"""Generates the Loom kernels of a ViT image encoder (GGUF mmproj) for loom-serve image components.
All activations are F32, row-major [rows, width]. Templates use $NAME placeholders (no f-string braces)."""
import math, re

def T(s, **kw):
    for k in sorted(kw, key=len, reverse=True):
        s = s.replace("$" + k, str(kw[k]))
    assert "$" not in s.replace("$$", ""), [l for l in s.splitlines() if "$" in l][:3]
    return s

MAXROWS = 65536   # rows (tokens x heads) a kernel accepts

# ---- row norms: LayerNorm (w, b), RMSNorm (w), RMSNorm (no weight); width W = 32 * G, G <= 256 ----
def norm(W, mode, eps):
    assert W % 32 == 0 and W // 32 <= 256
    G = W // 32
    name = f"v_norm_{mode}_{W}"
    ln = mode == "ln"
    wargs = {"ln": ", %w: buffer, %b: buffer", "rmsw": ", %w: buffer", "rms": ""}[mode]
    src = '''kernel.def export("$NAME") @$NAME(%rows: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%rows, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%rows_arg: index, %x: buffer$WARGS, %y: buffer) {
  %rows = index.assume %rows_arg [range(%rows_arg, 1, $MAXROWS)] : index
  %row_id = kernel.workgroup.id<x> : index
  %row = index.assume %row_id [lt(%row_id, %rows)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %groups = index.constant $G : index
  %zero_f = scalar.constant 0.0 : f32
  %zero32 = vector.constant 0.0 : vector<32xf32>
  %count = scalar.constant $W.0 : f32
  %active = index.cmp ult, %lane, %groups : index
  %x_view = buffer.view %x[%zero] : buffer -> view<[%rows]x$Gx32xf32>
  %xv = scf.if %active -> (vector<32xf32>) {
    %bl = index.assume %lane [range(%lane, 0, $GM1)] : index
    %a = vector.load %x_view[%row, %bl, %c0] : view<[%rows]x$Gx32xf32> -> vector<32xf32>
    scf.yield %a : vector<32xf32>
  } else {
    scf.yield %zero32 : vector<32xf32>
  }
$CENTER
  %sq = vector.mulf %xc, %xc : vector<32xf32>
  %local = vector.reduce<addf> %sq, %zero_f : vector<32xf32>, f32
  %sum = kernel.workgroup.reduce<addf> %local : f32
  %mean = scalar.divf %sum, %count : f32
  %eps = scalar.constant $EPS : f32
  %stable = scalar.addf %mean, %eps : f32
  %inv = scalar.rsqrtf %stable : f32
  %invs = vector.splat %inv : vector<32xf32>
  scf.if %active {
    %bl = index.assume %lane [range(%lane, 0, $GM1)] : index
    %xn = vector.mulf %xc, %invs : vector<32xf32>
$APPLY
    %y_view = buffer.view %y[%zero] : buffer -> view<[%rows]x$Gx32xf32>
    vector.store %r, %y_view[%row, %bl, %c0] : vector<32xf32>, view<[%rows]x$Gx32xf32>
  }
  kernel.return
}
'''
    center = ('''  %xlocal = vector.reduce<addf> %xv, %zero_f : vector<32xf32>, f32
  %xsum = kernel.workgroup.reduce<addf> %xlocal : f32
  %xmean = scalar.divf %xsum, %count : f32
  %xmeans = vector.splat %xmean : vector<32xf32>
  %xd = vector.subf %xv, %xmeans : vector<32xf32>
  %xc = scf.select %active, %xd, %zero32 : vector<32xf32>''' if ln else "  %xc = vector.addf %xv, %zero32 : vector<32xf32>")
    apply = {"ln": '''    %w_view = buffer.view %w[%zero] : buffer -> view<$Gx32xf32>
    %b_view = buffer.view %b[%zero] : buffer -> view<$Gx32xf32>
    %wv = vector.load %w_view[%bl, %c0] : view<$Gx32xf32> -> vector<32xf32>
    %bv = vector.load %b_view[%bl, %c0] : view<$Gx32xf32> -> vector<32xf32>
    %xw = vector.mulf %xn, %wv : vector<32xf32>
    %r = vector.addf %xw, %bv : vector<32xf32>''',
             "rmsw": '''    %w_view = buffer.view %w[%zero] : buffer -> view<$Gx32xf32>
    %wv = vector.load %w_view[%bl, %c0] : view<$Gx32xf32> -> vector<32xf32>
    %r = vector.mulf %xn, %wv : vector<32xf32>''',
             "rms": "    %r = vector.addf %xn, %zero32 : vector<32xf32>"}[mode]
    return name, T(src.replace("$CENTER", center).replace("$APPLY", apply), NAME=name, WARGS=wargs, G=G, GM1=G - 1, W=W,
                   EPS=repr(float(eps)), MAXROWS=MAXROWS)

# ---- elementwise over n = rows * width (flat), 256 threads x 4 values ----
EW_OPS = {
    # name: (extra args, body producing %r from %a [, %b])
    "add":       (", %b: buffer", "%r = scalar.addf %a, %bv : f32", True),          # y = x + b (same shape)
    "gelu_tanh": ("", "GELU_TANH", False),
    "gelu_erf":  ("", "GELU_ERF", False),
    "gate_gelu_tanh": (", %b: buffer", "GATE", True),                                # y = gelu_tanh(x) * b
    "gate_silu": (", %b: buffer", "GATE_SILU", True),                                # y = silu(x) * b
}
SILU_GATE = '''    %na = scalar.negf %a : f32
    %ea = scalar.expf<afn> %na : f32
    %one = scalar.constant 1.0 : f32
    %den = scalar.addf %ea, %one : f32
    %g = scalar.divf %a, %den : f32
    %r = scalar.mulf %g, %bv : f32'''
GELU_TANH = '''    %x3 = scalar.mulf %a, %a : f32
    %x3b = scalar.mulf %x3, %a : f32
    %k1 = scalar.constant 0.044715 : f32
    %inner0 = scalar.mulf %x3b, %k1 : f32
    %inner1 = scalar.addf %inner0, %a : f32
    %k2 = scalar.constant 1.5957691216057308 : f32
    %z = scalar.mulf %inner1, %k2 : f32
    %nz = scalar.negf %z : f32
    %ez = scalar.expf<afn> %nz : f32
    %one = scalar.constant 1.0 : f32
    %den = scalar.addf %ez, %one : f32
    %r = scalar.divf %a, %den : f32'''
# erf via Abramowitz-Stegun 7.1.26 is too coarse; use erf(x) = sign * (1 - t*exp(-x^2 + poly(t))) (Numerical Recipes erfc, 1.2e-7)
GELU_ERF = '''    %one = scalar.constant 1.0 : f32
    %half = scalar.constant 0.5 : f32
    %rs2 = scalar.constant 0.7071067811865476 : f32
    %u = scalar.mulf %a, %rs2 : f32
    %au = scalar.absf %u : f32
    %hz = scalar.mulf %au, %half : f32
    %tden = scalar.addf %hz, %one : f32
    %t = scalar.divf %one, %tden : f32
    %p9 = scalar.constant 0.17087277 : f32
    %p8 = scalar.constant -0.82215223 : f32
    %p7 = scalar.constant 1.48851587 : f32
    %p6 = scalar.constant -1.13520398 : f32
    %p5 = scalar.constant 0.27886807 : f32
    %p4 = scalar.constant -0.18628806 : f32
    %p3 = scalar.constant 0.09678418 : f32
    %p2 = scalar.constant 0.37409196 : f32
    %p1 = scalar.constant 1.00002368 : f32
    %p0 = scalar.constant -1.26551223 : f32
    %q8 = scalar.fmaf %t, %p9, %p8 : f32
    %q7 = scalar.fmaf %t, %q8, %p7 : f32
    %q6 = scalar.fmaf %t, %q7, %p6 : f32
    %q5 = scalar.fmaf %t, %q6, %p5 : f32
    %q4 = scalar.fmaf %t, %q5, %p4 : f32
    %q3 = scalar.fmaf %t, %q4, %p3 : f32
    %q2 = scalar.fmaf %t, %q3, %p2 : f32
    %q1 = scalar.fmaf %t, %q2, %p1 : f32
    %q0 = scalar.fmaf %t, %q1, %p0 : f32
    %u2 = scalar.mulf %au, %au : f32
    %e0 = scalar.subf %q0, %u2 : f32
    %e1 = scalar.expf<afn> %e0 : f32
    %erfc = scalar.mulf %t, %e1 : f32
    %erfa = scalar.subf %one, %erfc : f32
    %zf_e = scalar.constant 0.0 : f32
    %neg = scalar.cmpf olt, %u, %zf_e : f32
    %erfn = scalar.negf %erfa : f32
    %erf = scf.select %neg, %erfn, %erfa : f32
    %onep = scalar.addf %erf, %one : f32
    %hx = scalar.mulf %a, %half : f32
    %r = scalar.mulf %hx, %onep : f32'''

def ew(op):
    extra, body, has_b = EW_OPS[op]
    name = f"v_ew_{op}"
    if body == "GELU_TANH": body = GELU_TANH
    elif body == "GELU_ERF": body = GELU_ERF
    elif body == "GATE_SILU": body = SILU_GATE
    elif body == "GATE": body = GELU_TANH.replace("%r = scalar.divf %a, %den : f32", "%g = scalar.divf %a, %den : f32") + "\n    %r = scalar.mulf %g, %bv : f32"
    else: body = "    " + body
    src = '''kernel.def export("$NAME") @$NAME(%n: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %c255 = index.constant 255 : index
  %n_up = index.add %n, %c255 : index
  %groups = index.div %n_up, %c256 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%n_arg: index, %x: buffer$EXTRA, %y: buffer) {
  %n = index.assume %n_arg [range(%n_arg, 1, 268435456)] : index
  %c256 = index.constant 256 : index
  %gid = kernel.workgroup.id<x> : index
  %lid = kernel.workitem.id<x> : index
  %i = index.madd %gid, %c256, %lid : index
  %live = index.cmp ult, %i, %n : index
  scf.if %live {
    %bi = index.assume %i [lt(%i, %n)] : index
    %zero = index.constant 0 : offset
    %xv = buffer.view %x[%zero] : buffer -> view<[%n]xf32>
    %a = view.load %xv[%bi] : view<[%n]xf32> -> f32
$LOADB
$BODY
    %yv = buffer.view %y[%zero] : buffer -> view<[%n]xf32>
    view.store %r, %yv[%bi] : f32, view<[%n]xf32>
  }
  kernel.return
}
'''
    loadb = '''    %bview = buffer.view %b[%zero] : buffer -> view<[%n]xf32>
    %bv = view.load %bview[%bi] : view<[%n]xf32> -> f32''' if has_b else ""
    return name, T(src.replace("$BODY", body).replace("$LOADB", loadb), NAME=name, EXTRA=extra)

# ---- bias (per column) + optional activation + optional clamp: y[r, c] = act(x[r, c] + b[c]) ----
def bias_act(N, act):
    name = f"v_bias_{act}_{N}"
    body = {"none": "    %r = scalar.addf %a, %zf : f32", "gelu_tanh": GELU_TANH, "gelu_erf": GELU_ERF}[act]
    src = '''kernel.def export("$NAME") @$NAME(%rows: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cols = index.constant $N : index
  %cg = index.constant $CG : index
  kernel.launch.config workgroups(%rows, %cg, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%rows_arg: index, %x: buffer, %bias: buffer, %y: buffer) {
  %rows = index.assume %rows_arg [range(%rows_arg, 1, $MAXROWS)] : index
  %c256 = index.constant 256 : index
  %cols = index.constant $N : index
  %row_id = kernel.workgroup.id<x> : index
  %row = index.assume %row_id [lt(%row_id, %rows)] : index
  %cg = kernel.workgroup.id<y> : index
  %lid = kernel.workitem.id<x> : index
  %c = index.madd %cg, %c256, %lid : index
  %live = index.cmp ult, %c, %cols : index
  scf.if %live {
    %bc = index.assume %c [lt(%c, %cols)] : index
    %zero = index.constant 0 : offset
    %zf = scalar.constant 0.0 : f32
    %xv = buffer.view %x[%zero] : buffer -> view<[%rows]x$Nxf32>
    %bv = buffer.view %bias[%zero] : buffer -> view<$Nxf32>
    %x0 = view.load %xv[%row, %bc] : view<[%rows]x$Nxf32> -> f32
    %b0 = view.load %bv[%bc] : view<$Nxf32> -> f32
    %a = scalar.addf %x0, %b0 : f32
$BODY
    %yv = buffer.view %y[%zero] : buffer -> view<[%rows]x$Nxf32>
    view.store %r, %yv[%row, %bc] : f32, view<[%rows]x$Nxf32>
  }
  kernel.return
}
'''
    return name, T(src.replace("$BODY", body), NAME=name, N=N, MAXROWS=MAXROWS, CG=(N + 255) // 256)

# ---- linear layers: row-scaled F16 activations x WMMA GEMM (the starter's gen_gemm, rows padded to 64) ----
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))          # the starter (or kit) directory: gen_gemm.py
import gen_gemm as _gg                                # noqa: E402
import gen_gemm_rdna4 as _rdna4                       # noqa: E402  gfx12 twins of the WMMA kernels (LOOM_RDNA4=0: none)

# Projector weights are F16 / BF16 / F32: add those formats to gen_gemm's F16 staging.
_DEQ = {
    "f16": ("""
      %{o}_two_b = index.constant 2 : offset
      %{o}_col_bytes = index.scale %w_col, %{o}_two_b : index, offset -> offset
      %{o}_origin = index.add %w_row_origin, %{o}_col_bytes : offset
      %{o}_view = buffer.view %{w}[%{o}_origin] : buffer -> view<16xf16>
      %{o} = vector.load %{o}_view[0] : view<16xf16> -> vector<16xf16>""", 64),
    "bf16": ("""
      %{o}_two_b = index.constant 2 : offset
      %{o}_col_bytes = index.scale %w_col, %{o}_two_b : index, offset -> offset
      %{o}_origin = index.add %w_row_origin, %{o}_col_bytes : offset
      %{o}_view = buffer.view %{w}[%{o}_origin] : buffer -> view<16xbf16>
      %{o}_raw = vector.load %{o}_view[0] : view<16xbf16> -> vector<16xbf16>
      %{o}_wide = vector.extf %{o}_raw : vector<16xbf16> to vector<16xf32>
      %{o} = vector.fptrunc %{o}_wide : vector<16xf32> to vector<16xf16>""", 64),
    "f32": ("""
      %{o}_four_b = index.constant 4 : offset
      %{o}_col_bytes = index.scale %w_col, %{o}_four_b : index, offset -> offset
      %{o}_origin = index.add %w_row_origin, %{o}_col_bytes : offset
      %{o}_view = buffer.view %{w}[%{o}_origin] : buffer -> view<16xf32>
      %{o}_wide = vector.load %{o}_view[0] : view<16xf32> -> vector<16xf32>
      %{o} = vector.fptrunc %{o}_wide : vector<16xf32> to vector<16xf16>""", 128),
}
for _f, (_code, _bytes) in _DEQ.items():
    _gg.FMT.setdefault(_f, dict(block_bytes=_bytes, block_values=32))
_base_dequant16 = _gg.dequant16
def _dequant16(fmt, w, view_name, out):
    if fmt in _DEQ:
        return _DEQ[fmt][0].replace("{o}", out).replace("{w}", w)
    return _base_dequant16(fmt, w, view_name, out)
_gg.dequant16 = _dequant16

def gemm_any(fmt, K, N):
    """Y[rows, N] = inverse_scale[row] * X16[rows, K] . W[N, K]^T for shapes the WMMA GEMM cannot tile
    (K or N not a multiple of 64; Qwen3-VL/Qwen3.8 encoder FFN: 1152 x 4304). A workgroup computes 16 rows x 64
    columns (one lane per column, 16 row accumulators), reading 8 f16 values of a weight row per step and reusing
    them for the 16 rows; the encoder runs once per image, so this only has to be reasonable, not peak."""
    VK = 8 if K % 8 == 0 else 4   # values per load step (Qwen2.5-VL encoder FFN: K = 3420)
    assert fmt in ("f16", "bf16") and K % VK == 0, (fmt, K)
    name = f"v_gemm_any_{fmt}_{K}x{N}"
    R = 16
    accs = ", ".join(f"%a{r} = %zf : f32" for r in range(R))
    types = ", ".join(["f32"] * R)
    L = [T('''kernel.def export("$NAME") @$NAME(%rows: index, %n: index) {
  %c1 = index.constant 1 : index
  %c15 = index.constant 15 : index
  %c16 = index.constant 16 : index
  %c63 = index.constant 63 : index
  %c64 = index.constant 64 : index
  %rp = index.add %rows, %c15 : index
  %row_tiles = index.div %rp, %c16 : index
  %ncst = index.constant $N : index
  %np = index.add %ncst, %c63 : index
  %col_tiles = index.div %np, %c64 : index
  kernel.launch.config workgroups(%row_tiles, %col_tiles, %c1) workgroup_size(%c64, %c1, %c1) : index
} launch(%rows_arg: index, %k_arg: index, %n_arg: index, %x: buffer, %scales: buffer, %w: buffer, %y: buffer) {
  %rows = index.assume %rows_arg [range(%rows_arg, 1, $MAXROWS)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c64 = index.constant 64 : index
  %ck = index.constant $K : index
  %cn = index.constant $N : index
  %zf = scalar.constant 0.0 : f32
  %rt_id = kernel.workgroup.id<x> : index
  %ct_id = kernel.workgroup.id<y> : index
  %lid = kernel.workitem.id<x> : index
  %lane = index.assume %lid [range(%lid, 0, 63)] : index
  %col0 = index.mul %ct_id, %c64 : index
  %col = index.add %col0, %lane : index
  %col_live = index.cmp ult, %col, %cn : index
  %nlast = index.constant $NM1 : index
  %colc = index.min %col, %nlast : index
  %wcol = index.assume %colc [range(%colc, 0, $NM1)] : index
  %row0 = index.mul %rt_id, %c16 : index
  %x_view = buffer.view %x[%zero] : buffer -> view<[%rows]x$Kxf16>
  %w_view = buffer.view %w[%zero] : buffer -> view<$Nx$Kx$FMT>
  %s_view = buffer.view %scales[%zero] : buffer -> view<[%rows]x2xf32>
  %rlast = index.sub %rows, %c1 : index''', NAME=name, N=N, K=K, NM1=N - 1, MAXROWS=MAXROWS, FMT=fmt)]
    L.append(f"  {', '.join(f'%r{r}' for r in range(R))} = scf.for %kk = [%c0 to %ck step %c8]({accs}) -> ({types}) {{")
    L.append(T('''    %bk = index.assume %kk [range(%kk, 0, $KM8), multiple_of(%kk, 8)] : index
    %wh = vector.load %w_view[%wcol, %bk] : view<$Nx$Kx$FMT> -> vector<8x$FMT>
    %wv = vector.extf %wh : vector<8x$FMT> to vector<8xf32>''', KM8=K - VK, N=N, K=K, FMT=fmt))
    for r in range(R):
        L.append(T('''    %ri$r = index.constant $r : index
    %rr$r = index.add %row0, %ri$r : index
    %rc$r = index.min %rr$r, %rlast : index
    %rb$r = index.assume %rc$r [lt(%rc$r, %rows)] : index
    %xh$r = vector.load %x_view[%rb$r, %bk] : view<[%rows]x$Kxf16> -> vector<8xf16>
    %xv$r = vector.extf %xh$r : vector<8xf16> to vector<8xf32>
    %d$r = vector.dotf<reassoc|contract> %wv, %xv$r, %zf : vector<8xf32>, vector<8xf32>, f32
    %n$r = scalar.addf %a$r, %d$r : f32''', r=r, K=K))
    L.append(f"    scf.yield {', '.join(f'%n{r}' for r in range(R))} : {types}\n  }}")
    L.append("  %y_view = buffer.view %y[%zero] : buffer -> view<[%rows]x" + str(N) + "xf32>")
    for r in range(R):
        L.append(T('''  %ro$r = index.constant $r : index
  %orow$r = index.add %row0, %ro$r : index
  %olive$r = index.cmp ult, %orow$r, %rows : index
  %ok$r = scalar.andi %olive$r, %col_live : i1
  scf.if %ok$r {
    %ob$r = index.assume %orow$r [lt(%orow$r, %rows)] : index
    %inv$r = view.load %s_view[%ob$r, %c1] : view<[%rows]x2xf32> -> f32
    %val$r = scalar.mulf %r$r, %inv$r : f32
    view.store %val$r, %y_view[%ob$r, %wcol] : f32, view<[%rows]x$Nxf32>
  }''', r=r, N=N))
    L.append("  kernel.return\n}\n")
    src = "\n".join(L)
    if VK == 4:
        src = (src.replace("%c8 = index.constant 8 : index", "%c8 = index.constant 4 : index").replace("multiple_of(%kk, 8)", "multiple_of(%kk, 4)")
               .replace("vector<8x", "vector<4x"))
    return name, src


def gemm(fmt, K, N, mode="store"):
    """Shape-specialized GEMM: Y[rows, N] (+)= X16[rows, K] . W[N, K]^T; rows a multiple of 64.
    Shapes the WMMA tiles cannot cover (K or N not a multiple of 64) take gemm_any (same launch signature)."""
    if (K % 64 or N % 64) and mode == "store":
        return gemm_any(fmt, K, N)
    assert K % 64 == 0 and N % 64 == 0, (K, N)
    base = f"{_gg.PREFIX}_gemm_{fmt}_{mode}"; name = f"v_gemm_{fmt}_{mode}_{K}x{N}"
    s = _gg.gemm(fmt, mode).replace(base, name)
    s = s.replace("range(%rows_arg, 64, 8192)", f"range(%rows_arg, 64, {MAXROWS})")
    s = re.sub(r"%k = index.assume %k_arg \[[^\]]*\] : index", f"%k = index.constant {K} : index", s)
    s = re.sub(r"%n = index.assume %n_arg \[[^\]]*\] : index", f"%n = index.constant {N} : index", s)
    # the config region computes tiles from %n: make it constant too
    s = s.replace("  %out_tiles = index.div %n, %c64 : index", f"  %n_const = index.constant {N} : index\n  %out_tiles = index.div %n_const, %c64 : index", 1)
    return name, _rdna4.with_twins(s)

def row_scale(K=8):
    """Row-scaled F16 copy of rows K wide (K a multiple of 8; else a variant for multiples of 4)."""
    m = 8 if K % 8 == 0 else 4
    name = "v_row_scale_f16" + ("" if m == 8 else "_k4")
    src = ROW_SCALE.replace("g_row_scale_f16", name).replace("[range(%k_arg, 256, 65536), multiple_of(%k_arg, 256)]", f"[range(%k_arg, {m}, 65536), multiple_of(%k_arg, {m})]")
    return name, src

ROW_SCALE = open(_os.path.join(_HERE, "row_scale.loom")).read()

# ---- token order: token t -> patch (py, px) on a grid gw patches wide ----
def order_code(order, t="%t", gw="%gw"):
    """Loom lines defining %py, %px (index) for token t."""
    if order == "raster":
        return f'''  %py = index.div {t}, {gw} : index
  %px = index.rem {t}, {gw} : index'''
    if order == "block2":   # 2x2 merge blocks contiguous: t = ((ty * gw/2 + tx) * 2 + dy) * 2 + dx
        return f'''  %c2o = index.constant 2 : index
  %c4o = index.constant 4 : index
  %bw0 = index.div {gw}, %c2o : index
  %bw1 = index.max %bw0, %c1 : index
  %bw = index.assume %bw1 [range(%bw1, 1, 4096)] : index
  %blk = index.div {t}, %c4o : index
  %sub = index.rem {t}, %c4o : index
  %ty = index.div %blk, %bw : index
  %tx = index.rem %blk, %bw : index
  %dy = index.div %sub, %c2o : index
  %dx = index.rem %sub, %c2o : index
  %py = index.madd %ty, %c2o, %dy : index
  %px = index.madd %tx, %c2o, %dx : index'''
    raise ValueError(order)

def _wload(fmt, view, idx, out):
    """scalar weight load as f32 from a typed view."""
    if fmt == "f32":
        return f"    {out} = view.load {view}[{idx}] : view<[%wn]xf32> -> f32"
    return f'''    {out}_h = view.load {view}[{idx}] : view<[%wn]x{fmt}> -> {fmt}
    {out} = scalar.extf {out}_h : {fmt} to f32'''

# ---- patch embedding: y[t, e] = sum_k w[e, k] * patch_t[k] (+ w2) (+ b); k = (c, ky, kx) ----
def patch_embed(P, E, fmt, order, dual=False, bias=False):
    K = 3 * P * P; KP = (K + 255) // 256 * 256
    name = f"v_patch_{P}_{E}_{fmt}_{order}" + ("_dual" if dual else "") + ("_bias" if bias else "")
    wargs = ", %w: buffer" + (", %w2: buffer" if dual else "") + (", %b: buffer" if bias else "")
    lines = [T('''kernel.def export("$NAME") @$NAME(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %gw_arg: index, %img_w_arg: index, %pix: buffer$WARGS, %y: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, $MAXROWS)] : index
  %gw = index.assume %gw_arg [range(%gw_arg, 1, 1024)] : index
  %img_w = index.assume %img_w_arg [range(%img_w_arg, 1, 16384)] : index
  %t_id = kernel.workgroup.id<x> : index
  %t = index.assume %t_id [lt(%t_id, %tokens)] : index
  %lid = kernel.workitem.id<x> : index
  %lane = index.assume %lid [range(%lid, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c3 = index.constant 3 : index
  %c256 = index.constant 256 : index
  %cP = index.constant $P : index
  %cPP = index.constant $PP : index
  %cK = index.constant $K : index
  %cKP = index.constant $KP : index
  %cE = index.constant $E : index
  %zf = scalar.constant 0.0 : f32''', NAME=name, WARGS=wargs, MAXROWS=MAXROWS, P=P, PP=P * P, K=K, KP=KP, E=E),
             order_code(order),
             T('''  %stage_bytes = index.constant $KB : offset
  %stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %sv = buffer.view %stage[%zero] : buffer -> view<$KPxf32>
  %rows_px = index.mul %tokens, %c1 : index
  %pix_n = index.constant 268435456 : index
  %pv = buffer.view %pix[%zero] : buffer -> view<[%pix_n]xf32>
  %y0 = index.mul %py, %cP : index
  %x0 = index.mul %px, %cP : index
  scf.for %i = [%lane to %cKP step %c256] {
    %bi = index.assume %i [lt(%i, %cKP)] : index
    %inside = index.cmp ult, %i, %cK : index
    %v = scf.if %inside -> (f32) {
      %ch = index.div %i, %cPP : index
      %r = index.rem %i, %cPP : index
      %ky = index.div %r, %cP : index
      %kx = index.rem %r, %cP : index
      %yy = index.add %y0, %ky : index
      %xx = index.add %x0, %kx : index
      %rowpix = index.mul %yy, %img_w : index
      %pixel = index.add %rowpix, %xx : index
      %el = index.madd %pixel, %c3, %ch : index
      %bel = index.assume %el [lt(%el, %pix_n)] : index
      %pval = view.load %pv[%bel] : view<[%pix_n]xf32> -> f32
      scf.yield %pval : f32
    } else {
      scf.yield %zf : f32
    }
    view.store %v, %sv[%bi] : f32, view<$KPxf32>
  }
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %wn = index.constant $WN : index
  %wv = buffer.view %w[%zero] : buffer -> view<[%wn]x$WT>''', KB=KP * 4, KP=KP, WN=E * K, WT=fmt)]
    if dual:
        lines.append(T("  %wv2 = buffer.view %w2[%zero] : buffer -> view<[%wn]x$WT>", WT=fmt))
    lines.append(T('''  %yv = buffer.view %y[%zero] : buffer -> view<[%tokens]x$Exf32>
  scf.for %e = [%lane to %cE step %c256] {
    %be = index.assume %e [lt(%e, %cE)] : index
    %wrow = index.mul %be, %cK : index
    %acc = scf.for %k = [%c0 to %cK step %c1](%a = %zf : f32) -> (f32) {
      %bk = index.assume %k [lt(%k, %cKP)] : index
      %xk = view.load %sv[%bk] : view<$KPxf32> -> f32
      %wi = index.add %wrow, %k : index
      %bwi = index.assume %wi [lt(%wi, %wn)] : index''', E=E, KP=KP))
    lines.append(_wload(fmt, "%wv", "%bwi", "%wk"))
    if dual:
        lines.append(_wload(fmt, "%wv2", "%bwi", "%wk2"))
        lines.append("    %wks = scalar.addf %wk, %wk2 : f32")
    else:
        lines.append("    %wks = scalar.addf %wk, %zf : f32")
    lines.append('''      %na = scalar.fmaf %wks, %xk, %a : f32
      scf.yield %na : f32
    }''')
    if bias:
        lines.append(T('''    %bview = buffer.view %b[%zero] : buffer -> view<$Exf32>
    %bb = view.load %bview[%be] : view<$Exf32> -> f32
    %out = scalar.addf %acc, %bb : f32''', E=E))
    else:
        lines.append("    %out = scalar.addf %acc, %zf : f32")
    lines.append(T('''    view.store %out, %yv[%t, %be] : f32, view<[%tokens]x$Exf32>
  }
  kernel.return
}
''', E=E))
    return name, "\n".join(lines)

# ---- pixel normalization: f32 [n] = (u8 / 255 - mean[c]) / std[c] * gain + shift (per family) ----
def pixels(mean, std, scale2=False):
    """input: RGB u8 [n*3] (i8 bytes); output f32 [n*3]. scale2: x*2-1 (gemma4v does it in the graph)."""
    name = "v_pixels_" + "_".join(f"{m:g}" for m in mean).replace(".", "p") + ("_x2m1" if scale2 else "")
    a = [1.0 / 255.0 / s for s in std]; b = [-m / s for m, s in zip(mean, std)]
    if scale2: a = [2 * v for v in a]; b = [2 * v - 1 for v in b]
    src = T('''kernel.def export("$NAME") @$NAME(%n: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %c255 = index.constant 255 : index
  %n_up = index.add %n, %c255 : index
  %groups = index.div %n_up, %c256 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%n_arg: index, %src: buffer, %y: buffer) {
  %n = index.assume %n_arg [range(%n_arg, 1, 268435456)] : index
  %c256 = index.constant 256 : index
  %c3 = index.constant 3 : index
  %gid = kernel.workgroup.id<x> : index
  %lid = kernel.workitem.id<x> : index
  %i = index.madd %gid, %c256, %lid : index
  %live = index.cmp ult, %i, %n : index
  scf.if %live {
    %bi = index.assume %i [lt(%i, %n)] : index
    %zero = index.constant 0 : offset
    %sv = buffer.view %src[%zero] : buffer -> view<[%n]xi8>
    %raw = view.load %sv[%bi] : view<[%n]xi8> -> i8
    %u = scalar.extui %raw : i8 to i32
    %f = scalar.uitofp %u : i32 to f32
    %ch = index.rem %bi, %c3 : index
    %c0 = index.constant 0 : index
    %c1 = index.constant 1 : index
    %is0 = index.cmp eq, %ch, %c0 : index
    %is1 = index.cmp eq, %ch, %c1 : index
    %a0 = scalar.constant $A0 : f32
    %a1 = scalar.constant $A1 : f32
    %a2 = scalar.constant $A2 : f32
    %b0 = scalar.constant $B0 : f32
    %b1 = scalar.constant $B1 : f32
    %b2 = scalar.constant $B2 : f32
    %a12 = scf.select %is1, %a1, %a2 : f32
    %a = scf.select %is0, %a0, %a12 : f32
    %b12 = scf.select %is1, %b1, %b2 : f32
    %b = scf.select %is0, %b0, %b12 : f32
    %r = scalar.fmaf %f, %a, %b : f32
    %yv = buffer.view %y[%zero] : buffer -> view<[%n]xf32>
    view.store %r, %yv[%bi] : f32, view<[%n]xf32>
  }
  kernel.return
}
''', NAME=name, A0=repr(a[0]), A1=repr(a[1]), A2=repr(a[2]), B0=repr(b[0]), B1=repr(b[1]), B2=repr(b[2]))
    return name, src

# ---- learned position embeddings ----
def pos_xy(E, npos):
    """gemma4v: y[t] += tab[0][px] + tab[1][py] (tab [2][npos][E]); raster order."""
    name = f"v_pos_xy_{E}"
    return name, T('''kernel.def export("$NAME") @$NAME(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %gw_arg: index, %tab: buffer, %y: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, $MAXROWS)] : index
  %gw = index.assume %gw_arg [range(%gw_arg, 1, $NPOS)] : index
  %t_id = kernel.workgroup.id<x> : index
  %t = index.assume %t_id [lt(%t_id, %tokens)] : index
  %lid = kernel.workitem.id<x> : index
  %lane = index.assume %lid [range(%lid, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cE = index.constant $E : index
  %cmax = index.constant $NPOSM1 : index
$ORDER
  %bpx0 = index.min %px, %cmax : index
  %bpx = index.assume %bpx0 [range(%bpx0, 0, $NPOSM1)] : index
  %bpy0 = index.min %py, %cmax : index
  %bpy = index.assume %bpy0 [range(%bpy0, 0, $NPOSM1)] : index
  %tv = buffer.view %tab[%zero] : buffer -> view<2x$NPOSx$Exf32>
  %yv = buffer.view %y[%zero] : buffer -> view<[%tokens]x$Exf32>
  scf.for %e = [%lane to %cE step %c256] {
    %be = index.assume %e [lt(%e, %cE)] : index
    %a = view.load %yv[%t, %be] : view<[%tokens]x$Exf32> -> f32
    %ex = view.load %tv[%c0, %bpx, %be] : view<2x$NPOSx$Exf32> -> f32
    %ey = view.load %tv[%c1, %bpy, %be] : view<2x$NPOSx$Exf32> -> f32
    %s1 = scalar.addf %a, %ex : f32
    %s2 = scalar.addf %s1, %ey : f32
    view.store %s2, %yv[%t, %be] : f32, view<[%tokens]x$Exf32>
  }
  kernel.return
}
''', NAME=name, E=E, NPOS=npos, NPOSM1=npos - 1, MAXROWS=MAXROWS, ORDER=order_code("raster"))

def pos_interp(E, n, order, align_corners):
    """y[t] += bilinear(grid n x n of E, at the patch's (py, px) of a gh x gw grid)."""
    name = f"v_pos_interp_{E}_{n}_{order}_" + ("ac" if align_corners else "hp")
    if align_corners:   # src = i * (n-1)/(o-1)  (o == 1 -> 0)
        coord = lambda v, o: f'''  %{v}_of_i = index.cast %{v} : index to i32
  %{v}_of = scalar.sitofp %{v}_of_i : i32 to f32
  %{o}_m1 = index.sub %{o}, %c1 : index
  %{o}_m1f_i = index.cast %{o}_m1 : index to i32
  %{o}_m1f = scalar.sitofp %{o}_m1f_i : i32 to f32
  %{o}_den = scalar.maxnumf %{o}_m1f, %onef : f32
  %{v}_s0 = scalar.mulf %{v}_of, %nm1f : f32
  %{v}_s = scalar.divf %{v}_s0, %{o}_den : f32'''
    else:               # src = clamp((i + 0.5) * n / o - 0.5, 0, n - 1)
        coord = lambda v, o: f'''  %{v}_of_i = index.cast %{v} : index to i32
  %{v}_of = scalar.sitofp %{v}_of_i : i32 to f32
  %{o}_f_i = index.cast %{o} : index to i32
  %{o}_f = scalar.sitofp %{o}_f_i : i32 to f32
  %{v}_h = scalar.addf %{v}_of, %halff : f32
  %{v}_r = scalar.mulf %{v}_h, %nf : f32
  %{v}_q = scalar.divf %{v}_r, %{o}_f : f32
  %{v}_m = scalar.subf %{v}_q, %halff : f32
  %{v}_c = scalar.maxnumf %{v}_m, %zf : f32
  %{v}_s = scalar.minnumf %{v}_c, %nm1f : f32'''
    def split(v):
        return f'''  %{v}_fl = scalar.floorf %{v}_s : f32
  %{v}_fr = scalar.subf %{v}_s, %{v}_fl : f32
  %{v}_i0r = scalar.fptosi %{v}_fl : f32 to i32
  %{v}_i0c = scalar.maxsi %{v}_i0r, %c0i : i32
  %{v}_i0d = scalar.minsi %{v}_i0c, %nm1i : i32
  %{v}_i0a = scalar.assume %{v}_i0d [range(%{v}_i0d, 0, {n - 1})] : i32
  %{v}_i0 = index.cast %{v}_i0a : i32 to index
  %{v}_i1r = index.add %{v}_i0, %c1 : index
  %{v}_i1m = index.min %{v}_i1r, %nm1 : index
  %{v}_i1 = index.assume %{v}_i1m [range(%{v}_i1m, 0, {n - 1})] : index'''
    body = T('''kernel.def export("$NAME") @$NAME(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %gw_arg: index, %gh_arg: index, %tab: buffer, %y: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, $MAXROWS)] : index
  %gw = index.assume %gw_arg [range(%gw_arg, 1, 1024)] : index
  %gh = index.assume %gh_arg [range(%gh_arg, 1, 1024)] : index
  %t_id = kernel.workgroup.id<x> : index
  %t = index.assume %t_id [lt(%t_id, %tokens)] : index
  %lid = kernel.workitem.id<x> : index
  %lane = index.assume %lid [range(%lid, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cE = index.constant $E : index
  %nm1 = index.constant $NM1 : index
  %c0i = scalar.constant 0 : i32
  %nm1i = scalar.constant $NM1 : i32
  %zf = scalar.constant 0.0 : f32
  %onef = scalar.constant 1.0 : f32
  %halff = scalar.constant 0.5 : f32
  %nf = scalar.constant $N.0 : f32
  %nm1f = scalar.constant $NM1.0 : f32
$ORDER
''', NAME=name, E=E, NM1=n - 1, N=n, MAXROWS=MAXROWS, ORDER=order_code(order))
    body += "\n".join([coord("py", "gh"), coord("px", "gw"), split("py"), split("px")]) + "\n"
    body += T('''  %fy1 = scalar.addf %py_fr, %zf : f32
  %fy0 = scalar.subf %onef, %py_fr : f32
  %fx1 = scalar.addf %px_fr, %zf : f32
  %fx0 = scalar.subf %onef, %px_fr : f32
  %tv = buffer.view %tab[%zero] : buffer -> view<$Nx$Nx$Exf32>
  %yv = buffer.view %y[%zero] : buffer -> view<[%tokens]x$Exf32>
  scf.for %e = [%lane to %cE step %c256] {
    %be = index.assume %e [lt(%e, %cE)] : index
    %v00 = view.load %tv[%py_i0, %px_i0, %be] : view<$Nx$Nx$Exf32> -> f32
    %v01 = view.load %tv[%py_i0, %px_i1, %be] : view<$Nx$Nx$Exf32> -> f32
    %v10 = view.load %tv[%py_i1, %px_i0, %be] : view<$Nx$Nx$Exf32> -> f32
    %v11 = view.load %tv[%py_i1, %px_i1, %be] : view<$Nx$Nx$Exf32> -> f32
    %r0a = scalar.mulf %v00, %fx0 : f32
    %r0 = scalar.fmaf %v01, %fx1, %r0a : f32
    %r1a = scalar.mulf %v10, %fx0 : f32
    %r1 = scalar.fmaf %v11, %fx1, %r1a : f32
    %ra = scalar.mulf %r0, %fy0 : f32
    %r = scalar.fmaf %r1, %fy1, %ra : f32
    %a = view.load %yv[%t, %be] : view<[%tokens]x$Exf32> -> f32
    %s = scalar.addf %a, %r : f32
    view.store %s, %yv[%t, %be] : f32, view<[%tokens]x$Exf32>
  }
  kernel.return
}
''', E=E, N=n)
    return name, body

# ---- 2-D rotary embeddings, in place on rows [tokens, S] at column offset O (NH heads of D) ----
ROPE_MODES = {
    # mode: (theta, function(j, D) -> (lo, hi, axis 'x'|'y', exponent index i, n_dims for the exponent), position offset)
    "gemma_halves": (100.0, 0),     # neox pairs within each half; x on the first half, y on the second
    "qwen_vision": (10000.0, 0),    # pairs (j, j + D/2); y for j < D/4, x after
    "normal_halves": (10000.0, 1),  # adjacent pairs within each half; x+1 / y+1 (1-indexed)
}
def rope_pairs(mode, D):
    half, q = D // 2, D // 4; out = []
    for j in range(D // 2):
        if mode == "gemma_halves":
            jj = j % q; base = 0 if j < q else half
            out.append((base + jj, base + jj + q, "x" if j < q else "y", jj, half))
        elif mode == "qwen_vision":
            jj = j if j < q else j - q
            out.append((j, j + half, "y" if j < q else "x", jj, half))
        elif mode == "normal_halves":
            jj = j % q; base = 0 if j < q else half
            out.append((base + 2 * jj, base + 2 * jj + 1, "x" if j < q else "y", jj, half))
    return out

ROPE_PAIR = {   # pair j -> %lo_i, %hi_i (index), %isy (i1), %jj (frequency index)
    "gemma_halves": """    %isy = index.cmp uge, %j, %cq : index
    %jj = index.rem %j, %cq : index
    %c0p = index.constant 0 : index
    %base = scf.select %isy, %chalf, %c0p : index
    %lo_i = index.add %base, %jj : index
    %hi_i = index.add %lo_i, %cq : index""",
    "qwen_vision": """    %isx = index.cmp uge, %j, %cq : index
    %true = scalar.constant true : i1
    %isy = scalar.xori %isx, %true : i1
    %jj = index.rem %j, %cq : index
    %lo_i = index.add %j, %c0 : index
    %hi_i = index.add %j, %chalf : index""",
    "normal_halves": """    %isy = index.cmp uge, %j, %cq : index
    %jj = index.rem %j, %cq : index
    %c0p = index.constant 0 : index
    %c2p = index.constant 2 : index
    %base = scf.select %isy, %chalf, %c0p : index
    %lo_i = index.madd %jj, %c2p, %base : index
    %hi_i = index.add %lo_i, %c1 : index""",
}

def rope(mode, D, NH, S, O, order):
    theta, off = ROPE_MODES[mode]; pairs = rope_pairs(mode, D); NP = len(pairs)
    name = f"v_rope_{mode}_{D}x{NH}_s{S}_o{O}_{order}"
    src = T('''kernel.def export("$NAME") @$NAME(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %gw_arg: index, %x: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, $MAXROWS)] : index
  %gw = index.assume %gw_arg [range(%gw_arg, 1, 4096)] : index
  %t_id = kernel.workgroup.id<x> : index
  %t = index.assume %t_id [lt(%t_id, %tokens)] : index
  %lid = kernel.workitem.id<x> : index
  %lane = index.assume %lid [range(%lid, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cNP = index.constant $NP : index
  %cWORK = index.constant $WORK : index
  %cD = index.constant $D : index
  %cO = index.constant $O : index
$ORDER
  %pxi = index.cast %px : index to i32
  %pyi = index.cast %py : index to i32
  %offi = scalar.constant $OFF : i32
  %pxo = scalar.addi %pxi, %offi : i32
  %pyo = scalar.addi %pyi, %offi : i32
  %pxf = scalar.sitofp %pxo : i32 to f32
  %pyf = scalar.sitofp %pyo : i32 to f32
  %two_pi = scalar.constant 6.283185307179586 : f32
  %inv_two_pi = scalar.constant 0.15915494309189535 : f32
  %xv = buffer.view %x[%zero] : buffer -> view<[%tokens]x$Sxf32>
  %cq = index.constant $Q : index
  %chalf = index.constant $HALF : index
  %lnth = scalar.constant $LNTH : f32
  %halff = scalar.constant $HALF.0 : f32
  scf.for %w = [%lane to %cWORK step %c256] {
    %h = index.div %w, %cNP : index
    %j0 = index.rem %w, %cNP : index
    %j = index.assume %j0 [range(%j0, 0, $NPM1)] : index
$PAIR
    %ii = index.cast %jj : index to i32
    %if0 = scalar.sitofp %ii : i32 to f32
    %if1 = scalar.mulf %if0, %lnth : f32
    %if2 = scalar.divf %if1, %halff : f32
    %if3 = scalar.negf %if2 : f32
    %invf = scalar.expf<afn> %if3 : f32
    %pos = scf.select %isy, %pyf, %pxf : f32
    %ang0 = scalar.mulf %pos, %invf : f32
    %tr = scalar.mulf %ang0, %inv_two_pi : f32
    %trf = scalar.floorf %tr : f32
    %wrap = scalar.mulf %trf, %two_pi : f32
    %ang = scalar.subf %ang0, %wrap : f32
    %cs = scalar.cosf<afn> %ang : f32
    %sn = scalar.sinf<afn> %ang : f32
    %hcol = index.madd %h, %cD, %cO : index
    %clo0 = index.add %hcol, %lo_i : index
    %chi0 = index.add %hcol, %hi_i : index
    %clo = index.assume %clo0 [lt(%clo0, $S)] : index
    %chi = index.assume %chi0 [lt(%chi0, $S)] : index
    %a = view.load %xv[%t, %clo] : view<[%tokens]x$Sxf32> -> f32
    %b = view.load %xv[%t, %chi] : view<[%tokens]x$Sxf32> -> f32
    %acs = scalar.mulf %a, %cs : f32
    %bsn = scalar.mulf %b, %sn : f32
    %ra = scalar.subf %acs, %bsn : f32
    %asn = scalar.mulf %a, %sn : f32
    %rb = scalar.fmaf %b, %cs, %asn : f32
    view.store %ra, %xv[%t, %clo] : f32, view<[%tokens]x$Sxf32>
    view.store %rb, %xv[%t, %chi] : f32, view<[%tokens]x$Sxf32>
  }
  kernel.return
}
''', NAME=name, NP=NP, NPM1=NP - 1, WORK=NP * NH, D=D, DM1=D - 1, O=O, S=S, OFF=off, MAXROWS=MAXROWS, ORDER=order_code(order),
            Q=D // 4, HALF=D // 2, LNTH=repr(2.0 * math.log(theta)), PAIR=ROPE_PAIR[mode])
    return name, src

# ---- attention: one 32-lane wave per (query, head); online softmax over all (or same-window) keys ----
def attention(D, NH, S, OQ, OK, OV, scale, window=0, order="raster"):
    """o[t, h*D:(h+1)*D] = softmax(q_t . k_j * scale) v_j over keys j (window > 0: same window x window patch block)."""
    # q/k/v rows are read in VW-wide chunks: 32 when the head size allows, else 8 (Qwen3-VL/Qwen3.8 encoders: D = 72)
    VW = 32 if D % 32 == 0 else 8
    assert D % VW == 0, f"vision head_dim {D} must be a multiple of 8"
    G = D // VW
    name = f"v_attn_{D}x{NH}_s{S}_{OQ}_{OK}_{OV}" + (f"_w{window}" if window else "") + ("_" + order if window else "")
    L = [T('''kernel.def export("$NAME") @$NAME(%tokens: index) {
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  %heads = index.constant $NH : index
  kernel.launch.config workgroups(%tokens, %heads, %c1) workgroup_size(%c32, %c1, %c1) : index
} launch(%tokens_arg: index, %gw_arg: index, %q: buffer, %k: buffer, %v: buffer, %o: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, $MAXROWS)] : index
  %gw = index.assume %gw_arg [range(%gw_arg, 1, 4096)] : index
  %t_id = kernel.workgroup.id<x> : index
  %t = index.assume %t_id [lt(%t_id, %tokens)] : index
  %h_id = kernel.workgroup.id<y> : index
  %h = index.assume %h_id [range(%h_id, 0, $NHM1)] : index
  %lid = kernel.workitem.id<x> : index
  %lane = index.assume %lid [range(%lid, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  %cD = index.constant $D : index
  %zf = scalar.constant 0.0 : f32
  %neg_big = scalar.constant -3.0e38 : f32
  %scale = scalar.constant $SCALE : f32
  %zero32 = vector.constant 0.0 : vector<32xf32>
  %qv = buffer.view %q[%zero] : buffer -> view<[%tokens]x$Sxf32>
  %kv = buffer.view %k[%zero] : buffer -> view<[%tokens]x$Sxf32>
  %vv = buffer.view %v[%zero] : buffer -> view<[%tokens]x$Sxf32>
  %qcol = index.madd %h, %cD, %cOQ : index
  %kcol = index.madd %h, %cD, %cOK : index
  %vcol = index.madd %h, %cD, %cOV : index''', NAME=name, NH=NH, NHM1=NH - 1, MAXROWS=MAXROWS, D=D, S=S, SCALE=repr(float(scale)))]
    L.insert(1, "")
    L[0] = L[0].replace("  %qcol = index.madd", T("  %cOQ = index.constant $OQ : index\n  %cOK = index.constant $OK : index\n  %cOV = index.constant $OV : index\n", OQ=OQ, OK=OK, OV=OV) + "  %qcol = index.madd", 1)
    for g in range(G):
        L.append(T('''  %qc$g0 = index.constant $GO : index
  %qc$g1 = index.add %qcol, %qc$g0 : index
  %qc$g = index.assume %qc$g1 [range(%qc$g1, 0, $SM32)] : index
  %qr$g = vector.load %qv[%t, %qc$g] : view<[%tokens]x$Sxf32> -> vector<32xf32>
  %scv$g = vector.splat %scale : vector<32xf32>
  %q$g = vector.mulf %qr$g, %scv$g : vector<32xf32>''', g=g, GO=VW * g, SM32=S - VW, S=S))
    if window:
        L.append(order_code(order, t="%t") .replace("%py", "%qy").replace("%px", "%qx").replace("%c2o", "%c2q").replace("%c4o", "%c4q").replace("%bw", "%bwq").replace("%blk", "%blkq").replace("%sub", "%subq").replace("%ty", "%tyq").replace("%tx", "%txq").replace("%dy", "%dyq").replace("%dx", "%dxq"))
        L.append(T('''  %cwin = index.constant $WIN : index
  %qwy = index.div %qy, %cwin : index
  %qwx = index.div %qx, %cwin : index''', WIN=window))
    carried = ", ".join([f"%acc{g}_c = %zero32 : vector<32xf32>" for g in range(G)])
    ctypes = ", ".join(["vector<32xf32>"] * G)
    L.append(f'''  %m_l, %l_l, {", ".join(f"%acc{g}_l" for g in range(G))} = scf.for %j = [%lane to %tokens step %c32](%m_c = %neg_big : f32, %l_c = %zf : f32, {carried}) -> (f32, f32, {ctypes}) {{
    %bj = index.assume %j [lt(%j, %tokens)] : index''')
    if window:
        L.append(order_code(order, t="%bj").replace("%py", "%ky").replace("%px", "%kx").replace("%c2o", "%c2k").replace("%c4o", "%c4k").replace("%bw", "%bwk").replace("%blk", "%blkk").replace("%sub", "%subk").replace("%ty", "%tyk").replace("%tx", "%txk").replace("%dy", "%dyk").replace("%dx", "%dxk"))
        L.append('''    %kwy = index.div %ky, %cwin : index
    %kwx = index.div %kx, %cwin : index
    %same_y = index.cmp eq, %kwy, %qwy : index
    %same_x = index.cmp eq, %kwx, %qwx : index
    %allowed = scalar.andi %same_y, %same_x : i1''')
    else:
        L.append("    %allowed = index.cmp ult, %bj, %tokens : index")
    res = ", ".join(["f32", "f32"] + ["vector<32xf32>"] * G)
    L.append(f"    %m_n, %l_n, {', '.join(f'%acc{g}_n' for g in range(G))} = scf.if %allowed -> ({res}) {{")
    for g in range(G):
        L.append(T('''      %kc$g0 = index.constant $GO : index
      %kc$g1 = index.add %kcol, %kc$g0 : index
      %kc$g = index.assume %kc$g1 [range(%kc$g1, 0, $SM32)] : index
      %kr$g = vector.load %kv[%bj, %kc$g] : view<[%tokens]x$Sxf32> -> vector<32xf32>
      %vc$g1 = index.add %vcol, %kc$g0 : index
      %vc$g = index.assume %vc$g1 [range(%vc$g1, 0, $SM32)] : index
      %vr$g = vector.load %vv[%bj, %vc$g] : view<[%tokens]x$Sxf32> -> vector<32xf32>
      %pd$g = vector.mulf %q$g, %kr$g : vector<32xf32>
      %ps$g = vector.reduce<addf> %pd$g, %zf : vector<32xf32>, f32''', g=g, GO=VW * g, SM32=S - VW, S=S))
    ssum = "%ps0"
    for g in range(1, G):
        L.append(f"      %ss{g} = scalar.addf {ssum}, %ps{g} : f32"); ssum = f"%ss{g}"
    L.append(f'''      %s = scalar.addf {ssum}, %zf : f32
      %mx = scalar.maxnumf %m_c, %s : f32
      %dm = scalar.subf %m_c, %mx : f32
      %corr = scalar.expf<afn> %dm : f32
      %ds = scalar.subf %s, %mx : f32
      %p = scalar.expf<afn> %ds : f32
      %lc = scalar.mulf %l_c, %corr : f32
      %ln = scalar.addf %lc, %p : f32
      %corrv = vector.splat %corr : vector<32xf32>
      %pv = vector.splat %p : vector<32xf32>''')
    for g in range(G):
        L.append(f'''      %ac{g} = vector.mulf %acc{g}_c, %corrv : vector<32xf32>
      %an{g} = vector.fmaf %vr{g}, %pv, %ac{g} : vector<32xf32>''')
    L.append(f"      scf.yield %mx, %ln, {', '.join(f'%an{g}' for g in range(G))} : {res}\n    }} else {{")
    L.append(f"      scf.yield %m_c, %l_c, {', '.join(f'%acc{g}_c' for g in range(G))} : {res}\n    }}")
    L.append(f"    scf.yield %m_n, %l_n, {', '.join(f'%acc{g}_n' for g in range(G))} : {res}\n  }}")
    L.append(T('''  %mg = kernel.subgroup.reduce<maxnumf> %m_l : f32
  %dml = scalar.subf %m_l, %mg : f32
  %cl = scalar.expf<afn> %dml : f32
  %lw = scalar.mulf %l_l, %cl : f32
  %lg = kernel.subgroup.reduce<addf> %lw : f32
  %one = scalar.constant 1.0 : f32
  %invl = scalar.divf %one, %lg : f32
  %clv = vector.splat %cl : vector<32xf32>
  %stage_bytes = index.constant $SB : offset
  %stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %st = buffer.view %stage[%zero] : buffer -> view<32x$Dxf32>''', SB=32 * D * 4, D=D))
    for g in range(G):
        L.append(T('''  %aw$g = vector.mulf %acc$g_l, %clv : vector<32xf32>
  %sc$g = index.constant $GO : index
  vector.store %aw$g, %st[%lane, %sc$g] : vector<32xf32>, view<32x$Dxf32>''', g=g, GO=VW * g, D=D))
    L.append(T('''  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %ov = buffer.view %o[%zero] : buffer -> view<[%tokens]x$Exf32>
  %ocol0 = index.mul %h, %cD : index
  scf.for %col = [%lane to %cD step %c32] {
    %bcol = index.assume %col [lt(%col, %cD)] : index
    %tot = scf.for %r = [%c0 to %c32 step %c1](%a = %zf : f32) -> (f32) {
      %br = index.assume %r [range(%r, 0, 31)] : index
      %x = view.load %st[%br, %bcol] : view<32x$Dxf32> -> f32
      %na = scalar.addf %a, %x : f32
      scf.yield %na : f32
    }
    %res = scalar.mulf %tot, %invl : f32
    %oc0 = index.add %ocol0, %bcol : index
    %oc = index.assume %oc0 [lt(%oc0, $E)] : index
    view.store %res, %ov[%t, %oc] : f32, view<[%tokens]x$Exf32>
  }
  kernel.return
}
''', D=D, E=D * NH))
    return name, "\n".join(L).replace("vector<32xf32>", f"vector<{VW}xf32>")

# ---- merges ----
def avgpool(E, k, scale):
    """gemma4v: raster [gh, gw, E] -> [(gh/k)*(gw/k), E], mean over k x k, times scale."""
    name = f"v_avgpool{k}_{E}"
    return name, T('''kernel.def export("$NAME") @$NAME(%out_tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%out_tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%out_tokens_arg: index, %gw_arg: index, %tokens_arg: index, %x: buffer, %y: buffer) {
  %out_tokens = index.assume %out_tokens_arg [range(%out_tokens_arg, 1, $MAXROWS)] : index
  %gw = index.assume %gw_arg [range(%gw_arg, $K, 4096)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, $MAXROWS)] : index
  %o_id = kernel.workgroup.id<x> : index
  %o = index.assume %o_id [lt(%o_id, %out_tokens)] : index
  %lid = kernel.workitem.id<x> : index
  %lane = index.assume %lid [range(%lid, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %ck = index.constant $K : index
  %cE = index.constant $E : index
  %zf = scalar.constant 0.0 : f32
  %scale = scalar.constant $SCALE : f32
  %ow0 = index.div %gw, %ck : index
  %ow = index.assume %ow0 [range(%ow0, 1, 4096)] : index
  %oy = index.div %o, %ow : index
  %ox = index.rem %o, %ow : index
  %xv = buffer.view %x[%zero] : buffer -> view<[%tokens]x$Exf32>
  %yv = buffer.view %y[%zero] : buffer -> view<[%out_tokens]x$Exf32>
  scf.for %e = [%lane to %cE step %c256] {
    %be = index.assume %e [lt(%e, %cE)] : index
    %sum = scf.for %i = [%c0 to %cKK step %c1](%a = %zf : f32) -> (f32) {
      %dy = index.div %i, %ck : index
      %dx = index.rem %i, %ck : index
      %yy = index.madd %oy, %ck, %dy : index
      %xx = index.madd %ox, %ck, %dx : index
      %src = index.madd %yy, %gw, %xx : index
      %bs = index.assume %src [lt(%src, %tokens)] : index
      %val = view.load %xv[%bs, %be] : view<[%tokens]x$Exf32> -> f32
      %na = scalar.addf %a, %val : f32
      scf.yield %na : f32
    }
    %r = scalar.mulf %sum, %scale : f32
    view.store %r, %yv[%o, %be] : f32, view<[%out_tokens]x$Exf32>
  }
  kernel.return
}
''', NAME=name, E=E, K=k, SCALE=repr(float(scale) / (k * k)), MAXROWS=MAXROWS).replace("%cKK", "%ckk").replace("  %ck = index.constant", f"  %ckk = index.constant {k * k} : index\n  %ck = index.constant", 1)

def shuffle(E, m):
    """muse-glimmer pixel shuffle: raster [gh, gw, E] -> [(gh/m)*(gw/m), E*m*m], out[o][c*m*m + s] = x[(oy*m+ry, ox*m+rx)][c]."""
    name = f"v_shuffle{m}_{E}"; MM = m * m
    return name, T('''kernel.def export("$NAME") @$NAME(%out_tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%out_tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%out_tokens_arg: index, %gw_arg: index, %tokens_arg: index, %x: buffer, %y: buffer) {
  %out_tokens = index.assume %out_tokens_arg [range(%out_tokens_arg, 1, $MAXROWS)] : index
  %gw = index.assume %gw_arg [range(%gw_arg, $M, 4096)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, $MAXROWS)] : index
  %o_id = kernel.workgroup.id<x> : index
  %o = index.assume %o_id [lt(%o_id, %out_tokens)] : index
  %lid = kernel.workitem.id<x> : index
  %lane = index.assume %lid [range(%lid, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c256 = index.constant 256 : index
  %cm = index.constant $M : index
  %cmm = index.constant $MM : index
  %cEO = index.constant $EO : index
  %ow0 = index.div %gw, %cm : index
  %ow = index.assume %ow0 [range(%ow0, 1, 4096)] : index
  %oy = index.div %o, %ow : index
  %ox = index.rem %o, %ow : index
  %xv = buffer.view %x[%zero] : buffer -> view<[%tokens]x$Exf32>
  %yv = buffer.view %y[%zero] : buffer -> view<[%out_tokens]x$EOxf32>
  scf.for %f = [%lane to %cEO step %c256] {
    %bf = index.assume %f [lt(%f, %cEO)] : index
    %c = index.div %bf, %cmm : index
    %bc = index.assume %c [lt(%c, $E)] : index
    %s = index.rem %bf, %cmm : index
    %ry = index.div %s, %cm : index
    %rx = index.rem %s, %cm : index
    %yy = index.madd %oy, %cm, %ry : index
    %xx = index.madd %ox, %cm, %rx : index
    %src = index.madd %yy, %gw, %xx : index
    %bs = index.assume %src [lt(%src, %tokens)] : index
    %val = view.load %xv[%bs, %bc] : view<[%tokens]x$Exf32> -> f32
    view.store %val, %yv[%o, %bf] : f32, view<[%out_tokens]x$EOxf32>
  }
  kernel.return
}
''', NAME=name, E=E, M=m, MM=MM, EO=E * MM, MAXROWS=MAXROWS)

def place_cols(N, NT, off):
    """y[r, off + c] = x[r, c] for x [rows, N] and y [rows, NT]: one projection into its slice of a wider output row
    (Qwen3-VL DeepStack: the main projection followed by one per deepstack layer)."""
    assert off + N <= NT
    name = f"v_place_{N}_{NT}_{off}"
    return name, T('''kernel.def export("$NAME") @$NAME(%rows: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%rows, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%rows_arg: index, %x: buffer, %y: buffer) {
  %rows = index.assume %rows_arg [range(%rows_arg, 1, $MAXROWS)] : index
  %row_id = kernel.workgroup.id<x> : index
  %row = index.assume %row_id [lt(%row_id, %rows)] : index
  %lid = kernel.workitem.id<x> : index
  %lane = index.assume %lid [range(%lid, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c256 = index.constant 256 : index
  %cN = index.constant $N : index
  %coff = index.constant $OFF : index
  %xv = buffer.view %x[%zero] : buffer -> view<[%rows]x$Nxf32>
  %yv = buffer.view %y[%zero] : buffer -> view<[%rows]x$NTxf32>
  scf.for %c = [%lane to %cN step %c256] {
    %bc = index.assume %c [lt(%c, %cN)] : index
    %val = view.load %xv[%row, %bc] : view<[%rows]x$Nxf32> -> f32
    %d = index.add %bc, %coff : index
    %bd = index.assume %d [lt(%d, $NT)] : index
    view.store %val, %yv[%row, %bd] : f32, view<[%rows]x$NTxf32>
  }
  kernel.return
}
''', NAME=name, N=N, NT=NT, OFF=off, MAXROWS=MAXROWS)

# ---- clamp with bounds from 1-element parameter tensors (Gemma4ClippableLinear) ----
def clamp():
    name = "v_clamp"
    return name, '''kernel.def export("v_clamp") @v_clamp(%n: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %c255 = index.constant 255 : index
  %n_up = index.add %n, %c255 : index
  %groups = index.div %n_up, %c256 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%n_arg: index, %x: buffer, %lo: buffer, %hi: buffer, %y: buffer) {
  %n = index.assume %n_arg [range(%n_arg, 1, 268435456)] : index
  %c256 = index.constant 256 : index
  %c0 = index.constant 0 : index
  %gid = kernel.workgroup.id<x> : index
  %lid = kernel.workitem.id<x> : index
  %i = index.madd %gid, %c256, %lid : index
  %live = index.cmp ult, %i, %n : index
  scf.if %live {
    %bi = index.assume %i [lt(%i, %n)] : index
    %zero = index.constant 0 : offset
    %lov = buffer.view %lo[%zero] : buffer -> view<1xf32>
    %hiv = buffer.view %hi[%zero] : buffer -> view<1xf32>
    %l = view.load %lov[%c0] : view<1xf32> -> f32
    %h = view.load %hiv[%c0] : view<1xf32> -> f32
    %xv = buffer.view %x[%zero] : buffer -> view<[%n]xf32>
    %a = view.load %xv[%bi] : view<[%n]xf32> -> f32
    %b = scalar.maxnumf %a, %l : f32
    %r = scalar.minnumf %b, %h : f32
    %yv = buffer.view %y[%zero] : buffer -> view<[%n]xf32>
    view.store %r, %yv[%bi] : f32, view<[%n]xf32>
  }
  kernel.return
}
'''

# ---- Pillow-compatible separable resample (llama.cpp mtmd img_tool::resize_pillow), u8 -> u8 ----
FILTERS = {"bilinear": 1.0, "bicubic": 2.0, "lanczos": 3.0}
def _filter_code(filt, x, out):
    if filt == "bilinear":
        return f'''      {out}_a = scalar.absf {x} : f32
      {out}_b = scalar.subf %onef, {out}_a : f32
      {out} = scalar.maxnumf {out}_b, %zf : f32'''
    if filt == "bicubic":   # a = -0.5
        return f'''      {out}_x = scalar.absf {x} : f32
      {out}_lt1 = scalar.cmpf olt, {out}_x, %onef : f32
      {out}_lt2 = scalar.cmpf olt, {out}_x, %twof : f32
      {out}_p1a = scalar.mulf {out}_x, %c1p5 : f32
      {out}_p1b = scalar.subf {out}_p1a, %c2p5 : f32
      {out}_p1c = scalar.mulf {out}_p1b, {out}_x : f32
      {out}_p1d = scalar.mulf {out}_p1c, {out}_x : f32
      {out}_p1 = scalar.addf {out}_p1d, %onef : f32
      {out}_p2a = scalar.subf {out}_x, %fivef : f32
      {out}_p2b = scalar.mulf {out}_p2a, {out}_x : f32
      {out}_p2c = scalar.addf {out}_p2b, %eightf : f32
      {out}_p2d = scalar.mulf {out}_p2c, {out}_x : f32
      {out}_p2e = scalar.subf {out}_p2d, %fourf : f32
      {out}_p2 = scalar.mulf {out}_p2e, %mhalf : f32
      {out}_q = scf.select {out}_lt2, {out}_p2, %zf : f32
      {out} = scf.select {out}_lt1, {out}_p1, {out}_q : f32'''
    if filt == "lanczos":   # sinc(x) sinc(x/3) on [-3, 3)
        return f'''      {out}_in0 = scalar.cmpf oge, {x}, %m3f : f32
      {out}_in1 = scalar.cmpf olt, {x}, %threef : f32
      {out}_in = scalar.andi {out}_in0, {out}_in1 : i1
      {out}_px = scalar.mulf {x}, %pif : f32
      {out}_px3 = scalar.divf {out}_px, %threef : f32
      {out}_s1 = scalar.sinf<afn> {out}_px : f32
      {out}_s3 = scalar.sinf<afn> {out}_px3 : f32
      {out}_num = scalar.mulf {out}_s1, {out}_s3 : f32
      {out}_den0 = scalar.mulf {out}_px, {out}_px3 : f32
      {out}_tiny = scalar.cmpf olt, {out}_den0, %tinyf : f32
      {out}_den1 = scalar.absf {out}_den0 : f32
      {out}_den = scalar.maxnumf {out}_den1, %tinyf : f32
      {out}_q = scalar.divf {out}_num, {out}_den : f32
      {out}_v = scf.select {out}_tiny, %onef, {out}_q : f32
      {out} = scf.select {out}_in, {out}_v, %zf : f32'''
    raise ValueError(filt)

def resize_pass(filt, axis, cin):
    """axis 'h': [rows, in_w, cin] u8 -> [rows, out_w, 3]; axis 'v': [in_h, cols, 3] -> [out_h, cols, 3]. One thread per output pixel."""
    sup = FILTERS[filt]; name = f"v_resize_{filt}_{axis}{cin}"
    if axis == "h":
        idx_in = "%src_idx = index.madd %row, %in_n, %xi : index\n      %el0 = index.mul %src_idx, %ccin : index"
        idx_out = "%dst_idx = index.madd %row, %out_n, %o : index"
    else:
        idx_in = "%src_idx = index.madd %xi, %cols, %col : index\n      %el0 = index.mul %src_idx, %ccin : index"
        idx_out = "%dst_idx = index.madd %o, %cols, %col : index"
    other = "%row" if axis == "h" else "%col"
    src = T('''kernel.def export("$NAME") @$NAME(%count: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %c255 = index.constant 255 : index
  %n_up = index.add %count, %c255 : index
  %groups = index.div %n_up, %c256 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%count_arg: index, %in_n_arg: index, %out_n_arg: index, %other_arg: index, %src: buffer, %dst: buffer) {
  %count = index.assume %count_arg [range(%count_arg, 1, 268435456)] : index
  %in_n = index.assume %in_n_arg [range(%in_n_arg, 1, 65536)] : index
  %out_n = index.assume %out_n_arg [range(%out_n_arg, 1, 65536)] : index
  %other = index.assume %other_arg [range(%other_arg, 1, 65536)] : index
  %cols = index.add %other, %c0x : index
  %c256 = index.constant 256 : index
  %gid = kernel.workgroup.id<x> : index
  %lid = kernel.workitem.id<x> : index
  %i = index.madd %gid, %c256, %lid : index
  %live = index.cmp ult, %i, %count : index
  scf.if %live {
    %zero = index.constant 0 : offset
    %c0 = index.constant 0 : index
    %c1 = index.constant 1 : index
    %c3 = index.constant 3 : index
    %ccin = index.constant $CIN : index
    %zf = scalar.constant 0.0 : f32
    %onef = scalar.constant 1.0 : f32
    %halff = scalar.constant 0.5 : f32
    %twof = scalar.constant 2.0 : f32
    %threef = scalar.constant 3.0 : f32
    %m3f = scalar.constant -3.0 : f32
    %fourf = scalar.constant 4.0 : f32
    %fivef = scalar.constant 5.0 : f32
    %eightf = scalar.constant 8.0 : f32
    %c1p5 = scalar.constant 1.5 : f32
    %c2p5 = scalar.constant 2.5 : f32
    %mhalf = scalar.constant -0.5 : f32
    %pif = scalar.constant 3.141592653589793 : f32
    %tinyf = scalar.constant 1.0e-12 : f32
    %fxp = scalar.constant 4194304.0 : f32
    %o = index.rem %i, %out_n : index
    $OTHERV = index.div %i, %out_n : index
    %in_i = index.cast %in_n : index to i32
    %out_i = index.cast %out_n : index to i32
    %in_f = scalar.sitofp %in_i : i32 to f32
    %out_f = scalar.sitofp %out_i : i32 to f32
    %scale = scalar.divf %in_f, %out_f : f32
    %fs = scalar.maxnumf %scale, %onef : f32
    %supc = scalar.constant $SUP : f32
    %support = scalar.mulf %supc, %fs : f32
    %ss = scalar.divf %onef, %fs : f32
    %oi = index.cast %o : index to i32
    %of = scalar.sitofp %oi : i32 to f32
    %oh = scalar.addf %of, %halff : f32
    %center = scalar.mulf %oh, %scale : f32
    %lo0 = scalar.subf %center, %support : f32
    %lo1 = scalar.addf %lo0, %halff : f32
    %lo2 = scalar.fptosi %lo1 : f32 to i32
    %zi = scalar.constant 0 : i32
    %lo3 = scalar.maxsi %lo2, %zi : i32
    %hi0 = scalar.addf %center, %support : f32
    %hi1 = scalar.addf %hi0, %halff : f32
    %hi2 = scalar.fptosi %hi1 : f32 to i32
    %hi3 = scalar.minsi %hi2, %in_i : i32
    %lo4 = scalar.assume %lo3 [range(%lo3, 0, 65536)] : i32
    %hi4 = scalar.assume %hi3 [range(%hi3, 0, 65536)] : i32
    %xlo = index.cast %lo4 : i32 to index
    %xhi = index.cast %hi4 : i32 to index
    %ww = scf.for %xi = [%xlo to %xhi step %c1](%acc = %zf : f32) -> (f32) {
      %xii = index.cast %xi : index to i32
      %xf = scalar.sitofp %xii : i32 to f32
      %d0 = scalar.subf %xf, %center : f32
      %d1 = scalar.addf %d0, %halff : f32
      %d = scalar.mulf %d1, %ss : f32
$FILTER_W
      %na = scalar.addf %acc, %wv : f32
      scf.yield %na : f32
    }
    %wnz = scalar.cmpf one, %ww, %zf : f32
    %wdiv = scf.select %wnz, %ww, %onef : f32
    %srcn = index.constant 268435456 : index
    %sv = buffer.view %src[%zero] : buffer -> view<[%srcn]xi8>
    %bias = scalar.constant 2097152 : i32
    %s0, %s1, %s2 = scf.for %xi = [%xlo to %xhi step %c1](%a0 = %bias : i32, %a1 = %bias : i32, %a2 = %bias : i32) -> (i32, i32, i32) {
      %xii = index.cast %xi : index to i32
      %xf = scalar.sitofp %xii : i32 to f32
      %d0 = scalar.subf %xf, %center : f32
      %d1 = scalar.addf %d0, %halff : f32
      %d = scalar.mulf %d1, %ss : f32
$FILTER_W
      %wn = scalar.divf %wv, %wdiv : f32
      %wfx = scalar.mulf %wn, %fxp : f32
      %neg = scalar.cmpf olt, %wn, %zf : f32
      %rnd = scf.select %neg, %mhalf, %halff : f32
      %wr = scalar.addf %wfx, %rnd : f32
      %wi = scalar.fptosi %wr : f32 to i32
      $IDXIN
      %el = index.assume %el0 [lt(%el0, %srcn)] : index
      %el1r = index.add %el0, %c1 : index
      %el1 = index.assume %el1r [lt(%el1r, %srcn)] : index
      %el2r = index.add %el1r, %c1 : index
      %el2 = index.assume %el2r [lt(%el2r, %srcn)] : index
      %p0r = view.load %sv[%el] : view<[%srcn]xi8> -> i8
      %p1r = view.load %sv[%el1] : view<[%srcn]xi8> -> i8
      %p2r = view.load %sv[%el2] : view<[%srcn]xi8> -> i8
      %p0 = scalar.extui %p0r : i8 to i32
      %p1 = scalar.extui %p1r : i8 to i32
      %p2 = scalar.extui %p2r : i8 to i32
      %m0 = scalar.muli %p0, %wi : i32
      %m1 = scalar.muli %p1, %wi : i32
      %m2 = scalar.muli %p2, %wi : i32
      %n0 = scalar.addi %a0, %m0 : i32
      %n1 = scalar.addi %a1, %m1 : i32
      %n2 = scalar.addi %a2, %m2 : i32
      scf.yield %n0, %n1, %n2 : i32, i32, i32
    }
    %sh = scalar.constant 22 : i32
    %c255i = scalar.constant 255 : i32
    $IDXOUT
    %dstn = index.constant 268435456 : index
    %dv = buffer.view %dst[%zero] : buffer -> view<[%dstn]xi8>
    %de0 = index.mul %dst_idx, %c3 : index
$STORES
  }
  kernel.return
}
''', NAME=name, CIN=cin, SUP=repr(sup), OTHERV=other, IDXIN=idx_in, IDXOUT=idx_out,
       FILTER_W=_filter_code(filt, "%d", "%wv"),
       STORES="\n".join(f'''    %q{c}a = scalar.shrsi %s{c}, %sh : i32
    %q{c}b = scalar.maxsi %q{c}a, %zi : i32
    %q{c}c = scalar.minsi %q{c}b, %c255i : i32
    %q{c} = scalar.trunci %q{c}c : i32 to i8
    %dc{c}r = index.add %de0, %c{['0', '1', '1'][c]}x{c} : index
    %dc{c} = index.assume %dc{c}r [lt(%dc{c}r, %dstn)] : index
    view.store %q{c}, %dv[%dc{c}] : i8, view<[%dstn]xi8>''' for c in range(3)))
    src = src.replace("%c1x1", "%c1").replace("%c0x0", "%c0").replace("%c1x2", "%c2x")
    src = src.replace("    %de0 = index.mul %dst_idx, %c3 : index", "    %de0 = index.mul %dst_idx, %c3 : index\n    %c2x = index.constant 2 : index")
    src = src.replace("  %cols = index.add %other, %c0x : index\n", "")
    if axis == "v":
        src = src.replace("$OTHERV", "")
        src = src.replace("    %col = index.div %i, %out_n : index", "    %o_v = index.div %i, %other : index\n    %col = index.rem %i, %other : index").replace("    %o = index.rem %i, %out_n : index\n", "")
        src = src.replace("%oi = index.cast %o : index to i32", "%oi = index.cast %o_v : index to i32").replace("%dst_idx = index.madd %o, %cols, %col : index", "%dst_idx = index.madd %o_v, %other, %col : index").replace("%src_idx = index.madd %xi, %cols, %col : index", "%src_idx = index.madd %xi, %other, %col : index")
    return name, src
