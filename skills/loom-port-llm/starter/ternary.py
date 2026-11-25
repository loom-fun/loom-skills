"""Ternary rotated-basis checkpoints (PrismML Ternary Bonsai; llama_config "hadamard" + PTQ1_0 weights).

Two pieces, both emitted only when the model has them (no other model's programs change):

1. PTQ1_0 weights (GGML type 143, gguf_ext.py): 128 trits {-1, 0, 1} and one F16 scale per 28-byte block. The value helper
   (@g_ptq1_0_values, GEMM dequant and embedding) and the int8 code helper (@g_ptq1_0_icodes, the decode matvecs and the head:
   signed trit codes x one scale, as q8_0) decode one 32-value group of a block.

2. prism.hadamard: a folded weight W D H expects the input H (D x) (D: +/-1 per input column, H: normalized 1024-point Sylvester
   Walsh-Hadamard transform per block). The rotation is applied where the activations are quantized for the matmuls (the int8
   copy of the decode matvecs, the F16 row-scaled copy of the GEMMs): every trunk quantizer and the head's become rotating ones
   (wiring()), so the f32 activation stays in the model basis for the unrotated F32 projections that read it (gated delta-net
   alpha / beta) and for the MTP block (Q4_0, not folded). ssm_out takes its input permuted from the layer's tiled head order to
   the grouped order of the fold (gdn_v_grouped). The token embedding stores rotated rows: the lookup applies D (H z).
"""
import re

BLK = 1024   # prism.hadamard.block_size this module implements (inspect_gguf.py records it; asserted in setup())

PTQ1_0_HELPERS = r"""
// PTQ1_0 trit codes of 16 bytes at weight 3^n (%p): ((q * 3^n mod 256) * 3) >> 8 is 0 below 86, 1 below 171, else 2; code = trit - 1.
func.def inline @g_ptq1_0_tc16(%b: vector<16xi8>, %p: vector<16xi8>) -> (vector<16xi8>) {
  %m = vector.muli %b, %p : vector<16xi8>
  %t1 = vector.splati 86 : vector<16xi8>
  %t2 = vector.splati -85 : vector<16xi8>
  %ge1 = vector.cmpi uge, %m, %t1 : vector<16xi8> -> vector<16xi1>
  %ge2 = vector.cmpi uge, %m, %t2 : vector<16xi8> -> vector<16xi1>
  %one = vector.splati 1 : vector<16xi8>
  %zero = vector.splati 0 : vector<16xi8>
  %mone = vector.splati -1 : vector<16xi8>
  %a = vector.select %ge1, %zero, %mone : vector<16xi8>
  %c = vector.select %ge2, %one, %a : vector<16xi8>
  func.return %c : vector<16xi8>
}

// PTQ1_0 block (28 bytes): qs[24] then qh[2] then F16 d. Value v of the block: v < 80: byte v % 16, trit v / 16; v < 120: byte
// 16 + (v - 80) % 8, trit (v - 80) / 8; else byte 24 + (v - 120) % 2, trit (v - 120) / 2. Group %group (0..3) = values 32g..32g+31.
// Branch-free (lanes of one wave hold different groups of a block): each half of the group is 16 bytes x 16 powers of 3, picked by
// selects: g0 = A*1 | A*3, g1 = A*9 | A*27, g2 = A*81 | B*(1,3), g3 = B*(9,27) | B*81, Q*(1,1,3,3,9,9,27,27)  (A = bytes 0..15, B =
// bytes 16..23 twice, Q = bytes 24, 25 four times).
func.def inline @g_ptq1_0_icodes(%w: buffer, %origin: offset, %group: index) -> (vector<32xi8>, f32) {
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %o16 = index.constant 16 : offset
  %o24 = index.constant 24 : offset
  %o26 = index.constant 26 : offset
  %a_view = buffer.view %w[%origin] : buffer -> view<16xi8>
  %b_origin = index.add %origin, %o16 : offset
  %b_view = buffer.view %w[%b_origin] : buffer -> view<8xi8>
  %h_origin = index.add %origin, %o24 : offset
  %h_view = buffer.view %w[%h_origin] : buffer -> view<2xi8>
  %d_origin = index.add %origin, %o26 : offset
  %d_view = buffer.view %w[%d_origin] : buffer -> view<1xf16>
  %d_half = view.load %d_view[0] : view<1xf16> -> f16
  %d = scalar.extf %d_half : f16 to f32
  %a = vector.load %a_view[0] : view<16xi8> -> vector<16xi8>
  %b8 = vector.load %b_view[0] : view<8xi8> -> vector<8xi8>
  %h2 = vector.load %h_view[0] : view<2xi8> -> vector<2xi8>
  %bb = vector.concat<0> %b8, %b8 : vector<8xi8>, vector<8xi8> -> vector<16xi8>
  %h4 = vector.concat<0> %h2, %h2 : vector<2xi8>, vector<2xi8> -> vector<4xi8>
  %h8 = vector.concat<0> %h4, %h4 : vector<4xi8>, vector<4xi8> -> vector<8xi8>
  %bq = vector.concat<0> %b8, %h8 : vector<8xi8>, vector<8xi8> -> vector<16xi8>
  %is1 = index.cmp eq, %group, %c1 : index
  %is2 = index.cmp eq, %group, %c2 : index
  %is3 = index.cmp eq, %group, %c3 : index
  %le1 = index.cmp ult, %group, %c2 : index
  %m1 = vector.splat %is1 : vector<16xi1>
  %m2 = vector.splat %is2 : vector<16xi1>
  %m3 = vector.splat %is3 : vector<16xi1>
  %ml1 = vector.splat %le1 : vector<16xi1>
  %lo = vector.select %m3, %bb, %a : vector<16xi8>
  %hi_23 = vector.select %m2, %bb, %bq : vector<16xi8>
  %hi = vector.select %ml1, %a, %hi_23 : vector<16xi8>
  %i1 = scalar.constant 1 : i8
  %i3 = scalar.constant 3 : i8
  %i9 = scalar.constant 9 : i8
  %i27 = scalar.constant 27 : i8
  %i81 = scalar.constant 81 : i8
  %v1 = vector.splat %i1 : vector<16xi8>
  %v3 = vector.splat %i3 : vector<16xi8>
  %v9 = vector.splat %i9 : vector<16xi8>
  %v27 = vector.splat %i27 : vector<16xi8>
  %v81 = vector.splat %i81 : vector<16xi8>
  %p3lo = vector.from_elements %i9, %i9, %i9, %i9, %i9, %i9, %i9, %i9, %i27, %i27, %i27, %i27, %i27, %i27, %i27, %i27 : vector<16xi8>
  %p2hi = vector.from_elements %i1, %i1, %i1, %i1, %i1, %i1, %i1, %i1, %i3, %i3, %i3, %i3, %i3, %i3, %i3, %i3 : vector<16xi8>
  %p3hi = vector.from_elements %i81, %i81, %i81, %i81, %i81, %i81, %i81, %i81, %i1, %i1, %i3, %i3, %i9, %i9, %i27, %i27 : vector<16xi8>
  %plo_01 = vector.select %m1, %v9, %v1 : vector<16xi8>
  %plo_012 = vector.select %m2, %v81, %plo_01 : vector<16xi8>
  %plo = vector.select %m3, %p3lo, %plo_012 : vector<16xi8>
  %phi_01 = vector.select %m1, %v27, %v3 : vector<16xi8>
  %phi_012 = vector.select %m2, %p2hi, %phi_01 : vector<16xi8>
  %phi = vector.select %m3, %p3hi, %phi_012 : vector<16xi8>
  %clo = func.call @g_ptq1_0_tc16(%lo, %plo) : (vector<16xi8>, vector<16xi8>) -> (vector<16xi8>)
  %chi = func.call @g_ptq1_0_tc16(%hi, %phi) : (vector<16xi8>, vector<16xi8>) -> (vector<16xi8>)
  %codes = vector.concat<0> %clo, %chi : vector<16xi8>, vector<16xi8> -> vector<32xi8>
  func.return %codes, %d : vector<32xi8>, f32
}

func.def inline @g_ptq1_0_values(%w: buffer, %origin: offset, %group: index) -> (vector<32xf32>) {
  %codes, %d = func.call @g_ptq1_0_icodes(%w, %origin, %group) : (buffer, offset, index) -> (vector<32xi8>, f32)
  %f = vector.sitofp %codes : vector<32xi8> to vector<32xf32>
  %ds = vector.splat %d : vector<32xf32>
  %values = vector.mulf %f, %ds : vector<32xf32>
  func.return %values : vector<32xf32>
}
"""


def signs_rodata(cfg):
    """One read-only table per sign width: the +/-1 signs as bits (set = -1), little-endian 32-bit words."""
    had = cfg["hadamard"]
    return "".join(f'global.rodata.def @g_had_signs_{w} = align(64) bytes("{bits}")\n' for w, bits in sorted(had["signs"].items(), key=lambda kv: int(kv[0])))


def fwht(lds, nblk, lanes, pfx):
    """IR: in-place normalized Walsh-Hadamard transform of `nblk` consecutive 1024-value blocks held in workgroup buffer %{lds}
    (f32), all `lanes` lanes of the workgroup taking part (a barrier before and after every stage), then x 1/32."""
    n_bf = nblk * BLK // 2   # butterflies per stage
    per = (n_bf + lanes - 1) // lanes
    return f"""  %{pfx}_n = index.constant {nblk * BLK} : index
  %{pfx}_v = buffer.view %{lds}[%zero] : buffer -> view<{nblk * BLK}xf32>
  %{pfx}_nbf = index.constant {n_bf} : index
  %{pfx}_per = index.constant {per} : index
  %{pfx}_lanes = index.constant {lanes} : index
  %{pfx}_c1 = index.constant 1 : index
  %{pfx}_c0 = index.constant 0 : index
  %{pfx}_c10 = index.constant 10 : index
  %{pfx}_c2 = index.constant 2 : index
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %{pfx}_hend = scf.for %{pfx}_s = [%{pfx}_c0 to %{pfx}_c10 step %{pfx}_c1](%{pfx}_hc = %{pfx}_c1 : index) -> (index) {{
    %{pfx}_h = index.assume %{pfx}_hc [range(%{pfx}_hc, 1, 512)] : index
    scf.for %{pfx}_k = [%{pfx}_c0 to %{pfx}_per step %{pfx}_c1] unroll {{
      %{pfx}_p0 = index.mul %{pfx}_k, %{pfx}_lanes : index
      %{pfx}_p = index.add %{pfx}_p0, %lane : index
      %{pfx}_ok = index.cmp ult, %{pfx}_p, %{pfx}_nbf : index
      scf.if %{pfx}_ok {{
        %{pfx}_hi = index.div %{pfx}_p, %{pfx}_h : index
        %{pfx}_lo = index.rem %{pfx}_p, %{pfx}_h : index
        %{pfx}_hi2 = index.mul %{pfx}_hi, %{pfx}_h : index
        %{pfx}_hi3 = index.add %{pfx}_hi2, %{pfx}_hi2 : index
        %{pfx}_i0 = index.add %{pfx}_hi3, %{pfx}_lo : index
        %{pfx}_i = index.assume %{pfx}_i0 [range(%{pfx}_i0, 0, {nblk * BLK - 1})] : index
        %{pfx}_j0 = index.add %{pfx}_i, %{pfx}_h : index
        %{pfx}_j = index.assume %{pfx}_j0 [range(%{pfx}_j0, 0, {nblk * BLK - 1})] : index
        %{pfx}_a = view.load %{pfx}_v[%{pfx}_i] : view<{nblk * BLK}xf32> -> f32
        %{pfx}_b = view.load %{pfx}_v[%{pfx}_j] : view<{nblk * BLK}xf32> -> f32
        %{pfx}_sum = scalar.addf %{pfx}_a, %{pfx}_b : f32
        %{pfx}_dif = scalar.subf %{pfx}_a, %{pfx}_b : f32
        view.store %{pfx}_sum, %{pfx}_v[%{pfx}_i] : f32, view<{nblk * BLK}xf32>
        view.store %{pfx}_dif, %{pfx}_v[%{pfx}_j] : f32, view<{nblk * BLK}xf32>
      }}
    }}
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    %{pfx}_h2 = index.mul %{pfx}_h, %{pfx}_c2 : index
    scf.yield %{pfx}_h2 : index
  }}
"""


def _uniq(text, p):
    """Prefix every value `text` defines (and its uses) with p: the same snippet emitted twice in one kernel."""
    names = set(re.findall(r"%(\w+) = ", text)) | set(re.findall(r"scf\.for %(\w+) = ", text)) | set(re.findall(r"\(%(\w+) = ", text))
    for n in sorted(names, key=len, reverse=True):
        text = re.sub(r"%" + n + r"\b", "%" + p + n, text)
    return text


def _sign_of(width, idx, pfx):
    """IR: f32 %{pfx}_sg = -1.0 or 1.0, the sign of input column %{idx} (an index < width) of a width-`width` input."""
    return f"""  %{pfx}_tab = global.load @g_had_signs_{width} : buffer
  %{pfx}_tv = buffer.view %{pfx}_tab[%zero] : buffer -> view<{width // 32}xi32>
  %{pfx}_c32 = index.constant 32 : index
  %{pfx}_wi0 = index.div %{idx}, %{pfx}_c32 : index
  %{pfx}_wi = index.assume %{pfx}_wi0 [range(%{pfx}_wi0, 0, {width // 32 - 1})] : index
  %{pfx}_bi = index.rem %{idx}, %{pfx}_c32 : index
  %{pfx}_word = view.load %{pfx}_tv[%{pfx}_wi] : view<{width // 32}xi32> -> i32
  %{pfx}_sh = index.cast %{pfx}_bi : index to i32
  %{pfx}_bit0 = scalar.shrui %{pfx}_word, %{pfx}_sh : i32
  %{pfx}_one = scalar.constant 1 : i32
  %{pfx}_bit = scalar.andi %{pfx}_bit0, %{pfx}_one : i32
  %{pfx}_neg = scalar.cmpi ne, %{pfx}_bit, %zero_i : i32
  %{pfx}_pf = scalar.constant 1.0 : f32
  %{pfx}_nf = scalar.constant -1.0 : f32
  %{pfx}_sg = scf.select %{pfx}_neg, %{pfx}_nf, %{pfx}_pf : f32
"""


def _gather(width, perm, gdn):
    """IR (lanes of one workgroup, block %blk of row %row): load this lane's 4 inputs of the block into LDS %buf, in the fold's
    column order (perm: tiled -> grouped heads) and multiplied by the column's sign."""
    head = gdn["head_v"] if gdn else 128
    nk = gdn["k_heads"] if gdn else 1
    rep = (gdn["v_heads"] // gdn["k_heads"]) if gdn else 1
    src = "%col"
    pm = ""
    if perm:   # grouped column g = d + hd (r + rep k) reads tiled column d + hd (k + nk r)
        src = "%src"
        pm = f"""      %hd = index.constant {head} : index
      %rep = index.constant {rep} : index
      %nk = index.constant {nk} : index
      %hidx = index.div %col, %hd : index
      %dd = index.rem %col, %hd : index
      %rr = index.rem %hidx, %rep : index
      %kk = index.div %hidx, %rep : index
      %t0 = index.mul %nk, %rr : index
      %t1 = index.add %kk, %t0 : index
      %t2 = index.mul %t1, %hd : index
      %src0 = index.add %t2, %dd : index
      %src = index.assume %src0 [range(%src0, 0, {width - 1})] : index
"""
    return f"""  %buf_v = buffer.view %buf[%zero] : buffer -> view<{BLK}xf32>
  %xrow = buffer.view %x[%zero] : buffer -> view<[%rows]x{width}xf32>
  %blk_base = index.mul %blk, %c1024 : index
  scf.for %e = [%c0 to %c4 step %c1] unroll {{
      %e4 = index.mul %lane, %c4 : index
      %ei0 = index.add %e4, %e : index
      %ei = index.assume %ei0 [range(%ei0, 0, 1023)] : index
      %col0 = index.add %blk_base, %ei : index
      %col = index.assume %col0 [range(%col0, 0, {width - 1})] : index
{pm}      %xv = view.load %xrow[%row, {src}] : view<[%rows]x{width}xf32> -> f32
{_sign_of(width, "col", "sg").replace(chr(10) + "  ", chr(10) + "      ")}      %xs = scalar.mulf %xv, %sg_sg : f32
      view.store %xs, %buf_v[%ei] : f32, view<{BLK}xf32>
  }}
"""


def q8_kernel(width, perm, gdn):
    """g_quant_q8 with the rotation: grid (width / 1024 blocks, rows), 256 lanes per block; the same arguments and the same
    output layout as g_quant_q8 (int8 codes per 32-value group, then (d as F16-rounded f32, code sum) per group)."""
    name = f"g_had_q8_w{width}{'p' if perm else ''}"
    return f"""
kernel.def export("{name}") @{name}(%rows: index, %width: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %nb = index.constant {width // BLK} : index
  kernel.launch.config workgroups(%nb, %rows, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%rows_arg: index, %width_arg: index, %x: buffer, %xq: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 8192)] : index
  %width = index.constant {width} : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c4 = index.constant 4 : index
  %c32 = index.constant 32 : index
  %c1024 = index.constant 1024 : index
  %sg_c32w = index.constant {width // 32} : index
  %zero_f = scalar.constant 0.0 : f32
  %zero_i = scalar.constant 0 : i32
  %c127 = scalar.constant 127.0 : f32
  %zero8i = vector.splati 0 : vector<8xi32>
  %ones8 = vector.splati 1 : vector<32xi8>
  %zero32 = vector.splatf 0.0 : vector<32xf32>
  %blk_id = kernel.workgroup.id<x> : index
  %nb = index.constant {width // BLK} : index
  %blk = index.assume %blk_id [lt(%blk_id, %nb)] : index
  %row_id = kernel.workgroup.id<y> : index
  %row = index.assume %row_id [lt(%row_id, %rows)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %buf_bytes = index.constant {BLK * 4} : offset
  %buf = buffer.alloca<workgroup> align(16) %buf_bytes : buffer
{_gather(width, perm, gdn)}{fwht("buf", 1, 256, "fw")}  %is_q = index.cmp ult, %lane, %c32 : index
  scf.if %is_q {{
    %gl = index.assume %lane [range(%lane, 0, 31)] : index
    %groups = index.constant {width // 32} : index
    %bv32 = buffer.view %buf[%zero] : buffer -> view<32x32xf32>
    %raw = vector.load %bv32[%gl, %c0] : view<32x32xf32> -> vector<32xf32>
    %inv = vector.splatf 0.03125 : vector<32xf32>
    %v = vector.mulf %raw, %inv : vector<32xf32>
    %g0 = index.mul %blk, %c32 : index
    %g1 = index.add %g0, %gl : index
    %g = index.assume %g1 [range(%g1, 0, {width // 32 - 1})] : index
    %q_view = buffer.view %xq[%zero] : buffer -> view<[%rows]x{width // 32}x8xi32>
    %m_bytes = index.constant {width} : offset
    %m_off = index.scale %rows, %m_bytes : index, offset -> offset
    %m_view = buffer.view %xq[%m_off] : buffer -> view<[%rows]x{width // 32}x2xf32>
    %a = vector.absf %v : vector<32xf32>
    %amax = vector.reduce<maxnumf> %a, %zero_f : vector<32xf32>, f32
    %d = scalar.divf %amax, %c127 : f32
    %nz = scalar.cmpf one, %amax, %zero_f : f32
    %dv = vector.splat %d : vector<32xf32>
    %sc = vector.divf %v, %dv : vector<32xf32>
    %rd = vector.roundf %sc : vector<32xf32>
    %nzv = vector.splat %nz : vector<32xi1>
    %rs = vector.select %nzv, %rd, %zero32 : vector<32xf32>
    %q = vector.fptosi %rs : vector<32xf32> to vector<32xi8>
    %qw = vector.bitcast %q : vector<32xi8> to vector<8xi32>
    vector.store %qw, %q_view[%row, %g, %c0] : vector<8xi32>, view<[%rows]x{width // 32}x8xi32>
    %qs = vector.dot4i<s8s8> %q, %ones8, %zero8i : vector<32xi8>, vector<32xi8>, vector<8xi32>
    %qsi = vector.reduce<addi> %qs, %zero_i : vector<8xi32>, i32
    %qsf = scalar.sitofp %qsi : i32 to f32
    %dh = scalar.fptrunc %d : f32 to f16
    %da = scalar.extf %dh : f16 to f32
    %meta = vector.from_elements %da, %qsf : vector<2xf32>
    vector.store %meta, %m_view[%row, %g, %c0] : vector<2xf32>, view<[%rows]x{width // 32}x2xf32>
  }}
  kernel.return
}}
"""


def rs16_kernel(width, perm, gdn):
    """g_row_scale_f16 with the rotation: one workgroup per row (256 lanes); pass 1 rotates every block and takes the row's
    max |value|, pass 2 rotates again and writes value * scale as F16 (the same power-of-two scale rule and outputs as
    g_row_scale_f16)."""
    name = f"g_had_rs16_w{width}{'p' if perm else ''}"
    nb = width // BLK
    body = _gather(width, perm, gdn) + fwht("buf", 1, 256, "fw")
    body2 = _uniq(body, "p2")
    return f"""
kernel.def export("{name}") @{name}(%rows: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%rows, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%rows_arg: index, %k_arg: index, %x: buffer, %scales: buffer, %x16: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 2097152)] : index
  %width = index.constant {width} : index
  %row_id = kernel.workgroup.id<x> : index
  %row = index.assume %row_id [lt(%row_id, %rows)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c4 = index.constant 4 : index
  %c1024 = index.constant 1024 : index
  %sg_c32w = index.constant {width // 32} : index
  %zero_f = scalar.constant 0.0 : f32
  %zero_i = scalar.constant 0 : i32
  %inv32 = scalar.constant 0.03125 : f32
  %nb = index.constant {nb} : index
  %buf_bytes = index.constant {BLK * 4} : offset
  %buf = buffer.alloca<workgroup> align(16) %buf_bytes : buffer
  %local = scf.for %blk = [%c0 to %nb step %c1](%m = %zero_f : f32) -> (f32) {{
{body}    %bv = buffer.view %buf[%zero] : buffer -> view<256x4xf32>
    %v = vector.load %bv[%lane, %c0] : view<256x4xf32> -> vector<4xf32>
    %a = vector.absf %v : vector<4xf32>
    %n = vector.reduce<maxnumf> %a, %m : vector<4xf32>, f32
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    scf.yield %n : f32
  }}
  %maximum0 = kernel.workgroup.reduce<maxnumf> %local : f32
  %maximum = scalar.mulf %maximum0, %inv32 : f32
  %bits = scalar.bitcast %maximum : f32 to i32
  %c23 = scalar.constant 23 : i32
  %c255 = scalar.constant 255 : i32
  %shifted = scalar.shrui %bits, %c23 : i32
  %biased = scalar.andi %shifted, %c255 : i32
  %c127 = scalar.constant 127 : i32
  %c13 = scalar.constant 13 : i32
  %c0_i32 = scalar.constant 0 : i32
  %c100 = scalar.constant 100 : i32
  %unbiased = scalar.subi %biased, %c127 : i32
  %raw_e = scalar.subi %unbiased, %c13 : i32
  %positive = scalar.maxsi %raw_e, %c0_i32 : i32
  %e = scalar.minsi %positive, %c100 : i32
  %scale_exp = scalar.subi %c127, %e : i32
  %inverse_exp = scalar.addi %c127, %e : i32
  %scale_bits = scalar.shli %scale_exp, %c23 : i32
  %inverse_bits = scalar.shli %inverse_exp, %c23 : i32
  %scale = scalar.bitcast %scale_bits : i32 to f32
  %inverse = scalar.bitcast %inverse_bits : i32 to f32
  %scale32 = scalar.mulf %scale, %inv32 : f32
  %scale4 = vector.splat %scale32 : vector<4xf32>
  %k4 = index.constant {width // 4} : index
  %h4_view = buffer.view %x16[%zero] : buffer -> view<[%rows]x{width // 4}x4xf16>
  scf.for %blk = [%c0 to %nb step %c1] {{
{body2}    %bv = buffer.view %buf[%zero] : buffer -> view<256x4xf32>
    %v = vector.load %bv[%lane, %c0] : view<256x4xf32> -> vector<4xf32>
    %sv = vector.mulf %v, %scale4 : vector<4xf32>
    %hv = vector.fptrunc %sv : vector<4xf32> to vector<4xf16>
    %c256 = index.constant 256 : index
    %ob0 = index.mul %blk, %c256 : index
    %ob1 = index.add %ob0, %lane : index
    %ob = index.assume %ob1 [range(%ob1, 0, {width // 4 - 1})] : index
    vector.store %hv, %h4_view[%row, %ob, %c0] : vector<4xf16>, view<[%rows]x{width // 4}x4xf16>
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  }}
  %first = index.cmp eq, %lane, %c0 : index
  scf.if %first {{
    %s_view = buffer.view %scales[%zero] : buffer -> view<[%rows]x2xf32>
    view.store %scale, %s_view[%row, %c0] : f32, view<[%rows]x2xf32>
    view.store %inverse, %s_view[%row, %c1] : f32, view<[%rows]x2xf32>
  }}
  kernel.return
}}
"""


def sign_vec32(word, pfx):
    """IR: vector<32xf32> %{pfx}_s of +/-1 from the 32 sign bits of i32 %{word} (bit k = column k of the group)."""
    return f"""  %{pfx}_ws = vector.splat {word} : vector<32xi32>
  %{pfx}_z0 = scalar.constant 0 : i32
  %{pfx}_s1 = scalar.constant 1 : i32
  %{pfx}_k = vector.iota %{pfx}_z0 step %{pfx}_s1 : vector<32xi32>
  %{pfx}_sh = vector.shrui %{pfx}_ws, %{pfx}_k : vector<32xi32>
  %{pfx}_one = vector.splati 1 : vector<32xi32>
  %{pfx}_bit = vector.andi %{pfx}_sh, %{pfx}_one : vector<32xi32>
  %{pfx}_zi = vector.splati 0 : vector<32xi32>
  %{pfx}_neg = vector.cmpi ne, %{pfx}_bit, %{pfx}_zi : vector<32xi32> -> vector<32xi1>
  %{pfx}_pf = vector.splatf 1.0 : vector<32xf32>
  %{pfx}_nf = vector.splatf -1.0 : vector<32xf32>
  %{pfx}_s = vector.select %{pfx}_neg, %{pfx}_nf, %{pfx}_pf : vector<32xf32>
"""


def embed_patch(text, H):
    """The lookup of a Hadamard-latent token embedding (g_embed: one workgroup of 256 lanes per token, lane l < H/32 holding
    values 32l..32l+31 in %values): h = D (H z) for table rows; image feature rows (%is_feat) are in the model basis already."""
    anchor = "  %normalized = vector.addf %values, %zero32 : vector<32xf32>\n"
    m = re.search(r'kernel\.def export\("g_embed"\).*?\nkernel\.return\n?\}', text, re.S) or \
        re.search(r'kernel\.def export\("g_embed"\).*?\n  kernel\.return\n\}', text, re.S)
    assert m and anchor in m.group(0), "ternary.embed_patch: g_embed layout changed"
    nblk, groups = H // BLK, H // 32
    rot = f"""  // prism.hadamard: the table stores rotated rows z; the model row is D (H z) (llama.cpp build_tok)
  %he_bytes = index.constant {H * 4} : offset
  %he_buf = buffer.alloca<workgroup> align(16) %he_bytes : buffer
  %he_v32 = buffer.view %he_buf[%zero] : buffer -> view<{groups}x32xf32>
  scf.if %active {{
    %he_l = index.assume %lane [range(%lane, 0, {groups - 1})] : index
    vector.store %values, %he_v32[%he_l, %c0] : vector<32xf32>, view<{groups}x32xf32>
  }}
{fwht("he_buf", nblk, 256, "hf")}  %rotated = scf.if %active -> (vector<32xf32>) {{
    %hr_l = index.assume %lane [range(%lane, 0, {groups - 1})] : index
    %hr_raw = vector.load %he_v32[%hr_l, %c0] : view<{groups}x32xf32> -> vector<32xf32>
    %hr_inv = vector.splatf 0.03125 : vector<32xf32>
    %hr_n = vector.mulf %hr_raw, %hr_inv : vector<32xf32>
    %hr_tab = global.load @g_had_signs_{H} : buffer
    %hr_tv = buffer.view %hr_tab[%zero] : buffer -> view<{groups}xi32>
    %hr_word = view.load %hr_tv[%hr_l] : view<{groups}xi32> -> i32
{sign_vec32("%hr_word", "hs")}    %hr_v = vector.mulf %hr_n, %hs_s : vector<32xf32>
    scf.yield %hr_v : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}
FEATSEL  %normalized = vector.addf %he_values, %zero32 : vector<32xf32>
"""
    rot = rot.replace("FEATSEL", """  %he_featv = vector.splat %is_feat : vector<32xi1>
  %he_values = vector.select %he_featv, %values, %rotated : vector<32xf32>
""" if "%is_feat" in m.group(0) else "  %he_values = vector.addf %rotated, %zero32 : vector<32xf32>\n")
    k = m.group(0).replace(anchor, rot)
    return text[:m.start()] + k + text[m.end():]


_QLAUNCH = re.compile(r"kernel\.launch @(g_row_scale_f16|g_quant_q8)\[[^\]]*\]\((%\w+), (%\w+), (%\w+), (%\w+)(?:, (%\w+))?\)")


def wiring(program, cfg):
    """Rename every activation quantizer whose copy feeds only folded weights to its rotating variant. Per command program: the
    copy a quantizer writes (X16 of g_row_scale_f16, Xq of g_quant_q8) is followed to the launches that read it before the next
    quantizer rewrites it; their weight operands (command.parameter bindings) decide: all folded -> rotate (width = the weights'
    input width; ssm_out -> permuted variant), none folded -> keep, mixed -> error. Returns (program, {(kind, width, perm)})."""
    had = cfg["hadamard"]
    folded = set(had["weights"])
    shapes = {n: t["shape"] for n, t in cfg["tensors"].items()}
    used = set()
    out, pos = [], 0
    for pm in re.finditer(r"command\.def public[^\n]*\{\n.*?\n\}\n", program, re.S):
        out.append(program[pos:pm.start()])
        body = pm.group(0)
        params = dict(re.findall(r"(%\w+) = command\.parameter %\w+, \"([^\"]+)\"", body))
        launches = list(_QLAUNCH.finditer(body))
        edits = []
        for i, q in enumerate(launches):
            kind = q.group(1)
            dst = q.group(6) if kind == "g_row_scale_f16" else q.group(5)
            end = len(body)
            for q2 in launches[i + 1:]:
                d2 = q2.group(6) if q2.group(1) == "g_row_scale_f16" else q2.group(5)
                if d2 == dst:
                    end = q2.start()
                    break
            seg = body[q.end():end]
            names = set()
            for ln in re.findall(r"kernel\.launch @\w+\[[^\]]*\]\(([^)]*)\)", seg):
                args = [a.strip() for a in ln.split(",")]
                if dst in args:
                    names |= {params[a] for a in args if a in params}
            mats = {n for n in names if len(shapes.get(n, [])) == 2}
            if not mats or not (mats & folded):
                continue
            assert mats <= folded, f"ternary.wiring: {dst} feeds folded and unfolded weights: {sorted(mats)[:6]}"
            widths = {shapes[n][0] for n in mats}
            assert len(widths) == 1, f"ternary.wiring: {dst} feeds weights of input widths {widths}"
            w = widths.pop()
            perm = had.get("gdn_v_grouped", False) and any(n.endswith(".ssm_out.weight") for n in mats)
            assert not perm or all(n.endswith(".ssm_out.weight") for n in mats)
            assert str(w) in had["signs"] and w % BLK == 0, f"ternary.wiring: no sign vector / block split for width {w}"
            k = "rs16" if kind == "g_row_scale_f16" else "q8"
            used.add((k, w, perm))
            edits.append((q.start(), q.start() + len(f"kernel.launch @{kind}"), f"kernel.launch @g_had_{k}_w{w}{'p' if perm else ''}"))
        for a, b, new in reversed(edits):
            body = body[:a] + new + body[b:]
        out.append(body)
        pos = pm.end()
    out.append(program[pos:])
    return "".join(out), used


def kernels(cfg, used):
    """Sign tables and the rotating quantizers wiring() launched."""
    gdn = cfg.get("gdn")
    text = signs_rodata(cfg)
    for k, w, perm in sorted(used):
        text += (q8_kernel if k == "q8" else rs16_kernel)(w, perm, gdn)
    return text


def setup(cfg):
    """True when the model is a prism.hadamard checkpoint this module handles (asserts the parts it relies on)."""
    had = cfg.get("hadamard")
    if not had:
        return False
    assert had["block"] == BLK, f"prism.hadamard block {had['block']}: only {BLK} is implemented"
    assert set(had["inverse"]) <= {"token_embd.weight"}
    assert "token_embd.weight" in had["inverse"], "a Hadamard model whose embedding is not latent: not seen yet"
    return True


# ---- numpy references (make_kernel_tests.py) ----
def np_signs(cfg, width):
    import numpy as np
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(cfg["hadamard"]["signs"][str(width)]), np.uint8), bitorder="little")[:width]
    return np.where(bits == 1, -1.0, 1.0)


def np_hadamard(n=BLK):
    """The normalized Sylvester Walsh-Hadamard matrix (llama.cpp load_tensors_hadamard: entry (r, c) = (-1)^popcount(r & c) / sqrt(n))."""
    import numpy as np
    r = np.arange(n)
    par = np.vectorize(lambda v: bin(v).count("1") & 1)(r[:, None] & r[None, :])
    return np.where(par == 1, -1.0, 1.0) / np.sqrt(n)


def np_perm(cfg):
    """Grouped column g -> the tiled source column it reads (ssm_out input with gdn_v_grouped)."""
    import numpy as np
    gd = cfg["gdn"]; hd, nk = gd["head_v"], gd["k_heads"]; rep = gd["v_heads"] // nk
    g = np.arange(hd * nk * rep); d, h = g % hd, g // hd; r, k = h % rep, h // rep
    return d + hd * (k + nk * r)


def np_rotate(x, cfg, width, perm=False):
    """H (D x) per 1024-block of the last axis (x: rows x width), after the grouped permutation when perm."""
    import numpy as np
    x = np.asarray(x, np.float64)
    if perm:
        x = x[:, np_perm(cfg)]
    x = x * np_signs(cfg, width)
    Hm = np_hadamard()
    return (x.reshape(x.shape[0], -1, BLK) @ Hm).reshape(x.shape)


def np_unrotate_embed(z, cfg):
    """D (H z) of embedding rows (the lookup's inverse transform)."""
    import numpy as np
    z = np.asarray(z, np.float64)
    w = z.shape[-1]
    return (z.reshape(z.shape[0], -1, BLK) @ np_hadamard()).reshape(z.shape) * np_signs(cfg, w)
