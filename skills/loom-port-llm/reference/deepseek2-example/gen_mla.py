"""Multi-head latent attention (deepseek2 absorbed MLA, cfg["mla"]): the kernels around the attention core.

Per layer (llama.cpp src/models/deepseek2.cpp, is_mla path), x = attn_norm(h):
  qa = q_a x -> rms(qa) * q_a_norm (g_mla_rms_q) -> qb = q_b qa            (per head [q_nope (nope) | q_pe (rope)])
  kv = kv_a_mqa x                                                          ([c (kv_lora) | k_pe (rope)])
  g_mla_rope_store: cache row [rms(c) * kv_a_norm | rope(k_pe)] (F16, ONE plane per layer: V is the row's first
      kv_lora values) and Q[h][kv_lora:] = rope(q_pe[h])
  g_mla_bmv_kb: Q[h][:kv_lora] = k_b[h] q_nope[h]                          (k_b GGUF [nope, kv_lora, heads])
  attention (gen_llm split-K / listed / flash) on Q [heads x (kv_lora + rope)] against the one-head cache: o[h] (the
      tail rope dims of o are a by-product nobody reads)
  g_mla_bmv_vb: vo[h] = v_b[h] o[h][:kv_lora]                              (v_b GGUF [kv_lora, v_dims, heads])
  attn_output on vo (heads * v_dims -> hidden)
RoPE is llama.cpp's NORM type (pairs 2p, 2p+1) on the rope dims. Every kernel takes any token count (decode and prefill).
"""
import math
import os

ROWS_PER_WG = 256   # lanes per workgroup of the batched per-head matvec


def flash_on():
    """MLA prefill on the flash kernels (gen_flash2s.py MLA mode) for runs of 9+ tokens; MLA_FLASH=0: every prefill token
    takes the listed split-K kernels (gen_llm NO_FLASH lists). Shared by gen_llm.py and make_kernel_tests.py."""
    import os
    return os.environ.get("MLA_FLASH", "1") == "1"


def geometry(cfg):
    m = cfg.get("mla") or {}
    required = {"kv_lora": 512, "rope_dims": 64, "nope_dims": 128, "v_dims": 128}
    got = {name: m.get(name) for name in required}
    q_lora = m.get("q_lora", 0)
    if (cfg.get("heads") != 32 or got != required or not isinstance(q_lora, int) or q_lora <= 0
            or q_lora % 256 or cfg.get("sliding_layers")):
        raise ValueError("unsupported deepseek2 MLA geometry: expected 32 query heads, "
                         f"{required}, a positive q_lora divisible by 256, and no sliding layers; "
                         f"received heads={cfg.get('heads')}, {got}, q_lora={q_lora}, "
                         f"sliding_layers={cfg.get('sliding_layers')}")
    return dict(QH=cfg["heads"], QL=m["q_lora"], KL=m["kv_lora"], RD=m["rope_dims"], NOPE=m["nope_dims"],
                VD=m["v_dims"], D=m["kv_lora"] + m["rope_dims"], QB=m["nope_dims"] + m["rope_dims"])


def rms_q(cfg, eps):
    """In-place RMSNorm of the q_lora rows (q_a output) times attn_q_a_norm: one 256-lane workgroup per token."""
    g = geometry(cfg)
    W = g["QL"]
    assert W % 256 == 0
    NJ = W // 256
    loads = "\n".join(f"""  %x{j} = view.load %x_view[%token, {j}, %wi] : view<[%tokens]x{NJ}x256xf32> -> f32""" for j in range(NJ))
    sq = "  %s0 = scalar.mulf %x0, %x0 : f32\n" + "".join(f"  %s{j} = scalar.fmaf %x{j}, %x{j}, %s{j - 1} : f32\n" for j in range(1, NJ))
    stores = "\n".join(f"""  %w{j} = view.load %w_view[{j}, %wi] : view<{NJ}x256xf32> -> f32
  %n{j} = scalar.mulf %x{j}, %inverse : f32
  %y{j} = scalar.mulf %n{j}, %w{j} : f32
  view.store %y{j}, %x_view[%token, {j}, %wi] : f32, view<[%tokens]x{NJ}x256xf32>""" for j in range(NJ))
    return f"""
// MLA: q_a rows normalized in place, rms(x) * attn_q_a_norm ({W} values per token).
kernel.def export("g_mla_rms_q") @g_mla_rms_q(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %x: buffer, %w: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %wi_id = kernel.workitem.id<x> : index
  %wi = index.assume %wi_id [range(%wi_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{NJ}x256xf32>
  %w_view = buffer.view %w[%zero] : buffer -> view<{NJ}x256xf32>
{loads}
{sq}  %sum = kernel.workgroup.reduce<addf> %s{NJ - 1} : f32
  %count = scalar.constant {float(W)!r} : f32
  %eps = scalar.constant {eps!r} : f32
  %mean = scalar.divf %sum, %count : f32
  %stable = scalar.addf %mean, %eps : f32
  %inverse = scalar.rsqrtf %stable : f32
{stores}
  kernel.return
}}
"""


def rope_store(cfg, eps, theta, layers):
    """Per token (256 lanes, lane p of a wave = rotary pair p): the cache row [rms(c) * kv_a_norm | rope(k_pe)] stored
    as F16 at (row, position, plane = layer) by wave 0, and Q[h][kv_lora + 2p, +1] = rope(q_pe[h]) by every wave."""
    g = geometry(cfg)
    QH, KL, RD, NOPE, D, QB = g["QH"], g["KL"], g["RD"], g["NOPE"], g["D"], g["QB"]
    assert RD == 64 and KL == 512, "g_mla_rope_store: 32 rotary pairs (one per lane), 16 latent values per lane"
    ln_scale = 2.0 * math.log(theta) / RD   # angle of pair p = position * theta^(-2p/RD)
    return f"""
// MLA: latent cache row + rotary halves of Q (NORM RoPE pairs 2p, 2p+1 over {RD} dims; theta {theta}).
kernel.def export("g_mla_rope_store") @g_mla_rope_store(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %layer_arg: index, %context_arg: index, %pool_arg: index, %qb: buffer, %kv: buffer, %kv_norm: buffer, %q: buffer, %info: buffer, %origins: buffer, %arena: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {layers - 1})] : index
  %context = index.assume %context_arg [range(%context_arg, 16, 262144)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 2147483584)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, 7)] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c8 = index.constant 8 : index
  %c15 = index.constant 15 : index
  %c16 = index.constant 16 : index
  %cqh = index.constant {QH} : index
  %c0_i32 = scalar.constant 0 : i32
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %row32 = view.load %info_view[%token, 0] : view<[%tokens]x4xi32> -> i32
  %position32 = view.load %info_view[%token, 1] : view<[%tokens]x4xi32> -> i32
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %position_low = scalar.maxsi %position32, %c0_i32 : i32
  %position_f = scalar.sitofp %position_low : i32 to f32
  %ln_scale = scalar.constant {ln_scale!r} : f32
  %neg = scalar.constant -1.0 : f32
  %inv_two_pi = scalar.constant 0.15915494309189535 : f32
  %two_pi = scalar.constant 6.283185307179586 : f32
  %half = scalar.constant 0.5 : f32
  %zero_f = scalar.constant 0.0 : f32
  %li = index.cast %lane : index to i32
  %lf = scalar.sitofp %li : i32 to f32
  %exa = scalar.mulf %lf, %ln_scale : f32
  %exb = scalar.mulf %exa, %neg : f32
  %freq = scalar.expf<afn> %exb : f32
  %ra = scalar.mulf %position_f, %freq : f32
  %tr = scalar.mulf %ra, %inv_two_pi : f32
  %ts = scalar.addf %tr, %half : f32
  %ti = scalar.fptosi %ts : f32 to i32
  %tf = scalar.sitofp %ti : i32 to f32
  %wr = scalar.mulf %tf, %two_pi : f32
  %an = scalar.subf %ra, %wr : f32
  %cos = scalar.cosf<afn> %an : f32
  %sin = scalar.sinf<afn> %an : f32
  %e0 = index.mul %lane, %c2 : index
  %e1 = index.add %e0, %c1 : index
  %qe0c = index.constant {NOPE} : index
  %qe0 = index.add %qe0c, %e0 : index
  %qe1 = index.add %qe0, %c1 : index
  %ke0c = index.constant {KL} : index
  %ke0 = index.add %ke0c, %e0 : index
  %ke1 = index.add %ke0, %c1 : index
  %qb_view = buffer.view %qb[%zero] : buffer -> view<[%tokens]x{QH}x{QB}xf32>
  %q_view = buffer.view %q[%zero] : buffer -> view<[%tokens]x{QH}x{D}xf32>
  scf.for %head = [%wave to %cqh step %c8] {{
    %bh = index.assume %head [range(%head, 0, {QH - 1})] : index
    %a0 = view.load %qb_view[%token, %bh, %qe0] : view<[%tokens]x{QH}x{QB}xf32> -> f32
    %a1 = view.load %qb_view[%token, %bh, %qe1] : view<[%tokens]x{QH}x{QB}xf32> -> f32
    %ac0 = scalar.mulf %a0, %cos : f32
    %as1 = scalar.mulf %a1, %sin : f32
    %r0 = scalar.subf %ac0, %as1 : f32
    %as0 = scalar.mulf %a0, %sin : f32
    %ac1 = scalar.mulf %a1, %cos : f32
    %r1 = scalar.addf %as0, %ac1 : f32
    view.store %r0, %q_view[%token, %bh, %ke0] : f32, view<[%tokens]x{QH}x{D}xf32>
    view.store %r1, %q_view[%token, %bh, %ke1] : f32, view<[%tokens]x{QH}x{D}xf32>
  }}
  %first_wave = index.cmp eq, %wave, %c0 : index
  %store = scalar.andi %first_wave, %live : i1
  scf.if %store {{
    %kvc_view = buffer.view %kv[%zero] : buffer -> view<[%tokens]x{D // 16}x16xf32>
    %kvs_view = buffer.view %kv[%zero] : buffer -> view<[%tokens]x{D}xf32>
    %kn_view = buffer.view %kv_norm[%zero] : buffer -> view<32x16xf32>
    %c = vector.load %kvc_view[%token, %lane, %c0] : view<[%tokens]x{D // 16}x16xf32> -> vector<16xf32>
    %cc = vector.mulf %c, %c : vector<16xf32>
    %part = vector.reduce<addf> %cc, %zero_f : vector<16xf32>, f32
    %sum = kernel.subgroup.reduce<addf> %part : f32
    %count = scalar.constant {float(KL)!r} : f32
    %eps = scalar.constant {eps!r} : f32
    %mean = scalar.divf %sum, %count : f32
    %stable = scalar.addf %mean, %eps : f32
    %inverse = scalar.rsqrtf %stable : f32
    %inv_v = vector.splat %inverse : vector<16xf32>
    %wn = vector.load %kn_view[%lane, %c0] : view<32x16xf32> -> vector<16xf32>
    %cn = vector.mulf %c, %inv_v : vector<16xf32>
    %cw = vector.mulf %cn, %wn : vector<16xf32>
    %ch = vector.fptrunc %cw : vector<16xf32> to vector<16xf16>
    %b0 = view.load %kvs_view[%token, %ke0] : view<[%tokens]x{D}xf32> -> f32
    %b1 = view.load %kvs_view[%token, %ke1] : view<[%tokens]x{D}xf32> -> f32
    %bc0 = scalar.mulf %b0, %cos : f32
    %bs1 = scalar.mulf %b1, %sin : f32
    %k0 = scalar.subf %bc0, %bs1 : f32
    %bs0 = scalar.mulf %b0, %sin : f32
    %bc1 = scalar.mulf %b1, %cos : f32
    %k1 = scalar.addf %bs0, %bc1 : f32
    %kh0 = scalar.fptrunc %k0 : f32 to f16
    %kh1 = scalar.fptrunc %k1 : f32 to f16
    %row_bound = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
    %row = index.cast %row_bound : i32 to index
    %position_bound = scalar.assume %position_low [range(%position_low, 0, 262143)] : i32
    %position = index.cast %position_bound : i32 to index
    %padded = index.add %context, %c15 : index
    %blocks = index.div %padded, %c16 : index
    %k_offset = func.call @g_kv_offset(%origins, %row, %position, %blocks, %layer, %pool) : (buffer, index, index, index, index, index) -> (offset)
    %cache_c = buffer.view %arena[%k_offset] : buffer -> view<{D // 16}x16xf16>
    vector.store %ch, %cache_c[%lane, %c0] : vector<16xf16>, view<{D // 16}x16xf16>
    %cache_s = buffer.view %arena[%k_offset] : buffer -> view<{D}xf16>
    view.store %kh0, %cache_s[%ke0] : f16, view<{D}xf16>
    view.store %kh1, %cache_s[%ke1] : f16, view<{D}xf16>
  }}
  kernel.return
}}
"""


def bmv(name, fmt, K, N, H, XH, YH, FMT, SB, row_bytes, TGC=8):
    """Per-head batched matrix-vector: y[t, h, r] = sum_k W[h, r, k] x[t, h, k] for r < N, k < K (W rows of K values in
    `fmt`, head h's N rows after h * N rows); x rows of XH values per head (first K used), y rows of YH per head (first
    N written). Lane = (row, 32-value group): K/32 lanes per row reduce by xor shuffles. A 256-lane workgroup takes
    256 / (K/32) rows of one head and TGC tokens (weights decoded once per workgroup, kept in registers)."""
    LPR = K // 32
    assert K % 32 == 0 and 32 % LPR == 0 and XH % 32 == 0
    RPW = 32 // LPR
    RW = 8 * RPW
    assert N % RW == 0
    gpb = SB // 32
    rb = row_bytes
    sh = []
    d = 1
    while d < LPR:
        sh.append(f"""      %sh{d}r, %sh{d}u = kernel.subgroup.shuffle<xor> {"%p" if d == 1 else f"%acc{d // 2}"}, %sd{d}, %sw32 : f32, i32, i32
      %acc{d} = scalar.addf {"%p" if d == 1 else f"%acc{d // 2}"}, %sh{d}r : f32""")
        d *= 2
    final = "%p" if LPR == 1 else f"%acc{LPR // 2}"
    sconst = "".join(f"  %sd{d} = scalar.constant {d} : i32\n" for d in (1 << i for i in range(int(math.log2(LPR)))))
    return f"""
// MLA per-head matvec {name}: {H} heads x [{N} x {K}] {fmt}; x per head stride {XH}, y per head stride {YH}.
kernel.def export("{name}") @{name}(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %rblocks = index.constant {N // RW} : index
  %heads = index.constant {H} : index
  %ctg = index.constant {TGC} : index
  %ctg_less = index.constant {TGC - 1} : index
  %tpad = index.add %tokens, %ctg_less : index
  %tgroups = index.div %tpad, %ctg : index
  kernel.launch.config workgroups(%rblocks, %heads, %tgroups) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %x: buffer, %w: buffer, %y: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %rb_id = kernel.workgroup.id<x> : index
  %rblk = index.assume %rb_id [range(%rb_id, 0, {N // RW - 1})] : index
  %h_id = kernel.workgroup.id<y> : index
  %head = index.assume %h_id [range(%h_id, 0, {H - 1})] : index
  %tg_id = kernel.workgroup.id<z> : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, 7)] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %clpr = index.constant {LPR} : index
  %crpw = index.constant {RPW} : index
  %crw = index.constant {RW} : index
  %cn = index.constant {N} : index
  %cgpb = index.constant {gpb} : index
  %ctg = index.constant {TGC} : index
  %sw32 = scalar.constant 32 : i32
{sconst}  %zero_f = scalar.constant 0.0 : f32
  %g_v = index.rem %lane, %clpr : index
  %g = index.assume %g_v [range(%g_v, 0, {LPR - 1})] : index
  %rl_v = index.div %lane, %clpr : index
  %rw_base = index.mul %wave, %crpw : index
  %rr_v = index.add %rw_base, %rl_v : index
  %rblk_base = index.mul %rblk, %crw : index
  %r_v = index.add %rblk_base, %rr_v : index
  %r = index.assume %r_v [range(%r_v, 0, {N - 1})] : index
  %hrow = index.madd %head, %cn, %r : index
  %row_bytes = index.constant {rb} : offset
  %block_bytes = index.constant {FMT} : offset
  %ro = index.scale %hrow, %row_bytes : index, offset -> offset
  %blk = index.div %g, %cgpb : index
  %within = index.rem %g, %cgpb : index
  %ba = index.scale %blk, %block_bytes : index, offset -> offset
  %origin = index.add %ro, %ba : offset
  %wv = func.call @g_{fmt}_values(%w, %origin, %within) : (buffer, offset, index) -> (vector<32xf32>)
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{H}x{XH // 32}x32xf32>
  %y_view = buffer.view %y[%zero] : buffer -> view<[%tokens]x{H}x{YH}xf32>
  %first = index.cmp eq, %g, %c0 : index
  %t0 = index.mul %tg_id, %ctg : index
  scf.for %j = [%c0 to %ctg step %c1] {{
    %tok_raw = index.add %t0, %j : index
    %in = index.cmp ult, %tok_raw, %tokens : index
    scf.if %in {{
      %tok = index.assume %tok_raw [lt(%tok_raw, %tokens)] : index
      %xv = vector.load %x_view[%tok, %head, %g, %c0] : view<[%tokens]x{H}x{XH // 32}x32xf32> -> vector<32xf32>
      %p = vector.dotf<reassoc|contract> %wv, %xv, %zero_f : vector<32xf32>, vector<32xf32>, f32
{chr(10).join(sh)}
      scf.if %first {{
        view.store {final}, %y_view[%tok, %head, %r] : f32, view<[%tokens]x{H}x{YH}xf32>
      }}
    }}
  }}
  kernel.return
}}
"""


def bmm(name, fmt, K, N, H, XH, YH, FMT, SB, row_bytes, TT=4):
    """Prefill twin of bmv() on WMMA (epochs of 64+ tokens, a multiple of 64): y[t, h, r] = sum_k W[h, r, k] x[t, h, k] with f16
    operands and f32 accumulation. Workgroup = (64 rows, head, 64 tokens), 4 waves of 16 rows each; per 32-value group a lane
    decodes its row's weights once (B fragments: column = row) for the wave's TT 16-token tiles (A fragments: row = token)."""
    assert K % 32 == 0 and N % 64 == 0 and XH % 16 == 0
    gpb = SB // 32
    T = [f"""
// MLA per-head WMMA product {name}: {H} heads x [{N} x {K}] {fmt} (prefill: tokens a multiple of 64).
kernel.def export("{name}") @{name}(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c64 = index.constant 64 : index
  %c128 = index.constant 128 : index
  %rblocks = index.constant {N // 64} : index
  %heads = index.constant {H} : index
  %tblocks = index.div %tokens, %c64 : index
  kernel.launch.config workgroups(%rblocks, %heads, %tblocks) workgroup_size(%c128, %c1, %c1) : index
}} launch(%tokens_arg: index, %x: buffer, %w: buffer, %y: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 64, 8192), multiple_of(%tokens_arg, 64)] : index
  %rb_id = kernel.workgroup.id<x> : index
  %rblk = index.assume %rb_id [range(%rb_id, 0, {N // 64 - 1})] : index
  %h_id = kernel.workgroup.id<y> : index
  %head = index.assume %h_id [range(%h_id, 0, {H - 1})] : index
  %c64 = index.constant 64 : index
  %tblocks = index.div %tokens, %c64 : index
  %tb_id = kernel.workgroup.id<z> : index
  %tblk = index.assume %tb_id [lt(%tb_id, %tblocks)] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, 3)] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c16 = index.constant 16 : index
  %cn = index.constant {N} : index
  %cgpb = index.constant {gpb} : index
  %col_v = index.rem %lane, %c16 : index
  %col = index.assume %col_v [range(%col_v, 0, 15)] : index
  %wrow = index.mul %wave, %c16 : index
  %rblk_base = index.mul %rblk, %c64 : index
  %r0 = index.add %rblk_base, %wrow : index
  %r_v = index.add %r0, %col : index
  %r = index.assume %r_v [range(%r_v, 0, {N - 1})] : index
  %hrow = index.madd %head, %cn, %r : index
  %row_bytes = index.constant {row_bytes} : offset
  %block_bytes = index.constant {FMT} : offset
  %ro = index.scale %hrow, %row_bytes : index, offset -> offset
  %t0 = index.mul %tblk, %c64 : index
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{H}x{XH // 16}x16xf32>
  %zero8 = vector.splatf 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %groups = index.constant {K // 32} : index"""]
    for j in range(TT):
        T.append(f"""  %tj{j}c = index.constant {16 * j} : index
  %tj{j}a = index.add %t0, %tj{j}c : index
  %tj{j}b = index.add %tj{j}a, %col : index
  %tok{j} = index.assume %tj{j}b [lt(%tj{j}b, %tokens)] : index""")
    accs = ", ".join(f"%a{j} = %init : vector<8xf32>" for j in range(TT))
    T.append(f"""  {", ".join(f"%o{j}" for j in range(TT))} = scf.for %g = [%c0 to %groups step %c1]({accs}) -> ({", ".join(["vector<8xf32>"] * TT)}) {{
    %gb = index.assume %g [range(%g, 0, {K // 32 - 1})] : index
    %blk = index.div %gb, %cgpb : index
    %within = index.rem %gb, %cgpb : index
    %ba = index.scale %blk, %block_bytes : index, offset -> offset
    %origin = index.add %ro, %ba : offset
    %wv = func.call @g_{fmt}_values(%w, %origin, %within) : (buffer, offset, index) -> (vector<32xf32>)
    %wlo = vector.slice %wv[0] : vector<32xf32> -> vector<16xf32>
    %whi = vector.slice %wv[16] : vector<32xf32> -> vector<16xf32>
    %wlo_h = vector.fptrunc %wlo : vector<16xf32> to vector<16xf16>
    %whi_h = vector.fptrunc %whi : vector<16xf32> to vector<16xf16>
    %b0 = vector.fragment<rhs> %wlo_h shape [%c16, %c16] : vector<16xf16>
    %b1 = vector.fragment<rhs> %whi_h shape [%c16, %c16] : vector<16xf16>
    %k0 = index.mul %gb, %c2 : index
    %k1 = index.add %k0, %c1 : index""")
    for j in range(TT):
        for kk in range(2):
            T.append(f"""    %x{j}_{kk} = vector.load %x_view[%tok{j}, %head, %k{kk}, %c0] : view<[%tokens]x{H}x{XH // 16}x16xf32> -> vector<16xf32>
    %xh{j}_{kk} = vector.fptrunc %x{j}_{kk} : vector<16xf32> to vector<16xf16>
    %af{j}_{kk} = vector.fragment<lhs> %xh{j}_{kk} shape [%c16, %c16] : vector<16xf16>""")
        T.append(f"""    %m{j}_0 = vector.mma %af{j}_0, %b0, %a{j} : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %m{j}_1 = vector.mma %af{j}_1, %b1, %m{j}_0 : vector<16xf16>, vector<16xf16>, vector<8xf32>""")
    T.append(f"""    scf.yield {", ".join(f"%m{j}_1" for j in range(TT))} : {", ".join(["vector<8xf32>"] * TT)}
  }}
  %cyt = index.constant {H * YH} : index
  %y_layout = encoding.layout.strided [%cyt, %c1] : encoding<layout>
  %cyh = index.constant {YH} : index
  %four = index.constant 4 : offset
  %hcol = index.mul %head, %cyh : index
  %hr0 = index.add %hcol, %r0 : index""")
    for j in range(TT):
        T.append(f"""  %yt{j} = index.mul %tj{j}a, %cyt : index
  %ye{j} = index.add %yt{j}, %hr0 : index
  %yo{j} = index.scale %ye{j}, %four : index, offset -> offset
  %yv{j} = buffer.view %y[%yo{j}] : buffer -> view<16x16xf32, %y_layout>
  vector.fragment.store<result> %o{j}, %yv{j}[%c0, %c0] shape [%c16, %c16] : vector<8xf32>, view<16x16xf32, %y_layout>""")
    T.append("  kernel.return\n}\n")
    text = "\n".join(T)
    return text.replace("  %c16 = index.constant 16 : index\n  %cn", "  %c16 = index.constant 16 : index\n  %c2 = index.constant 2 : index\n  %cn", 1)


def rope_kb(rope, kb, name, TGC=8):
    """MLA_ROPE_KB: g_mla_rope_store and the decode k_b matvec in one launch. The k_b grid (row blocks, heads, token
    groups of TGC) gets one more row block; its workgroups (y < TGC) run the rope / cache store of token z * TGC + y,
    the rest return. Both bodies unchanged (each in its own branch region). Args: the rope kernel's, then k_b's weights."""
    import re as _re
    def body(t):
        i = t.index("} launch(")
        i = t.index("\n", i) + 1
        j = t.rindex("  kernel.return")
        return t[i:j]
    rb, kbb = body(rope), body(kb)
    kbb = _re.sub(r"%x\b", "%qb", kbb)
    kbb = _re.sub(r"%y\b", "%q", kbb)
    rb = rb.replace("  %token_id = kernel.workgroup.id<x> : index\n", "  %token_id = index.madd %rk_z, %rk_tgc, %rk_y : index\n", 1)
    cfg = kb[:kb.index("} launch(")]
    cfg = cfg.replace("kernel.launch.config workgroups(%rblocks, %heads, %tgroups)", "%rblocks1 = index.add %rblocks, %c1 : index\n  kernel.launch.config workgroups(%rblocks1, %heads, %tgroups)")
    kname = _re.search(r'export\("(\w+)"\)', kb).group(1)
    cfg = cfg.replace(f'export("{kname}") @{kname}(', f'export("{name}") @{name}(')
    nrb = _re.search(r"%rblocks = index.constant (\d+) : index", kb).group(1)
    args = rope[rope.index("} launch(") + len("} launch("):rope.index(") {", rope.index("} launch("))]
    return cfg + f"""}} launch({args}, %w: buffer) {{
  %rk_x = kernel.workgroup.id<x> : index
  %rk_y = kernel.workgroup.id<y> : index
  %rk_z = kernel.workgroup.id<z> : index
  %rk_nrb = index.constant {nrb} : index
  %rk_tgc = index.constant {TGC} : index
  %rk_isr = index.cmp eq, %rk_x, %rk_nrb : index
  scf.if %rk_isr {{
    %rk_ylt = index.cmp ult, %rk_y, %rk_tgc : index
    %rk_tok = index.madd %rk_z, %rk_tgc, %rk_y : index
    %rk_tin = index.cmp ult, %rk_tok, %tokens_arg : index
    %rk_do = scalar.andi %rk_ylt, %rk_tin : i1
    scf.if %rk_do {{
{rb}    }}
  }} else {{
{kbb}  }}
  kernel.return
}}
"""


def kernels(cfg, eps, theta, layers, FMT, SB, row_bytes, kb_fmts, vb_fmts):
    """Every MLA kernel the layers launch: the q_a norm, the rope/cache store, and the k_b / v_b per-head matvecs
    (g_mla_kb_<fmt>, g_mla_vb_<fmt>) for each format in use."""
    g = geometry(cfg)
    QH, KL, NOPE, VD, D, QB = g["QH"], g["KL"], g["NOPE"], g["VD"], g["D"], g["QB"]
    return (rms_q(cfg, eps) + rope_store(cfg, eps, theta, layers)
            + "".join(bmv(f"g_mla_kb_{f}", f, NOPE, KL, QH, QB, D, FMT[f], SB[f], row_bytes(f, NOPE)) for f in kb_fmts)
            + "".join(rope_kb(rope_store(cfg, eps, theta, layers), bmv(f"g_mla_kb_{f}", f, NOPE, KL, QH, QB, D, FMT[f], SB[f], row_bytes(f, NOPE)),
                              f"g_mla_ropekb_{f}") for f in kb_fmts)
            + "".join(bmv(f"g_mla_vb_{f}", f, KL, VD, QH, D, VD, FMT[f], SB[f], row_bytes(f, KL)) for f in vb_fmts)
            + "".join(_bmm("kbm")(f"g_mla_kbm_{f}", f, NOPE, KL, QH, QB, D, FMT[f], SB[f], row_bytes(f, NOPE)) for f in kb_fmts)
            + "".join(_bmm("vbm")(f"g_mla_vbm_{f}", f, KL, VD, QH, D, VD, FMT[f], SB[f], row_bytes(f, KL)) for f in vb_fmts))


def _bmm(kind=None):
    f = _bmm0(kind)
    if os.environ.get("MLA_BMM_LIVE", "0") not in ("", "0"):
        return lambda *a, **k: bmm_live(f(*a, **k))
    return f


def _bmm0(kind=None):
    """MLA_BMM2: 1 = bmm2 for both products, or a list (kbm,vbm). kbench joyai_mla (run 45234212231): vbm -28 % (1k) / -13 % (4k),
    kbm +16 % / tie, so the useful setting is MLA_BMM2=vbm."""
    v = os.environ.get("MLA_BMM2", "0").strip()
    return bmm2 if v not in ("", "0") and (v == "1" or kind is None or kind in v.split(",")) else bmm


def bmm2(name, fmt, K, N, H, XH, YH, FMT, SB, row_bytes, TT=4):
    """MLA_BMM2 (choice mla.prefill.bmm2): bmm() with the activations staged once per workgroup. Workgroup = (128 rows (all
    of them when N = 128), head, 64 tokens), 8 waves of 16 rows; per 128-K chunk the 64 tokens' activations are converted to f16
    into LDS once and every wave's A fragments come from there (bmm() made each of its 4 waves re-read the f32 rows from
    memory, and N/64 workgroups re-read them again). Same values and K order as bmm(): bit-identical."""
    RW = min(N, 128)
    KC = min(K, 128)
    assert K % KC == 0 and N % RW == 0 and XH % 16 == 0 and KC % 32 == 0
    NW = RW // 16
    NT = NW * 32
    gpb = SB // 32
    LD = KC + 8
    per = 64 * KC // NT            # f32 values staged per thread per chunk
    assert per % 16 == 0
    T = [f"""
// MLA per-head WMMA product {name} (bmm2: shared f16 activation stage): {H} heads x [{N} x {K}] {fmt}.
kernel.def export("{name}") @{name}(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c64 = index.constant 64 : index
  %cnt = index.constant {NT} : index
  %rblocks = index.constant {N // RW} : index
  %heads = index.constant {H} : index
  %tblocks = index.div %tokens, %c64 : index
  kernel.launch.config workgroups(%rblocks, %heads, %tblocks) workgroup_size(%cnt, %c1, %c1) : index
}} launch(%tokens_arg: index, %x: buffer, %w: buffer, %y: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 64, 8192), multiple_of(%tokens_arg, 64)] : index
  %rb_id = kernel.workgroup.id<x> : index
  %rblk = index.assume %rb_id [range(%rb_id, 0, {N // RW - 1})] : index
  %h_id = kernel.workgroup.id<y> : index
  %head = index.assume %h_id [range(%h_id, 0, {H - 1})] : index
  %c64 = index.constant 64 : index
  %tblocks = index.div %tokens, %c64 : index
  %tb_id = kernel.workgroup.id<z> : index
  %tblk = index.assume %tb_id [lt(%tb_id, %tblocks)] : index
  %wi_id = kernel.workitem.id<x> : index
  %wi = index.assume %wi_id [range(%wi_id, 0, {NT - 1})] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, {NW - 1})] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c16 = index.constant 16 : index
  %cn = index.constant {N} : index
  %cgpb = index.constant {gpb} : index
  %crw = index.constant {RW} : index
  %col_v = index.rem %lane, %c16 : index
  %col = index.assume %col_v [range(%col_v, 0, 15)] : index
  %wrow = index.mul %wave, %c16 : index
  %rblk_base = index.mul %rblk, %crw : index
  %r0 = index.add %rblk_base, %wrow : index
  %r_v = index.add %r0, %col : index
  %r = index.assume %r_v [range(%r_v, 0, {N - 1})] : index
  %hrow = index.madd %head, %cn, %r : index
  %row_bytes = index.constant {row_bytes} : offset
  %block_bytes = index.constant {FMT} : offset
  %ro = index.scale %hrow, %row_bytes : index, offset -> offset
  %t0 = index.mul %tblk, %c64 : index
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{H}x{XH // 16}x16xf32>
  %st_bytes = index.constant {64 * LD * 2} : offset
  %st = buffer.alloca<workgroup> align(16) %st_bytes : buffer
  %st_view = buffer.view %st[%zero] : buffer -> view<64x{LD}xf16>
  %cper = index.constant {per} : index
  %cpr = index.constant {KC // per} : index
  %s_tok_l = index.div %wi, %cpr : index
  %s_q = index.rem %wi, %cpr : index
  %s_col = index.mul %s_q, %cper : index
  %s_tok_v = index.add %t0, %s_tok_l : index
  %s_tok = index.assume %s_tok_v [lt(%s_tok_v, %tokens)] : index
  %zero8 = vector.splatf 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %chunks = index.constant {K // KC} : index"""]
    accs = ", ".join(f"%a{j} = %init : vector<8xf32>" for j in range(TT))
    T.append(f"""  {", ".join(f"%o{j}" for j in range(TT))} = scf.for %ch = [%c0 to %chunks step %c1]({accs}) -> ({", ".join(["vector<8xf32>"] * TT)}) {{
    %ckc = index.constant {KC} : index
    %kc0 = index.mul %ch, %ckc : index
    %s_k = index.add %kc0, %s_col : index
    %s_k16 = index.div %s_k, %c16 : index""")
    for i in range(per // 16):
        T.append(f"""    %s_ki{i} = index.add %s_k16, %c{i if i < 3 else 0}{'' if i < 3 else ''} : index""" if i < 3 else
                 f"""    %s_kc{i} = index.constant {i} : index\n    %s_ki{i} = index.add %s_k16, %s_kc{i} : index""")
        T.append(f"""    %s_kb{i} = index.assume %s_ki{i} [range(%s_ki{i}, 0, {XH // 16 - 1})] : index
    %s_x{i} = vector.load %x_view[%s_tok, %head, %s_kb{i}, %c0] : view<[%tokens]x{H}x{XH // 16}x16xf32> -> vector<16xf32>
    %s_h{i} = vector.fptrunc %s_x{i} : vector<16xf32> to vector<16xf16>
    %s_ci{i} = index.constant {16 * i} : index
    %s_c{i} = index.add %s_col, %s_ci{i} : index
    vector.store %s_h{i}, %st_view[%s_tok_l, %s_c{i}] : vector<16xf16>, view<64x{LD}xf16>""")
    T.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    cur = {j: f"%a{j}" for j in range(TT)}
    for g in range(KC // 32):
        T.append(f"""    %cg{g} = index.constant {g} : index
    %g4_{g} = index.constant {KC // 32} : index
    %gq{g} = index.mul %ch, %g4_{g} : index
    %gi{g} = index.add %gq{g}, %cg{g} : index
    %gb{g} = index.assume %gi{g} [range(%gi{g}, 0, {K // 32 - 1})] : index
    %blk{g} = index.div %gb{g}, %cgpb : index
    %within{g} = index.rem %gb{g}, %cgpb : index
    %ba{g} = index.scale %blk{g}, %block_bytes : index, offset -> offset
    %origin{g} = index.add %ro, %ba{g} : offset
    %wv{g} = func.call @g_{fmt}_values(%w, %origin{g}, %within{g}) : (buffer, offset, index) -> (vector<32xf32>)
    %wlo{g} = vector.slice %wv{g}[0] : vector<32xf32> -> vector<16xf32>
    %whi{g} = vector.slice %wv{g}[16] : vector<32xf32> -> vector<16xf32>
    %wlo_h{g} = vector.fptrunc %wlo{g} : vector<16xf32> to vector<16xf16>
    %whi_h{g} = vector.fptrunc %whi{g} : vector<16xf32> to vector<16xf16>
    %b0_{g} = vector.fragment<rhs> %wlo_h{g} shape [%c16, %c16] : vector<16xf16>
    %b1_{g} = vector.fragment<rhs> %whi_h{g} shape [%c16, %c16] : vector<16xf16>""")
        for kk in range(2):
            T.append(f"    %fc{g}{kk} = index.constant {g * 32 + kk * 16} : index")
            for j in range(TT):
                T.append(f"""    %fr{g}{kk}{j} = index.constant {16 * j} : index
    %af{g}{kk}{j} = vector.fragment.load<lhs> %st_view[%fr{g}{kk}{j}, %fc{g}{kk}] shape [%c16, %c16] : view<64x{LD}xf16> -> vector<16xf16>
    %m{g}{kk}{j} = vector.mma %af{g}{kk}{j}, %b{kk}_{g}, {cur[j]} : vector<16xf16>, vector<16xf16>, vector<8xf32>""")
                cur[j] = f"%m{g}{kk}{j}"
    T.append(f"""    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    scf.yield {", ".join(cur[j] for j in range(TT))} : {", ".join(["vector<8xf32>"] * TT)}
  }}
  %cyt = index.constant {H * YH} : index
  %y_layout = encoding.layout.strided [%cyt, %c1] : encoding<layout>
  %cyh = index.constant {YH} : index
  %four = index.constant 4 : offset
  %hcol = index.mul %head, %cyh : index
  %hr0 = index.add %hcol, %r0 : index""")
    for j in range(TT):
        T.append(f"""  %tj{j}c = index.constant {16 * j} : index
  %tj{j}a = index.add %t0, %tj{j}c : index
  %yt{j} = index.mul %tj{j}a, %cyt : index
  %ye{j} = index.add %yt{j}, %hr0 : index
  %yo{j} = index.scale %ye{j}, %four : index, offset -> offset
  %yv{j} = buffer.view %y[%yo{j}] : buffer -> view<16x16xf32, %y_layout>
  vector.fragment.store<result> %o{j}, %yv{j}[%c0, %c0] shape [%c16, %c16] : vector<8xf32>, view<16x16xf32, %y_layout>""")
    T.append("  kernel.return\n}\n")
    return "\n".join(T)


def bmm_live(text):
    """MLA_BMM_LIVE (choice mla.prefill.bmm_live): a bmm/bmm2 kernel given the epoch's %info skips 64-token blocks that are all
    padding (padding is the epoch's tail: the block's first token has info word 2 = 0, a workgroup-uniform test; an epoch runs at the next power-of-two shape, so a 1151-token prompt pads 897 rows of a 2048
    epoch). Live tokens get the same values; padding rows of the output are not written (flash and v_b of padding queries are
    never read: their row scales are 0)."""
    a = "launch(%tokens_arg: index, %x: buffer, %w: buffer, %y: buffer) {"
    assert text.count(a) == 1
    text = text.replace(a, "launch(%tokens_arg: index, %x: buffer, %w: buffer, %y: buffer, %info: buffer) {")
    a = "  %tblk = index.assume %tb_id [lt(%tb_id, %tblocks)] : index\n"
    assert text.count(a) == 1
    text = text.replace(a, a + """  %lv_zero = index.constant 0 : offset
  %lv_view = buffer.view %info[%lv_zero] : buffer -> view<[%tokens]x4xi32>
  %lv_c64 = index.constant 64 : index
  %lv_t0 = index.mul %tblk, %lv_c64 : index
  %lv_a = index.assume %lv_t0 [lt(%lv_t0, %tokens)] : index
  %lv_w = view.load %lv_view[%lv_a, 2] : view<[%tokens]x4xi32> -> i32
  %lv_dead = scalar.constant 0 : i32
  %lv_live = scalar.cmpi ne, %lv_w, %lv_dead : i32
  scf.if %lv_live {
""")
    a = "  kernel.return\n}\n"
    assert text.count(a) == 1
    return text.replace(a, "  }\n" + a)
