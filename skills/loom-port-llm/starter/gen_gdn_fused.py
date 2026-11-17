"""gfx12 fused gated delta-net prefill in two launches (choice prefill.gdn.fused2, env GDN_FUSED2): `g_gdn_fa_rdna4` + `g_gdn_fb_rdna4`.

They replace, for an epoch of GEMM shape (>= 64 rows), the six launches g_gdn_conv, g_gdn_conv_commit, g_gdn_qk_norm, g_gdn_recurrence /
g_gdn_chunk_rdna4 and g_gdn_norm_gate. The chunked (WY) form is that of gen_gdn_chunk.py; the difference is WHERE the work runs:

  A (g_gdn_fa_rdna4, one workgroup per (value head, 32-token chunk, span): ~1500 workgroups at 1024 rows, no serial dependence):
     conv1d + SiLU of the chunk's q / k / v channels (inputs before the span from the stored history), q / k L2 norms, g = softplus(alpha +
     dt_bias) a, beta = sigmoid(beta), gc = cumsum(g), A = beta exp(gc_i - gc_j) k_i.k_j, T = (I + A)^-1 (one wave, as gen_gdn_chunk), then
         W = T (beta exp(gc) K)   U = T (beta V)   P = exp(gc_i - gc_j) q_i.k_j (j <= i)
     written to a workspace slot of the chunk (f16; gc f32); the normalized Q and K^T of the key head by its first value head.
  B (g_gdn_fb_rdna4, one workgroup per (value head, span), wave w owns value columns [16 w, 16 w + 16)): the serial pass over the chunks
     holds only the three state products (no solve, no staging, no conv):
         d = U - W S    O = exp(gc) (Q S) + P d    S <- exp(G) S + K^T (exp(G - gc) d)
     then the norm-gate epilogue of O (RMSNorm over the head, x w_norm x silu(z)), the state store and the conv history commit.

The serial pass in g_gdn_chunk_rdna4 ran the 32-row triangular solve, the staging loads and five barriers per chunk on 48 workgroups; here
the solve runs once per chunk in parallel, and the serial chain is three WMMA products per chunk with one LDS round trip (the norm).

Workspace slot of chunk c of span s: first_s / 32 + rank_s + c, rank_s = the span's rank by first row among the live spans (distinct for any
packing order; at most ceil(tokens / 32) + spans slots).
Padding rows of a chunk (beyond the span's end) have q = k = v = 0, beta = 0 and g = 0, so W, U, P, Q, K^T are zero there and change nothing.
"""

import re

CN = 32


def _common_consts(ci, extra=()):
    return "\n".join(f"  %k{k} = index.constant {k} : index" for k in sorted(set(ci) | set(extra)))


def nslots_bytes(G):
    """Workspace bytes per slot (W, U f16 [32][DV], P f16 [32][32], gc f32 [32] per value head; Q f16 [32][DK], K^T f16 [DK][32] per key head)."""
    HV, HK, DK, DV = G["HV"], G["HK"], G["DK"], G["DV"]
    return HV * (2 * CN * DV * 2 + CN * CN * 2 + CN * 4) + HK * 2 * CN * DK * 2


def _regions(G):
    """(name, bytes per slot) in workspace order."""
    HV, HK, DK, DV = G["HV"], G["HK"], G["DK"], G["DV"]
    return [("w", HV * CN * DV * 2), ("u", HV * CN * DV * 2), ("p", HV * CN * CN * 2), ("g", HV * CN * 4), ("q", HK * CN * DK * 2), ("kt", HK * DK * CN * 2)]


def _region_text(G):
    """Workspace region bases (offsets) from %nslots."""
    out, prev = [], None
    for name, b in _regions(G):
        out.append(f"  %wsz_{name} = index.constant {b} : offset")
        out.append(f"  %rs_{name} = index.scale %nslots, %wsz_{name} : index, offset -> offset")
    out.append("  %base_w = index.constant 0 : offset")
    names = [n for n, _ in _regions(G)]
    for a, b in zip(names, names[1:]):
        out.append(f"  %base_{b} = index.add %base_{a}, %rs_{a} : offset")
    return "\n".join(out)


def _blk(name, idx, nbytes, shape, ind="  ", sfx=""):
    """A view %v_<name><sfx> of one (slot, head) block of region `name`: `idx` = slot * heads + head (index SSA)."""
    n = name + sfx
    return (f"{ind}%bo_{n}_c = index.constant {nbytes} : offset\n"
            f"{ind}%bo_{n}_r = index.scale {idx}, %bo_{n}_c : index, offset -> offset\n"
            f"{ind}%bo_{n} = index.add %base_{name}, %bo_{n}_r : offset\n"
            f"{ind}%v_{n} = buffer.view %work[%bo_{n}] : buffer -> view<{shape}>")


def _head_ids(G, grouped, ratio):
    if grouped:
        kh = f"  %vratio = index.constant {ratio} : index\n  %kh = index.div %vh, %vratio : index\n  %vrem = index.rem %vh, %vratio : index\n  %kfirst = index.cmp eq, %vrem, %k0 : index"
    else:
        kh = f"  %chk = index.constant {G['HK']} : index\n  %kh = index.rem %vh, %chk : index\n  %kfirst = index.cmp ult, %vh, %chk : index"
    return kh


def phase_a(G, NG, LB, eps, grouped, ratio, target=True, abl=()):
    """abl: kbench timing ablations (wrong results): nosolve (no T solve), nostore (no workspace stores), noconv (no conv taps)."""
    HK, HV, DK, DV, C, K, SB = G["HK"], G["HV"], G["DK"], G["DV"], G["C"], G["K"], G["state_bytes"]
    assert DK == 128 and DV == 128 and K == 4, "g_gdn_fa_rdna4: 128 x 128 heads, conv kernel 4"
    NT = CN // 16
    vec8 = "vector<8xf32>"
    SH = "shape [%k16, %k16]"
    KHS, QAS, TS = DK + 8, DK + 8, CN + 8
    consts = set([0, 1, 2, 3, 8, 16, 32, 64, 128, 256, CN - 1])
    out = []
    P = out.append

    def ci(n):
        consts.add(n)
        return f"%k{n}"

    def bar():
        P("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    # ---- S1: per-token scalars (threads < CN)
    P("  %tcn = index.cmp ult, %thread, %k32 : index")
    P("  scf.if %tcn {")
    P(f"    %sr = index.assume %thread [range(%thread, 0, {CN - 1})] : index")
    P("    %slive = index.cmp ult, %sr, %m : index")
    P("    %stoff = index.add %tok0, %sr : index")
    P("    %stokc = scf.select %slive, %stoff, %tok0 : index")
    P("    %stok = index.assume %stokc [lt(%stokc, %tokens)] : index")
    P(f"    %al = view.load %alpha_view[%stok, %vh] : view<[%tokens]x{HV}xf32> -> f32")
    P(f"    %be = view.load %beta_view[%stok, %vh] : view<[%tokens]x{HV}xf32> -> f32")
    P("    %al_b = scalar.addf %al, %dt_bias_value : f32")
    P("    %al_p = scalar.softplusf<afn> %al_b : f32")
    P("    %gate = scalar.mulf %al_p, %a_value : f32")
    P("    %g_live = scf.select %slive, %gate, %zero_f : f32")
    P("    %bt = scalar.logisticf<afn> %be : f32")
    P("    %bt_live = scf.select %slive, %bt, %zero_f : f32")
    P(f"    view.store %g_live, %g_s[%sr] : f32, view<{CN}xf32>")
    P(f"    view.store %bt_live, %beta_s[%sr] : f32, view<{CN}xf32>")
    P("  }")
    bar()
    P("  scf.if %tcn {")
    P(f"    %tt = index.assume %thread [range(%thread, 0, {CN - 1})] : index")
    P("    %tt1 = index.add %tt, %k1 : index")
    P("    %gc = scf.for %j = [%k0 to %tt1 step %k1](%acc = %zero_f : f32) -> (f32) {")
    P(f"      %jb = index.assume %j [range(%j, 0, {CN - 1})] : index")
    P(f"      %gj = view.load %g_s[%jb] : view<{CN}xf32> -> f32")
    P("      %acc_n = scalar.addf %acc, %gj : f32")
    P("      scf.yield %acc_n : f32")
    P("    }")
    P("    %egv = scalar.expf<afn> %gc : f32")
    P(f"    view.store %gc, %gc_s[%tt] : f32, view<{CN}xf32>")
    P(f"    view.store %egv, %eg_s[%tt] : f32, view<{CN}xf32>")
    P(f"    view.store %gc, %v_g[%tt] : f32, view<{CN}xf32>")
    P("  }")
    # ---- S2: conv1d + SiLU; thread t: q/k channel t (q: kh's channels, k: kh's channels) and v channel t (t < DV)
    # rows -3..-1 come from the previous chunk of the span (in %qkv) or, for chunk 0, from the stored history (zeros when fresh)
    def conv_channel(tag, chv):
        P(f"  %w{tag}0 = view.load %w_view[{chv}, 0] : view<{C}x{K}xf32> -> f32")
        for j in range(1, K):
            P(f"  %w{tag}{j} = view.load %w_view[{chv}, {j}] : view<{C}x{K}xf32> -> f32")
        for r in range(-(K - 1), 0):
            hk = K - 1 + r
            u = f"{tag}{-r}"
            P(f"  %x{tag}m{-r} = scf.if %first_chunk -> (f32) {{")
            P(f"    %hz{u} = scf.if %fresh -> (f32) {{")
            P("      scf.yield %zero_f : f32")
            P("    } else {")
            P(f"      %hv{u} = view.load %hist_view[{ci(hk)}, {chv}] : view<{K - 1}x{C}xf32> -> f32")
            P(f"      scf.yield %hv{u} : f32")
            P("    }")
            P(f"    scf.yield %hz{u} : f32")
            P("  } else {")
            P(f"    %pr{u} = index.sub %tok0, {ci(-r)} : index")
            P(f"    %prb{u} = index.assume %pr{u} [lt(%pr{u}, %tokens)] : index")
            P(f"    %pv{u} = view.load %qkv_view[%prb{u}, {chv}] : view<[%tokens]x{C}xf32> -> f32")
            P(f"    scf.yield %pv{u} : f32")
            P("  }")
        for r in range(CN):
            P(f"  %x{tag}{r}_r = view.load %qkv_view[%trow{r}, {chv}] : view<[%tokens]x{C}xf32> -> f32")
        def xname(r):
            return f"%x{tag}m{-r}" if r < 0 else f"%x{tag}{r}_r"
        for r in range(CN):
            acc = "%zero_f"
            for j in range(K):
                P(f"  %c{tag}{r}_{j} = scalar.fmaf %w{tag}{j}, {xname(r - (K - 1) + j)}, {acc} : f32")
                acc = f"%c{tag}{r}_{j}"
            P(f"  %s{tag}{r}_g = scalar.logisticf<afn> {acc} : f32")
            P(f"  %s{tag}{r}_m = scalar.mulf {acc}, %s{tag}{r}_g : f32")
            P(f"  %s{tag}{r} = scf.select %rl{r}, %s{tag}{r}_m, %zero_f : f32")
    for r in range(CN):
        P(f"  %rl{r} = index.cmp ult, {ci(r)}, %m : index")
        P(f"  %tr{r}_a = index.add %tok0, {ci(r)} : index")
        P(f"  %tr{r}_c = scf.select %rl{r}, %tr{r}_a, %tok0 : index")
        P(f"  %trow{r} = index.assume %tr{r}_c [lt(%tr{r}_c, %tokens)] : index")
    P("  %first_chunk = index.cmp eq, %chunk, %k0 : index")
    # q/k channel of this thread: threads 0..127 q (kh * DK + t), 128..255 k (HK * DK + kh * DK + t - 128)
    P("  %isk = index.cmp uge, %thread, %k128 : index")
    P("  %tq = index.rem %thread, %k128 : index")
    P(f"  %qk_base_q = index.mul %kh, {ci(DK)} : index")
    P(f"  %qk_base_k = index.add %qk_base_q, {ci(HK * DK)} : index")
    P("  %qk_base = scf.select %isk, %qk_base_k, %qk_base_q : index")
    P("  %chqk_v = index.add %qk_base, %tq : index")
    P(f"  %chqk = index.assume %chqk_v [range(%chqk_v, 0, {C - 1})] : index")
    conv_channel("a", "%chqk")
    # q/k L2 norm: per row, a subgroup sum over the wave's 32 channels, partials of the 4 waves in LDS
    for r in range(CN):
        P(f"  %sq{r} = scalar.mulf %sa{r}, %sa{r} : f32")
        P(f"  %ss{r} = kernel.subgroup.reduce<addf> %sq{r} : f32")
    P("  %lead = index.cmp eq, %lane, %k0 : index")
    P("  scf.if %lead {")
    for r in range(CN):
        P(f"    view.store %ss{r}, %red_view[{ci(r)}, %wave] : f32, view<{CN}x8xf32>")
    P("  }")
    # v channel of threads < DV: beta * v as f16 (U's right operand)
    P(f"  %isv = index.cmp ult, %thread, {ci(DV)} : index")
    P("  scf.if %isv {")
    P(f"    %vtv = index.assume %thread [range(%thread, 0, {DV - 1})] : index")
    P(f"    %vbase_c = index.constant {2 * HK * DK} : index")
    P(f"    %vhb = index.mul %vh, {ci(DV)} : index")
    P("    %vch0 = index.add %vbase_c, %vhb : index")
    P("    %chv_v = index.add %vch0, %vtv : index")
    P(f"    %chv = index.assume %chv_v [range(%chv_v, 0, {C - 1})] : index")
    body_start = len(out)
    conv_channel("b", "%chv")
    for r in range(CN):
        P(f"  %bt{r} = view.load %beta_s[{ci(r)}] : view<{CN}xf32> -> f32")
        P(f"  %bv{r} = scalar.mulf %sb{r}, %bt{r} : f32")
        P(f"  %bvh{r} = scalar.fptrunc %bv{r} : f32 to f16")
        P(f"  view.store %bvh{r}, %bv_view[{ci(r)}, %vtv] : f16, view<{CN}x{KHS}xf16>")
    for i in range(body_start, len(out)):
        out[i] = "  " + out[i]
    P("  }")
    bar()
    # ---- S3: normalized q (x DK^-0.5) / k as f16 in LDS; beta exp(gc) k (W's right operand); the key head's Q / K^T by its first value head
    P("  %wq0 = index.cmp uge, %wave, %k4 : index")
    consts.add(4)
    P("  %wb = scf.select %wq0, %k4, %k0 : index")
    for r in range(CN):
        P(f"  %rd{r}_0 = view.load %red_view[{ci(r)}, %wb] : view<{CN}x8xf32> -> f32")
        acc = f"%rd{r}_0"
        for w in range(1, 4):
            P(f"  %rw{r}_{w} = index.add %wb, {ci(w)} : index")
            P(f"  %rd{r}_{w} = view.load %red_view[{ci(r)}, %rw{r}_{w}] : view<{CN}x8xf32> -> f32")
            P(f"  %ra{r}_{w} = scalar.addf {acc}, %rd{r}_{w} : f32")
            acc = f"%ra{r}_{w}"
        P(f"  %rs{r} = scalar.addf {acc}, %eps : f32")
        P(f"  %ri{r} = scalar.rsqrtf %rs{r} : f32")
        P(f"  %rq{r} = scalar.mulf %ri{r}, %qscale_c : f32")
        P(f"  %rf{r} = scf.select %isk, %ri{r}, %rq{r} : f32")
        P(f"  %n{r} = scalar.mulf %sa{r}, %rf{r} : f32")
        P(f"  %nh{r} = scalar.fptrunc %n{r} : f32 to f16")
    P("  scf.if %isk {")
    for r in range(CN):
        P(f"    view.store %nh{r}, %kh_view[{ci(r)}, %tq] : f16, view<{CN}x{KHS}xf16>")
        P(f"    %egr{r} = view.load %eg_s[{ci(r)}] : view<{CN}xf32> -> f32")
        P(f"    %btr{r} = view.load %beta_s[{ci(r)}] : view<{CN}xf32> -> f32")
        P(f"    %bek{r}_a = scalar.mulf %egr{r}, %btr{r} : f32")
        P(f"    %bek{r} = scalar.mulf %bek{r}_a, %n{r} : f32")
        P(f"    %bekh{r} = scalar.fptrunc %bek{r} : f32 to f16")
        P(f"    view.store %bekh{r}, %bek_view[{ci(r)}, %tq] : f16, view<{CN}x{KHS}xf16>")
    P("    scf.if %kfirst {")
    P(f"      %ktv = vector.from_elements {', '.join(f'%nh{r}' for r in range(CN))} : vector<{CN}xf16>")
    P(f"      vector.store %ktv, %v_kt[%tq, %k0] : vector<{CN}xf16>, view<{DK}x{CN}xf16>")
    P("    }")
    P("  } else {")
    for r in range(CN):
        P(f"    view.store %nh{r}, %q_view[{ci(r)}, %tq] : f16, view<{CN}x{QAS}xf16>")
    P("  }")
    bar()
    # the key head's normalized Q to the workspace by its first value head: rows of the LDS copy, 16-byte stores
    P("  scf.if %kfirst {")
    for r in range(2):
        P(f"    %qc{r}_v = index.add %thread, {ci(256 * r)} : index")
        P(f"    %qc{r} = index.assume %qc{r}_v [range(%qc{r}_v, 0, 511)] : index")
        P(f"    %qcr{r} = index.div %qc{r}, %k16 : index")
        P(f"    %qcq{r} = index.rem %qc{r}, %k16 : index")
        P(f"    %qcc{r} = index.mul %qcq{r}, %k8 : index")
        P(f"    %qcv{r} = vector.load %q_view[%qcr{r}, %qcc{r}] : view<{CN}x{QAS}xf16> -> vector<8xf16>")
        P(f"    vector.store %qcv{r}, %v_q[%qcr{r}, %qcc{r}] : vector<8xf16>, view<{CN}x{DK}xf16>")
    P("  }")
    # ---- S4: A = -beta exp(gc_i - gc_j) K K^T (waves 0-3, j < i) to LDS; P = exp(gc_i - gc_j) Q K^T (waves 4-7, j <= i) to the workspace
    P(f"  %tqw = index.rem %wave, {ci(NT * NT)} : index")
    P(f"  %tmt = index.div %tqw, {ci(NT)} : index")
    P(f"  %tnt = index.rem %tqw, {ci(NT)} : index")
    P("  %tr = index.mul %tmt, %k16 : index")
    P("  %tc = index.mul %tnt, %k16 : index")
    P("  %rb_v = index.add %tr, %lane_h8 : index")
    P(f"  %rb = index.assume %rb_v [range(%rb_v, 0, {CN - 8})] : index")
    P("  %cj_v = index.add %tc, %lane_row : index")
    P(f"  %cj = index.assume %cj_v [range(%cj_v, 0, {CN - 1})] : index")
    P("  %rb_i = index.cast %rb : index to i32")
    P("  %cj_i = index.cast %cj : index to i32")
    P("  %rows = vector.iota %rb_i step %one_i32 : vector<8xi32>")
    P("  %colv = vector.splat %cj_i : vector<8xi32>")
    P(f"  %gci = vector.load %gc_s[%rb] : view<{CN}xf32> -> {vec8}")
    P(f"  %gcj_s = view.load %gc_s[%cj] : view<{CN}xf32> -> f32")
    P(f"  %gcj = vector.splat %gcj_s : {vec8}")
    P(f"  %dg = vector.subf %gci, %gcj : {vec8}")
    P(f"  %dec = vector.expf<afn> %dg : {vec8}")
    P("  %is_kk = index.cmp ult, %wave, %k4 : index")
    P("  scf.if %is_kk {")
    prev = "%init"
    for kb in range(8):
        P(f"    %kkl{kb} = vector.fragment.load<lhs> %kh_view[%tr, {ci(16 * kb)}] {SH} : view<{CN}x{KHS}xf16> -> vector<8xf16>")
        P(f"    %kkr{kb} = vector.fragment.load<rhs> %kh_t[{ci(16 * kb)}, %tc] {SH} : view<{KHS}x{CN}xf16, %kh_tl> -> vector<8xf16>")
        P(f"    %kk{kb} = vector.mma %kkl{kb}, %kkr{kb}, {prev} : vector<8xf16>, vector<8xf16>, {vec8}")
        prev = f"%kk{kb}"
    P(f"    %bti = vector.load %beta_s[%rb] : view<{CN}xf32> -> {vec8}")
    P("    %lts = vector.cmpi sgt, %rows, %colv : vector<8xi32> -> vector<8xi1>")
    P(f"    %kkd = vector.mulf {prev}, %dec : {vec8}")
    P(f"    %kkb = vector.mulf %kkd, %bti : {vec8}")
    P(f"    %kkn = vector.subf %zero8, %kkb : {vec8}")
    P(f"    %av = vector.select %lts, %kkn, %zero8 : {vec8}")
    P(f"    vector.fragment.store<result> %av, %a_view[%tr, %tc] {SH} : {vec8}, view<{CN}x{CN}xf32>")
    P("  } else {")
    prev = "%init"
    for kb in range(8):
        P(f"    %qkl{kb} = vector.fragment.load<lhs> %q_view[%tr, {ci(16 * kb)}] {SH} : view<{CN}x{QAS}xf16> -> vector<8xf16>")
        P(f"    %qkr{kb} = vector.fragment.load<rhs> %kh_t[{ci(16 * kb)}, %tc] {SH} : view<{KHS}x{CN}xf16, %kh_tl> -> vector<8xf16>")
        P(f"    %qk{kb} = vector.mma %qkl{kb}, %qkr{kb}, {prev} : vector<8xf16>, vector<8xf16>, {vec8}")
        prev = f"%qk{kb}"
    P("    %les = vector.cmpi sge, %rows, %colv : vector<8xi32> -> vector<8xi1>")
    P(f"    %qkd = vector.mulf {prev}, %dec : {vec8}")
    P(f"    %pv = vector.select %les, %qkd, %zero8 : {vec8}")
    P(f"    vector.fragment.store<result> %pv, %v_p[%tr, %tc] {SH} : {vec8}, view<{CN}x{CN}xf16>")
    P("  }")
    bar()
    # ---- S5: T = (I + A)^-1 (threads < CN, one per column; gen_gdn_chunk's solve with its run-time-zero load dependence)
    P("  scf.if %tcn {")
    P(f"    %cc = index.assume %thread [range(%thread, 0, {CN - 1})] : index")
    dep = 2
    for i in range(CN):
        P(f"    %oh{i} = index.cmp eq, %cc, {ci(i)} : index")
        P(f"    %tv{i}_0 = scf.select %oh{i}, %one_f, %zero_f : f32")
        if i >= dep:
            P(f"    %db{i} = scalar.bitcast %T{i - dep} : f32 to i32")
            P(f"    %dz{i} = scalar.andi %db{i}, %zero_i32 : i32")
            P(f"    %dx{i} = index.cast %dz{i} : i32 to index")
            P(f"    %rv{i} = index.add {ci(i)}, %dx{i} : index")
            P(f"    %rw{i} = index.assume %rv{i} [range(%rv{i}, 0, {CN - 1})] : index")
            row = f"%rw{i}"
        else:
            row = ci(i)
        acc = f"%tv{i}_0"
        nfull = i // 4
        for g in range(nfull):
            P(f"    %ar{i}_{g} = vector.load %a_view[{row}, {ci(4 * g)}] : view<{CN}x{CN}xf32> -> vector<4xf32>")
            for e in range(4):
                m_ = 4 * g + e
                P(f"    %ae{i}_{m_} = vector.extract %ar{i}_{g}[{e}] : vector<4xf32> -> f32")
                P(f"    %tf{i}_{m_} = scalar.fmaf %ae{i}_{m_}, %T{m_}, {acc} : f32")
                acc = f"%tf{i}_{m_}"
        for m_ in range(4 * nfull, i):
            P(f"    %ae{i}_{m_} = view.load %a_view[{row}, {ci(m_)}] : view<{CN}x{CN}xf32> -> f32")
            P(f"    %tf{i}_{m_} = scalar.fmaf %ae{i}_{m_}, %T{m_}, {acc} : f32")
            acc = f"%tf{i}_{m_}"
        P(f"    %T{i} = scalar.addf {acc}, %zero_f : f32")
        P(f"    %T{i}h = scalar.fptrunc %T{i} : f32 to f16")
        P(f"    view.store %T{i}h, %t_view[{ci(i)}, %cc] : f16, view<{CN}x{TS}xf16>")
    P("  }")
    bar()
    # ---- S6: W = T (beta exp(gc) K), U = T (beta V): wave w owns value / key columns [16 w, 16 w + 16)
    P("  %wcol = index.mul %wave, %k16 : index")
    for mt in range(NT):
        for nm_, src, dst in (("w", "%bek_view", "%v_w"), ("u", "%bv_view", "%v_u")):
            prev = "%init"
            for kt in range(mt + 1):
                P(f"  %{nm_}tl{mt}_{kt} = vector.fragment.load<lhs> %t_view[{ci(16 * mt)}, {ci(16 * kt)}] {SH} : view<{CN}x{TS}xf16> -> vector<8xf16>")
                P(f"  %{nm_}rr{mt}_{kt} = vector.fragment.load<rhs> {src}[{ci(16 * kt)}, %wcol] {SH} : view<{CN}x{KHS}xf16> -> vector<8xf16>")
                P(f"  %{nm_}a{mt}_{kt} = vector.mma %{nm_}tl{mt}_{kt}, %{nm_}rr{mt}_{kt}, {prev} : vector<8xf16>, vector<8xf16>, {vec8}")
                prev = f"%{nm_}a{mt}_{kt}"
            if nm_ == "u":   # U^T: a lane's 8 rows are 16 contiguous bytes
                P(f"  vector.fragment.store<result> {prev}, {dst}[{ci(16 * mt)}, %wcol] {SH} : {vec8}, view<{CN}x{DV}xf16, %ut_l>")
            else:            # W through LDS (the K buffer is free after S4): row stores below
                P(f"  vector.fragment.store<result> {prev}, %kh_view[{ci(16 * mt)}, %wcol] {SH} : {vec8}, view<{CN}x{KHS}xf16>")

    bar()
    for r in range(2):
        P(f"  %wc{r}_v = index.add %thread, {ci(256 * r)} : index")
        P(f"  %wc{r} = index.assume %wc{r}_v [range(%wc{r}_v, 0, 511)] : index")
        P(f"  %wcr{r} = index.div %wc{r}, %k16 : index")
        P(f"  %wcq{r} = index.rem %wc{r}, %k16 : index")
        P(f"  %wcc{r} = index.mul %wcq{r}, %k8 : index")
        P(f"  %wcv{r} = vector.load %kh_view[%wcr{r}, %wcc{r}] : view<{CN}x{KHS}xf16> -> vector<8xf16>")
        P(f"  vector.store %wcv{r}, %v_w[%wcr{r}, %wcc{r}] : vector<8xf16>, view<{CN}x{DV}xf16>")
    if "nosolve" in abl:
        a = out.index("  scf.if %tcn {", out.index("    %cc = index.assume %thread [range(%thread, 0, 31)] : index") - 1)
        b = out.index("  %wcol = index.mul %wave, %k16 : index")
        out[a:b] = ["  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)"]
    if "nostore" in abl:
        out = [l for l in out if not re.search(r"(vector\.store|view\.store|vector\.fragment\.store<result>) [^\n]*%v_(w|u|p|g|q|kt)\[", l)]
    if "noconv" in abl:
        out = [re.sub(r"scalar\.fmaf %w[ab]\d+, (%x[ab](m\d|\d+_r)), (%\w+) : f32", r"scalar.addf \1, \3 : f32", l) for l in out]
    body = "\n".join(out)
    tgt = "target(@rdna4) " if target else ""
    kname = "g_gdn_fa_rdna4" + "".join("_" + x for x in abl)
    head = f"""
kernel.def {tgt}export("{kname}") @{kname}(%tokens: index, %spans: index) {{
  %c31 = index.constant {CN - 1} : index
  %c32 = index.constant {CN} : index
  %heads = index.constant {HV} : index
  %c256 = index.constant 256 : index
  %c1 = index.constant 1 : index
  %ta = index.add %tokens, %c31 : index
  %chunks = index.div %ta, %c32 : index
  kernel.launch.config workgroups(%heads, %chunks, %spans) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %layer_arg: index, %metadata: buffer, %origins: buffer, %arena: buffer, %qkv: buffer, %weight: buffer, %alpha: buffer, %beta: buffer, %ssm_a: buffer, %dt_bias: buffer, %work: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {NG - 1})] : index
  %head_id = kernel.workgroup.id<x> : index
  %vh = index.assume %head_id [range(%head_id, 0, {HV - 1})] : index
  %chunk_id = kernel.workgroup.id<y> : index
  %chunk = index.assume %chunk_id [range(%chunk_id, 0, 65534)] : index
  %span_id = kernel.workgroup.id<z> : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
  %thread_id = kernel.workitem.id<x> : index
  %thread = index.assume %thread_id [range(%thread_id, 0, 255)] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, 7)] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %zero_f = scalar.constant 0.0 : f32
  %one_f = scalar.constant 1.0 : f32
  %one_i32 = scalar.constant 1 : i32
  %eps = scalar.constant {eps!r} : f32
@CONSTS@
  %active, %length, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
  %c0tok = index.mul %chunk, %k32 : index
  %has = index.cmp ult, %c0tok, %length : index
  %mine = scalar.andi %active, %has : i1
  scf.if %mine {{
  %rem = index.sub %length, %c0tok : index
  %m = index.min %rem, %k32 : index
  %tok0_v = index.add %first, %c0tok : index
  %tok0 = index.assume %tok0_v [lt(%tok0_v, %tokens)] : index
  %layer_stride = index.constant {LB} : offset
  %layer_bytes = index.scale %layer, %layer_stride : index, offset -> offset
  %reader, %writer, %fresh = func.call @g_gdn_slots(%origins, %row, %start32, %layer_bytes) : (buffer, index, i32, offset) -> (offset, offset, i1)
  %history_skip = index.constant {SB} : offset
  %history_at = index.add %reader, %history_skip : offset
  %hist_view = buffer.view %arena[%history_at] : buffer -> view<{K - 1}x{C}xf32>
  %qkv_view = buffer.view %qkv[%zero] : buffer -> view<[%tokens]x{C}xf32>
  %w_view = buffer.view %weight[%zero] : buffer -> view<{C}x{K}xf32>
  %alpha_view = buffer.view %alpha[%zero] : buffer -> view<[%tokens]x{HV}xf32>
  %beta_view = buffer.view %beta[%zero] : buffer -> view<[%tokens]x{HV}xf32>
  %a_view_g = buffer.view %ssm_a[%zero] : buffer -> view<{HV}xf32>
  %bias_view = buffer.view %dt_bias[%zero] : buffer -> view<{HV}xf32>
  %a_value = view.load %a_view_g[%vh] : view<{HV}xf32> -> f32
  %dt_bias_value = view.load %bias_view[%vh] : view<{HV}xf32> -> f32
  %qscale_c = scalar.constant {DK ** -0.5!r} : f32
  %zero8 = vector.splatf 0.0 : vector<8xf32>
  %zero16 = vector.splati 0 : vector<16xi32>
  %init = vector.fragment<init> %zero8 {SH} : vector<8xf32>
{_head_ids(G, grouped, ratio)}
  %lane_row = index.rem %lane, %k16 : index
  %lane_half = index.div %lane, %k16 : index
  %lane_h8_v = index.mul %lane_half, %k8 : index
  %lane_h8 = index.assume %lane_h8_v [range(%lane_h8_v, 0, 8)] : index
  %ta2 = index.add %tokens, %k31 : index
  %nchk = index.div %ta2, %k32 : index
  %ns_a = index.add %nchk, %spans : index
  %nslots = index.add %ns_a, %k1 : index
{_region_text(G)}
  %fdiv = index.div %first, %k32 : index
  %rank = func.call @g_gdn_fused_rank(%metadata, %spans, %first) : (buffer, index, index) -> (index)
  %slot_a = index.add %fdiv, %rank : index
  %slot = index.add %slot_a, %chunk : index
  %sv_v = index.mul %slot, {ci(HV)} : index
  %svh = index.add %sv_v, %vh : index
  %sk_v = index.mul %slot, {ci(HK)} : index
  %skh = index.add %sk_v, %kh : index
{_blk("w", "%svh", CN * DV * 2, f"{CN}x{DV}xf16")}
  %ut_l = encoding.layout.strided [1, {CN}] : encoding<layout>
{_blk("u", "%svh", CN * DV * 2, f"{CN}x{DV}xf16, %ut_l")}
{_blk("p", "%svh", CN * CN * 2, f"{CN}x{CN}xf16")}
{_blk("g", "%svh", CN * 4, f"{CN}xf32")}
{_blk("q", "%skh", CN * DK * 2, f"{CN}x{DK}xf16")}
{_blk("kt", "%skh", DK * CN * 2, f"{DK}x{CN}xf16")}
  %kh_bytes = index.constant {CN * KHS * 2} : offset
  %kh_buf = buffer.alloca<workgroup> align(16) %kh_bytes : buffer
  %kh_view = buffer.view %kh_buf[%zero] : buffer -> view<{CN}x{KHS}xf16>
  %kh_tl = encoding.layout.strided [1, {KHS}] : encoding<layout>
  %kh_t = buffer.view %kh_buf[%zero] : buffer -> view<{KHS}x{CN}xf16, %kh_tl>
  %q_buf = buffer.alloca<workgroup> align(16) %kh_bytes : buffer
  %q_view = buffer.view %q_buf[%zero] : buffer -> view<{CN}x{QAS}xf16>
  %bek_buf = buffer.alloca<workgroup> align(16) %kh_bytes : buffer
  %bek_view = buffer.view %bek_buf[%zero] : buffer -> view<{CN}x{KHS}xf16>
  %bv_buf = buffer.alloca<workgroup> align(16) %kh_bytes : buffer
  %bv_view = buffer.view %bv_buf[%zero] : buffer -> view<{CN}x{KHS}xf16>
  %a_bytes = index.constant {CN * CN * 4} : offset
  %a_buf = buffer.alloca<workgroup> align(16) %a_bytes : buffer
  %a_view = buffer.view %a_buf[%zero] : buffer -> view<{CN}x{CN}xf32>
  %t_bytes = index.constant {CN * TS * 2} : offset
  %t_buf = buffer.alloca<workgroup> align(16) %t_bytes : buffer
  %t_view = buffer.view %t_buf[%zero] : buffer -> view<{CN}x{TS}xf16>
  %red_bytes = index.constant {CN * 8 * 4} : offset
  %red_buf = buffer.alloca<workgroup> align(16) %red_bytes : buffer
  %red_view = buffer.view %red_buf[%zero] : buffer -> view<{CN}x8xf32>
  %sc_bytes = index.constant 1024 : offset
  %sc_buf = buffer.alloca<workgroup> align(16) %sc_bytes : buffer
  %o1 = index.constant 128 : offset
  %o2 = index.constant 256 : offset
  %o3 = index.constant 384 : offset
  %o5 = index.constant 512 : offset
  %g_s = buffer.view %sc_buf[%zero] : buffer -> view<{CN}xf32>
  %gc_s = buffer.view %sc_buf[%o1] : buffer -> view<{CN}xf32>
  %eg_s = buffer.view %sc_buf[%o2] : buffer -> view<{CN}xf32>
  %beta_s = buffer.view %sc_buf[%o3] : buffer -> view<{CN}xf32>
  %zero_cell = buffer.view %sc_buf[%o5] : buffer -> view<16xi32>
  %thread_is0 = index.cmp eq, %thread, %k0 : index
  scf.if %thread_is0 {{
    vector.store %zero16, %zero_cell[%k0] : vector<16xi32>, view<16xi32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %zero_i32 = view.load %zero_cell[%k0] : view<16xi32> -> i32
"""
    head = head.replace("@CONSTS@", _common_consts(consts))
    text = head + body + "\n  }\n  kernel.return\n}\n"
    from gdn_fa_liveness import enabled as share_enabled, rewrite as share_lds
    return share_lds(text) if not abl and share_enabled() else text


def phase_b(G, NG, LB, eps, grouped, ratio, guard_open="", guard_close="", target=True, abl=()):
    """abl: kbench timing ablations: nonorm (no norm-gate epilogue)."""
    HK, HV, DK, DV, C, K, VD, SB = G["HK"], G["HV"], G["DK"], G["DV"], G["C"], G["K"], G["VD"], G["state_bytes"]
    NS = HV * DV * DK
    NT = CN // 16
    vec8 = "vector<8xf32>"
    SH = "shape [%k16, %k16]"
    OS = DV + 4
    consts = set([0, 1, 4, 8, 16, 31, 32, CN - 1])
    out = []
    P = out.append

    def ci(n):
        consts.add(n)
        return f"%k{n}"
    WP, TP = DV + 8, CN + 8   # stage pitches (f16): the 32 x 128 W / Q rows, the 128 x 32 K^T / U^T and 32 x 32 P rows
    # per thread: W, Q, K^T, U^T pieces t and t + 256 (8 f16), P piece t < 128, gc piece t < 8 (4 f32)
    PIECES = [("w", "%svh", CN * DV * 2, f"{CN}x{DV}xf16", f"{CN}x{WP}xf16", 16, 2), ("q", "%skh", CN * DK * 2, f"{CN}x{DK}xf16", f"{CN}x{WP}xf16", 16, 2),
              ("kt", "%skh", DK * CN * 2, f"{DK}x{CN}xf16", f"{DK}x{TP}xf16", 4, 2), ("u", "%svh", CN * DV * 2, f"{DV}x{CN}xf16", f"{DV}x{TP}xf16", 4, 2),
              ("p", "%svh", CN * CN * 2, f"{CN}x{CN}xf16", f"{CN}x{TP}xf16", 4, 1)]

    def prefetch(chunk, ind):
        sx = "_" + chunk.strip("%")
        L = [f"{ind}%pslot{sx} = index.add %slot0, {chunk} : index", f"{ind}%psv{sx}_v = index.mul %pslot{sx}, {ci(HV)} : index",
             f"{ind}%svh{sx} = index.add %psv{sx}_v, %vh : index", f"{ind}%psk{sx}_v = index.mul %pslot{sx}, {ci(HK)} : index",
             f"{ind}%skh{sx} = index.add %psk{sx}_v, %kh : index"]
        L.append(_blk("g", "%svh" + sx, CN * 4, f"{CN}xf32", ind, sx))
        for nm_, idx, nb, gshp, _, per_row, reps in PIECES:
            L.append(_blk(nm_, idx + sx, nb, gshp, ind, sx))
            for r in range(reps):
                pc = f"%pc_{nm_}{r}{sx}"
                L.append(f"{ind}{pc}_v = index.add %thread, {ci(256 * r)} : index")
                L.append(f"{ind}{pc} = index.assume {pc}_v [range({pc}_v, 0, {256 * reps - 1})] : index")
                L.append(f"{ind}%prow_{nm_}{r}{sx} = index.div {pc}, {ci(per_row)} : index")
                L.append(f"{ind}%pcq_{nm_}{r}{sx} = index.rem {pc}, {ci(per_row)} : index")
                L.append(f"{ind}%pcol_{nm_}{r}{sx} = index.mul %pcq_{nm_}{r}{sx}, %k8 : index")
                ld = f"vector.load %v_{nm_}{sx}[%prow_{nm_}{r}{sx}, %pcol_{nm_}{r}{sx}] : view<{gshp}> -> vector<8xf16>"
                if nm_ == "p":
                    L += [f"{ind}%pf_{nm_}{r}{sx} = scf.if %t128 -> (vector<8xf16>) {{", f"{ind}  %pl_{nm_}{r}{sx} = {ld}", f"{ind}  scf.yield %pl_{nm_}{r}{sx} : vector<8xf16>",
                          f"{ind}}} else {{", f"{ind}  scf.yield %zero8h : vector<8xf16>", f"{ind}}}"]
                else:
                    L.append(f"{ind}%pf_{nm_}{r}{sx} = {ld}")
        L += [f"{ind}%pf_g{sx} = scf.if %t8 -> (vector<4xf32>) {{", f"{ind}  %pl_g{sx} = vector.load %v_g{sx}[%t4o] : view<{CN}xf32> -> vector<4xf32>",
              f"{ind}  scf.yield %pl_g{sx} : vector<4xf32>", f"{ind}}} else {{", f"{ind}  scf.yield %zero4 : vector<4xf32>", f"{ind}}}"]
        return L

    def stage_stores(ind, sx):
        L = []
        for nm_, idx, nb, gshp, sshp, per_row, reps in PIECES:
            for r in range(reps):
                st = f"vector.store %pf_{nm_}{r}{sx}, %s_{nm_}[%prow_{nm_}{r}{sx}, %pcol_{nm_}{r}{sx}] : vector<8xf16>, view<{sshp}>"
                L += ([f"{ind}scf.if %t128 {{", f"{ind}  {st}", f"{ind}}}"] if nm_ == "p" else [f"{ind}{st}"])
        L += [f"{ind}scf.if %t8 {{", f"{ind}  vector.store %pf_g{sx}, %s_g[%t4o] : vector<4xf32>, view<{CN}xf32>", f"{ind}}}"]
        return L
    P("    %c0tok = index.mul %ci, %k32 : index")
    P("    %rem = index.sub %length, %c0tok : index")
    P("    %m = index.min %rem, %k32 : index")
    P("    %tok0 = index.add %first, %c0tok : index")
    # z of this lane's norm rows (independent of the chain: in flight during it)
    for rr in range(4):
        P(f"    %nr{rr}_a = index.mul %wave, %k4 : index")
        P(f"    %nr{rr}_b = index.add %nr{rr}_a, {ci(rr)} : index")
        P(f"    %nr{rr} = index.assume %nr{rr}_b [range(%nr{rr}_b, 0, {CN - 1})] : index")
        P(f"    %olive{rr} = index.cmp ult, %nr{rr}, %m : index")
        P(f"    %otok{rr}_a = index.add %tok0, %nr{rr} : index")
        P(f"    %otok{rr}_c = scf.select %olive{rr}, %otok{rr}_a, %tok0 : index")
        P(f"    %otok{rr} = index.assume %otok{rr}_c [lt(%otok{rr}_c, %tokens)] : index")
    for kb in range(8):
        P(f"    %sh{kb} = vector.fptrunc %s{kb} : {vec8} to vector<8xf16>")
        P(f"    %sf{kb} = vector.fragment<rhs> %sh{kb} {SH} : vector<8xf16>")
    # d = U - W S (result layout: lane holds rows 16 mt + lane_h8 + e of value column dv)
    for mt in range(NT):
        prev = "%init"
        wr = ci(16 * mt)   # v0.2: a fragment.load base must be uniform (fragment_memory.dynamic_stride): no run-time-zero ordering term here
        for kb in range(8):
            P(f"    %wl{mt}_{kb} = vector.fragment.load<lhs> %s_w[{wr}, {ci(16 * kb)}] {SH} : view<{CN}x{WP}xf16> -> vector<8xf16>")
            P(f"    %ws{mt}_{kb} = vector.mma %wl{mt}_{kb}, %sf{kb}, {prev} : vector<8xf16>, vector<8xf16>, {vec8}")
            prev = f"%ws{mt}_{kb}"
        P(f"    %ri{mt}_v = index.add {ci(16 * mt)}, %lane_h8 : index")
        P(f"    %ri{mt} = index.assume %ri{mt}_v [range(%ri{mt}_v, 0, {CN - 8})] : index")
        P(f"    %uh{mt} = vector.load %s_u[%dv, %ri{mt}] : view<{DV}x{TP}xf16> -> vector<8xf16>")
        P(f"    %uv{mt} = vector.extf %uh{mt} : vector<8xf16> to {vec8}")
        P(f"    %d{mt} = vector.subf %uv{mt}, {prev} : {vec8}")
        P(f"    %dh{mt} = vector.fptrunc %d{mt} : {vec8} to vector<8xf16>")
        P(f"    %df{mt} = vector.fragment<rhs> %dh{mt} {SH} : vector<8xf16>")
        P(f"    %gcv{mt} = vector.load %s_g[%ri{mt}] : view<{CN}xf32> -> {vec8}")
    P(f"    %gl = view.load %s_g[%k{CN - 1}] : view<{CN}xf32> -> f32")
    P(f"    %glv = vector.splat %gl : {vec8}")
    # O = exp(gc) (Q S) + P d (v0.2: the Q, W and K^T fragment loads carry no run-time-zero ordering term; the SDK requires a uniform fragment.load base)
    for mt in range(NT):
        prev = "%init"
        for kb in range(8):
            P(f"    %ql{mt}_{kb} = vector.fragment.load<lhs> %s_q[{ci(16 * mt)}, {ci(16 * kb)}] {SH} : view<{CN}x{WP}xf16> -> vector<8xf16>")
            P(f"    %qs{mt}_{kb} = vector.mma %ql{mt}_{kb}, %sf{kb}, {prev} : vector<8xf16>, vector<8xf16>, {vec8}")
            prev = f"%qs{mt}_{kb}"
        P(f"    %egc{mt} = vector.expf<afn> %gcv{mt} : {vec8}")
        P(f"    %oi{mt} = vector.mulf %egc{mt}, {prev} : {vec8}")
        prev = f"%oi{mt}"
        for kt in range(mt + 1):
            P(f"    %pl{mt}_{kt} = vector.fragment.load<lhs> %s_p[{ci(16 * mt)}, {ci(16 * kt)}] {SH} : view<{CN}x{TP}xf16> -> vector<8xf16>")
            P(f"    %oo{mt}_{kt} = vector.mma %pl{mt}_{kt}, %df{kt}, {prev} : vector<8xf16>, vector<8xf16>, {vec8}")
            prev = f"%oo{mt}_{kt}"
        P(f"    vector.fragment.store<result> {prev}, %o_lds[{ci(16 * mt)}, %wcol] {SH} : {vec8}, view<{CN}x{OS}xf32>")
    # S <- exp(G) S + K^T (exp(G - gc) d)
    for mt in range(NT):
        P(f"    %dgm{mt} = vector.subf %glv, %gcv{mt} : {vec8}")
        P(f"    %edm{mt} = vector.expf<afn> %dgm{mt} : {vec8}")
        P(f"    %de{mt} = vector.mulf %d{mt}, %edm{mt} : {vec8}")
        P(f"    %deh{mt} = vector.fptrunc %de{mt} : {vec8} to vector<8xf16>")
        P(f"    %dfe{mt} = vector.fragment<rhs> %deh{mt} {SH} : vector<8xf16>")
    # the next chunk prefetch and the z loads take a run-time-zero term of O (after the Q S chain)
    P(f"    %dz2_e = vector.extract %oo{NT - 1}_{NT - 1}[0] : vector<8xf32> -> f32")
    P("    %dz2_b = scalar.bitcast %dz2_e : f32 to i32")
    P("    %dz2_i = scalar.andi %dz2_b, %zero_i32 : i32")
    P("    %dz2 = index.cast %dz2_i : i32 to index")
    P("    %cn_a = index.add %ci, %k1 : index")
    P("    %cn_last = index.sub %nchunks, %k1 : index")
    P("    %cn_b = index.min %cn_a, %cn_last : index")
    P("    %cn = index.add %cn_b, %dz2 : index")
    out.extend(prefetch("%cn", "    "))
    for rr in range(4):   # z of this lane's norm rows: in flight during the S update (after O: not over the W S / Q S chains)
        P(f"    %ztok{rr} = index.add %otok{rr}, %dz2 : index")
        P(f"    %zta{rr} = index.assume %ztok{rr} [range(%ztok{rr}, 0, 8191)] : index")
        P(f"    %ztb{rr} = index.assume %zta{rr} [lt(%zta{rr}, %tokens)] : index")
        P(f"    %zv{rr} = vector.load %z_view[%ztb{rr}, %zcol] : view<[%tokens]x{VD}xf32> -> vector<4xf32>")
    P(f"    %eg_last = scalar.expf<afn> %gl : f32")
    P(f"    %eg_v = vector.splat %eg_last : {vec8}")
    for kb in range(8):
        P(f"    %sdv{kb} = vector.mulf %s{kb}, %eg_v : {vec8}")
        P(f"    %sd{kb} = vector.fragment<init> %sdv{kb} {SH} : {vec8}")
        prev = f"%sd{kb}"
        for kt in range(NT):
            P(f"    %ul{kb}_{kt} = vector.fragment.load<lhs> %s_kt[{ci(16 * kb)}, {ci(16 * kt)}] {SH} : view<{DK}x{TP}xf16> -> vector<8xf16>")
            P(f"    %un{kb}_{kt} = vector.mma %ul{kb}_{kt}, %dfe{kt}, {prev} : vector<8xf16>, vector<8xf16>, {vec8}")
            prev = f"%un{kb}_{kt}"
        P(f"    %sn{kb} = vector.addf {prev}, %zero8 : {vec8}")
    P("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    out.extend(stage_stores("    ", "_cn"))
    # norm-gate epilogue: wave w takes rows 4 w .. 4 w + 3, lane l value columns 4 l .. 4 l + 3
    for rr in range(4):
        P(f"    %ov{rr} = vector.load %o_lds[%nr{rr}, %lc4] : view<{CN}x{OS}xf32> -> vector<4xf32>")
        P(f"    %osq{rr} = vector.mulf %ov{rr}, %ov{rr} : vector<4xf32>")
        P(f"    %osr{rr} = vector.reduce<addf> %osq{rr}, %zero_f : vector<4xf32>, f32")
        P(f"    %osum{rr} = kernel.subgroup.reduce<addf> %osr{rr} : f32")
        P(f"    %omean{rr} = scalar.divf %osum{rr}, %count : f32")
        P(f"    %ost{rr} = scalar.addf %omean{rr}, %eps : f32")
        P(f"    %oinv{rr} = scalar.rsqrtf %ost{rr} : f32")
        P(f"    %oiv{rr} = vector.splat %oinv{rr} : vector<4xf32>")
        P(f"    %zs{rr} = vector.siluf<afn> %zv{rr} : vector<4xf32>")
        P(f"    %on{rr}_a = vector.mulf %ov{rr}, %oiv{rr} : vector<4xf32>")
        P(f"    %on{rr}_b = vector.mulf %on{rr}_a, %nwv : vector<4xf32>")
        P(f"    %on{rr} = vector.mulf %on{rr}_b, %zs{rr} : vector<4xf32>")
    for rr in range(4):
        P(f"    scf.if %olive{rr} {{")
        P(f"      vector.store %on{rr}, %o_view[%otok{rr}, %zcol] : vector<4xf32>, view<[%tokens]x{VD}xf32>")
        P("    }")
    P("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    P(f"    scf.yield {', '.join(f'%sn{kb}' for kb in range(8))} : {', '.join([vec8] * 8)}")
    if "nonorm" in abl:
        txt = "\n".join(out)
        txt = re.sub(r"\n    scf\.if %olive\d \{\n[^\n]*\n    \}", "", txt)
        out = [l for l in txt.split("\n") if not re.search(r"%(ov|osq|osr|osum|omean|ost|oinv|oiv|zs|on)\d", l)]
    loop = "\n".join(out)
    # conv history commit (the span's last K-1 inputs; after phase A read the old ones): v channels of this head (threads < DV), the key
    # head's q and k channels by its first value head (thread t: q channel t, k channel t, t < DK)
    commit = []
    for k in range(K - 1):
        need = K - 1 - k
        commit.append(f"""      %need{k}~ = index.constant {need} : index
      %has{k}~ = index.cmp uge, %length, %need{k}~ : index
      %val{k}~ = scf.if %has{k}~ -> (f32) {{
        %off{k}~a = index.add %first, %length : index
        %src{k}~ = index.sub %off{k}~a, %need{k}~ : index
        %bsrc{k}~ = index.assume %src{k}~ [lt(%src{k}~, %tokens)] : index
        %v{k}~ = view.load %qkv_view[%bsrc{k}~, %ch~] : view<[%tokens]x{C}xf32> -> f32
        scf.yield %v{k}~ : f32
      }} else {{
        %o{k}~ = scf.if %fresh -> (f32) {{
          scf.yield %zero_f : f32
        }} else {{
          %hk{k}~a = index.constant {k} : index
          %hk{k}~b = index.add %hk{k}~a, %length : index
          %hk{k}~ = index.assume %hk{k}~b [range(%hk{k}~b, 0, {K - 2})] : index
          %hv{k}~ = view.load %old_view[%hk{k}~, %ch~] : view<{K - 1}x{C}xf32> -> f32
          scf.yield %hv{k}~ : f32
        }}
        scf.yield %o{k}~ : f32
      }}""")
    cwrites = [f"      view.store %val{k}~, %new_view[{k}, %ch~] : f32, view<{K - 1}x{C}xf32>" for k in range(K - 1)]

    def commit_for(tag, chexpr):
        t = "\n".join(commit).replace("~", tag) + "\n" + guard_open.replace("%publish", f"%publish{tag}") + "\n".join(cwrites).replace("~", tag) + guard_close
        return f"      %ch{tag}_v = {chexpr}\n      %ch{tag} = index.assume %ch{tag}_v [range(%ch{tag}_v, 0, {C - 1})] : index\n" + t
    commits = f"""
  %history_skip = index.constant {SB} : offset
  %old_at = index.add %reader, %history_skip : offset
  %new_at = index.add %writer, %history_skip : offset
  %old_view = buffer.view %arena[%old_at] : buffer -> view<{K - 1}x{C}xf32>
  %new_view = buffer.view %arena[%new_at] : buffer -> view<{K - 1}x{C}xf32>
  %cdk = index.constant {DK} : index
  %cmt = index.cmp ult, %thread, %cdk : index
  scf.if %cmt {{
    %ct = index.assume %thread [range(%thread, 0, {DK - 1})] : index
    %vb_c = index.constant {2 * HK * DK} : index
    %vb_h = index.mul %vh, %cdk : index
    %vb = index.add %vb_c, %vb_h : index
{commit_for("v", "index.add %vb, %ct : index")}
    scf.if %kfirst {{
      %qb = index.mul %kh, %cdk : index
      %kb_c = index.constant {HK * DK} : index
      %kb = index.add %kb_c, %qb : index
{commit_for("q", "index.add %qb, %ct : index")}
{commit_for("k", "index.add %kb, %ct : index")}
    }}
  }}
"""
    tgt = "target(@rdna4) " if target else ""
    kname = "g_gdn_fb_rdna4" + "".join("_" + x for x in abl)
    head = f"""
kernel.def {tgt}export("{kname}") @{kname}(%spans: index) {{
  %c1 = index.constant 1 : index
  %heads = index.constant {HV} : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%heads, %c1, %spans) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %layer_arg: index, %metadata: buffer, %origins: buffer, %arena: buffer, %qkv: buffer, %work: buffer, %weight: buffer, %z: buffer, %out: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {NG - 1})] : index
  %head_id = kernel.workgroup.id<x> : index
  %vh = index.assume %head_id [range(%head_id, 0, {HV - 1})] : index
  %span_id = kernel.workgroup.id<z> : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
  %thread_id = kernel.workitem.id<x> : index
  %thread = index.assume %thread_id [range(%thread_id, 0, 255)] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, 7)] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %zero_f = scalar.constant 0.0 : f32
  %eps = scalar.constant {eps!r} : f32
  %count = scalar.constant {float(DV)!r} : f32
@CONSTS@
  %active, %length, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
  scf.if %active {{
  %layer_stride = index.constant {LB} : offset
  %layer_bytes = index.scale %layer, %layer_stride : index, offset -> offset
  %reader, %writer, %fresh = func.call @g_gdn_slots(%origins, %row, %start32, %layer_bytes) : (buffer, index, i32, offset) -> (offset, offset, i1)
  %qkv_view = buffer.view %qkv[%zero] : buffer -> view<[%tokens]x{C}xf32>
  %z_view = buffer.view %z[%zero] : buffer -> view<[%tokens]x{VD}xf32>
  %o_view = buffer.view %out[%zero] : buffer -> view<[%tokens]x{VD}xf32>
  %nw_view = buffer.view %weight[%zero] : buffer -> view<{DV}xf32>
  %state_view = buffer.view %arena[%reader] : buffer -> view<{NS}xf32>
  %next_view = buffer.view %arena[%writer] : buffer -> view<{NS}xf32>
  %zero8 = vector.splatf 0.0 : vector<8xf32>
  %zero16 = vector.splati 0 : vector<16xi32>
  %init = vector.fragment<init> %zero8 {SH} : vector<8xf32>
{_head_ids(G, grouped, ratio)}
  %k_dk = index.constant {DK} : index
  %k_dv = index.constant {DV} : index
  %lane_row = index.rem %lane, %k16 : index
  %lane_half = index.div %lane, %k16 : index
  %lane_h8_v = index.mul %lane_half, %k8 : index
  %lane_h8 = index.assume %lane_h8_v [range(%lane_h8_v, 0, 8)] : index
  %wcol = index.mul %wave, %k16 : index
  %dv_local = index.add %wcol, %lane_row : index
  %dv = index.assume %dv_local [range(%dv_local, 0, {DV - 1})] : index
  %head_cols = index.mul %vh, %k_dv : index
  %srow_dv = index.add %head_cols, %dv : index
  %sidx0 = index.mul %srow_dv, %k_dk : index
  %sidx_l = index.add %sidx0, %lane_h8 : index
  %lc4_v = index.mul %lane, %k4 : index
  %lc4 = index.assume %lc4_v [range(%lc4_v, 0, {DV - 4})] : index
  %zcol = index.add %head_cols, %lc4 : index
  %nwv = vector.load %nw_view[%lc4] : view<{DV}xf32> -> vector<4xf32>
  %ta2 = index.add %tokens, %k31 : index
  %nchk = index.div %ta2, %k32 : index
  %ns_a = index.add %nchk, %spans : index
  %nslots = index.add %ns_a, %k1 : index
{_region_text(G)}
  %fdiv = index.div %first, %k32 : index
  %rank = func.call @g_gdn_fused_rank(%metadata, %spans, %first) : (buffer, index, index) -> (index)
  %slot0 = index.add %fdiv, %rank : index
  %ob = index.constant {CN * OS * 4} : offset
  %o_buf = buffer.alloca<workgroup> align(16) %ob : buffer
  %o_lds = buffer.view %o_buf[%zero] : buffer -> view<{CN}x{OS}xf32>
@STAGE@
  %sc_bytes = index.constant 64 : offset
  %sc_buf = buffer.alloca<workgroup> align(16) %sc_bytes : buffer
  %zero_cell = buffer.view %sc_buf[%zero] : buffer -> view<16xi32>
  %thread_is0 = index.cmp eq, %thread, %k0 : index
  scf.if %thread_is0 {{
    vector.store %zero16, %zero_cell[%k0] : vector<16xi32>, view<16xi32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %zero_i32 = view.load %zero_cell[%k0] : view<16xi32> -> i32
"""
    init = []
    init.append(f"  {', '.join(f'%ist{kb}' for kb in range(8))} = scf.if %fresh -> ({', '.join([vec8] * 8)}) {{")
    init.append(f"    scf.yield {', '.join(['%zero8'] * 8)} : {', '.join([vec8] * 8)}")
    init.append("  } else {")
    for kb in range(8):
        init.append(f"    %sk{kb}_i = index.add %sidx_l, {ci(16 * kb)} : index")
        init.append(f"    %sl{kb} = vector.load %state_view[%sk{kb}_i] : view<{NS}xf32> -> {vec8}")
    init.append(f"    scf.yield {', '.join(f'%sl{kb}' for kb in range(8))} : {', '.join([vec8] * 8)}")
    init.append("  }")
    nch = ["  %len_r = index.add %length, %k31 : index", "  %nchunks = index.div %len_r, %k32 : index"] + prefetch("%k0", "  ") + stage_stores("  ", "_k0") + \
          ["  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)"]
    stage = [f"  %zero8h = vector.splatf 0.0 : vector<8xf16>", "  %zero4 = vector.splatf 0.0 : vector<4xf32>",
             f"  %t128 = index.cmp ult, %thread, {ci(128)} : index", f"  %t8 = index.cmp ult, %thread, %k8 : index",
             "  %t4o_v = index.mul %thread, %k4 : index", f"  %t4o = index.assume %t4o_v [range(%t4o_v, 0, {CN - 4})] : index"]
    off = 0
    for nm_, b in (("w", CN * WP * 2), ("q", CN * WP * 2), ("kt", DK * TP * 2), ("u", DV * TP * 2), ("p", CN * TP * 2), ("g", CN * 4)):
        shp = {"w": f"{CN}x{WP}xf16", "q": f"{CN}x{WP}xf16", "kt": f"{DK}x{TP}xf16", "u": f"{DV}x{TP}xf16", "p": f"{CN}x{TP}xf16", "g": f"{CN}xf32"}[nm_]
        stage += [f"  %sb_{nm_} = index.constant {b} : offset", f"  %sbuf_{nm_} = buffer.alloca<workgroup> align(16) %sb_{nm_} : buffer",
                  f"  %s_{nm_} = buffer.view %sbuf_{nm_}[%zero] : buffer -> view<{shp}>"]
    head = head.replace("@STAGE@", "\n".join(stage))
    carry = ", ".join(f"%s{kb} = %ist{kb} : {vec8}" for kb in range(8))
    outs = ", ".join(f"%fin{kb}" for kb in range(8))
    loop_head = f"  {outs} = scf.for %ci = [%k0 to %nchunks step %k1]({carry}) -> ({', '.join([vec8] * 8)}) {{"
    stores = [f"    %so{kb}_i = index.add %sidx_l, {ci(16 * kb)} : index\n    vector.store %fin{kb}, %next_view[%so{kb}_i] : {vec8}, view<{NS}xf32>" for kb in range(8)]
    head = head.replace("@CONSTS@", _common_consts(consts))
    # the history commit first: its loads are independent of the chunk loop (after it, the scheduler hoisted them over the loop)
    return (head + commits + "\n".join(init) + "\n" + "\n".join(nch) + "\n" + loop_head + "\n" + loop + "\n  }\n"
            + guard_open + "\n".join(stores) + "\n" + guard_close + "\n  }\n  kernel.return\n}\n")


RANK = """
// Rank of a live span (length > 0) among the epoch's live spans ordered by first row: the workspace slots of span s are
// floor(first_s / 32) + rank_s + chunk, distinct for any packing order of the spans (rows of different spans never overlap).
func.def inline @g_gdn_fused_rank(%metadata: buffer, %spans: index, %first: index) -> (index) {
  %zero = index.constant 0 : offset
  %c16 = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %count_bound = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count_raw = index.cast %count_bound : i32 to index
  %count = index.min %count_raw, %spans : index
  %packets = buffer.view %metadata[%c16] : buffer -> view<[%spans]x9xi32>
  %rank = scf.for %j = [%c0 to %count step %c1](%acc = %c0 : index) -> (index) {
    %jb = index.assume %j [lt(%j, %spans)] : index
    %len32 = view.load %packets[%jb, 0] : view<[%spans]x9xi32> -> i32
    %fst32 = view.load %packets[%jb, 4] : view<[%spans]x9xi32> -> i32
    %len_b = scalar.assume %len32 [range(%len32, 0, 8192)] : i32
    %fst_b = scalar.assume %fst32 [range(%fst32, 0, 8191)] : i32
    %len = index.cast %len_b : i32 to index
    %fst = index.cast %fst_b : i32 to index
    %ne = index.cmp ugt, %len, %c0 : index
    %lt = index.cmp ult, %fst, %first : index
    %before = scalar.andi %ne, %lt : i1
    %inc = scf.select %before, %c1, %c0 : index
    %acc_n = index.add %acc, %inc : index
    scf.yield %acc_n : index
  }
  %rank_b = index.assume %rank [range(%rank, 0, 8191)] : index
  func.return %rank_b : index
}
"""


def fused_kernels(G, NG, LB, eps, grouped, ratio, guard_open="", guard_close="", target=True):
    return RANK + phase_a(G, NG, LB, eps, grouped, ratio, target) + "\n" + phase_b(G, NG, LB, eps, grouped, ratio, guard_open, guard_close, target)
