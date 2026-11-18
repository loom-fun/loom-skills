"""gfx12 chunked (WY) gated delta-net prefill kernel: `g_gdn_chunk_rdna4`.

The sequential recurrence (gen_gdn.py `_recurrence_rows`) is ~7.5 % of Qwen3.8-27B 2k prefill on the R9700 (VALU-bound, one serial token
loop per value head). This kernel runs a span of >= SPAN_MIN tokens in chunks of CN = 32 tokens per value head with f16 WMMA (f32
accumulation, f32 state): the llama.cpp `build_delta_net_chunking` / fla `chunk_gated_delta_rule` form.

Per chunk, with gc = cumsum(g) over the chunk, Q rows q_i (L2-normalised, 1/sqrt(DK) folded in), K rows k_i, S = state^T [dk][dv]:
    A[i][j] = beta_i exp(gc_i - gc_j) k_i.k_j  (j < i)             T = (I + A)^-1  (CN sequential rows, f32, one thread per column)
    R = beta v - beta exp(gc) (K S)            d = T R             (the delta-rule updates, [t][dv])
    O = exp(gc) (Q S) + P d,   P[i][j] = exp(gc_i - gc_j) q_i.k_j (j <= i)
    S <- exp(G) S + (K * exp(G - gc))^T d                          (G = gc of the chunk's last token)

The state S lives in registers as the f32 accumulator tiles of the state update (wave w owns value columns [16w, 16w + 16), all 128 key
rows = 8 result fragments); its f16 copy is the rhs fragment of the K S and Q S products as it is (flash2_rdna4.py: the result layout of S^T
is the rhs layout). K, Q, T, P and K^T exp(G - gc) are staged in LDS as f16, v as f32. Padding tokens of the last chunk have beta = 0, g = 0
and zero K/Q rows, so they change nothing. Same launch signature as g_gdn_recurrence; a span of fewer than SPAN_MIN tokens is not this
kernel's (the sequential kernel takes it).

Why a run-time-zero dependence in the solve: Loom's scheduler hoists every independent load of a basic block to its top, so the unrolled
CN-row solve (CN * (CN - 1) / 2 independent LDS loads of A) ran out of registers (peak 270-1000 VGPRs at CN = 32-64; scf.if regions are
if-converted away). The A loads of row i take their row index plus a term that is zero at run time (the bits of T[i - 2] and an LDS zero),
which the compiler cannot know: at most two rows of loads are in flight.
"""

CN = 32
NT = CN // 16
SPAN_MIN = 64   # spans of at least this many tokens (the GEMM epoch shape) run the chunked kernel; shorter ones the sequential twin


def chunk_kernel(G, NG, LB, guard_open="", guard_close="", target=True, dep=2):
    from gen_gdn import kh_line, _V_GROUPED
    chk_def = "" if _V_GROUPED[0] else f"%chk = index.constant {G['HK']} : index\n  "   # kh_line's tiled form reads %chk; the grouped form needs none
    HK, HV, DK, DV, C, VD = G["HK"], G["HV"], G["DK"], G["DV"], G["C"], G["VD"]
    assert DK == 128 and DV == 128 and HV % HK == 0, "g_gdn_chunk_rdna4 is built for 128 x 128 heads (8 waves x 16 value columns)"
    NS = HV * DV * DK
    consts = set()

    def ci(n):
        consts.add(n)
        return f"%k{n}"
    vec8 = "vector<8xf32>"
    SH = "shape [%k16, %k16]"
    KHS, QAS, PS, TS, KDS, VS = DK + 8, DK + 8, CN + 8, CN + 8, CN + 4, DV + 4
    out = []
    P = out.append

    def bar():
        P("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")

    # ------------------------------------------------------------------------------------------------------------------
    P(f"    %c0tok = index.mul %ci, {ci(CN)} : index")
    P(f"    %rem = index.sub %length, %c0tok : index")
    P(f"    %m = index.min %rem, {ci(CN)} : index")
    P(f"    %m_last_a = index.sub %m, %k1 : index")
    P(f"    %m_last = index.assume %m_last_a [range(%m_last_a, 0, {CN - 1})] : index")
    P(f"    %vbase_tok = index.add %first, %c0tok : index")
    # ---- P0: stage the chunk (thread -> row thread / 8, 16 columns of K, Q and V)
    P(f"    %srow = index.div %thread, {ci(8)} : index")
    P(f"    %spart = index.rem %thread, {ci(8)} : index")
    P(f"    %scol = index.mul %spart, {ci(16)} : index")
    P(f"    %slive = index.cmp ult, %srow, %m : index")
    P(f"    %stoff = index.add %c0tok, %srow : index")
    P(f"    %stokc = scf.select %slive, %stoff, %c0tok : index")
    P(f"    %stabs_v = index.add %first, %stokc : index")
    P(f"    %stabs = index.assume %stabs_v [lt(%stabs_v, %tokens)] : index")
    P(f"    %kinv0 = view.load %inv_view[%stabs, %k_norm_index] : view<[%tokens]x{2 * HK}xf32> -> f32")
    P(f"    %qinv0 = view.load %inv_view[%stabs, %q_norm_index] : view<[%tokens]x{2 * HK}xf32> -> f32")
    P(f"    %qinv1 = scalar.mulf %qinv0, %qscale_c : f32")
    P(f"    %kinv = scf.select %slive, %kinv0, %zero_f : f32")
    P(f"    %qinv = scf.select %slive, %qinv1, %zero_f : f32")
    P(f"    %kinv_v = vector.splat %kinv : vector<16xf32>")
    P(f"    %qinv_v = vector.splat %qinv : vector<16xf32>")
    P(f"    %kcol = index.add %kcol_base, %scol : index")
    P(f"    %qcol = index.add %qcol_base, %scol : index")
    P(f"    %vcol16 = index.add %vcol_head, %scol : index")
    P(f"    %kr = vector.load %x_view[%stabs, %kcol] : view<[%tokens]x{C}xf32> -> vector<16xf32>")
    P(f"    %qr = vector.load %x_view[%stabs, %qcol] : view<[%tokens]x{C}xf32> -> vector<16xf32>")
    P(f"    %vr = vector.load %x_view[%stabs, %vcol16] : view<[%tokens]x{C}xf32> -> vector<16xf32>")
    P(f"    %kn = vector.mulf %kr, %kinv_v : vector<16xf32>")
    P(f"    %qn = vector.mulf %qr, %qinv_v : vector<16xf32>")
    P(f"    %kh16 = vector.fptrunc %kn : vector<16xf32> to vector<16xf16>")
    P(f"    %qh16 = vector.fptrunc %qn : vector<16xf32> to vector<16xf16>")
    P(f"    vector.store %kh16, %kh_view[%srow, %scol] : vector<16xf16>, view<{CN}x{KHS}xf16>")
    P(f"    vector.store %qh16, %q_view[%srow, %scol] : vector<16xf16>, view<{CN}x{QAS}xf16>")
    P(f"    vector.store %vr, %v_view[%srow, %scol] : vector<16xf32>, view<{CN}x{VS}xf32>")
    # per-token scalars: threads with part 0 (rows 0..CN-1 live in threads 0, 8, 16, ...)
    P(f"    %is_p0 = index.cmp eq, %spart, %k0 : index")
    P(f"    scf.if %is_p0 {{")
    P(f"      %al = view.load %alpha_view[%stabs, %vh] : view<[%tokens]x{HV}xf32> -> f32")
    P(f"      %be = view.load %beta_view[%stabs, %vh] : view<[%tokens]x{HV}xf32> -> f32")
    P(f"      %al_b = scalar.addf %al, %dt_bias_value : f32")
    P(f"      %al_p = scalar.softplusf<afn> %al_b : f32")
    P(f"      %gate = scalar.mulf %al_p, %a_value : f32")
    P(f"      %g_live = scf.select %slive, %gate, %zero_f : f32")
    P(f"      %bt = scalar.logisticf<afn> %be : f32")
    P(f"      %bt_live = scf.select %slive, %bt, %zero_f : f32")
    P(f"      view.store %g_live, %g_s[%srow] : f32, view<{CN}xf32>")
    P(f"      view.store %bt_live, %beta_s[%srow] : f32, view<{CN}xf32>")
    P(f"    }}")
    bar()
    # ---- P1: inclusive scan of g (threads < CN)
    P(f"    %tcn = index.cmp ult, %thread, {ci(CN)} : index")
    P(f"    scf.if %tcn {{")
    P(f"      %tt = index.assume %thread [range(%thread, 0, {CN - 1})] : index")
    P(f"      %tt1 = index.add %tt, %k1 : index")
    P(f"      %gc = scf.for %j = [%k0 to %tt1 step %k1](%acc = %zero_f : f32) -> (f32) {{")
    P(f"        %jb = index.assume %j [range(%j, 0, {CN - 1})] : index")
    P(f"        %gj = view.load %g_s[%jb] : view<{CN}xf32> -> f32")
    P(f"        %acc_n = scalar.addf %acc, %gj : f32")
    P(f"        scf.yield %acc_n : f32")
    P(f"      }}")
    P(f"      %egv = scalar.expf<afn> %gc : f32")
    P(f"      view.store %gc, %gc_s[%tt] : f32, view<{CN}xf32>")
    P(f"      view.store %egv, %eg_s[%tt] : f32, view<{CN}xf32>")
    P(f"    }}")
    bar()
    # ---- P2: S as rhs fragments; K S and Q S; this wave's tile of K K^T (waves 0-3) or Q K^T (waves 4-7)
    for kb in range(8):
        P(f"    %sh{kb} = vector.fptrunc %s{kb} : {vec8} to vector<8xf16>")
        P(f"    %sf{kb} = vector.fragment<rhs> %sh{kb} {SH} : vector<8xf16>")
    for nmx, vw, sz in (("zk", "%kh_view", KHS), ("zq", "%q_view", QAS)):
        for mt in range(NT):
            prev = "%init"
            for kb in range(8):
                P(f"    %{nmx}l{mt}_{kb} = vector.fragment.load<lhs> {vw}[{ci(16 * mt)}, {ci(16 * kb)}] {SH} : view<{CN}x{sz}xf16> -> vector<8xf16>")
                P(f"    %{nmx}{mt}_{kb} = vector.mma %{nmx}l{mt}_{kb}, %sf{kb}, {prev} : vector<8xf16>, vector<8xf16>, {vec8}")
                prev = f"%{nmx}{mt}_{kb}"
    P(f"    %tq = index.rem %wave, {ci(NT * NT)} : index")
    P(f"    %tmt = index.div %tq, {ci(NT)} : index")
    P(f"    %tnt = index.rem %tq, {ci(NT)} : index")
    P(f"    %tr = index.mul %tmt, %k16 : index")
    P(f"    %tc = index.mul %tnt, %k16 : index")
    P(f"    %rb_v = index.add %tr, %lane_h8 : index")
    P(f"    %rb = index.assume %rb_v [range(%rb_v, 0, {CN - 8})] : index")
    P(f"    %cj_v = index.add %tc, %lane_row : index")
    P(f"    %cj = index.assume %cj_v [range(%cj_v, 0, {CN - 1})] : index")
    P(f"    %rb_i = index.cast %rb : index to i32")
    P(f"    %cj_i = index.cast %cj : index to i32")
    P(f"    %rows = vector.iota %rb_i step %one_i32 : vector<8xi32>")
    P(f"    %colv = vector.splat %cj_i : vector<8xi32>")
    P(f"    %gci = vector.load %gc_s[%rb] : view<{CN}xf32> -> {vec8}")
    P(f"    %gcj_s = view.load %gc_s[%cj] : view<{CN}xf32> -> f32")
    P(f"    %gcj = vector.splat %gcj_s : {vec8}")
    P(f"    %dg = vector.subf %gci, %gcj : {vec8}")
    P(f"    %dec = vector.expf<afn> %dg : {vec8}")
    P(f"    %is_kk = index.cmp ult, %wave, {ci(4)} : index")
    P(f"    scf.if %is_kk {{")
    prev = "%init"
    for kb in range(8):
        P(f"      %kkl{kb} = vector.fragment.load<lhs> %kh_view[%tr, {ci(16 * kb)}] {SH} : view<{CN}x{KHS}xf16> -> vector<8xf16>")
        P(f"      %kkr{kb} = vector.fragment.load<rhs> %kh_t[{ci(16 * kb)}, %tc] {SH} : view<{KHS}x{CN}xf16, %kh_tl> -> vector<8xf16>")
        P(f"      %kk{kb} = vector.mma %kkl{kb}, %kkr{kb}, {prev} : vector<8xf16>, vector<8xf16>, {vec8}")
        prev = f"%kk{kb}"
    P(f"      %bti = vector.load %beta_s[%rb] : view<{CN}xf32> -> {vec8}")
    P(f"      %lts = vector.cmpi sgt, %rows, %colv : vector<8xi32> -> vector<8xi1>")
    P(f"      %kkd = vector.mulf {prev}, %dec : {vec8}")
    P(f"      %kkb = vector.mulf %kkd, %bti : {vec8}")
    P(f"      %kkn = vector.subf %zero8, %kkb : {vec8}")
    P(f"      %av = vector.select %lts, %kkn, %zero8 : {vec8}")
    P(f"      vector.fragment.store<result> %av, %a_view[%tr, %tc] {SH} : {vec8}, view<{CN}x{CN}xf32>")
    P(f"    }} else {{")
    prev = "%init"
    for kb in range(8):
        P(f"      %qkl{kb} = vector.fragment.load<lhs> %q_view[%tr, {ci(16 * kb)}] {SH} : view<{CN}x{QAS}xf16> -> vector<8xf16>")
        P(f"      %qkr{kb} = vector.fragment.load<rhs> %kh_t[{ci(16 * kb)}, %tc] {SH} : view<{KHS}x{CN}xf16, %kh_tl> -> vector<8xf16>")
        P(f"      %qk{kb} = vector.mma %qkl{kb}, %qkr{kb}, {prev} : vector<8xf16>, vector<8xf16>, {vec8}")
        prev = f"%qk{kb}"
    P(f"      %les = vector.cmpi sge, %rows, %colv : vector<8xi32> -> vector<8xi1>")
    P(f"      %qkd = vector.mulf {prev}, %dec : {vec8}")
    P(f"      %pv = vector.select %les, %qkd, %zero8 : {vec8}")
    P(f"      vector.fragment.store<result> %pv, %p_view[%tr, %tc] {SH} : {vec8}, view<{CN}x{PS}xf16>")
    P(f"    }}")
    bar()
    # ---- P3: waves 0 solves T = (I + A)^-1 (one thread per column), waves 1-4 stage K^T exp(G - gc) as f16 [dk][t]
    P(f"    scf.if %tcn {{")
    P(f"      %cc = index.assume %thread [range(%thread, 0, {CN - 1})] : index")
    for i in range(CN):
        P(f"      %oh{i} = index.cmp eq, %cc, {ci(i)} : index")
        P(f"      %tv{i}_0 = scf.select %oh{i}, %one_f, %zero_f : f32")
        if i >= dep:
            P(f"      %db{i} = scalar.bitcast %T{i - dep} : f32 to i32")
            P(f"      %dz{i} = scalar.andi %db{i}, %zero_i32 : i32")
            P(f"      %dx{i} = index.cast %dz{i} : i32 to index")
            P(f"      %rv{i} = index.add {ci(i)}, %dx{i} : index")
            P(f"      %rw{i} = index.assume %rv{i} [range(%rv{i}, 0, {CN - 1})] : index")
            row = f"%rw{i}"
        else:
            row = ci(i)
        acc = f"%tv{i}_0"
        nfull = i // 4
        for g in range(nfull):
            P(f"      %ar{i}_{g} = vector.load %a_view[{row}, {ci(4 * g)}] : view<{CN}x{CN}xf32> -> vector<4xf32>")
            for e in range(4):
                m_ = 4 * g + e
                P(f"      %ae{i}_{m_} = vector.extract %ar{i}_{g}[{e}] : vector<4xf32> -> f32")
                P(f"      %tf{i}_{m_} = scalar.fmaf %ae{i}_{m_}, %T{m_}, {acc} : f32")
                acc = f"%tf{i}_{m_}"
        for m_ in range(4 * nfull, i):
            P(f"      %ae{i}_{m_} = view.load %a_view[{row}, {ci(m_)}] : view<{CN}x{CN}xf32> -> f32")
            P(f"      %tf{i}_{m_} = scalar.fmaf %ae{i}_{m_}, %T{m_}, {acc} : f32")
            acc = f"%tf{i}_{m_}"
        P(f"      %T{i} = scalar.addf {acc}, %zero_f : f32")
        P(f"      %T{i}h = scalar.fptrunc %T{i} : f32 to f16")
        P(f"      view.store %T{i}h, %t_view[{ci(i)}, %cc] : f16, view<{CN}x{TS}xf16>")
    P(f"    }} else {{")
    P(f"      %ktid = index.sub %thread, {ci(CN)} : index")
    P(f"      %kstage = index.cmp ult, %ktid, {ci(DK)} : index")
    P(f"      scf.if %kstage {{")
    P(f"        %kdk = index.assume %ktid [range(%ktid, 0, {DK - 1})] : index")
    P(f"        %gl = view.load %gc_s[%k{CN - 1}] : view<{CN}xf32> -> f32")
    P(f"        %glv = vector.splat %gl : vector<8xf32>")
    for q4 in range(CN // 8):
        P(f"        %gcq{q4} = vector.load %gc_s[{ci(8 * q4)}] : view<{CN}xf32> -> {vec8}")
        P(f"        %dgq{q4} = vector.subf %glv, %gcq{q4} : {vec8}")
        P(f"        %ewq{q4} = vector.expf<afn> %dgq{q4} : {vec8}")
        els = []
        for e in range(8):
            P(f"        %kv{q4}_{e} = view.load %kh_view[{ci(8 * q4 + e)}, %kdk] : view<{CN}x{KHS}xf16> -> f16")
            els.append(f"%kv{q4}_{e}")
        P(f"        %kvv{q4} = vector.from_elements {', '.join(els)} : vector<8xf16>")
        P(f"        %kvf{q4} = vector.extf %kvv{q4} : vector<8xf16> to {vec8}")
        P(f"        %kdv{q4}_f = vector.mulf %kvf{q4}, %ewq{q4} : {vec8}")
        P(f"        %kdv{q4} = vector.fptrunc %kdv{q4}_f : {vec8} to vector<8xf16>")
        P(f"        vector.store %kdv{q4}, %kdt_view[%kdk, {ci(8 * q4)}] : vector<8xf16>, view<{DK}x{KDS}xf16>")
    P(f"      }}")
    P(f"    }}")
    bar()
    # ---- P4: R = beta (v - exp(gc) K S), d = T R
    for mt in range(NT):
        P(f"    %ri{mt}_v = index.add {ci(16 * mt)}, %lane_h8 : index")
        P(f"    %ri{mt} = index.assume %ri{mt}_v [range(%ri{mt}_v, 0, {CN - 8})] : index")
        P(f"    %bti_{mt} = vector.load %beta_s[%ri{mt}] : view<{CN}xf32> -> {vec8}")
        P(f"    %egi_{mt} = vector.load %eg_s[%ri{mt}] : view<{CN}xf32> -> {vec8}")
        els = []
        for e in range(8):
            P(f"    %vt{mt}_{e}_o = index.add %ri{mt}, {ci(e)} : index")
            P(f"    %vv{mt}_{e} = view.load %v_view[%vt{mt}_{e}_o, %dv] : view<{CN}x{VS}xf32> -> f32")
            els.append(f"%vv{mt}_{e}")
        P(f"    %vvec{mt} = vector.from_elements {', '.join(els)} : {vec8}")
        P(f"    %ezk{mt} = vector.mulf %egi_{mt}, %zk{mt}_7 : {vec8}")
        P(f"    %vmz{mt} = vector.subf %vvec{mt}, %ezk{mt} : {vec8}")
        P(f"    %rr{mt} = vector.mulf %bti_{mt}, %vmz{mt} : {vec8}")
        P(f"    %rh{mt} = vector.fptrunc %rr{mt} : {vec8} to vector<8xf16>")
        P(f"    %rf{mt} = vector.fragment<rhs> %rh{mt} {SH} : vector<8xf16>")
    for mt in range(NT):
        prev = "%init"
        for kt in range(mt + 1):
            P(f"    %tl{mt}_{kt} = vector.fragment.load<lhs> %t_view[{ci(16 * mt)}, {ci(16 * kt)}] {SH} : view<{CN}x{TS}xf16> -> vector<8xf16>")
            P(f"    %dd{mt}_{kt} = vector.mma %tl{mt}_{kt}, %rf{kt}, {prev} : vector<8xf16>, vector<8xf16>, {vec8}")
            prev = f"%dd{mt}_{kt}"
        P(f"    %dh{mt} = vector.fptrunc {prev} : {vec8} to vector<8xf16>")
        P(f"    %df{mt} = vector.fragment<rhs> %dh{mt} {SH} : vector<8xf16>")
    # ---- P5: O = exp(gc) Q S + P d
    for mt in range(NT):
        P(f"    %oi{mt} = vector.mulf %egi_{mt}, %zq{mt}_7 : {vec8}")
        prev = f"%oi{mt}"
        for kt in range(mt + 1):
            P(f"    %pl{mt}_{kt} = vector.fragment.load<lhs> %p_view[{ci(16 * mt)}, {ci(16 * kt)}] {SH} : view<{CN}x{PS}xf16> -> vector<8xf16>")
            P(f"    %oo{mt}_{kt} = vector.mma %pl{mt}_{kt}, %df{kt}, {prev} : vector<8xf16>, vector<8xf16>, {vec8}")
            prev = f"%oo{mt}_{kt}"
        for e in range(8):
            P(f"    %olt{mt}_{e}_r = index.add {ci(16 * mt + e)}, %lane_h8 : index")
            P(f"    %olt{mt}_{e} = index.cmp ult, %olt{mt}_{e}_r, %m : index")
            P(f"    scf.if %olt{mt}_{e} {{")
            P(f"      %ov{mt}_{e} = vector.extract {prev}[{e}] : {vec8} -> f32")
            P(f"      %orw{mt}_{e}_a = index.add %vbase_tok, %olt{mt}_{e}_r : index")
            P(f"      %orw{mt}_{e} = index.assume %orw{mt}_{e}_a [lt(%orw{mt}_{e}_a, %tokens)] : index")
            P(f"      view.store %ov{mt}_{e}, %out_view[%orw{mt}_{e}, %ocol_l] : f32, view<[%tokens]x{VD}xf32>")
            P(f"    }}")
    # ---- P6: S <- exp(G) S + (K exp(G - gc))^T d
    P(f"    %gl2 = view.load %gc_s[%k{CN - 1}] : view<{CN}xf32> -> f32")
    P(f"    %eg_last = scalar.expf<afn> %gl2 : f32")
    P(f"    %eg_v = vector.splat %eg_last : {vec8}")
    for kb in range(8):
        P(f"    %sdv{kb} = vector.mulf %s{kb}, %eg_v : {vec8}")
        P(f"    %sd{kb} = vector.fragment<init> %sdv{kb} {SH} : {vec8}")
        prev = f"%sd{kb}"
        for kt in range(NT):
            P(f"    %ul{kb}_{kt} = vector.fragment.load<lhs> %kdt_view[{ci(16 * kb)}, {ci(16 * kt)}] {SH} : view<{DK}x{KDS}xf16> -> vector<8xf16>")
            P(f"    %un{kb}_{kt} = vector.mma %ul{kb}_{kt}, %df{kt}, {prev} : vector<8xf16>, vector<8xf16>, {vec8}")
            prev = f"%un{kb}_{kt}"
        P(f"    %sn{kb} = vector.addf {prev}, %zero8 : {vec8}")
    bar()
    P(f"    scf.yield {', '.join(f'%sn{kb}' for kb in range(8))} : {', '.join([vec8] * 8)}")
    chunk_text = out
    # ------------------------------------------------------------------------------------------------------------------
    tgt = "target(@rdna4) " if target else ""
    head = f"""
kernel.def {tgt}export("g_gdn_chunk_rdna4") @g_gdn_chunk_rdna4(%spans: index) {{
  %c1 = index.constant 1 : index
  %heads = index.constant {HV} : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%heads, %c1, %spans) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %layer_arg: index, %metadata: buffer, %origins: buffer, %arena: buffer, %x: buffer, %inverse: buffer, %alpha: buffer, %beta: buffer, %ssm_a: buffer, %dt_bias: buffer, %out: buffer) {{
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
  %one_f = scalar.constant 1.0 : f32
  %one_i32 = scalar.constant 1 : i32
  %active, %length, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
  %k_min_len = index.constant {SPAN_MIN} : index
  %in_class = index.cmp uge, %length, %k_min_len : index
  %mine = scalar.andi %active, %in_class : i1
  scf.if %mine {{
  %layer_stride = index.constant {LB} : offset
  %layer_bytes = index.scale %layer, %layer_stride : index, offset -> offset
  %reader, %writer, %fresh = func.call @g_gdn_slots(%origins, %row, %start32, %layer_bytes) : (buffer, index, i32, offset) -> (offset, offset, i1)
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{C}xf32>
  %inv_view = buffer.view %inverse[%zero] : buffer -> view<[%tokens]x{2 * HK}xf32>
  %alpha_view = buffer.view %alpha[%zero] : buffer -> view<[%tokens]x{HV}xf32>
  %beta_view = buffer.view %beta[%zero] : buffer -> view<[%tokens]x{HV}xf32>
  %a_view_g = buffer.view %ssm_a[%zero] : buffer -> view<{HV}xf32>
  %bias_view = buffer.view %dt_bias[%zero] : buffer -> view<{HV}xf32>
  %out_view = buffer.view %out[%zero] : buffer -> view<[%tokens]x{VD}xf32>
  %state_view = buffer.view %arena[%reader] : buffer -> view<{NS}xf32>
  %next_view = buffer.view %arena[%writer] : buffer -> view<{NS}xf32>
@CONSTS@
  %a_value = view.load %a_view_g[%vh] : view<{HV}xf32> -> f32
  %dt_bias_value = view.load %bias_view[%vh] : view<{HV}xf32> -> f32
  %qscale_c = scalar.constant {DK ** -0.5!r} : f32
  %zero8 = vector.splatf 0.0 : vector<8xf32>
  %zero16 = vector.splati 0 : vector<16xi32>
  %init = vector.fragment<init> %zero8 {SH} : vector<8xf32>
  {chk_def}{kh_line()}
  %kh_off = index.mul %kh, %k_dk : index
  %kcol_base_v = index.constant {HK * DK} : index
  %kcol_base = index.add %kcol_base_v, %kh_off : index
  %qcol_base = index.add %kh_off, %k0 : index
  %c2i = index.constant 2 : index
  %q_norm_index = index.mul %kh, %c2i : index
  %k_norm_index = index.add %q_norm_index, %k1 : index
  %lane_row = index.rem %lane, %k16 : index
  %lane_half = index.div %lane, %k16 : index
  %lane_h8_v = index.mul %lane_half, %k8 : index
  %lane_h8 = index.assume %lane_h8_v [range(%lane_h8_v, 0, 8)] : index
  %wcol = index.mul %wave, %k16 : index
  %dv_local = index.add %wcol, %lane_row : index
  %dv = index.assume %dv_local [range(%dv_local, 0, {DV - 1})] : index
  %head_cols = index.mul %vh, %k_dv : index
  %ocol_l = index.add %head_cols, %dv : index
  %vbase = index.constant {2 * HK * DK} : index
  %vcol_head = index.add %vbase, %head_cols : index
  // state tile rows (dk = 16 kb + 8 half + i) of value column dv
  %srow_dv = index.add %head_cols, %dv : index
  %sidx0 = index.mul %srow_dv, %k_dk : index
  %sidx_l = index.add %sidx0, %lane_h8 : index
"""
    lds = f"""
  %kh_bytes = index.constant {CN * KHS * 2} : offset
  %kh_buf = buffer.alloca<workgroup> align(16) %kh_bytes : buffer
  %kh_view = buffer.view %kh_buf[%zero] : buffer -> view<{CN}x{KHS}xf16>
  %kh_tl = encoding.layout.strided [1, {KHS}] : encoding<layout>
  %kh_t = buffer.view %kh_buf[%zero] : buffer -> view<{KHS}x{CN}xf16, %kh_tl>
  %q_bytes = index.constant {CN * QAS * 2} : offset
  %q_buf = buffer.alloca<workgroup> align(16) %q_bytes : buffer
  %q_view = buffer.view %q_buf[%zero] : buffer -> view<{CN}x{QAS}xf16>
  %v_bytes = index.constant {CN * VS * 4} : offset
  %v_buf = buffer.alloca<workgroup> align(16) %v_bytes : buffer
  %v_view = buffer.view %v_buf[%zero] : buffer -> view<{CN}x{VS}xf32>
  %a_bytes = index.constant {CN * CN * 4} : offset
  %a_buf = buffer.alloca<workgroup> align(16) %a_bytes : buffer
  %a_view = buffer.view %a_buf[%zero] : buffer -> view<{CN}x{CN}xf32>
  %p_bytes = index.constant {CN * PS * 2} : offset
  %p_buf = buffer.alloca<workgroup> align(16) %p_bytes : buffer
  %p_view = buffer.view %p_buf[%zero] : buffer -> view<{CN}x{PS}xf16>
  %t_bytes = index.constant {CN * TS * 2} : offset
  %t_buf = buffer.alloca<workgroup> align(16) %t_bytes : buffer
  %t_view = buffer.view %t_buf[%zero] : buffer -> view<{CN}x{TS}xf16>
  %kdt_bytes = index.constant {DK * KDS * 2} : offset
  %kdt_buf = buffer.alloca<workgroup> align(16) %kdt_bytes : buffer
  %kdt_view = buffer.view %kdt_buf[%zero] : buffer -> view<{DK}x{KDS}xf16>
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
    init = []
    ys = ", ".join(f"%ist{kb}" for kb in range(8))
    init.append(f"  {ys} = scf.if %fresh -> ({', '.join([vec8] * 8)}) {{")
    init.append(f"    scf.yield {', '.join(['%zero8'] * 8)} : {', '.join([vec8] * 8)}")
    init.append("  } else {")
    for kb in range(8):
        init.append(f"    %sk{kb}_i = index.add %sidx_l, {ci(16 * kb)} : index")
        init.append(f"    %sl{kb} = vector.load %state_view[%sk{kb}_i] : view<{NS}xf32> -> {vec8}")
    init.append(f"    scf.yield {', '.join(f'%sl{kb}' for kb in range(8))} : {', '.join([vec8] * 8)}")
    init.append("  }")
    nch = [f"  %len_r = index.add %length, {ci(CN - 1)} : index", f"  %nchunks = index.div %len_r, {ci(CN)} : index"]
    carry = ", ".join(f"%s{kb} = %ist{kb} : {vec8}" for kb in range(8))
    outs = ", ".join(f"%fin{kb}" for kb in range(8))
    loop_head = f"  {outs} = scf.for %ci = [%k0 to %nchunks step %k1]({carry}) -> ({', '.join([vec8] * 8)}) {{"
    stores = [f"    %so{kb}_i = index.add %sidx_l, {ci(16 * kb)} : index\n    vector.store %fin{kb}, %next_view[%so{kb}_i] : {vec8}, view<{NS}xf32>" for kb in range(8)]
    consts.update({0, 1, 8, 16})
    cdefs = "\n".join(f"  %k{k} = index.constant {k} : index" for k in sorted(consts))
    head = head.replace("@CONSTS@", cdefs + f"\n  %k_dk = index.constant {DK} : index\n  %k_dv = index.constant {DV} : index")
    text = (head + lds + "\n".join(init) + "\n" + "\n".join(nch) + "\n" + loop_head + "\n" + "\n".join(chunk_text) + "\n  }\n"
            + guard_open + "\n".join(stores) + "\n" + guard_close + "\n  }\n  kernel.return\n}\n")
    return text
