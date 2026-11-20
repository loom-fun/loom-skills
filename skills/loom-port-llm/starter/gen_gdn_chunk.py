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


def chunk_kernel(G, NG, LB, guard_open="", guard_close="", target=True, dep=2, arch="rdna4", fuse=False, eps=None, gate=False, cn=None, alias=False, opts=(), abl=()):
    """arch "rdna4": g_gdn_chunk_rdna4 (gfx12 WMMA layouts). arch "rdna3": g_gdn_chunk, the gfx11 form of the same dataflow (RDNA3 WMMA:
    vector<16xf16> operands holding all 16 k in each lane half, f32 results holding rows 2 i + lane / 16 of column lane % 16): the result ->
    rhs conversions (the state copy, R, d) exchange the other row parity with the partner lane (xor 16; f16 pairs packed in i32) and
    interleave; the per-row scalars and rows of a result fragment are gathered with stride 2.
    fuse: the chunk's staging also runs the conv1d + SiLU of its q / k / v channels (inputs before the span from the stored history) and the
    q / k L2 norms (gen_gdn_fused phase A's staging), so %x is the RAW qkv projection and a last argument %weight holds the conv weights;
    g_gdn_conv / g_gdn_qk_norm then only serve the spans under SPAN_MIN, and the history commit runs after this kernel.
    gate: the gated RMSNorm epilogue (g_gdn_norm_gate's out = rmsnorm(o) * w * silu(z)) runs here: each wave's O tile goes to LDS (the V rows
    it alone read), then wave w normalizes rows 4 w .. 4 w + 3 (lane l: value columns 4 l .. 4 l + 3) and stores the gated output to %o;
    arguments %norm_w, %z, %o follow (the raw output %out is not written).
    cn: the chunk length (default CN = 32; 16 halves the staging LDS and the serial solve, gfx11 only, without fuse / gate).
    alias: K^T exp(G - gc) (written in P3, read in P6) lives in Q's LDS rows (Q is dead after P2): -9 KB LDS, so 3 workgroups fit a WGP.
    opts: "kbo" (P2 kb-outer: one state rhs copy live, -56 VGPR peak), latency options: "scan" (the g prefix sum unrolled over vector loads instead of a loop of dependent LDS loads), "acc4" (the
    solve's dot products in 4 partial sums: a quarter of the dependent FMA chain).
    abl: timing-only ablations (wrong results): "noscan", "nosolve", "nostage" (no global loads in the staging)."""
    CN = cn or globals()["CN"]
    NT = CN // 16
    assert CN in (16, 32)
    RPW = CN // 8   # gate epilogue: rows per wave
    R3 = arch == "rdna3"
    FH = "vector<16xf16>" if R3 else "vector<8xf16>"
    from gen_gdn import kh_line, _V_GROUPED
    chk_def = "" if _V_GROUPED[0] else f"%chk = index.constant {G['HK']} : index\n  "   # kh_line's tiled form reads %chk; the grouped form needs none
    HK, HV, DK, DV, C, VD = G["HK"], G["HV"], G["DK"], G["DV"], G["C"], G["VD"]
    assert DK == 128 and DV == 128 and HV % HK == 0, "g_gdn_chunk(_rdna4) is built for 128 x 128 heads (8 waves x 16 value columns)"
    NS = HV * DV * DK
    consts = set()

    def ci(n):
        consts.add(n)
        return f"%k{n}"
    vec8 = "vector<8xf32>"
    SH = "shape [%k16, %k16]"
    KHS, QAS, PS, TS, KDS, VS = DK + 8, DK + 8, CN + 8, CN + 8, CN + 4, DV + 4
    if alias:   # K^T exp(G - gc) [DK][KDS] in Q's [CN][QAS] bytes
        KDS = CN * QAS // DK
        assert KDS >= CN + 2
    out = []
    P = out.append

    def bar():
        P("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")

    def to_rhs(n, src, h4, ind="    "):
        """%n = the rhs fragment (f16) of the f32 result fragment src (rows of the result = k of the rhs); h4: the gfx12 f16 copy's name."""
        if not R3:
            P(f"{ind}%{h4} = vector.fptrunc {src} : {vec8} to vector<8xf16>")
            P(f"{ind}%{n} = vector.fragment<rhs> %{h4} {SH} : vector<8xf16>")
            return
        P(f"{ind}%{n}_h = vector.fptrunc {src} : {vec8} to vector<8xf16>")
        # gfx11: this lane holds rows 2 i + half; the partner lane (xor 16) the other parity. Exchange f16 pairs as i32, interleave.
        P(f"{ind}%{n}_w = vector.bitcast %{n}_h : vector<8xf16> to vector<4xi32>")
        for j in range(4):
            P(f"{ind}%{n}_e{j} = vector.extract %{n}_w[{j}] : vector<4xi32> -> i32")
            P(f"{ind}%{n}_o{j}, %{n}_u{j} = kernel.subgroup.shuffle<xor> %{n}_e{j}, %exchange, %width : i32, i32, i32")
        P(f"{ind}%{n}_pw = vector.from_elements {', '.join(f'%{n}_o{j}' for j in range(4))} : vector<4xi32>")
        P(f"{ind}%{n}_ph = vector.bitcast %{n}_pw : vector<4xi32> to vector<8xf16>")
        P(f"{ind}%{n}_ev = scf.select %hi_half, %{n}_ph, %{n}_h : vector<8xf16>")
        P(f"{ind}%{n}_od = scf.select %hi_half, %{n}_h, %{n}_ph : vector<8xf16>")
        P(f"{ind}%{n}_f = vector.interleave<0> %{n}_ev, %{n}_od : vector<8xf16>, vector<8xf16> -> vector<16xf16>")
        P(f"{ind}%{n} = vector.fragment<rhs> %{n}_f {SH} : vector<16xf16>")

    def rows8(n, view, base, ind="    "):
        """%n = vector<8xf32> of the CN-entry LDS vector `view` at the 8 result rows from `base` (gfx12: base + i; gfx11: base + 2 i)."""
        if not R3:
            P(f"{ind}%{n} = vector.load {view}[{base}] : view<{CN}xf32> -> {vec8}")
            return
        for e in range(8):
            P(f"{ind}%{n}_a{e} = index.add {base}, {ci(2 * e)} : index")
            P(f"{ind}%{n}_{e} = view.load {view}[%{n}_a{e}] : view<{CN}xf32> -> f32")
        P(f"{ind}%{n} = vector.from_elements {', '.join(f'%{n}_{e}' for e in range(8))} : {vec8}")

    RS = 2 if R3 else 1                       # row step of a result fragment's elements
    LH = "%lane_half" if R3 else "%lane_h8"   # this lane's first result row

    # ------------------------------------------------------------------------------------------------------------------
    P(f"    %c0tok = index.mul %ci, {ci(CN)} : index")
    P(f"    %rem = index.sub %length, %c0tok : index")
    P(f"    %m = index.min %rem, {ci(CN)} : index")
    P(f"    %m_last_a = index.sub %m, %k1 : index")
    P(f"    %m_last = index.assume %m_last_a [range(%m_last_a, 0, {CN - 1})] : index")
    P(f"    %vbase_tok = index.add %first, %c0tok : index")
    # ---- P0: stage the chunk (thread -> row thread / 8, 16 columns of K, Q and V)
    p0_start = len(out)
    SPT = 256 // CN          # staging threads per row
    W = DK // SPT            # columns per staging thread
    P(f"    %srow = index.div %thread, {ci(SPT)} : index")
    P(f"    %spart = index.rem %thread, {ci(SPT)} : index")
    P(f"    %scol = index.mul %spart, {ci(W)} : index")
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
    P(f"    %kinv_v = vector.splat %kinv : vector<{W}xf32>")
    P(f"    %qinv_v = vector.splat %qinv : vector<{W}xf32>")
    P(f"    %kcol = index.add %kcol_base, %scol : index")
    P(f"    %qcol = index.add %qcol_base, %scol : index")
    P(f"    %vcol16 = index.add %vcol_head, %scol : index")
    if "nostage" in abl:
        P(f"    %kr = vector.splat %kinv : vector<{W}xf32>")
        P(f"    %qr = vector.splat %qinv : vector<{W}xf32>")
        P(f"    %vr = vector.splat %kinv0 : vector<{W}xf32>")
    else:
        P(f"    %kr = vector.load %x_view[%stabs, %kcol] : view<[%tokens]x{C}xf32> -> vector<{W}xf32>")
        P(f"    %qr = vector.load %x_view[%stabs, %qcol] : view<[%tokens]x{C}xf32> -> vector<{W}xf32>")
        P(f"    %vr = vector.load %x_view[%stabs, %vcol16] : view<[%tokens]x{C}xf32> -> vector<{W}xf32>")
    P(f"    %kn = vector.mulf %kr, %kinv_v : vector<{W}xf32>")
    P(f"    %qn = vector.mulf %qr, %qinv_v : vector<{W}xf32>")
    P(f"    %kh16 = vector.fptrunc %kn : vector<{W}xf32> to vector<{W}xf16>")
    P(f"    %qh16 = vector.fptrunc %qn : vector<{W}xf32> to vector<{W}xf16>")
    P(f"    vector.store %kh16, %kh_view[%srow, %scol] : vector<{W}xf16>, view<{CN}x{KHS}xf16>")
    P(f"    vector.store %qh16, %q_view[%srow, %scol] : vector<{W}xf16>, view<{CN}x{QAS}xf16>")
    P(f"    vector.store %vr, %v_view[%srow, %scol] : vector<{W}xf32>, view<{CN}x{VS}xf32>")
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
    if fuse:
        out[p0_start:] = _fused_stage(G, CN, ci, KHS, QAS, VS)
    # ---- P1: inclusive scan of g (threads < CN)
    P(f"    %tcn = index.cmp ult, %thread, {ci(CN)} : index")
    P(f"    scf.if %tcn {{")
    P(f"      %tt = index.assume %thread [range(%thread, 0, {CN - 1})] : index")
    if "noscan" in abl:
        P(f"      %gc = view.load %g_s[%tt] : view<{CN}xf32> -> f32")
    elif "scan" in opts:   # same order as the loop (adding exact zeros past %tt): every load independent
        acc = "%zero_f"
        for q4 in range(CN // 4):
            P(f"      %gsq{q4} = vector.load %g_s[{ci(4 * q4)}] : view<{CN}xf32> -> vector<4xf32>")
            for e in range(4):
                j = 4 * q4 + e
                P(f"      %gse{j} = vector.extract %gsq{q4}[{e}] : vector<4xf32> -> f32")
                P(f"      %gsl{j} = index.cmp ule, {ci(j)}, %tt : index")
                P(f"      %gsm{j} = scf.select %gsl{j}, %gse{j}, %zero_f : f32")
                P(f"      %gsa{j} = scalar.addf {acc}, %gsm{j} : f32")
                acc = f"%gsa{j}"
        P(f"      %gc = scalar.addf {acc}, %zero_f : f32")
    else:
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
    if "kbo" in opts:   # each state fragment's rhs copy feeds its K S / Q S steps right away (one 8-VGPR copy live, not 64; same sums)
        for kb in range(8):
            to_rhs(f"sf{kb}", f"%s{kb}", f"sh{kb}")
            for nmx, vw, sz in (("zk", "%kh_view", KHS), ("zq", "%q_view", QAS)):
                for mt in range(NT):
                    prev = "%init" if kb == 0 else f"%{nmx}{mt}_{kb - 1}"
                    P(f"    %{nmx}l{mt}_{kb} = vector.fragment.load<lhs> {vw}[{ci(16 * mt)}, {ci(16 * kb)}] {SH} : view<{CN}x{sz}xf16> -> {FH}")
                    P(f"    %{nmx}{mt}_{kb} = vector.mma %{nmx}l{mt}_{kb}, %sf{kb}, {prev} : {FH}, {FH}, {vec8}")
    else:
        for kb in range(8):
            to_rhs(f"sf{kb}", f"%s{kb}", f"sh{kb}")
        for nmx, vw, sz in (("zk", "%kh_view", KHS), ("zq", "%q_view", QAS)):
            for mt in range(NT):
                prev = "%init"
                for kb in range(8):
                    P(f"    %{nmx}l{mt}_{kb} = vector.fragment.load<lhs> {vw}[{ci(16 * mt)}, {ci(16 * kb)}] {SH} : view<{CN}x{sz}xf16> -> {FH}")
                    P(f"    %{nmx}{mt}_{kb} = vector.mma %{nmx}l{mt}_{kb}, %sf{kb}, {prev} : {FH}, {FH}, {vec8}")
                    prev = f"%{nmx}{mt}_{kb}"
    P(f"    %tq = index.rem %wave, {ci(NT * NT)} : index")
    P(f"    %tmt = index.div %tq, {ci(NT)} : index")
    P(f"    %tnt = index.rem %tq, {ci(NT)} : index")
    P(f"    %tr = index.mul %tmt, %k16 : index")
    P(f"    %tc = index.mul %tnt, %k16 : index")
    P(f"    %rb_v = index.add %tr, {LH} : index")
    P(f"    %rb = index.assume %rb_v [range(%rb_v, 0, {CN - 16 + (1 if R3 else 8)})] : index")
    P(f"    %cj_v = index.add %tc, %lane_row : index")
    P(f"    %cj = index.assume %cj_v [range(%cj_v, 0, {CN - 1})] : index")
    P(f"    %rb_i = index.cast %rb : index to i32")
    P(f"    %cj_i = index.cast %cj : index to i32")
    P(f"    %rows = vector.iota %rb_i step {'%two_i32' if R3 else '%one_i32'} : vector<8xi32>")
    P(f"    %colv = vector.splat %cj_i : vector<8xi32>")
    rows8("gci", "%gc_s", "%rb")
    P(f"    %gcj_s = view.load %gc_s[%cj] : view<{CN}xf32> -> f32")
    P(f"    %gcj = vector.splat %gcj_s : {vec8}")
    P(f"    %dg = vector.subf %gci, %gcj : {vec8}")
    P(f"    %dec = vector.expf<afn> %dg : {vec8}")
    P(f"    %is_kk = index.cmp ult, %wave, {ci(4)} : index")
    P(f"    scf.if %is_kk {{")
    prev = "%init"
    for kb in range(8):
        P(f"      %kkl{kb} = vector.fragment.load<lhs> %kh_view[%tr, {ci(16 * kb)}] {SH} : view<{CN}x{KHS}xf16> -> {FH}")
        P(f"      %kkr{kb} = vector.fragment.load<rhs> %kh_t[{ci(16 * kb)}, %tc] {SH} : view<{KHS}x{CN}xf16, %kh_tl> -> {FH}")
        P(f"      %kk{kb} = vector.mma %kkl{kb}, %kkr{kb}, {prev} : {FH}, {FH}, {vec8}")
        prev = f"%kk{kb}"
    rows8("bti", "%beta_s", "%rb", "      ")
    P(f"      %lts = vector.cmpi sgt, %rows, %colv : vector<8xi32> -> vector<8xi1>")
    P(f"      %kkd = vector.mulf {prev}, %dec : {vec8}")
    P(f"      %kkb = vector.mulf %kkd, %bti : {vec8}")
    P(f"      %kkn = vector.subf %zero8, %kkb : {vec8}")
    P(f"      %av = vector.select %lts, %kkn, %zero8 : {vec8}")
    P(f"      vector.fragment.store<result> %av, %a_view[%tr, %tc] {SH} : {vec8}, view<{CN}x{CN}xf32>")
    P(f"    }} else {{")
    prev = "%init"
    for kb in range(8):
        P(f"      %qkl{kb} = vector.fragment.load<lhs> %q_view[%tr, {ci(16 * kb)}] {SH} : view<{CN}x{QAS}xf16> -> {FH}")
        P(f"      %qkr{kb} = vector.fragment.load<rhs> %kh_t[{ci(16 * kb)}, %tc] {SH} : view<{KHS}x{CN}xf16, %kh_tl> -> {FH}")
        P(f"      %qk{kb} = vector.mma %qkl{kb}, %qkr{kb}, {prev} : {FH}, {FH}, {vec8}")
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
        nfull = 0 if "nosolve" in abl else i // 4
        tail = [] if "nosolve" in abl else list(range(4 * nfull, i))
        # "acc4": four partial sums (column m into sum m % 4), then (s0 + s1) + (s2 + s3): a quarter of the dependent chain
        parts = [acc, "%zero_f", "%zero_f", "%zero_f"] if "acc4" in opts and i >= 8 else None
        for g in range(nfull):
            P(f"      %ar{i}_{g} = vector.load %a_view[{row}, {ci(4 * g)}] : view<{CN}x{CN}xf32> -> vector<4xf32>")
            for e in range(4):
                m_ = 4 * g + e
                P(f"      %ae{i}_{m_} = vector.extract %ar{i}_{g}[{e}] : vector<4xf32> -> f32")
                if parts:
                    P(f"      %tf{i}_{m_} = scalar.fmaf %ae{i}_{m_}, %T{m_}, {parts[e]} : f32")
                    parts[e] = f"%tf{i}_{m_}"
                    continue
                P(f"      %tf{i}_{m_} = scalar.fmaf %ae{i}_{m_}, %T{m_}, {acc} : f32")
                acc = f"%tf{i}_{m_}"
        for m_ in tail:
            P(f"      %ae{i}_{m_} = view.load %a_view[{row}, {ci(m_)}] : view<{CN}x{CN}xf32> -> f32")
            if parts:
                P(f"      %tf{i}_{m_} = scalar.fmaf %ae{i}_{m_}, %T{m_}, {parts[m_ % 4]} : f32")
                parts[m_ % 4] = f"%tf{i}_{m_}"
                continue
            P(f"      %tf{i}_{m_} = scalar.fmaf %ae{i}_{m_}, %T{m_}, {acc} : f32")
            acc = f"%tf{i}_{m_}"
        if parts:
            P(f"      %ts{i}_a = scalar.addf {parts[0]}, {parts[1]} : f32")
            P(f"      %ts{i}_b = scalar.addf {parts[2]}, {parts[3]} : f32")
            P(f"      %ts{i}_c = scalar.addf %ts{i}_a, %ts{i}_b : f32")
            acc = f"%ts{i}_c"
        P(f"      %T{i} = scalar.addf {acc}, %zero_f : f32")
        P(f"      %T{i}h = scalar.fptrunc %T{i} : f32 to f16")
        P(f"      view.store %T{i}h, %t_view[{ci(i)}, %cc] : f16, view<{CN}x{TS}xf16>")
    P(f"    }} else {{")
    P(f"      %ktid = index.sub %thread, {ci(CN)} : index")
    P(f"      %kstage = index.cmp ult, %ktid, {ci(DK)} : index")
    P(f"      scf.if %kstage {{")
    P(f"        %kdk = index.assume %ktid [range(%ktid, 0, {DK - 1})] : index")
    P(f"        %gl = view.load %gc_s[{ci(CN - 1)}] : view<{CN}xf32> -> f32")
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
        P(f"    %ri{mt}_v = index.add {ci(16 * mt)}, {LH} : index")
        P(f"    %ri{mt} = index.assume %ri{mt}_v [range(%ri{mt}_v, 0, {CN - 16 + (1 if R3 else 8)})] : index")
        rows8(f"bti_{mt}", "%beta_s", f"%ri{mt}")
        rows8(f"egi_{mt}", "%eg_s", f"%ri{mt}")
        els = []
        for e in range(8):
            P(f"    %vt{mt}_{e}_o = index.add %ri{mt}, {ci(RS * e)} : index")
            P(f"    %vv{mt}_{e} = view.load %v_view[%vt{mt}_{e}_o, %dv] : view<{CN}x{VS}xf32> -> f32")
            els.append(f"%vv{mt}_{e}")
        P(f"    %vvec{mt} = vector.from_elements {', '.join(els)} : {vec8}")
        P(f"    %ezk{mt} = vector.mulf %egi_{mt}, %zk{mt}_7 : {vec8}")
        P(f"    %vmz{mt} = vector.subf %vvec{mt}, %ezk{mt} : {vec8}")
        P(f"    %rr{mt} = vector.mulf %bti_{mt}, %vmz{mt} : {vec8}")
        to_rhs(f"rf{mt}", f"%rr{mt}", f"rh{mt}")
    for mt in range(NT):
        prev = "%init"
        for kt in range(mt + 1):
            P(f"    %tl{mt}_{kt} = vector.fragment.load<lhs> %t_view[{ci(16 * mt)}, {ci(16 * kt)}] {SH} : view<{CN}x{TS}xf16> -> {FH}")
            P(f"    %dd{mt}_{kt} = vector.mma %tl{mt}_{kt}, %rf{kt}, {prev} : {FH}, {FH}, {vec8}")
            prev = f"%dd{mt}_{kt}"
        to_rhs(f"df{mt}", prev, f"dh{mt}")
    # ---- P5: O = exp(gc) Q S + P d
    for mt in range(NT):
        P(f"    %oi{mt} = vector.mulf %egi_{mt}, %zq{mt}_7 : {vec8}")
        prev = f"%oi{mt}"
        for kt in range(mt + 1):
            P(f"    %pl{mt}_{kt} = vector.fragment.load<lhs> %p_view[{ci(16 * mt)}, {ci(16 * kt)}] {SH} : view<{CN}x{PS}xf16> -> {FH}")
            P(f"    %oo{mt}_{kt} = vector.mma %pl{mt}_{kt}, %df{kt}, {prev} : {FH}, {FH}, {vec8}")
            prev = f"%oo{mt}_{kt}"
        if gate:
            P(f"    vector.fragment.store<result> {prev}, %v_view[{ci(16 * mt)}, %wcol] {SH} : {vec8}, view<{CN}x{VS}xf32>")
            continue
        for e in range(8):
            P(f"    %olt{mt}_{e}_r = index.add {ci(16 * mt + RS * e)}, {LH} : index")
            P(f"    %olt{mt}_{e} = index.cmp ult, %olt{mt}_{e}_r, %m : index")
            P(f"    scf.if %olt{mt}_{e} {{")
            P(f"      %ov{mt}_{e} = vector.extract {prev}[{e}] : {vec8} -> f32")
            P(f"      %orw{mt}_{e}_a = index.add %vbase_tok, %olt{mt}_{e}_r : index")
            P(f"      %orw{mt}_{e} = index.assume %orw{mt}_{e}_a [lt(%orw{mt}_{e}_a, %tokens)] : index")
            P(f"      view.store %ov{mt}_{e}, %out_view[%orw{mt}_{e}, %ocol_l] : f32, view<[%tokens]x{VD}xf32>")
            P(f"    }}")
    if gate:   # gated RMSNorm of the chunk's O rows (fb's epilogue): rows 4 w + rr, columns 4 l .. 4 l + 3 of this head
        bar()
        for rr in range(RPW):
            P(f"    %nr{rr}_a = index.mul %wave, {ci(RPW)} : index")
            P(f"    %nr{rr}_b = index.add %nr{rr}_a, {ci(rr)} : index")
            P(f"    %nr{rr} = index.assume %nr{rr}_b [range(%nr{rr}_b, 0, {CN - 1})] : index")
            P(f"    %olive{rr} = index.cmp ult, %nr{rr}, %m : index")
            P(f"    %otok{rr}_a = index.add %vbase_tok, %nr{rr} : index")
            P(f"    %otok{rr}_c = scf.select %olive{rr}, %otok{rr}_a, %vbase_tok : index")
            P(f"    %otok{rr} = index.assume %otok{rr}_c [lt(%otok{rr}_c, %tokens)] : index")
            P(f"    %zv{rr} = vector.load %z_view[%otok{rr}, %zcol] : view<[%tokens]x{VD}xf32> -> vector<4xf32>")
            P(f"    %gov{rr} = vector.load %v_view[%nr{rr}, %lc4] : view<{CN}x{VS}xf32> -> vector<4xf32>")
            P(f"    %gosq{rr} = vector.mulf %gov{rr}, %gov{rr} : vector<4xf32>")
            P(f"    %gosr{rr} = vector.reduce<addf> %gosq{rr}, %zero_f : vector<4xf32>, f32")
            P(f"    %gosum{rr} = kernel.subgroup.reduce<addf> %gosr{rr} : f32")
            P(f"    %gomean{rr} = scalar.divf %gosum{rr}, %gcount : f32")
            P(f"    %gost{rr} = scalar.addf %gomean{rr}, %eps : f32")
            P(f"    %goinv{rr} = scalar.rsqrtf %gost{rr} : f32")
            P(f"    %goiv{rr} = vector.splat %goinv{rr} : vector<4xf32>")
            P(f"    %gzs{rr} = vector.siluf<afn> %zv{rr} : vector<4xf32>")
            P(f"    %gon{rr}_a = vector.mulf %gov{rr}, %goiv{rr} : vector<4xf32>")
            P(f"    %gon{rr}_b = vector.mulf %gon{rr}_a, %nwv : vector<4xf32>")
            P(f"    %gon{rr} = vector.mulf %gon{rr}_b, %gzs{rr} : vector<4xf32>")
        for rr in range(RPW):
            P(f"    scf.if %olive{rr} {{")
            P(f"      vector.store %gon{rr}, %o_view[%otok{rr}, %zcol] : vector<4xf32>, view<[%tokens]x{VD}xf32>")
            P("    }")
    # ---- P6: S <- exp(G) S + (K exp(G - gc))^T d
    P(f"    %gl2 = view.load %gc_s[{ci(CN - 1)}] : view<{CN}xf32> -> f32")
    P(f"    %eg_last = scalar.expf<afn> %gl2 : f32")
    P(f"    %eg_v = vector.splat %eg_last : {vec8}")
    for kb in range(8):
        P(f"    %sdv{kb} = vector.mulf %s{kb}, %eg_v : {vec8}")
        P(f"    %sd{kb} = vector.fragment<init> %sdv{kb} {SH} : {vec8}")
        prev = f"%sd{kb}"
        for kt in range(NT):
            P(f"    %ul{kb}_{kt} = vector.fragment.load<lhs> %kdt_view[{ci(16 * kb)}, {ci(16 * kt)}] {SH} : view<{DK}x{KDS}xf16> -> {FH}")
            P(f"    %un{kb}_{kt} = vector.mma %ul{kb}_{kt}, %df{kt}, {prev} : {FH}, {FH}, {vec8}")
            prev = f"%un{kb}_{kt}"
        P(f"    %sn{kb} = vector.addf {prev}, %zero8 : {vec8}")
    bar()
    P(f"    scf.yield {', '.join(f'%sn{kb}' for kb in range(8))} : {', '.join([vec8] * 8)}")
    chunk_text = out
    # ------------------------------------------------------------------------------------------------------------------
    tgt = "target(@rdna4) " if target and not R3 else ""
    FUSE_HEAD = (f"  %history_skip = index.constant {G['state_bytes']} : offset\n  %history_at = index.add %reader, %history_skip : offset\n"
                 f"  %hist_view = buffer.view %arena[%history_at] : buffer -> view<{G['K'] - 1}x{C}xf32>\n"
                 f"  %w_view = buffer.view %weight[%zero] : buffer -> view<{C}x{G['K']}xf32>\n  %eps = scalar.constant {eps!r} : f32\n") if fuse else ""
    GATE_HEAD = (f"  %z_view = buffer.view %z[%zero] : buffer -> view<[%tokens]x{VD}xf32>\n  %o_view = buffer.view %o[%zero] : buffer -> view<[%tokens]x{VD}xf32>\n"
                 f"  %nw_view = buffer.view %norm_w[%zero] : buffer -> view<{DV}xf32>\n  %gcount = scalar.constant {float(DV)!r} : f32\n"
                 + ("" if fuse else f"  %eps = scalar.constant {eps!r} : f32\n")) if gate else ""
    GATE_LANE = ("  %lc4_c = index.constant 4 : index\n  %lc4_v = index.mul %lane, %lc4_c : index\n"
                 f"  %lc4 = index.assume %lc4_v [range(%lc4_v, 0, {DV - 4})] : index\n  %zcol = index.add %head_cols, %lc4 : index\n"
                 f"  %nwv = vector.load %nw_view[%lc4] : view<{DV}xf32> -> vector<4xf32>\n") if gate else ""
    R3_CONSTS = ("  %hi_half = index.cmp ne, %lane_half, %k0 : index\n  %exchange = scalar.constant 16 : i32\n  %width = scalar.constant 32 : i32\n"
                 "  %two_i32 = scalar.constant 2 : i32\n") if R3 else ""
    kname = "g_gdn_chunk" if R3 else "g_gdn_chunk_rdna4"
    head = f"""
kernel.def {tgt}export("{kname}") @{kname}(%spans: index) {{
  %c1 = index.constant 1 : index
  %heads = index.constant {HV} : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%heads, %c1, %spans) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %layer_arg: index, %metadata: buffer, %origins: buffer, %arena: buffer, %x: buffer, %inverse: buffer, %alpha: buffer, %beta: buffer, %ssm_a: buffer, %dt_bias: buffer, %out: buffer{", %weight: buffer" if fuse else ""}{", %norm_w: buffer, %z: buffer, %o: buffer" if gate else ""}) {{
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
{FUSE_HEAD}{GATE_HEAD}@CONSTS@
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
{GATE_LANE}  %vcol_head = index.add %vbase, %head_cols : index
  // state tile rows (dk = 16 kb + 8 half + i) of value column dv
  %srow_dv = index.add %head_cols, %dv : index
  %sidx0 = index.mul %srow_dv, %k_dk : index
  %sidx_l = index.add %sidx0, {LH} : index
{R3_CONSTS}"""
    KDT_LDS = (f"  %kdt_view = buffer.view %q_buf[%zero] : buffer -> view<{DK}x{KDS}xf16>" if alias else
               f"  %kdt_bytes = index.constant {DK * KDS * 2} : offset\n  %kdt_buf = buffer.alloca<workgroup> align(16) %kdt_bytes : buffer\n"
               f"  %kdt_view = buffer.view %kdt_buf[%zero] : buffer -> view<{DK}x{KDS}xf16>")
    RED = (f"  %red_bytes = index.constant {CN * 8 * 4} : offset\n  %red_buf = buffer.alloca<workgroup> align(16) %red_bytes : buffer\n"
           f"  %red_view = buffer.view %red_buf[%zero] : buffer -> view<{CN}x8xf32>\n") if fuse else ""
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
{KDT_LDS}
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
{RED}  %thread_is0 = index.cmp eq, %thread, %k0 : index
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
        if R3:   # gfx11 result rows 16 kb + 2 i + half: strided (once per span)
            for e in range(8):
                init.append(f"    %sk{kb}_{e} = index.add %sidx_l, {ci(16 * kb + 2 * e)} : index")
                init.append(f"    %sl{kb}_{e} = view.load %state_view[%sk{kb}_{e}] : view<{NS}xf32> -> f32")
            init.append(f"    %sl{kb} = vector.from_elements {', '.join(f'%sl{kb}_{e}' for e in range(8))} : {vec8}")
            continue
        init.append(f"    %sk{kb}_i = index.add %sidx_l, {ci(16 * kb)} : index")
        init.append(f"    %sl{kb} = vector.load %state_view[%sk{kb}_i] : view<{NS}xf32> -> {vec8}")
    init.append(f"    scf.yield {', '.join(f'%sl{kb}' for kb in range(8))} : {', '.join([vec8] * 8)}")
    init.append("  }")
    nch = [f"  %len_r = index.add %length, {ci(CN - 1)} : index", f"  %nchunks = index.div %len_r, {ci(CN)} : index"]
    carry = ", ".join(f"%s{kb} = %ist{kb} : {vec8}" for kb in range(8))
    outs = ", ".join(f"%fin{kb}" for kb in range(8))
    loop_head = f"  {outs} = scf.for %ci = [%k0 to %nchunks step %k1]({carry}) -> ({', '.join([vec8] * 8)}) {{"
    if R3:
        stores = [f"    %so{kb}_{e} = index.add %sidx_l, {ci(16 * kb + 2 * e)} : index\n    %sov{kb}_{e} = vector.extract %fin{kb}[{e}] : {vec8} -> f32\n"
                  f"    view.store %sov{kb}_{e}, %next_view[%so{kb}_{e}] : f32, view<{NS}xf32>" for kb in range(8) for e in range(8)]
    else:
        stores = [f"    %so{kb}_i = index.add %sidx_l, {ci(16 * kb)} : index\n    vector.store %fin{kb}, %next_view[%so{kb}_i] : {vec8}, view<{NS}xf32>" for kb in range(8)]
    consts.update({0, 1, 8, 16})
    cdefs = "\n".join(f"  %k{k} = index.constant {k} : index" for k in sorted(consts))
    head = head.replace("@CONSTS@", cdefs + f"\n  %k_dk = index.constant {DK} : index\n  %k_dv = index.constant {DV} : index")
    text = (head + lds + "\n".join(init) + "\n" + "\n".join(nch) + "\n" + loop_head + "\n" + "\n".join(chunk_text) + "\n  }\n"
            + guard_open + "\n".join(stores) + "\n" + guard_close + "\n  }\n  kernel.return\n}\n")
    return text


def _fused_stage(G, CN, ci, KHS, QAS, VS):
    """Loop-body lines of the fused staging (chunk_kernel fuse=True): g / beta per token (threads < CN); conv1d + SiLU of the chunk's rows of
    q / k channel `thread % 128` of the key head (threads < 128: q, else k) and of value channel `thread` (threads < DV); the q / k L2 norms
    (a subgroup sum per row, the 4 waves' partials through LDS); K, Q (x DK^-0.5) as f16 rows and V as f32 rows in LDS. Rows past the
    span's end are zero. Same arithmetic as g_gdn_conv + g_gdn_qk_norm (the conv taps in the same order)."""
    HV, DV, C, K = G["HV"], G["DV"], G["C"], G["K"]
    out = []
    P = out.append
    P(f"    %ftcn = index.cmp ult, %thread, {ci(CN)} : index")
    P("    scf.if %ftcn {")
    P(f"      %sr = index.assume %thread [range(%thread, 0, {CN - 1})] : index")
    P("      %slive = index.cmp ult, %sr, %m : index")
    P("      %stoff = index.add %vbase_tok, %sr : index")
    P("      %stokc = scf.select %slive, %stoff, %vbase_tok : index")
    P("      %stok = index.assume %stokc [lt(%stokc, %tokens)] : index")
    P(f"      %al = view.load %alpha_view[%stok, %vh] : view<[%tokens]x{HV}xf32> -> f32")
    P(f"      %be = view.load %beta_view[%stok, %vh] : view<[%tokens]x{HV}xf32> -> f32")
    P("      %al_b = scalar.addf %al, %dt_bias_value : f32")
    P("      %al_p = scalar.softplusf<afn> %al_b : f32")
    P("      %gate = scalar.mulf %al_p, %a_value : f32")
    P("      %g_live = scf.select %slive, %gate, %zero_f : f32")
    P("      %bt = scalar.logisticf<afn> %be : f32")
    P("      %bt_live = scf.select %slive, %bt, %zero_f : f32")
    P(f"      view.store %g_live, %g_s[%sr] : f32, view<{CN}xf32>")
    P(f"      view.store %bt_live, %beta_s[%sr] : f32, view<{CN}xf32>")
    P("    }")
    for r in range(CN):
        P(f"    %frl{r} = index.cmp ult, {ci(r)}, %m : index")
        P(f"    %ftr{r}_a = index.add %vbase_tok, {ci(r)} : index")
        P(f"    %ftr{r}_c = scf.select %frl{r}, %ftr{r}_a, %vbase_tok : index")
        P(f"    %ftrow{r} = index.assume %ftr{r}_c [lt(%ftr{r}_c, %tokens)] : index")
    P("    %ffirst = index.cmp eq, %ci, %k0 : index")

    def conv_channel(tag, chv, ind):
        def Q(t):
            P(ind + t)
        for j in range(K):
            Q(f"%fw{tag}{j} = view.load %w_view[{chv}, {ci(j)}] : view<{C}x{K}xf32> -> f32")
        for r in range(-(K - 1), 0):
            u = f"{tag}{-r}"
            Q(f"%fx{tag}m{-r} = scf.if %ffirst -> (f32) {{")
            Q(f"  %fhz{u} = scf.if %fresh -> (f32) {{")
            Q("    scf.yield %zero_f : f32")
            Q("  } else {")
            Q(f"    %fhv{u} = view.load %hist_view[{ci(K - 1 + r)}, {chv}] : view<{K - 1}x{C}xf32> -> f32")
            Q(f"    scf.yield %fhv{u} : f32")
            Q("  }")
            Q(f"  scf.yield %fhz{u} : f32")
            Q("} else {")
            Q(f"  %fpr{u} = index.sub %vbase_tok, {ci(-r)} : index")
            Q(f"  %fprb{u} = index.assume %fpr{u} [lt(%fpr{u}, %tokens)] : index")
            Q(f"  %fpv{u} = view.load %x_view[%fprb{u}, {chv}] : view<[%tokens]x{C}xf32> -> f32")
            Q(f"  scf.yield %fpv{u} : f32")
            Q("}")
        for r in range(CN):
            Q(f"%fx{tag}{r}_r = view.load %x_view[%ftrow{r}, {chv}] : view<[%tokens]x{C}xf32> -> f32")

        def xname(r):
            return f"%fx{tag}m{-r}" if r < 0 else f"%fx{tag}{r}_r"
        for r in range(CN):
            acc = "%zero_f"
            for j in range(K):
                Q(f"%fc{tag}{r}_{j} = scalar.fmaf %fw{tag}{j}, {xname(r - (K - 1) + j)}, {acc} : f32")
                acc = f"%fc{tag}{r}_{j}"
            Q(f"%fs{tag}{r}_g = scalar.logisticf<afn> {acc} : f32")
            Q(f"%fs{tag}{r}_m = scalar.mulf {acc}, %fs{tag}{r}_g : f32")
            Q(f"%fs{tag}{r} = scf.select %frl{r}, %fs{tag}{r}_m, %zero_f : f32")
    P(f"    %fisk = index.cmp uge, %thread, {ci(128)} : index")
    P(f"    %ftq = index.rem %thread, {ci(128)} : index")
    P("    %fqk_base = scf.select %fisk, %kcol_base, %qcol_base : index")
    P("    %fchqk_v = index.add %fqk_base, %ftq : index")
    P(f"    %fchqk = index.assume %fchqk_v [range(%fchqk_v, 0, {C - 1})] : index")
    conv_channel("a", "%fchqk", "    ")
    for r in range(CN):
        P(f"    %fsq{r} = scalar.mulf %fsa{r}, %fsa{r} : f32")
        P(f"    %fss{r} = kernel.subgroup.reduce<addf> %fsq{r} : f32")
    P("    %flead = index.cmp eq, %lane, %k0 : index")
    P("    scf.if %flead {")
    for r in range(CN):
        P(f"      view.store %fss{r}, %red_view[{ci(r)}, %wave] : f32, view<{CN}x8xf32>")
    P("    }")
    P(f"    %fisv = index.cmp ult, %thread, {ci(DV)} : index")
    P("    scf.if %fisv {")
    P(f"      %fvt = index.assume %thread [range(%thread, 0, {DV - 1})] : index")
    P("      %fchv_v = index.add %vcol_head, %fvt : index")
    P(f"      %fchv = index.assume %fchv_v [range(%fchv_v, 0, {C - 1})] : index")
    conv_channel("b", "%fchv", "      ")
    for r in range(CN):
        P(f"      view.store %fsb{r}, %v_view[{ci(r)}, %fvt] : f32, view<{CN}x{VS}xf32>")
    P("    }")
    P("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    P(f"    %fwq = index.cmp uge, %wave, {ci(4)} : index")
    P(f"    %fwb = scf.select %fwq, {ci(4)}, %k0 : index")
    for r in range(CN):
        P(f"    %frd{r}_0 = view.load %red_view[{ci(r)}, %fwb] : view<{CN}x8xf32> -> f32")
        acc = f"%frd{r}_0"
        for w in range(1, 4):
            P(f"    %frw{r}_{w} = index.add %fwb, {ci(w)} : index")
            P(f"    %frd{r}_{w} = view.load %red_view[{ci(r)}, %frw{r}_{w}] : view<{CN}x8xf32> -> f32")
            P(f"    %fra{r}_{w} = scalar.addf {acc}, %frd{r}_{w} : f32")
            acc = f"%fra{r}_{w}"
        P(f"    %frs{r} = scalar.addf {acc}, %eps : f32")
        P(f"    %fri{r} = scalar.rsqrtf %frs{r} : f32")
        P(f"    %frq{r} = scalar.mulf %fri{r}, %qscale_c : f32")
        P(f"    %frf{r} = scf.select %fisk, %fri{r}, %frq{r} : f32")
        P(f"    %fn{r} = scalar.mulf %fsa{r}, %frf{r} : f32")
        P(f"    %fnh{r} = scalar.fptrunc %fn{r} : f32 to f16")
    P("    scf.if %fisk {")
    for r in range(CN):
        P(f"      view.store %fnh{r}, %kh_view[{ci(r)}, %ftq] : f16, view<{CN}x{KHS}xf16>")
    P("    } else {")
    for r in range(CN):
        P(f"      view.store %fnh{r}, %q_view[{ci(r)}, %ftq] : f16, view<{CN}x{QAS}xf16>")
    P("    }")
    P("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    return out
