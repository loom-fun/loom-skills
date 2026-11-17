"""Key-split flash loop with two query heads per wave for the gfx12 head_dim 512 twin (gen_flash2s.py, FLASH2_B=2; llm_config rdna4_f16_ffn sets it).

The staged twin lets the four waves of a query head compute every 16-key S tile together (each over its 128 dims, an f32 partial exchange,
four redundant softmaxes) and every wave streams its own K / V^T fragments for ONE head. This loop reorganises the work the way llama.cpp's
Vulkan flash_attn_cm1 does and shares the K / V fragments between two heads:
  * a block is 64 keys; wave j of a workgroup computes the S^T tile of keys 16 j .. 16 j + 15 over ALL 512 dims (32 WMMAs per head; the K
    fragments straight from the paged KV arena as four 128-byte loads per lane, the Q^T fragments from workgroup memory where the four
    waves staged them once) for BOTH heads of the workgroup's head pair, so a K fragment feeds two WMMAs;
  * ONE softmax per S tile: the column maxima of the four tiles of a head are exchanged (one barrier), every wave derives the same running
    max, exponentiates its own tile and publishes P^T (f16, the rhs fragment as it is) and its row sum;
  * PV: every wave reads the four P^T tiles of both heads (second barrier) and accumulates O^T of ITS 128 dims over all 64 keys; a V^T
    fragment (global_load_tr_b128 from the arena) feeds both heads' WMMAs.
Per 64 keys and head pair: two barriers, 128 WMMAs per wave. A workgroup is the four waves of a head pair (128 lanes); the exchange buffers
are single (a wave cannot reach the next block's store before every wave has finished reading this block's), so the workgroup needs 40 KB of
workgroup memory and three of them share a WGP. Keys past the tile's last written page re-read the last page (finite, fully masked, weight 0);
the partial kernel zeroes V^T for positions past its last written key. The first head's output epilogue is repeated for the second head.
Measured on E2B (R9700, vs the staged twin): prefill +7 / +12 / +19 % at 8k / 16k / 32k.
"""
import re


def rewrite_b2(text, kvh, d, partial, qh=None, nh=2, opts=None):
    """d = 512 (gemma4 global layers: 8 query heads per KV head) or 256 (Qwen3.5-class 27B: 6 per KV head, three head pairs per KV head).
    A wave owns d / 4 output dims (HALF of gen_flash2s.py: 128 / 64) and sums its 16 keys' S^T over all d dims."""
    assert d in (256, 512) and nh in (2, 3)
    waves = 4
    SUF = ["", "B", "C"][:nh]                                   # per head: names of its running max / sum / S tile / epilogue ("" = the first head)
    AP = ["a", "b", "cc"][:nh]                                  # per head: its accumulator names
    OP = ["o", "oB", "oC"][:nh]                                 # per head: its loop results
    HV = ["%head", "%head1", "%head2"][:nh]
    qh = qh or kvh * 8
    R = kvh * d
    NF = d // 16
    NO = d // 64                   # output fragments of a wave per head (8 / 4)
    NKW = NF // 8                  # 64-dim K loads per lane for one 16-key S tile (4 / 2)
    FPW = NF // waves              # Q^T fragments each wave stages (8 / 4)
    opts = opts or {}
    QP = NF * 8 + 8 - 8 * int(opts.get("qreg", 0))
    pos_bytes = int(re.search(r"%pos_bytes = index\.constant (\d+) : offset", text).group(1))
    marker = "  // Partial tile: O^T / l" if partial else "  // O^T / l, gate, store through"
    text = re.sub(r"^ *%k_bytes = .*\n *%k_tile = .*\n *%k_view = .*\n", "", text, flags=re.M)
    text = re.sub(r"^ *%vt_bytes = .*\n *%vt_tile = .*\n *%vt_view = .*\n", "", text, flags=re.M)
    text = re.sub(r"^ *%s_bytes = .*\n *%s_tile = .*\n *%s_view = .*\n", "", text, flags=re.M)
    h0 = text.index("  %m_final, %l_final,")
    h1 = text.index(marker)
    epi = text[h1:]
    body_end = epi.index("  }\n  kernel.return")        # the epilogue ends at the scf.if's closing brace
    epilogue, tail = epi[:body_end], epi[body_end:]
    # --- the other heads' epilogues: every name the epilogue defines gets the head's suffix; the loop results and the head index are renamed
    defined = set(re.findall(r"^\s*(%\w+)(?:, %\w+)* =", epilogue, flags=re.M))
    defined |= set(m for line in re.findall(r"^\s*((?:%\w+, )+%\w+) =", epilogue, flags=re.M) for m in line.split(", "))
    epis = []
    for hh in range(1, nh):
        epi_b = epilogue
        for nm in sorted(defined, key=len, reverse=True):
            epi_b = re.sub(re.escape(nm) + r"\b(?!\w)", nm + SUF[hh], epi_b)
        for k in range(NO):
            epi_b = re.sub(rf"%o{k}\b(?!\w)", f"%{OP[hh]}{k}", epi_b)
        epi_b = re.sub(r"%l_final\b(?!\w)", f"%l{SUF[hh]}_final", epi_b)
        epi_b = re.sub(r"(?<![\w%])%head\b(?!\w)", HV[hh], epi_b)
        epi_b = re.sub(r"%head\b(?!\w)", HV[hh], epi_b)
        epis.append(epi_b)
    # Q^T staging of both heads
    cfpw = "%fb_c8" if FPW == 8 else "%fb_cfpw"   # d 512 keeps its original text (the gemma4 program stays byte-identical)
    qst = ("  %fb_c8 = index.constant 8 : index\n  %fb_c64 = index.constant " + str(FPW * 8) + " : index\n  %fb_c256 = index.constant " + str(d // 2) + " : index\n"
           + ("" if FPW == 8 else "  %fb_cfpw = index.constant " + str(FPW) + " : index\n") +
           "  %fb_dh8 = index.mul %dim_half, " + cfpw + " : index\n  %fb_dh64 = index.mul %dim_half, %fb_c64 : index\n"
           "  %fb_h256 = index.mul %lane_half, %fb_c256 : index\n  %fb_qb = index.add %fb_h256, %fb_dh64 : index\n"
           + "".join(f"  %head{hh}_v = index.add %head, %c{hh} : index\n  %head{hh} = index.assume %head{hh}_v [range(%head{hh}_v, {hh}, {qh - 1})] : index\n" for hh in range(1, nh)))
    if opts.get("qreg"):   # the wave(s) owning row elements 0 .. 8 NQR - 1 stage nothing: every wave loads those fragments into registers
        qst += f"  %fb_nz = index.cmp ne, %dim_half, %c0 : index\n  %fb_cnqr = index.constant {8 * int(opts['qreg'])} : index\n  scf.if %fb_nz {{\n"
    for hh, hn in enumerate(HV):
        for c in range(FPW):
            qst += (f"  %fqs{hh}_{c}_v = index.add %fb_dh8, %e{c} : index\n  %fqs{hh}_{c} = index.assume %fqs{hh}_{c}_v [range(%fqs{hh}_{c}_v, 0, {NF - 1})] : index\n"
                    f"  %fqe{hh}_{c}_v = index.mul %fqs{hh}_{c}, %fb_c8 : index\n  %fqe{hh}_{c} = index.assume %fqe{hh}_{c}_v [range(%fqe{hh}_{c}_v, 0, {QP - 8})] : index\n"
                    f"  %fqn{hh}_{c}_o = index.constant {8 * c} : index\n  %fqn{hh}_{c}_v = index.add %fb_qb, %fqn{hh}_{c}_o : index\n"
                    f"  %fqn{hh}_{c} = index.assume %fqn{hh}_{c}_v [range(%fqn{hh}_{c}_v, 0, {d - 8})] : index\n"
                    f"  %fqraw{hh}_{c} = vector.load %q_view[%q_token, {hn}, %fqn{hh}_{c}] : view<[%tokens]x{qh}x{d}xf32> -> vector<8xf32>\n"
                    f"  %fqh{hh}_{c} = vector.fptrunc %fqraw{hh}_{c} : vector<8xf32> to vector<8xf16>\n"
                    f"  vector.store %fqh{hh}_{c}, %ql_view[%c{hh}_i, %lane, %fqe{hh}_{c}] : vector<8xf16>, view<{nh}x32x{QP}xf16>\n")
    if opts.get("qreg"):
        qst = re.sub(r"(%fqe\w+)_v = index\.mul (%fqs\w+), %fb_c8 : index", r"\1_w = index.mul \2, %fb_c8 : index\n  \1_v = index.sub \1_w, %fb_cnqr : index", qst)
        qst += "  }\n"
        for hh, hn in enumerate(HV):
            for r in range(int(opts["qreg"])):
                qst += (f"  %fqnR{hh}_{r}_o = index.constant {8 * r} : index\n  %fqnR{hh}_{r}_v = index.add %fb_h256, %fqnR{hh}_{r}_o : index\n"
                        f"  %fqnR{hh}_{r} = index.assume %fqnR{hh}_{r}_v [range(%fqnR{hh}_{r}_v, 0, {d - 8})] : index\n"
                        f"  %fqrawR{hh}_{r} = vector.load %q_view[%q_token, {hn}, %fqnR{hh}_{r}] : view<[%tokens]x{qh}x{d}xf32> -> vector<8xf32>\n"
                        f"  %fqhR{hh}_{r} = vector.fptrunc %fqrawR{hh}_{r} : vector<8xf32> to vector<8xf16>\n"
                        f"  %qgf{hh}_{r} = vector.fragment<rhs> %fqhR{hh}_{r} shape [%c16, %c16] : vector<8xf16>\n")
    qst += "  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"
    alloc = (f"  %c0_i = index.constant 0 : index\n  %c1_i = index.constant 1 : index\n" + ("  %c2_i = index.constant 2 : index\n" if nh == 3 else "") +
             f"  %ql_bytes = index.constant {nh * 32 * QP * 2} : offset\n  %ql_tile = buffer.alloca<workgroup> align(16) %ql_bytes : buffer\n"
             f"  %ql_view = buffer.view %ql_tile[%origin] : buffer -> view<{nh}x32x{QP}xf16>\n"
             f"  %pl_bytes = index.constant {nh * waves * 32 * 16} : offset\n  %pl_tile = buffer.alloca<workgroup> align(16) %pl_bytes : buffer\n"
             f"  %pl_view = buffer.view %pl_tile[%origin] : buffer -> view<1x{nh}x{waves}x32x8xf16>\n"
             f"  %mx_bytes = index.constant {nh * waves * 32 * 4} : offset\n  %mx_tile = buffer.alloca<workgroup> align(16) %mx_bytes : buffer\n"
             f"  %mx_view = buffer.view %mx_tile[%origin] : buffer -> view<1x{nh}x{waves}x32xf32>\n"
             f"  %ls_tile = buffer.alloca<workgroup> align(16) %mx_bytes : buffer\n"
             f"  %ls_view = buffer.view %ls_tile[%origin] : buffer -> view<1x{nh}x{waves}x32xf32>\n")
    defs = f"""  %vt_g_c1 = index.constant 1 : index
  %vt_g_crow = index.constant {R} : index
  %vt_g_layout = encoding.layout.strided [%vt_g_c1, %vt_g_crow] : encoding<layout>
  %k_g_layout = encoding.layout.strided [%vt_g_crow, %vt_g_c1] : encoding<layout>
  %vd_cd = index.constant {d} : index
  %vd_kvd = index.mul %kv, %vd_cd : index
  %vd_dk = index.add %vd_kvd, %dim_base : index
  %vd_c2b = index.constant 2 : offset
  %kv_boff = index.scale %vd_kvd, %vd_c2b : index, offset -> offset
  %vd_boff = index.scale %vd_dk, %vd_c2b : index, offset -> offset
  %kh_lane0 = scalar.constant 0 : i32
  %kh_rec_stride = index.constant 40 : offset
  %kh_row = index.assume %row [range(%row, 0, 2147483646)] : index
  %kh_rec_off = index.scale %kh_row, %kh_rec_stride : index, offset -> offset
  %kh_records = buffer.view %origins[%kh_rec_off] : buffer -> view<5xi64>
  %kh_map64 = view.load %kh_records[4] : view<5xi64> -> i64
  %kh_map_b = scalar.assume %kh_map64 [range(%kh_map64, 0, 9223372036854775807)] : i64
  %kh_map_origin = index.cast %kh_map_b : i64 to offset
  %kh_map = buffer.view %origins[%kh_map_origin] : buffer -> view<[%blocks]xi32>
  %kh_last = index.sub %blocks, %c1 : index
  %kh_pp = index.mul %k_plane, %pool : index
  %kh_ppo = index.scale %kh_pp, %pos_bytes : index, offset -> offset
  %kb_o = index.add %kh_ppo, %kv_boff : offset
  %vb_o0 = index.add %kh_ppo, %vd_boff : offset
  %vb_o = index.add %vb_o0, %v_plane_delta : offset
  %pl_rows_m = index.sub %pool, %c16 : index
  %k_pv = buffer.view %arena[%kb_o] : buffer -> view<[%pool]x{d}xf16, %k_g_layout>
  %pg_live_v = index.div %pos0, %c16 : index
  %pg_live = index.assume %pg_live_v [range(%pg_live_v, 0, %kh_last), lt(%pg_live_v, %blocks)] : index
  %c64 = index.constant 64 : index
  %c63 = index.constant 63 : index
  %zv8 = vector.splatf 0.0 : vector<8xf16>
  %nblk_round = index.add %key_count, %c63 : index
  %nblk = index.div %nblk_round, %c64 : index
  %dh16 = index.mul %dim_half, %c16 : index
  %dh16_i32 = index.cast %dh16 : index to i32
  %fb_is0 = index.cmp eq, %dim_half, %c0 : index
  %fb_is1 = index.cmp eq, %dim_half, %c1 : index
  %fb_is2 = index.cmp eq, %dim_half, %c2 : index
  %fb_wv = index.assume %wave [range(%wave, 0, 3)] : index
"""
    for f in range(NF):
        defs += f"  %fq{f} = index.constant {f} : index\n  %kcol{f} = index.constant {16 * f} : index\n"
    if partial:
        pass
    ABC = "ABC"[:nh]
    acc = ", ".join(f"%{AP[hh]}{k} = %init : vector<8xf32>" for hh in range(nh) for k in range(NO))
    outs = ", ".join(f"%{OP[hh]}{k}" for hh in range(nh) for k in range(NO))
    ty16 = ", ".join(["vector<8xf32>"] * (nh * NO))
    mls = ", ".join(f"%m{SUF[hh]}_final, %l{SUF[hh]}_final" for hh in range(nh))
    mli = ", ".join(f"%m{SUF[hh]} = %neg_big : f32, %l{SUF[hh]} = %zero_f : f32" for hh in range(nh))
    mlt = ", ".join(["f32, f32"] * nh)
    def gen_loop(fm):
        L = []
        A = L.append
        A(f"  {mls}, {outs} = scf.for %bt = [%c0 to %nblk step %c1]({mli}, {acc}) -> ({mlt}, {ty16}) {{")
        A("    %bt64 = index.mul %bt, %c64 : index\n    %blk_first = index.add %bt64, %key_low : index\n    %blk_pg0 = index.div %blk_first, %c16 : index\n"
          "    %par_i = index.rem %bt, %c2 : index\n    %par = index.assume %par_i [range(%par_i, 0, 1)] : index")
        for j in range(4):
            A(f"""    %pg{j}_a = index.add %blk_pg0, %fq{j} : index
    %pg{j}_c = index.min %pg{j}_a, %pg_live : index
    %pg{j} = index.assume %pg{j}_c [range(%pg{j}_c, 0, %kh_last), lt(%pg{j}_c, %blocks)] : index
    %ph{j}_r = view.load %kh_map[%pg{j}] : view<[%blocks]xi32> -> i32
    %ph{j}_s = scalar.addi %ph{j}_r, %kh_lane0 : i32
    %ph{j}_b = scalar.assume %ph{j}_s [range(%ph{j}_s, 0, 65535)] : i32
    %ph{j}_i = index.cast %ph{j}_b : i32 to index
    %rw{j}_v = index.mul %ph{j}_i, %c16 : index
    %rw{j} = index.assume %rw{j}_v [range(%rw{j}_v, 0, 2147483568), lt(%rw{j}_v, %pl_rows_m)] : index""")
        A("    %rk_a = scf.select %fb_is2, %rw2, %rw3 : index\n    %rk_b = scf.select %fb_is1, %rw1, %rk_a : index\n    %rk_c = scf.select %fb_is0, %rw0, %rk_b : index\n    %rk = index.assume %rk_c [range(%rk_c, 0, 2147483568), lt(%rk_c, %pl_rows_m)] : index")
        VPF = opts.get("vpf", "")      # "" = the V^T loads inside the PV loop; "top" / "mid" / "top2": issued at the block start / after the QK loop / half each

        def vloads(jjs):
            out = ""
            for jj in jjs:
                out += (f"    %vj{jj}_ro = index.scale %rw{jj}, %pos_bytes : index, offset -> offset\n    %vj{jj}_o = index.add %vb_o, %vj{jj}_ro : offset\n"
                        f"    %vj{jj} = buffer.view %arena[%vj{jj}_o] : buffer -> view<{d // 4}x16xf16, %vt_g_layout>\n")
                for k in range(NO):
                    out += f"    %vc{jj}_{k} = vector.fragment.load<lhs> %vj{jj}[%kcol{k}, %c0] shape [%c16, %c16] : view<{d // 4}x16xf16, %vt_g_layout> -> vector<8xf16>\n"
            return out
        if VPF in ("top", "top2"):
            A(vloads(range(4) if VPF == "top" else (0, 1)))
        A("    %kw_lr_v = index.add %rk, %lane_col : index\n    %kw_lr = index.assume %kw_lr_v [range(%kw_lr_v, 0, 2147483583), lt(%kw_lr_v, %pool)] : index\n"
          f"    %kw_c256 = index.constant {d // 2} : index\n    %kw_h = index.mul %lane_half, %kw_c256 : index\n    %c64k = index.constant 64 : index\n    %fq8c = index.constant 8 : index")
        NQR = int(opts.get("qreg", 0))   # Q^T fragments per head held in registers (row elements 0 .. 8 NQR - 1; the rest in workgroup memory)
        if NQR:
            assert NKW == 2 and NQR == FPW
            cur = ["%init"] * nh
            for kw in range(NKW):
                A(f"""    %kwc{kw}_o = index.constant {64 * kw} : index
    %kwc{kw}_v = index.add %kw_h, %kwc{kw}_o : index
    %kwc{kw} = index.assume %kwc{kw}_v [range(%kwc{kw}_v, 0, {d - 64})] : index
    %kwv{kw} = vector.load %k_pv[%kw_lr, %kwc{kw}] : view<[%pool]x{d}xf16, %k_g_layout> -> vector<64xf16>""")
                nl = 64 - 8 * NQR if kw == 0 else 64
                qoff = "%c0" if kw == 0 else f"%kqo{kw}"
                if kw:
                    A(f"    %kqo{kw} = index.constant {64 * kw - 8 * NQR} : index")
                for hh, x in enumerate(ABC):
                    A(f"    %qrw{x}{kw} = vector.load %ql_view[%c{hh}_i, %lane, {qoff}] : view<{nh}x32x{QP}xf16> -> vector<{nl}xf16>")
                for sl in range(8):
                    A(f"    %kfs{kw}_{sl} = vector.slice %kwv{kw}[{8 * sl}] : vector<64xf16> -> vector<8xf16>\n"
                      f"    %kf{kw}_{sl} = vector.fragment<lhs> %kfs{kw}_{sl} shape [%c16, %c16] : vector<8xf16>")
                    for hh, x in enumerate(ABC):
                        if kw == 0 and sl < NQR:
                            qb = f"%qgf{hh}_{sl}"
                        else:
                            o = 8 * sl - (8 * NQR if kw == 0 else 0)
                            A(f"    %qr{x}{kw}_{sl} = vector.slice %qrw{x}{kw}[{o}] : vector<{nl}xf16> -> vector<8xf16>\n"
                              f"    %qb{x}{kw}_{sl} = vector.fragment<rhs> %qr{x}{kw}_{sl} shape [%c16, %c16] : vector<8xf16>")
                            qb = f"%qb{x}{kw}_{sl}"
                        nm = f"%s{x}_f" if (kw, sl) == (NKW - 1, 7) else f"%sq{x}{kw}_{sl}"
                        A(f"    {nm} = vector.mma %kf{kw}_{sl}, {qb}, {cur[hh]} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
                        cur[hh] = nm
        else:
            CH = int(opts.get("sacc", 1))   # independent S^T accumulation chains per head (2: even / odd fragments, summed after the loop)
            SN = [[f"%s{x}" + ("" if c == 0 else f"c{c}") for c in range(CH)] for x in ABC]
            sinit = ", ".join(f"{SN[hh][c]} = {'%sinit' + x if fm and c == 0 else '%init'} : vector<8xf32>" for c in range(CH) for hh, x in enumerate(ABC))
            sres = ", ".join(f"%s{x}_f" for x in ABC) if CH == 1 else ", ".join(f"%s{x}_p{c}" for c in range(CH) for x in ABC)
            sty = ", ".join(["vector<8xf32>"] * (nh * CH))
            A(f"    {sres} = scf.for %kw = [%c0 to %c{NKW} step %c1]({sinit}) -> ({sty}){' unroll' if opts.get('kwu') else ''}{{")
            A(f"""      %kw_a = index.assume %kw [range(%kw, 0, {NKW - 1})] : index
      %kw_o = index.mul %kw_a, %c64k : index
      %kw_c = index.add %kw_h, %kw_o : index
      %kw_ci = index.assume %kw_c [range(%kw_c, 0, {d - 64})] : index
      %kwv = vector.load %k_pv[%kw_lr, %kw_ci] : view<[%pool]x{d}xf16, %k_g_layout> -> vector<64xf16>
      %kw_f0 = index.mul %kw_a, %fq8c : index
      %kqo_v = index.mul %kw_f0, %fq8c : index
      %kqo = index.assume %kqo_v [range(%kqo_v, 0, {QP - 64})] : index""" + "".join(f"""
      %qrw{x} = vector.load %ql_view[%c{hh}_i, %lane, %kqo] : view<{nh}x32x{QP}xf16> -> vector<64xf16>""" for hh, x in enumerate(ABC)))
            cur = [f"%s{x}" for x in ABC]
            curc = {(hh, c): SN[hh][c] for hh in range(nh) for c in range(CH)}
            for sl in range(8):
                if CH > 1:
                    cur = [curc[(hh, sl % CH)] for hh in range(nh)]
                A(f"""      %kfs{sl} = vector.slice %kwv[{8 * sl}] : vector<64xf16> -> vector<8xf16>
      %kf{sl} = vector.fragment<lhs> %kfs{sl} shape [%c16, %c16] : vector<8xf16>
""" + "".join(f"      %qr{x}{sl} = vector.slice %qrw{x}[{8 * sl}] : vector<64xf16> -> vector<8xf16>\n" for x in ABC)
                  + "".join(f"      %qb{x}{sl} = vector.fragment<rhs> %qr{x}{sl} shape [%c16, %c16] : vector<8xf16>\n" for x in ABC)
                  + "\n".join(f"      %sq{x}{sl} = vector.mma %kf{sl}, %qb{x}{sl}, {cur[hh]} : vector<8xf16>, vector<8xf16>, vector<8xf32>" for hh, x in enumerate(ABC)))
                cur = [f"%sq{x}{sl}" for x in ABC]
                for hh, x in enumerate(ABC):
                    curc[(hh, sl % CH)] = f"%sq{x}{sl}"
            if CH > 1:
                A(f"      scf.yield {', '.join(curc[(hh, c)] for c in range(CH) for hh in range(nh))} : {sty}\n    }}")
                for x in ABC:
                    A(f"    %s{x}_f = vector.addf %s{x}_p0, %s{x}_p1 : vector<8xf32>")
            else:
                A(f"      scf.yield {', '.join(cur)} : {sty}\n    }}")
        if VPF in ("mid", "top2"):
            A(vloads(range(4) if VPF == "mid" else (2, 3)))
        A(f"""    %tf32 = index.cast %blk_first : index to i32
    %kbase32 = scalar.addi %tf32, %dh16_i32 : i32
    %base_v = vector.splat %kbase32 : vector<8xi32>
    %kpos = vector.addi %base_v, %row_off : vector<8xi32>
    %causal = vector.cmpi sle, %kpos, %qp_v : vector<8xi32> -> vector<8xi1>
    %inwin = vector.cmpi sge, %kpos, %low_v : vector<8xi32> -> vector<8xi1>
    %winok = vector.ori %inwin, %ns_v : vector<8xi1>
    %valid = vector.andi %causal, %winok : vector<8xi1>""")
        if fm:   # fixed reference (S^T came out of the MMAs shifted by -m0, log2 units): no running max, no exchange, no rescale
            for hh, s_ in enumerate(SUF):
                ovl = opts.get("ovl")   # overflow tracked as the largest per-lane block sum (> 2^15 falls back: conservative), not the max score
                A(f"""    %masked{s_} = vector.select %valid, %s{ABC[hh]}_f, %neg_v : vector<8xf32>""" + ("" if ovl else f"""
    %lmax{s_} = vector.reduce<maxnumf> %masked{s_}, %neg_big : vector<8xf32>, f32
    %m{s_}_new = scalar.maxnumf %m{s_}, %lmax{s_} : f32""") + f"""
    %p{s_} = vector.exp2f<afn> %masked{s_} : vector<8xf32>
    %ls{s_} = vector.reduce<addf> %p{s_}, %zero_f : vector<8xf32>, f32""" + (f"""
    %m{s_}_new = scalar.maxnumf %m{s_}, %ls{s_} : f32""" if ovl else "") + f"""
    %l{s_}_new = scalar.addf %l{s_}, %ls{s_} : f32
    %pt_h{s_} = vector.fptrunc %p{s_} : vector<8xf32> to vector<8xf16>
    vector.store %pt_h{s_}, %pl_view[%c0_i, %c{hh}_i, %fb_wv, %lane, %c0] : vector<8xf16>, view<1x{nh}x{waves}x32x8xf16>""")
            A("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        else:
            for hh, s_ in enumerate(SUF):
                A(f"""    %sc{s_} = vector.mulf %s{ABC[hh]}_f, %scale_v : vector<8xf32>
    %masked{s_} = vector.select %valid, %sc{s_}, %neg_v : vector<8xf32>
    %lmax{s_} = vector.reduce<maxnumf> %masked{s_}, %neg_big : vector<8xf32>, f32
    %omax{s_}, %omax{s_}_u = kernel.subgroup.shuffle<xor> %lmax{s_}, %exchange, %width : f32, i32, i32
    %tmax{s_} = scalar.maxnumf %lmax{s_}, %omax{s_} : f32
    view.store %tmax{s_}, %mx_view[%c0_i, %c{hh}_i, %fb_wv, %lane] : f32, view<1x{nh}x{waves}x32xf32>""")
            A("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
            for hh, s_ in enumerate(SUF):
                for i in range(4):
                    A(f"    %mx{s_}{i} = view.load %mx_view[%c0_i, %c{hh}_i, %fq{i}, %lane] : view<1x{nh}x{waves}x32xf32> -> f32")
                A(f"""    %mxa{s_} = scalar.maxnumf %mx{s_}0, %mx{s_}1 : f32
    %mxb{s_} = scalar.maxnumf %mx{s_}2, %mx{s_}3 : f32
    %bm{s_} = scalar.maxnumf %mxa{s_}, %mxb{s_} : f32
    %m{s_}_new = scalar.maxnumf %m{s_}, %bm{s_} : f32
    %m{s_}_shift = scalar.subf %m{s_}, %m{s_}_new : f32
    %alpha{s_} = scalar.expf<afn> %m{s_}_shift : f32
    %m{s_}_new_v = vector.splat %m{s_}_new : vector<8xf32>
    %sh{s_} = vector.subf %masked{s_}, %m{s_}_new_v : vector<8xf32>
    %pe{s_} = vector.expf<afn> %sh{s_} : vector<8xf32>
    %p{s_} = vector.select %valid, %pe{s_}, %zero8 : vector<8xf32>
    %ls{s_} = vector.reduce<addf> %p{s_}, %zero_f : vector<8xf32>, f32
    %osum{s_}, %osum{s_}_u = kernel.subgroup.shuffle<xor> %ls{s_}, %exchange, %width : f32, i32, i32
    %tsum{s_} = scalar.addf %ls{s_}, %osum{s_} : f32
    %pt_h{s_} = vector.fptrunc %p{s_} : vector<8xf32> to vector<8xf16>
    vector.store %pt_h{s_}, %pl_view[%c0_i, %c{hh}_i, %fb_wv, %lane, %c0] : vector<8xf16>, view<1x{nh}x{waves}x32x8xf16>
    view.store %tsum{s_}, %ls_view[%c0_i, %c{hh}_i, %fb_wv, %lane] : f32, view<1x{nh}x{waves}x32xf32>""")
            A("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
            for hh, s_ in enumerate(SUF):
                for j in range(4):
                    A(f"    %lv{s_}{j} = view.load %ls_view[%c0_i, %c{hh}_i, %fq{j}, %lane] : view<1x{nh}x{waves}x32xf32> -> f32")
                A(f"""    %lt01{s_} = scalar.addf %lv{s_}0, %lv{s_}1 : f32
    %lt012{s_} = scalar.addf %lt01{s_}, %lv{s_}2 : f32
    %lt{s_} = scalar.addf %lt012{s_}, %lv{s_}3 : f32
    %l_scaled{s_} = scalar.mulf %l{s_}, %alpha{s_} : f32
    %l{s_}_new = scalar.addf %l_scaled{s_}, %lt{s_} : f32
    %alpha{s_}_v = vector.splat %alpha{s_} : vector<8xf32>""")
        if partial:
            A("    %pos_hi_v = vector.splat %pos_hi_i32 : vector<8xi32>")
        PA = ["pa", "pb", "pcc"]
        BN = ["bn", "bnB", "bnC"]
        ON = ["on", "onB", "onC"]
        for k in range(NO if not fm else 0):
            A("\n".join(f"    %os{SUF[hh]}{k} = vector.mulf %{AP[hh]}{k}, %alpha{SUF[hh]}_v : vector<8xf32>" for hh in range(nh)))
        if VPF:   # straight-line PV over the four pages of the block (V^T fragments already loaded)
            cur = {(hh, k): (f"%{AP[hh]}{k}" if fm else f"%os{SUF[hh]}{k}") for hh in range(nh) for k in range(NO)}
            for jj in range(4):
                for hh, x in enumerate(ABC):
                    A(f"    %pr{x}{jj} = vector.load %pl_view[%c0_i, %c{hh}_i, %fq{jj}, %lane, %c0] : view<1x{nh}x{waves}x32x8xf16> -> vector<8xf16>\n"
                      f"    %pf{x}{jj} = vector.fragment<rhs> %pr{x}{jj} shape [%c16, %c16] : vector<8xf16>")
                if partial:
                    A(f"""    %vkc{jj} = scalar.constant {16 * jj} : i32
    %vkb{jj} = scalar.addi %tf32, %vkc{jj} : i32
    %vkb{jj}_v = vector.splat %vkb{jj} : vector<8xi32>
    %vkp{jj} = vector.addi %vkb{jj}_v, %row_off : vector<8xi32>
    %vok{jj} = vector.cmpi sle, %vkp{jj}, %pos_hi_v : vector<8xi32> -> vector<8xi1>""")
                for k in range(NO):
                    vname = f"%vc{jj}_{k}"
                    if partial:
                        A(f"    %vs{jj}_{k} = vector.select %vok{jj}, %vc{jj}_{k}, %zv8 : vector<8xf16>\n    %vm{jj}_{k} = vector.fragment<lhs> %vs{jj}_{k} shape [%c16, %c16] : vector<8xf16>")
                        vname = f"%vm{jj}_{k}"
                    for hh, x in enumerate(ABC):
                        nm = f"%{ON[hh]}{k}" if jj == 3 else f"%{BN[hh]}{jj}_{k}"
                        A(f"    {nm} = vector.mma {vname}, %pf{x}{jj}, {cur[(hh, k)]} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
                        cur[(hh, k)] = nm
        else:
            pvin = ", ".join(f"%{PA[hh]}{k} = %os{SUF[hh]}{k} : vector<8xf32>" for hh in range(nh) for k in range(NO))
            pvout = ", ".join(f"%{ON[hh]}{k}" for hh in range(nh) for k in range(NO))
            A(f"    {pvout} = scf.for %jj = [%c0 to %c4 step %c1]({pvin}) -> ({ty16}) unroll {{")
            VJ = """      %rj_a = scf.select %jj_c2, %rw2, %rw3 : index
      %rj_b = scf.select %jj_c1, %rw1, %rj_a : index
      %rj_c = scf.select %jj_c0, %rw0, %rj_b : index
      %rj = index.assume %rj_c [range(%rj_c, 0, 2147483568), lt(%rj_c, %pl_rows_m)] : index
      %vj_ro = index.scale %rj, %pos_bytes : index, offset -> offset
      %vj_o = index.add %vb_o, %vj_ro : offset
"""
            A(f"""      %jj_a = index.assume %jj [range(%jj, 0, 3)] : index
      %jj_c1 = index.cmp eq, %jj, %c1 : index
      %jj_c2 = index.cmp eq, %jj, %c2 : index
      %jj_c0 = index.cmp eq, %jj, %c0 : index
{VJ}      %vj = buffer.view %arena[%vj_o] : buffer -> view<{d // 4}x16xf16, %vt_g_layout>""" + "".join(f"""
      %pr{x} = vector.load %pl_view[%c0_i, %c{hh}_i, %jj_a, %lane, %c0] : view<1x{nh}x{waves}x32x8xf16> -> vector<8xf16>
      %pf{x} = vector.fragment<rhs> %pr{x} shape [%c16, %c16] : vector<8xf16>""" for hh, x in enumerate(ABC)))
            if partial:
                A("""      %jj16 = index.mul %jj_a, %c16 : index
      %jj16_i = index.cast %jj16 : index to i32
      %vkb = scalar.addi %tf32, %jj16_i : i32
      %vkb_v = vector.splat %vkb : vector<8xi32>
      %vkp = vector.addi %vkb_v, %row_off : vector<8xi32>
      %vok = vector.cmpi sle, %vkp, %pos_hi_v : vector<8xi32> -> vector<8xi1>""")
            for k in range(NO):
                A(f"      %vc{k} = vector.fragment.load<lhs> %vj[%kcol{k}, %c0] shape [%c16, %c16] : view<{d // 4}x16xf16, %vt_g_layout> -> vector<8xf16>")
            for k in range(NO):
                vname = f"%vc{k}"
                if partial:
                    A(f"      %vs{k} = vector.select %vok, %vc{k}, %zv8 : vector<8xf16>\n      %vm{k} = vector.fragment<lhs> %vs{k} shape [%c16, %c16] : vector<8xf16>")
                    vname = f"%vm{k}"
                A("\n".join(f"      %{BN[hh]}{k} = vector.mma {vname}, %pf{x}, %{PA[hh]}{k} : vector<8xf16>, vector<8xf16>, vector<8xf32>" for hh, x in enumerate(ABC)))
            A("      scf.yield " + ", ".join(f"%{BN[hh]}{k}" for hh in range(nh) for k in range(NO)) + f" : {ty16}")
            A("    }")
        A("    scf.yield " + ", ".join(f"%m{SUF[hh]}_new, %l{SUF[hh]}_new" for hh in range(nh)) + ", " + ", ".join(f"%{ON[hh]}{k}" for hh in range(nh) for k in range(NO)) + f" : {mlt}, {ty16}\n  }}\n")
        return L

    FM = bool(opts.get("fmax"))
    if not FM:
        L = gen_loop(False)
    else:
        assert nh == 2 and opts.get("vpf") and opts.get("e2") and not opts.get("qreg"), "fmax: two heads, V^T prefetch, e2"
        def fres(sfx):
            return ([f"%m{SUF[hh]}_final{sfx}" for hh in range(nh)], [f"%l{SUF[hh]}_final{sfx}" for hh in range(nh)],
                    [f"%{OP[hh]}{k}{sfx}" for hh in range(nh) for k in range(NO)])
        def renamed(lines, sfx):
            t = "".join(x if x.endswith("\n") else x + "\n" for x in lines)
            h, rest = t.split("\n", 1)
            for a in [x for g in fres("") for x in g]:
                h = re.sub(re.escape(a) + r"\b(?!\w)", a + sfx, h, count=1)
            return h + "\n" + rest
        P = []
        P.append(f"""  // fixed reference m0 = max over the first page and the diagonal page (wave w: page w / 2, head w % 2); a later score above m0 + 15 (log2)
  // makes the whole tile fall back to the running-max loop
  %fm_isd = index.cmp uge, %fb_wv, %c2 : index
  %fm_pgf_v = index.div %key_low, %c16 : index
  %fm_pgf = index.min %fm_pgf_v, %pg_live : index
  %fm_pg_s = scf.select %fm_isd, %pg_live, %fm_pgf : index
  %fm_pg = index.assume %fm_pg_s [range(%fm_pg_s, 0, %kh_last), lt(%fm_pg_s, %blocks)] : index
  %fm_ph_r = view.load %kh_map[%fm_pg] : view<[%blocks]xi32> -> i32
  %fm_ph_s = scalar.addi %fm_ph_r, %kh_lane0 : i32
  %fm_ph_b = scalar.assume %fm_ph_s [range(%fm_ph_s, 0, 65535)] : i32
  %fm_ph_i = index.cast %fm_ph_b : i32 to index
  %fm_rw_v = index.mul %fm_ph_i, %c16 : index
  %fm_rw = index.assume %fm_rw_v [range(%fm_rw_v, 0, 2147483568), lt(%fm_rw_v, %pl_rows_m)] : index
  %fm_lr_v = index.add %fm_rw, %lane_col : index
  %fm_lr = index.assume %fm_lr_v [range(%fm_lr_v, 0, 2147483583), lt(%fm_lr_v, %pool)] : index
  %fm_hq_v = index.rem %fb_wv, %c2 : index
  %fm_hq = index.assume %fm_hq_v [range(%fm_hq_v, 0, 1)] : index
  %fm_c128 = index.constant {d // 2} : index
  %fm_kh = index.mul %lane_half, %fm_c128 : index""")
        cur = "%init"
        for kw in range(NKW):
            P.append(f"""  %fm_kc{kw}_o = index.constant {64 * kw} : index
  %fm_kc{kw}_v = index.add %fm_kh, %fm_kc{kw}_o : index
  %fm_kc{kw} = index.assume %fm_kc{kw}_v [range(%fm_kc{kw}_v, 0, {d - 64})] : index
  %fm_kv{kw} = vector.load %k_pv[%fm_lr, %fm_kc{kw}] : view<[%pool]x{d}xf16, %k_g_layout> -> vector<64xf16>
  %fm_qv{kw} = vector.load %ql_view[%fm_hq, %lane, %fm_kc{kw}_o] : view<{nh}x32x{QP}xf16> -> vector<64xf16>""")
            for sl in range(8):
                P.append(f"""  %fm_kf{kw}_{sl}s = vector.slice %fm_kv{kw}[{8 * sl}] : vector<64xf16> -> vector<8xf16>
  %fm_kf{kw}_{sl} = vector.fragment<lhs> %fm_kf{kw}_{sl}s shape [%c16, %c16] : vector<8xf16>
  %fm_qf{kw}_{sl}s = vector.slice %fm_qv{kw}[{8 * sl}] : vector<64xf16> -> vector<8xf16>
  %fm_qf{kw}_{sl} = vector.fragment<rhs> %fm_qf{kw}_{sl}s shape [%c16, %c16] : vector<8xf16>
  %fm_s{kw}_{sl} = vector.mma %fm_kf{kw}_{sl}, %fm_qf{kw}_{sl}, {cur} : vector<8xf16>, vector<8xf16>, vector<8xf32>""")
                cur = f"%fm_s{kw}_{sl}"
        P.append(f"""  %fm_pk = index.mul %fm_pg, %c16 : index
  %fm_pk32 = index.cast %fm_pk : index to i32
  %fm_pkv = vector.splat %fm_pk32 : vector<8xi32>
  %fm_kpos = vector.addi %fm_pkv, %row_off : vector<8xi32>
  %fm_ok = vector.cmpi sle, %fm_kpos, %qp_v : vector<8xi32> -> vector<8xi1>
  %fm_msk = vector.select %fm_ok, {cur}, %neg_v : vector<8xf32>
  %fm_lm = vector.reduce<maxnumf> %fm_msk, %neg_big : vector<8xf32>, f32
  %fm_om, %fm_om_u = kernel.subgroup.shuffle<xor> %fm_lm, %exchange, %width : f32, i32, i32
  %fm_tm = scalar.maxnumf %fm_lm, %fm_om : f32
  view.store %fm_tm, %mx_view[%c0_i, %fm_hq, %fb_wv, %lane] : f32, view<1x{nh}x{waves}x32xf32>
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)""")
        for hh, x in enumerate(ABC):
            s_ = SUF[hh]
            P.append(f"""  %fm_ma{s_} = view.load %mx_view[%c0_i, %c{hh}_i, %fq{hh}, %lane] : view<1x{nh}x{waves}x32xf32> -> f32
  %fm_mb{s_} = view.load %mx_view[%c0_i, %c{hh}_i, %fq{hh + 2}, %lane] : view<1x{nh}x{waves}x32xf32> -> f32
  %fm_m{s_} = scalar.maxnumf %fm_ma{s_}, %fm_mb{s_} : f32
  %fm_nm{s_} = scalar.subf %zero_f, %fm_m{s_} : f32
  %fm_nmv{s_} = vector.splat %fm_nm{s_} : vector<8xf32>
  %sinit{x} = vector.fragment<init> %fm_nmv{s_} shape [%c16, %c16] : vector<8xf32>""")
        P.append("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        def fast_loops(sa, su, si):
            """The fixed-reference loops with result suffixes SA (masked / only loop) and SU (unmasked loop), S^T initialised from %sinit{SI}x."""
            fast = renamed(gen_loop(True), sa)
            if opts.get("peel"):   # blocks wholly at or below the tile's first position need no mask: an unmasked loop, then the masked tail
                un = renamed(gen_loop(True), su)
                un = un.replace("[%c0 to %nblk step %c1]", "[%c0 to %fm_nfull step %c1]", 1)
                un = re.sub(r"(%masked\w*) = vector\.select %valid, (%s\w+_f), %neg_v : vector<8xf32>", r"\1 = vector.addf \2, %zero8 : vector<8xf32>", un)
                mu, lu, ou = fres(su)
                init = ", ".join(f"%m{SUF[hh]} = {mu[hh]} : f32, %l{SUF[hh]} = {lu[hh]} : f32" for hh in range(nh)) + ", " + ", ".join(
                    f"%{AP[hh]}{k} = {ou[hh * NO + k]} : vector<8xf32>" for hh in range(nh) for k in range(NO))
                h, rest = fast.split("\n", 1)
                h = h.replace("[%c0 to %nblk step %c1]", "[%fm_nfull to %nblk step %c1]").replace(f"({mli}, {acc})", f"({init})")
                assert "%fm_nfull" in h and init in h
                fast = un + h + "\n" + rest
            return fast.replace("%sinitA", f"%sinit{si}A").replace("%sinitB", f"%sinit{si}B")

        def comb(sa, q):
            """Per head: l summed and the overflow (max S^T - m0) maxed over the lane pair and the four waves: %fl{q}{h}_t, %fo{q}{h}_t."""
            mf, lf, _ = fres(sa)
            C = ["  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)"] if q else []
            for hh, s_ in enumerate(SUF):
                C.append(f"""  %fl{q}{s_}_o, %fl{q}{s_}_u = kernel.subgroup.shuffle<xor> {lf[hh]}, %exchange, %width : f32, i32, i32
  %fl{q}{s_}_w = scalar.addf {lf[hh]}, %fl{q}{s_}_o : f32
  %fo{q}{s_}_o, %fo{q}{s_}_u = kernel.subgroup.shuffle<xor> {mf[hh]}, %exchange, %width : f32, i32, i32
  %fo{q}{s_}_w = scalar.maxnumf {mf[hh]}, %fo{q}{s_}_o : f32
  view.store %fl{q}{s_}_w, %ls_view[%c0_i, %c{hh}_i, %fb_wv, %lane] : f32, view<1x{nh}x{waves}x32xf32>
  view.store %fo{q}{s_}_w, %mx_view[%c0_i, %c{hh}_i, %fb_wv, %lane] : f32, view<1x{nh}x{waves}x32xf32>""")
            C.append("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
            for hh, s_ in enumerate(SUF):
                for j in range(4):
                    C.append(f"  %fl{q}{s_}{j} = view.load %ls_view[%c0_i, %c{hh}_i, %fq{j}, %lane] : view<1x{nh}x{waves}x32xf32> -> f32\n"
                             f"  %fo{q}{s_}{j} = view.load %mx_view[%c0_i, %c{hh}_i, %fq{j}, %lane] : view<1x{nh}x{waves}x32xf32> -> f32")
                C.append(f"""  %fl{q}{s_}01 = scalar.addf %fl{q}{s_}0, %fl{q}{s_}1 : f32
  %fl{q}{s_}23 = scalar.addf %fl{q}{s_}2, %fl{q}{s_}3 : f32
  %fl{q}{s_}_t = scalar.addf %fl{q}{s_}01, %fl{q}{s_}23 : f32
  %fo{q}{s_}01 = scalar.maxnumf %fo{q}{s_}0, %fo{q}{s_}1 : f32
  %fo{q}{s_}23 = scalar.maxnumf %fo{q}{s_}2, %fo{q}{s_}3 : f32
  %fo{q}{s_}_t = scalar.maxnumf %fo{q}{s_}01, %fo{q}{s_}23 : f32""")
            return "\n".join(C) + "\n"

        assert not opts.get("ovl"), "fmax: the second pass needs the true score max"
        prel = (f"  %fm_span = index.sub %pos0, %key_low : index\n  %fm_span1 = index.add %fm_span, %c1 : index\n"
                f"  %fm_nf_v = index.div %fm_span1, %c64 : index\n  %fm_nfull = index.min %fm_nf_v, %nblk : index\n") if opts.get("peel") else ""
        NW = 12 if opts.get("wg") == 3 else 4   # waves of the workgroup: the second pass is taken by all of them or none
        flag = f"""  %c0_f = index.constant 0 : index
  %fm_lim = scalar.constant 15.0 : f32
  %fm_ol = scalar.maxnumf %fo_t, %foB_t : f32
  %fm_svf = kernel.subgroup.reduce<maxnumf> %fm_ol : f32
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  view.store %fm_svf, %ls_view[%c0_f, %c0_f, %c0_f, %wave] : f32, view<1x{nh}x{waves}x32xf32>
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)""" + "".join(f"""
  %fm_wc{j} = index.constant {j} : index
  %fm_v{j} = view.load %ls_view[%c0_f, %c0_f, %c0_f, %fm_wc{j}] : view<1x{nh}x{waves}x32xf32> -> f32""" for j in range(NW)) + "".join(f"""
  %fm_vm{j} = scalar.maxnumf {"%fm_v0" if j == 1 else f"%fm_vm{j - 1}"}, %fm_v{j} : f32""" for j in range(1, NW)) + f"""
  %fm_vm = scalar.addf %fm_vm{NW - 1}, %zero_f : f32
  %fm_any = scalar.cmpf ogt, %fm_vm, %fm_lim : f32
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  {mls}, {outs} = scf.if %fm_any -> ({mlt}, {ty16}) {{
"""
        # second pass (some lane of the workgroup had a score above its m0 + 15): every lane reruns the same loops, a lane that overflowed
        # with m0 raised to its true max, every other lane with its m0 unchanged (bit-identical to the first pass: the result of a token
        # depends on that token only, so a prefill cut anywhere gives the same bits)
        p2 = ""
        for hh, x in enumerate(ABC):
            s_ = SUF[hh]
            p2 += f"""  %fm2_ov{s_} = scalar.cmpf ogt, %fo{s_}_t, %fm_lim : f32
  %fm2_up{s_} = scalar.addf %fm_m{s_}, %fo{s_}_t : f32
  %fm2_m{s_} = scf.select %fm2_ov{s_}, %fm2_up{s_}, %fm_m{s_} : f32
  %fm2_nm{s_} = scalar.subf %zero_f, %fm2_m{s_} : f32
  %fm2_nmv{s_} = vector.splat %fm2_nm{s_} : vector<8xf32>
  %sinit2{x} = vector.fragment<init> %fm2_nmv{s_} shape [%c16, %c16] : vector<8xf32>
"""
        p2 += fast_loops("_fb", "_fv", "2") + comb("_fb", "2")
        _, _, o2 = fres("_fb")
        _, _, of = fres("_fa")
        tail_if = (f"    scf.yield {', '.join(f'%fo2{SUF[hh]}_t, %fl2{SUF[hh]}_t' for hh in range(nh))}, {', '.join(o2)} : {mlt}, {ty16}\n  }} else {{\n"
                   f"    scf.yield {', '.join(f'%fo{SUF[hh]}_t, %fl{SUF[hh]}_t' for hh in range(nh))}, {', '.join(of)} : {mlt}, {ty16}\n  }}\n")
        L = ["\n".join(P) + "\n" + prel + fast_loops("_fa", "_fu", "") + comb("_fa", "") + flag + p2 + tail_if]
    loop = "".join(x if x.endswith("\n") else x + "\n" for x in L)
    loop = _ablate(loop, (opts or {}).get("abl", ""))
    if opts.get("e2"):   # exp2 domain: Q^T staged pre-scaled by scale * log2(e); S^T comes out in log2 units, P = exp2(S - m)
        c2 = float(re.search(r"%scale = scalar\.constant (\S+) : f32", text).group(1)) * 1.4426950408889634
        qst = (f"  %e2c = scalar.constant {c2!r} : f32\n  %e2c_v = vector.splat %e2c : vector<8xf32>\n"
               + re.sub(r"(%fqh\w+) = vector\.fptrunc (%fqraw\w+) : vector<8xf32>",
                        r"\2s = vector.mulf \2, %e2c_v : vector<8xf32>\n  \1 = vector.fptrunc \2s : vector<8xf32>", qst))
        for m in list(re.finditer(r"^ *(%sc\w*) = vector\.mulf (%s\w+_f), %scale_v : vector<8xf32>\n", loop, flags=re.M)):
            loop = loop.replace(m.group(0), "")
            loop = re.sub(re.escape(m.group(1)) + r"\b(?!\w)", m.group(2), loop)
        loop = loop.replace("vector.expf<afn>", "vector.exp2f<afn>").replace("scalar.expf<afn>", "scalar.exp2f<afn>")
    mid = alloc + defs + qst + loop
    head_text = text[:h0]
    if opts.get("wg") == 3:   # 12-wave workgroup: the three head pairs of the KV head (gen_flash2s.py WG3); wave = 4 pair + quarter
        assert nh == 2 and d == 256
        np_ = 3
        head_text = text[:h0].replace("  %head_value = index.madd %kv, %cg, %head_in_group : index",
                            "  %hig2 = index.mul %head_in_group, %c2 : index\n  %head_value = index.madd %kv, %cg, %hig2 : index")
        mid = mid.replace("%fb_wv = index.assume %wave [range(%wave, 0, 3)]", "%fb_wv = index.assume %dim_half [range(%dim_half, 0, 3)]")
        pre = "  %wg_pair = index.assume %head_in_group [range(%head_in_group, 0, 2)] : index\n  %wg_cnh = index.constant 2 : index\n" + "".join(
            f"  %wg_q{hh}_v = index.madd %wg_pair, %wg_cnh, %c{hh}_i : index\n  %wg_q{hh} = index.assume %wg_q{hh}_v [range(%wg_q{hh}_v, 0, 5)] : index\n"
            for hh in range(nh))
        if not opts.get("qshare"):   # qshare: timing ablation, the three pairs read pair 0's Q^T region (wrong results)
            mid = mid.replace(f"view<{nh}x32x{QP}xf16>", f"view<{np_ * nh}x32x{QP}xf16>").replace(
                f"%ql_bytes = index.constant {nh * 32 * QP * 2}", f"%ql_bytes = index.constant {np_ * nh * 32 * QP * 2}")
            for hh in range(nh):
                mid = mid.replace(f"%ql_view[%c{hh}_i,", f"%ql_view[%wg_q{hh},")
        mid = mid.replace(f"view<1x{nh}x{waves}x", f"view<{np_}x{nh}x{waves}x").replace("[%c0_i, %c", "[%wg_pair, %c").replace("[%c0_i, %fm_hq", "[%wg_pair, %fm_hq")
        mid = mid.replace(f"%pl_bytes = index.constant {nh * waves * 32 * 16}", f"%pl_bytes = index.constant {np_ * nh * waves * 32 * 16}").replace(
            f"%mx_bytes = index.constant {nh * waves * 32 * 4}", f"%mx_bytes = index.constant {np_ * nh * waves * 32 * 4}")
        i = mid.index("  %pl_bytes")
        mid = mid[:i] + pre + mid[i:]
    text = head_text + mid + epilogue + "".join(epis) + tail
    return text


def _ablate(loop, abl):
    """Timing ablations of the loop (kbench only; results are wrong): kv0 every K / V^T page is page 0 (cache-resident), noqk no QK
    WMMAs (nor K loads), nopv no PV WMMAs (nor V^T loads), nobar no barriers, noexp no exponentials."""
    for a in [x for x in abl.split(",") if x]:
        if a == "kv0":
            loop = re.sub(r"(%rw\d) = index\.assume %rw\d_v \[range\(%rw\d_v,", r"\1 = index.assume %c0 [range(%c0,", loop)
            loop = re.sub(r"lt\(%rw\d_v, %pl_rows_m\)", "lt(%c0, %pl_rows_m)", loop)
        elif a in ("noqk", "nopv"):
            pat = r"^ *(%sq\w+) = vector\.mma \S+, \S+, (%\w+) :.*\n" if a == "noqk" else r"^ *(%(?:bn|on)\w+) = vector\.mma \S+, \S+, (%\w+) :.*\n"
            while (m := re.search(pat, loop, flags=re.M)):   # one at a time: a rename rewrites the later MMAs of a chain
                loop = loop.replace(m.group(0), "", 1)
                loop = re.sub(re.escape(m.group(1)) + r"\b(?!\w)", m.group(2), loop)
        elif a == "k0":   # K rows of page 0 only (V as is)
            loop = loop.replace("%kw_lr_v = index.add %rk, %lane_col", "%kw_lr_v = index.add %c0, %lane_col")
        elif a == "v0":   # V^T rows of page 0 only (K as is)
            assert "%vj0_ro" in loop, "v0: V^T prefetch variants only"   # page jj -> row 16 jj: four distinct cache-resident pages
            loop = re.sub(r"(%vj(\d)_ro) = index\.scale %rw\d,", r"\1 = index.scale %kcol\2,", loop)
        elif a == "nobar":
            loop = re.sub(r"^ *kernel\.barrier.*\n", "", loop, flags=re.M)
        elif a == "noexp":
            loop = re.sub(r"vector\.expf<afn> (%\w+) : vector<8xf32>", r"vector.addf \1, %zero8 : vector<8xf32>", loop)
        else:
            raise ValueError("flash2_b2: unknown ablation " + a)
    return loop
