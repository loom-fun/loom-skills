"""MOE_CW_Q4K (choice moe.prefill.cw_q4k): column-wave grouped MoE GEMMs for Q4_K and Q5_K experts, drop-in for
gen_moe.grouped_gemm's g_moe_ggu_{q4_k,q5_k}_k* (gate/up, swiglu) and g_moe_gdn_{q4_k,q5_k}_k* (down, store): same names, launch
arguments and bucketed 64-row expert tiles, with the gfx12 twin `<name>_rdna4` (target(@rdna4)) built by the same generator.
Q5_K: the same header and code layout (codes at byte 48 of 176) plus qh[32] at byte 16; each lane also loads the qh bytes of
its K16 pieces (gfx11 16, gfx12 8), shifts them right 4 once per chunk for the upper half, and per K16 piece of group g ORs
(qh << (4 - g)) & 0x10101010 into the masked nibbles (3 VALU per code word): the codes equal gen_gemm.dequant16's Q5_K codes.

Why: at a 1k-2k prompt with 256 experts top-8 an expert holds ~32-64 rows, and the 64x64-tile grouped GEMM is bound by its
per-64-K step (half the threads dequantize a 64x64 weight tile into LDS, 2 barriers, 4 row waves read it back). Here each wave
owns output columns instead (the reference kit's gen_moe_cw.py for IQ4_XS, ported to Q4_K):
  - every lane dequantizes its own column's Q4_K values from global straight into WMMA B fragments (no weight LDS stage, no
    weight barrier); the activation rows (the gathered x16 rows for gate/up, the sorted rows for down) are staged in LDS 128 K
    per barrier pair and shared by the 8 waves;
  - only the tile's live 16-row fragments get MMAs (nf = ceil(rows / 16) from tile_info: a uniform branch into a copy of the
    K loop specialized for nf = 1..4), so the WMMA work follows the real rows and is spread over all 8 waves;
  - the six-bit scales/mins of the chunk's four 32-groups are decoded word-wide once per chunk (two masks and shifts per word
    instead of a g_k_metadata call per 16 values) and scaled in f16: d * sc and dmin * m are exact in f32 (11 + 6 bits), so the
    f16 product rounds to the same value as the 64-row kernel's fptrunc(f32 product);
  - gfx11 (WMMA B wants both half-waves to hold the same 16 K values): each lane decodes its column's 16 values per K16;
    gfx12 (lanes 16-31 hold K 8-15 of a B fragment): each lane decodes only its 8, no duplicated work;
    MOE_CW_Q4K token 'xch' (gfx11 only, experiment): lanes 0-15 decode the even, 16-31 the odd group of a code word pair and an
    xor-16 shuffle swaps them (half the decode, plus 8 shuffles and 16 selects per 16 values: about a tie in VALU);
  - the next chunk's weights (header + 4 code vectors per weight fragment) and activations are loaded right after the chunk's
    first barrier and carried into the next iteration.
swiglu: wave w owns gate and up columns out*128 + 16w .. +16 (128-column tiles); store: out*256 + 32w .. +32 (256-column tiles).
Values: the dequant is gen_gemm.dequant16's Q4_K (uitofp(code) fma f16 scale + f16 -min), the MMAs accumulate K in the same
16-wide ascending order and the epilogue is the 64-row kernel's, so live rows are bit-identical to the current kernels; rows of
skipped fragments are not written (padding rows: never combined). moe.prefill.store16 (MOE_STORE16) stores the down rows as f16."""

KC = 128   # K per staged chunk (half a Q4_K superblock: 4 groups of 32)


def _suffix(text, sfx):
    """rename every SSA value defined in `text` (one specialized loop body) with suffix `sfx`, so sibling bodies do not collide"""
    import re
    names = set(re.findall(r"%([A-Za-z_]\w*)\s*(?=(?:,\s*%[A-Za-z_]\w*\s*)*=)", text))
    names |= set(re.findall(r"scf\.for %([A-Za-z_]\w*)", text))
    names |= set(re.findall(r"[(,] ?%([A-Za-z_]\w*) = %", text))
    return re.sub(r"%([A-Za-z_]\w*)\b", lambda m: f"%{m.group(1)}{sfx}" if m.group(1) in names else m.group(0), text)


def _meta(o, hdr):
    """%{o}_meta vector<8xf16>: f16 scale (d * sc) of the chunk's groups 0-3, then f16 -(dmin * m) of the same groups, from the
    superblock header `hdr` (vector<4xi32>: d, dmin, 12 scale bytes) and the uniform chunk half %sb_hi (groups 4-7)."""
    return f"""      %{o}_w1 = vector.extract {hdr}[1] : vector<4xi32> -> i32
      %{o}_w2 = vector.extract {hdr}[2] : vector<4xi32> -> i32
      %{o}_w3 = vector.extract {hdr}[3] : vector<4xi32> -> i32
      %{o}_p = vector.from_elements %{o}_w1, %{o}_w2 : vector<2xi32>
      %{o}_lo = vector.andi %{o}_p, %cq_m63 : vector<2xi32>
      %{o}_q3 = vector.from_elements %{o}_w3, %{o}_w3 : vector<2xi32>
      %{o}_q3s = vector.shrui %{o}_q3, %cq_sh04 : vector<2xi32>
      %{o}_hl = vector.andi %{o}_q3s, %cq_m15w : vector<2xi32>
      %{o}_ps = vector.shrui %{o}_p, %cq_sh2 : vector<2xi32>
      %{o}_hh = vector.andi %{o}_ps, %cq_m48 : vector<2xi32>
      %{o}_hi = vector.ori %{o}_hl, %{o}_hh : vector<2xi32>
      %{o}_sm = scf.select %sb_hi, %{o}_hi, %{o}_lo : vector<2xi32>
      %{o}_smb = vector.bitcast %{o}_sm : vector<2xi32> to vector<8xi8>
      %{o}_smf = vector.uitofp %{o}_smb : vector<8xi8> to vector<8xf16>
      %{o}_halves = vector.bitcast {hdr} : vector<4xi32> to vector<8xf16>
      %{o}_d = vector.extract %{o}_halves[0] : vector<8xf16> -> f16
      %{o}_dm = vector.extract %{o}_halves[1] : vector<8xf16> -> f16
      %{o}_ndm = scalar.negf %{o}_dm : f16
      %{o}_dd = vector.from_elements %{o}_d, %{o}_d, %{o}_d, %{o}_d, %{o}_ndm, %{o}_ndm, %{o}_ndm, %{o}_ndm : vector<8xf16>
      %{o}_meta = vector.mulf %{o}_smf, %{o}_dd : vector<8xf16>"""


def _values(o, meta, words, g, VW, VF, qh=None):
    """%{o}_v vector<VF x f16>: the code words `words` (vector<VW x i32>) of group g (0-3 in the chunk) dequantized.
    qh (Q5_K): the matching qh words, already shifted right by 4 for the chunk's upper half, so bit g of each byte is the
    value's fifth bit: (qh << (4 - g)) & 0x10 per byte moves it to bit 4 of the code byte (gen_gemm.dequant16's Q5_K code)."""
    sh = (f"      %{o}_sh = vector.shrui {words}, %cq_f4 : vector<{VW}xi32>\n" if g % 2 else "")
    src = f"%{o}_sh" if g % 2 else words
    code = f"%{o}_mk"
    if qh is not None:
        code = f"%{o}_mq"
        sh5 = f"""      %{o}_hs = vector.shli {qh}, %cq_{"f4" if g == 0 else "s" + str(4 - g)} : vector<{VW}xi32>
      %{o}_hb = vector.andi %{o}_hs, %cq_m16 : vector<{VW}xi32>
      %{o}_mq = vector.ori %{o}_mk, %{o}_hb : vector<{VW}xi32>
"""
    return f"""{sh}      %{o}_mk = vector.andi {src}, %cq_m15 : vector<{VW}xi32>
{sh5 if qh is not None else ""}      %{o}_b = vector.bitcast {code} : vector<{VW}xi32> to vector<{VF}xi8>
      %{o}_c = vector.uitofp %{o}_b : vector<{VF}xi8> to vector<{VF}xf16>
      %{o}_s = vector.extract {meta}[{g}] : vector<8xf16> -> f16
      %{o}_n = vector.extract {meta}[{4 + g}] : vector<8xf16> -> f16
      %{o}_ss = vector.splat %{o}_s : vector<{VF}xf16>
      %{o}_ns = vector.splat %{o}_n : vector<{VF}xf16>
      %{o}_v = vector.fmaf %{o}_c, %{o}_ss, %{o}_ns : vector<{VF}xf16>"""


def _xch_values(o, meta, words, p, VW, VF):
    """gfx11 half-wave exchange: lanes 0-15 decode group 2p, lanes 16-31 group 2p + 1 of the same code words (low / high
    nibbles), an xor-16 shuffle hands each half the other's values. -> %{o}_x (group 2p), %{o}_y (group 2p + 1), vector<16xf16>."""
    t = f"""      %{o}_sh = vector.shrui {words}, %cq_lsh : vector<{VW}xi32>
      %{o}_mk = vector.andi %{o}_sh, %cq_m15 : vector<{VW}xi32>
      %{o}_b = vector.bitcast %{o}_mk : vector<{VW}xi32> to vector<{VF}xi8>
      %{o}_c = vector.uitofp %{o}_b : vector<{VF}xi8> to vector<{VF}xf16>
      %{o}_s0 = vector.extract {meta}[{2 * p}] : vector<8xf16> -> f16
      %{o}_s1 = vector.extract {meta}[{2 * p + 1}] : vector<8xf16> -> f16
      %{o}_n0 = vector.extract {meta}[{4 + 2 * p}] : vector<8xf16> -> f16
      %{o}_n1 = vector.extract {meta}[{5 + 2 * p}] : vector<8xf16> -> f16
      %{o}_s = scf.select %cq_hi, %{o}_s1, %{o}_s0 : f16
      %{o}_n = scf.select %cq_hi, %{o}_n1, %{o}_n0 : f16
      %{o}_ss = vector.splat %{o}_s : vector<{VF}xf16>
      %{o}_ns = vector.splat %{o}_n : vector<{VF}xf16>
      %{o}_own = vector.fmaf %{o}_c, %{o}_ss, %{o}_ns : vector<{VF}xf16>
      %{o}_own32 = vector.bitcast %{o}_own : vector<16xf16> to vector<8xf32>
"""
    for i in range(8):
        t += f"""      %{o}_e{i} = vector.extract %{o}_own32[{i}] : vector<8xf32> -> f32
      %{o}_o{i}, %{o}_u{i} = kernel.subgroup.shuffle<xor> %{o}_e{i}, %cq_x16, %cq_w32 : f32, i32, i32
"""
    t += f"""      %{o}_oth32 = vector.from_elements {", ".join(f"%{o}_o{i}" for i in range(8))} : vector<8xf32>
      %{o}_oth = vector.bitcast %{o}_oth32 : vector<8xf32> to vector<16xf16>
      %{o}_x = scf.select %cq_hi, %{o}_oth, %{o}_own : vector<16xf16>
      %{o}_y = scf.select %cq_hi, %{o}_own, %{o}_oth : vector<16xf16>"""
    return t


def grouped_gemm_cw_q4k(mode, N, K, E, KU, store16=False, arch="rdna3", tile=64, xch=False, fmt="q4_k"):
    """One kernel: `g_moe_g{gu,dn}_{fmt}_k{K}` (arch rdna3, gfx11 fragments) or its gfx12 twin `..._rdna4` (arch rdna4).
    fmt q5_k: the Q4_K header and code layout (codes at byte 48 of a 176-byte superblock) plus qh[32] at byte 16 (bit g of
    qh[l] = fifth bit of value l of group g); the chunk's qh words ride along with its code words (xch: Q4_K only)."""
    assert mode in ("swiglu", "store") and arch in ("rdna3", "rdna4") and fmt in ("q4_k", "q5_k"), (mode, arch, fmt)
    q5 = fmt == "q5_k"
    xch = xch and not q5
    assert K % 256 == 0 and tile == 64
    sw = mode == "swiglu"
    g12 = arch == "rdna4"
    VF = 8 if g12 else 16        # f16 values of a B / A fragment per lane
    VW = VF // 4                 # code words per lane per (group pair, K16)
    TN = 128 if sw else 256      # output columns per workgroup
    assert N % TN == 0, (N, TN)
    WF = [("w", 0), ("w2", 0)] if sw else [("w", 0), ("w", 16)]   # (weight buffer, column offset within the wave's columns)
    WCOLS = 16 if sw else 32
    BB = 176 if q5 else 144      # superblock bytes
    rb = K // 256 * BB
    eb = rb * N
    base = f"g_moe_g{'gu' if sw else 'dn'}_{fmt}_k{K}"
    name = base + ("_rdna4" if g12 else "")
    nch = K // KC
    max_rows = 8192 * KU + tile * E
    wargs = "%w: buffer, %w2: buffer" if sw else "%w: buffer"
    YT = "f16" if (store16 and not sw) else "f32"
    tgt = "target(@rdna4) " if g12 else ""
    L = [f'''kernel.def {tgt}export("{name}") @{name}(%rows: index, %n: index) {{
  %c1 = index.constant 1 : index
  %c64 = index.constant 64 : index
  %cTN = index.constant {TN} : index
  %c256 = index.constant 256 : index
  %row_tiles = index.div %rows, %c64 : index
  %out_tiles = index.div %n, %cTN : index
  kernel.launch.config workgroups(%out_tiles, %row_tiles, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%rows_arg: index, %k_arg: index, %n_arg: index, %xrows_arg: index, %tile_info: buffer, %rowmap: buffer, %x: buffer, %scales: buffer, {wargs}, %y: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 64, {max_rows}), multiple_of(%rows_arg, 64)] : index
  %k = index.assume %k_arg [range(%k_arg, {K}, {K})] : index
  %n = index.assume %n_arg [range(%n_arg, {N}, {N})] : index
  %origin = index.constant 0 : offset
  %xrows = index.assume %xrows_arg [range(%xrows_arg, 1, 1048576)] : index
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %c128 = index.constant 128 : index
  %c256 = index.constant 256 : index
  %cTN = index.constant {TN} : index
  %c0_i32 = scalar.constant 0 : i32
  %row_tile_count = index.div %rows, %c64 : index
  %out_tile_count = index.div %n, %cTN : index
  %row_group_id = kernel.workgroup.id<y> : index
  %out_group_id = kernel.workgroup.id<x> : index
  %row_group = index.assume %row_group_id [lt(%row_group_id, %row_tile_count)] : index
  %out_group = index.assume %out_group_id [lt(%out_group_id, %out_tile_count)] : index
  %ti_view = buffer.view %tile_info[%origin] : buffer -> view<[%row_tile_count]x2xi32>
  %tile_expert = view.load %ti_view[%row_group, 0] : view<[%row_tile_count]x2xi32> -> i32
  %tile_used = scalar.cmpi sge, %tile_expert, %c0_i32 : i32
  scf.if %tile_used {{
  %cEm1 = scalar.constant {E - 1} : i32
  %e_lo = scalar.minsi %tile_expert, %cEm1 : i32
  %e_b = scalar.assume %e_lo [range(%e_lo, 0, {E - 1})] : i32
  %expert = index.cast %e_b : i32 to index
  %expert_bytes = index.constant {eb} : offset
  %expert_off = index.scale %expert, %expert_bytes : index, offset -> offset
  %tile_rows32 = view.load %ti_view[%row_group, 1] : view<[%row_tile_count]x2xi32> -> i32
  %tile_rows_b = scalar.assume %tile_rows32 [range(%tile_rows32, 0, 64)] : i32
  %tile_rows = index.cast %tile_rows_b : i32 to index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, 255)] : index
  %subgroup = kernel.subgroup.id : index
  %wave = index.assume %subgroup [range(%subgroup, 0, 7)] : index
  %lane = kernel.subgroup.lane.id : index
  %lane16 = index.rem %lane, %c16 : index
  %cw_true = scalar.constant true : i1
  %cq_m63 = vector.splati 1061109567 : vector<2xi32>
  %cq_m15w = vector.splati 252645135 : vector<2xi32>
  %cq_m48 = vector.splati 808464432 : vector<2xi32>
  %cq_sh2 = vector.splati 2 : vector<2xi32>
  %cq_i0 = scalar.constant 0 : i32
  %cq_i4 = scalar.constant 4 : i32
  %cq_sh04 = vector.from_elements %cq_i0, %cq_i4 : vector<2xi32>
  %cq_m15 = vector.splati 252645135 : vector<{VW}xi32>
  %cq_f4 = vector.splati 4 : vector<{VW}xi32>
  %row_origin = index.mul %row_group, %c64 : index
  %out_origin = index.mul %out_group, %cTN : index
  %wcols = index.constant {WCOLS} : index
  %wave_col_v = index.mul %wave, %wcols : index
  %wave_col = index.assume %wave_col_v [range(%wave_col_v, 0, {TN - WCOLS})] : index
  %block_bytes = index.constant {BB} : offset
  %row_bytes = index.constant {rb} : offset
  %code_bytes = index.constant {48 if q5 else 16} : offset
  %half_bytes = index.constant 64 : offset''']
    if q5:
        L.append(f'''  %qh_bytes = index.constant 16 : offset
  %cq_m16 = vector.splati 269488144 : vector<{VW}xi32>
  %cq_s3 = vector.splati 3 : vector<{VW}xi32>
  %cq_s2 = vector.splati 2 : vector<{VW}xi32>
  %cq_s1 = vector.splati 1 : vector<{VW}xi32>''')
    xch = xch and not g12
    if xch:
        L.append(f'''  %cq_hi = index.cmp uge, %lane, %c16 : index
  %cq_l0 = vector.splati 0 : vector<{VW}xi32>
  %cq_l4 = vector.splati 4 : vector<{VW}xi32>
  %cq_lsh = scf.select %cq_hi, %cq_l4, %cq_l0 : vector<{VW}xi32>
  %cq_x16 = scalar.constant 16 : i32
  %cq_w32 = scalar.constant 32 : i32''')
    if g12:   # gfx12 B fragment: lanes 16-31 hold K 8-15 of the column, i.e. code words 2-3 of each 4-word K16 piece
        L.append('''  %cq_hi = index.cmp uge, %lane, %c16 : index
  %cq_hb8 = index.constant 8 : offset
  %cq_hb0 = index.constant 0 : offset
  %lane_code = scf.select %cq_hi, %cq_hb8, %cq_hb0 : offset''')
    L.append(f'''  %x_view = buffer.view %x[%origin] : buffer -> view<[%xrows]x[%k]xf16>
  %s_view = buffer.view %scales[%origin] : buffer -> view<[%xrows]x2xf32>
  %stage_bytes = index.constant {64 * (KC + 8) * 2} : offset
  %a_stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %a_view = buffer.view %a_stage[%origin] : buffer -> view<64x{KC + 8}xf16>
  %inv_bytes = index.constant 4096 : offset
  %inv_tile = buffer.alloca<workgroup> align(16) %inv_bytes : buffer
  %inv_view = buffer.view %inv_tile[%origin] : buffer -> view<64x16xf32>
  // thread t stages activation row t/4, K columns (t%4)*32 .. +32 of each chunk
  %stage_row = index.div %workitem, %c4 : index
  %stage_q = index.rem %workitem, %c4 : index
  %stage_col0 = index.mul %stage_q, %c32 : index
  %stage_col1 = index.add %stage_col0, %c16 : index
  %stage_tile_row = index.add %row_origin, %stage_row : index''')
    if sw:   # gather: activation row = token rowmap[a] / KU; padding rows (a < 0) are staged as zeros once
        L.append(f'''  %rm_rows = index.mul %row_tile_count, %c64 : index
  %rm_view = buffer.view %rowmap[%origin] : buffer -> view<[%rm_rows]xi32>
  %st_prow = index.assume %stage_tile_row [lt(%stage_tile_row, %rm_rows)] : index
  %st_a32 = view.load %rm_view[%st_prow] : view<[%rm_rows]xi32> -> i32
  %st_valid = scalar.cmpi sge, %st_a32, %c0_i32 : i32
  %st_ac = scalar.maxsi %st_a32, %c0_i32 : i32
  %st_ab = scalar.assume %st_ac [range(%st_ac, 0, 1048575)] : i32
  %st_a = index.cast %st_ab : i32 to index
  %cKU = index.constant {KU} : index
  %st_t = index.div %st_a, %cKU : index
  %x_row = index.assume %st_t [lt(%st_t, %xrows)] : index''')
    else:
        L.append('''  %x_row = index.assume %stage_tile_row [lt(%stage_tile_row, %xrows)] : index''')
    for i, (wb, co) in enumerate(WF):   # the lane's weight row (output column) and its expert byte origin
        L.append(f'''  %wc{i}_v = index.add %out_origin, %wave_col : index
  %wc{i}_l = index.add %wc{i}_v, %lane16 : index
  %wc{i}_o = index.add %wc{i}_l, %c{co} : index
  %wcol{i} = index.assume %wc{i}_o [lt(%wc{i}_o, %n)] : index
  %wro{i}_0 = index.scale %wcol{i}, %row_bytes : index, offset -> offset
  %wro{i} = index.add %wro{i}_0, %expert_off : offset''')

    NWV = 14 if g12 else 16   # code words in the lane's view of the chunk's 64 code bytes (gfx12: from its 8-byte lane offset)

    def wload(ch, sfx):
        """header + code vectors (group pair p, K16 half kk) of chunk `ch` (an index SSA name) for every weight fragment"""
        s = [f"    %sb{sfx} = index.div {ch}, %c2 : index", f"    %half{sfx} = index.rem {ch}, %c2 : index",
             f"    %sbo{sfx} = index.scale %sb{sfx}, %block_bytes : index, offset -> offset",
             f"    %hqo{sfx} = index.scale %half{sfx}, %half_bytes : index, offset -> offset"]
        outs = []
        for i, (wb, _) in enumerate(WF):
            s.append(f"""    %bo{i}{sfx} = index.add %wro{i}, %sbo{sfx} : offset
    %hv{i}{sfx} = buffer.view %{wb}[%bo{i}{sfx}] : buffer -> view<4xi32>
    %h{i}{sfx} = vector.load %hv{i}{sfx}[0] : view<4xi32> -> vector<4xi32>
    %qb{i}{sfx}_1 = index.add %bo{i}{sfx}, %code_bytes : offset
    %qb{i}{sfx}_2 = index.add %qb{i}{sfx}_1, %hqo{sfx} : offset""" + (f"""
    %qb{i}{sfx} = index.add %qb{i}{sfx}_2, %lane_code : offset""" if g12 else f"""
    %qb{i}{sfx} = index.add %qb{i}{sfx}_2, %origin : offset""") + f"""
    %qv{i}{sfx} = buffer.view %{wb}[%qb{i}{sfx}] : buffer -> view<{NWV}xi32>""")
            outs.append((f"%h{i}{sfx}", "vector<4xi32>"))
            for p in range(2):
                for kk in range(2):
                    s.append(f"    %q{i}{p}{kk}{sfx} = vector.load %qv{i}{sfx}[{p * 8 + kk * 4}] : view<{NWV}xi32> -> vector<{VW}xi32>")
                    outs.append((f"%q{i}{p}{kk}{sfx}", f"vector<{VW}xi32>"))
            if q5:   # qh[32]: K16 half kk = bytes 16kk .. +16 (gfx12: the lane's 8 of them)
                NHV = 6 if g12 else 8
                s.append(f"""    %hb{i}{sfx}_1 = index.add %bo{i}{sfx}, %qh_bytes : offset
    %hb{i}{sfx} = index.add %hb{i}{sfx}_1, {"%lane_code" if g12 else "%origin"} : offset
    %hqv{i}{sfx} = buffer.view %{wb}[%hb{i}{sfx}] : buffer -> view<{NHV}xi32>""")
                for kk in range(2):
                    s.append(f"    %qh{i}{kk}{sfx} = vector.load %hqv{i}{sfx}[{kk * 4}] : view<{NHV}xi32> -> vector<{VW}xi32>")
                    outs.append((f"%qh{i}{kk}{sfx}", f"vector<{VW}xi32>"))
        return "\n".join(s), outs

    def xload(ch, sfx):
        s = [f"    %xk{sfx} = index.mul {ch}, %c128 : index", f"    %xc0{sfx} = index.add %xk{sfx}, %stage_col0 : index",
             f"    %xc1{sfx} = index.add %xk{sfx}, %stage_col1 : index"]
        outs = []
        for j in range(2):
            s.append(f"    %xr{j}{sfx} = vector.load %x_view[%x_row, %xc{j}{sfx}] : view<[%xrows]x[%k]xf16> -> vector<16xf16>")
            outs.append((f"%xr{j}{sfx}", "vector<16xf16>"))
        return "\n".join(s), outs

    L.append(f'''  %zero16h = vector.splatf 0.0 : vector<16xf16>
  %zero8 = vector.splatf 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %nch = index.constant {nch} : index
  %ch_last = index.constant {nch - 1} : index
  // inverse row scales of the tile (result layout) for the epilogue; the K loop's first barrier publishes them
  %writer = index.cmp ult, %workitem, %c64 : index
  scf.if %writer {{
    %iv_tr = index.add %row_origin, %workitem : index''')
    if sw:
        L.append('''    %iv_prow = index.assume %iv_tr [lt(%iv_tr, %rm_rows)] : index
    %iv_a32 = view.load %rm_view[%iv_prow] : view<[%rm_rows]xi32> -> i32
    %iv_ac = scalar.maxsi %iv_a32, %c0_i32 : i32
    %iv_ab = scalar.assume %iv_ac [range(%iv_ac, 0, 1048575)] : i32
    %iv_a = index.cast %iv_ab : i32 to index
    %iv_t = index.div %iv_a, %cKU : index
    %iv_row = index.assume %iv_t [lt(%iv_t, %xrows)] : index''')
    else:
        L.append("    %iv_row = index.assume %iv_tr [lt(%iv_tr, %xrows)] : index")
    L.append(f'''    %inverse = view.load %s_view[%iv_row, %c1] : view<[%xrows]x2xf32> -> f32
    %inverse16 = vector.splat %inverse : vector<16xf32>
    %bw = index.assume %workitem [range(%workitem, 0, 63)] : index
    vector.store %inverse16, %inv_view[%bw, %c0] : vector<16xf32>, view<64x16xf32>
  }}
  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %tile_first = index.madd %row_origin, %n, %out_origin : index
  %y_esize = index.constant {2 if YT == "f16" else 4} : offset
  %tile_offset = index.scale %tile_first, %y_esize : index, offset -> offset
  %y_view = buffer.view %y[%tile_offset] : buffer -> view<64x{TN}x{YT}, %y_layout>''')

    def body(nf):
        """the K loop + epilogue for nf live 16-row fragments"""
        accs = [f"%acc{i}_{f}" for i in range(len(WF)) for f in range(nf)]
        s = [f"    // nf = {nf}", f"    %nfrows = index.constant {nf * 16} : index",
             "    %stage_live = index.cmp ult, %stage_row, %nfrows : index"]
        live = "%stage_live"
        if sw:   # padding rows of the live fragments: zeros once; valid rows are staged per chunk
            s.append(f"""    %zs_not = scalar.xori %st_valid, %cw_true : i1
    %zs_pad = scalar.andi %stage_live, %zs_not : i1
    scf.if %zs_pad {{
      vector.store %zero16h, %a_view[%stage_row, %stage_col0] : vector<16xf16>, view<64x{KC + 8}xf16>
      vector.store %zero16h, %a_view[%stage_row, %stage_col1] : vector<16xf16>, view<64x{KC + 8}xf16>
    }}
    %stage_ok = scalar.andi %stage_live, %st_valid : i1""")
            live = "%stage_ok"
        wt, wouts = wload("%c0", "p")
        xt, xouts = xload("%c0", "p")
        s += [wt, xt]
        carried = wouts + xouts
        args = ", ".join([f"{a} = %init : vector<8xf32>" for a in accs] +
                         [f"%cr{j} = {v} : {t}" for j, (v, t) in enumerate(carried)])
        types = ", ".join(["vector<8xf32>"] * len(accs) + [t for _, t in carried])
        res = [a.replace("%acc", "%fin") for a in accs] + [f"%unused{j}" for j in range(len(carried))]
        s.append(f"    {', '.join(res)} = scf.for %ch = [%c0 to %nch step %c1]({args}) -> ({types}) {{")
        cr = {v: f"%cr{j}" for j, (v, _) in enumerate(carried)}
        s.append(f"""    scf.if {live} {{
      vector.store {cr[xouts[0][0]]}, %a_view[%stage_row, %stage_col0] : vector<16xf16>, view<64x{KC + 8}xf16>
      vector.store {cr[xouts[1][0]]}, %a_view[%stage_row, %stage_col1] : vector<16xf16>, view<64x{KC + 8}xf16>
    }}
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    %ch1 = index.add %ch, %c1 : index
    %chm = index.min %ch1, %ch_last : index
    %chn = index.assume %chm [lt(%chm, %nch)] : index""")
        wt, wouts_n = wload("%chn", "n")
        xt, xouts_n = xload("%chn", "n")
        s += [wt, xt]
        s.append("    %half_c = index.rem %ch, %c2 : index\n    %sb_hi = index.cmp ne, %half_c, %c0 : index")
        for i in range(len(WF)):
            s.append(_meta(f"m{i}", cr[f"%h{i}p"]))
            if q5:   # upper chunk half (groups 4-7): their fifth bits are bits 4-7 of the qh bytes
                for kk in range(2):
                    s.append(f"""    %qr{i}{kk} = vector.shrui {cr[f"%qh{i}{kk}p"]}, %cq_f4 : vector<{VW}xi32>
    %qs{i}{kk} = scf.select %sb_hi, %qr{i}{kk}, {cr[f"%qh{i}{kk}p"]} : vector<{VW}xi32>""")
        cur = {a: a for a in accs}
        for g in range(4):
            for kk in range(2):
                s.append(f"    %fc{g}{kk} = index.constant {g * 32 + kk * 16} : index")
                for i in range(len(WF)):
                    if xch:
                        if g % 2 == 0:
                            s.append(_xch_values(f"d{i}{g // 2}{kk}", f"%m{i}_meta", cr[f"%q{i}{g // 2}{kk}p"], g // 2, VW, VF))
                        val = f"%d{i}{g // 2}{kk}_{'y' if g % 2 else 'x'}"
                    else:
                        s.append(_values(f"d{i}{g}{kk}", f"%m{i}_meta", cr[f"%q{i}{g // 2}{kk}p"], g, VW, VF,
                                         qh=f"%qs{i}{kk}" if q5 else None))
                        val = f"%d{i}{g}{kk}_v"
                    s.append(f"    %rhs{i}{g}{kk} = vector.fragment<rhs> {val} shape [%c16, %c16] : vector<{VF}xf16>")
                for f in range(nf):
                    s.append(f"    %fr{f}{g}{kk} = index.constant {f * 16} : index\n"
                             f"    %lhs{f}{g}{kk} = vector.fragment.load<lhs> %a_view[%fr{f}{g}{kk}, %fc{g}{kk}] shape [%c16, %c16] : view<64x{KC + 8}xf16> -> vector<{VF}xf16>")
                    for i in range(len(WF)):
                        a = f"%acc{i}_{f}"
                        nv = f"%m{i}_{f}_{g}{kk}"
                        s.append(f"    {nv} = vector.mma %lhs{f}{g}{kk}, %rhs{i}{g}{kk}, {cur[a]} : vector<{VF}xf16>, vector<{VF}xf16>, vector<8xf32>")
                        cur[a] = nv
        s.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        s.append(f"    scf.yield {', '.join([cur[a] for a in accs] + [v for v, _ in wouts_n + xouts_n])} : {types}\n    }}")
        for f in range(nf):   # epilogue: the 64-row kernel's (raw * inverse row scale; swiglu: silu(g) * u)
            s.append(f"    %er{f} = index.constant {f * 16} : index\n"
                     f"    %inv{f} = vector.fragment.load<result> %inv_view[%er{f}, %c0] shape [%c16, %c16] : view<64x16xf32> -> vector<8xf32>")
            if sw:
                s.append(f"""    %g{f} = vector.mulf %fin0_{f}, %inv{f} : vector<8xf32>
    %u{f} = vector.mulf %fin1_{f}, %inv{f} : vector<8xf32>
    %sg{f} = vector.siluf<afn> %g{f} : vector<8xf32>
    %act{f} = vector.mulf %sg{f}, %u{f} : vector<8xf32>
    vector.fragment.store<result> %act{f}, %y_view[%er{f}, %wave_col] shape [%c16, %c16] : vector<8xf32>, view<64x{TN}x{YT}, %y_layout>""")
            else:
                s.append(f"""    %o0_{f} = vector.mulf %fin0_{f}, %inv{f} : vector<8xf32>
    %o1_{f} = vector.mulf %fin1_{f}, %inv{f} : vector<8xf32>
    %wc16_{f} = index.add %wave_col, %c16 : index
    vector.fragment.store<result> %o0_{f}, %y_view[%er{f}, %wave_col] shape [%c16, %c16] : vector<8xf32>, view<64x{TN}x{YT}, %y_layout>
    vector.fragment.store<result> %o1_{f}, %y_view[%er{f}, %wc16_{f}] shape [%c16, %c16] : vector<8xf32>, view<64x{TN}x{YT}, %y_layout>""")
        return _suffix("\n".join(s), f"_n{nf}")

    # uniform dispatch on the live fragment count
    L.append("  %c48 = index.constant 48 : index")
    L.append("  %nf1 = index.cmp ule, %tile_rows, %c16 : index\n  scf.if %nf1 {\n" + body(1) + "\n  } else {")
    L.append("  %nf2 = index.cmp ule, %tile_rows, %c32 : index\n  scf.if %nf2 {\n" + body(2) + "\n  } else {")
    L.append("  %nf3 = index.cmp ule, %tile_rows, %c48 : index\n  scf.if %nf3 {\n" + body(3) + "\n  } else {\n" + body(4) + "\n  }\n  }\n  }")
    L.append("  }\n  kernel.return\n}\n")
    return "\n".join(L)


def _tokens():
    import os
    v = os.environ.get("MOE_CW_Q4K", "0").strip()
    return set() if v.lower() in ("", "0", "false", "no", "off", "none") else set(v.split(","))


def modes():
    """MOE_CW_Q4K: '1' both grouped GEMMs, or a comma list of swiglu,store (plus 'xch': the gfx11 half-wave exchange
    dequant); '0' / unset: off (the 64-row tile kernels)."""
    t = _tokens()
    return {"swiglu", "store"} if "1" in t else t & {"swiglu", "store"}


def xch():
    return "xch" in _tokens()


def applies(fmt, mode, N, K, fused=False, bias=False, act=None):
    """Whether grouped_gemm(fmt, mode, ...) takes the column-wave kernel (Q4_K or Q5_K experts, one format for gate and up)."""
    return (mode in modes() and fmt in ("q4_k", "q5_k") and not (fused or bias or act not in (None, "silu"))
            and K % 256 == 0 and N % (128 if mode == "swiglu" else 256) == 0)
