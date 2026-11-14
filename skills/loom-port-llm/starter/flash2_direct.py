"""Direct-load variant of the gfx12 head_dim 512 flash twin (gen_flash2s.py, FLASH2_DIRECT=1/2; gfx12 twin only, opt-in).

The twin stages every 16-key tile through workgroup memory: K [16 x D] (one 16-byte store per lane), V^T [D x 16] (sixteen
16-bit scatter stores per lane: the transpose), three workgroup barriers per tile. gfx12 has `global_load_tr_b128`, which
`vector.fragment.load<lhs|rhs>` emits for a view whose fragment k dimension is the strided one, so a wave can load its V^T
fragments (m = dim, k = key) straight from the paged KV arena (a [dim, key] view with strides [1, KVH*D]) with no staging, and
its K fragments (m = key, k = dim) as plain 16-byte global loads.

FLASH2_DIRECT=2: K and V^T direct. The only workgroup memory left is the S-partial exchange of the four waves of a head, double
buffered by key-tile parity, so a tile needs ONE barrier (a wave cannot reach the store of tile t + 2 before every wave has passed
the barrier of tile t + 1, i.e. finished reading tile t's partials).

Keys past the block's last written position (the partial kernel's last tile) are stale memory, possibly NaN: the staged path
wrote zeros for them; here K needs nothing (their S entries are masked by the causal select) and the V^T fragments are
zeroed per key (`%vok`) in the partial kernel (its hot copy drops the select: flash2_trim.hot).
"""
import re


LDS_LOADS = """{ind}%lf_k0 = vector.fragment.load<lhs> %k_g[%kb0_row, %lf_col0] shape [%c16, %c16] : view<16x{R}xf16, %k_g_layout> -> vector<8xf16>
{ind}%lf_k1 = vector.fragment.load<lhs> %k_g[%kb0_row, %lf_col1] shape [%c16, %c16] : view<16x{R}xf16, %k_g_layout> -> vector<8xf16>
{ind}%lf_v0 = vector.fragment.load<lhs> %v_g[%lf_col0, %kb0_row] shape [%c16, %c16] : view<{R}x16xf16, %vt_g_layout> -> vector<8xf16>
{ind}%lf_v1 = vector.fragment.load<lhs> %v_g[%lf_col1, %kb0_row] shape [%c16, %c16] : view<{R}x16xf16, %vt_g_layout> -> vector<8xf16>
{ind}vector.store %lf_k0, %kvl_view[%lf_w0, %lane, %c0] : vector<8xf16>, view<64x32x8xf16>
{ind}vector.store %lf_k1, %kvl_view[%lf_w1, %lane, %c0] : vector<8xf16>, view<64x32x8xf16>
{ind}vector.store %lf_v0, %kvl_view[%lf_wv0, %lane, %c0] : vector<8xf16>, view<64x32x8xf16>
{ind}vector.store %lf_v1, %kvl_view[%lf_wv1, %lane, %c0] : vector<8xf16>, view<64x32x8xf16>
{ind}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
"""


def direct(text, mode, kvh, d, partial, kwide=False):
    assert mode in (2, 3), 'FLASH2_DIRECT=2 (K and V^T direct) or 3 (direct, shared through workgroup memory)'
    lds = mode == 3
    R = kvh * d
    pos_bytes = int(re.search(r"%pos_bytes = index\.constant (\d+) : offset", text).group(1))
    HALF = int(__import__("os").environ.get("FLASH2_HALF", "128"))   # dims per wave
    m = re.search(r"^( *)scf\.if %st_live \{\n.*?^ *\}\n( *)kernel\.barrier<workgroup> scope\(workgroup\) ordering\(acq_rel\)\n", text, re.M | re.S)
    assert m, "flash2_direct: staging block not found"
    stage = m.group(0)
    call = re.search(r"%k_offset = func\.call @(\w+)\(%origins, %row, %kp, %blocks, %k_plane, %pool\) : \(buffer, index, index, index, index, index\) -> \(offset\)", stage)
    assert call, "flash2_direct: kv offset call not found"
    kvfn = call.group(1)
    ind = m.group(1)
    assert kvfn == "g_kv_offset", f"flash2_direct: the plain KV layout only (got {kvfn})"
    # The tile is page aligned: one page lookup, broadcast from lane 0 so the arena offset is wave-uniform (a per-lane value cannot be
    # the scalar base of a fragment load: its 64-bit range does not fit the 32-bit lane offset).
    kv_lines = f"""{ind}%kp = index.assume %tile_first [range(%tile_first, 0, 262143)] : index
{ind}%kh_pg_v = index.div %kp, %kh_c16 : index
{ind}%kh_pg = index.assume %kh_pg_v [range(%kh_pg_v, 0, %kh_last), lt(%kh_pg_v, %blocks)] : index
{ind}%kh_ph32r = view.load %kh_map[%kh_pg] : view<[%blocks]xi32> -> i32
{ind}%kh_ph32 = kernel.subgroup.broadcast %kh_ph32r from %kh_lane0 : i32, i32
{ind}%kh_phb = scalar.assume %kh_ph32 [range(%kh_ph32, 0, 65535)] : i32
{ind}%kh_ph = index.cast %kh_phb : i32 to index
{ind}%kh_pg16_v = index.mul %kh_ph, %kh_c16 : index
{ind}%krow_v = index.add %kh_pp, %kh_pg16_v : index
{ind}%vrow_v = index.add %krow_v, %pool : index
{ind}%krow_i = index.cast %krow_v : index to i32
{ind}%vrow_i = index.cast %vrow_v : index to i32
{ind}%krow_u = kernel.subgroup.broadcast %krow_i from %kh_lane0 : i32, i32
{ind}%vrow_u = kernel.subgroup.broadcast %vrow_i from %kh_lane0 : i32, i32
{ind}%krow_b = scalar.assume %krow_u [range(%krow_u, 0, 67108863)] : i32
{ind}%vrow_b = scalar.assume %vrow_u [range(%vrow_u, 0, 67108863)] : i32
{ind}%krow = index.cast %krow_b : i32 to index
{ind}%vrow = index.cast %vrow_b : i32 to index
{ind}%k_wbase_a = index.scale %krow, %pos_bytes : index, offset -> offset
{ind}%v_wbase_a = index.scale %vrow, %pos_bytes : index, offset -> offset
{ind}%k_wbase = index.add %k_wbase_a, %vd_boff : offset
{ind}%v_wbase = index.add %v_wbase_a, %vd_boff : offset
{ind}%v_g = buffer.view %arena[%v_wbase] : buffer -> view<{HALF}x16xf16, %vt_g_layout>
{ind}%k_g = buffer.view %arena[%k_wbase] : buffer -> view<16x{HALF}xf16, %k_g_layout>
"""
    if lds:
        kv_lines = kv_lines.replace("%vd_boff : offset", "%origin : offset").replace(f"view<{HALF}x16xf16", f"view<{R}x16xf16").replace(f"view<16x{HALF}xf16", f"view<16x{R}xf16")
        kv_lines += LDS_LOADS.replace("{ind}", ind).replace("{R}", str(R))
    text = text.replace(stage, kv_lines, 1)
    # allocas: no K / V^T tiles; the S exchange is double buffered
    text = re.sub(r"^ *%k_bytes = .*\n *%k_tile = .*\n *%k_view = .*\n", "", text, flags=re.M)
    text = re.sub(r"^ *%vt_bytes = .*\n *%vt_tile = .*\n *%vt_view = .*\n", "", text, flags=re.M)
    text = re.sub(r"^ *%vtf_view = .*\n *%vtf_c = .*\n", "", text, flags=re.M)
    sb = re.search(r"%s_bytes = index\.constant (\d+) : offset", text)
    s_bytes = int(sb.group(1))
    text = text.replace(sb.group(0), f"%s_bytes = index.constant {2 * s_bytes} : offset")
    text = re.sub(r"(%s_view = buffer\.view %s_tile\[%origin\] : buffer -> view<)(\d+)(x16xf32>)", lambda mm: f"{mm.group(1)}{2 * int(mm.group(2))}{mm.group(3)}", text)
    sx = re.search(r"%sx_view = buffer\.view %s_tile\[%origin\] : buffer -> view<(\d+)x32x8xf32>", text)
    assert sx, "flash2_direct: needs FLASH2_SX (the lane-vector S exchange)"
    waves = int(sx.group(1))
    text = text.replace(sx.group(0), f"%sx_view = buffer.view %s_tile[%origin] : buffer -> view<{2 * waves}x32x8xf32>")
    # the sliding / transposed global V^T layout, once before the key loop
    hdr = re.search(r"^( *)%m_final, %l_final, [^=]*= scf\.for %kt = ", text, re.M)
    hi = hdr.group(1)
    defs = (f"{hi}%vt_g_c1 = index.constant 1 : index\n{hi}%vt_g_crow = index.constant {R} : index\n"
            f"{hi}%vt_g_layout = encoding.layout.strided [%vt_g_c1, %vt_g_crow] : encoding<layout>\n"
            f"{hi}%k_g_layout = encoding.layout.strided [%vt_g_crow, %vt_g_c1] : encoding<layout>\n"
            f"{hi}%vd_c2 = index.constant 2 : index\n{hi}%vd_cw = index.constant {waves} : index\n{hi}%vd_cd = index.constant {d} : index\n"
            f"{hi}%vd_kvd = index.mul %kv, %vd_cd : index\n{hi}%vd_base = index.add %vd_kvd, %dim_base : index\n{hi}%vd_c2b = index.constant 2 : offset\n{hi}%vd_boff = index.scale %vd_base, %vd_c2b : index, offset -> offset\n"
            f"""{hi}%kh_c1 = index.constant 1 : index
{hi}%kh_c16 = index.constant 16 : index
{hi}%kh_lane0 = scalar.constant 0 : i32
{hi}%kh_rec_stride = index.constant 40 : offset
{hi}%kh_row = index.assume %row [range(%row, 0, 2147483646)] : index
{hi}%kh_rec_off = index.scale %kh_row, %kh_rec_stride : index, offset -> offset
{hi}%kh_records = buffer.view %origins[%kh_rec_off] : buffer -> view<5xi64>
{hi}%kh_map64 = view.load %kh_records[4] : view<5xi64> -> i64
{hi}%kh_map_b = scalar.assume %kh_map64 [range(%kh_map64, 0, 9223372036854775807)] : i64
{hi}%kh_map_origin = index.cast %kh_map_b : i64 to offset
{hi}%kh_map = buffer.view %origins[%kh_map_origin] : buffer -> view<[%blocks]xi32>
{hi}%kh_last = index.sub %blocks, %kh_c1 : index
{hi}%kh_pp = index.mul %k_plane, %pool : index
{hi}%kh_pagebytes64 = scalar.constant {16 * pos_bytes} : i64
{hi}%kh_ppo = index.scale %kh_pp, %pos_bytes : index, offset -> offset
{hi}%k_inv_o = index.add %kh_ppo, %vd_boff : offset
{hi}%v_inv_o = index.add %k_inv_o, %v_plane_delta : offset
{hi}%k_inv_i = index.cast %k_inv_o : offset to i64
{hi}%v_inv_i = index.cast %v_inv_o : offset to i64
""")
    text = text[:hdr.start()] + defs + text[hdr.start():]
    # parity of the tile selects the S exchange buffer; the one barrier is the one after the S partial store
    text = re.sub(r"^( *)%tf32 = index\.cast %tile_first : index to i32\n",
                  lambda mm: (mm.group(0) + f"{mm.group(1)}%par_i = index.rem %kt, %vd_c2 : index\n{mm.group(1)}%par_o = index.mul %par_i, %vd_cw : index\n"
                              f"{mm.group(1)}%wave_pv = index.add %wave, %par_o : index\n{mm.group(1)}%wave_p = index.assume %wave_pv [range(%wave_pv, 0, {2 * waves - 1})] : index\n"
                              + ("".join(f"{mm.group(1)}%sxp{j}_v = index.add %sx_hw{j if j else ''}, %par_o : index\n{mm.group(1)}%sxp{j} = index.assume %sxp{j}_v [range(%sxp{j}_v, 0, {2 * waves - 1})] : index\n" for j in range(4))
                                 if "%sx_hw " in text else f"{mm.group(1)}%sxpp_v = index.add %partner_id, %par_o : index\n{mm.group(1)}%sxpp = index.assume %sxpp_v [range(%sxpp_v, 0, {2 * waves - 1})] : index\n")), text, count=1, flags=re.M)
    text = text.replace("%sx_view[%wave, %lane, %c0]", "%sx_view[%wave_p, %lane, %c0]")
    text = text.replace("%sx_view[%partner_id, %lane, %c0]", "%sx_view[%sxpp, %lane, %c0]")
    for j in range(4):
        text = re.sub(rf"(%sp\d+_{j}(?:_pt)?) = vector\.load %sx_view\[%sx_hw{j if j else ''}, %lane, %c0\] : view<{waves}x32x8xf32>",
                      lambda mm: f"{mm.group(1)} = vector.load %sx_view[%sxp{j}, %lane, %c0] : view<{2 * waves}x32x8xf32>", text)
    text = text.replace(f"view<{waves}x32x8xf32>", f"view<{2 * waves}x32x8xf32>")
    # K fragments (S = K Q^T) from the arena, V^T fragments from the transposed arena view
    text = re.sub(r"^( *)(%kc\d+_\d+) = index\.add %dim_base, (%qd\d+_o) : index$", r"\1\2 = index.add \3, %c0 : index", text, flags=re.M)
    text = re.sub(r"%k_view\[(%kb\d+_row), (%kc\d+_\d+)\] shape \[%c16, %c16\] : view<\d+x\d+xf16>", rf"%k_g[\1, \2] shape [%c16, %c16] : view<16x{HALF}xf16, %k_g_layout>", text)
    assert not kwide, "wide K load: not ported to the page-coordinate views"
    if kwide:
        # K fragments from ONE 128-byte load per lane (eight 16-byte loads off one address): lane (key row l % 16, half h) holds dims
        # 64 h + 8 c + j of the wave's 128 as fragment c's k slots. The dims of a fragment's k axis are any bijection as long as Q^T uses
        # the same one (the products are summed over k), so Q^T's fragment c lane half h holds those dims too.
        text = text.replace("%lane_h8 = index.assume", "%kw_c64 = index.constant 64 : index\n  %kw_h64_v = index.mul %lane_half, %kw_c64 : index\n  %kw_h64 = index.assume %kw_h64_v [range(%kw_h64_v, 0, 64)] : index\n  %lane_h8 = index.assume", 1)
        text, nq = re.subn(r"^( *)(%qdh(\d+)_v) = index\.add %qd\d+, %lane_h8 : index$",
                           lambda mm: f"{mm.group(1)}%kw_o{mm.group(3)} = index.constant {8 * int(mm.group(3))} : index\n{mm.group(1)}%kw_a{mm.group(3)} = index.add %kw_h64, %kw_o{mm.group(3)} : index\n{mm.group(1)}{mm.group(2)} = index.add %dim_base, %kw_a{mm.group(3)} : index", text, flags=re.M)
        assert nq == 8, nq
        def kw(mm):
            ind, nm, kb, c, view = mm.group(1), mm.group(2), mm.group(3), int(mm.group(4)), mm.group(5)
            out = ""
            if c == 0:
                out += f"{ind}%kraw{kb}{mm.group(6)} = vector.load {view}[%kh_pg_lane, %kw_h64] : view<16x{HALF}xf16, %k_g_layout> -> vector<64xf16>\n"
            out += f"{ind}%ks{kb}_{c}{mm.group(6)} = vector.slice %kraw{kb}{mm.group(6)}[{8 * c}] : vector<64xf16> -> vector<8xf16>\n"
            out += f"{ind}{nm} = vector.fragment<lhs> %ks{kb}_{c}{mm.group(6)} shape [%c16, %c16] : vector<8xf16>"
            return out
        text, nk = re.subn(r"^( *)(%kl(\d+)_(\d+)(_pt)?) = vector\.fragment\.load<lhs> (%k_g(?:_pt)?)\[%kb\d+_row, %kc\d+_\d+(?:_pt)?\] shape \[%c16, %c16\] : view<16x\d+xf16, %k_g_layout> -> vector<8xf16>$", lambda mm: kw(type("M", (), {"group": lambda self, i, mm=mm: {1: mm.group(1), 2: mm.group(2), 3: mm.group(3), 4: mm.group(4), 5: mm.group(6), 6: mm.group(5) or ""}[i]})()), text, flags=re.M)
        assert nk >= 8, nk
    text = re.sub(r"^( *)(%vr\d+) = index\.add %dim_base, (%qd\d+_o) : index$", r"\1\2 = index.add \3, %c0 : index", text, flags=re.M)
    text = re.sub(r"%vt_view\[(%vr\d+), (%kb\d+_row)\] shape \[%c16, %c16\] : view<\d+x\d+xf16>", rf"%v_g[\1, \2] shape [%c16, %c16] : view<{HALF}x16xf16, %vt_g_layout>", text)
    # the third barrier (end of the tile) goes: the double-buffered exchange makes it redundant. Exactly the last one of the loop body.
    # barriers: [after staging (removed with the staging block)], [after the S store: kept], [loop end]
    ends = [mm for mm in re.finditer(r"^ *kernel\.barrier<workgroup> scope\(workgroup\) ordering\(acq_rel\)\n( *)scf\.yield %m_new, %l_new, ", text, re.M)]
    assert len(ends) == 1, len(ends)
    text = text[:ends[0].start()] + ends[0].group(0).split("\n", 1)[1] + text[ends[0].end():]
    if lds:
        # K / V^T fragments: each wave loads two fragments of its dim slice from the arena (the four head waves of a slice split its eight),
        # stores them in lane layout, and every wave reads the slice's eight back after the barrier (no per-wave global address work).
        text = text.replace("  %s_tile = buffer.alloca<workgroup>", "  %kvl_bytes = index.constant 32768 : offset\n  %kvl_tile = buffer.alloca<workgroup> align(16) %kvl_bytes : buffer\n  %kvl_view = buffer.view %kvl_tile[%origin] : buffer -> view<64x32x8xf16>\n  %s_tile = buffer.alloca<workgroup>", 1)
        pre = ("{hi}%lf_c8 = index.constant 8 : index\n{hi}%lf_c32 = index.constant 32 : index\n{hi}%lf_kb = index.mul %dim_half, %lf_c8 : index\n"
               "{hi}%lf_2h = index.mul %head_in_group, %c2 : index\n{hi}%lf_w0_v = index.add %lf_kb, %lf_2h : index\n"
               "{hi}%lf_w0 = index.assume %lf_w0_v [range(%lf_w0_v, 0, 62)] : index\n{hi}%lf_w1_v = index.add %lf_w0, %c1 : index\n"
               "{hi}%lf_w1 = index.assume %lf_w1_v [range(%lf_w1_v, 0, 63)] : index\n{hi}%lf_wv0_v = index.add %lf_w0, %lf_c32 : index\n"
               "{hi}%lf_wv0 = index.assume %lf_wv0_v [range(%lf_wv0_v, 0, 63)] : index\n{hi}%lf_wv1_v = index.add %lf_w1, %lf_c32 : index\n"
               "{hi}%lf_wv1 = index.assume %lf_wv1_v [range(%lf_wv1_v, 0, 63)] : index\n"
               "{hi}%lf_h32 = index.mul %head_in_group, %lf_c32 : index\n{hi}%lf_col0_v = index.add %vd_base, %lf_h32 : index\n"
               "{hi}%lf_col0 = index.assume %lf_col0_v [range(%lf_col0_v, 0, " + str(R - 32) + ")] : index\n{hi}%lf_col1_v = index.add %lf_col0, %c16 : index\n"
               "{hi}%lf_col1 = index.assume %lf_col1_v [range(%lf_col1_v, 0, " + str(R - 16) + ")] : index\n").replace("{hi}", hi)
        for c in range(8):
            pre += (f"{hi}%lf_kc{c}_o = index.add %lf_kb, %e{c} : index\n{hi}%lf_ks{c} = index.assume %lf_kc{c}_o [range(%lf_kc{c}_o, 0, 63)] : index\n"
                    f"{hi}%lf_vc{c}_o = index.add %lf_ks{c}, %lf_c32 : index\n{hi}%lf_vs{c} = index.assume %lf_vc{c}_o [range(%lf_vc{c}_o, 0, 63)] : index\n")
        hdr2 = re.search(r"^( *)%m_final, %l_final, [^=]*= scf\.for %kt = ", text, re.M)
        text = text[:hdr2.start()] + pre + text[hdr2.start():]
        def klds(mm):
            ind_, nm, c = mm.group(1), mm.group(2), int(mm.group(3))
            return (f"{ind_}%{nm[1:]}_raw = vector.load %kvl_view[%lf_ks{c}, %lane, %c0] : view<64x32x8xf16> -> vector<8xf16>\n"
                    f"{ind_}{nm} = vector.fragment<lhs> %{nm[1:]}_raw shape [%c16, %c16] : vector<8xf16>")
        text, nk = re.subn(r"^( *)(%kl0_(\d+)) = vector\.fragment\.load<lhs> %k_g\[[^\n]*$", klds, text, flags=re.M)
        assert nk == 8, nk
        # V^T fragments are read right after the S store (before the exchange barrier) and held through the softmax
        vr = "".join(f"%vrw{c} = vector.load %kvl_view[%lf_vs{c}, %lane, %c0] : view<64x32x8xf16> -> vector<8xf16>\n    " for c in range(8))
        text, ns = re.subn(r"^( *)(vector\.store %s0_7, %sx_view\[%wave_p, %lane, %c0\] : [^\n]*\n)", lambda mm: mm.group(0) + "".join(f"{mm.group(1)}%vrw{c} = vector.load %kvl_view[%lf_vs{c}, %lane, %c0] : view<64x32x8xf16> -> vector<8xf16>\n" for c in range(8)), text, flags=re.M)
        assert ns == 1, ns
        text, nv = re.subn(r"^( *)(%vl(\d+)_0) = vector\.fragment\.load<lhs> %v_g\[[^\n]*$", lambda mm: f"{mm.group(1)}{mm.group(2)} = vector.fragment<lhs> %vrw{mm.group(3)} shape [%c16, %c16] : vector<8xf16>", text, flags=re.M)
        assert nv >= 8, nv
    if partial:   # stale keys: V^T of keys past the last written position is zeroed (see the module doc)
        text = re.sub(r"^( *)(%iota16 = vector\.iota [^\n]*\n)",
                      lambda mm: mm.group(0) + f"{mm.group(1)}%vk_base_v = vector.splat %tf32 : vector<8xi32>\n{mm.group(1)}%vk_pos = vector.addi %vk_base_v, %iota16 : vector<8xi32>\n"
                                 f"{mm.group(1)}%vk_hi_v = vector.splat %pos_hi_i32 : vector<8xi32>\n{mm.group(1)}%vok = vector.cmpi sle, %vk_pos, %vk_hi_v : vector<8xi32> -> vector<8xi1>\n", text, flags=re.M)
        text = re.sub(r"^( *)(%vl(\d+_\d+)) = (vector\.fragment\.load<lhs> %v_g\[[^\n]*)$" if not lds else r"^( *)(%vl(\d+_\d+)) = (vector\.fragment<lhs> %vrw\d+ [^\n]*)$",
                      lambda mm: (f"{mm.group(1)}%vlraw{mm.group(3)} = {mm.group(4)}\n{mm.group(1)}%vlsel{mm.group(3)} = vector.select %vok, %vlraw{mm.group(3)}, %zero16h : vector<8xf16>\n"
                                  f"{mm.group(1)}{mm.group(2)} = vector.fragment<lhs> %vlsel{mm.group(3)} shape [%c16, %c16] : vector<8xf16>"), text, flags=re.M)
    return text
