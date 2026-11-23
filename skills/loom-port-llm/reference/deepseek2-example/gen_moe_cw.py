"""MOE_CW (choice moe.prefill.cw): column-wave grouped MoE GEMMs for IQ4_XS experts (drop-in for gen_moe.grouped_gemm's
g_moe_ggu_* / g_moe_gdn_*: same name, launch arguments and bucketed 64-row expert tiles).

Why: at a 1k-4k chunk an expert holds ~32-128 rows, so a 64-row tile is often half padding, and the 64x64 tile kernel stages
weights through LDS (dequant by half the threads, 2 barriers per 64 K). Here each wave owns output columns instead of rows:
  - every lane dequantizes its own column's IQ4_XS 32-groups from global straight into WMMA B fragments (no weight LDS
    stage: weights are used by exactly one wave);
  - the activation rows (gathered x16 rows for gate/up, the sorted rows for down) are staged in LDS, 128 K per barrier pair,
    shared by all 8 waves;
  - only the tile's live 16-row fragments get MMAs (nf = ceil(rows/16) from tile_info, a uniform branch into a copy of the
    K loop specialized for nf = 1..4), so the WMMA work follows the real rows instead of the 64-row tile.
swiglu: wave w owns gate and up columns out*128 + 16w .. +16 (128-column tiles); store: out*256 + 32w .. +32 (256-column tiles).
Values: the dequant is _g32_iq4's (iq4_f16lut values), the MMAs accumulate K in the same 16-wide order and the epilogue is the
64-row kernel's, so live rows are bit-identical to GEMM_T64_G32; rows of skipped fragments are not written (never combined)."""

KV = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
KC = 128   # K per staged chunk
import os
# MOE_CW_SWAP=1: lanes 0-15 dequantize a group's low nibbles (K 0-15), lanes 16-31 the high nibbles (K 16-31) of the same
# column, and an xor-16 shuffle swaps the halves (WMMA B wants both half-waves to hold the same 16 values): half the dequant.
SWAP = os.environ.get("MOE_CW_SWAP", "0") not in ("", "0")


PLAIN_TAIL = '      %{o}_lo_i = vector.andi {packed}, %{o}_m15 : vector<16xi8>\n      %{o}_hn_s = vector.shrui {packed}, %{o}_f4 : vector<16xi8>\n      %{o}_hi_i = vector.andi %{o}_hn_s, %{o}_m15 : vector<16xi8>\n      %{o}_lo_u = vector.table.lookup %{o}_book[%{o}_lo_i] : vector<16xi8>, vector<16xi8> -> vector<16xi8>\n      %{o}_hi_u = vector.table.lookup %{o}_book[%{o}_hi_i] : vector<16xi8>, vector<16xi8> -> vector<16xi8>\n      %{o}_lo_f = vector.uitofp %{o}_lo_u : vector<16xi8> to vector<16xf16>\n      %{o}_hi_f = vector.uitofp %{o}_hi_u : vector<16xi8> to vector<16xf16>\n      %{o}_m128 = scalar.constant -128.0 : f32\n      %{o}_off32 = scalar.mulf %{o}_dlh32, %{o}_m128 : f32\n      %{o}_off_h = scalar.fptrunc %{o}_off32 : f32 to f16\n      %{o}_scales = vector.splat %{o}_dlh : vector<16xf16>\n      %{o}_offsets = vector.splat %{o}_off_h : vector<16xf16>\n      %{o}_v0 = vector.fmaf %{o}_lo_f, %{o}_scales, %{o}_offsets : vector<16xf16>\n      %{o}_v1 = vector.fmaf %{o}_hi_f, %{o}_scales, %{o}_offsets : vector<16xf16>\n'
SWAP_TAIL = '      %{o}_shc = vector.shrui {packed}, %{o}_f4 : vector<16xi8>\n      %{o}_sh = scf.select %cw_hi, %{o}_shc, {packed} : vector<16xi8>\n      %{o}_ni = vector.andi %{o}_sh, %{o}_m15 : vector<16xi8>\n      %{o}_nu = vector.table.lookup %{o}_book[%{o}_ni] : vector<16xi8>, vector<16xi8> -> vector<16xi8>\n      %{o}_nf = vector.uitofp %{o}_nu : vector<16xi8> to vector<16xf16>\n      %{o}_m128 = scalar.constant -128.0 : f32\n      %{o}_off32 = scalar.mulf %{o}_dlh32, %{o}_m128 : f32\n      %{o}_off_h = scalar.fptrunc %{o}_off32 : f32 to f16\n      %{o}_scales = vector.splat %{o}_dlh : vector<16xf16>\n      %{o}_offsets = vector.splat %{o}_off_h : vector<16xf16>\n      %{o}_own = vector.fmaf %{o}_nf, %{o}_scales, %{o}_offsets : vector<16xf16>\n      %{o}_own32 = vector.bitcast %{o}_own : vector<16xf16> to vector<8xf32>\n      %{o}_e0 = vector.extract %{o}_own32[0] : vector<8xf32> -> f32\n      %{o}_x0, %{o}_xu0 = kernel.subgroup.shuffle<xor> %{o}_e0, %cw_x16, %cw_w32 : f32, i32, i32\n      %{o}_e1 = vector.extract %{o}_own32[1] : vector<8xf32> -> f32\n      %{o}_x1, %{o}_xu1 = kernel.subgroup.shuffle<xor> %{o}_e1, %cw_x16, %cw_w32 : f32, i32, i32\n      %{o}_e2 = vector.extract %{o}_own32[2] : vector<8xf32> -> f32\n      %{o}_x2, %{o}_xu2 = kernel.subgroup.shuffle<xor> %{o}_e2, %cw_x16, %cw_w32 : f32, i32, i32\n      %{o}_e3 = vector.extract %{o}_own32[3] : vector<8xf32> -> f32\n      %{o}_x3, %{o}_xu3 = kernel.subgroup.shuffle<xor> %{o}_e3, %cw_x16, %cw_w32 : f32, i32, i32\n      %{o}_e4 = vector.extract %{o}_own32[4] : vector<8xf32> -> f32\n      %{o}_x4, %{o}_xu4 = kernel.subgroup.shuffle<xor> %{o}_e4, %cw_x16, %cw_w32 : f32, i32, i32\n      %{o}_e5 = vector.extract %{o}_own32[5] : vector<8xf32> -> f32\n      %{o}_x5, %{o}_xu5 = kernel.subgroup.shuffle<xor> %{o}_e5, %cw_x16, %cw_w32 : f32, i32, i32\n      %{o}_e6 = vector.extract %{o}_own32[6] : vector<8xf32> -> f32\n      %{o}_x6, %{o}_xu6 = kernel.subgroup.shuffle<xor> %{o}_e6, %cw_x16, %cw_w32 : f32, i32, i32\n      %{o}_e7 = vector.extract %{o}_own32[7] : vector<8xf32> -> f32\n      %{o}_x7, %{o}_xu7 = kernel.subgroup.shuffle<xor> %{o}_e7, %cw_x16, %cw_w32 : f32, i32, i32\n      %{o}_oth32 = vector.from_elements %{o}_x0, %{o}_x1, %{o}_x2, %{o}_x3, %{o}_x4, %{o}_x5, %{o}_x6, %{o}_x7 : vector<8xf32>\n      %{o}_oth = vector.bitcast %{o}_oth32 : vector<8xf32> to vector<16xf16>\n      %{o}_v0 = scf.select %cw_hi, %{o}_oth, %{o}_own : vector<16xf16>\n      %{o}_v1 = scf.select %cw_hi, %{o}_own, %{o}_oth : vector<16xf16>\n'


def _dequant(o, hdr, packed, grp, swap=False):
    """%{o}_v0 / %{o}_v1: weights 0-15 / 16-31 of 32-group `grp` (an index SSA name, 0..7 in the superblock) as f16."""
    book = "\n".join(f"      %{o}_k{i} = scalar.constant {v + 128 - (256 if v + 128 > 127 else 0)} : i8" for i, v in enumerate(KV))
    return f"""      %{o}_halves = vector.bitcast {hdr} : vector<2xi32> to vector<4xf16>
      %{o}_d_half = vector.extract %{o}_halves[0] : vector<4xf16> -> f16
      %{o}_d = scalar.extf %{o}_d_half : f16 to f32
      %{o}_word0 = vector.extract {hdr}[0] : vector<2xi32> -> i32
      %{o}_word1 = vector.extract {hdr}[1] : vector<2xi32> -> i32
      %{o}_i16 = scalar.constant 16 : i32
      %{o}_i15 = scalar.constant 15 : i32
      %{o}_i3 = scalar.constant 3 : i32
      %{o}_i4 = scalar.constant 4 : i32
      %{o}_i2 = scalar.constant 2 : i32
      %{o}_i32 = scalar.constant 32 : i32
      %{o}_g32 = index.cast {grp} : index to i32
      %{o}_scales_h = scalar.shrui %{o}_word0, %{o}_i16 : i32
      %{o}_lo_shift = scalar.muli %{o}_g32, %{o}_i4 : i32
      %{o}_hi_shift = scalar.muli %{o}_g32, %{o}_i2 : i32
      %{o}_lo_s = scalar.shrui %{o}_word1, %{o}_lo_shift : i32
      %{o}_lo = scalar.andi %{o}_lo_s, %{o}_i15 : i32
      %{o}_hi_s = scalar.shrui %{o}_scales_h, %{o}_hi_shift : i32
      %{o}_hi_m = scalar.andi %{o}_hi_s, %{o}_i3 : i32
      %{o}_hi = scalar.shli %{o}_hi_m, %{o}_i4 : i32
      %{o}_ls = scalar.ori %{o}_lo, %{o}_hi : i32
      %{o}_centered = scalar.subi %{o}_ls, %{o}_i32 : i32
      %{o}_lf = scalar.sitofp %{o}_centered : i32 to f32
      %{o}_dl = scalar.mulf %{o}_d, %{o}_lf : f32
      %{o}_dlh = scalar.fptrunc %{o}_dl : f32 to f16
      %{o}_dlh32 = scalar.extf %{o}_dlh : f16 to f32
{book}
      %{o}_book = vector.from_elements {", ".join(f"%{o}_k{i}" for i in range(16))} : vector<16xi8>
      %{o}_m15 = vector.splati 15 : vector<16xi8>
      %{o}_f4 = vector.splati 4 : vector<16xi8>
{(SWAP_TAIL if swap else PLAIN_TAIL).format(o=o, packed=packed)}"""


def _suffix(text, sfx):
    """rename every SSA value defined in `text` (one specialized loop body) with suffix `sfx`, so sibling bodies do not collide"""
    import re
    names = set(re.findall(r"%([A-Za-z_]\w*)\s*(?=(?:,\s*%[A-Za-z_]\w*\s*)*=)", text))
    names |= set(re.findall(r"scf\.for %([A-Za-z_]\w*)", text))
    names |= set(re.findall(r"[(,] ?%([A-Za-z_]\w*) = %", text))
    return re.sub(r"%([A-Za-z_]\w*)\b", lambda m: f"%{m.group(1)}{sfx}" if m.group(1) in names else m.group(0), text)


def grouped_gemm_cw(fmt, mode, N, K, E, KU, tile=64):
    assert fmt == "iq4_xs" and mode in ("swiglu", "store"), (fmt, mode)
    assert K % 256 == 0 and tile == 64
    sw = mode == "swiglu"
    TN = 128 if sw else 256          # output columns per workgroup
    assert N % TN == 0, (N, TN)
    # weight fragments per wave: (name of the weight buffer, column offset within the wave's columns)
    WF = [("w", 0), ("w2", 0)] if sw else [("w", 0), ("w", 16)]
    WCOLS = 16 if sw else 32
    rb = K // 256 * 136
    eb = rb * N
    name = f"g_moe_g{'gu' if sw else 'dn'}_{fmt}_k{K}"
    nch = K // KC
    max_rows = 8192 * KU + tile * E
    wargs = "%w: buffer, %w2: buffer" if sw else "%w: buffer"
    L = [f'''kernel.def export("{name}") @{name}(%rows: index, %n: index) {{
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
  %c8 = index.constant 8 : index
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
  %cw_hi = index.cmp uge, %lane, %c16 : index
  %cw_s0 = vector.splati 0 : vector<16xi8>
  %cw_s4 = vector.splati 4 : vector<16xi8>
  %cw_hshift = scf.select %cw_hi, %cw_s4, %cw_s0 : vector<16xi8>
  %cw_x16 = scalar.constant 16 : i32
  %cw_w32 = scalar.constant 32 : i32
  %row_origin = index.mul %row_group, %c64 : index
  %out_origin = index.mul %out_group, %cTN : index
  %wcols = index.constant {WCOLS} : index
  %wave_col_v = index.mul %wave, %wcols : index
  %wave_col = index.assume %wave_col_v [range(%wave_col_v, 0, {TN - WCOLS})] : index
  %block_bytes = index.constant 136 : offset
  %row_bytes = index.constant {rb} : offset
  %x_view = buffer.view %x[%origin] : buffer -> view<[%xrows]x[%k]xf16>
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
  %stage_tile_row = index.add %row_origin, %stage_row : index''']
    if sw:   # gather: activation row = token rowmap[a] / KU; padding rows (a < 0) stage zeros
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
    # the wave's weight rows (output columns) and their expert byte origins
    for i, (wb, co) in enumerate(WF):
        L.append(f'''  %wc{i}_v = index.add %out_origin, %wave_col : index
  %wc{i}_l = index.add %wc{i}_v, %lane16 : index
  %wc{i}_o = index.add %wc{i}_l, %c{co if co else 0} : index
  %wcol{i} = index.assume %wc{i}_o [lt(%wc{i}_o, %n)] : index
  %wro{i}_0 = index.scale %wcol{i}, %row_bytes : index, offset -> offset
  %wro{i} = index.add %wro{i}_0, %expert_off : offset''')

    def wload(ch, sfx):
        """header + 4 group code vectors of chunk `ch` (an index SSA name) for every weight fragment"""
        s = [f"    %sb{sfx} = index.div {ch}, %c2 : index", f"    %half{sfx} = index.rem {ch}, %c2 : index",
             f"    %sbo{sfx} = index.scale %sb{sfx}, %block_bytes : index, offset -> offset",
             f"    %hq{sfx} = index.constant 64 : offset", f"    %hqo{sfx} = index.scale %half{sfx}, %hq{sfx} : index, offset -> offset"]
        outs = []
        for i, (wb, _) in enumerate(WF):
            s.append(f"""    %bo{i}{sfx} = index.add %wro{i}, %sbo{sfx} : offset
    %hv{i}{sfx} = buffer.view %{wb}[%bo{i}{sfx}] : buffer -> view<2xi32>
    %h{i}{sfx} = vector.load %hv{i}{sfx}[0] : view<2xi32> -> vector<2xi32>
    %qb{i}{sfx}_0 = index.constant 8 : offset
    %qb{i}{sfx}_1 = index.add %bo{i}{sfx}, %qb{i}{sfx}_0 : offset
    %qb{i}{sfx} = index.add %qb{i}{sfx}_1, %hqo{sfx} : offset
    %qv{i}{sfx} = buffer.view %{wb}[%qb{i}{sfx}] : buffer -> view<64xi8>""")
            outs.append((f"%h{i}{sfx}", "vector<2xi32>"))
            for g in range(4):
                s.append(f"    %q{i}{g}{sfx} = vector.load %qv{i}{sfx}[{g * 16}] : view<64xi8> -> vector<16xi8>")
                outs.append((f"%q{i}{g}{sfx}", "vector<16xi8>"))
        return "\n".join(s), outs

    def xload(ch, sfx):
        s = [f"    %xk{sfx} = index.mul {ch}, %c128 : index", f"    %xc0{sfx} = index.add %xk{sfx}, %stage_col0 : index",
             f"    %xc1{sfx} = index.add %xk{sfx}, %stage_col1 : index"]
        outs = []
        for j in range(2):
            s.append(f"    %xr{j}{sfx} = vector.load %x_view[%x_row, %xc{j}{sfx}] : view<[%xrows]x[%k]xf16> -> vector<16xf16>")
            if sw:
                s.append(f"    %xs{j}{sfx} = scf.select %st_valid, %xr{j}{sfx}, %zero16h : vector<16xf16>")
                outs.append((f"%xs{j}{sfx}", "vector<16xf16>"))
            else:
                outs.append((f"%xr{j}{sfx}", "vector<16xf16>"))
        return "\n".join(s), outs

    L.append(f'''  %zero16h = vector.splatf 0.0 : vector<16xf16>
  %zero8 = vector.splatf 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %nch = index.constant {nch} : index
  %ch_last = index.constant {nch - 1} : index
  // inverse row scales of the tile (result layout) for the epilogue
  %writer = index.cmp ult, %workitem, %c64 : index
  scf.if %writer {{
    %iv_tr = index.add %row_origin, %workitem : index''')
    if sw:
        L.append(f'''    %iv_prow = index.assume %iv_tr [lt(%iv_tr, %rm_rows)] : index
    %iv_a32 = view.load %rm_view[%iv_prow] : view<[%rm_rows]xi32> -> i32
    %iv_ac = scalar.maxsi %iv_a32, %c0_i32 : i32
    %iv_ab = scalar.assume %iv_ac [range(%iv_ac, 0, 1048575)] : i32
    %iv_a = index.cast %iv_ab : i32 to index
    %iv_t = index.div %iv_a, %cKU : index
    %iv_row = index.assume %iv_t [lt(%iv_t, %xrows)] : index''')
    else:
        L.append("    %iv_row = index.assume %iv_tr [lt(%iv_tr, %xrows)] : index")
    L.append('''    %inverse = view.load %s_view[%iv_row, %c1] : view<[%xrows]x2xf32> -> f32
    %inverse16 = vector.splat %inverse : vector<16xf32>
    %bw = index.assume %workitem [range(%workitem, 0, 63)] : index
    vector.store %inverse16, %inv_view[%bw, %c0] : vector<16xf32>, view<64x16xf32>
  }
  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %tile_first = index.madd %row_origin, %n, %out_origin : index
  %four = index.constant 4 : offset
  %tile_offset = index.scale %tile_first, %four : index, offset -> offset''')
    L.append(f"  %y_view = buffer.view %y[%tile_offset] : buffer -> view<64x{TN}xf32, %y_layout>")

    def body(nf):
        """the K loop + epilogue for nf live 16-row fragments"""
        accs = [f"%acc{i}_{f}" for i in range(len(WF)) for f in range(nf)]
        s = [f"    // nf = {nf}"]
        s.append(f"    %stage_live = index.cmp ult, %stage_row, %c{nf * 16 if nf * 16 in (16, 32, 64) else 0} : index" if nf * 16 in (16, 32, 64)
                 else f"    %nfrows{nf} = index.constant {nf * 16} : index\n    %stage_live = index.cmp ult, %stage_row, %nfrows{nf} : index")
        wt, wouts = wload("%c0", "p")
        xt, xouts = xload("%c0", "p")
        s += [wt, xt]
        carried = wouts + xouts
        args = ", ".join([f"{a} = %init : vector<8xf32>" for a in accs] +
                         [f"%cr{j} = {v} : {t}" for j, (v, t) in enumerate(carried)])
        types = ", ".join(["vector<8xf32>"] * len(accs) + [t for _, t in carried])
        res = [a.replace("%acc", "%fin") for a in accs] + [f"%unused{nf}_{j}" for j in range(len(carried))]
        s.append(f"    {', '.join(res)} = scf.for %ch = [%c0 to %nch step %c1]({args}) -> ({types}) {{")
        cr = {v: f"%cr{j}" for j, (v, _) in enumerate(carried)}
        s.append(f"""    scf.if %stage_live {{
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
        s.append(f"    %half_c = index.rem %ch, %c2 : index\n    %gbase = index.mul %half_c, %c4 : index")
        cur = {a: a for a in accs}
        for g in range(4):
            s.append(f"    %grp{g}_v = index.add %gbase, %c{g if g else 0} : index" if g in (0, 1, 2, 4)
                     else f"    %cg{g} = index.constant {g} : index\n    %grp{g}_v = index.add %gbase, %cg{g} : index")
            s.append(f"    %grp{g} = index.assume %grp{g}_v [range(%grp{g}_v, 0, 7)] : index")
            for i in range(len(WF)):
                s.append(_dequant(f"d{i}{g}", cr[f"%h{i}p"], cr[f"%q{i}{g}p"], f"%grp{g}", swap=SWAP))
            for kk in range(2):
                col = g * 32 + kk * 16
                s.append(f"    %fc{g}{kk} = index.constant {col} : index")
                for i in range(len(WF)):
                    s.append(f"    %rhs{i}{g}{kk} = vector.fragment<rhs> %d{i}{g}_v{kk} shape [%c16, %c16] : vector<16xf16>")
                for f in range(nf):
                    s.append(f"    %fr{f}{g}{kk} = index.constant {f * 16} : index\n"
                             f"    %lhs{f}{g}{kk} = vector.fragment.load<lhs> %a_view[%fr{f}{g}{kk}, %fc{g}{kk}] shape [%c16, %c16] : view<64x{KC + 8}xf16> -> vector<16xf16>")
                    for i in range(len(WF)):
                        a = f"%acc{i}_{f}"
                        nv = f"%m{i}_{f}_{g}{kk}"
                        s.append(f"    {nv} = vector.mma %lhs{f}{g}{kk}, %rhs{i}{g}{kk}, {cur[a]} : vector<16xf16>, vector<16xf16>, vector<8xf32>")
                        cur[a] = nv
        s.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        s.append(f"    scf.yield {', '.join([cur[a] for a in accs] + [v for v, _ in wouts_n + xouts_n])} : {types}\n    }}")
        # epilogue (the inverse-scale tile was written before the loop's first barrier)
        for f in range(nf):
            s.append(f"    %er{f} = index.constant {f * 16} : index\n"
                     f"    %inv{f} = vector.fragment.load<result> %inv_view[%er{f}, %c0] shape [%c16, %c16] : view<64x16xf32> -> vector<8xf32>")
            if sw:
                s.append(f"""    %g{f} = vector.mulf %fin0_{f}, %inv{f} : vector<8xf32>
    %u{f} = vector.mulf %fin1_{f}, %inv{f} : vector<8xf32>
    %sg{f} = vector.siluf<afn> %g{f} : vector<8xf32>
    %act{f} = vector.mulf %sg{f}, %u{f} : vector<8xf32>
    vector.fragment.store<result> %act{f}, %y_view[%er{f}, %wave_col] shape [%c16, %c16] : vector<8xf32>, view<64x{TN}xf32, %y_layout>""")
            else:
                s.append(f"""    %o0_{f} = vector.mulf %fin0_{f}, %inv{f} : vector<8xf32>
    %o1_{f} = vector.mulf %fin1_{f}, %inv{f} : vector<8xf32>
    %wc16_{f} = index.add %wave_col, %c16 : index
    vector.fragment.store<result> %o0_{f}, %y_view[%er{f}, %wave_col] shape [%c16, %c16] : vector<8xf32>, view<64x{TN}xf32, %y_layout>
    vector.fragment.store<result> %o1_{f}, %y_view[%er{f}, %wc16_{f}] shape [%c16, %c16] : vector<8xf32>, view<64x{TN}xf32, %y_layout>""")
        return _suffix("\n".join(s), f"_n{nf}")

    # uniform dispatch on the live fragment count
    L.append("  %c48 = index.constant 48 : index")
    L.append("  %nf1 = index.cmp ule, %tile_rows, %c16 : index\n  scf.if %nf1 {\n" + body(1) + "\n  } else {")
    L.append("  %nf2 = index.cmp ule, %tile_rows, %c32 : index\n  scf.if %nf2 {\n" + body(2) + "\n  } else {")
    L.append("  %nf3 = index.cmp ule, %tile_rows, %c48 : index\n  scf.if %nf3 {\n" + body(3) + "\n  } else {\n" + body(4) + "\n  }\n  }\n  }")
    L.append("  }\n  kernel.return\n}\n")
    text = "\n".join(L)
    # constants the bodies reference by %c<N>
    return text.replace("%c48 = index.constant 48 : index", "%c48 = index.constant 48 : index", 1)
