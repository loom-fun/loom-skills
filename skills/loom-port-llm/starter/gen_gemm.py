"""Generates Glimmer prefill GEMM kernels (adapted from the Flint port) (WMMA, RDNA3 wave32) as Loom source.

Y[rows, n] (+)= X[rows, k] . W^T (X: F16, pre-scaled per row by g_row_scale_f16) with GGUF Q8_0 or Q4_K weights dequantized to
F16 in workgroup memory. A 256-thread workgroup owns a 64x64 output tile;
eight waves own a 4x2 grid of 16x32 outputs. K advances 64 values per stage.
Rows and outputs must be multiples of 64; k a multiple of 64 (Q8_0) or 256 (Q4_K).

Variants:
  store / acc   single weight, Y = XW^T or Y += XW^T
  swiglu        two weights (gate, up), Y = silu(X Wg^T) * (X Wu^T)
"""
import os, re as _re, sys
PREFIX = os.environ.get("GEMM_PREFIX", "g")
META = os.environ.get("GEMM_KMETA", "@g_k_metadata")

FMT = {
    "q4_1": dict(block_bytes=20, block_values=32),
    "q8_0": dict(block_bytes=34, block_values=32),
    "q4_k": dict(block_bytes=144, block_values=256),
    "q5_k": dict(block_bytes=176, block_values=256),
    "q6_k": dict(block_bytes=210, block_values=256),
    # dequantized through gen_llm.py's @g_<fmt>_values helpers (quant_values.loom)
    "q4_0": dict(block_bytes=18, block_values=32),
    "q5_0": dict(block_bytes=22, block_values=32),
    "iq4_nl": dict(block_bytes=18, block_values=32),
    "iq4_xs": dict(block_bytes=136, block_values=256),
    "q3_k": dict(block_bytes=110, block_values=256),
    "iq3_s": dict(block_bytes=110, block_values=256),
    "f16": dict(block_bytes=64, block_values=32),
    "mxfp4": dict(block_bytes=17, block_values=32),
    # BF16 (vendor drafters published only in BF16, used as-is): 16 values load directly, rounded to F16 for the MMA
    "bf16": dict(block_bytes=64, block_values=32),
    # F32 (small projections some quants keep unquantized, e.g. Gemma 3n/4 per-layer-input gate/proj): 16 values load
    # directly, rounded to F16 for the MMA
    "f32": dict(block_bytes=128, block_values=32),
}
HELPER = {"q4_0", "q5_0", "iq4_nl", "iq4_xs", "q3_k", "iq3_s", "f16", "mxfp4"}

def dequant16(fmt, w, view_name, out):
    """Loom snippet: dequantize 16 consecutive weights of row %w_row at column %w_col into F16 vector %{out}."""
    if fmt in HELPER:   # 16 consecutive weights = one half of a 32-value group from the shared value helper
        bv = FMT[fmt]["block_values"]
        return f'''
      %{out}_bv = index.constant {bv} : index
      %{out}_block = index.div %w_col, %{out}_bv : index
      %{out}_within = index.rem %w_col, %{out}_bv : index
      %{out}_group_value = index.div %{out}_within, %c32 : index
      %{out}_group = index.assume %{out}_group_value [range(%{out}_group_value, 0, {bv // 32 - 1})] : index
      %{out}_in_group = index.rem %{out}_within, %c32 : index
      %{out}_half = index.div %{out}_in_group, %c16 : index
      %{out}_block_add = index.scale %{out}_block, %block_bytes : index, offset -> offset
      %{out}_block_origin = index.add %w_row_origin, %{out}_block_add : offset
      %{out}_all = func.call @g_{fmt}_values(%{w}, %{out}_block_origin, %{out}_group) : (buffer, offset, index) -> (vector<32xf32>)
      %{out}_lo = vector.slice %{out}_all[0] : vector<32xf32> -> vector<16xf32>
      %{out}_hi = vector.slice %{out}_all[16] : vector<32xf32> -> vector<16xf32>
      %{out}_upper = index.cmp ne, %{out}_half, %c0 : index
      %{out}_part = scf.select %{out}_upper, %{out}_hi, %{out}_lo : vector<16xf32>
      %{out} = vector.fptrunc %{out}_part : vector<16xf32> to vector<16xf16>'''
    if fmt == "q4_1":
        return f'''
      %{out}_block = index.div %w_col, %c32 : index
      %{out}_in_block = index.rem %w_col, %c32 : index
      %{out}_half = index.div %{out}_in_block, %c16 : index
      %{out}_block_add = index.scale %{out}_block, %block_bytes : index, offset -> offset
      %{out}_block_origin = index.add %w_row_origin, %{out}_block_add : offset
      %{out}_four = index.constant 4 : offset
      %{out}_code_origin = index.add %{out}_block_origin, %{out}_four : offset
      %{out}_dm_view = buffer.view %{w}[%{out}_block_origin] : buffer -> view<2xf16>
      %{out}_code_view = buffer.view %{w}[%{out}_code_origin] : buffer -> view<4xi32>
      %{out}_d = view.load %{out}_dm_view[0] : view<2xf16> -> f16
      %{out}_m = view.load %{out}_dm_view[1] : view<2xf16> -> f16
      %{out}_packed = vector.load %{out}_code_view[0] : view<4xi32> -> vector<4xi32>
      %{out}_shift_bits = index.mul %{out}_half, %c4 : index
      %{out}_shift_i32 = index.cast %{out}_shift_bits : index to i32
      %{out}_shift = vector.splat %{out}_shift_i32 : vector<4xi32>
      %{out}_shifted = vector.shrui %{out}_packed, %{out}_shift : vector<4xi32>
      %{out}_masked = vector.andi %{out}_shifted, %nibble_mask : vector<4xi32>
      %{out}_bytes = vector.bitcast %{out}_masked : vector<4xi32> to vector<16xi8>
      %{out}_codes = vector.uitofp %{out}_bytes : vector<16xi8> to vector<16xf16>
      %{out}_ds = vector.splat %{out}_d : vector<16xf16>
      %{out}_ms = vector.splat %{out}_m : vector<16xf16>
      %{out} = vector.fmaf %{out}_codes, %{out}_ds, %{out}_ms : vector<16xf16>'''
    if fmt == "bf16":
        return f'''
      %{out}_col_bytes = index.scale %w_col, %two : index, offset -> offset
      %{out}_origin = index.add %w_row_origin, %{out}_col_bytes : offset
      %{out}_view = buffer.view %{w}[%{out}_origin] : buffer -> view<16xbf16>
      %{out}_raw = vector.load %{out}_view[0] : view<16xbf16> -> vector<16xbf16>
      %{out}_wide = vector.extf %{out}_raw : vector<16xbf16> to vector<16xf32>
      %{out} = vector.fptrunc %{out}_wide : vector<16xf32> to vector<16xf16>'''
    if fmt == "f32":
        return f'''
      %{out}_four = index.constant 4 : offset
      %{out}_col_bytes = index.scale %w_col, %{out}_four : index, offset -> offset
      %{out}_origin = index.add %w_row_origin, %{out}_col_bytes : offset
      %{out}_view = buffer.view %{w}[%{out}_origin] : buffer -> view<16xf32>
      %{out}_wide = vector.load %{out}_view[0] : view<16xf32> -> vector<16xf32>
      %{out} = vector.fptrunc %{out}_wide : vector<16xf32> to vector<16xf16>'''
    if fmt == "q8_0":
        return f'''
      %{out}_block = index.div %w_col, %c32 : index
      %{out}_half = index.rem %w_col, %c32 : index
      %{out}_block_add = index.scale %{out}_block, %block_bytes : index, offset -> offset
      %{out}_block_origin = index.add %w_row_origin, %{out}_block_add : offset
      %{out}_half_bytes = index.cast %{out}_half : index to offset
      %{out}_code_origin0 = index.add %{out}_block_origin, %two : offset
      %{out}_code_origin = index.add %{out}_code_origin0, %{out}_half_bytes : offset
      %{out}_scale_view = buffer.view %{w}[%{out}_block_origin] : buffer -> view<1xf16>
      %{out}_code_view = buffer.view %{w}[%{out}_code_origin] : buffer -> view<16xi8>
      %{out}_scale_half = view.load %{out}_scale_view[0] : view<1xf16> -> f16
      %{out}_codes = vector.load %{out}_code_view[0] : view<16xi8> -> vector<16xi8>
      // q (8 bits) * d (f16) is exact in f32, so one f16 multiply rounds to the same value as f32 math + fptrunc.
      %{out}_values = vector.sitofp %{out}_codes : vector<16xi8> to vector<16xf16>
      %{out}_scales = vector.splat %{out}_scale_half : vector<16xf16>
      %{out} = vector.mulf %{out}_values, %{out}_scales : vector<16xf16>'''
    if fmt == "q6_k":
        # Q6_K: value c of a superblock: half = c/128, j = (c%128)/32, l = c%32;
        # q = ((ql[half*64 + (j%2)*32 + l] >> 4*(j/2)) & 15) | (((qh[half*32 + l] >> 2j) & 3) << 4);
        # w = d * sc[half*8 + 2j + l/16] * (q - 32). 16 consecutive values share one scale.
        return f'''
      %{out}_c128 = index.constant 128 : index
      %{out}_c64 = index.constant 64 : index
      %{out}_qh_base = index.constant 128 : index
      %{out}_sc_base = index.constant 192 : index
      %{out}_d_bytes = index.constant 208 : offset
      %{out}_super = index.div %w_col, %c256 : index
      %{out}_within = index.rem %w_col, %c256 : index
      %{out}_half = index.div %{out}_within, %{out}_c128 : index
      %{out}_in_half = index.rem %{out}_within, %{out}_c128 : index
      %{out}_j = index.div %{out}_in_half, %c32 : index
      %{out}_l0 = index.rem %{out}_in_half, %c32 : index
      %{out}_jl = index.rem %{out}_j, %c2 : index
      %{out}_jh = index.div %{out}_j, %c2 : index
      %{out}_ql_a = index.mul %{out}_half, %{out}_c64 : index
      %{out}_ql_b = index.mul %{out}_jl, %c32 : index
      %{out}_ql_c = index.add %{out}_ql_a, %{out}_ql_b : index
      %{out}_ql_i = index.add %{out}_ql_c, %{out}_l0 : index
      %{out}_qh_a = index.mul %{out}_half, %c32 : index
      %{out}_qh_b = index.add %{out}_qh_base, %{out}_qh_a : index
      %{out}_qh_i = index.add %{out}_qh_b, %{out}_l0 : index
      %{out}_sc_a = index.mul %{out}_half, %c8 : index
      %{out}_sc_b = index.mul %{out}_j, %c2 : index
      %{out}_sc_c = index.div %{out}_l0, %c16 : index
      %{out}_sc_d = index.add %{out}_sc_a, %{out}_sc_b : index
      %{out}_sc_e = index.add %{out}_sc_d, %{out}_sc_c : index
      %{out}_sc_i = index.add %{out}_sc_base, %{out}_sc_e : index
      %{out}_block_add = index.scale %{out}_super, %block_bytes : index, offset -> offset
      %{out}_block_origin = index.add %w_row_origin, %{out}_block_add : offset
      %{out}_ql_off = index.cast %{out}_ql_i : index to offset
      %{out}_qh_off = index.cast %{out}_qh_i : index to offset
      %{out}_sc_off = index.cast %{out}_sc_i : index to offset
      %{out}_ql_o = index.add %{out}_block_origin, %{out}_ql_off : offset
      %{out}_qh_o = index.add %{out}_block_origin, %{out}_qh_off : offset
      %{out}_sc_o = index.add %{out}_block_origin, %{out}_sc_off : offset
      %{out}_d_o = index.add %{out}_block_origin, %{out}_d_bytes : offset
      %{out}_ql_v = buffer.view %{w}[%{out}_ql_o] : buffer -> view<4xi32>
      %{out}_qh_v = buffer.view %{w}[%{out}_qh_o] : buffer -> view<4xi32>
      %{out}_sc_v = buffer.view %{w}[%{out}_sc_o] : buffer -> view<1xi8>
      %{out}_d_v = buffer.view %{w}[%{out}_d_o] : buffer -> view<1xf16>
      %{out}_ql = vector.load %{out}_ql_v[0] : view<4xi32> -> vector<4xi32>
      %{out}_qh = vector.load %{out}_qh_v[0] : view<4xi32> -> vector<4xi32>
      %{out}_sc8 = view.load %{out}_sc_v[0] : view<1xi8> -> i8
      %{out}_dh = view.load %{out}_d_v[0] : view<1xf16> -> f16
      %{out}_s1 = index.mul %{out}_jh, %c4 : index
      %{out}_s2 = index.mul %{out}_j, %c2 : index
      %{out}_s1_32 = index.cast %{out}_s1 : index to i32
      %{out}_s2_32 = index.cast %{out}_s2 : index to i32
      %{out}_s1_v = vector.splat %{out}_s1_32 : vector<4xi32>
      %{out}_s2_v = vector.splat %{out}_s2_32 : vector<4xi32>
      %{out}_m15 = vector.constant 252645135 : vector<4xi32>
      %{out}_m3 = vector.constant 50529027 : vector<4xi32>
      %{out}_f4 = vector.constant 4 : vector<4xi32>
      // Word-wide bit work (4 codes per i32): a byte's bits shifted in from its neighbour are masked off.
      %{out}_lo_s = vector.shrui %{out}_ql, %{out}_s1_v : vector<4xi32>
      %{out}_lo = vector.andi %{out}_lo_s, %{out}_m15 : vector<4xi32>
      %{out}_hi_s = vector.shrui %{out}_qh, %{out}_s2_v : vector<4xi32>
      %{out}_hi_m = vector.andi %{out}_hi_s, %{out}_m3 : vector<4xi32>
      %{out}_hi = vector.shli %{out}_hi_m, %{out}_f4 : vector<4xi32>
      %{out}_qw = vector.ori %{out}_lo, %{out}_hi : vector<4xi32>
      %{out}_q = vector.bitcast %{out}_qw : vector<4xi32> to vector<16xi8>
      %{out}_codes = vector.uitofp %{out}_q : vector<16xi8> to vector<16xf16>
      %{out}_d = scalar.extf %{out}_dh : f16 to f32
      %{out}_scf = scalar.sitofp %{out}_sc8 : i8 to f32
      %{out}_scale = scalar.mulf %{out}_d, %{out}_scf : f32
      %{out}_bias32 = scalar.constant -32.0 : f32
      %{out}_offset = scalar.mulf %{out}_scale, %{out}_bias32 : f32
      %{out}_scale_h = scalar.fptrunc %{out}_scale : f32 to f16
      %{out}_offset_h = scalar.fptrunc %{out}_offset : f32 to f16
      %{out}_scales = vector.splat %{out}_scale_h : vector<16xf16>
      %{out}_offsets = vector.splat %{out}_offset_h : vector<16xf16>
      %{out} = vector.fmaf %{out}_codes, %{out}_scales, %{out}_offsets : vector<16xf16>'''
    if fmt == "q5_k":
        # Q5_K: 176-byte superblocks = Q4_K's 16-byte header (d, dmin, 6-bit scales/mins),
        # qh[32] (bit g of qh[l] is value l of group g's fifth bit), then qs[128] as in Q4_K.
        return f'''
      %{out}_super = index.div %w_col, %c256 : index
      %{out}_within = index.rem %w_col, %c256 : index
      %{out}_group_value = index.div %{out}_within, %c32 : index
      %{out}_group = index.assume %{out}_group_value [range(%{out}_group_value, 0, 7)] : index
      %{out}_in_group = index.rem %{out}_within, %c32 : index
      %{out}_half = index.div %{out}_in_group, %c16 : index
      %{out}_block_add = index.scale %{out}_super, %block_bytes : index, offset -> offset
      %{out}_block_origin = index.add %w_row_origin, %{out}_block_add : offset
      %{out}_header_view = buffer.view %{w}[%{out}_block_origin] : buffer -> view<4xi32>
      %{out}_header = vector.load %{out}_header_view[0] : view<4xi32> -> vector<4xi32>
      %{out}_scale, %{out}_minimum = func.call {META}(%{out}_header, %{out}_group) : (vector<4xi32>, index) -> (f32, f32)
      %{out}_page = index.div %{out}_group, %c2 : index
      %{out}_nibble = index.rem %{out}_group, %c2 : index
      %{out}_word_page = index.mul %{out}_page, %c8 : index
      %{out}_word_half = index.mul %{out}_half, %c4 : index
      %{out}_word = index.add %{out}_word_page, %{out}_word_half : index
      %{out}_qs_bytes = index.constant 48 : offset
      %{out}_payload = index.add %{out}_block_origin, %{out}_qs_bytes : offset
      %{out}_words = buffer.view %{w}[%{out}_payload] : buffer -> view<32xi32>
      %{out}_packed = vector.load %{out}_words[%{out}_word] : view<32xi32> -> vector<4xi32>
      %{out}_shift_bits = index.mul %{out}_nibble, %c4 : index
      %{out}_shift_i32 = index.cast %{out}_shift_bits : index to i32
      %{out}_shift = vector.splat %{out}_shift_i32 : vector<4xi32>
      %{out}_shifted = vector.shrui %{out}_packed, %{out}_shift : vector<4xi32>
      %{out}_masked = vector.andi %{out}_shifted, %nibble_mask : vector<4xi32>
      %{out}_low = vector.bitcast %{out}_masked : vector<4xi32> to vector<16xi8>
      %{out}_qh_half = index.mul %{out}_half, %c16 : index
      %{out}_qh_half_bytes = index.cast %{out}_qh_half : index to offset
      %{out}_qh_origin = index.add %{out}_block_origin, %sixteen : offset
      %{out}_qh_at = index.add %{out}_qh_origin, %{out}_qh_half_bytes : offset
      %{out}_qh_view = buffer.view %{w}[%{out}_qh_at] : buffer -> view<4xi32>
      %{out}_qh = vector.load %{out}_qh_view[0] : view<4xi32> -> vector<4xi32>
      %{out}_g32 = index.cast %{out}_group : index to i32
      %{out}_gv = vector.splat %{out}_g32 : vector<4xi32>
      %{out}_one = vector.constant 16843009 : vector<4xi32>
      %{out}_four = vector.constant 4 : vector<4xi32>
      // Word-wide: bit g of every qh byte (0x01010101 mask after the shift), moved to bit 4 of the code byte.
      %{out}_hb_s = vector.shrui %{out}_qh, %{out}_gv : vector<4xi32>
      %{out}_hb = vector.andi %{out}_hb_s, %{out}_one : vector<4xi32>
      %{out}_hi = vector.shli %{out}_hb, %{out}_four : vector<4xi32>
      %{out}_lw = vector.bitcast %{out}_low : vector<16xi8> to vector<4xi32>
      %{out}_bw = vector.ori %{out}_lw, %{out}_hi : vector<4xi32>
      %{out}_bytes = vector.bitcast %{out}_bw : vector<4xi32> to vector<16xi8>
      %{out}_codes = vector.uitofp %{out}_bytes : vector<16xi8> to vector<16xf16>
      %{out}_scale_h = scalar.fptrunc %{out}_scale : f32 to f16
      %{out}_neg_min = scalar.negf %{out}_minimum : f32
      %{out}_neg_min_h = scalar.fptrunc %{out}_neg_min : f32 to f16
      %{out}_scales = vector.splat %{out}_scale_h : vector<16xf16>
      %{out}_minima = vector.splat %{out}_neg_min_h : vector<16xf16>
      %{out} = vector.fmaf %{out}_codes, %{out}_scales, %{out}_minima : vector<16xf16>'''
    # Q4_K: 256-value superblocks of 144 bytes; 8 groups of 32 with 6-bit scale/min.
    return f'''
      %{out}_super = index.div %w_col, %c256 : index
      %{out}_within = index.rem %w_col, %c256 : index
      %{out}_group_value = index.div %{out}_within, %c32 : index
      %{out}_group = index.assume %{out}_group_value [range(%{out}_group_value, 0, 7)] : index
      %{out}_in_group = index.rem %{out}_within, %c32 : index
      %{out}_half = index.div %{out}_in_group, %c16 : index
      %{out}_block_add = index.scale %{out}_super, %block_bytes : index, offset -> offset
      %{out}_block_origin = index.add %w_row_origin, %{out}_block_add : offset
      %{out}_header_view = buffer.view %{w}[%{out}_block_origin] : buffer -> view<4xi32>
      %{out}_header = vector.load %{out}_header_view[0] : view<4xi32> -> vector<4xi32>
      %{out}_scale, %{out}_minimum = func.call {META}(%{out}_header, %{out}_group) : (vector<4xi32>, index) -> (f32, f32)
      %{out}_page = index.div %{out}_group, %c2 : index
      %{out}_nibble = index.rem %{out}_group, %c2 : index
      %{out}_word_page = index.mul %{out}_page, %c8 : index
      %{out}_word_half = index.mul %{out}_half, %c4 : index
      %{out}_word = index.add %{out}_word_page, %{out}_word_half : index
      %{out}_payload = index.add %{out}_block_origin, %sixteen : offset
      %{out}_words = buffer.view %{w}[%{out}_payload] : buffer -> view<32xi32>
      %{out}_packed = vector.load %{out}_words[%{out}_word] : view<32xi32> -> vector<4xi32>
      %{out}_shift_bits = index.mul %{out}_nibble, %c4 : index
      %{out}_shift_i32 = index.cast %{out}_shift_bits : index to i32
      %{out}_shift = vector.splat %{out}_shift_i32 : vector<4xi32>
      %{out}_shifted = vector.shrui %{out}_packed, %{out}_shift : vector<4xi32>
      %{out}_masked = vector.andi %{out}_shifted, %nibble_mask : vector<4xi32>
      %{out}_bytes = vector.bitcast %{out}_masked : vector<4xi32> to vector<16xi8>
      %{out}_codes = vector.uitofp %{out}_bytes : vector<16xi8> to vector<16xf16>
      %{out}_scale_h = scalar.fptrunc %{out}_scale : f32 to f16
      %{out}_neg_min = scalar.negf %{out}_minimum : f32
      %{out}_neg_min_h = scalar.fptrunc %{out}_neg_min : f32 to f16
      %{out}_scales = vector.splat %{out}_scale_h : vector<16xf16>
      %{out}_minima = vector.splat %{out}_neg_min_h : vector<16xf16>
      %{out} = vector.fmaf %{out}_codes, %{out}_scales, %{out}_minima : vector<16xf16>'''


def gemm(fmt, mode):
    f = FMT[fmt]
    two_w = mode == "swiglu"
    name = f"{PREFIX}_gemm_{fmt}_{mode}"
    wargs = "%w: buffer, %w2: buffer" if two_w else "%w: buffer"
    accs = ["g0", "g1", "u0", "u1"] if two_w else ["a0", "a1"]
    stages = ["w_stage", "w2_stage"] if two_w else ["w_stage"]
    k_min = 256 if FMT[fmt]["block_values"] == 256 else 64
    lines = []
    lines.append(f'''kernel.def export("{name}") @{name}(%rows: index, %n: index) {{
  %c1 = index.constant 1 : index
  %c64 = index.constant 64 : index
  %c256 = index.constant 256 : index
  %row_tiles = index.div %rows, %c64 : index
  %out_tiles = index.div %n, %c64 : index
  kernel.launch.config workgroups(%row_tiles, %out_tiles, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%rows_arg: index, %k_arg: index, %n_arg: index, %x: buffer, %scales: buffer, {wargs}, %y: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 64, 8192), multiple_of(%rows_arg, 64)] : index
  %k = index.assume %k_arg [range(%k_arg, {k_min}, 65536), multiple_of(%k_arg, {k_min})] : index
  %n = index.assume %n_arg [range(%n_arg, 64, 262144), multiple_of(%n_arg, 64)] : index
  %origin = index.constant 0 : offset
  %two = index.constant 2 : offset
  %sixteen = index.constant 16 : offset
  %block_bytes = index.constant {f["block_bytes"]} : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %c256 = index.constant 256 : index
  %nibble_mask = vector.constant 252645135 : vector<4xi32>
  %stage_bytes = index.constant 9216 : offset
  %row_tile_count = index.div %rows, %c64 : index
  %out_tile_count = index.div %n, %c64 : index
  %row_group_id = kernel.workgroup.id<x> : index
  %out_group_id = kernel.workgroup.id<y> : index
  %row_group = index.assume %row_group_id [lt(%row_group_id, %row_tile_count)] : index
  %out_group = index.assume %out_group_id [lt(%out_group_id, %out_tile_count)] : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, 255)] : index
  %subgroup = kernel.subgroup.id : index
  %lane = kernel.subgroup.lane.id : index
  %lane_row = index.rem %lane, %c16 : index
  %bounded_subgroup = index.assume %subgroup [range(%subgroup, 0, 7)] : index
  %wave_row_group = index.div %bounded_subgroup, %c2 : index
  %wave_out_group = index.rem %bounded_subgroup, %c2 : index
  %wave_row = index.mul %wave_row_group, %c16 : index
  %wave_out = index.mul %wave_out_group, %c32 : index
  %w_row0 = index.add %wave_out, %lane_row : index
  %w_row1 = index.add %w_row0, %c16 : index
  %row_origin = index.mul %row_group, %c64 : index
  %out_origin = index.mul %out_group, %c64 : index
  %result_row = index.add %row_origin, %wave_row : index
  %result_col0 = index.add %out_origin, %wave_out : index
  %result_col1 = index.add %result_col0, %c16 : index
  %blocks = index.div %k, %c{f["block_values"]} : index
  %row_bytes = index.scale %blocks, %block_bytes : index, offset -> offset
  %x_view = buffer.view %x[%origin] : buffer -> view<[%rows]x[%k]xf16>
  // 64x64 output tile with the full row stride n.
  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %tile_first = index.madd %row_origin, %n, %out_origin : index
  %four = index.constant 4 : offset
  %tile_offset = index.scale %tile_first, %four : index, offset -> offset
  %y_view = buffer.view %y[%tile_offset] : buffer -> view<64x64xf32, %y_layout>
  %bounded_wave_row = index.assume %wave_row [range(%wave_row, 0, 48)] : index
  %bounded_wave_out = index.assume %wave_out [range(%wave_out, 0, 32)] : index
  %wave_out1 = index.add %bounded_wave_out, %c16 : index
  %x_stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %x_stage_view = buffer.view %x_stage[%origin] : buffer -> view<64x72xf16>''')
    for st in stages:
        lines.append(f'''  %{st} = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %{st}_view = buffer.view %{st}[%origin] : buffer -> view<64x72xf16>''')
    lines.append('''  // Thread t stages 16 values: row t/4, columns (t%4)*16.
  %stage_row = index.div %workitem, %c4 : index
  %stage_quarter = index.rem %workitem, %c4 : index
  %stage_col = index.mul %stage_quarter, %c16 : index
  %x_row = index.add %row_origin, %stage_row : index
  %bounded_x_row = index.assume %x_row [lt(%x_row, %rows)] : index
  %w_out_row = index.add %out_origin, %stage_row : index
  %bounded_w_out_row = index.assume %w_out_row [lt(%w_out_row, %n)] : index
  %w_row_origin = index.scale %bounded_w_out_row, %row_bytes : index, offset -> offset
  %zero8 = vector.constant 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %k_steps = index.div %k, %c64 : index
  %s_view = buffer.view %scales[%origin] : buffer -> view<[%rows]x2xf32>
  %x_scale = view.load %s_view[%bounded_x_row, %c0] : view<[%rows]x2xf32> -> f32
  %x_scales = vector.splat %x_scale : vector<16xf32>''')
    acc_args = ", ".join(f"%{a}_acc = %init : vector<8xf32>" for a in accs)
    acc_types = ", ".join("vector<8xf32>" for _ in accs)
    lines.append(f'''  {", ".join("%" + a + "_raw" for a in accs)} = scf.for %step = [%c0 to %k_steps step %c1]({acc_args}) -> ({acc_types}) {{
    %k_origin = index.mul %step, %c64 : index
    %x_col = index.add %k_origin, %stage_col : index
    %x_half = vector.load %x_view[%bounded_x_row, %x_col] : view<[%rows]x[%k]xf16> -> vector<16xf16>
    vector.store %x_half, %x_stage_view[%stage_row, %stage_col] : vector<16xf16>, view<64x72xf16>
    %w_col = index.add %k_origin, %stage_col : index''')
    for i, (st, w) in enumerate(zip(stages, ["w", "w2"])):
        lines.append(dequant16(fmt, w, None, f"deq{i}"))
        lines.append(f"    vector.store %deq{i}, %{st}_view[%stage_row, %stage_col] : vector<16xf16>, view<64x72xf16>")
    lines.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    cur = {a: f"%{a}_acc" for a in accs}
    for kk in range(4):
        col = kk * 16
        lines.append(f'''    %fc{kk} = index.constant {col} : index
    %lhs{kk} = vector.fragment.load<lhs> %x_stage_view[%wave_row, %fc{kk}] shape [%c16, %c16] : view<64x72xf16> -> vector<16xf16>''')
        for si, st in enumerate(stages):
            pre = "g" if (two_w and si == 0) else ("u" if two_w else "a")
            for r in range(2):
                wr = f"%w_row{r}"
                lines.append(f'''    %rv{si}{r}{kk} = vector.load %{st}_view[{wr}, %fc{kk}] : view<64x72xf16> -> vector<16xf16>
    %rhs{si}{r}{kk} = vector.fragment<rhs> %rv{si}{r}{kk} shape [%c16, %c16] : vector<16xf16>
    %n{pre}{r}_{kk} = vector.mma %lhs{kk}, %rhs{si}{r}{kk}, {cur[pre + str(r)]} : vector<16xf16>, vector<16xf16>, vector<8xf32>''')
                cur[pre + str(r)] = f"%n{pre}{r}_{kk}"
    lines.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    lines.append(f"    scf.yield {', '.join(cur[a] for a in accs)} : {acc_types}\n  }}")
    lines.append('''  %inv_bytes = index.constant 4096 : offset
  %inv_tile = buffer.alloca<workgroup> align(16) %inv_bytes : buffer
  %inv_view = buffer.view %inv_tile[%origin] : buffer -> view<64x16xf32>
  %writer = index.cmp ult, %workitem, %c64 : index
  scf.if %writer {
    %inv_row_value = index.add %row_origin, %workitem : index
    %inv_row = index.assume %inv_row_value [lt(%inv_row_value, %rows)] : index
    %inverse = view.load %s_view[%inv_row, %c1] : view<[%rows]x2xf32> -> f32
    %inverse16 = vector.splat %inverse : vector<16xf32>
    %bw = index.assume %workitem [range(%workitem, 0, 63)] : index
    vector.store %inverse16, %inv_view[%bw, %c0] : vector<16xf32>, view<64x16xf32>
  }
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %inv_frag = vector.fragment.load<result> %inv_view[%wave_row, %c0] shape [%c16, %c16] : view<64x16xf32> -> vector<8xf32>''')
    raw = {a: a + "_raw" for a in accs}
    for a in accs:
        lines.append(f"  %{a} = vector.mulf %{a}_raw, %inv_frag : vector<8xf32>")
    if mode == "store":
        lines.append('''  vector.fragment.store<result> %a0, %y_view[%bounded_wave_row, %bounded_wave_out] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>
  vector.fragment.store<result> %a1, %y_view[%bounded_wave_row, %wave_out1] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>''')
    elif mode == "acc":
        lines.append('''  %old0 = vector.fragment.load<result> %y_view[%bounded_wave_row, %bounded_wave_out] shape [%c16, %c16] : view<64x64xf32, %y_layout> -> vector<8xf32>
  %old1 = vector.fragment.load<result> %y_view[%bounded_wave_row, %wave_out1] shape [%c16, %c16] : view<64x64xf32, %y_layout> -> vector<8xf32>
  %sum0 = vector.addf %old0, %a0 : vector<8xf32>
  %sum1 = vector.addf %old1, %a1 : vector<8xf32>
  vector.fragment.store<result> %sum0, %y_view[%bounded_wave_row, %bounded_wave_out] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>
  vector.fragment.store<result> %sum1, %y_view[%bounded_wave_row, %wave_out1] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>''')
    else:
        lines.append('''  %sg0 = vector.siluf<afn> %g0 : vector<8xf32>
  %sg1 = vector.siluf<afn> %g1 : vector<8xf32>
  %act0 = vector.mulf %sg0, %u0 : vector<8xf32>
  %act1 = vector.mulf %sg1, %u1 : vector<8xf32>
  vector.fragment.store<result> %act0, %y_view[%bounded_wave_row, %bounded_wave_out] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>
  vector.fragment.store<result> %act1, %y_view[%bounded_wave_row, %wave_out1] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>''')
    lines.append("  kernel.return\n}\n")
    return "\n".join(lines)


def gemm_big(fmt, mode, tn=None, suffix=""):
    """128-row WMMA GEMM. store/acc: 128x128 tiles, waves 4 (32 rows) x 2 (64 cols).
    swiglu: 128x64 tiles per weight, waves 4 x 2 (32 cols each of gate and up)."""
    f = FMT[fmt]
    two_w = mode == "swiglu"
    TN = tn or (64 if two_w else 128)
    wave_cols = TN // 2
    cfr = wave_cols // 16
    name = f"{PREFIX}_gemm_{fmt}_{mode}_m128" + suffix
    weights = ["w", "w2"] if two_w else ["w"]
    k_min = 256 if FMT[fmt]["block_values"] == 256 else 64
    L = []
    L.append(f'''kernel.def export("{name}") @{name}(%rows: index, %n: index) {{
  %c1 = index.constant 1 : index
  %c128 = index.constant 128 : index
  %tn = index.constant {TN} : index
  %c256 = index.constant 256 : index
  %row_tiles = index.div %rows, %c128 : index
  %out_tiles = index.div %n, %tn : index
  kernel.launch.config workgroups(%row_tiles, %out_tiles, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%rows_arg: index, %k_arg: index, %n_arg: index, %x: buffer, %scales: buffer, {", ".join(f"%{w}: buffer" for w in weights)}, %y: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 128, 8192), multiple_of(%rows_arg, 128)] : index
  %k = index.assume %k_arg [range(%k_arg, {k_min}, 65536), multiple_of(%k_arg, {k_min})] : index
  %n = index.assume %n_arg [range(%n_arg, {TN}, 262144), multiple_of(%n_arg, {TN})] : index
  %origin = index.constant 0 : offset
  %two = index.constant 2 : offset
  %sixteen = index.constant 16 : offset
  %four = index.constant 4 : offset
  %block_bytes = index.constant {f["block_bytes"]} : offset
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
  %tn = index.constant {TN} : index
  %nibble_mask = vector.constant 252645135 : vector<4xi32>
  %row_tile_count = index.div %rows, %c128 : index
  %out_tile_count = index.div %n, %tn : index
  %row_group_id = kernel.workgroup.id<x> : index
  %out_group_id = kernel.workgroup.id<y> : index
  %row_group = index.assume %row_group_id [lt(%row_group_id, %row_tile_count)] : index
  %out_group = index.assume %out_group_id [lt(%out_group_id, %out_tile_count)] : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, 255)] : index
  %subgroup = kernel.subgroup.id : index
  %bounded_subgroup = index.assume %subgroup [range(%subgroup, 0, 7)] : index
  %lane = kernel.subgroup.lane.id : index
  %lane_row = index.rem %lane, %c16 : index
  %wave_row_group = index.div %bounded_subgroup, %c2 : index
  %wave_col_group = index.rem %bounded_subgroup, %c2 : index
  %wave_row_value = index.mul %wave_row_group, %c32 : index
  %wave_row = index.assume %wave_row_value [range(%wave_row_value, 0, 96)] : index
  %wave_row1 = index.add %wave_row, %c16 : index
  %wave_cols = index.constant {wave_cols} : index
  %wave_col_value = index.mul %wave_col_group, %wave_cols : index
  %wave_col = index.assume %wave_col_value [range(%wave_col_value, 0, {TN - wave_cols})] : index
  %row_origin = index.mul %row_group, %c128 : index
  %out_origin = index.mul %out_group, %tn : index
  %blocks = index.div %k, %c{f["block_values"]} : index
  %row_bytes = index.scale %blocks, %block_bytes : index, offset -> offset
  %x_view = buffer.view %x[%origin] : buffer -> view<[%rows]x[%k]xf16>
  %s_view = buffer.view %scales[%origin] : buffer -> view<[%rows]x2xf32>
  %x_bytes = index.constant 18432 : offset
  %x_stage = buffer.alloca<workgroup> align(16) %x_bytes : buffer
  %x_stage_view = buffer.view %x_stage[%origin] : buffer -> view<128x72xf16>
  %w_bytes = index.constant {TN * 144} : offset''')
    for w in weights:
        L.append(f'''  %{w}_stage = buffer.alloca<workgroup> align(16) %w_bytes : buffer
  %{w}_stage_view = buffer.view %{w}_stage[%origin] : buffer -> view<{TN}x72xf16>''')
    per_thread_w = TN * 64 // 256
    L.append(f'''  // A: thread t stages 32 values of row t/2. W: {per_thread_w} values of row t/{256 // TN}.
  %x_stage_row = index.div %workitem, %c2 : index
  %x_half = index.rem %workitem, %c2 : index
  %x_stage_col = index.mul %x_half, %c32 : index
  %x_row_value = index.add %row_origin, %x_stage_row : index
  %x_row = index.assume %x_row_value [lt(%x_row_value, %rows)] : index
  %x_scale = view.load %s_view[%x_row, %c0] : view<[%rows]x2xf32> -> f32
  %x_scales = vector.splat %x_scale : vector<32xf32>
  %w_per_row = index.constant {256 // TN} : index
  %stage_row = index.div %workitem, %w_per_row : index
  %w_seg = index.rem %workitem, %w_per_row : index
  %w_seg_width = index.constant {per_thread_w} : index
  %stage_col = index.mul %w_seg, %w_seg_width : index
  %w_out_row_value = index.add %out_origin, %stage_row : index
  %w_out_row = index.assume %w_out_row_value [lt(%w_out_row_value, %n)] : index
  %w_row_origin = index.scale %w_out_row, %row_bytes : index, offset -> offset
  %zero8 = vector.constant 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %k_steps = index.div %k, %c64 : index''')
    accs = [f"{wi}r{r}c{c}" for wi in range(len(weights)) for r in range(2) for c in range(cfr)]
    import re as _re
    # Register prefetch: each weight segment's dequant splits into an address +
    # load part (issued for step+1 before this step's MMAs) and a decode part
    # that consumes the raw values carried through the loop.
    segs = []
    for wi, w in enumerate(weights):
        for seg in range(per_thread_w // 16):
            out = f"dq{wi}_{seg}"
            full = dequant16(fmt, w, None, out).replace("%w_col,", f"%wcol_{wi}_{seg},").replace("%w_col :", f"%wcol_{wi}_{seg} :")
            lines = [l for l in full.strip("\n").split("\n") if l.strip()]
            last = max(i for i, l in enumerate(lines) if ".load " in l)
            addr = lines[: last + 1]
            loads = [(_re.match(r"\s*(%\S+) = ", l).group(1), l.split("-> ")[-1].strip()) for l in addr if ".load " in l]
            segs.append((wi, seg, w, out, lines, addr, loads))
    def addr_text(sfx, kvar):
        t = []
        for wi, seg, w, out, lines, addr, loads in segs:
            t.append(f"    %wc{wi}_{seg}_c{sfx} = index.constant {seg * 16} : index\n    %wc{wi}_{seg}_l{sfx} = index.add %stage_col, %wc{wi}_{seg}_c{sfx} : index\n    %wcol_{wi}_{seg}{sfx} = index.add {kvar}, %wc{wi}_{seg}_l{sfx} : index")
            body = "\n".join(addr).replace(f"%wcol_{wi}_{seg}", f"%wcol_{wi}_{seg}{sfx}")
            body = _re.sub(rf"%{out}_", f"%{out}{sfx}_", body)
            t.append(body)
        if sfx == "p":
            t.append(f"    %x_col{sfx} = index.add {kvar}, %x_stage_col : index\n    %x_raw{sfx} = vector.load %x_view[%x_row, %x_col{sfx}] : view<[%rows]x[%k]xf16> -> vector<32xf16>")
        else:
            t.append(f"""    %x_more = index.cmp ult, %x_next_step, %k_steps : index
    %x_raw{sfx} = scf.if %x_more -> (vector<32xf16>) {{
      %x_ns = index.assume %x_next_step [lt(%x_next_step, %k_steps)] : index
      %x_ko = index.mul %x_ns, %c64 : index
      %x_col{sfx} = index.add %x_ko, %x_stage_col : index
      %xv = vector.load %x_view[%x_row, %x_col{sfx}] : view<[%rows]x[%k]xf16> -> vector<32xf16>
      scf.yield %xv : vector<32xf16>
    }} else {{
      scf.yield %cr_x : vector<32xf16>
    }}""")
        return "\n".join(t)
    carried = []   # (iter name, init name, next name, type)
    for wi, seg, w, out, lines, addr, loads in segs:
        for name, ty in loads:
            base = name[1:]
            carried.append((f"%cr_{base}", f"%{base.replace(out + '_', out + 'p_', 1)}", f"%{base.replace(out + '_', out + 'n_', 1)}", ty))
    carried.append(("%cr_x", "%x_rawp", "%x_rawn", "vector<32xf16>"))
    L.append(addr_text("p", "%c0").replace("    ", "  ", 1))
    L.append("  %k_last = index.sub %k_steps, %c1 : index")
    L.append("  " + ", ".join(f"%o{a}" for a in accs) + ", " + ", ".join(f"%unused_{i}" for i in range(len(carried))) + " = scf.for %step = [%c0 to %k_steps step %c1](" +
             ", ".join(f"%i{a} = %init : vector<8xf32>" for a in accs) + ", " + ", ".join(f"{c[0]} = {c[1]} : {c[3]}" for c in carried) + ") -> (" +
             ", ".join("vector<8xf32>" for _ in accs) + ", " + ", ".join(c[3] for c in carried) + ") {")
    L.append("    %k_origin = index.mul %step, %c64 : index\n    vector.store %cr_x, %x_stage_view[%x_stage_row, %x_stage_col] : vector<32xf16>, view<128x72xf16>")
    for wi, seg, w, out, lines, addr, loads in segs:
        L.append(f"    %wc{wi}_{seg}_c = index.constant {seg * 16} : index\n    %wc{wi}_{seg}_l = index.add %stage_col, %wc{wi}_{seg}_c : index\n    %wcol_{wi}_{seg} = index.add %k_origin, %wc{wi}_{seg}_l : index")
        body = [l for l in lines if ".load " not in l]
        text = "\n".join(body)
        for name, ty in loads:
            text = _re.sub(_re.escape(name) + r"\b", f"%cr_{name[1:]}", text)
        L.append(text)
        L.append(f"    vector.store %{out}, %{w}_stage_view[%stage_row, %wc{wi}_{seg}_l] : vector<16xf16>, view<{TN}x72xf16>")
    L.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    L.append("    %k_next_raw = index.add %k_origin, %c64 : index\n    %k_limit = index.sub %k, %c64 : index\n    %k_next = index.min %k_next_raw, %k_limit : index\n    %x_next_step = index.add %step, %c1 : index")
    L.append(addr_text("n", "%k_next"))
    cur = {a: f"%i{a}" for a in accs}
    for kk in range(4):
        L.append(f'''    %fk{kk} = index.constant {kk * 16} : index
    %l0_{kk} = vector.fragment.load<lhs> %x_stage_view[%wave_row, %fk{kk}] shape [%c16, %c16] : view<128x72xf16> -> vector<16xf16>
    %l1_{kk} = vector.fragment.load<lhs> %x_stage_view[%wave_row1, %fk{kk}] shape [%c16, %c16] : view<128x72xf16> -> vector<16xf16>''')
        for wi, w in enumerate(weights):
            for c in range(cfr):
                L.append(f'''    %wr{wi}{c}_{kk}_c = index.constant {c * 16} : index
    %wr{wi}{c}_{kk}_b = index.add %wave_col, %wr{wi}{c}_{kk}_c : index
    %wr{wi}{c}_{kk} = index.add %wr{wi}{c}_{kk}_b, %lane_row : index
    %rv{wi}{c}_{kk} = vector.load %{w}_stage_view[%wr{wi}{c}_{kk}, %fk{kk}] : view<{TN}x72xf16> -> vector<16xf16>
    %rf{wi}{c}_{kk} = vector.fragment<rhs> %rv{wi}{c}_{kk} shape [%c16, %c16] : vector<16xf16>''')
                for r in range(2):
                    a = f"{wi}r{r}c{c}"
                    L.append(f"    %m{a}_{kk} = vector.mma %l{r}_{kk}, %rf{wi}{c}_{kk}, {cur[a]} : vector<16xf16>, vector<16xf16>, vector<8xf32>")
                    cur[a] = f"%m{a}_{kk}"
    L.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    L.append("    scf.yield " + ", ".join(cur[a] for a in accs) + ", " + ", ".join(c[2] for c in carried) + " : " + ", ".join("vector<8xf32>" for _ in accs) + ", " + ", ".join(c[3] for c in carried) + "\n  }")
    L.append(f'''  %inv_bytes = index.constant 8192 : offset
  %inv_tile = buffer.alloca<workgroup> align(16) %inv_bytes : buffer
  %inv_view = buffer.view %inv_tile[%origin] : buffer -> view<128x16xf32>
  %writer = index.cmp ult, %workitem, %c128 : index
  scf.if %writer {{
    %bw = index.assume %workitem [range(%workitem, 0, 127)] : index
    %inv_row_value = index.add %row_origin, %bw : index
    %inv_row = index.assume %inv_row_value [lt(%inv_row_value, %rows)] : index
    %inverse = view.load %s_view[%inv_row, %c1] : view<[%rows]x2xf32> -> f32
    %inverse16 = vector.splat %inverse : vector<16xf32>
    vector.store %inverse16, %inv_view[%bw, %c0] : vector<16xf32>, view<128x16xf32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %inv0 = vector.fragment.load<result> %inv_view[%wave_row, %c0] shape [%c16, %c16] : view<128x16xf32> -> vector<8xf32>
  %inv1 = vector.fragment.load<result> %inv_view[%wave_row1, %c0] shape [%c16, %c16] : view<128x16xf32> -> vector<8xf32>
  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %tile_first = index.madd %row_origin, %n, %out_origin : index
  %tile_offset = index.scale %tile_first, %four : index, offset -> offset
  %y_view = buffer.view %y[%tile_offset] : buffer -> view<128x{TN}xf32, %y_layout>''')
    for r in range(2):
        for c in range(cfr):
            row = "%wave_row" if r == 0 else "%wave_row1"
            L.append(f"  %oc{r}{c}_c = index.constant {c * 16} : index\n  %oc{r}{c} = index.add %wave_col, %oc{r}{c}_c : index")
            if two_w:
                L.append(f'''  %g{r}{c} = vector.mulf %o0r{r}c{c}, %inv{r} : vector<8xf32>
  %u{r}{c} = vector.mulf %o1r{r}c{c}, %inv{r} : vector<8xf32>
  %sg{r}{c} = vector.siluf<afn> %g{r}{c} : vector<8xf32>
  %v{r}{c} = vector.mulf %sg{r}{c}, %u{r}{c} : vector<8xf32>''')
            else:
                L.append(f"  %v{r}{c}_s = vector.mulf %o0r{r}c{c}, %inv{r} : vector<8xf32>")
                if mode == "acc":
                    L.append(f"  %old{r}{c} = vector.fragment.load<result> %y_view[{row}, %oc{r}{c}] shape [%c16, %c16] : view<128x{TN}xf32, %y_layout> -> vector<8xf32>\n  %v{r}{c} = vector.addf %old{r}{c}, %v{r}{c}_s : vector<8xf32>")
                else:
                    L.append(f"  %v{r}{c} = vector.addf %v{r}{c}_s, %zero8 : vector<8xf32>")
            L.append(f"  vector.fragment.store<result> %v{r}{c}, %y_view[{row}, %oc{r}{c}] shape [%c16, %c16] : vector<8xf32>, view<128x{TN}xf32, %y_layout>")
    L.append("  kernel.return\n}\n")
    return "\n".join(L)



def _split_dequant(fmt, wsrc, out, col):
    full = dequant16(fmt, wsrc, None, out).replace("%w_col,", f"{col},").replace("%w_col :", f"{col} :")
    lines = [l for l in full.strip("\n").split("\n") if l.strip()]
    last = max(i for i, l in enumerate(lines) if ".load " in l)
    addr = lines[: last + 1]
    loads = [(_re.match(r"\s*(%\S+) = ", l).group(1), l.split("-> ")[-1].strip()) for l in addr if ".load " in l]
    return lines, addr, loads

def gemm_big256(fmt, mode):
    """256-row WMMA GEMM: 256 threads (8 waves in a 4 x 2 grid), each wave 64 rows x 64 columns (4 x 4
    16 x 16 accumulators). store/acc: 256 x 128 tiles, swiglu: 256 rows x 64 outputs of each of gate and up.
    Each staged weight value feeds 256 rows, and each wave reuses an A fragment across 4 B fragments and a
    B fragment across 4 A fragments (0.5 LDS fragment reads per MMA; 32 x 64 waves need 0.75).
    LDS: activations 256 x 72 f16 (padded rows, 36 KB) + weights 128 x 72 (18 KB): two workgroups per WGP.
    Thread t stages activation rows t/2 and t/2 + 128 (32 values each) and a 32-value K group of weight
    row t/2 (two 16-value slices) of the gate/up-interleaved weight tile."""
    f = FMT[fmt]
    two_w = mode == "swiglu"
    NT, TW = 256, 128
    TN = TW // 2 if two_w else TW
    cfr = 2 if two_w else 4
    RF = 4                               # 16-row A fragments per wave
    wave_cols = 16 * cfr
    WVR = TW - (TN - wave_cols) - 15
    name = f"{PREFIX}_gemm_{fmt}_{mode}_m256"
    weights = ["w", "w2"] if two_w else ["w"]
    k_min = 256 if f["block_values"] == 256 else 64
    L = []
    L.append(f'''kernel.def export("{name}") @{name}(%rows: index, %n: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cnt = index.constant {NT} : index
  %tn = index.constant {TN} : index
  %row_tiles = index.div %rows, %c256 : index
  %out_tiles = index.div %n, %tn : index
  kernel.launch.config workgroups(%row_tiles, %out_tiles, %c1) workgroup_size(%cnt, %c1, %c1) : index
}} launch(%rows_arg: index, %k_arg: index, %n_arg: index, %x: buffer, %scales: buffer, {", ".join(f"%{w}: buffer" for w in weights)}, %y: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 256, 8192), multiple_of(%rows_arg, 256)] : index
  %k = index.assume %k_arg [range(%k_arg, {k_min}, 65536), multiple_of(%k_arg, {k_min})] : index
  %n = index.assume %n_arg [range(%n_arg, {TN}, 262144), multiple_of(%n_arg, {TN})] : index
  %origin = index.constant 0 : offset
  %two = index.constant 2 : offset
  %sixteen = index.constant 16 : offset
  %four = index.constant 4 : offset
  %block_bytes = index.constant {f["block_bytes"]} : offset
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
  %tn = index.constant {TN} : index
  %c144_off = index.constant 144 : offset
  %nibble_mask = vector.constant 252645135 : vector<4xi32>
  %row_tile_count = index.div %rows, %c256 : index
  %out_tile_count = index.div %n, %tn : index
  %row_group_id = kernel.workgroup.id<x> : index
  %out_group_id = kernel.workgroup.id<y> : index
  %row_group = index.assume %row_group_id [lt(%row_group_id, %row_tile_count)] : index
  %out_group = index.assume %out_group_id [lt(%out_group_id, %out_tile_count)] : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, {NT - 1})] : index
  %subgroup = kernel.subgroup.id : index
  %bounded_subgroup = index.assume %subgroup [range(%subgroup, 0, {NT // 32 - 1})] : index
  %lane = kernel.subgroup.lane.id : index
  %lane_row = index.rem %lane, %c16 : index
  %wave_row_group = index.div %bounded_subgroup, %c2 : index
  %wave_col_group = index.rem %bounded_subgroup, %c2 : index
  %wave_row_value = index.mul %wave_row_group, %c64 : index
  %wave_row = index.assume %wave_row_value [range(%wave_row_value, 0, 192)] : index''')
    for r in range(1, RF):
        L.append(f"  %wave_row{r}_c = index.constant {16 * r} : index\n  %wave_row{r} = index.add %wave_row, %wave_row{r}_c : index")
    L.append(f'''  %wave_cols = index.constant {wave_cols} : index
  %wave_col_value = index.mul %wave_col_group, %wave_cols : index
  %wave_col = index.assume %wave_col_value [range(%wave_col_value, 0, {TN - wave_cols})] : index
  %row_origin = index.mul %row_group, %c256 : index
  %out_origin = index.mul %out_group, %tn : index
  %blocks = index.div %k, %c{f["block_values"]} : index
  %row_bytes = index.scale %blocks, %block_bytes : index, offset -> offset
  %x_view = buffer.view %x[%origin] : buffer -> view<[%rows]x[%k]xf16>
  %s_view = buffer.view %scales[%origin] : buffer -> view<[%rows]x2xf32>
  %x_bytes = index.constant 36864 : offset
  %x_stage = buffer.alloca<workgroup> align(16) %x_bytes : buffer
  %x_stage_view = buffer.view %x_stage[%origin] : buffer -> view<256x72xf16>
  %w_bytes = index.constant {TW * 144} : offset
  %w_stage = buffer.alloca<workgroup> align(16) %w_bytes : buffer
  %w_stage_view = buffer.view %w_stage[%origin] : buffer -> view<{TW}x72xf16>
  %x_stage_row = index.div %workitem, %c2 : index
  %x_stage_row1 = index.add %x_stage_row, %c128 : index
  %x_half = index.rem %workitem, %c2 : index
  %x_stage_col = index.mul %x_half, %c32 : index
  %x_row_value = index.add %row_origin, %x_stage_row : index
  %x_row = index.assume %x_row_value [lt(%x_row_value, %rows)] : index
  %x_row1_value = index.add %x_row, %c128 : index
  %x_row1 = index.assume %x_row1_value [lt(%x_row1_value, %rows)] : index
  %stage_row = index.div %workitem, %c2 : index
  %stage_ks = index.rem %workitem, %c2 : index
  %stage_col = index.mul %stage_ks, %c32 : index
  %stage_col1 = index.add %stage_col, %c16 : index''')
    if two_w:
        L.append(f'''  %half_tw = index.constant {TN} : index
  %w_is_up = index.cmp uge, %stage_row, %half_tw : index
  %w_local_up = index.sub %stage_row, %half_tw : index
  %w_local = scf.select %w_is_up, %w_local_up, %stage_row : index
  %w_out_row_value = index.add %out_origin, %w_local : index
  %wsel = scf.select %w_is_up, %w2, %w : buffer''')
    else:
        L.append("  %w_out_row_value = index.add %out_origin, %stage_row : index")
    L.append(f'''  %w_out_row = index.assume %w_out_row_value [lt(%w_out_row_value, %n)] : index
  %w_row_origin = index.scale %w_out_row, %row_bytes : index, offset -> offset
  %zero8 = vector.constant 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %k_steps = index.div %k, %c64 : index
  %wb_row = index.add %wave_col, %lane_row : index
  %wb_off = index.scale %wb_row, %c144_off : index, offset -> offset
  %w_wave_view = buffer.view %w_stage[%wb_off] : buffer -> view<{WVR}x72xf16>''')
    wsrc = "wsel" if two_w else "w"
    slices = [("dqa", "%wcola", "%stage_col"), ("dqb", "%wcolb", "%stage_col1")]
    parts = [(out, col, sc) + _split_dequant(fmt, wsrc, out, col) for out, col, sc in slices]
    def addr_text(sfx, kvar):
        t = []
        for out, col, sc, lines, addr, loads in parts:
            c2 = col + sfx
            t.append(f"    {c2} = index.add {kvar}, {sc} : index")
            body = "\n".join(addr).replace(col + " ", c2 + " ").replace(col + ",", c2 + ",")
            body = _re.sub(rf"%{out}_", f"%{out}{sfx}_", body)
            t.append(body)
        if sfx == "p":
            t.append(f"""    %x_colp = index.add {kvar}, %x_stage_col : index
    %x_rawp = vector.load %x_view[%x_row, %x_colp] : view<[%rows]x[%k]xf16> -> vector<32xf16>
    %x_rawp1 = vector.load %x_view[%x_row1, %x_colp] : view<[%rows]x[%k]xf16> -> vector<32xf16>""")
        else:
            t.append(f"""    %x_more = index.cmp ult, %x_next_step, %k_steps : index
    %x_rawn, %x_rawn1 = scf.if %x_more -> (vector<32xf16>, vector<32xf16>) {{
      %x_ns = index.assume %x_next_step [lt(%x_next_step, %k_steps)] : index
      %x_ko = index.mul %x_ns, %c64 : index
      %x_coln = index.add %x_ko, %x_stage_col : index
      %xv = vector.load %x_view[%x_row, %x_coln] : view<[%rows]x[%k]xf16> -> vector<32xf16>
      %xv1 = vector.load %x_view[%x_row1, %x_coln] : view<[%rows]x[%k]xf16> -> vector<32xf16>
      scf.yield %xv, %xv1 : vector<32xf16>, vector<32xf16>
    }} else {{
      scf.yield %cr_x, %cr_x1 : vector<32xf16>, vector<32xf16>
    }}""")
        return "\n".join(t)
    carried = []
    for out, col, sc, lines, addr, loads in parts:
        for nm, ty in loads:
            base = nm[1:]
            carried.append((f"%cr_{base}", f"%{base.replace(out + '_', out + 'p_', 1)}", f"%{base.replace(out + '_', out + 'n_', 1)}", ty))
    carried.append(("%cr_x", "%x_rawp", "%x_rawn", "vector<32xf16>"))
    carried.append(("%cr_x1", "%x_rawp1", "%x_rawn1", "vector<32xf16>"))
    accs = [f"{wi}r{r}c{c}" for wi in range(len(weights)) for r in range(RF) for c in range(cfr)]
    L.append(addr_text("p", "%c0").replace("    ", "  ", 1))
    L.append("  " + ", ".join(f"%o{a}" for a in accs) + ", " + ", ".join(f"%unused_{i}" for i in range(len(carried))) + " = scf.for %step = [%c0 to %k_steps step %c1](" +
             ", ".join(f"%i{a} = %init : vector<8xf32>" for a in accs) + ", " + ", ".join(f"{c[0]} = {c[1]} : {c[3]}" for c in carried) + ") -> (" +
             ", ".join("vector<8xf32>" for _ in accs) + ", " + ", ".join(c[3] for c in carried) + ") {")
    L.append("""    %k_origin = index.mul %step, %c64 : index
    vector.store %cr_x, %x_stage_view[%x_stage_row, %x_stage_col] : vector<32xf16>, view<256x72xf16>
    vector.store %cr_x1, %x_stage_view[%x_stage_row1, %x_stage_col] : vector<32xf16>, view<256x72xf16>""")
    for out, col, sc, lines, addr, loads in parts:
        L.append(f"    {col} = index.add %k_origin, {sc} : index")
        text = "\n".join(l for l in lines if ".load " not in l)
        for nm, ty in loads:
            text = _re.sub(_re.escape(nm) + r"\b", f"%cr_{nm[1:]}", text)
        L.append(text)
        L.append(f"    vector.store %{out}, %w_stage_view[%stage_row, {sc}] : vector<16xf16>, view<{TW}x72xf16>")
    L.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    L.append("    %k_next_raw = index.add %k_origin, %c64 : index\n    %k_limit = index.sub %k, %c64 : index\n    %k_next = index.min %k_next_raw, %k_limit : index\n    %x_next_step = index.add %step, %c1 : index")
    L.append(addr_text("n", "%k_next"))
    cur = {a: f"%i{a}" for a in accs}
    rowv = ["%wave_row"] + [f"%wave_row{r}" for r in range(1, RF)]
    for kk in range(4):
        L.append(f"    %fk{kk} = index.constant {kk * 16} : index")
        for r in range(RF):
            L.append(f"    %l{r}_{kk} = vector.fragment.load<lhs> %x_stage_view[{rowv[r]}, %fk{kk}] shape [%c16, %c16] : view<256x72xf16> -> vector<16xf16>")
        for wi in range(len(weights)):
            for c in range(cfr):
                L.append(f'''    %wr{wi}{c}_{kk} = index.constant {wi * TN + c * 16} : index
    %rv{wi}{c}_{kk} = vector.load %w_wave_view[%wr{wi}{c}_{kk}, %fk{kk}] : view<{WVR}x72xf16> -> vector<16xf16>
    %rf{wi}{c}_{kk} = vector.fragment<rhs> %rv{wi}{c}_{kk} shape [%c16, %c16] : vector<16xf16>''')
        for wi in range(len(weights)):
            for c in range(cfr):
                for r in range(RF):
                    a = f"{wi}r{r}c{c}"
                    L.append(f"    %m{a}_{kk} = vector.mma %l{r}_{kk}, %rf{wi}{c}_{kk}, {cur[a]} : vector<16xf16>, vector<16xf16>, vector<8xf32>")
                    cur[a] = f"%m{a}_{kk}"
    L.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    L.append("    scf.yield " + ", ".join(cur[a] for a in accs) + ", " + ", ".join(c[2] for c in carried) + " : " + ", ".join("vector<8xf32>" for _ in accs) + ", " + ", ".join(c[3] for c in carried) + "\n  }")
    L.append(f'''  %inv_view = buffer.view %x_stage[%origin] : buffer -> view<256x16xf32>
  %inv_row_value = index.add %row_origin, %workitem : index
  %inv_row = index.assume %inv_row_value [lt(%inv_row_value, %rows)] : index
  %inverse = view.load %s_view[%inv_row, %c1] : view<[%rows]x2xf32> -> f32
  %inverse16 = vector.splat %inverse : vector<16xf32>
  vector.store %inverse16, %inv_view[%workitem, %c0] : vector<16xf32>, view<256x16xf32>
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)''')
    for r in range(RF):
        L.append(f"  %inv{r} = vector.fragment.load<result> %inv_view[{rowv[r]}, %c0] shape [%c16, %c16] : view<256x16xf32> -> vector<8xf32>")
    L.append(f'''  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %tile_first = index.madd %row_origin, %n, %out_origin : index
  %tile_offset = index.scale %tile_first, %four : index, offset -> offset
  %y_view = buffer.view %y[%tile_offset] : buffer -> view<256x{TN}xf32, %y_layout>''')
    for r in range(RF):
        for c in range(cfr):
            row = rowv[r]
            L.append(f"  %oc{r}{c}_c = index.constant {c * 16} : index\n  %oc{r}{c} = index.add %wave_col, %oc{r}{c}_c : index")
            if two_w:
                L.append(f'''  %g{r}{c} = vector.mulf %o0r{r}c{c}, %inv{r} : vector<8xf32>
  %u{r}{c} = vector.mulf %o1r{r}c{c}, %inv{r} : vector<8xf32>
  %sg{r}{c} = vector.siluf<afn> %g{r}{c} : vector<8xf32>
  %v{r}{c} = vector.mulf %sg{r}{c}, %u{r}{c} : vector<8xf32>''')
            else:
                L.append(f"  %v{r}{c}_s = vector.mulf %o0r{r}c{c}, %inv{r} : vector<8xf32>")
                if mode == "acc":
                    L.append(f"  %old{r}{c} = vector.fragment.load<result> %y_view[{row}, %oc{r}{c}] shape [%c16, %c16] : view<256x{TN}xf32, %y_layout> -> vector<8xf32>\n  %v{r}{c} = vector.addf %old{r}{c}, %v{r}{c}_s : vector<8xf32>")
                else:
                    L.append(f"  %v{r}{c} = vector.addf %v{r}{c}_s, %zero8 : vector<8xf32>")
            L.append(f"  vector.fragment.store<result> %v{r}{c}, %y_view[{row}, %oc{r}{c}] shape [%c16, %c16] : vector<8xf32>, view<256x{TN}xf32, %y_layout>")
    L.append("  kernel.return\n}\n")
    return "\n".join(L)


def gemm_q8(fmt, mode):
    """llama.cpp-style int8 prefill GEMM (mmq): Q8_1 activations (g_quant_q8: int8 codes per 32 values, then a
    (d, sum q) table) times Q4_K weight codes on int8 WMMA, f32 per 32-value block:
        y += (sum_j qw_j qx_j) * dx * (d sc) - (dx sum_j qx_j) * (dmin m)      (llama.cpp vec_dot_q4_K_q8_1)
    128-row tiles (rows multiple of 128); store: 128 x 128 outputs, swiglu: 128 x 64 of each of gate and up.
    A 64-wide K step is one 32-byte page of a Q4_K superblock: its low nibbles are group 2p, high nibbles 2p+1.
    Waves: 4 (32 rows) x 2 (half the output columns); a lane's result column is lane % 16, its 8 elements rows
    2i + lane / 16 (the RDNA3 wave32 result layout), so the per-row activation scales are an 8-wide gather.
    No row scale: the activations are quantized from the f32 input itself."""
    assert fmt == "q4_k"
    two_w = mode == "swiglu"
    TN = 64 if two_w else 128
    WC = TN // 2                    # output columns per wave (per weight)
    CF = WC // 16                   # 16-column fragments per wave per weight
    weights = ["w", "w2"] if two_w else ["w"]
    name = f"{PREFIX}_gemm_{fmt}_{mode}_q8"
    XS = 80                         # padded LDS row (bytes) for 64 int8 codes
    wargs = ", ".join(f"%{w}: buffer" for w in weights)
    L = []
    A = L.append
    A(f"""kernel.def export("{name}") @{name}(%rows: index, %n: index) {{
  %c1 = index.constant 1 : index
  %c128 = index.constant 128 : index
  %tn = index.constant {TN} : index
  %c256 = index.constant 256 : index
  %row_tiles = index.div %rows, %c128 : index
  %out_tiles = index.div %n, %tn : index
  kernel.launch.config workgroups(%out_tiles, %row_tiles, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%rows_arg: index, %k_arg: index, %n_arg: index, %xq: buffer, {wargs}, %y: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 128, 8192), multiple_of(%rows_arg, 128)] : index
  %k = index.assume %k_arg [range(%k_arg, 512, 65536), multiple_of(%k_arg, 512)] : index
  %n = index.assume %n_arg [range(%n_arg, {TN}, 262144), multiple_of(%n_arg, {TN})] : index
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c7 = index.constant 7 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %c128 = index.constant 128 : index
  %c256 = index.constant 256 : index
  %tn = index.constant {TN} : index
  %row_tile_count = index.div %rows, %c128 : index
  %out_tile_count = index.div %n, %tn : index
  %out_group_id = kernel.workgroup.id<x> : index
  %row_group_id = kernel.workgroup.id<y> : index
  %row_group = index.assume %row_group_id [lt(%row_group_id, %row_tile_count)] : index
  %out_group = index.assume %out_group_id [lt(%out_group_id, %out_tile_count)] : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, 255)] : index
  %subgroup = kernel.subgroup.id : index
  %sg = index.assume %subgroup [range(%subgroup, 0, 7)] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %lane_col = index.rem %lane, %c16 : index
  %lane_half = index.div %lane, %c16 : index
  %wave_row_group = index.div %sg, %c2 : index
  %wave_col_group = index.rem %sg, %c2 : index
  %wave_row_value = index.mul %wave_row_group, %c32 : index
  %wave_row = index.assume %wave_row_value [range(%wave_row_value, 0, 96)] : index
  %cwc = index.constant {WC} : index
  %wave_col_value = index.mul %wave_col_group, %cwc : index
  %wave_col = index.assume %wave_col_value [range(%wave_col_value, 0, {TN - WC})] : index
  %row_origin = index.mul %row_group, %c128 : index
  %out_origin = index.mul %out_group, %tn : index
  %groups = index.div %k, %c32 : index
  %blocks = index.div %k, %c256 : index
  %block_bytes = index.constant 144 : offset
  %row_bytes = index.scale %blocks, %block_bytes : index, offset -> offset
  // Activations: codes [rows][groups][32] i8, then at byte rows*k: [rows][groups] x (d, sum q) f32.
  %xc_view = buffer.view %xq[%origin] : buffer -> view<[%rows]x[%groups]x8xi32>
  %k_bytes = index.cast %k : index to offset
  %xm_off = index.scale %rows, %k_bytes : index, offset -> offset
  %xm_view = buffer.view %xq[%xm_off] : buffer -> view<[%rows]x[%groups]x2xf32>
  // LDS: X codes [128][80 B], W codes per weight [{TN}][80 B], dx/sx [2 blocks][128], W (scale, min) [2][{TN}][2].
  %xs_bytes = index.constant {128 * XS} : offset
  %xs = buffer.alloca<workgroup> align(16) %xs_bytes : buffer
  %xs_view = buffer.view %xs[%origin] : buffer -> view<128x{XS // 4}xi32>
  %xs8 = buffer.view %xs[%origin] : buffer -> view<128x{XS}xi8>
  %xd_bytes = index.constant 1024 : offset
  %xd = buffer.alloca<workgroup> align(16) %xd_bytes : buffer
  %xd_view = buffer.view %xd[%origin] : buffer -> view<2x128xf32>
  %xsum = buffer.alloca<workgroup> align(16) %xd_bytes : buffer
  %xsum_view = buffer.view %xsum[%origin] : buffer -> view<2x128xf32>
  %xd0 = buffer.view %xd[%origin] : buffer -> view<128xf32>
  %xs0 = buffer.view %xsum[%origin] : buffer -> view<128xf32>
  %half_kb = index.constant 512 : offset
  %xd1 = buffer.view %xd[%half_kb] : buffer -> view<128xf32>
  %xs1 = buffer.view %xsum[%half_kb] : buffer -> view<128xf32>
  // Min term (dmin m)(dx sum qx) folded with one f16 WMMA per tile every 16 blocks: ring buffers, f16.
  %sxr_bytes = index.constant {128 * 48} : offset
  %sxr = buffer.alloca<workgroup> align(16) %sxr_bytes : buffer
  %sxr_view = buffer.view %sxr[%origin] : buffer -> view<128x24xf16>
  %mnr_bytes = index.constant {TN * 48} : offset
  %ws_bytes = index.constant {TN * XS} : offset
  %wm_bytes = index.constant {TN * 16} : offset""")
    for w in weights:
        A(f"""  %{w}_s = buffer.alloca<workgroup> align(16) %ws_bytes : buffer
  %{w}_sv = buffer.view %{w}_s[%origin] : buffer -> view<{TN}x{XS // 4}xi32>
  %{w}_s8 = buffer.view %{w}_s[%origin] : buffer -> view<{TN}x{XS}xi8>
  %{w}_m = buffer.alloca<workgroup> align(16) %wm_bytes : buffer
  %{w}_mv = buffer.view %{w}_m[%origin] : buffer -> view<{TN}x2x2xf32>
  %{w}_mr = buffer.alloca<workgroup> align(16) %mnr_bytes : buffer
  %{w}_mrv = buffer.view %{w}_mr[%origin] : buffer -> view<{TN}x24xf16>""")
    # staging assignment: X: thread t -> row t/2, half t%2 (32 codes + one (d, s)).
    # W: store: thread t -> row t/2, nibble half t%2; swiglu: threads 0-127 gate, 128-255 up, row (t%128)/2.
    A(f"""  %x_srow = index.div %workitem, %c2 : index
  %x_half = index.rem %workitem, %c2 : index
  %x_grow_value = index.add %row_origin, %x_srow : index
  %x_grow = index.assume %x_grow_value [lt(%x_grow_value, %rows)] : index
  %x_col_words = index.mul %x_half, %c8 : index
  %w_t = index.rem %workitem, %c_wt : index
  %w_srow_value = index.div %w_t, %c2 : index
  %w_srow = index.assume %w_srow_value [range(%w_srow_value, 0, {TN - 1})] : index
  %w_half = index.rem %w_t, %c2 : index
  %w_grow_value = index.add %out_origin, %w_srow : index
  %w_grow = index.assume %w_grow_value [lt(%w_grow_value, %n)] : index
  %w_row_origin = index.scale %w_grow, %row_bytes : index, offset -> offset
  %w_shift_bits = index.mul %w_half, %c4 : index
  %w_shift_i = index.cast %w_shift_bits : index to i32
  %w_shift = vector.splat %w_shift_i : vector<8xi32>
  %nib = vector.constant 252645135 : vector<8xi32>
  %w_col_words = index.mul %w_half, %c8 : index""")
    if two_w:
        A("  %c128i = index.constant 128 : index\n  %is_up = index.cmp uge, %workitem, %c128i : index")
    A(f"""  %schema = encoding.define #encoding.operand<element_format=i8, payload_elements=16, payload_registers=4> : encoding<schema>
  %zf = vector.constant 0.0 : vector<8xf32>
  %finit = vector.fragment<init> %zf shape [%c16, %c16] : vector<8xf32>
  %zi = vector.constant 0 : vector<8xi32>
  %iinit = vector.fragment<init> %zi shape [%c16, %c16] : vector<8xi32>
  %two_i = scalar.constant 2 : i32
  %k_steps = index.div %k, %c64 : index""")
    accs = [f"a{wi}r{r}c{c}" for wi in range(len(weights)) for r in range(2) for c in range(CF)]
    V8 = "vector<8xf32>"
    A("  " + ", ".join(f"%o{a}" for a in accs) + " = scf.for %step = [%c0 to %k_steps step %c1](" +
      ", ".join(f"%i{a} = %finit : {V8}" for a in accs) + ") -> (" + ", ".join(V8 for _ in accs) + ") {")
    # ---- stage X
    A(f"""    %g0 = index.mul %step, %c2 : index
    %xg_value = index.add %g0, %x_half : index
    %xg = index.assume %xg_value [lt(%xg_value, %groups)] : index
    %xcodes = vector.load %xc_view[%x_grow, %xg, %c0] : view<[%rows]x[%groups]x8xi32> -> vector<8xi32>
    vector.store %xcodes, %xs_view[%x_srow, %x_col_words] : vector<8xi32>, view<128x{XS // 4}xi32>
    %xmeta = vector.load %xm_view[%x_grow, %xg, %c0] : view<[%rows]x[%groups]x2xf32> -> vector<2xf32>
    %xdv = vector.extract %xmeta[0] : vector<2xf32> -> f32
    %xqs = vector.extract %xmeta[1] : vector<2xf32> -> f32
    %xsv = scalar.mulf %xdv, %xqs : f32
    %xsr = index.assume %x_srow [range(%x_srow, 0, 127)] : index
    view.store %xdv, %xd_view[%x_half, %xsr] : f32, view<2x128xf32>
    %ring_a = index.mul %step, %c2 : index
    %ring_b = index.add %ring_a, %x_half : index
    %ring_x = index.rem %ring_b, %c16 : index
    %xsvh = scalar.fptrunc %xsv : f32 to f16
    view.store %xsvh, %sxr_view[%xsr, %ring_x] : f16, view<128x24xf16>
    // ---- stage W (Q4_K page p = step % 4 of superblock step / 4)
    %super = index.div %step, %c4 : index
    %page = index.rem %step, %c4 : index
    %super_bytes = index.scale %super, %block_bytes : index, offset -> offset
    %page_words_a = index.mul %page, %c8 : index
    %group_a = index.mul %page, %c2 : index
    %group_value = index.add %group_a, %w_half : index
    %group = index.assume %group_value [range(%group_value, 0, 7)] : index""")
    def stage_w(w):
        return f"""      %{w}_bo = index.add %w_row_origin, %super_bytes : offset
      %{w}_hv = buffer.view %{w}[%{w}_bo] : buffer -> view<4xi32>
      %{w}_hdr = vector.load %{w}_hv[0] : view<4xi32> -> vector<4xi32>
      %{w}_sc, %{w}_mn = func.call {META}(%{w}_hdr, %group) : (vector<4xi32>, index) -> (f32, f32)
      %{w}_po = index.add %{w}_bo, %sixteen : offset
      %{w}_pv = buffer.view %{w}[%{w}_po] : buffer -> view<32xi32>
      %{w}_pk = vector.load %{w}_pv[%page_words_a] : view<32xi32> -> vector<8xi32>
      %{w}_sh = vector.shrui %{w}_pk, %w_shift : vector<8xi32>
      %{w}_cd = vector.andi %{w}_sh, %nib : vector<8xi32>
      vector.store %{w}_cd, %{w}_sv[%w_srow, %w_col_words] : vector<8xi32>, view<{TN}x{XS // 4}xi32>
      %{w}_meta = vector.from_elements %{w}_sc, %{w}_mn : vector<2xf32>
      vector.store %{w}_meta, %{w}_mv[%w_srow, %w_half, %c0] : vector<2xf32>, view<{TN}x2x2xf32>
      %{w}_nmn = scalar.negf %{w}_mn : f32
      %{w}_nmnh = scalar.fptrunc %{w}_nmn : f32 to f16
      %{w}_ra = index.mul %step, %c2 : index
      %{w}_rb = index.add %{w}_ra, %w_half : index
      %{w}_rc = index.rem %{w}_rb, %c16 : index
      view.store %{w}_nmnh, %{w}_mrv[%w_srow, %{w}_rc] : f16, view<{TN}x24xf16>"""
    if two_w:
        A("    scf.if %is_up {\n" + stage_w("w2") + "\n    } else {\n" + stage_w("w") + "\n    }")
    else:
        A("    scf.if %true_c {\n" + stage_w("w") + "\n    }")
    A("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    cur = {a: f"%i{a}" for a in accs}
    for b in range(2):     # 32-value block within the step
        xdv, xsv = ("%xd0", "%xs0") if b == 0 else ("%xd1", "%xs1")
        A("".join(f"    %lc{b}{kk} = index.constant {b * 32 + kk * 16} : index\n" for kk in range(2)).rstrip("\n"))
        for r in range(2):
            A(f"""    %rb{b}{r}_a = index.constant {r * 16} : index
    %rb{b}{r}_b = index.add %wave_row, %rb{b}{r}_a : index
    %rb{b}{r}_c = index.add %rb{b}{r}_b, %lane_half : index
    %rb{b}{r} = index.assume %rb{b}{r}_c [range(%rb{b}{r}_c, 0, 113)] : index
    %dx{b}{r} = vector.gather {xdv}[%rb{b}{r}][%gather_off] : view<128xf32>, vector<8xindex> -> vector<8xf32>""")
            for kk in range(2):
                col = b * 32 + kk * 16
                A(f"""    %lr{b}{r}{kk}_a = index.add %wave_row, %rb{b}{r}_a : index
    %lr{b}{r}{kk} = index.add %lr{b}{r}{kk}_a, %lane_col : index
    %lb{b}{r}{kk} = vector.load %xs8[%lr{b}{r}{kk}, %lc{b}{kk}] : view<128x{XS}xi8> -> vector<16xi8>
    %lp{b}{r}{kk} = vector.bitcast %lb{b}{r}{kk} : vector<16xi8> to vector<4xi32>
    %lf{b}{r}{kk} = vector.fragment<lhs> %lp{b}{r}{kk} shape [%c16, %c16] using {{schema = %schema : encoding<schema>}} : vector<4xi32>""")
        for wi, w in enumerate(weights):
            for c in range(CF):
                A(f"""    %wc{b}{wi}{c}_a = index.constant {c * 16} : index
    %wc{b}{wi}{c}_b = index.add %wave_col, %wc{b}{wi}{c}_a : index
    %wc{b}{wi}{c} = index.add %wc{b}{wi}{c}_b, %lane_col : index
    %wcb{b}{wi}{c} = index.assume %wc{b}{wi}{c} [range(%wc{b}{wi}{c}, 0, {TN - 1})] : index
    %wsc{b}{wi}{c} = view.load %{w}_mv[%wcb{b}{wi}{c}, %c{b}_idx, %c0] : view<{TN}x2x2xf32> -> f32
    %wscv{b}{wi}{c} = vector.splat %wsc{b}{wi}{c} : vector<8xf32>""")
                for kk in range(2):
                    col = b * 32 + kk * 16
                    A(f"""    %rb{b}{wi}{c}{kk} = vector.load %{w}_s8[%wcb{b}{wi}{c}, %lc{b}{kk}] : view<{TN}x{XS}xi8> -> vector<16xi8>
    %rp{b}{wi}{c}{kk} = vector.bitcast %rb{b}{wi}{c}{kk} : vector<16xi8> to vector<4xi32>
    %rf{b}{wi}{c}{kk} = vector.fragment<rhs> %rp{b}{wi}{c}{kk} shape [%c16, %c16] using {{schema = %schema : encoding<schema>}} : vector<4xi32>""")
                for r in range(2):
                    a = f"a{wi}r{r}c{c}"
                    A(f"""    %im{b}{a}0 = vector.mma %lf{b}{r}0, %rf{b}{wi}{c}0, %iinit : vector<4xi32>, vector<4xi32>, vector<8xi32>
    %im{b}{a}1 = vector.mma %lf{b}{r}1, %rf{b}{wi}{c}1, %im{b}{a}0 : vector<4xi32>, vector<4xi32>, vector<8xi32>
    %fm{b}{a} = vector.sitofp %im{b}{a}1 : vector<8xi32> to vector<8xf32>
    %fx{b}{a} = vector.mulf %fm{b}{a}, %dx{b}{r} : vector<8xf32>
    %f1{b}{a} = vector.fmaf %fx{b}{a}, %wscv{b}{wi}{c}, {cur[a]} : vector<8xf32>""")
                    cur[a] = f"%f1{b}{a}"
    A("    %step_in8 = index.rem %step, %c8 : index\n    %fold = index.cmp eq, %step_in8, %c7 : index")
    A("    " + ", ".join(f"%mf{a}" for a in accs) + " = scf.if %fold -> (" + ", ".join(V8 for _ in accs) + ") {")
    for r in range(2):
        A(f"""      %mr{r}_a = index.constant {r * 16} : index
      %mr{r} = index.add %wave_row, %mr{r}_a : index
      %ml{r} = vector.fragment.load<lhs> %sxr_view[%mr{r}, %c0] shape [%c16, %c16] : view<128x24xf16> -> vector<16xf16>""")
    for wi, w in enumerate(weights):
        for c in range(CF):
            A(f"""      %mc{wi}{c}_a = index.constant {c * 16} : index
      %mc{wi}{c}_b = index.add %wave_col, %mc{wi}{c}_a : index
      %mc{wi}{c} = index.add %mc{wi}{c}_b, %lane_col : index
      %mcv{wi}{c} = vector.load %{w}_mrv[%mc{wi}{c}, %c0] : view<{TN}x24xf16> -> vector<16xf16>
      %mcf{wi}{c} = vector.fragment<rhs> %mcv{wi}{c} shape [%c16, %c16] : vector<16xf16>""")
            for r in range(2):
                a = f"a{wi}r{r}c{c}"
                A(f"      %mm{a} = vector.mma %ml{r}, %mcf{wi}{c}, {cur[a]} : vector<16xf16>, vector<16xf16>, vector<8xf32>")
    A("      scf.yield " + ", ".join(f"%mm{a}" for a in accs) + " : " + ", ".join(V8 for _ in accs))
    A("    } else {\n      scf.yield " + ", ".join(cur[a] for a in accs) + " : " + ", ".join(V8 for _ in accs) + "\n    }")
    A("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    A("    scf.yield " + ", ".join(f"%mf{a}" for a in accs) + " : " + ", ".join(V8 for _ in accs) + "\n  }")
    A(f"""  %c4o = index.constant 4 : offset
  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %tile_first = index.madd %row_origin, %n, %out_origin : index
  %tile_offset = index.scale %tile_first, %c4o : index, offset -> offset
  %y_view = buffer.view %y[%tile_offset] : buffer -> view<128x{TN}xf32, %y_layout>""")
    for r in range(2):
        for c in range(CF):
            A(f"""  %yr{r}{c}_a = index.constant {r * 16} : index
  %yr{r}{c} = index.add %wave_row, %yr{r}{c}_a : index
  %yc{r}{c}_a = index.constant {c * 16} : index
  %yc{r}{c} = index.add %wave_col, %yc{r}{c}_a : index""")
            if two_w:
                A(f"""  %sg{r}{c} = vector.siluf<afn> %oa0r{r}c{c} : vector<8xf32>
  %v{r}{c} = vector.mulf %sg{r}{c}, %oa1r{r}c{c} : vector<8xf32>""")
            else:
                A(f"  %v{r}{c} = vector.addf %oa0r{r}c{c}, %zf : vector<8xf32>")
            A(f"  vector.fragment.store<result> %v{r}{c}, %y_view[%yr{r}{c}, %yc{r}{c}] shape [%c16, %c16] : vector<8xf32>, view<128x{TN}xf32, %y_layout>")
    A("  kernel.return\n}\n")
    t = "\n".join(L)
    consts = ("  %sixteen = index.constant 16 : offset\n  %true_c = scalar.constant true : i1\n  %c0_idx = index.constant 0 : index\n"
              "  %c1_idx = index.constant 1 : index\n  %c_wt = index.constant " + str(2 * TN) + " : index\n"
              "  %gather_off = vector.iota %c0 step %c2 : vector<8xindex>\n")
    return t.replace("  %tn = index.constant " + str(TN) + " : index\n  %row_tile_count", consts + "  %tn = index.constant " + str(TN) + " : index\n  %row_tile_count", 1)

if __name__ == "__main__":
    out = [f"// Generated by gen_gemm.py: {PREFIX} prefill WMMA GEMMs over GGUF weights.\n"]
    spec = os.environ.get("GEMM_FORMATS", "q4_k:store+acc+swiglu,q6_k:store")
    def skip_padding(text):
        # GEMM_SKIP_PADDING: a row tile whose rows all have scale 0 (padding rows, as
        # marked by g_row_scale_f16_live) does no work. Every row is checked: padding
        # can sit between live rows, so the first row alone does not decide the tile.
        anchor = "  %s_view = buffer.view %scales[%origin] : buffer -> view<[%rows]x2xf32>\n"
        assert anchor in text
        tm = next(m for m in (256, 128, 64) if f"%row_origin = index.mul %row_group, %c{m} : index" in text)
        assert f"%row_origin = index.mul %row_group, %c{tm} : index" in text
        check = anchor + f"""  %skip_tid0 = kernel.workitem.id<x> : index
  %skip_tid = index.assume %skip_tid0 [range(%skip_tid0, 0, 511)] : index
  %skip_tm = index.constant {tm} : index
  %skip_in = index.cmp ult, %skip_tid, %skip_tm : index
  %skip_row = index.add %row_origin, %skip_tid : index
  %skip_rc = scf.select %skip_in, %skip_row, %row_origin : index
  %tile_row = index.assume %skip_rc [lt(%skip_rc, %rows)] : index
  %tile_scale = view.load %s_view[%tile_row, %c0] : view<[%rows]x2xf32> -> f32
  %tile_abs = scalar.absf %tile_scale : f32
  %tile_any = kernel.workgroup.reduce<maxnumf> %tile_abs : f32
  %no_scale = scalar.constant 0.0 : f32
  %tile_live = scalar.cmpf one, %tile_any, %no_scale : f32
  scf.if %tile_live {{
"""
        text = text.replace(anchor, check, 1)
        tail = "  kernel.return\n}\n"
        assert text.endswith(tail)
        return text[: -len(tail)] + "  }\n" + tail
    def n_fastest(text):
        # GEMM_N_FASTEST: walk output-column tiles fastest so consecutive
        # workgroups share an A row tile (kept in L2 instead of re-read from DRAM).
        text = text.replace("workgroups(%row_tiles, %out_tiles, %c1)", "workgroups(%out_tiles, %row_tiles, %c1)")
        text = text.replace("  %row_group_id = kernel.workgroup.id<x> : index\n  %out_group_id = kernel.workgroup.id<y> : index",
                            "  %row_group_id = kernel.workgroup.id<y> : index\n  %out_group_id = kernel.workgroup.id<x> : index")
        return text
    on = lambda k: os.environ.get(k, "0") not in ("", "0", "false", "no")  # GEMM_X=0 really means off
    skip = skip_padding if on("GEMM_SKIP_PADDING") else (lambda t: t)
    order = n_fastest if on("GEMM_N_FASTEST") else (lambda t: t)
    post = lambda t: order(skip(t))
    on256 = os.environ.get("GEMM_M256", "1") not in ("", "0", "false", "no")   # 256-row kernels (_m256)
    for item in spec.split(","):
        fmt, modes = item.split(":")
        for mode in modes.split("+"):
            out.append(post(gemm(fmt, mode)))
            if fmt in HELPER:   # the 64-row tile kernel under the 128-row name (its dequant has no prefetchable load)
                name = f"{PREFIX}_gemm_{fmt}_{mode}"
                out.append(post(gemm(fmt, mode)).replace(f'export("{name}") @{name}(', f'export("{name}_m128") @{name}_m128('))
            else:
                out.append(post(gemm_big(fmt, mode)))
                if on256:
                    out.append(post(gemm_big256(fmt, mode)))
    sys.stdout.write("\n".join(out))
