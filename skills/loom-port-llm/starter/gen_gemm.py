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
    "ptq1_0": dict(block_bytes=28, block_values=128),
    "q2_0": dict(block_bytes=18, block_values=64),
    "q5_0": dict(block_bytes=22, block_values=32),
    "iq4_nl": dict(block_bytes=18, block_values=32),
    "iq4_xs": dict(block_bytes=136, block_values=256),
    "q3_k": dict(block_bytes=110, block_values=256),
    "iq3_s": dict(block_bytes=110, block_values=256),
    "f16": dict(block_bytes=64, block_values=32),
    "mxfp4": dict(block_bytes=17, block_values=32),
    # MXFP4 split planes (safetensors, Quark): the kernel takes the nibble plane as %w (16 bytes per 32 values) and the E8M0
    # scale plane as %ws (one byte per 32 values); @g_mxfp4s_values reads both.
    "mxfp4s": dict(block_bytes=16, block_values=32),
    # BF16 (vendor drafters published only in BF16, used as-is): 16 values load directly, rounded to F16 for the MMA
    "bf16": dict(block_bytes=64, block_values=32),
    # F32 (small projections some quants keep unquantized, e.g. Gemma 3n/4 per-layer-input gate/proj): 16 values load
    # directly, rounded to F16 for the MMA
    "f32": dict(block_bytes=128, block_values=32),
}
def bv_name(bv):
    """The index constant of a format's block size: %c32 / %c256 exist in every GEMM kernel; another size (PTQ1_0: 128)
    gets its own constant (bv_const), so the other formats' kernels stay byte-identical."""
    return f"%c{bv}" if bv in (32, 256) else f"%cbv{bv}"


def bv_const(bv):
    return "" if bv in (32, 256) else f"  %cbv{bv} = index.constant {bv} : index\n"


HELPER = {"q4_0", "q5_0", "iq4_nl", "iq4_xs", "q3_k", "iq3_s", "f16", "mxfp4", "mxfp4s", "ptq1_0", "q2_0"}
# GEMM_Q4_0_NATIVE (choice prefill.gemm.q4_0_native): Q4_0 decodes its 16-value half-block straight to F16 (moe_q4_0.dequant16,
# the same values as @g_q4_0_values + fptrunc) and so gets the 128/256-row kernels instead of the 64-row tile only.
Q4_0_NATIVE = os.environ.get("GEMM_Q4_0_NATIVE", "0").strip().lower() not in ("", "0", "false", "no", "off")
if Q4_0_NATIVE:
    HELPER = HELPER - {"q4_0"}
# GEMM_Q2_0_NATIVE (choice prefill.gemm.q2_0_native): Q2_0 (Ternary Bonsai) decodes its 16-value quarter-block straight to F16
# (ternary.q2_0_dequant16: one 4-byte code load, word-wide shift-and-mask, the values of @g_q2_0_values + fptrunc) and so gets the
# 128/256-row kernels instead of the 64-row tile that decodes a whole 32-value group per 16 weights.
Q2_0_NATIVE = os.environ.get("GEMM_Q2_0_NATIVE", "0").strip().lower() not in ("", "0", "false", "no", "off")
if Q2_0_NATIVE:
    HELPER = HELPER - {"q2_0"}
# GEMM_IQ4_NATIVE (choice prefill.gemm.iq4_native): IQ4_XS / IQ4_NL decode their 16-value half-group inline (iq4_dequant16: the
# loads exposed, so the 128/256-row kernels prefetch them and the mixed-format m256 swiglu takes iq4_xs) instead of through the
# @g_<fmt>_values helper, which limits them to the 64-row tile. Same values as the helper + fptrunc.
IQ4_NATIVE = os.environ.get("GEMM_IQ4_NATIVE", "0").strip().lower() not in ("", "0", "false", "no", "off")
if IQ4_NATIVE:
    HELPER = HELPER - {"iq4_xs", "iq4_nl"}

# GEMM_Q3_K_NATIVE (choice prefill.gemm.q3_k_native): Q3_K decodes its 16-value half-group inline (q3_k_dequant16) and gets the
# 128/256-row kernels and the mixed-format swiglu m256 (unsloth UD: one ffn_up); off: @g_q3_k_values and the 64/128-row tiles.
Q3_K_NATIVE = os.environ.get("GEMM_Q3_K_NATIVE", "0").strip().lower() not in ("", "0", "false", "no", "off")
if Q3_K_NATIVE:
    HELPER = HELPER - {"q3_k"}


def q3_k_dequant16(w, out):
    """Q3_K (110-byte superblocks of 256: hmask[32], qs[64], scales[12], F16 d): 16 consecutive weights of row %w_row_origin at
    column %w_col -> F16 vector %{out}, all loads first. Half h of group g (scale 2g + h): l = 16h..16h+15,
    q = ((qs[32 (g/4) + l] >> 2 (g%4)) & 3) | ((hmask[l] >> g) & 1) << 2, value (q - 4) * d * (scale - 32) in F32 then fptrunc
    (the arithmetic of @g_q3_k_values / @g_q3_k_scale)."""
    o = out
    return f'''
      %{o}_c4 = index.constant 4 : index
      %{o}_c8 = index.constant 8 : index
      %{o}_c256 = index.constant 256 : index
      %{o}_block = index.div %w_col, %{o}_c256 : index
      %{o}_within = index.rem %w_col, %{o}_c256 : index
      %{o}_group_value = index.div %{o}_within, %c32 : index
      %{o}_group = index.assume %{o}_group_value [range(%{o}_group_value, 0, 7)] : index
      %{o}_in_group = index.rem %{o}_within, %c32 : index
      %{o}_half = index.div %{o}_in_group, %c16 : index
      %{o}_l = index.mul %{o}_half, %c16 : index
      %{o}_n = index.div %{o}_group, %{o}_c4 : index
      %{o}_j = index.rem %{o}_group, %{o}_c4 : index
      %{o}_qs_pos0 = index.madd %{o}_n, %c32, %c32 : index
      %{o}_qs_pos = index.add %{o}_qs_pos0, %{o}_l : index
      %{o}_s0 = index.mul %{o}_group, %c2 : index
      %{o}_s_value = index.add %{o}_s0, %{o}_half : index
      %{o}_s = index.assume %{o}_s_value [range(%{o}_s_value, 0, 15)] : index
      %{o}_lo_idx = index.rem %{o}_s, %{o}_c8 : index
      %{o}_nib = index.div %{o}_s, %{o}_c8 : index
      %{o}_hi_idx0 = index.rem %{o}_s, %{o}_c4 : index
      %{o}_hi_sh = index.div %{o}_s, %{o}_c4 : index
      %{o}_c96 = index.constant 96 : index
      %{o}_c104 = index.constant 104 : index
      %{o}_lo_pos = index.add %{o}_c96, %{o}_lo_idx : index
      %{o}_hi_pos = index.add %{o}_c104, %{o}_hi_idx0 : index
      %{o}_block_add = index.scale %{o}_block, %block_bytes : index, offset -> offset
      %{o}_block_origin = index.add %w_row_origin, %{o}_block_add : offset
      %{o}_l_off = index.cast %{o}_l : index to offset
      %{o}_qs_off = index.cast %{o}_qs_pos : index to offset
      %{o}_lo_off = index.cast %{o}_lo_pos : index to offset
      %{o}_hi_off = index.cast %{o}_hi_pos : index to offset
      %{o}_hm_origin = index.add %{o}_block_origin, %{o}_l_off : offset
      %{o}_qs_origin = index.add %{o}_block_origin, %{o}_qs_off : offset
      %{o}_lo_origin = index.add %{o}_block_origin, %{o}_lo_off : offset
      %{o}_hi_origin = index.add %{o}_block_origin, %{o}_hi_off : offset
      %{o}_d_bytes = index.constant 108 : offset
      %{o}_d_origin = index.add %{o}_block_origin, %{o}_d_bytes : offset
      %{o}_hm_view = buffer.view %{w}[%{o}_hm_origin] : buffer -> view<16xi8>
      %{o}_qs_view = buffer.view %{w}[%{o}_qs_origin] : buffer -> view<16xi8>
      %{o}_lo_view = buffer.view %{w}[%{o}_lo_origin] : buffer -> view<1xi8>
      %{o}_hi_view = buffer.view %{w}[%{o}_hi_origin] : buffer -> view<1xi8>
      %{o}_d_view = buffer.view %{w}[%{o}_d_origin] : buffer -> view<1xf16>
      %{o}_hm = vector.load %{o}_hm_view[0] : view<16xi8> -> vector<16xi8>
      %{o}_qs = vector.load %{o}_qs_view[0] : view<16xi8> -> vector<16xi8>
      %{o}_lo_b = view.load %{o}_lo_view[0] : view<1xi8> -> i8
      %{o}_hi_b = view.load %{o}_hi_view[0] : view<1xi8> -> i8
      %{o}_d_half = view.load %{o}_d_view[0] : view<1xf16> -> f16
      %{o}_lo32 = scalar.extui %{o}_lo_b : i8 to i32
      %{o}_hi32 = scalar.extui %{o}_hi_b : i8 to i32
      %{o}_nib32 = index.cast %{o}_nib : index to i32
      %{o}_hsh32 = index.cast %{o}_hi_sh : index to i32
      %{o}_i4 = scalar.constant 4 : i32
      %{o}_i2 = scalar.constant 2 : i32
      %{o}_i15 = scalar.constant 15 : i32
      %{o}_i3 = scalar.constant 3 : i32
      %{o}_i32 = scalar.constant 32 : i32
      %{o}_lsh = scalar.muli %{o}_nib32, %{o}_i4 : i32
      %{o}_hsh = scalar.muli %{o}_hsh32, %{o}_i2 : i32
      %{o}_lo_s = scalar.shrui %{o}_lo32, %{o}_lsh : i32
      %{o}_lo = scalar.andi %{o}_lo_s, %{o}_i15 : i32
      %{o}_hi_s = scalar.shrui %{o}_hi32, %{o}_hsh : i32
      %{o}_hi_m = scalar.andi %{o}_hi_s, %{o}_i3 : i32
      %{o}_hi = scalar.shli %{o}_hi_m, %{o}_i4 : i32
      %{o}_sc = scalar.ori %{o}_lo, %{o}_hi : i32
      %{o}_centered_s = scalar.subi %{o}_sc, %{o}_i32 : i32
      %{o}_scf = scalar.sitofp %{o}_centered_s : i32 to f32
      %{o}_d = scalar.extf %{o}_d_half : f16 to f32
      %{o}_dl = scalar.mulf %{o}_d, %{o}_scf : f32
      %{o}_jq = index.mul %{o}_j, %c2 : index
      %{o}_jq32 = index.cast %{o}_jq : index to i32
      %{o}_jq8 = scalar.trunci %{o}_jq32 : i32 to i8
      %{o}_shift_q = vector.splat %{o}_jq8 : vector<16xi8>
      %{o}_g32 = index.cast %{o}_group : index to i32
      %{o}_g8 = scalar.trunci %{o}_g32 : i32 to i8
      %{o}_shift_h = vector.splat %{o}_g8 : vector<16xi8>
      %{o}_three = vector.splati 3 : vector<16xi8>
      %{o}_one = vector.splati 1 : vector<16xi8>
      %{o}_two8 = vector.splati 2 : vector<16xi8>
      %{o}_qsh = vector.shrui %{o}_qs, %{o}_shift_q : vector<16xi8>
      %{o}_low = vector.andi %{o}_qsh, %{o}_three : vector<16xi8>
      %{o}_hmsh = vector.shrui %{o}_hm, %{o}_shift_h : vector<16xi8>
      %{o}_hbit = vector.andi %{o}_hmsh, %{o}_one : vector<16xi8>
      %{o}_hb2 = vector.shli %{o}_hbit, %{o}_two8 : vector<16xi8>
      %{o}_code = vector.ori %{o}_low, %{o}_hb2 : vector<16xi8>
      %{o}_codef = vector.uitofp %{o}_code : vector<16xi8> to vector<16xf32>
      %{o}_four = vector.splatf 4.0 : vector<16xf32>
      %{o}_centered = vector.subf %{o}_codef, %{o}_four : vector<16xf32>
      %{o}_dls = vector.splat %{o}_dl : vector<16xf32>
      %{o}_scaled = vector.mulf %{o}_centered, %{o}_dls : vector<16xf32>
      %{o} = vector.fptrunc %{o}_scaled : vector<16xf32> to vector<16xf16>'''


_IQ4_KVALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)   # ggml kvalues_iq4nl


def _iq4_f16lut_on():
    """GEMM_IQ4_F16LUT (choice prefill.gemm.iq4_f16lut): iq4_dequant16 with the unsigned book (kvalue + 128), the 0x64 magic
    byte -> f16 conversion and one f16 fma per value pair (as _g32_iq4) instead of sign-extend + F32 convert/multiply/round.
    GEMM_IQ4_G32 requires it (set by the model's scope rule for the whole build, gfx12 twins included), so every tile size of a model
    stages the same f16 weights (_g32_iq4's values); a device-scoped G32 never changes the other target's kernels."""
    on = os.environ.get("GEMM_IQ4_F16LUT", "0").strip().lower() not in ("", "0", "false", "no", "off")
    if _g32_iq4_on() and IQ4_NATIVE and not on:
        raise SystemExit("GEMM_IQ4_G32 needs GEMM_IQ4_F16LUT (choice prefill.gemm.iq4_f16lut): every tile size must stage the same IQ4 values")
    return on


def _iq4_f16_tail(o, dl, pre="", dl32=None):
    """unsigned-book lookup of %{o}_idx (vector<16xi8>, 0..15) -> %{o} = (book - 128) * dl, dl an f16 SSA value"""
    book = "\n".join(f"      %{o}_k{i} = scalar.constant {v + 128 - (256 if v + 128 > 127 else 0)} : i8" for i, v in enumerate(_IQ4_KVALUES))
    return f'''
{book}
{pre}      %{o}_book = vector.from_elements {", ".join(f"%{o}_k{i}" for i in range(16))} : vector<16xi8>
      %{o}_ucodes = vector.table.lookup %{o}_book[%{o}_idx] : vector<16xi8>, vector<16xi8> -> vector<16xi8>
      %{o}_uf = vector.uitofp %{o}_ucodes : vector<16xi8> to vector<16xf16>
      %{o}_m128 = scalar.constant -128.0 : f32
      %{o}_off32 = scalar.mulf {dl32}, %{o}_m128 : f32
      %{o}_off_h = scalar.fptrunc %{o}_off32 : f32 to f16
      %{o}_scl = vector.splat {dl} : vector<16xf16>
      %{o}_offs = vector.splat %{o}_off_h : vector<16xf16>
      %{o} = vector.fmaf %{o}_uf, %{o}_scl, %{o}_offs : vector<16xf16>'''


def iq4_dequant16(fmt, w, out):
    """IQ4_XS / IQ4_NL: 16 consecutive weights of row %w_row_origin at column %w_col -> F16 vector %{out}. All loads first.
    IQ4_NL (18-byte blocks of 32): d * kvalues[q], one F16 multiply (int8 code times an F16 scale is exact in F32: same rounding).
    IQ4_XS (136-byte superblocks of 256): dl = d * (ls - 32) per 32-group (ls: 4 low bits from scales_l, 2 high from scales_h),
    dl * kvalues[q] in F32 then fptrunc (as @g_iq4_xs_values)."""
    o = out
    book = "\n".join(f"      %{o}_k{i} = scalar.constant {v} : i8" for i, v in enumerate(_IQ4_KVALUES))
    lookup = f'''
{book}
      %{o}_book = vector.from_elements {", ".join(f"%{o}_k{i}" for i in range(16))} : vector<16xi8>
      %{o}_shift_bits = index.mul %{o}_half, %{o}_c4 : index
      %{o}_shift_i8 = index.cast %{o}_shift_bits : index to i8
      %{o}_shift = vector.splat %{o}_shift_i8 : vector<16xi8>
      %{o}_shifted = vector.shrui %{o}_packed, %{o}_shift : vector<16xi8>
      %{o}_mask = vector.splati 15 : vector<16xi8>
      %{o}_idx = vector.andi %{o}_shifted, %{o}_mask : vector<16xi8>
      %{o}_codes = vector.table.lookup %{o}_book[%{o}_idx] : vector<16xi8>, vector<16xi8> -> vector<16xi8>'''
    f16 = _iq4_f16lut_on()
    lookup_idx = lookup[lookup.index(f"\n      %{o}_shift_bits = "): lookup.index(f"\n      %{o}_codes = ")]
    if fmt == "iq4_nl":
        head = f'''
      %{o}_c4 = index.constant 4 : index
      %{o}_block = index.div %w_col, %c32 : index
      %{o}_in_block = index.rem %w_col, %c32 : index
      %{o}_half = index.div %{o}_in_block, %c16 : index
      %{o}_block_add = index.scale %{o}_block, %block_bytes : index, offset -> offset
      %{o}_block_origin = index.add %w_row_origin, %{o}_block_add : offset
      %{o}_two = index.constant 2 : offset
      %{o}_code_origin = index.add %{o}_block_origin, %{o}_two : offset
      %{o}_scale_view = buffer.view %{w}[%{o}_block_origin] : buffer -> view<1xf16>
      %{o}_code_view = buffer.view %{w}[%{o}_code_origin] : buffer -> view<16xi8>
      %{o}_scale = view.load %{o}_scale_view[0] : view<1xf16> -> f16
      %{o}_packed = vector.load %{o}_code_view[0] : view<16xi8> -> vector<16xi8>'''
        if f16:
            return head + lookup_idx + _iq4_f16_tail(o, f"%{o}_scale", f"      %{o}_scale32 = scalar.extf %{o}_scale : f16 to f32\n", f"%{o}_scale32")
        return head + lookup + f'''
      %{o}_values = vector.sitofp %{o}_codes : vector<16xi8> to vector<16xf16>
      %{o}_scales = vector.splat %{o}_scale : vector<16xf16>
      %{o} = vector.mulf %{o}_values, %{o}_scales : vector<16xf16>'''
    assert fmt == "iq4_xs", fmt
    head = f'''
      %{o}_c4 = index.constant 4 : index
      %{o}_c256 = index.constant 256 : index
      %{o}_block = index.div %w_col, %{o}_c256 : index
      %{o}_within = index.rem %w_col, %{o}_c256 : index
      %{o}_group_value = index.div %{o}_within, %c32 : index
      %{o}_group = index.assume %{o}_group_value [range(%{o}_group_value, 0, 7)] : index
      %{o}_in_group = index.rem %{o}_within, %c32 : index
      %{o}_half = index.div %{o}_in_group, %c16 : index
      %{o}_block_add = index.scale %{o}_block, %block_bytes : index, offset -> offset
      %{o}_block_origin = index.add %w_row_origin, %{o}_block_add : offset
      %{o}_qs_base = index.constant 8 : offset
      %{o}_sixteen = index.constant 16 : offset
      %{o}_qs_add = index.scale %{o}_group, %{o}_sixteen : index, offset -> offset
      %{o}_qs0 = index.add %{o}_block_origin, %{o}_qs_base : offset
      %{o}_qs_origin = index.add %{o}_qs0, %{o}_qs_add : offset
      %{o}_header_view = buffer.view %{w}[%{o}_block_origin] : buffer -> view<2xi32>
      %{o}_code_view = buffer.view %{w}[%{o}_qs_origin] : buffer -> view<16xi8>
      %{o}_header = vector.load %{o}_header_view[0] : view<2xi32> -> vector<2xi32>
      %{o}_packed = vector.load %{o}_code_view[0] : view<16xi8> -> vector<16xi8>
      %{o}_halves = vector.bitcast %{o}_header : vector<2xi32> to vector<4xf16>
      %{o}_d_half = vector.extract %{o}_halves[0] : vector<4xf16> -> f16
      %{o}_d = scalar.extf %{o}_d_half : f16 to f32
      %{o}_word0 = vector.extract %{o}_header[0] : vector<2xi32> -> i32
      %{o}_word1 = vector.extract %{o}_header[1] : vector<2xi32> -> i32
      %{o}_i16 = scalar.constant 16 : i32
      %{o}_i15 = scalar.constant 15 : i32
      %{o}_i3 = scalar.constant 3 : i32
      %{o}_i4 = scalar.constant 4 : i32
      %{o}_i2 = scalar.constant 2 : i32
      %{o}_i32 = scalar.constant 32 : i32
      %{o}_g32 = index.cast %{o}_group : index to i32
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
      %{o}_dl = scalar.mulf %{o}_d, %{o}_lf : f32'''
    if f16:
        return head + lookup_idx + _iq4_f16_tail(o, f"%{o}_dlh", f"      %{o}_dlh = scalar.fptrunc %{o}_dl : f32 to f16\n      %{o}_dlh32 = scalar.extf %{o}_dlh : f16 to f32\n", f"%{o}_dlh32")
    return head + lookup + f'''
      %{o}_wide = vector.sitofp %{o}_codes : vector<16xi8> to vector<16xf32>
      %{o}_dls = vector.splat %{o}_dl : vector<16xf32>
      %{o}_scaled = vector.mulf %{o}_wide, %{o}_dls : vector<16xf32>
      %{o} = vector.fptrunc %{o}_scaled : vector<16xf32> to vector<16xf16>'''

def dequant16(fmt, w, view_name, out):
    """Loom snippet: dequantize 16 consecutive weights of row %w_row at column %w_col into F16 vector %{out}."""
    if fmt == "q4_0" and Q4_0_NATIVE:
        import moe_q4_0
        return moe_q4_0.dequant16(w, out)
    if fmt in ("iq4_xs", "iq4_nl") and IQ4_NATIVE:
        return iq4_dequant16(fmt, w, out)
    if fmt == "q3_k" and Q3_K_NATIVE:
        return q3_k_dequant16(w, out)
    if fmt == "q2_0" and Q2_0_NATIVE:
        import ternary
        return ternary.q2_0_dequant16(w, out)
    if fmt == "ptq1_0":   # 16 consecutive trits = one (byte vector, power of 3) pair of the block (ternary.py @g_ptq1_0_half16)
        return f'''
      %{out}_bv = index.constant 128 : index
      %{out}_block = index.div %w_col, %{out}_bv : index
      %{out}_within = index.rem %w_col, %{out}_bv : index
      %{out}_h0 = index.div %{out}_within, %c16 : index
      %{out}_h = index.assume %{out}_h0 [range(%{out}_h0, 0, 7)] : index
      %{out}_block_add = index.scale %{out}_block, %block_bytes : index, offset -> offset
      %{out}_block_origin = index.add %w_row_origin, %{out}_block_add : offset
      %{out}_v = func.call @g_ptq1_0_half16(%{w}, %{out}_block_origin, %{out}_h) : (buffer, offset, index) -> (vector<16xf32>)
      %{out} = vector.fptrunc %{out}_v : vector<16xf32> to vector<16xf16>'''
    if fmt in HELPER:   # 16 consecutive weights = one half of a 32-value group from the shared value helper
        bv = FMT[fmt]["block_values"]
        sc_setup = f"""      %{out}_sidx = index.madd %bounded_w_out_row, %blocks, %{out}_block : index
      %{out}_sone = index.constant 1 : offset
      %{out}_so = index.scale %{out}_sidx, %{out}_sone : index, offset -> offset
""" if fmt == "mxfp4s" else ""
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
{sc_setup}      %{out}_all = func.call @g_{fmt}_values(%{w}, {"%" + w + "s, " if fmt == "mxfp4s" else ""}%{out}_block_origin, {"%" + out + "_so, " if fmt == "mxfp4s" else ""}%{out}_group) : (buffer, {"buffer, " if fmt == "mxfp4s" else ""}offset, {"offset, " if fmt == "mxfp4s" else ""}index) -> (vector<32xf32>)
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
      %{out}_m15 = vector.splati 252645135 : vector<4xi32>
      %{out}_m3 = vector.splati 50529027 : vector<4xi32>
      %{out}_f4 = vector.splati 4 : vector<4xi32>
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
      %{out}_one = vector.splati 16843009 : vector<4xi32>
      %{out}_four = vector.splati 4 : vector<4xi32>
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


def gemm(fmt, mode, fmt2=None):
    """fmt2 (swiglu only): the up weight's format when it differs from the gate's (a Q4_K_M MoE quantizing expert
    gate and up differently, MiniCPM-V-4.7): its own block size, row stride and dequant. None: one format (unchanged)."""
    f = FMT[fmt]
    two_w = mode == "swiglu"
    fmt2 = fmt2 if two_w and fmt2 not in (None, fmt) else None
    name = f"{PREFIX}_gemm_{fmt}_{mode}" if fmt2 is None else f"{PREFIX}_gemm_{fmt}_{fmt2}_{mode}"
    assert fmt2 is None or FMT[fmt2]["block_values"] in (32, 256), fmt2   # %c32 / %c256 below
    wargs = "%w: buffer, %w2: buffer" if two_w else "%w: buffer" + (", %ws: buffer" if fmt == "mxfp4s" else "")
    assert not (two_w and fmt == "mxfp4s"), "mxfp4s has no swiglu GEMM: gate and up are two projections and g_silu_mul"
    accs = ["g0", "g1", "u0", "u1"] if two_w else ["a0", "a1"]
    stages = ["w_stage", "w2_stage"] if two_w else ["w_stage"]
    k_min = 256 if 256 in (FMT[fmt]["block_values"], FMT[fmt2 or fmt]["block_values"]) else 64
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
  %nibble_mask = vector.splati 252645135 : vector<4xi32>
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
{bv_const(f["block_values"])}  %blocks = index.div %k, {bv_name(f["block_values"])} : index
  %row_bytes = index.scale %blocks, %block_bytes : index, offset -> offset{{MIXED_ROW}}
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
  %w_row_origin = index.scale %bounded_w_out_row, %row_bytes : index, offset -> offset{MIXED_ORIGIN}
  %zero8 = vector.splatf 0.0 : vector<8xf32>
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
        if i and fmt2:   # the up weight in its own format: its block size and row origin
            lines.append(dequant16(fmt2, w, None, f"deq{i}").replace("%w_row_origin", "%w2_row_origin").replace("%block_bytes", "%block_bytes2"))
        else:
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
    text = "\n".join(lines)
    if fmt2:
        f2 = FMT[fmt2]
        text = text.replace("{MIXED_ROW}", f"""
  %block_bytes2 = index.constant {f2["block_bytes"]} : offset
  %blocks2 = index.div %k, %c{f2["block_values"]} : index
  %row_bytes2 = index.scale %blocks2, %block_bytes2 : index, offset -> offset""").replace("{MIXED_ORIGIN}", """
  %w2_row_origin = index.scale %bounded_w_out_row, %row_bytes2 : index, offset -> offset""")
    return text.replace("{MIXED_ROW}", "").replace("{MIXED_ORIGIN}", "")


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
  %nibble_mask = vector.splati 252645135 : vector<4xi32>
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
{bv_const(f["block_values"])}  %blocks = index.div %k, {bv_name(f["block_values"])} : index
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
  %zero8 = vector.splatf 0.0 : vector<8xf32>
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

# q6_k: correct but 5-7 % slower (local kbench 2026-11-19); opt in with GEMM_M256_G32_FMTS=q4_k,q6_k
def _g32_fmts():
    f = tuple(os.environ.get("GEMM_M256_G32_FMTS", "q4_k").split(","))
    return f + ("iq4_xs", "iq4_nl") if _g32_iq4_on() and IQ4_NATIVE else f


def _g32_dequant(fmt, w, out, col):
    """GEMM_M256_G32 (choice prefill.gemm.m256_g32): the m256 thread's 32 consecutive weights of one K-quant group in one
    piece (lines, addr, loads as _split_dequant) -> %{out} vector<32xf16>. col (the step's K origin, a multiple of 64) is
    uniform, so the block, page and shifts are uniform and the thread's own part is only %stage_ks (which 32-group of the
    64): one header/scale decode instead of two, one 32-byte code load, word-wide bit work on 8 words."""
    o = out
    if fmt == "q4_k":
        text = f'''
      %{o}_super = index.div {col}, %c256 : index
      %{o}_within = index.rem {col}, %c256 : index
      %{o}_page = index.div %{o}_within, %c64 : index
      %{o}_block_add = index.scale %{o}_super, %block_bytes : index, offset -> offset
      %{o}_block_origin = index.add %w_row_origin, %{o}_block_add : offset
      %{o}_header_view = buffer.view %{w}[%{o}_block_origin] : buffer -> view<4xi32>
      %{o}_payload = index.add %{o}_block_origin, %sixteen : offset
      %{o}_words = buffer.view %{w}[%{o}_payload] : buffer -> view<32xi32>
      %{o}_pw = index.mul %{o}_page, %c8 : index
      %{o}_header = vector.load %{o}_header_view[0] : view<4xi32> -> vector<4xi32>
      %{o}_packed = vector.load %{o}_words[%{o}_pw] : view<32xi32> -> vector<8xi32>
      %{o}_gv = index.madd %{o}_page, %c2, %stage_ks : index
      %{o}_group = index.assume %{o}_gv [range(%{o}_gv, 0, 7)] : index
      %{o}_scale, %{o}_minimum = func.call {META}(%{o}_header, %{o}_group) : (vector<4xi32>, index) -> (f32, f32)
      %{o}_shifted = vector.shrui %{o}_packed, %g32_shift : vector<8xi32>
      %{o}_masked = vector.andi %{o}_shifted, %g32_m15 : vector<8xi32>
      %{o}_bytes = vector.bitcast %{o}_masked : vector<8xi32> to vector<32xi8>
      %{o}_codes = vector.uitofp %{o}_bytes : vector<32xi8> to vector<32xf16>
      %{o}_scale_h = scalar.fptrunc %{o}_scale : f32 to f16
      %{o}_neg_min = scalar.negf %{o}_minimum : f32
      %{o}_neg_min_h = scalar.fptrunc %{o}_neg_min : f32 to f16
      %{o}_scales = vector.splat %{o}_scale_h : vector<32xf16>
      %{o}_minima = vector.splat %{o}_neg_min_h : vector<32xf16>
      %{o} = vector.fmaf %{o}_codes, %{o}_scales, %{o}_minima : vector<32xf16>'''
    elif fmt == "q6_k":
        # thread group j = 2 * (u % 2) + stage_ks of half u / 2 (u = within / 64): ql bytes half*64 + stage_ks*32 .. +32,
        # qh bytes 128 + half*32 .. +32 (uniform), ql shift 4 * (u % 2), qh shift 2 * j, scales 192 + half*8 + 2j (+1 for
        # the upper 16), d at 208.
        text = f'''
      %{o}_super = index.div {col}, %c256 : index
      %{o}_within = index.rem {col}, %c256 : index
      %{o}_u = index.div %{o}_within, %c64 : index
      %{o}_half = index.div %{o}_u, %c2 : index
      %{o}_jh = index.rem %{o}_u, %c2 : index
      %{o}_j = index.madd %{o}_jh, %c2, %stage_ks : index
      %{o}_block_add = index.scale %{o}_super, %block_bytes : index, offset -> offset
      %{o}_block_origin = index.add %w_row_origin, %{o}_block_add : offset
      %{o}_ql_i = index.madd %{o}_half, %c64, %g32_ks32 : index
      %{o}_qh_i = index.madd %{o}_half, %c32, %c128 : index
      %{o}_sc_a = index.madd %{o}_half, %c8, %g32_c192 : index
      %{o}_sc_i = index.madd %{o}_j, %c2, %{o}_sc_a : index
      %{o}_ql_off = index.cast %{o}_ql_i : index to offset
      %{o}_qh_off = index.cast %{o}_qh_i : index to offset
      %{o}_sc_off = index.cast %{o}_sc_i : index to offset
      %{o}_ql_o = index.add %{o}_block_origin, %{o}_ql_off : offset
      %{o}_qh_o = index.add %{o}_block_origin, %{o}_qh_off : offset
      %{o}_sc_o = index.add %{o}_block_origin, %{o}_sc_off : offset
      %{o}_d_o = index.add %{o}_block_origin, %g32_d208 : offset
      %{o}_ql_v = buffer.view %{w}[%{o}_ql_o] : buffer -> view<8xi32>
      %{o}_qh_v = buffer.view %{w}[%{o}_qh_o] : buffer -> view<8xi32>
      %{o}_sc_v = buffer.view %{w}[%{o}_sc_o] : buffer -> view<2xi8>
      %{o}_d_v = buffer.view %{w}[%{o}_d_o] : buffer -> view<1xf16>
      %{o}_ql = vector.load %{o}_ql_v[0] : view<8xi32> -> vector<8xi32>
      %{o}_qh = vector.load %{o}_qh_v[0] : view<8xi32> -> vector<8xi32>
      %{o}_sc2 = vector.load %{o}_sc_v[0] : view<2xi8> -> vector<2xi8>
      %{o}_dh = view.load %{o}_d_v[0] : view<1xf16> -> f16
      %{o}_s1 = index.mul %{o}_jh, %c4 : index
      %{o}_s2 = index.mul %{o}_j, %c2 : index
      %{o}_s1_32 = index.cast %{o}_s1 : index to i32
      %{o}_s2_32 = index.cast %{o}_s2 : index to i32
      %{o}_s1_v = vector.splat %{o}_s1_32 : vector<8xi32>
      %{o}_s2_v = vector.splat %{o}_s2_32 : vector<8xi32>
      %{o}_lo_s = vector.shrui %{o}_ql, %{o}_s1_v : vector<8xi32>
      %{o}_lo = vector.andi %{o}_lo_s, %g32_m15 : vector<8xi32>
      %{o}_hi_s = vector.shrui %{o}_qh, %{o}_s2_v : vector<8xi32>
      %{o}_hi_m = vector.andi %{o}_hi_s, %g32_m3 : vector<8xi32>
      %{o}_hi = vector.shli %{o}_hi_m, %g32_f4 : vector<8xi32>
      %{o}_qw = vector.ori %{o}_lo, %{o}_hi : vector<8xi32>
      %{o}_q = vector.bitcast %{o}_qw : vector<8xi32> to vector<32xi8>
      %{o}_codes = vector.uitofp %{o}_q : vector<32xi8> to vector<32xf16>
      %{o}_d = scalar.extf %{o}_dh : f16 to f32
      %{o}_bias32 = scalar.constant -32.0 : f32'''
        for h in range(2):
            text += f'''
      %{o}_sc8_{h} = vector.extract %{o}_sc2[{h}] : vector<2xi8> -> i8
      %{o}_scf{h} = scalar.sitofp %{o}_sc8_{h} : i8 to f32
      %{o}_scale{h} = scalar.mulf %{o}_d, %{o}_scf{h} : f32
      %{o}_offset{h} = scalar.mulf %{o}_scale{h}, %{o}_bias32 : f32
      %{o}_scale_h{h} = scalar.fptrunc %{o}_scale{h} : f32 to f16
      %{o}_offset_h{h} = scalar.fptrunc %{o}_offset{h} : f32 to f16
      %{o}_scales{h} = vector.splat %{o}_scale_h{h} : vector<16xf16>
      %{o}_offsets{h} = vector.splat %{o}_offset_h{h} : vector<16xf16>
      %{o}_c{h} = vector.slice %{o}_codes[{16 * h}] : vector<32xf16> -> vector<16xf16>
      %{o}_v{h} = vector.fmaf %{o}_c{h}, %{o}_scales{h}, %{o}_offsets{h} : vector<16xf16>'''
    elif fmt in ("iq4_xs", "iq4_nl"):
        text = _g32_iq4(fmt, w, o, col)
    else:
        raise ValueError(fmt)
    lines = [l for l in text.strip("\n").split("\n") if l.strip()]
    last = max(i for i, l in enumerate(lines) if ".load " in l)
    addr = lines[: last + 1]
    loads = [(_re.match(r"\s*(%\S+) = ", l).group(1), l.split("-> ")[-1].strip()) for l in addr if ".load " in l]
    return lines, addr, loads


def _g32_iq4(fmt, w, o, col):
    """GEMM_IQ4_G32 (choice prefill.gemm.iq4_g32): the thread's whole 32-weight IQ4 group (group = col / 32 + %stage_ks) as
    %{o}_v0 (weights 0-15: low nibbles) and %{o}_v1 (16-31: high nibbles). The nibble book holds kvalue + 128 (unsigned
    1..241), so the byte -> f16 conversion is the 0x64 magic (v_perm + pk_add) instead of sign-extend + cvt_f32 + mul_f32 +
    cvt_f16; the scale goes on as one f16 fma per value pair: (u - 128) * dl = fma(u, dl, -128 dl), and -128 dl is exact in
    f16, so the value is round_f16(kvalue * f16(dl)) (one rounding of the f16 scale product, as the K-quant kernels).
    IQ4_XS: dl = d * (ls - 32) in F32 (one header decode per group, not per 16 values); IQ4_NL: dl = d."""
    book = "\n".join(f"      %{o}_k{i} = scalar.constant {v + 128 - (256 if v + 128 > 127 else 0)} : i8" for i, v in enumerate(_IQ4_KVALUES))
    if fmt == "iq4_nl":
        head = f'''
      %{o}_bq = index.div {col}, %c32 : index
      %{o}_bi = index.add %{o}_bq, %stage_ks : index
      %{o}_block_add = index.scale %{o}_bi, %block_bytes : index, offset -> offset
      %{o}_block_origin = index.add %w_row_origin, %{o}_block_add : offset
      %{o}_two = index.constant 2 : offset
      %{o}_code_origin = index.add %{o}_block_origin, %{o}_two : offset
      %{o}_scale_view = buffer.view %{w}[%{o}_block_origin] : buffer -> view<1xf16>
      %{o}_code_view = buffer.view %{w}[%{o}_code_origin] : buffer -> view<16xi8>
      %{o}_dh = view.load %{o}_scale_view[0] : view<1xf16> -> f16
      %{o}_packed8 = vector.load %{o}_code_view[0] : view<16xi8> -> vector<16xi8>'''
        dl, dl32 = f"%{o}_dh", f"%{o}_d32"
        head += f"\n      %{o}_d32 = scalar.extf %{o}_dh : f16 to f32"
    else:
        head = f'''
      %{o}_super = index.div {col}, %c256 : index
      %{o}_within = index.rem {col}, %c256 : index
      %{o}_gq = index.div %{o}_within, %c32 : index
      %{o}_gv = index.add %{o}_gq, %stage_ks : index
      %{o}_group = index.assume %{o}_gv [range(%{o}_gv, 0, 7)] : index
      %{o}_block_add = index.scale %{o}_super, %block_bytes : index, offset -> offset
      %{o}_block_origin = index.add %w_row_origin, %{o}_block_add : offset
      %{o}_qs_base = index.constant 8 : offset
      %{o}_sixteen = index.constant 16 : offset
      %{o}_qs_add = index.scale %{o}_group, %{o}_sixteen : index, offset -> offset
      %{o}_qs0 = index.add %{o}_block_origin, %{o}_qs_base : offset
      %{o}_qs_origin = index.add %{o}_qs0, %{o}_qs_add : offset
      %{o}_header_view = buffer.view %{w}[%{o}_block_origin] : buffer -> view<2xi32>
      %{o}_code_view = buffer.view %{w}[%{o}_qs_origin] : buffer -> view<16xi8>
      %{o}_header = vector.load %{o}_header_view[0] : view<2xi32> -> vector<2xi32>
      %{o}_packed8 = vector.load %{o}_code_view[0] : view<16xi8> -> vector<16xi8>
      %{o}_halves = vector.bitcast %{o}_header : vector<2xi32> to vector<4xf16>
      %{o}_d_half = vector.extract %{o}_halves[0] : vector<4xf16> -> f16
      %{o}_d = scalar.extf %{o}_d_half : f16 to f32
      %{o}_word0 = vector.extract %{o}_header[0] : vector<2xi32> -> i32
      %{o}_word1 = vector.extract %{o}_header[1] : vector<2xi32> -> i32
      %{o}_i16 = scalar.constant 16 : i32
      %{o}_i15 = scalar.constant 15 : i32
      %{o}_i3 = scalar.constant 3 : i32
      %{o}_i4 = scalar.constant 4 : i32
      %{o}_i2 = scalar.constant 2 : i32
      %{o}_i32 = scalar.constant 32 : i32
      %{o}_g32 = index.cast %{o}_group : index to i32
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
      %{o}_dlh32 = scalar.extf %{o}_dlh : f16 to f32'''
        dl, dl32 = f"%{o}_dlh", f"%{o}_dlh32"   # the offset from the rounded scale: (u - 128) * f16(dl) exactly, subnormal dl too
    return head + f'''
{book}
      %{o}_book = vector.from_elements {", ".join(f"%{o}_k{i}" for i in range(16))} : vector<16xi8>
      %{o}_m15 = vector.splati 15 : vector<16xi8>
      %{o}_f4 = vector.splati 4 : vector<16xi8>
      %{o}_lo_i = vector.andi %{o}_packed8, %{o}_m15 : vector<16xi8>
      %{o}_hn_s = vector.shrui %{o}_packed8, %{o}_f4 : vector<16xi8>
      %{o}_hi_i = vector.andi %{o}_hn_s, %{o}_m15 : vector<16xi8>
      %{o}_lo_u = vector.table.lookup %{o}_book[%{o}_lo_i] : vector<16xi8>, vector<16xi8> -> vector<16xi8>
      %{o}_hi_u = vector.table.lookup %{o}_book[%{o}_hi_i] : vector<16xi8>, vector<16xi8> -> vector<16xi8>
      %{o}_lo_f = vector.uitofp %{o}_lo_u : vector<16xi8> to vector<16xf16>
      %{o}_hi_f = vector.uitofp %{o}_hi_u : vector<16xi8> to vector<16xf16>
      %{o}_m128 = scalar.constant -128.0 : f32
      %{o}_off32 = scalar.mulf {dl32}, %{o}_m128 : f32
      %{o}_off_h = scalar.fptrunc %{o}_off32 : f32 to f16
      %{o}_scales = vector.splat {dl} : vector<16xf16>
      %{o}_offsets = vector.splat %{o}_off_h : vector<16xf16>
      %{o}_v0 = vector.fmaf %{o}_lo_f, %{o}_scales, %{o}_offsets : vector<16xf16>
      %{o}_v1 = vector.fmaf %{o}_hi_f, %{o}_scales, %{o}_offsets : vector<16xf16>'''


def _g32_iq4_on():
    return os.environ.get("GEMM_IQ4_G32", "0").strip().lower() not in ("", "0", "false", "no", "off")


def _g32_stores(fmt, out):
    """(value, width, column offset) of the LDS stores of _g32_dequant's result"""
    return [(f"%{out}", 32, 0)] if fmt == "q4_k" else [(f"%{out}_v0", 16, 0), (f"%{out}_v1", 16, 16)]


G32_CONSTS = """  %g32_ks = index.rem %workitem, %c2 : index
  %g32_shift_i = index.mul %g32_ks, %c4 : index
  %g32_shift_32 = index.cast %g32_shift_i : index to i32
  %g32_shift = vector.splat %g32_shift_32 : vector<8xi32>
  %g32_m15 = vector.splati 252645135 : vector<8xi32>
  %g32_m3 = vector.splati 50529027 : vector<8xi32>
  %g32_f4 = vector.splati 4 : vector<8xi32>
  %g32_ks32 = index.mul %g32_ks, %c32 : index
  %g32_c192 = index.constant 192 : index
  %g32_d208 = index.constant 208 : offset"""


def _swizzle_text():
    """GEMM_M256_SWZ=G (choice prefill.gemm.m256_swz): workgroup order in groups of G output tiles, each group walking
    every row tile with the output tile fastest, so G weight tiles stay cached while the activations stream once per
    group (default order streams all activations once per ~1 output tile: at 8k rows they do not fit the cache).
    0/unset: launch order (row tile fastest)."""
    g = int(os.environ.get("GEMM_M256_SWZ", "0") or 0)
    plain = """  %row_group_id = kernel.workgroup.id<x> : index
  %out_group_id = kernel.workgroup.id<y> : index"""
    if g > 0:
        plain = f"""  %wg_x = kernel.workgroup.id<x> : index
  %wg_y = kernel.workgroup.id<y> : index
  %swz_g = index.constant {g} : index
  %swz_lin = index.madd %wg_y, %row_tile_count, %wg_x : index
  %swz_span = index.mul %swz_g, %row_tile_count : index
  %swz_grp = index.div %swz_lin, %swz_span : index
  %swz_first = index.mul %swz_grp, %swz_g : index
  %swz_within = index.rem %swz_lin, %swz_span : index
  %swz_left = index.sub %out_tile_count, %swz_first : index
  %swz_size_v = index.min %swz_g, %swz_left : index
  %swz_size = index.assume %swz_size_v [range(%swz_size_v, 1, {g})] : index
  %swz_oo = index.rem %swz_within, %swz_size : index
  %out_group_id = index.add %swz_first, %swz_oo : index
  %row_group_id = index.div %swz_within, %swz_size : index"""
        return plain + """
  %row_group = index.assume %row_group_id [range(%row_group_id, 0, 31), lt(%row_group_id, %row_tile_count)] : index
  %out_group = index.assume %out_group_id [range(%out_group_id, 0, 4095), lt(%out_group_id, %out_tile_count)] : index"""
    return plain + """
  %row_group = index.assume %row_group_id [lt(%row_group_id, %row_tile_count)] : index
  %out_group = index.assume %out_group_id [lt(%out_group_id, %out_tile_count)] : index"""


def a32_on():
    return os.environ.get("GEMM_M256_A32", "0").strip().lower() not in ("", "0", "false", "no", "off")


def a32(text):
    """GEMM_M256_A32 (choice prefill.gemm.m256_a32): cheaper weight addressing in an m256 kernel text.
    1. The thread's weight row origin is computed as an index proven < 2^31 (every m256 weight tensor of a port is < 2 GiB), so
       the loads take a 32-bit offset from the buffer's scalar base instead of 64-bit per-load address arithmetic
       (v_add_co / v_add_co_ci / v_mul_hi pairs).
    2. A dequant column col = K + S (K the step's origin, a multiple of 64; S < 64 the thread's slice) splits into the uniform
       superblock / block K / B and the thread's small remainder, instead of a per-thread division of the sum."""
    for v, bb in (("%w_row_origin", "%row_bytes"), ("%w2_row_origin", "%row_bytes2")):
        old = f"  {v} = index.scale %w_out_row, {bb} : index, offset -> offset\n"
        if old not in text:
            continue
        n = v[1:]
        bbv = "%block_bytes" if v == "%w_row_origin" else "%block_bytes2"
        cst = _re.search(rf"  {_re.escape(bbv)} = index.constant (\d+) : offset", text).group(1)
        text = text.replace(old, f"""  %{n}_bb = index.constant {cst} : index
  %{n}_rb = index.mul %blocks, %{n}_bb : index
  %{n}_ro = index.mul %w_out_row, %{n}_rb : index
  %{n}_roa = index.assume %{n}_ro [range(%{n}_ro, 0, 2147000000)] : index
  %{n}_one = index.constant 1 : offset
  {v} = index.scale %{n}_roa, %{n}_one : index, offset -> offset
""")
    # 3. the gate/up weight choice of a swiglu tile is wave-uniform (waves 0-3 stage gate rows 0-63, 4-7 up rows): from the
    #    subgroup id it stays scalar, so the selected buffer base is an SGPR pair and the loads keep the scalar-base form.
    up = "  %w_is_up = index.cmp uge, %stage_row, %half_tw : index\n"
    if up in text and "%stage_row = index.div %workitem, %c2 : index" in text:
        text = text.replace(up, "  %w_is_up = index.cmp uge, %bounded_subgroup, %c4 : index\n")
    cols = dict((m.group(1), (m.group(2), m.group(3))) for m in _re.finditer(r"^\s*(%\w+) = index\.add (%\w+), (%\w+) : index$", text, flags=_re.M))
    def sub_div(m):
        ind, x, col, c = m.group(1), m.group(2), m.group(3), m.group(4)
        if col not in cols or cols[col][0] not in ("%k_origin", "%k_next", "%c0") and not cols[col][0].startswith("%k_"):
            return m.group(0)
        k, sv = cols[col]
        return f"{ind}{x}_kq = index.div {k}, {c} : index\n{ind}{x}_sq = index.div {sv}, {c} : index\n{ind}{x} = index.add {x}_kq, {x}_sq : index"
    def sub_rem(m):
        ind, x, col, c = m.group(1), m.group(2), m.group(3), m.group(4)
        if col not in cols or cols[col][0] not in ("%k_origin", "%k_next", "%c0") and not cols[col][0].startswith("%k_"):
            return m.group(0)
        k, sv = cols[col]
        hi = int(c[2:]) - 1
        return (f"{ind}{x}_kr = index.rem {k}, {c} : index\n{ind}{x}_sr = index.rem {sv}, {c} : index\n"
                f"{ind}{x}_wv = index.add {x}_kr, {x}_sr : index\n{ind}{x} = index.assume {x}_wv [range({x}_wv, 0, {hi})] : index")
    text = _re.sub(r"^(\s*)(%\w+) = index\.div (%\w+), (%c256|%c32) : index$", sub_div, text, flags=_re.M)
    text = _re.sub(r"^(\s*)(%\w+) = index\.rem (%\w+), (%c256|%c32) : index$", sub_rem, text, flags=_re.M)
    return text


def gemm_big256(fmt, mode, fmt2=None, g32=None, bal=None, a32_=None):
    """256-row WMMA GEMM: 256 threads (8 waves in a 4 x 2 grid), each wave 64 rows x 64 columns (4 x 4
    16 x 16 accumulators). store/acc: 256 x 128 tiles, swiglu: 256 rows x 64 outputs of each of gate and up.
    Each staged weight value feeds 256 rows, and each wave reuses an A fragment across 4 B fragments and a
    B fragment across 4 A fragments (0.5 LDS fragment reads per MMA; 32 x 64 waves need 0.75).
    LDS: activations 256 x 72 f16 (padded rows, 36 KB) + weights 128 x 72 (18 KB): two workgroups per WGP.
    Thread t stages activation rows t/2 and t/2 + 128 (32 values each) and a 32-value K group of weight
    row t/2 (two 16-value slices) of the gate/up-interleaved weight tile."""
    f = FMT[fmt]
    two_w = mode == "swiglu"
    # fmt2 (swiglu, choice prefill.gemm.m256_mixed): the up weight in another format (Q4_K_M's q4_k gate + q6_k up layers).
    # Every thread loads its row of both weights (same local row, unconditional: loads yielded from a branch stall), and
    # dequantizes the one of its wave's half of the tile in a wave-uniform scf.if (waves 0-3 stage gate rows, 4-7 up rows).
    fmt2 = fmt2 if two_w and fmt2 not in (None, fmt) else None
    # GEMM_M256_MIXED_BAL (choice prefill.gemm.m256_mixed_bal): the mixed kernel with every thread staging 16 values of a gate row
    # AND 16 of the matching up row (no wave-uniform branch): every wave does the same q4_k + q6_k work, so the barrier does not
    # wait for the waves that dequantize the costlier format, and each thread loads only what it dequantizes.
    bal = bool(fmt2) and ((os.environ.get("GEMM_M256_MIXED_BAL", "0").strip().lower() not in ("", "0", "false", "no", "off")) if bal is None else bal)
    NT, TW = 256, 128
    TN = TW // 2 if two_w else TW
    cfr = 2 if two_w else 4
    RF = 4                               # 16-row A fragments per wave
    wave_cols = 16 * cfr
    WVR = TW - (TN - wave_cols) - 15
    name = f"{PREFIX}_gemm_{fmt}_{mode}_m256" if fmt2 is None else f"{PREFIX}_gemm_{fmt}_{fmt2}_{mode}_m256"
    # GEMM_M256_MIXED_NL (choice prefill.gemm.m256_mixed_nl): a 32-value-block weight (IQ4_NL, inline under prefill.gemm.iq4_native)
    # on either side of the mixed kernel; its row bytes come from its own block count (%blocks2 when the formats' blocks differ)
    nl = os.environ.get("GEMM_M256_MIXED_NL", "0").strip().lower() not in ("", "0", "false", "no", "off")
    assert fmt2 is None or (f["block_values"] == 256 and FMT[fmt2]["block_values"] == 256) \
        or (nl and {f["block_values"], FMT[fmt2]["block_values"]} <= {32, 256}), (fmt, fmt2)
    weights = ["w", "w2"] if two_w else ["w"]
    k_min = 256 if 256 in (f["block_values"], FMT[fmt2 or fmt]["block_values"]) else 64
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
  %block_bytes2 = index.constant {FMT[fmt2 or fmt]["block_bytes"]} : offset
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
  %nibble_mask = vector.splati 252645135 : vector<4xi32>
  %row_tile_count = index.div %rows, %c256 : index
  %out_tile_count = index.div %n, %tn : index
{_swizzle_text()}
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
{bv_const(f["block_values"])}  %blocks = index.div %k, {bv_name(f["block_values"])} : index{"" if not fmt2 or FMT[fmt2]["block_values"] == f["block_values"] else chr(10) + "  %blocks2 = index.div %k, %c" + str(FMT[fmt2]["block_values"]) + " : index"}
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
    if bal:
        L[-1] = L[-1].replace("""  %stage_row = index.div %workitem, %c2 : index
  %stage_ks = index.rem %workitem, %c2 : index
  %stage_col = index.mul %stage_ks, %c32 : index
  %stage_col1 = index.add %stage_col, %c16 : index""", f"""  %stage_row_v = index.div %workitem, %c4 : index
  %stage_row = index.assume %stage_row_v [range(%stage_row_v, 0, {TN - 1})] : index
  %stage_ks = index.rem %workitem, %c4 : index
  %stage_col = index.mul %stage_ks, %c16 : index
  %stage_row_up = index.add %stage_row, %tn : index""")
        L.append("  %w_out_row_value = index.add %out_origin, %stage_row : index")
    elif two_w:
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
  %row_bytes2 = index.scale %blocks{"2" if fmt2 and FMT[fmt2]["block_values"] != f["block_values"] else ""}, %block_bytes2 : index, offset -> offset
  %w2_row_origin = index.scale %w_out_row, %row_bytes2 : index, offset -> offset
  %zero8 = vector.splatf 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %k_steps = index.div %k, %c64 : index
  %wb_row = index.add %wave_col, %lane_row : index
  %wb_off = index.scale %wb_row, %c144_off : index, offset -> offset
  %w_wave_view = buffer.view %w_stage[%wb_off] : buffer -> view<{WVR}x72xf16>''')
    wsrc = "wsel" if two_w and not fmt2 else "w"
    def fmt_parts(fm, ws, pre, second):
        """(g32, parts) of one weight's dequant: parts = [(out, col, stage col, lines, addr, loads)]"""
        g = ((os.environ.get("GEMM_M256_G32", "0").strip().lower() not in ("", "0", "false", "no", "off")) if g32 is None else g32) and fm in _g32_fmts()
        fix = (lambda t: t.replace("%w_row_origin", "%w2_row_origin").replace("%block_bytes", "%block_bytes2")) if second else (lambda t: t)
        def fixp(out, col, sc, res):
            lines, addr, loads = res
            return (out, col, sc, [fix(l) for l in lines], [fix(l) for l in addr], loads)
        if bal:
            return False, [fixp(f"{pre}a", f"%{pre}cola", "%stage_col", _split_dequant(fm, ws, f"{pre}a", f"%{pre}cola"))]
        if g:
            return g, [fixp(f"{pre}g", f"%{pre}colg", "%c0", _g32_dequant(fm, ws, f"{pre}g", f"%{pre}colg"))]
        return g, [fixp(f"{pre}{x}", f"%{pre}col{x}", sc, _split_dequant(fm, ws, f"{pre}{x}", f"%{pre}col{x}"))
                   for x, sc in (("a", "%stage_col"), ("b", "%stage_col1"))]
    g32, parts = fmt_parts(fmt, wsrc, "dq", False)
    groups = [(fmt, g32, parts)]
    if fmt2:
        g32b, parts2 = fmt_parts(fmt2, "w2", "du", True)
        groups.append((fmt2, g32b, parts2))
        parts = parts + parts2
    if any(g for _, g, _ in groups):
        L.append(G32_CONSTS)
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
        elif (os.environ.get("GEMM_M256_XCLAMP", "0").strip().lower() not in ("", "0", "false", "no", "off")):   # choice prefill.gemm.m256_xclamp: unconditional loads at the clamped next K (the last step
            # reloads its own slice): no scf.if, whose yielded loads made the compiler wait for them (vmcnt(0)) and copy them
            # into the carried registers right after the barrier, before the MMAs (q6_k)
            t.append(f"""    %x_coln_v = index.add {kvar}, %x_stage_col : index
    %x_c31 = index.constant 31 : index
    %x_lim = index.sub %k, %x_c31 : index
    %x_coln = index.assume %x_coln_v [lt(%x_coln_v, %x_lim)] : index
    %x_rawn = vector.load %x_view[%x_row, %x_coln] : view<[%rows]x[%k]xf16> -> vector<32xf16>
    %x_rawn1 = vector.load %x_view[%x_row1, %x_coln] : view<[%rows]x[%k]xf16> -> vector<32xf16>""")
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
    # GEMM_M256_EDQ (choice prefill.gemm.m256_edq): the next step's weights are dequantized right after this step's MMAs (before the
    # overwrite barrier) and carried as f16, so the window between the overwrite barrier and the publication barrier holds only
    # the LDS stores.
    edq = (os.environ.get("GEMM_M256_EDQ", "0").strip().lower() not in ("", "0", "false", "no", "off")) and not (fmt2 and not bal)
    def dq_block(kvar, sfx, lmap):
        """dequant of the loads lmap names (original load name -> value) at K origin kvar: (lines, [(value, row, col, width)])"""
        out_l, st = [], []
        plist = [(groups[0][2][0], "%stage_row", groups[0]), (groups[1][2][0], "%stage_row_up", groups[1])] if bal else [(pp, "%stage_row", groups[0]) for pp in parts]
        for (out, col, sc, lines, addr, loads), row, (fm, g, _) in plist:
            c2 = col + sfx
            out_l.append(f"    {c2} = index.add {kvar}, {sc} : index")
            text = "\n".join(l for l in lines if ".load " not in l)
            text = _re.sub(_re.escape(col) + r"(?![A-Za-z0-9])", c2, text)
            for nm, ty in loads:
                text = _re.sub(_re.escape(nm) + r"(?![A-Za-z0-9_])", lmap[nm], text)
            text = _re.sub(rf"%{out}(?![A-Za-z0-9])", f"%{out}{sfx}", text)
            out_l.append(text)
            if g and not bal:
                for v, wd, co in _g32_stores(fm, out):
                    st.append((v.replace(f"%{out}", f"%{out}{sfx}", 1), row, "%stage_col" if co == 0 else "%stage_col1", wd))
            else:
                st.append((f"%{out}{sfx}", row, sc, 16))
        return out_l, st
    if edq:
        pmap, nmap = {}, {}
        for out, col, sc, lines, addr, loads in parts:
            for nm, ty in loads:
                pmap[nm] = "%" + nm[1:].replace(out + "_", out + "p_", 1)
                nmap[nm] = "%" + nm[1:].replace(out + "_", out + "n_", 1)
        L.append(addr_text("p", "%c0").replace("    ", "  ", 1))
        pl, pst = dq_block("%c0", "e0", pmap)
        _, nst = dq_block("%k_next", "e", nmap)
        L += pl
        edq_st = pst
        carried = [(f"%cr_dq{i}", v, nst[i][0], f"vector<{wd}xf16>") for i, (v, row, cc, wd) in enumerate(pst)] + carried[-2:]
    else:
        L.append(addr_text("p", "%c0").replace("    ", "  ", 1))
    L.append("  " + ", ".join(f"%o{a}" for a in accs) + ", " + ", ".join(f"%unused_{i}" for i in range(len(carried))) + " = scf.for %step = [%c0 to %k_steps step %c1](" +
             ", ".join(f"%i{a} = %init : vector<8xf32>" for a in accs) + ", " + ", ".join(f"{c[0]} = {c[1]} : {c[3]}" for c in carried) + ") -> (" +
             ", ".join("vector<8xf32>" for _ in accs) + ", " + ", ".join(c[3] for c in carried) + ") {")
    L.append("""    %k_origin = index.mul %step, %c64 : index
    vector.store %cr_x, %x_stage_view[%x_stage_row, %x_stage_col] : vector<32xf16>, view<256x72xf16>
    vector.store %cr_x1, %x_stage_view[%x_stage_row1, %x_stage_col] : vector<32xf16>, view<256x72xf16>""")
    def compute(gparts):
        t = []
        for out, col, sc, lines, addr, loads in gparts:
            t.append(f"    {col} = index.add %k_origin, {sc} : index")
            text = "\n".join(l for l in lines if ".load " not in l)
            for nm, ty in loads:
                text = _re.sub(_re.escape(nm) + r"\b", f"%cr_{nm[1:]}", text)
            t.append(text)
        return t
    def halves(fm, g, gparts):
        """the two 16-value halves (stage_col, stage_col1) of one weight's dequant, as values (and the slices making them)"""
        if not g:
            return [], [f"%{gparts[0][0]}", f"%{gparts[1][0]}"]
        st = _g32_stores(fm, gparts[0][0])
        if len(st) == 2:
            return [], [st[0][0], st[1][0]]
        v = st[0][0]
        return [f"    {v}_h0 = vector.slice {v}[0] : vector<32xf16> -> vector<16xf16>",
                f"    {v}_h1 = vector.slice {v}[16] : vector<32xf16> -> vector<16xf16>"], [f"{v}_h0", f"{v}_h1"]
    if edq:
        L += [f"    vector.store %cr_dq{i}, %w_stage_view[{row}, {cc}] : vector<{wd}xf16>, view<{TW}x72xf16>" for i, (v, row, cc, wd) in enumerate(edq_st)]
    elif bal:
        (fa, ga, pa), (fb, gb, pb) = groups
        L += compute(pa) + [f"    vector.store %{pa[0][0]}, %w_stage_view[%stage_row, %stage_col] : vector<16xf16>, view<{TW}x72xf16>"]
        L += compute(pb) + [f"    vector.store %{pb[0][0]}, %w_stage_view[%stage_row_up, %stage_col] : vector<16xf16>, view<{TW}x72xf16>"]
    elif fmt2:
        (fa, ga, pa), (fb, gb, pb) = groups
        sa, va = halves(fa, ga, pa)
        sb, vb = halves(fb, gb, pb)
        st = lambda v: [f"    vector.store {v[0]}, %w_stage_view[%stage_row, %stage_col] : vector<16xf16>, view<{TW}x72xf16>",
                        f"    vector.store {v[1]}, %w_stage_view[%stage_row, %stage_col1] : vector<16xf16>, view<{TW}x72xf16>"]
        L.append("    scf.if %w_is_up {")
        L += compute(pb) + sb + st(vb)
        L.append("    } else {")
        L += compute(pa) + sa + st(va)
        L.append("    }")
    else:
      for out, col, sc, lines, addr, loads in parts:
        L += compute([(out, col, sc, lines, addr, loads)])
        if g32:
            for v, wd, co in _g32_stores(fmt, out):
                cc = "%stage_col" if co == 0 else "%stage_col1"
                L.append(f"    vector.store {v}, %w_stage_view[%stage_row, {cc}] : vector<{wd}xf16>, view<{TW}x72xf16>")
        else:
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
    if edq:
        L += dq_block("%k_next", "e", nmap)[0]
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
    if mode == "acc" and int(os.environ.get("GEMM_ACC_BATCH", "16") or 0) > 0:
        # acc: a batch of fragments' old values are all loaded before any of the batch's stores (a load issued after a store to the same
        # buffer waits for it: the per-fragment load / add / store form was one DRAM round trip per fragment, 116 in a row per thread)
        B = int(os.environ.get("GEMM_ACC_BATCH", "16"))
        frs = [(r, c) for r in range(RF) for c in range(cfr)]
        for i in range(0, len(frs), B):
            grp = frs[i:i + B]
            for r, c in grp:
                L.append(f"  %oc{r}{c}_c = index.constant {c * 16} : index\n  %oc{r}{c} = index.add %wave_col, %oc{r}{c}_c : index")
                L.append(f"  %old{r}{c} = vector.fragment.load<result> %y_view[{rowv[r]}, %oc{r}{c}] shape [%c16, %c16] : view<256x{TN}xf32, %y_layout> -> vector<8xf32>")
            for r, c in grp:
                L.append(f"  %v{r}{c}_s = vector.mulf %o0r{r}c{c}, %inv{r} : vector<8xf32>\n  %v{r}{c} = vector.addf %old{r}{c}, %v{r}{c}_s : vector<8xf32>")
            for r, c in grp:
                L.append(f"  vector.fragment.store<result> %v{r}{c}, %y_view[{rowv[r]}, %oc{r}{c}] shape [%c16, %c16] : vector<8xf32>, view<256x{TN}xf32, %y_layout>")
        L.append("  kernel.return\n}\n")
        return a32("\n".join(L)) if (a32_on() if a32_ is None else a32_) else "\n".join(L)
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
    return a32("\n".join(L)) if (a32_on() if a32_ is None else a32_) else "\n".join(L)


def f16out_m256(text):
    """GEMM_M256_F16O (choice prefill.gemm.m256_f16o; gfx11): the `<K>_f16o` kernel of a skip-padding swiglu m256 kernel K: the
    result is stored as f16 (an f32 fragment stored into an f16 view narrows; the down GEMM that reads it takes scale 1) and the
    kernel writes the scales its consumer reads: scales_out[row] = (1, 1) for a live row, (0, 0) for a dead one (input scale 0),
    from the tile's output tile 0. Replaces `<K>` + g_row_scale_f16[_live][_lv] of the f32 result (f16o_fuse.fuse_gfx11): no f32
    round trip, no second pass. Same text transform as gen_gemm_rdna4.f16out (gfx12)."""
    old = _re.search(r'kernel\.def export\("(\w+)"\)', text).group(1)
    assert old.endswith("_m256") and "swiglu" in old, old
    t = text.replace(f'export("{old}") @{old}(', f'export("{old}_f16o") @{old}_f16o(', 1)
    assert t.count(", %y: buffer) {") == 1
    t = t.replace(", %y: buffer) {", ", %y: buffer, %scales_out: buffer) {", 1)
    a = "%tile_offset = index.scale %tile_first, %four : index, offset -> offset"
    assert t.count(a) == 1
    t = t.replace(a, a.replace("%four", "%two"))
    t, n1 = _re.subn(r"(view<\d+x\d+x)f32(, %y_layout>)", r"\1f16\2", t)
    assert n1 >= 3, n1
    live = "  %tile_live = scalar.cmpf one, %tile_any, %no_scale : f32\n"
    assert t.count(live) == 1, "f16out_m256 needs GEMM_SKIP_PADDING=1 (gen_llm's default)"
    return t.replace(live, live + """  %so_view = buffer.view %scales_out[%origin] : buffer -> view<[%rows]x2xf32>
  %so_n0 = index.cmp eq, %out_group, %c0 : index
  %so_act = scalar.andi %so_n0, %skip_in : i1
  scf.if %so_act {
    %so_one = scalar.constant 1.0 : f32
    %so_live = scalar.cmpf one, %tile_scale, %no_scale : f32
    %so_val = scf.select %so_live, %so_one, %no_scale : f32
    view.store %so_val, %so_view[%tile_row, %c0] : f32, view<[%rows]x2xf32>
    view.store %so_val, %so_view[%tile_row, %c1] : f32, view<[%rows]x2xf32>
  }
""")


def qg_split(text, D):
    """GEMM_QGS_FMTS / GEMM_QGS_D (gen_llm QG_SPLIT, choice prefill.fuse.qg_split): the `<K>_qgs` kernel of a store m256 kernel (or its gfx12
    twin) for the fused query+gate projection (output column c = head c / 2D, part (c / D) % 2, dim c % D): the tile (128 columns, inside
    one D-wide half) is stored to %y (query) or %y2 (gate) at [row, head * D + dim] with row stride n / 2: g_split_qgate's work in the
    epilogue, no %qg round trip."""
    old = _re.search(r'kernel\.def (?:target\(@\w+\) )?export\("(\w+)"\)', text).group(1)
    new = old.replace("_store_m256", "_store_m256_qgs")
    assert new != old, old
    t = text.replace(f'export("{old}") @{old}(', f'export("{new}") @{new}(', 1)
    assert t.count(", %y: buffer) {") == 1
    t = t.replace(", %y: buffer) {", ", %y: buffer, %y2: buffer) {", 1)
    a = """  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %tile_first = index.madd %row_origin, %n, %out_origin : index
  %tile_offset = index.scale %tile_first, %four : index, offset -> offset
  %y_view = buffer.view %y[%tile_offset] : buffer"""
    assert t.count(a) == 1, old
    b = f"""  %qgs_n2 = index.div %n, %c2 : index
  %y_layout = encoding.layout.strided [%qgs_n2, %c1] : encoding<layout>
  %qgs_d = index.constant {D} : index
  %qgs_2d = index.constant {2 * D} : index
  %qgs_h = index.div %out_origin, %qgs_2d : index
  %qgs_r = index.rem %out_origin, %qgs_2d : index
  %qgs_gate = index.cmp uge, %qgs_r, %qgs_d : index
  %qgs_dd = index.rem %out_origin, %qgs_d : index
  %qgs_col = index.madd %qgs_h, %qgs_d, %qgs_dd : index
  %tile_first = index.madd %row_origin, %qgs_n2, %qgs_col : index
  %tile_offset = index.scale %tile_first, %four : index, offset -> offset
  %qgs_buf = scf.select %qgs_gate, %y2, %y : buffer
  %y_view = buffer.view %qgs_buf[%tile_offset] : buffer"""
    return t.replace(a, b, 1)


def _with_twin(gfx11, base):
    """gfx11 kernel text + the gfx12 twin (gen_gemm_rdna4) of base (the default kernel of the same name): every kernel a dual-target
    program launches needs its K_rdna4, or gfx12 runs the gfx11 WMMA fragment layout."""
    import gen_gemm_rdna4 as _r4
    return gfx11 + _r4.with_twins(base)[len(base):]


def gemm_big256_db(fmt, mode):
    """GEMM_M256_DB (choice prefill.gemm.m256_db): gemm_big256 with double-buffered LDS stages of 32 K values.
    Same tile and waves (256 threads, 4 x 2 waves of 64 x 64). Two stage pairs (activations 256 x 40 f16, weights
    128 x 40; 60 KB, still two workgroups per WGP). One barrier per stage: the dequant and stores of stage s+1 sit in
    the same barrier interval as the MMAs of stage s (other buffer), so a wave never waits between its stores and
    its MMAs. Thread t stages activation row t (32 values) and one 16-value weight slice (row t/2, half t%2).
    The K loop is unrolled by two so each half names its own buffers (no aliasing between the store and read)."""
    f = FMT[fmt]
    two_w = mode == "swiglu"
    NT, TW, KS, ST = 256, 128, 32, 40
    fence = (os.environ.get("GEMM_M256_DB_FENCE", "1").strip().lower() not in ("", "0", "false", "no", "off"))
    TN = TW // 2 if two_w else TW
    cfr = 2 if two_w else 4
    RF = 4
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
  %wrow_off = index.constant {2 * ST} : offset
  %nibble_mask = vector.splati 252645135 : vector<4xi32>
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
{bv_const(f["block_values"])}  %blocks = index.div %k, {bv_name(f["block_values"])} : index
  %row_bytes = index.scale %blocks, %block_bytes : index, offset -> offset
  %x_view = buffer.view %x[%origin] : buffer -> view<[%rows]x[%k]xf16>
  %s_view = buffer.view %scales[%origin] : buffer -> view<[%rows]x2xf32>
  %x_bytes = index.constant {256 * ST * 2} : offset
  %w_bytes = index.constant {TW * ST * 2} : offset''')
    for b in range(2):
        L.append(f'''  %x_stage{b} = buffer.alloca<workgroup> align(16) %x_bytes : buffer
  %x_stage_view{b} = buffer.view %x_stage{b}[%origin] : buffer -> view<256x{ST}xf16>
  %w_stage{b} = buffer.alloca<workgroup> align(16) %w_bytes : buffer
  %w_stage_view{b} = buffer.view %w_stage{b}[%origin] : buffer -> view<{TW}x{ST}xf16>''')
    L.append(f'''  %x_srow = index.div %workitem, %c2 : index
  %x_srow1 = index.add %x_srow, %c128 : index
  %x_scol_h = index.rem %workitem, %c2 : index
  %x_scol = index.mul %x_scol_h, %c16 : index
  %x_row_value = index.add %row_origin, %x_srow : index
  %x_row = index.assume %x_row_value [lt(%x_row_value, %rows)] : index
  %x_row1_value = index.add %x_row, %c128 : index
  %x_row1 = index.assume %x_row1_value [lt(%x_row1_value, %rows)] : index
  %stage_row = index.div %workitem, %c2 : index
  %stage_ks = index.rem %workitem, %c2 : index
  %stage_col = index.mul %stage_ks, %c16 : index''')
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
  %zero8 = vector.splatf 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %ks32 = index.constant {KS} : index
  %k_steps = index.div %k, %ks32 : index
  %k_pairs = index.div %k_steps, %c2 : index
  %k_last = index.sub %k_steps, %c1 : index
  %wb_row = index.add %wave_col, %lane_row : index
  %wb_off = index.scale %wb_row, %wrow_off : index, offset -> offset''')
    for b in range(2):
        L.append(f"  %w_wave_view{b} = buffer.view %w_stage{b}[%wb_off] : buffer -> view<{WVR}x{ST}xf16>")
    wsrc = "wsel" if two_w else "w"
    out, col = "dq", "%wcol"
    lines, addr, loads = _split_dequant(fmt, wsrc, out, col)

    def _kc(sfx, svar):
        """%kc<sfx>: the K offset of stage svar, clamped to the last stage (a stage past the end reloads the last one)"""
        return (f"    %sm{sfx} = index.min {svar}, %k_last : index\n    %sa{sfx} = index.assume %sm{sfx} [lt(%sm{sfx}, %k_steps)] : index\n"
                f"    %kc{sfx} = index.mul %sa{sfx}, %ks32 : index")

    def load_text(sfx, kvar):
        """global loads of the stage at K offset kvar (clamped to the last stage) -> names with suffix sfx"""
        c2 = col + sfx
        t = [_kc(sfx, kvar), f"    {c2} = index.add %kc{sfx}, %stage_col : index"]
        body = "\n".join(addr).replace(col + " ", c2 + " ").replace(col + ",", c2 + ",")
        t.append(_re.sub(rf"%{out}_", f"%{out}{sfx}_", body))
        t.append(f"    %xc{sfx} = index.add %kc{sfx}, %x_scol : index")
        t.append(f"    %x_raw{sfx} = vector.load %x_view[%x_row, %xc{sfx}] : view<[%rows]x[%k]xf16> -> vector<16xf16>")
        t.append(f"    %x_raw1{sfx} = vector.load %x_view[%x_row1, %xc{sfx}] : view<[%rows]x[%k]xf16> -> vector<16xf16>")
        return "\n".join(t)

    def store_text(sfx, kvar, buf, cur):
        """dequantize the carried raw values `cur` (stage at K offset kvar) and store the stage into buffer buf"""
        c2 = col + sfx
        t = [_kc(sfx, kvar), f"    {c2} = index.add %kc{sfx}, %stage_col : index"]
        text = "\n".join(l for l in lines if ".load " not in l).replace(col + " ", c2 + " ").replace(col + ",", c2 + ",")
        for nm, ty in loads:
            text = _re.sub(_re.escape(nm) + r"\b", cur[nm], text)
        text = _re.sub(rf"%{out}_", f"%{out}{sfx}_", text)
        text = _re.sub(rf"%{out}\b(?!_)", f"%{out}{sfx}", text)
        t.append(text)
        t.append(f"    vector.store %{out}{sfx}, %w_stage_view{buf}[%stage_row, %stage_col] : vector<16xf16>, view<{TW}x{ST}xf16>")
        t.append(f"    vector.store {cur['x']}, %x_stage_view{buf}[%x_srow, %x_scol] : vector<16xf16>, view<256x{ST}xf16>")
        t.append(f"    vector.store {cur['x1']}, %x_stage_view{buf}[%x_srow1, %x_scol] : vector<16xf16>, view<256x{ST}xf16>")
        return "\n".join(t)

    carried = [(nm, ty) for nm, ty in loads] + [("x", "vector<16xf16>"), ("x1", "vector<16xf16>")]
    def names(sfx):
        return {nm: (f"%{out}{sfx}_" + nm[len(out) + 2:] if nm not in ("x", "x1") else f"%x_raw{nm[1:]}{sfx}") for nm, _ in carried}
    accs = [f"{wi}r{r}c{c}" for wi in range(len(weights)) for r in range(RF) for c in range(cfr)]
    rowv = ["%wave_row"] + [f"%wave_row{r}" for r in range(1, RF)]

    def mma_text(buf, sfx, cur):
        t = []
        for kk in range(KS // 16):
            t.append(f"    %fk{sfx}{kk} = index.constant {kk * 16} : index")
            for r in range(RF):
                t.append(f"    %l{sfx}{r}_{kk} = vector.fragment.load<lhs> %x_stage_view{buf}[{rowv[r]}, %fk{sfx}{kk}] shape [%c16, %c16] : view<256x{ST}xf16> -> vector<16xf16>")
            for wi in range(len(weights)):
                for c in range(cfr):
                    t.append(f'''    %wr{sfx}{wi}{c}_{kk} = index.constant {wi * TN + c * 16} : index
    %rv{sfx}{wi}{c}_{kk} = vector.load %w_wave_view{buf}[%wr{sfx}{wi}{c}_{kk}, %fk{sfx}{kk}] : view<{WVR}x{ST}xf16> -> vector<16xf16>
    %rf{sfx}{wi}{c}_{kk} = vector.fragment<rhs> %rv{sfx}{wi}{c}_{kk} shape [%c16, %c16] : vector<16xf16>''')
            for wi in range(len(weights)):
                for c in range(cfr):
                    for r in range(RF):
                        a = f"{wi}r{r}c{c}"
                        t.append(f"    %m{sfx}{a}_{kk} = vector.mma %l{sfx}{r}_{kk}, %rf{sfx}{wi}{c}_{kk}, {cur[a]} : vector<16xf16>, vector<16xf16>, vector<8xf32>")
                        cur[a] = f"%m{sfx}{a}_{kk}"
        return "\n".join(t)

    # prologue: stage 0 -> buffer 0; stage 1 raw values in registers
    L.append(load_text("p0", "%c0").replace("    ", "  ", 1))
    L.append(store_text("s0", "%c0", 0, names("p0")))
    L.append(load_text("p1", "%c1"))
    L.append("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    p1 = names("p1")
    L.append("  " + ", ".join(f"%o{a}" for a in accs) + ", " + ", ".join(f"%unused_{i}" for i in range(len(carried))) +
             " = scf.for %pair = [%c0 to %k_pairs step %c1](" + ", ".join(f"%i{a} = %init : vector<8xf32>" for a in accs) + ", " +
             ", ".join(f"%cr{i} = {p1[nm]} : {ty}" for i, (nm, ty) in enumerate(carried)) + ") -> (" +
             ", ".join("vector<8xf32>" for _ in accs) + ", " + ", ".join(ty for _, ty in carried) + ") {")
    cr = {nm: f"%cr{i}" for i, (nm, _) in enumerate(carried)}
    cur = {a: f"%i{a}" for a in accs}
    L.append("""    %s0 = index.mul %pair, %c2 : index
    %k1 = index.add %s0, %c1 : index
    %k2 = index.add %s0, %c2 : index
    %k3 = index.add %k2, %c1 : index""")
    # half A: MMAs on buffer 0 (stage 2i); store stage 2i+1 (carried) into buffer 1; load stage 2i+2
    L.append(store_text("a", "%k1", 1, cr))
    L.append(load_text("na", "%k2"))
    L.append(mma_text(0, "a", cur))
    L.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    if fence:   # keep half B's dequant (of the raw values loaded in half A) out of half A's MMAs: it would wait on those loads
        L.append("    scf.schedule.fence")
    # half B: MMAs on buffer 1 (stage 2i+1); store stage 2i+2 into buffer 0; load stage 2i+3
    L.append(store_text("b", "%k2", 0, names("na")))
    L.append(load_text("nb", "%k3"))
    L.append(mma_text(1, "b", cur))
    L.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    if fence:
        L.append("    scf.schedule.fence")
    nb = names("nb")
    L.append("    scf.yield " + ", ".join(cur[a] for a in accs) + ", " + ", ".join(nb[nm] for nm, _ in carried) + " : " +
             ", ".join("vector<8xf32>" for _ in accs) + ", " + ", ".join(ty for _, ty in carried) + "\n  }")
    L.append(f'''  %inv_view = buffer.view %x_stage0[%origin] : buffer -> view<256x16xf32>
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


def gemm_m256_variant(fmt, mode, fmt2=None):
    """The gfx11 m256 kernel text the env switches select (None: the default gemm_big256)."""
    import gen_gemm_xg as _xg
    if _xg.xg_mode() and fmt in _xg.xg_fmts():   # choice prefill.gemm.m256_xg (gfx11): activations straight into A fragments
        return _xg.gemm_xg(fmt, mode, fmt2)
    if fmt2 is not None:
        return None
    import gen_gemm_ws as _ws
    if _ws.ws_on() and fmt in _ws.ws_fmts():   # choice prefill.gemm.m256_ws (gfx11): 1 producer / consumer waves, 2 software split barrier
        return _ws.gemm_sb(fmt, mode) if _ws.ws_mode() == 2 else _ws.gemm_ws(fmt, mode)
    if (os.environ.get("GEMM_M256_DB", "0").strip().lower() not in ("", "0", "false", "no", "off")):
        return gemm_big256_db(fmt, mode)
    if (os.environ.get("GEMM_M256_G32", "0").strip().lower() not in ("", "0", "false", "no", "off")) and fmt in _g32_fmts():   # gfx11 only: the gfx12 twin comes from the default kernel
        return gemm_big256(fmt, mode)
    if a32_on():   # gfx11 only, like G32
        return gemm_big256(fmt, mode)
    return None


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
  %nib = vector.splati 252645135 : vector<8xi32>
  %w_col_words = index.mul %w_half, %c8 : index""")
    if two_w:
        A("  %c128i = index.constant 128 : index\n  %is_up = index.cmp uge, %workitem, %c128i : index")
    A(f"""  %schema = encoding.define #encoding.operand<element_format=i8, payload_elements=16, payload_registers=4> : encoding<schema>
  %zf = vector.splatf 0.0 : vector<8xf32>
  %finit = vector.fragment<init> %zf shape [%c16, %c16] : vector<8xf32>
  %zi = vector.splati 0 : vector<8xi32>
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
    import gen_gemm_rdna4 as _rdna4   # LOOM_RDNA4=0: no gfx12 twins (output byte-identical to the gfx11-only generator)
    post = lambda t: order(skip(t))
    on256 = os.environ.get("GEMM_M256", "1") not in ("", "0", "false", "no")   # 256-row kernels (_m256)
    acc256 = set(filter(None, os.environ.get("GEMM_ACC_M256_ONLY", "").split(",")))   # gen_llm RESID_EPI: these formats' acc kernels at 256 rows only
    for item in spec.split(","):
        fmt, modes = item.split(":")
        for mode in modes.split("+"):
            only256 = mode == "acc" and fmt in acc256 and fmt not in HELPER and on256
            if not only256:
                out.append(_rdna4.with_twins(post(gemm(fmt, mode))))
            if fmt in HELPER:   # the 64-row tile kernel under the 128-row name (its dequant has no prefetchable load)
                name = f"{PREFIX}_gemm_{fmt}_{mode}"
                out.append(_rdna4.with_twins(post(gemm(fmt, mode)).replace(f'export("{name}") @{name}(', f'export("{name}_m128") @{name}_m128(')))
                if fmt == "mxfp4s" and mode == "store":   # gfx12 fp8 WMMA kernels (gen_mxfp4s.fuse_mxfp4s_f8 launches them): the plain GEMM and its fused SwiGLU pair
                    import gen_mxfp4s as _mx
                    if _mx.f8_enabled():
                        out.append(post(_mx.gemm_mxfp4s_f8_m256("store", PREFIX, **_mx.f8_kernel_kwargs("store"))))
                        out.append(post(_mx.gemm_mxfp4s_f8_m256("swiglu", PREFIX, **_mx.f8_kernel_kwargs("swiglu"))))
            else:
                if not only256:
                    out.append(_rdna4.with_twins(post(gemm_big(fmt, mode))))
                if on256:
                    base256 = post(gemm_big256(fmt, mode, g32=False, a32_=False))
                    var256 = gemm_m256_variant(fmt, mode)
                    if var256 is None:
                        out.append(_rdna4.with_twins(base256))
                    else:   # gfx11 variant; the gfx12 twin still comes from the default kernel
                        out.append(post(var256) + _rdna4.with_twins(base256)[len(base256):])
                    if mode == "store" and fmt in set(filter(None, os.environ.get("GEMM_QGS_FMTS", "").split(","))):   # gen_llm QG_SPLIT
                        _D = int(os.environ["GEMM_QGS_D"])
                        _g11 = base256 if var256 is None else post(var256)
                        out.append(qg_split(_g11, _D) + "\n" + (qg_split(_rdna4.with_twins(base256)[len(base256):].lstrip("\n"), _D) if _rdna4.enabled() else ""))
                    if mode == "swiglu" and (os.environ.get("GEMM_M256_F16O", "0").strip().lower() not in ("", "0", "false", "no", "off")):   # f16-output swiglu (f16o_fuse.fuse_gfx11) + its gfx12 twin
                        out.append(_with_twin(f16out_m256(base256 if var256 is None else post(var256)), f16out_m256(base256)))
    for pair in filter(None, os.environ.get("GEMM_M256_MIXED", "").split(",")):   # gen_llm sets it (choice prefill.gemm.m256_mixed)
        f1, f2 = pair.split(":")
        out.append(_with_twin(f16out_m256(post(gemm_big256(f1, "swiglu", f2))), f16out_m256(post(gemm_big256(f1, "swiglu", f2, g32=False, bal=False, a32_=False)))))
    sys.stdout.write("\n".join(out))
