"""Q4_0 expert weight staging, shared by grouped gate/up and down GEMMs.

A block is a signed F16 scale followed by 16 packed bytes. Columns 0..15 are
low nibbles and 16..31 are high nibbles, both centered at eight. Decode matvecs
use the existing exact F32 @g_q4_0_values helper; grouped WMMA kernels only need
one half-block per staging lane. Keep this specialization local to new Q4_0
experts so dense and all previously supported expert formats stay unchanged.
"""


def dequant16(w, out):
    """Decode the selected 16-value half of %w_col into F16 %{out}."""
    return f'''
      %{out}_block = index.div %w_col, %c32 : index
      %{out}_in_block = index.rem %w_col, %c32 : index
      %{out}_half = index.div %{out}_in_block, %c16 : index
      %{out}_block_add = index.scale %{out}_block, %block_bytes : index, offset -> offset
      %{out}_block_origin = index.add %w_row_origin, %{out}_block_add : offset
      %{out}_two = index.constant 2 : offset
      %{out}_code_origin = index.add %{out}_block_origin, %{out}_two : offset
      %{out}_scale_view = buffer.view %{w}[%{out}_block_origin] : buffer -> view<1xf16>
      %{out}_code_view = buffer.view %{w}[%{out}_code_origin] : buffer -> view<16xi8>
      %{out}_scale = view.load %{out}_scale_view[0] : view<1xf16> -> f16
      %{out}_packed = vector.load %{out}_code_view[0] : view<16xi8> -> vector<16xi8>
      %{out}_four = index.constant 4 : index
      %{out}_shift_bits = index.mul %{out}_half, %{out}_four : index
      %{out}_shift_i8 = index.cast %{out}_shift_bits : index to i8
      %{out}_shift = vector.splat %{out}_shift_i8 : vector<16xi8>
      %{out}_shifted = vector.shrui %{out}_packed, %{out}_shift : vector<16xi8>
      %{out}_mask = vector.splati 15 : vector<16xi8>
      %{out}_eight = vector.splati 8 : vector<16xi8>
      %{out}_unsigned = vector.andi %{out}_shifted, %{out}_mask : vector<16xi8>
      %{out}_centered = vector.subi %{out}_unsigned, %{out}_eight : vector<16xi8>
      %{out}_codes = vector.sitofp %{out}_centered : vector<16xi8> to vector<16xf16>
      %{out}_scales = vector.splat %{out}_scale : vector<16xf16>
      // Integer [-8, 7] times F16 is exact in F32. This F16 multiply has the
      // same one rounding as @g_q4_0_values followed by vector.fptrunc.
      %{out} = vector.mulf %{out}_codes, %{out}_scales : vector<16xf16>'''


def specialize_grouped(text, fmt, mode, fmt2=""):
    """Replace only Q4_0 staging in the unmodified 64x64 GEMM template.

    Apply before expert/fused offsets and before RDNA4 twins. Assert each
    replacement so a template change cannot silently leave half the path old.
    """
    import gen_gemm
    formats = [fmt, fmt2 or fmt] if mode == "swiglu" else [fmt]
    for i, weight_fmt in enumerate(formats):
        if weight_fmt != "q4_0":
            continue
        w = "w" if i == 0 else "w2"
        old = gen_gemm.dequant16("q4_0", w, None, f"deq{i}")
        new = dequant16(w, f"deq{i}")
        if i and fmt2:
            old = old.replace("%w_row_origin", "%w2_row_origin").replace("%block_bytes", "%block_bytes2")
            new = new.replace("%w_row_origin", "%w2_row_origin").replace("%block_bytes", "%block_bytes2")
        if text.count(old) != 1:
            raise AssertionError(("Q4_0 expert staging", mode, i, text.count(old)))
        text = text.replace(old, new)
    return text
