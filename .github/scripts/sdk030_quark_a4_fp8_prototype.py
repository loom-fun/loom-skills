"""DevLab Quark experiment preserving the Radeon FP8 carrier and fused epilogues."""

import re

from quark_a4_prefill_prototype import activation_codes


def _rewrite_kernel(text, pattern, edit):
    pattern = re.compile(rf"^kernel\.def [^\n]*?@({pattern})\(", re.M)
    matches = tuple(pattern.finditer(text))
    assert matches, pattern
    for match in reversed(matches):
        start = match.start()
        end = text.index("\n}\n", start) + 3
        changed = edit(text[start:end])
        assert changed != text[start:end], match[1]
        text = text[:start] + changed + text[end:]
    return text, len(matches)


def _round_fragment_stores(kernel):
    stores = re.compile(r"^([ ]*)vector\.fragment\.store<result> (%\w+), (%y_view[^\n]+: vector<8xf32>[^\n]+)$", re.M)

    def replace(match):
        indent, value, output = match.groups()
        return (f"{indent}{value}_bf16 = vector.fptrunc {value} : vector<8xf32> to vector<8xbf16>\n"
                f"{indent}{value}_rounded = vector.extf {value}_bf16 : vector<8xbf16> to vector<8xf32>\n"
                f"{indent}vector.fragment.store<result> {value}_rounded, {output}")

    result, changed = stores.subn(replace, kernel)
    assert changed > 0
    return result


def _round_swiglu_inputs(kernel):
    # These are scaled accumulator fragments, before the gate activation and multiply.
    inputs = re.compile(r"^([ ]*)(%(?:g|u)\d+) = (vector\.mulf [^\n]+ : vector<8xf32>)$", re.M)

    def replace(match):
        indent, value, operation = match.groups()
        return (f"{indent}{value}_raw = {operation}\n"
                f"{indent}{value}_bf16 = vector.fptrunc {value}_raw : vector<8xf32> to vector<8xbf16>\n"
                f"{indent}{value} = vector.extf {value}_bf16 : vector<8xbf16> to vector<8xf32>")

    result, changed = inputs.subn(replace, kernel)
    assert changed > 0 and changed % 2 == 0
    return result


def _quantize_norm_output(kernel):
    marker = "    %hsv = vector.mulf %hsel, %scale32 : vector<32xf32>\n"
    assert kernel.count(marker) == 1
    code = """    %a4_code, %a4_step = func.call @g_quark_a4_codes(%hsel) : (vector<32xf32>) -> (vector<32xf32>, f32)
    %a4_scale = vector.splat %a4_step : vector<32xf32>
    %a4_input = vector.mulf %a4_code, %a4_scale : vector<32xf32>
"""
    result = kernel.replace(marker, code + marker.replace("%hsel", "%a4_input"))
    marker = "    %hsv8 = vector.mulf %hsel, %f8_scale32 : vector<32xf32>\n"
    assert result.count(marker) == 1
    return result.replace(marker, marker.replace("%hsel", "%a4_input"))


def _quantize_scaled_rows(kernel):
    start = kernel.index("  %scale4 = vector.splat %scale : vector<4xf32>\n")
    end = kernel.index("  %first = index.cmp eq, %lane, %c0 : index\n", start)
    original = kernel[start:end]
    assert "vector.store %q8, %x8_view" in original and "%w16_on" in original
    code = """  %a4_width = index.constant 32 : index
  %a4_groups = index.div %k, %a4_width : index
  %a4_in = buffer.view %x[%zero] : buffer -> view<[%rows]x[%a4_groups]x32xf32>
  %a4_out16 = buffer.view %x16[%zero] : buffer -> view<[%rows]x[%a4_groups]x32xf16>
  %a4_out8 = buffer.view %x8[%zero] : buffer -> view<[%rows]x[%a4_groups]x32xf8E4M3>
  %w16_on = index.cmp ne, %w16, %c0 : index
  scf.for %a4_base = [%lane to %a4_groups step %c256] {
    %a4_group = index.assume %a4_base [lt(%a4_base, %a4_groups)] : index
    %a4_values = vector.load %a4_in[%row, %a4_group, %c0] : view<[%rows]x[%a4_groups]x32xf32> -> vector<32xf32>
    %a4_code, %a4_step = func.call @g_quark_a4_codes(%a4_values) : (vector<32xf32>) -> (vector<32xf32>, f32)
    scf.if %w16_on {
      %out_step16 = scalar.mulf %a4_step, %scale : f32
      %out_scales16 = vector.splat %out_step16 : vector<32xf32>
      %a4_value16 = vector.mulf %a4_code, %out_scales16 : vector<32xf32>
      %a4_bits16 = vector.fptrunc %a4_value16 : vector<32xf32> to vector<32xf16>
      vector.store %a4_bits16, %a4_out16[%row, %a4_group, %c0] : vector<32xf16>, view<[%rows]x[%a4_groups]x32xf16>
    }
    %out_step8 = scalar.mulf %a4_step, %f8_scale : f32
    %out_scales8 = vector.splat %out_step8 : vector<32xf32>
    %a4_value8 = vector.mulf %a4_code, %out_scales8 : vector<32xf32>
    %a4_bits8 = vector.fptrunc %a4_value8 : vector<32xf32> to vector<32xf8E4M3>
    vector.store %a4_bits8, %a4_out8[%row, %a4_group, %c0] : vector<32xf8E4M3>, view<[%rows]x[%a4_groups]x32xf8E4M3>
  }
"""
    return kernel[:start] + code + kernel[end:]


def apply(text, config):
    """Radeon-only experiment. Keep the existing group schedules and select the serving variant on DevLab."""
    spec = config["quark"]["global_quant_config"]["input_tensors"]
    assert config["model_id"] == "qwen3.8-27b-mxfp4"
    assert (spec["dtype"], spec["group_size"], spec["scale_calculation_mode"]) == ("fp4", 32, "even")

    text, store_count = _rewrite_kernel(text,
        r"g_gemm_mxfp4s_store(?:_m128)?(?:_rdna4)?|g_gemm_mxfp4s_store_f8_m256_rdna4|g_ab_gemm_mxfp4s_store_pair_f8_m256_rdna4",
        _round_fragment_stores)
    text, swiglu_count = _rewrite_kernel(text, "g_gemm_mxfp4s_swiglu_f8_m256_rdna4", _round_swiglu_inputs)
    text, row_count = _rewrite_kernel(text, "g_row_scale_f16_live_lv_f8_rdna4", _quantize_scaled_rows)
    text, norm_count = _rewrite_kernel(text, "g_post_add_norm_rs_live_lv_f8_rdna4", _quantize_norm_output)
    assert (store_count, swiglu_count, row_count, norm_count) == (6, 1, 1, 1)
    first_kernel = text.index("\nkernel.def ") + 1
    print("Quark FP8 candidate fused A4 packing in two producers and BF16 rounding in seven GEMM kernels")
    return text[:first_kernel] + activation_codes() + text[first_kernel:]
