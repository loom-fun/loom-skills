"""DevLab-only Qwen Quark A4 activation experiment for the half prefill path."""

import re


def activation_codes():
    """Encode 32 normal BF16 activations as signed doubled E2M1 and a shared scale."""
    lines = ["""
func.def inline @g_quark_a4_codes(%source: vector<32xf32>) -> (vector<32xf32>, f32) {
  %bf16 = vector.fptrunc %source : vector<32xf32> to vector<32xbf16>
  %converted = vector.extf %bf16 : vector<32xbf16> to vector<32xf32>
  %zero = scalar.constant 0.0 : f32
  %zeros = vector.splatf 0.0 : vector<32xf32>
  %magnitude = vector.absf %converted : vector<32xf32>
  %maximum = vector.reduce<maxnumf> %magnitude, %zero : vector<32xf32>, f32
  %maximum_bits = scalar.bitcast %maximum : f32 to i32
  %scale_bias = scalar.constant 2097152 : i32
  %mantissa_rounded = scalar.addi %maximum_bits, %scale_bias : i32
  %exp_shift = scalar.constant 23 : i32
  %maximum_exponent = scalar.shrui %mantissa_rounded, %exp_shift : i32
  %two = scalar.constant 2 : i32
  %one = scalar.constant 1 : i32
  %max_exp = scalar.constant 254 : i32
  %scale_exp_raw = scalar.subi %maximum_exponent, %two : i32
  %scale_exp_lower = scalar.maxsi %scale_exp_raw, %one : i32
  %scale_exp = scalar.minsi %scale_exp_lower, %max_exp : i32
  %scale_bits = scalar.shli %scale_exp, %exp_shift : i32
  %scale = scalar.bitcast %scale_bits : i32 to f32
  %scales = vector.splat %scale : vector<32xf32>
  %normalized = vector.divf %magnitude, %scales : vector<32xf32>
"""]
    value = "%zeros"
    for n, (limit, op, twice_value) in enumerate([
        (0.25, "ogt", 1.0), (0.75, "oge", 2.0), (1.25, "ogt", 3.0),
        (1.75, "oge", 4.0), (2.5, "ogt", 6.0), (3.5, "oge", 8.0),
        (5.0, "ogt", 12.0),
    ]):
        lines += [f"  %edge{n} = vector.splatf {limit} : vector<32xf32>",
                  f"  %choose{n} = vector.cmpf {op}, %normalized, %edge{n} : vector<32xf32> -> vector<32xi1>",
                  f"  %step{n} = vector.splatf {twice_value} : vector<32xf32>",
                  f"  %code{n} = vector.select %choose{n}, %step{n}, {value} : vector<32xf32>"]
        value = f"%code{n}"
    lines += [f"  %negative = vector.cmpf olt, %converted, %zeros : vector<32xf32> -> vector<32xi1>",
              f"  %minus = vector.subf %zeros, {value} : vector<32xf32>",
              f"  %signed = vector.select %negative, %minus, {value} : vector<32xf32>",
              "  %half = scalar.constant 0.5 : f32",
              "  %step = scalar.mulf %scale, %half : f32",
              "  func.return %signed, %step : vector<32xf32>, f32", "}"]
    return "\n".join(lines) + "\n"


GEOMETRY = r"""(%rows: index, %width: index) {
  %one = index.constant 1 : index
  %thirtytwo = index.constant 32 : index
  %sixtythree = index.constant 63 : index
  %sixtyfour = index.constant 64 : index
  %groups = index.div %width, %thirtytwo : index
  %groups_padded = index.add %groups, %sixtythree : index
  %workgroups = index.div %groups_padded, %sixtyfour : index
  kernel.launch.config workgroups(%workgroups, %rows, %one) workgroup_size(%sixtyfour, %one, %one) : index
} launch(%rows_arg: index, %width_arg: index, %input: buffer, %output: buffer, %scales: buffer) {
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 8192)] : index
  %width = index.assume %width_arg [range(%width_arg, 32, 65536), multiple_of(%width_arg, 32)] : index
  %zero = index.constant 0 : offset
  %begin = index.constant 0 : index
  %thirtytwo = index.constant 32 : index
  %sixtyfour = index.constant 64 : index
  %groups = index.div %width, %thirtytwo : index
  %workgroup = kernel.workgroup.id<x> : index
  %row_index = kernel.workgroup.id<y> : index
  %row = index.assume %row_index [lt(%row_index, %rows)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 63)] : index
  %block = index.mul %workgroup, %sixtyfour : index
  %group_index = index.add %block, %lane : index
  %in_range = index.cmp ult, %group_index, %groups : index
  scf.if %in_range {
    %group = index.assume %group_index [lt(%group_index, %groups)] : index
"""

OPERAND_F16 = r"""
    %scale_view = buffer.view %scales[%zero] : buffer -> view<[%rows]x2xf32>
    %row_scale = view.load %scale_view[%row, %begin] : view<[%rows]x2xf32> -> f32
    %zero_f = scalar.constant 0.0 : f32
    %live = scalar.cmpf one, %row_scale, %zero_f : f32
    scf.if %live {
      %input_view = buffer.view %input[%zero] : buffer -> view<[%rows]x[%groups]x32xf32>
      %output_view = buffer.view %output[%zero] : buffer -> view<[%rows]x[%groups]x32xf16>
      %values = vector.load %input_view[%row, %group, %begin] : view<[%rows]x[%groups]x32xf32> -> vector<32xf32>
      %codes, %step = func.call @g_quark_a4_codes(%values) : (vector<32xf32>) -> (vector<32xf32>, f32)
      %output_scale = scalar.mulf %row_scale, %step : f32
      %output_scales = vector.splat %output_scale : vector<32xf32>
      %qdq = vector.mulf %codes, %output_scales : vector<32xf32>
      %nearest = vector.fptrunc %qdq : vector<32xf32> to vector<32xf16>
      vector.store %nearest, %output_view[%row, %group, %begin] : vector<32xf16>, view<[%rows]x[%groups]x32xf16>
    }
"""

OPERAND_I8 = r"""
    %input_view = buffer.view %input[%zero] : buffer -> view<[%rows]x[%groups]x32xf32>
    %output_view = buffer.view %output[%zero] : buffer -> view<[%rows]x[%groups]x8xi32>
    %values = vector.load %input_view[%row, %group, %begin] : view<[%rows]x[%groups]x32xf32> -> vector<32xf32>
    %codes, %step = func.call @g_quark_a4_codes(%values) : (vector<32xf32>) -> (vector<32xf32>, f32)
    %bytes = vector.fptosi %codes : vector<32xf32> to vector<32xi8>
    %words = vector.bitcast %bytes : vector<32xi8> to vector<8xi32>
    vector.store %words, %output_view[%row, %group, %begin] : vector<8xi32>, view<[%rows]x[%groups]x8xi32>
    %zero_i = scalar.constant 0 : i32
    %zero8 = vector.splati 0 : vector<8xi32>
    %ones = vector.splati 1 : vector<32xi8>
    %parts = vector.dot4i<s8s8> %bytes, %ones, %zero8 : vector<32xi8>, vector<32xi8>, vector<8xi32>
    %signed_sum = vector.reduce<addi> %parts, %zero_i : vector<8xi32>, i32
    %sum = scalar.sitofp %signed_sum : i32 to f32
    %width_bytes = index.cast %width : index to offset
    %metadata_start = index.scale %rows, %width_bytes : index, offset -> offset
    %metadata_view = buffer.view %output[%metadata_start] : buffer -> view<[%rows]x[%groups]x2xf32>
    %metadata = vector.from_elements %step, %sum : vector<2xf32>
    vector.store %metadata, %metadata_view[%row, %group, %begin] : vector<2xf32>, view<[%rows]x[%groups]x2xf32>
"""


def _kernel(name, operand):
    geometry = GEOMETRY if name.endswith("f16") else GEOMETRY.replace(", %scales: buffer", "")
    return f'kernel.def export("{name}") @{name}' + geometry + operand + "  }\n  kernel.return\n}\n"


def _round_linear_results(command):
    """Match the BF16 result boundary of the reference Quark linear."""
    gemm = re.compile(r"^([ ]*)template\.apply<@sel\d+_g_gemm_mxfp4s_store(?:_m128)?>\(([^)]*)\) : [^\n]+$", re.M)
    skinny = re.compile(r"^([ ]*)kernel\.launch @g_mv_mxfp4s_k5120_t16_q8\[%nHV, %t\]\([^\n]*(%galpha|%gbeta)\) : [^\n]+$", re.M)

    def append_round(match, rows, width, output):
        indent = match.group(1)
        return (f"{match[0]}\n{indent}kernel.launch @g_quark_round_bf16[{rows}, {width}]"
                f"({rows}, {width}, {output}) : [index, index](index, index, buffer)")

    def round_gemm(match):
        operands = match[2].split(", ")
        assert len(operands) == 10, operands
        return append_round(match, operands[2], operands[4], operands[9])

    command, gemms = gemm.subn(round_gemm, command)
    command, skinny_calls = skinny.subn(lambda m: append_round(m, "%t", "%nHV", m[2]), command)
    assert gemms > 500 and skinny_calls == 96, (gemms, skinny_calls)
    print(f"Quark prototype rounded {gemms} GEMM branches and {skinny_calls} GDN outputs to BF16")
    return command


def _round_bf16_kernel():
    geometry = GEOMETRY.replace("%input: buffer, %output: buffer, %scales: buffer", "%output: buffer")
    geometry = geometry.replace("thirtytwo", "sixteen").replace("constant 32 ", "constant 16 ")
    geometry = geometry.replace(" 32, 65536", " 16, 65536").replace("multiple_of(%width_arg, 32)", "multiple_of(%width_arg, 16)")
    return 'kernel.def export("g_quark_round_bf16") @g_quark_round_bf16' + geometry + """
    %output_view = buffer.view %output[%zero] : buffer -> view<[%rows]x[%groups]x16xf32>
    %values = vector.load %output_view[%row, %group, %begin] : view<[%rows]x[%groups]x16xf32> -> vector<16xf32>
    %narrow = vector.fptrunc %values : vector<16xf32> to vector<16xbf16>
    %rounded = vector.extf %narrow : vector<16xbf16> to vector<16xf32>
    vector.store %rounded, %output_view[%row, %group, %begin] : vector<16xf32>, view<[%rows]x[%groups]x16xf32>
  }
  kernel.return
}
"""


def apply(text, config, *, round_outputs=False):
    """Evaluate A4 only in the initial prefill source, keeping the qualification gate unchanged."""
    spec = config["quark"]["global_quant_config"]["input_tensors"]
    assert (spec["dtype"], spec["group_size"], spec["scale_calculation_mode"]) == ("fp4", 32, "even")
    assert config["model_id"] == "qwen3.8-27b-mxfp4"
    advance = re.search(r"^command\.def public (?:target\(@\w+\) )?@llm_advance\(", text, re.M)
    assert advance, "no emitted llm_advance command"
    start = advance.start()
    end = text.index("\n}\n", start) + len("\n}\n")
    command = text[start:end]
    assert "g_gemm_mxfp4s_store_m128" in command
    assert not any(f"@{name}[" in command for name in (
        "g_post_add_norm_rs_live_lv", "g_gdn_norm_gate_rs", "g_silu_mul_rs", "g_gemm_mxfp4s_store_f8_m256_rdna4",
    ))
    row_scale = re.compile(
        r"^([ ]*)(kernel\.launch @g_row_scale_f16(?:_live(?:_lv)?)?\[%t\]\(%t, (?:%info, )?"
        r"(%\w+), (%h|%o|%f), (%s_h|%s_o|%s_f), (%h16|%o16|%f16)\) : [^\n]+)$", re.M,
    )

    def add_qdq(match):
        indent, existing, width, source, scales, output = match.groups()
        return (f"{indent}{existing}\n{indent}kernel.launch @g_quark_a4_f16[%t, {width}]"
                f"(%t, {width}, {source}, {output}, {scales}) : [index, index](index, index, buffer, buffer, buffer)")

    command, f16_count = row_scale.subn(add_qdq, command)
    assert f16_count >= 3 * config["layers"], f16_count
    # GDN alpha and beta use an int8 carrier even when the other projections use GEMM.
    command, skinny_count = re.subn(
        r"(scf\.if %gemm \{\n[ ]*kernel\.launch )@g_quant_q8(\[%t, %nH\]\(%t, %nH, %h, %h16q\))",
        r"\1@g_quark_a4_i8\2", command,
    )
    assert skinny_count == len(config["recurrent_layers"]), skinny_count
    print(f"Quark prototype rewrote {f16_count} prefill A operands and {skinny_count} GDN input packs")
    library = activation_codes() + _kernel("g_quark_a4_f16", OPERAND_F16) + _kernel("g_quark_a4_i8", OPERAND_I8)
    if round_outputs:
        command = _round_linear_results(command)
        library += _round_bf16_kernel()
    return text[:start] + library + command + text[end:]
