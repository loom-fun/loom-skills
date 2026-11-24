"""Build a gate/up kernel that reuses B for adjacent tiles of the same expert."""
import os
import re

ARGS = ("g0", "g1", "u0", "u1")
ACC = "vector<8xf32>"


def clone(text, overrides):
    names = set()
    for line in text.splitlines():
        lhs, sep, _ = line.partition(" = ")
        if sep:
            names.update(re.findall(r"%\w+", lhs))
    substitutions = {name: "%p_" + name[1:] for name in names} | overrides
    return re.sub(r"%\w+", lambda match: substitutions.get(match[0], match[0]), text)


def insert_at(text, anchor, insert, after=False):
    assert text.count(anchor) == 1, anchor
    return text.replace(anchor, anchor + insert if after else insert + anchor)


def generate(gemm, moe, n, k, experts, topk):
    os.environ["MOE_PREFETCH"] = "0"
    kernel = moe.grouped_gemm("q4_k", "swiglu", n, k, experts, topk, gemm.gemm)
    tile_gate = "  scf.if %tile_used {\n"
    selection = """  %pair_parity = index.rem %row_group, %c2 : index
  %pair_even = index.cmp eq, %pair_parity, %c0 : index
  %pair_odd = index.cmp ne, %pair_parity, %c0 : index
  %pair_prev_raw = index.sub %row_group, %pair_parity : index
  %pair_prev = index.assume %pair_prev_raw [lt(%pair_prev_raw, %row_tile_count)] : index
  %pair_next_raw = index.add %row_group, %c1 : index
  %pair_has_next = index.cmp ult, %pair_next_raw, %row_tile_count : index
  %pair_next_clamped = scf.select %pair_has_next, %pair_next_raw, %row_group : index
  %pair_next = index.assume %pair_next_clamped [lt(%pair_next_clamped, %row_tile_count)] : index
  %pair_eprev = view.load %ti_view[%pair_prev, 0] : view<[%row_tile_count]x2xi32> -> i32
  %pair_enext = view.load %ti_view[%pair_next, 0] : view<[%row_tile_count]x2xi32> -> i32
  %pair_same_prev = scalar.cmpi eq, %tile_expert, %pair_eprev : i32
  %pair_same_next = scalar.cmpi eq, %tile_expert, %pair_enext : i32
  %pair_redundant = scalar.andi %pair_odd, %pair_same_prev : i1
  %pair_false = scalar.constant false : i1
  %pair_needed = scalar.cmpi eq, %pair_redundant, %pair_false : i1
  %pair_tile_used = scalar.andi %tile_used, %pair_needed : i1
  %pair_even_with_next = scalar.andi %pair_even, %pair_has_next : i1
  %paired = scalar.andi %pair_even_with_next, %pair_same_next : i1
  scf.if %pair_tile_used {
"""
    assert kernel.count(tile_gate) == 1
    kernel = kernel.replace(tile_gate, selection)
    setup = """  %pair_row_origin = index.mul %pair_next, %c64 : index
  %pair_first = index.madd %pair_row_origin, %n, %out_origin : index
  %pair_offset = index.scale %pair_first, %four : index, offset -> offset
  %pair_y_view = buffer.view %y[%pair_offset] : buffer -> view<64x64xf32, %y_layout>
  %pair_x_stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %pair_x_view = buffer.view %pair_x_stage[%origin] : buffer -> view<64x72xf16>
  %pair_count32 = view.load %ti_view[%pair_next, 1] : view<[%row_tile_count]x2xi32> -> i32
  %pair_count_b = scalar.assume %pair_count32 [range(%pair_count32, 0, 64)] : i32
  %pair_count = index.cast %pair_count_b : i32 to index
  %pair_wave_has_rows = index.cmp ult, %wave_row, %pair_count : index
  %pair_wave_live = scalar.andi %paired, %pair_wave_has_rows : i1
"""
    kernel = insert_at(kernel, "  %zero8 = vector.splatf 0.0 : vector<8xf32>\n", setup)
    gather = kernel[kernel.index("  %gx_prow_v = "):kernel.index("  %w_out_row = ")]
    gather = clone(gather, {"%row_origin": "%pair_row_origin"})
    kernel = insert_at(kernel, "  %zero8 = vector.splatf 0.0 : vector<8xf32>\n", gather)

    stage_start = kernel.index("    %x_half_raw = vector.load ")
    stage_end = kernel.index("    %w_col = ", stage_start)
    pair_stage = clone(kernel[stage_start:stage_end], {
        "%bounded_x_row": "%p_bounded_x_row", "%gx_valid": "%p_gx_valid", "%x_stage_view": "%pair_x_view",
    })
    anchor = "    // A stage for the adjacent tile\n"
    kernel = insert_at(kernel, "    %w_col = index.add %k_origin, %stage_col : index\n", anchor)

    loop = re.search(r"(?m)^(  %g0_raw.* = scf.for %step .*\)) -> \(([^\n]*)\) \{$", kernel)
    head = loop[1]
    outputs, definition = head.split(" = scf.for", 1)
    second_raw = ", ".join(f"%p_{name}_raw" for name in ARGS)
    types = ", ".join((ACC,) * 2 * len(ARGS))
    second_args = ", ".join(f"%p_{name}_acc = %init : {ACC}" for name in ARGS)
    new_loop = f"{outputs}, {second_raw} = scf.for{definition[:-1]}, {second_args}) -> ({types}) {{"
    kernel = kernel.replace(loop[0], new_loop)
    mma_start = kernel.index("    %live_")
    barrier = "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"
    mma_end = kernel.index(barrier, mma_start)
    mma = kernel[mma_start:mma_end]
    second_mma = clone(mma, {
        "%wave_live": "%pair_wave_live", "%x_stage_view": "%pair_x_view",
        **{f"%{name}_acc": f"%p_{name}_acc" for name in ARGS},
    })
    first_yield = re.match(r"    scf.yield (.*?) : (.*)\n", kernel[mma_end + len(barrier):])
    second_yield = ", ".join("%p_" + name[1:] for name in first_yield[1].split(", "))
    yield_line = f"    scf.yield {first_yield[1]}, {second_yield} : {types}\n"
    kernel = kernel[:mma_end] + second_mma + barrier + yield_line + kernel[mma_end + len(barrier) + len(first_yield[0]):]

    ep_start = kernel.index("  %inv_bytes =")
    ep_end = kernel.index("\n  }\n  kernel.return", ep_start)
    ep = clone(kernel[ep_start:ep_end], {
        "%row_origin": "%pair_row_origin", "%y_view": "%pair_y_view",
        **{f"%{name}_raw": f"%p_{name}_raw" for name in ARGS},
    })
    kernel = kernel[:ep_end] + "\n  scf.if %paired {\n" + ep + "\n  }" + kernel[ep_end:]

    # Preserve the baseline's clean prefetch for the primary A and shared B.
    # The second A is loaded only for an actual pair and uses the same gather.
    kernel = moe._prefetch_stages(kernel)
    assert kernel.count(anchor) == 1
    return kernel.replace(anchor, "    scf.if %paired {\n" + pair_stage + "    }\n")
