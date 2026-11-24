#!/usr/bin/env python3
"""Generate Strix-only A/B cases for paired Q4_K expert weight staging."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.request import urlopen

import numpy as np

H, F, E, U = 2048, 512, 256, 8
KERNEL = "g_moe_ggu_q4_k_k2048"
PINS = {
    "gen35": ("unsloth-Qwen3.5-35B-A3B-loom", "8c46496151b2ed2f64cce911e2ba87d2c2b72fc1",
              "ae4e3152193d5982a4cbddd61c3aeca94699c76653eb67402b19209e7e279f29"),
    "gen36": ("ggml-org-Qwen3.6-35B-A3B-loom", "a671b626bc2dc0b860828a634f7d45fe3fcc1002",
              "23b9024eb2f9c4616a0082138e7f6b2109184e83cfdce24cde98d153c30f97ca"),
}
DTYPES = {np.dtype("int8"): "i8", np.dtype("int32"): "i32",
          np.dtype("float16"): "f16", np.dtype("float32"): "f32"}


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def staged_kernel(generation):
    slug, revision, digest = PINS[generation]
    url = f"https://huggingface.co/loom-fun-staging/{slug}/resolve/{revision}/program/llm.loom"
    with urlopen(url, timeout=60) as response:
        program = response.read()
    assert hashlib.sha256(program).hexdigest() == digest, generation
    text = program.decode()
    kernel = re.search(rf'kernel.def target\(@rdna3\) export\("{KERNEL}"\).*?kernel.return\n\}}', text, re.S)
    helpers = [
        re.search(rf"func.def inline @{name}\(.*?\n\}}", text, re.S)[0]
        for name in ("g_k_scale_minimum", "g_k_metadata")
    ]
    target = re.search(r"(?m)^amdgpu.target<gfx11-generic> @rdna3\b[^\n]*", text)
    return kernel[0].replace(" target(@rdna3)", "", 1), "\n".join(helpers), target[0].replace(" @rdna3", " @expert_strix")


def g32_kernel(gemm, moe):
    # Generate the unpipelined source and replace the weight stage before
    # reusing the deployed A/B load prefetch over the new layout.
    os.environ["MOE_PREFETCH"] = "0"
    source = moe.grouped_gemm("q4_k", "swiglu", F, H, E, U, gemm.gemm)
    start = source.index("    %w_col = index.add %k_origin, %stage_col : index")
    end = source.index("    kernel.barrier<workgroup>", start)

    # Four whole wave32 groups stage gate, the remaining four stage up. In each
    # group two lanes per output row write the two 32-code halves of K64.
    setup = """  %g32_c128 = index.constant 128 : index
  %g32_gate = index.cmp ult, %workitem, %g32_c128 : index
  %g32_local = index.rem %workitem, %g32_c128 : index
  %g32_row = index.div %g32_local, %c2 : index
  %stage_ks = index.rem %g32_local, %c2 : index
  %g32_ks32 = index.mul %stage_ks, %c32 : index
  %g32_shift_index = index.mul %stage_ks, %c4 : index
  %g32_shift_i32 = index.cast %g32_shift_index : index to i32
  %g32_shift = vector.splat %g32_shift_i32 : vector<8xi32>
  %g32_m15 = vector.splati 252645135 : vector<8xi32>
  %g32_row_value = index.add %out_origin, %g32_row : index
  %g32_out_row = index.assume %g32_row_value [lt(%g32_row_value, %n)] : index
  %g32_row_origin0 = index.scale %g32_out_row, %row_bytes : index, offset -> offset
  %g32_row_origin = index.add %g32_row_origin0, %expert_off : offset
  %g32_weight = scf.select %g32_gate, %w, %w2 : buffer
"""
    lines, _, _ = gemm._g32_dequant("q4_k", "g32_weight", "g32_val", "%k_origin")
    paired = "\n".join(line.replace("%w_row_origin", "%g32_row_origin") for line in lines) + "\n"
    paired += """    scf.if %g32_gate {
      vector.store %g32_val, %w_stage_view[%g32_row, %g32_ks32] : vector<32xf16>, view<64x72xf16>
    } else {
      vector.store %g32_val, %w2_stage_view[%g32_row, %g32_ks32] : vector<32xf16>, view<64x72xf16>
    }
"""
    source = source[:start] + paired + source[end:]
    source = replace_once(source, "  %zero8 = vector.splatf 0.0 : vector<8xf32>\n",
                          setup + "  %zero8 = vector.splatf 0.0 : vector<8xf32>\n")
    return moe._prefetch_stages(source)


def weight_data(rng, out):
    for name in ("wg", "wu"):
        blocks = rng.integers(0, 256, (E * F * H // 256, 144), dtype=np.uint8)
        # Real GGML blocks with different signed finite scales/minima and packed codes.
        params = rng.uniform(-0.003, 0.003, (blocks.shape[0], 2)).astype("<f2")
        blocks[:, :4] = params.view(np.uint8).reshape(-1, 4)
        np.save(out / f"{name}.npy", blocks.reshape(-1).view(np.int8))


def case_data(rng, tokens):
    if tokens < 1024:
        rows = [[253, 64], [253, 17], [127, 1], [-1, 0]] if tokens == 63 else [
            [253, 64], [127, 17], [127, 1], [-1, 0]]
        tile_info = np.asarray(rows, dtype=np.int32)
        row_map = np.full(256, -1, dtype=np.int32)
        for tile, (_, active) in enumerate(tile_info):
            row_map[64 * tile : 64 * tile + active] = rng.choice(tokens * U, active, replace=False)
    else:
        # Approximately eight expert assignments per token, preserving the same
        # full, partial, and unused tile layout as the serving router/bucket.
        routing = rng.integers(E, size=tokens * U)
        tile_info, row_map = [], []
        for expert in range(E):
            assignments = np.flatnonzero(routing == expert)
            for start in range(0, len(assignments), 64):
                part = assignments[start : start + 64]
                tile_info.append([expert, len(part)])
                row_map.extend(np.pad(part, (0, 64 - len(part)), constant_values=-1))
        for _ in range(E + (tokens * U + 63) // 64 - len(tile_info)):
            tile_info.append([-1, 0])
            row_map.extend([-1] * 64)
        tile_info = np.asarray(tile_info, dtype=np.int32)
        row_map = np.asarray(row_map, dtype=np.int32)
    x = rng.standard_normal((tokens, H)).astype(np.float16) * np.float16(0.1)
    scales = rng.choice(np.asarray([0.5, 1.0, 2.0], dtype=np.float32), (tokens, 2))
    return {"tile_info": tile_info, "row_map": row_map, "x": x, "scales": scales}


def render_case(out, data, name, arm, full_name, alternative, negative=False):
    operators = {"base": [("base", "base")], alternative: [(alternative, "new")],
                 "ab": [("base", "base"), (alternative, "new")]}
    rows, tokens = data["row_map"].size, data["x"].shape[0]
    tensor = f"tensor<{rows}x{F}xf32>"
    weights = f"tensor<{E * F * H // 256 * 144}xi8>"
    types = {
        field: "tensor<" + "x".join(map(str, value.shape)) + "x" + DTYPES[value.dtype] + ">"
        for field, value in data.items()
    }
    text = [f"check.case public @{full_name} {{"]
    for field, value in data.items():
        path = f"{name}_{field}.npy"
        if not (out / path).is_file():
            np.save(out / path, value)
        text.append(f'  %{field} = check.file.read.npy path("{path}") : {types[field]}')
    for name_, value in (("rows", rows), ("t", tokens), ("h", H), ("f", F)):
        text.append(f"  %{name_} = check.literal value({value}) : index")
    for weight in ("wg", "wu"):
        text.append(f'  %{weight} = check.file.read.npy path("{weight}.npy") : {weights}')
    for variant, output in operators[arm]:
        text += [f"  %{output} = check.generate.fill value(42.0) : {tensor}",
                 f"  kernel.launch @k_{variant}[%rows, %f](%rows, %h, %f, %t, %tile_info, %row_map, %x, %scales, %wg, %wu, %{output})"
                 f" : [index, index](index, index, index, index, {', '.join(types.values())}, {weights}, {weights}, {tensor})"]
    if arm == "ab":
        text.append(f"  check.expect.equal actual(%new) expected(%base) : {tensor}")
    if negative:
        text += [f"  %fill = check.generate.fill value(42.0) : {tensor}",
                 f"  check.expect.approximate actual(%new) expected(%fill) atol(0.0) rtol(0.0) nan(different) : {tensor}, {tensor}"]
    return "\n".join(text + ["  check.return", "}"]) + "\n"


def name_kernel(text, name):
    return replace_once(text, f'kernel.def export("{KERNEL}") @{KERNEL}(',
                        f'kernel.def target(@expert_strix) export("k_{name}") @k_{name}(')


def main(root, out, generation, alternative="g32"):
    sys.path.insert(0, str(root / "skills/loom-port-llm/starter"))
    import gen_gemm
    import gen_moe

    os.environ.update(LOOM_RDNA4="0", MOE_PREFETCH="1", MOE_SKIP_EMPTY="1", MOE_N_FASTEST="swiglu,store")
    exact, helper, target = staged_kernel(generation)
    old = gen_moe.grouped_gemm("q4_k", "swiglu", F, H, E, U, gen_gemm.gemm)
    assert old.strip() == exact.strip(), f"{generation}: kit baseline differs from deployed Strix expert kernel"
    base = name_kernel(exact, "base")
    if alternative == "g32":
        variant = g32_kernel(gen_gemm, gen_moe)
    else:
        assert alternative == "paired"
        from paired_rows import generate
        variant = generate(gen_gemm, gen_moe, F, H, E, U)
    candidate = name_kernel(variant, alternative)
    pre = target + "\n" + helper + "\n"
    weight_data(np.random.default_rng(12), out)
    fixtures = {f"t{tokens}": case_data(np.random.default_rng(tokens), tokens) for tokens in (63, 65, 1024, 4096)}
    for arm, kernel in (("base", base), (alternative, candidate)):
        body = "".join(render_case(out, data, name, arm, name, alternative)
                       for name, data in fixtures.items() if name in ("t1024", "t4096"))
        benches = "".join(f"check.benchmark<@{name}> @k_case_{name}\n" for name in ("t1024", "t4096"))
        (out / f"bench_{arm}.loom").write_text(pre + kernel + "\n" + body + benches, encoding="utf-8")
    (out / "test.loom").write_text(pre + base + "\n" + candidate + "\n" + "".join(
        render_case(out, data, name, "ab", name, alternative) for name, data in fixtures.items()), encoding="utf-8")
    (out / "negative.loom").write_text(pre + candidate + "\n" + render_case(
        out, fixtures["t65"], "t65", alternative, "negative_nonzero", alternative, negative=True), encoding="utf-8")
    (out / "geometry.json").write_text(json.dumps({
        "generation": generation, "alternative": alternative, "pinned_source": PINS[generation], "rows": {
            name: data["row_map"].size for name, data in fixtures.items()
        }, "gpu": "gfx1151", "hidden": H, "expert_ffn": F, "experts": E, "used": U,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), *sys.argv[3:])
