"""Measure cache locality while keeping each staged down-projection workgroup intact."""
from pathlib import Path
import sys
import numpy as np
import gen_bench as expert


def schedules(base):
    row = "%row_group_id = kernel.workgroup.id<y> : index"
    col = "%out_group_id = kernel.workgroup.id<x> : index"
    yield "base", base
    yield "rows", expert.replace_once(expert.replace_once(expert.replace_once(
        base, "workgroups(%out_tiles, %row_tiles", "workgroups(%row_tiles, %out_tiles"),
        row, row.replace("<y>", "<x>")), col, col.replace("<x>", "<y>"))
    # For even numbers of row tiles transpose the low bits of the two IDs.
    # The final unpaired row is its own identity mapping.
    yield "pairs", expert.replace_once(expert.replace_once(base, row, row.replace("%row_group_id", "%raw_y")),
        col, """%raw_x = kernel.workgroup.id<x> : index
  %yb = index.rem %raw_y, %c2 : index
  %xb = index.rem %raw_x, %c2 : index
  %y0 = index.sub %raw_y, %yb : index
  %x0 = index.sub %raw_x, %xb : index
  %yn = index.add %y0, %c1 : index
  %paired = index.cmp ult, %yn, %row_tile_count : index
  %mapped_y = index.add %y0, %xb : index
  %mapped_x = index.add %x0, %yb : index
  %row_group_id = scf.select %paired, %mapped_y, %raw_y : index
  %out_group_id = scf.select %paired, %mapped_x, %raw_x : index""")


def main(out, gen):
    q, block_bytes = ("q5_k", 176) if gen == "gen35" else ("q4_k", 144)
    expert.KERNEL = f"g_moe_gdn_{q}_k512"
    deployed, helpers, target = expert.staged_kernel(gen)
    pre = target + "\n" + helpers + "\n"
    kernels = {name: expert.name_kernel(ir, name) for name, ir in schedules(deployed)}
    rng = np.random.default_rng(12)
    w = rng.integers(0, 256, (expert.E * expert.F * expert.H // 256, block_bytes), dtype=np.uint8)
    w[:, :4] = rng.uniform(-0.003, 0.003, (len(w), 2)).astype("<f2").view(np.uint8).reshape(-1, 4)
    np.save(out / "w.npy", w.reshape(-1).view(np.int8))
    wt = f"tensor<{w.size}xi8>"
    fixtures = {}
    for t in (63, 65, 1024, 4096):
        data = expert.case_data(rng, t)
        rows = data["row_map"].size
        data["x"] = rng.standard_normal((rows, expert.F), dtype=np.float32).astype(np.float16)
        data["scales"] = rng.choice(np.array([0.5, 1, 2], np.float32), size=(rows, 2))
        header, types = [], []
        for name, value in data.items():
            path = f"t{t}_{name}.npy"
            np.save(out / path, value)
            typ = "tensor<" + "x".join(map(str, value.shape)) + "x" + expert.DTYPES[value.dtype] + ">"
            types.append(typ)
            header.append(f'  %{name} = check.file.read.npy path("{path}") : {typ}')
        header.append(f'  %w = check.file.read.npy path("w.npy") : {wt}')
        for name, n in (("rows", rows), ("f", expert.F), ("h", expert.H)):
            header.append(f"  %{name} = check.literal value({n}) : index")
        output = f"tensor<{rows}x{expert.H}xf32>"
        def case(arms, check=False, negative=False):
            name = "negative_nonzero" if negative else f"t{t}"
            lines = [f"check.case public @{name} {{"] + header
            for arm in arms:
                lines += [f"  %result_{arm} = check.generate.fill value(42.0) : {output}",
                          f"  kernel.launch @k_{arm}[%rows, %h](%rows, %f, %h, %rows, %tile_info, %row_map, %x, %scales, %w, %result_{arm}) : [index, index](index, index, index, index, {', '.join(types)}, {wt}, {output})"]
                if check and arm != "base":
                    lines.append(f"  check.expect.equal actual(%result_{arm}) expected(%result_base) : {output}")
            if negative:
                lines += [f"  %fill = check.generate.fill value(42.0) : {output}",
                          f"  check.expect.approximate actual(%result_pairs) expected(%fill) atol(0.0) rtol(0.0) nan(different) : {output}, {output}"]
            return "\n".join(lines + ["  check.return", "}"]) + "\n"
        fixtures[t] = {arm: case([arm]) for arm in kernels}
        fixtures[t]["test"] = case(list(kernels), check=True)
        if t == 65:
            (out / "negative.loom").write_text(pre + kernels["pairs"] + "\n" + case(["pairs"], negative=True))
    for arm, ir in kernels.items():
        bench = "".join(fixtures[t][arm] + f"check.benchmark<@t{t}> @k_case_t{t}\n" for t in (1024, 4096))
        (out / f"bench_{arm}.loom").write_text(pre + ir + "\n" + bench)
    (out / "test.loom").write_text(pre + "\n".join(kernels.values()) + "\n" + "".join(f["test"] for f in fixtures.values()))


if __name__ == "__main__":
    main(Path(sys.argv[2]), sys.argv[3])
