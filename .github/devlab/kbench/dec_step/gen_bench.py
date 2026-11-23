#!/usr/bin/env python3
"""Generate the dec_step kbench suite: ONE single-token decode step of JoyAI-LLM-Flash (deepseek2 / MLA, 40 layers, IQ4_XS)
exactly as the deepseek2 example's program launches it (the per-kernel profile of decode without a published card).

usage: gen_bench.py ROOT OUTDIR [VARIANT ...]
The program is generated from this checkout (apply_example.sh deepseek2 + gen_llm.py with joyai_cfg.json, the GGUF's
header config); its llm_advance command is specialized to tokens = spans = 1 (the scf.if branches a 1-token epoch takes)
and becomes one check.case per context (cases CTXS) with every launch kept in order. Each kernel is renamed
k_<kernel>__<output buffer> so profile rows separate roles (q_a vs kv_a ...). Benchmark rows: k_case_<ctx> = host wall of
the whole step (launch gaps included); k_* = mean device ms per dispatch (x 39-40 per step for the per-layer kernels).
Weights: zero bytes (same dequant work and bytes as real ones, no NaN), norms 1.0, routers random f32 per layer (from
rng, so each layer picks its own 8 of 256 experts; the 256-expert tensors are one shared buffer per size, 0.7 GB,
re-read across layers like a real step), x / h random. KV cache zero (attention reads every key; timing only).
VARIANT = base or "+"-joined keys of VARIANTS (generator environment)."""
import json, os, re, shutil, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, "../../../.."))
HEAD = "amdgpu.target<gfx11-generic> @w32 {subgroup_size = 32}\n\n"
VARIANTS = {"split": {"MLA_DEC": "split"}, "pl1": {"MLA_DEC_PL": "1"}, "s32": {"MLA_DEC_S": "32"}, "b8": {"MLA_DEC_BMIN": "8"},
            "b2": {"MLA_DEC_BMIN": "2"}, "b1": {"MLA_DEC_BMIN": "1"}, "shslot": {"MOE_SHSLOT": "1"}}
CTXS = [("short", 300), ("2k", 2048), ("8k", 8192)]
OUT = None


def program(variant):
    tmp = tempfile.mkdtemp()
    kit = os.path.join(tmp, "kit")
    subprocess.run([os.path.join(ROOT, "skills/loom-port-llm/reference/apply_example.sh"), "deepseek2", kit], check=True, stdout=subprocess.DEVNULL)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MLA_DEC", "LLM_"))}
    if variant != "base":
        for x in variant.split("+"):
            env.update(VARIANTS[x])
    cfg = os.path.join(HERE, "joyai_cfg.json")
    env.update(LLM_CONFIG=cfg, LLM_OUT=os.path.join(tmp, "gen"))
    subprocess.run([sys.executable, os.path.join(kit, "gen_llm.py"), cfg], check=True, env=env, stdout=subprocess.DEVNULL)
    text = open(os.path.join(tmp, "gen", "model.loom")).read()
    shutil.rmtree(tmp, ignore_errors=True)
    return text


def split_program(text):
    i = text.index("command.def public @llm_advance(")
    j = text.index("\n}\n", i) + 3
    lib = text[:i] + text[j:]
    lib = re.sub(r"^config\.(decl|def) .*\n", "", lib, flags=re.M)
    lib = lib.replace('amdgpu.target<gfx11-generic> @w32 {subgroup_size = 32}\n', "")
    return lib, text[i:j]


def specialize(body, ctx, pool):
    """[(kind, data)] in program order for the 1-token epoch: ("launch", line) / ("param", (ssa, name, type));
    vals: the static value of every index / offset / i32 / i1 SSA the body computes (tokens = spans = 1)."""
    vals = {"%t": 1, "%spans": 1, "%context": ctx, "%pool": pool, "%logits_on": 0, "%logit_slots": 0}
    allocs, out = {}, []
    stack = []   # [parent active, condition, in else]

    def active():
        return all(p and (c if not e else not c) for p, c, e in stack)

    def v(x):
        return vals[x]
    for raw in body.split("\n")[1:]:
        ln = raw.strip()
        if not ln or ln.startswith("//") or ln in ("command.return",):
            continue
        m = re.match(r"scf\.if (%\w+) \{$", ln)
        if m:
            c = m.group(1)
            par = active()
            if par and c not in vals:
                raise SystemExit("unknown condition " + c)
            stack.append([par, bool(vals.get(c, 0)), False])
            continue
        if ln == "} else {":
            stack[-1][2] = True
            continue
        if ln == "}":
            if stack:
                stack.pop()
            continue
        if not active():
            continue
        if ln.startswith("kernel.launch @g_embed["):   # x, h come from the suite's random rows
            continue
        if ln.startswith("kernel.launch"):
            out.append(("launch", ln))
            continue
        m = re.match(r'(%\w+) = command\.parameter %weights, "([^"]+)" : view<(\d+)x(\w+)>', ln)
        if m and m.group(2) == "token_embd.weight":
            continue
        if m:
            out.append(("param", (m.group(1), m.group(2), int(m.group(3)), m.group(4))))
            continue
        m = re.match(r"(%\w+) = (.*)$", ln)
        if not m:
            raise SystemExit("unhandled: " + ln)
        d, e = m.groups()
        if e.startswith("config.get"):
            assert d in vals, ln
            continue
        mm = re.match(r"(?:index|scalar)\.constant (-?[\d.e]+) : (\w+)", e)
        if mm:
            vals[d] = float(mm.group(1)) if "." in mm.group(1) else int(mm.group(1))
            continue
        mm = re.match(r"buffer\.alloca<global> align\(\d+\) (%\w+) : buffer", e)
        if mm:
            allocs[d] = v(mm.group(1))
            continue
        mm = re.match(r"index\.(scale|mul|add|div|rem|sub) (%\w+), (%\w+) :", e)
        if mm:
            a, b = v(mm.group(2)), v(mm.group(3))
            vals[d] = {"scale": a * b, "mul": a * b, "add": a + b, "div": a // b, "rem": a % b, "sub": a - b}[mm.group(1)]
            continue
        mm = re.match(r"index\.cmp (\w+), (%\w+), (%\w+) :", e)
        if mm:
            a, b = v(mm.group(2)), v(mm.group(3))
            vals[d] = int({"ule": a <= b, "uge": a >= b, "eq": a == b, "ult": a < b, "ugt": a > b, "ne": a != b}[mm.group(1)])
            continue
        mm = re.match(r"scalar\.andi (%\w+), (%\w+) :", e)
        if mm:
            vals[d] = v(mm.group(1)) & v(mm.group(2))
            continue
        mm = re.match(r"scf\.select (%\w+), (%\w+), (%\w+) :", e)
        if mm:
            vals[d] = v(mm.group(2)) if v(mm.group(1)) else v(mm.group(3))
            continue
        raise SystemExit("unhandled: " + ln)
    return out, vals, allocs


def origins(blocks):
    o = np.zeros(10, np.int32)
    o[8] = 40
    return np.concatenate([o, np.arange(blocks, dtype=np.int32)])


def step_case(lib, body, tag, ctx):
    pool = (ctx + 15) // 16 * 16
    seq, vals, allocs = specialize(body, ctx, pool)
    L = 40
    blocks = pool // 16
    np.save(os.path.join(OUT, f"meta_{tag}.npy"), np.array([1, 0, 0, 0, 1, ctx - 1, 0, 0, 0, 0, 0, 0, 0], np.int32))
    np.save(os.path.join(OUT, f"orig_{tag}.npy"), origins(blocks))
    types = {"%metadata": "tensor<13xi32>", "%origins": f"tensor<{10 + blocks}xi32>", "%arena": f"tensor<{L * pool * 1152}xi8>",
             "%outputs": "tensor<4096xi8>", "%residual": "tensor<4096xi8>", "%tokens": "tensor<64xi8>"}
    lines = [f'  %metadata = check.file.read.npy path("meta_{tag}.npy") : tensor<13xi32>',
             f'  %origins = check.file.read.npy path("orig_{tag}.npy") : tensor<{10 + blocks}xi32>',
             f"  %arena = check.generate.fill value(0) : tensor<{L * pool * 1152}xi8>",
             "  %outputs = check.generate.fill value(0) : tensor<4096xi8>"]
    for b, n in allocs.items():
        n = max(int(n), 256)
        if b in ("%x", "%h"):
            types[b] = "tensor<2048xf32>"
            lines.append(f'  {b} = check.file.read.npy path("{b[1:]}0.npy") : tensor<2048xf32>')
        else:
            types[b] = f"tensor<{n}xi8>"
            lines.append(f"  {b} = check.generate.fill value(0) : tensor<{n}xi8>")
    shared = {}
    renamed = {}
    defined = set()
    for kind, data in seq:
        if kind == "param":
            ssa, name, n, ty = data
            if ty == "f32":
                types[ssa] = f"tensor<{n}xf32>"
                if "ffn_gate_inp" in name:
                    lines.append(f'  {ssa} = check.file.read.npy path("router_{name.split(".")[1]}.npy") : tensor<{n}xf32>')
                else:
                    lines.append(f"  {ssa} = check.generate.fill value({'0.0' if 'bias' in name else '1.0'}) : tensor<{n}xf32>")
            else:
                types[ssa] = f"tensor<{n}x{ty}>"
                if "_exps." in name:   # one shared buffer per expert-tensor size
                    if n not in shared:
                        shared[n] = f"%exps_{len(shared)}"
                        lines.append(f"  {shared[n]} = check.generate.fill value(0) : tensor<{n}x{ty}>")
                    renamed[ssa] = shared[n]
                else:
                    lines.append(f"  {ssa} = check.generate.fill value(0) : tensor<{n}x{ty}>")
            continue
        m = re.match(r"kernel\.launch @(\w+)\[([^\]]*)\]\(([^)]*)\) : \[([^\]]*)\]\(([^)]*)\)$", data)
        assert m, data
        kn, grid, args, gtys, atys = m.groups()
        args = [a.strip() for a in args.split(",")] if args.strip() else []
        atys = [a.strip() for a in atys.split(",")] if atys.strip() else []
        grid = [g.strip() for g in grid.split(",")] if grid.strip() else []
        outb = [a for a, ty in zip(args, atys) if ty == "buffer"]
        role = re.sub(r"_\d+$", "", outb[-1][1:]) if outb else "x"
        new = f"k_{kn}__{role}"
        renamed_k = renamed.setdefault(("k", kn, role), new)
        ty2 = []
        args2 = []
        for a, ty in zip(args, atys):
            if ty in ("index", "i32", "i1", "offset"):
                lit = f"%lit_{a[1:]}"
                if lit not in defined:
                    lines.append(f"  {lit} = check.literal value({int(vals[a])}) : {ty}")
                    defined.add(lit)
                args2.append(lit)
                ty2.append(ty)
            else:
                a2 = renamed.get(a, a)
                args2.append(a2)
                ty2.append(types[a])
        grid2 = []
        for g in grid:
            lit = f"%lit_{g[1:]}"
            if lit not in defined:
                lines.append(f"  {lit} = check.literal value({int(vals[g])}) : index")
                defined.add(lit)
            grid2.append(lit)
        lines.append(f"  kernel.launch @{renamed_k}[{', '.join(grid2)}]({', '.join(args2)}) : [{gtys}]({', '.join(ty2)})")
    kdefs = []
    for key, new in renamed.items():
        if isinstance(key, tuple):
            _, kn, role = key
            mm = re.search(rf'^kernel\.def export\("{kn}"\) @{kn}\((?:.|\n)*?^\}}\n', lib, re.M)
            assert mm, kn
            kdefs.append(mm.group(0).replace(f'export("{kn}") @{kn}(', f'export("{new}") @{new}(', 1))
    case = f"check.case public @c{tag} {{\n" + "\n".join(lines) + "\n  check.return\n}\n" + f"check.benchmark<@c{tag}> @{tag}\n"
    return kdefs, case


def strip_kernels(lib):
    """the library without its kernel.defs (funcs, rodata stay)"""
    return re.sub(r'^kernel\.def [^\n]*\n(?:.|\n)*?^\}\n', "", lib, flags=re.M)


def bench(variant):
    lib, body = split_program(program(variant))
    parts, kd = [strip_kernels(lib)], {}
    cases = []
    for tag, ctx in CTXS:
        kdefs, case = step_case(lib, body, tag, ctx)
        for k in kdefs:
            kd[re.search(r'export\("(\w+)"\)', k).group(1)] = k
        cases.append(case)
    return HEAD + "\n".join(parts + list(kd.values()) + cases)


if __name__ == "__main__" and len(sys.argv) > 2:
    OUT = sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(3)
    for i in range(40):
        np.save(os.path.join(OUT, f"router_{i}.npy"), rng.standard_normal(256 * 2048).astype(np.float32) * 0.05)
    np.save(os.path.join(OUT, "x0.npy"), rng.standard_normal(2048).astype(np.float32))
    np.save(os.path.join(OUT, "h0.npy"), rng.standard_normal(2048).astype(np.float32))
    for v in sys.argv[3:] or ["base"]:
        if v != "base" and not all(x in VARIANTS for x in v.split("+")):
            raise SystemExit("unknown variant " + v)
        open(os.path.join(OUT, "bench%s.loom" % ("" if v == "base" else "_" + v)), "w").write(bench(v))
