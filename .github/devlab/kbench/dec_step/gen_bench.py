#!/usr/bin/env python3
"""Generate the dec_step kbench suite: ONE single-token decode step of JoyAI-LLM-Flash (deepseek2 / MLA, 40 layers, IQ4_XS)
exactly as the deepseek2 example's program launches it (the per-kernel profile of decode without a published card).

usage: gen_bench.py ROOT OUTDIR [VARIANT ...]
The program is generated from this checkout (apply_example.sh deepseek2 + gen_llm.py with joyai_cfg.json, the GGUF's
header config); its llm_advance command is specialized to tokens = spans = 1 (the scf.if branches a 1-token epoch takes)
and becomes one check.case per context (cases CTXS) with every launch kept in order. Each kernel is renamed
k_<kernel>__<output buffer> so profile rows separate roles (q_a vs kv_a ...). Benchmark rows: k_case_<ctx> = host wall of
the whole step (launch gaps included); k_case_role_<kernel> = host wall of one role's launches alone (all layers, ROLE_CTX):
the per-role profile (the replay profile keeps only the first rows of a 760-dispatch step).
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
            "b2": {"MLA_DEC_BMIN": "2"}, "b1": {"MLA_DEC_BMIN": "1"}, "shslot": {"MOE_SHSLOT": "1"},
            "iq4q8": {"IQ4_Q8": "1"}, "wg1k": {"MV_WG1": "1024"}, "wg256": {"MV_WG1": "256"}, "moer4": {"MOE_R": "4"}, "moer1": {"MOE_R": "1"},
            "rt1": {"MOE_ROUTE_SPLIT": "0"}, "rt2": {"MOE_ROUTE_FUSE": "0"}, "ropekb": {"MLA_ROPE_KB": "1"}, "dd4": {"MOE_DEDUP": "4"}, "wg128": {"MV_WG1": "128"}, "wg64": {"MV_WG1": "64"}, "sh0": {"MOE_SHSLOT": "0"}, "qadual": {"MLA_QA_DUAL": "1"}, "rmsx": {"MLA_RMSQ_FUSE": "1"},
            "dnr1": {"MOE_R_DN": "1"}, "dnr4": {"MOE_R_DN": "4"}, "dnr8": {"MOE_R_DN": "8"}, "dnu2": {"MOE_DN_U": "2"}, "dnu4": {"MOE_DN_U": "4"},
            "old": {"MLA_DEC": "split", "MOE_ROUTE_SPLIT": "0", "MOE_SHSLOT": "0", "MLA_QA_DUAL": "0", "MLA_RMSQ_FUSE": "0", "MV_WG1": "4096"}}
CTXS = [("short", 300, 1), ("2k", 2048, 1), ("8k", 8192, 1), ("c4", 1024, 4)]   # (tag, context, tokens = streams)
ROLE_CTX = "none"   # role_<kernel> cases (host wall of that role's 39-40 launches alone) at this context
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


def specialize(body, ctx, pool, T=1):
    """[(kind, data)] in program order for the 1-token epoch: ("launch", line) / ("param", (ssa, name, type));
    vals: the static value of every index / offset / i32 / i1 SSA the body computes (tokens = spans = 1)."""
    vals = {"%t": T, "%spans": T, "%context": ctx, "%pool": pool, "%logits_on": 0, "%logit_slots": 0}
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


def origins(blocks, T=1):
    """T rows: records (10 i32 each, the page map's byte offset at +32), then each row's page map (its own pages)"""
    o = [np.zeros(10 * T, np.int32)]
    for r in range(T):
        o[0][10 * r + 8] = 40 * T + 4 * blocks * r
    o += [np.arange(blocks, dtype=np.int32) + r * blocks for r in range(T)]
    return np.concatenate(o)


def step_case(lib, body, tag, ctx, T=1, roles=True):
    """one step of T tokens, one per row (T = 4: 4 streams, each at context ctx)"""
    blocks = (ctx + 15) // 16
    pool = blocks * 16 * T
    seq, vals, allocs = specialize(body, blocks * 16, pool, T)
    L = 40
    meta = np.zeros(4 + 9 * T, np.int32)
    meta[0] = T
    for r in range(T):
        meta[4 + 9 * r:4 + 9 * r + 5] = (1, ctx - 1, 0, r, r)
    np.save(os.path.join(OUT, f"meta_{tag}.npy"), meta)
    org = origins(blocks, T)
    np.save(os.path.join(OUT, f"orig_{tag}.npy"), org)
    types = {"%metadata": f"tensor<{len(meta)}xi32>", "%origins": f"tensor<{len(org)}xi32>", "%arena": f"tensor<{L * pool * 1152}xi8>",
             "%outputs": "tensor<4096xi8>", "%residual": "tensor<4096xi8>", "%tokens": "tensor<64xi8>"}
    lines = [f'  %metadata = check.file.read.npy path("meta_{tag}.npy") : tensor<{len(meta)}xi32>',
             f'  %origins = check.file.read.npy path("orig_{tag}.npy") : tensor<{len(org)}xi32>',
             f"  %arena = check.generate.fill value(0) : tensor<{L * pool * 1152}xi8>",
             "  %outputs = check.generate.fill value(0) : tensor<4096xi8>"]
    for b, n in allocs.items():
        n = max(int(n), 256)
        if b in ("%x", "%h"):
            types[b] = f"tensor<{T}x2048xf32>"
            lines.append(f'  {b} = check.file.read.npy path("{b[1:]}{T - 1}.npy") : tensor<{T}x2048xf32>')
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
    if tag == ROLE_CTX and roles:   # one case per kernel role: only that role's launches (all layers) and the values they read
        defs = {}
        for ln in lines:
            m = re.match(r"  (%\w+) = ", ln)
            if m:
                defs[m.group(1)] = ln
        for k in sorted({re.match(r"  kernel\.launch @(\w+)", ln).group(1) for ln in lines if ln.startswith("  kernel.launch")}):
            ls = [ln for ln in lines if ln.startswith(f"  kernel.launch @{k}[")]
            if any(", %ids," in ln for ln in ls):   # alone, the routed matvecs get each layer's own random experts
                ls2 = []
                for n, ln in enumerate(ls):
                    ls2.append(f'  %ids_{n} = check.file.read.npy path("ids_{n}.npy") : {types["%ids"]}')
                    ls2.append(ln.replace(", %ids,", f", %ids_{n},", 1))
                ls = ls2
            used = []
            for ln in ls:
                for v in re.findall(r"%\w+", ln.split(" : ")[0]):
                    if v in defs and defs[v] not in used:
                        used.append(defs[v])
            if "%info" in defs and defs["%info"] in used:   # alone: a live token at the context end (g_token_info's output)
                n = int(re.search(r"tensor<(\d+)xi8>", defs["%info"]).group(1))
                a = np.zeros(n, np.int8); a[:16] = np.array([0, ctx - 1, 1, 0], np.int32).view(np.int8)
                np.save(os.path.join(OUT, f"info_{tag}.npy"), a)
                used[used.index(defs["%info"])] = f'  %info = check.file.read.npy path("info_{tag}.npy") : tensor<{n}xi8>'
            r = k[2:]
            # the step without this role: its in-situ cost (= k_case_<ROLE_CTX> - k_case_minus_<role>)
            case += (f"check.case public @m_{r} {{\n" + "\n".join(ln for ln in lines if not ln.startswith(f"  kernel.launch @{k}["))
                     + "\n  check.return\n}\n" + f"check.benchmark<@m_{r}> @minus_{r}\n")
            case += f"check.case public @r_{r} {{\n" + "\n".join(used + ls) + "\n  check.return\n}\n" + f"check.benchmark<@r_{r}> @role_{r}\n"
    return kdefs, case


def strip_kernels(lib):
    """the library without its kernel.defs (funcs, rodata stay)"""
    return re.sub(r'^kernel\.def [^\n]*\n(?:.|\n)*?^\}\n', "", lib, flags=re.M)


def bench(variant):
    lib, body = split_program(program(variant))
    parts, kd = [strip_kernels(lib)], {}
    cases = []
    for tag, ctx, T in CTXS:
        kdefs, case = step_case(lib, body, tag, ctx, T, roles=variant == "base")   # role / minus cases: base only
        for k in kdefs:
            kd[re.search(r'export\("(\w+)"\)', k).group(1)] = k
        cases.append(case)
    return HEAD + "\n".join(parts + list(kd.values()) + cases)


def route_test(lib):
    """test.loom: the split decode router (g_moe_route_lg + g_moe_route_tk) against g_moe_route, ids and weights equal,
    4 tokens of random h, random router and bias (random npy written by main)."""
    ks = []
    for k in ("g_moe_route", "g_moe_route_lg", "g_moe_route_tk", "g_moe_route_lgk", "g_moe_cnt_zero"):
        m = re.search(rf'^kernel\.def export\("{k}"\) @{k}\((?:.|\n)*?^\}}\n', lib, re.M)
        assert m, k
        ks.append(m.group(0))
    T = 4
    case = f"""check.case public @route_split {{
  %t = check.literal value({T}) : index
  %h = check.file.read.npy path("rt_h.npy") : tensor<{T}x2048xf32>
  %w = check.file.read.npy path("rt_w.npy") : tensor<{256 * 2048}xf32>
  %b = check.file.read.npy path("rt_b.npy") : tensor<256xf32>
  %ids_a = check.generate.fill value(0) : tensor<{T}x8xi32>
  %wts_a = check.generate.fill value(0.0) : tensor<{T}x8xf32>
  %ids_b = check.generate.fill value(0) : tensor<{T}x8xi32>
  %wts_b = check.generate.fill value(0.0) : tensor<{T}x8xf32>
  %lg = check.generate.fill value(0.0) : tensor<{T}x256xf32>
  kernel.launch @g_moe_route[%t](%t, %h, %w, %b, %ids_a, %wts_a) : [index](index, tensor<{T}x2048xf32>, tensor<{256 * 2048}xf32>, tensor<256xf32>, tensor<{T}x8xi32>, tensor<{T}x8xf32>)
  kernel.launch @g_moe_route_lg[%t](%t, %h, %w, %lg) : [index](index, tensor<{T}x2048xf32>, tensor<{256 * 2048}xf32>, tensor<{T}x256xf32>)
  kernel.launch @g_moe_route_tk[%t](%t, %lg, %b, %ids_b, %wts_b) : [index](index, tensor<{T}x256xf32>, tensor<256xf32>, tensor<{T}x8xi32>, tensor<{T}x8xf32>)
  check.expect.equal actual(%ids_b) expected(%ids_a) : tensor<{T}x8xi32>
  check.expect.equal actual(%wts_b) expected(%wts_a) : tensor<{T}x8xf32>
  %cnt = check.generate.fill value(7) : tensor<{T}xi32>
  kernel.launch @g_moe_cnt_zero[%t](%t, %cnt) : [index](index, tensor<{T}xi32>)
""" + "".join(f"""  %ids_c{r} = check.generate.fill value(0) : tensor<{T}x8xi32>
  %wts_c{r} = check.generate.fill value(0.0) : tensor<{T}x8xf32>
  %lg{r} = check.generate.fill value(0.0) : tensor<{T}x256xf32>
  kernel.launch @g_moe_route_lgk[%t](%t, %h, %w, %lg{r}, %cnt, %b, %ids_c{r}, %wts_c{r}) : [index](index, tensor<{T}x2048xf32>, tensor<{256 * 2048}xf32>, tensor<{T}x256xf32>, tensor<{T}xi32>, tensor<256xf32>, tensor<{T}x8xi32>, tensor<{T}x8xf32>)
  check.expect.equal actual(%ids_c{r}) expected(%ids_a) : tensor<{T}x8xi32>
  check.expect.equal actual(%wts_c{r}) expected(%wts_a) : tensor<{T}x8xf32>
""" for r in range(3)) + f"""  check.return
}}
"""
    return HEAD + strip_kernels(lib) + "\n".join(ks) + case


def kname(lib, pat):
    """the one exported kernel whose name fully matches PAT (the R suffixes follow the generator's choices)"""
    ns = sorted(set(re.findall(r'^kernel\.def export\("(' + pat.replace("(_r", "(?:_r") + r')"\)', lib, re.M)))
    assert len(ns) == 1, (pat, ns)
    return ns[0]


def iq4xs_rand(rng, nbytes):
    """random IQ4_XS blocks (136 bytes per 256 values) with a sane f16 scale d = 0.02 (random bytes give inf / NaN)"""
    b = rng.integers(0, 256, nbytes, dtype=np.uint8).reshape(-1, 136)
    b[:, 0:2] = np.array([0.02], np.float16).view(np.uint8)
    return b.reshape(-1).view(np.int8)


def qadual_test(lib, rng):
    """mv_dual (MLA_QA_DUAL: q_a + kv_a in one launch) against the two single launches: equal outputs, 1 and 4 tokens."""
    D1, D4 = kname(lib, r"g_mv_iq4_xs_k2048(_r\d+)?_dual"), kname(lib, r"g_mv_iq4_xs_k2048_t4(_r\d+)?_dual")
    names = sorted({D1, D4} | set(re.findall(r'^kernel\.def export\("(g_mv_iq4_xs_k2048(?:_t4)?(?:_r\d+)?)"\)', lib, re.M)))
    ks = []
    for k in names:
        m = re.search(rf'^kernel\.def export\("{k}"\) @{k}\((?:.|\n)*?^\}}\n', lib, re.M)
        assert m, k
        ks.append(m.group(0).replace(f'export("{k}") @{k}(', f'export("q_{k}") @q_{k}('))
    H, NA, NB = 2048, 1536, 576
    np.save(os.path.join(OUT, "qd_wa.npy"), iq4xs_rand(rng, NA * H // 256 * 136))
    np.save(os.path.join(OUT, "qd_wb.npy"), iq4xs_rand(rng, NB * H // 256 * 136))
    out = []
    def single(t4):
        c = [n for n in names if not n.endswith("_dual") and ("_t4" in n) == t4]
        r = lambda n: int(n.rsplit("_r", 1)[1]) if "_r" in n else 16
        return [n for n in c if 576 % r(n) == 0][0]
    for T, k1, kd in ((1, single(False), D1), (4, single(True), D4)):
        np.save(os.path.join(OUT, f"qd_h{T}.npy"), rng.standard_normal((T, H)).astype(np.float32))
        WA, WB, HT, AT, BT = f"tensor<{NA * H // 256 * 136}xi8>", f"tensor<{NB * H // 256 * 136}xi8>", f"tensor<{T}x{H}xf32>", f"tensor<{T}x{NA}xf32>", f"tensor<{T}x{NB}xf32>"
        out.append(f"""check.case public @qadual_t{T} {{
  %t = check.literal value({T}) : index
  %na = check.literal value({NA}) : index
  %nb = check.literal value({NB}) : index
  %nab = check.literal value({NA + NB}) : index
  %z = check.literal value(0) : i32
  %h = check.file.read.npy path("qd_h{T}.npy") : {HT}
  %wa = check.file.read.npy path("qd_wa.npy") : {WA}
  %wb = check.file.read.npy path("qd_wb.npy") : {WB}
  %ya = check.generate.fill value(0.0) : {AT}
  %yb = check.generate.fill value(0.0) : {BT}
  %ya2 = check.generate.fill value(0.0) : {AT}
  %yb2 = check.generate.fill value(0.0) : {BT}
  kernel.launch @q_{k1}[%na, %t](%na, %t, %z, %h, %wa, %ya) : [index, index](index, index, i32, {HT}, {WA}, {AT})
  kernel.launch @q_{k1}[%nb, %t](%nb, %t, %z, %h, %wb, %yb) : [index, index](index, index, i32, {HT}, {WB}, {BT})
  kernel.launch @q_{kd}[%nab, %t](%nab, %t, %z, %h, %wa, %wb, %ya2, %yb2, %na) : [index, index](index, index, i32, {HT}, {WA}, {WB}, {AT}, {BT}, index)
  check.expect.equal actual(%ya2) expected(%ya) : {AT}
  check.expect.equal actual(%yb2) expected(%yb) : {BT}
  check.return
}}
""")
    return "\n".join(ks) + "".join(out)


def rmsx_test(lib, rng):
    """mv_rmsx (MLA_RMSQ_FUSE) against g_mla_rms_q + the q_b matvec, 1 and 4 tokens (approximate: the norm's sum order)."""
    out, ks = [], []
    for T, k1 in ((1, "g_mv_iq4_xs_k1536"), (4, "g_mv_iq4_xs_k1536_t4")):
        for k in (k1, k1 + "_rmsx") + (("g_mla_rms_q",) if T == 1 else ()):
            m = re.search(rf'^kernel\.def export\("{k}"\) @{k}\((?:.|\n)*?^\}}\n', lib, re.M)
            assert m, k
            ks.append(m.group(0).replace(f'export("{k}") @{k}(', f'export("r_{k}") @r_{k}('))
        K, N = 1536, 6144
        wb = N * K // 256 * 136
        np.save(os.path.join(OUT, f"rx_x{T}.npy"), (rng.standard_normal((T, K)) * 3).astype(np.float32))
        if T == 1:
            np.save(os.path.join(OUT, "rx_w.npy"), iq4xs_rand(rng, wb))
            np.save(os.path.join(OUT, "rx_n.npy"), rng.uniform(0.5, 1.5, K).astype(np.float32))
        XT, YT, WT, NT = f"tensor<{T}x{K}xf32>", f"tensor<{T}x{N}xf32>", f"tensor<{wb}xi8>", f"tensor<{K}xf32>"
        out.append(f"""check.case public @rmsx_t{T} {{
  %t = check.literal value({T}) : index
  %n = check.literal value({N}) : index
  %z = check.literal value(0) : i32
  %x = check.file.read.npy path("rx_x{T}.npy") : {XT}
  %x2 = check.file.read.npy path("rx_x{T}.npy") : {XT}
  %w = check.file.read.npy path("rx_w.npy") : {WT}
  %nw = check.file.read.npy path("rx_n.npy") : {NT}
  %y = check.generate.fill value(0.0) : {YT}
  %y2 = check.generate.fill value(0.0) : {YT}
  kernel.launch @r_g_mla_rms_q[%t](%t, %x, %nw) : [index](index, {XT}, {NT})
  kernel.launch @r_{k1}[%n, %t](%n, %t, %z, %x, %w, %y) : [index, index](index, index, i32, {XT}, {WT}, {YT})
  kernel.launch @r_{k1}_rmsx[%n, %t](%n, %t, %z, %x2, %w, %y2, %nw) : [index, index](index, index, i32, {XT}, {WT}, {YT}, {NT})
  check.expect.approximate actual(%y2) expected(%y) atol(1.0e-3) rtol(1.0e-4) nan(different) : {YT}, {YT}
  check.return
}}
""")
    return "\n".join(ks) + "".join(out)


def ropekb_test(lib, rng):
    """g_mla_ropekb (MLA_ROPE_KB) against g_mla_rope_store + g_mla_kb: q and the cache equal, 2 live tokens + 1 dead."""
    ks = []
    for k in ("g_mla_rope_store", "g_mla_kb_iq4_nl", "g_mla_ropekb_iq4_nl"):
        m = re.search(rf'^kernel\.def export\("{k}"\) @{k}\((?:.|\n)*?^\}}\n', lib, re.M)
        assert m, k
        ks.append(m.group(0).replace(f'export("{k}") @{k}(', f'export("rk_{k}") @rk_{k}('))
    T, L, CTX = 3, 40, 64
    blocks = CTX // 16
    info = np.array([[0, 37, 1, 0], [0, 5, 0, 0], [0, 38, 1, 0]], np.int32)
    o = np.zeros(10, np.int32); o[8] = 40
    org = np.concatenate([o, rng.permutation(blocks).astype(np.int32)])
    wkb = rng.integers(0, 256, 1179648, dtype=np.uint8).reshape(-1, 18)
    wkb[:, 0:2] = np.array([0.02], np.float16).view(np.uint8)
    for n, a in (("info", info), ("org", org), ("qb", rng.standard_normal((T, 32 * 192)).astype(np.float32)),
                 ("kv", rng.standard_normal((T, 576)).astype(np.float32)), ("kvn", rng.uniform(0.5, 1.5, 512).astype(np.float32)),
                 ("wkb", wkb.reshape(-1).view(np.int8))):
        np.save(os.path.join(OUT, f"rk_{n}.npy"), a)
    QB, KV, QT, AR, IT, OT, WT, NT = (f"tensor<{T}x{32 * 192}xf32>", f"tensor<{T}x576xf32>", f"tensor<{T}x{32 * 576}xf32>",
                                      f"tensor<{L * CTX * 1152}xi8>", f"tensor<{T}x4xi32>", f"tensor<{len(org)}xi32>", "tensor<1179648xi8>", "tensor<512xf32>")
    case = f"""check.case public @ropekb {{
  %t = check.literal value({T}) : index
  %l = check.literal value(3) : index
  %ctx = check.literal value({CTX}) : index
  %pool = check.literal value({CTX}) : index
  %info = check.file.read.npy path("rk_info.npy") : {IT}
  %org = check.file.read.npy path("rk_org.npy") : {OT}
  %qb = check.file.read.npy path("rk_qb.npy") : {QB}
  %kv = check.file.read.npy path("rk_kv.npy") : {KV}
  %kv2 = check.file.read.npy path("rk_kv.npy") : {KV}
  %kvn = check.file.read.npy path("rk_kvn.npy") : {NT}
  %w = check.file.read.npy path("rk_wkb.npy") : {WT}
  %q = check.generate.fill value(0.0) : {QT}
  %q2 = check.generate.fill value(0.0) : {QT}
  %ar = check.generate.fill value(0) : {AR}
  %ar2 = check.generate.fill value(0) : {AR}
  kernel.launch @rk_g_mla_rope_store[%t](%t, %l, %ctx, %pool, %qb, %kv, %kvn, %q, %info, %org, %ar) : [index](index, index, index, index, {QB}, {KV}, {NT}, {QT}, {IT}, {OT}, {AR})
  kernel.launch @rk_g_mla_kb_iq4_nl[%t](%t, %qb, %w, %q) : [index](index, {QB}, {WT}, {QT})
  kernel.launch @rk_g_mla_ropekb_iq4_nl[%t](%t, %l, %ctx, %pool, %qb, %kv2, %kvn, %q2, %info, %org, %ar2, %w) : [index](index, index, index, index, {QB}, {KV}, {NT}, {QT}, {IT}, {OT}, {AR}, {WT})
  check.expect.equal actual(%q2) expected(%q) : {QT}
  check.expect.equal actual(%ar2) expected(%ar) : {AR}
  check.expect.equal actual(%kv2) expected(%kv) : {KV}
  check.return
}}
"""
    return "\n".join(ks) + case


def dedup_test(lib, rng):
    """expert_mv_dedup (MOE_DEDUP: each distinct expert read once) against the shared-slot kernels: act and ys equal for
    1..4 tokens whose routed experts overlap (16 experts, IQ4_XS gate/up/down)."""
    GU = re.search(r'export\("(g_moe_gu_iq4_xs_k2048_r\d+_sh)"\)', lib).group(1)
    DN = re.search(r'export\("(g_moe_dn_iq4_xs_k768_r\d+_sh)"\)', lib).group(1)
    ks = []
    for k in (GU, DN, GU + "d", DN + "d"):
        m = re.search(rf'^kernel\.def export\("{k}"\) @{k}\((?:.|\n)*?^\}}\n', lib, re.M)
        assert m, k
        ks.append(m.group(0).replace(f'export("{k}") @{k}(', f'export("d_{k}") @d_{k}('))
    E, KU, H, F = 16, 8, 2048, 768
    eb = F * H // 256 * 136
    for n, sz in (("wg", E * eb), ("wu", E * eb), ("wd", E * eb), ("sg", eb), ("su", eb), ("sd", eb)):
        np.save(os.path.join(OUT, f"dd_{n}.npy"), iq4xs_rand(rng, sz))
    W, S = f"tensor<{E * eb}xi8>", f"tensor<{eb}xi8>"
    out = []
    for T in (1, 2, 4):
        ids = np.stack([rng.choice(10, KU, replace=False) for _ in range(T)]).astype(np.int32)   # 10 of 16: overlaps
        np.save(os.path.join(OUT, f"dd_ids{T}.npy"), ids)
        np.save(os.path.join(OUT, f"dd_h{T}.npy"), rng.standard_normal((T, H)).astype(np.float32))
        HT, IT = f"tensor<{T}x{H}xf32>", f"tensor<{T}x{KU}xi32>"
        AT, YS = f"tensor<{T * (KU + 1) * F}xf32>", f"tensor<{T * (KU + 1) * H}xf32>"
        out.append(f"""check.case public @dedup_t{T} {{
  %ta = check.literal value({T * KU}) : index
  %nH = check.literal value({H}) : index
  %nF = check.literal value({F}) : index
  %h = check.file.read.npy path("dd_h{T}.npy") : {HT}
  %ids = check.file.read.npy path("dd_ids{T}.npy") : {IT}
""" + "".join(f'  %{n} = check.file.read.npy path("dd_{n}.npy") : {W if n[0] == "w" else S}\n' for n in ("wg", "wu", "wd", "sg", "su", "sd")) + f"""  %act = check.generate.fill value(0.0) : {AT}
  %ys = check.generate.fill value(0.0) : {YS}
  %act2 = check.generate.fill value(0.0) : {AT}
  %ys2 = check.generate.fill value(0.0) : {YS}
  kernel.launch @d_{GU}[%nF, %ta](%nF, %ta, %ids, %h, %wg, %wu, %sg, %su, %act) : [index, index](index, index, {IT}, {HT}, {W}, {W}, {S}, {S}, {AT})
  kernel.launch @d_{DN}[%nH, %ta](%nH, %ta, %ids, %act, %wd, %sd, %ys) : [index, index](index, index, {IT}, {AT}, {W}, {S}, {YS})
  kernel.launch @d_{GU}d[%nF, %ta](%nF, %ta, %ids, %h, %wg, %wu, %sg, %su, %act2) : [index, index](index, index, {IT}, {HT}, {W}, {W}, {S}, {S}, {AT})
  kernel.launch @d_{DN}d[%nH, %ta](%nH, %ta, %ids, %act2, %wd, %sd, %ys2) : [index, index](index, index, {IT}, {AT}, {W}, {S}, {YS})
  check.expect.equal actual(%act2) expected(%act) : {AT}
  check.expect.equal actual(%ys2) expected(%ys) : {YS}
  check.return
}}
""")
    return "\n".join(ks) + "".join(out)


def shslot_test(lib, rng):
    """the shared expert as a ninth slot (g_moe_gu/dn_..._sh + g_moe_combine_sh) against the separate launches (shared
    swiglu matvec + down matvec into y, routed gu / dn, g_moe_combine accumulating): 2 tokens, 16 experts, IQ4_XS."""
    MS = re.search(r"kernel\.launch @(g_mv_iq4_xs_k2048_swiglu(?:_r\d+)?)\[%nFS, %t\]", lib).group(1)
    MD = re.search(r"kernel\.launch @(g_mv_iq4_xs_k768(?:_r\d+)?)\[%nH, %t\]", lib).group(1)
    DSH = re.search(r"kernel\.launch @(g_moe_dn_iq4_xs_k768_r\d+_sh)\[", lib).group(1)
    names = [MS, MD, "g_moe_gu_iq4_xs_k2048_r2", "g_moe_dn_iq4_xs_k768_r2",
             "g_moe_combine", "g_moe_gu_iq4_xs_k2048_r2_sh", DSH, "g_moe_combine_sh"]
    ks = []
    for k in names:
        m = re.search(rf'^kernel\.def export\("{k}"\) @{k}\((?:.|\n)*?^\}}\n', lib, re.M)
        assert m, k
        ks.append(m.group(0))
    T, E, KU, H, F = 2, 16, 8, 2048, 768
    eb = F * H // 256 * 136
    np.save(os.path.join(OUT, "sh_h.npy"), rng.standard_normal((T, H)).astype(np.float32))
    for n, sz in (("wg", E * eb), ("wu", E * eb), ("wd", E * eb), ("sg", eb), ("su", eb), ("sd", eb)):
        np.save(os.path.join(OUT, f"sh_{n}.npy"), iq4xs_rand(rng, sz))
    ids = np.stack([rng.choice(E, KU, replace=False) for _ in range(T)]).astype(np.int32)
    np.save(os.path.join(OUT, "sh_ids.npy"), ids)
    np.save(os.path.join(OUT, "sh_wts.npy"), rng.uniform(0.05, 0.3, (T, KU)).astype(np.float32))
    W, S = f"tensor<{E * eb}xi8>", f"tensor<{eb}xi8>"
    HT, IT, WT = f"tensor<{T}x{H}xf32>", f"tensor<{T}x{KU}xi32>", f"tensor<{T}x{KU}xf32>"
    AT, YS, AT2, YS2 = (f"tensor<{T * KU * F}xf32>", f"tensor<{T * KU * H}xf32>", f"tensor<{T * (KU + 1) * F}xf32>",
                        f"tensor<{T * (KU + 1) * H}xf32>")
    case = f"""check.case public @shslot {{
  %t = check.literal value({T}) : index
  %ta = check.literal value({T * KU}) : index
  %nH = check.literal value({H}) : index
  %nF = check.literal value({F}) : index
  %z = check.literal value(0) : i32
  %h = check.file.read.npy path("sh_h.npy") : {HT}
  %ids = check.file.read.npy path("sh_ids.npy") : {IT}
  %wts = check.file.read.npy path("sh_wts.npy") : {WT}
""" + "".join(f'  %{n} = check.file.read.npy path("sh_{n}.npy") : {W if n[0] == "w" else S}\n' for n in ("wg", "wu", "wd", "sg", "su", "sd")) + f"""  %f = check.generate.fill value(0.0) : tensor<{T}x{F}xf32>
  %y = check.generate.fill value(0.0) : {HT}
  %act = check.generate.fill value(0.0) : {AT}
  %ys = check.generate.fill value(0.0) : {YS}
  %y2 = check.generate.fill value(0.0) : {HT}
  %act2 = check.generate.fill value(0.0) : {AT2}
  %ys2 = check.generate.fill value(0.0) : {YS2}
  kernel.launch @{MS}[%nF, %t](%nF, %t, %z, %h, %sg, %su, %f) : [index, index](index, index, i32, {HT}, {S}, {S}, tensor<{T}x{F}xf32>)
  kernel.launch @{MD}[%nH, %t](%nH, %t, %z, %f, %sd, %y) : [index, index](index, index, i32, tensor<{T}x{F}xf32>, {S}, {HT})
  kernel.launch @g_moe_gu_iq4_xs_k2048_r2[%nF, %ta](%nF, %ta, %ids, %h, %wg, %wu, %act) : [index, index](index, index, {IT}, {HT}, {W}, {W}, {AT})
  kernel.launch @g_moe_dn_iq4_xs_k768_r2[%nH, %ta](%nH, %ta, %ids, %act, %wd, %ys) : [index, index](index, index, {IT}, {AT}, {W}, {YS})
  kernel.launch @g_moe_combine[%t](%t, %ys, %wts, %y) : [index](index, {YS}, {WT}, {HT})
  kernel.launch @g_moe_gu_iq4_xs_k2048_r2_sh[%nF, %ta](%nF, %ta, %ids, %h, %wg, %wu, %sg, %su, %act2) : [index, index](index, index, {IT}, {HT}, {W}, {W}, {S}, {S}, {AT2})
  kernel.launch @{DSH}[%nH, %ta](%nH, %ta, %ids, %act2, %wd, %sd, %ys2) : [index, index](index, index, {IT}, {AT2}, {W}, {S}, {YS2})
  kernel.launch @g_moe_combine_sh[%t](%t, %ys2, %wts, %y2) : [index](index, {YS2}, {WT}, {HT})
  check.expect.approximate actual(%y2) expected(%y) atol(1.0e-2) rtol(1.0e-3) nan(different) : {HT}, {HT}
  check.return
}}
"""
    return "\n".join(ks) + case


if __name__ == "__main__" and len(sys.argv) > 2:
    OUT = sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(3)
    for i in range(40):
        np.save(os.path.join(OUT, f"router_{i}.npy"), rng.standard_normal(256 * 2048).astype(np.float32) * 0.05)
    for T in (1, 4):
        np.save(os.path.join(OUT, f"x{T - 1}.npy"), rng.standard_normal((T, 2048)).astype(np.float32))
        np.save(os.path.join(OUT, f"h{T - 1}.npy"), rng.standard_normal((T, 2048)).astype(np.float32))
    for n in range(40):   # role cases: each layer's 8 distinct experts (int32, as the program's ids buffer, 256 bytes)
        e = np.zeros(64, np.int32); e[:8] = (8 * n + np.arange(8)) % 256   # no expert re-read within 32 layers (MALL 32 MB)
        np.save(os.path.join(OUT, f"ids_{n}.npy"), e.view(np.int8))
    np.save(os.path.join(OUT, "rt_h.npy"), rng.standard_normal((4, 2048)).astype(np.float32))
    np.save(os.path.join(OUT, "rt_w.npy"), (rng.standard_normal(256 * 2048) * 0.05).astype(np.float32))
    np.save(os.path.join(OUT, "rt_b.npy"), (rng.standard_normal(256) * 0.01).astype(np.float32))
    prog0 = program("base")
    lib0, cmd0 = split_program(prog0)
    lib1, _ = split_program(program("qadual"))
    open(os.path.join(OUT, "test.loom"), "w").write(route_test(lib0) + shslot_test(lib0 + cmd0, rng) + qadual_test(lib1, rng)
                                                           + rmsx_test(split_program(program("rmsx"))[0], rng)
                                                           + ropekb_test(split_program(program("ropekb"))[0], rng)
                                                           + dedup_test(split_program(program("dd4"))[0], rng))
    for v in sys.argv[3:] or ["base"]:
        if v != "base" and not all(x in VARIANTS for x in v.split("+")):
            raise SystemExit("unknown variant " + v)
        open(os.path.join(OUT, "bench%s.loom" % ("" if v == "base" else "_" + v)), "w").write(bench(v))
