"""Chooses the speculative-decoding method and depth for a port by measuring decode speed (qualify.sh runs it).

usage: tune.py WORKDIR PORT MEMORY_GIB MODEL_ID (MEMORY_GIB is ignored: build.py sizes each candidate)   (WORKDIR holds cfg.json, tok/tokenizer.json and pkg/; Loom must not be
                                                 serving another model: one model is loaded at a time)
Candidates: no drafting; MTP (the GGUF's NextN block) at depths 1-4; DFlash (cfg["drafter"], found by find_drafter.py)
at depths 3, 5, 7 (capped by its block); a Gemma-4 assistant (cfg["assistant"]) at depths 1-4; n-gram prompt lookup (any model, no weights) at depths 1, 2, 3, 5. Each method
is built once; a depth is only a model.json setting (the width-specialized stages compile on first use). Every
candidate is served alone and timed on decode-heavy requests (short essay, code, a tool-call turn, a reasoning answer,
a code edit that returns most of its input as agents' file edits do, and one decode after a 16k-token document, at most
half the context; greedy, client-side tokens/s) and, with a multi-row preset, on `rows` concurrent essays (streams4).
Each candidate's margin is its worst row relative to depth 0. A drafter is used only if no row is slower than 0.98 x depth 0
(prose, long-document and 4-stream decode are the card's headline rows: low-acceptance drafting loses 20-40 % there) and
its single-stream rate (harmonic mean) beats no drafting by at least 3 %; the best margin wins; its kind/depth go to cfg["drafter"] and pkg/ is rebuilt with it
(build.py --draft auto reads them). Before it is kept, the fastest candidate must pass a concurrency check and the
spec gate's equivalence check (loom-benchmark spec_check.py: the same requests at the candidate depth and at
draft_depth 0 of the same package, identical); a candidate that fails either is recorded with the reason and the
next one is checked. Prints TUNE_RESULT {...} with the full table (the report shows it).
SPEC=off skips tuning and builds without drafting.
TUNE_BASELINE (a JSON file, or inline JSON) = {"decode": X, "decode_32k": Y, "c4": Z}: the fastest llama.cpp tok/s of the card's
rows; by default read from the cfg["reference"] card's info.json (find_reference.py apply). With it a candidate's margin is its worst row relative to llama.cpp (work* rows -> decode, "long" -> decode_32k, streams4
-> c4; a row without a reference is ignored), depth 0 is scored the same way, and a drafter is used when its margin beats
depth 0's (the 3 % single-stream rule stays): a drafter that wins every single-stream row but is 10 % under depth 0 at 4
streams still beats llama.cpp better than no drafting does. Without it the margin is relative to depth 0 as above.
"""
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STARTER = os.path.dirname(HERE)
# The skills root: LOOM_SKILLS (qualify.sh exports it), else relative to the starter. A kit copied into
# WORKDIR/kit (an applied reference example) is not inside the skills tree, so the relative path fails there.
SKILLS = os.environ.get("LOOM_SKILLS") or os.path.join(STARTER, "..", "..")
SETUP = os.path.join(SKILLS, "loom-setup", "scripts")
BENCH = os.path.join(SKILLS, "loom-benchmark", "scripts")
sys.path.insert(0, BENCH)
from bench_common import DOC_SIZES, doc, stream  # noqa: E402
from shell_command import bash_script  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "search", "description": "Search the web.",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]
WORK = [
    ([{"role": "user", "content": "Write a 300-word essay about the history of the printing press."}], {}),
    ([{"role": "user", "content": "Write a Python class implementing an LRU cache with get and put, with type hints and docstrings."}], {}),
    ([{"role": "user", "content": "Find recent news about fusion energy and summarize it. Use the search tool first."}], {"tools": TOOLS}),
    ([{"role": "user", "content": "Explain step by step why the sum of the first n odd numbers is n squared."}], {}),
    ([{"role": "user", "content": "Add a docstring to each method of this class and return the whole class, otherwise unchanged:\n\n"
       "class Inventory:\n    def __init__(self):\n        self.items = {}\n\n    def add(self, name, count=1):\n"
       "        self.items[name] = self.items.get(name, 0) + count\n        return self.items[name]\n\n"
       "    def remove(self, name, count=1):\n        if self.items.get(name, 0) < count:\n"
       "            raise ValueError(f'not enough {name}')\n        self.items[name] -= count\n"
       "        if self.items[name] == 0:\n            del self.items[name]\n\n    def total(self):\n"
       "        return sum(self.items.values())\n\n    def report(self):\n"
       "        return '\\n'.join(f'{k}: {v}' for k, v in sorted(self.items.items()))\n"}], {}),
]


def sh(*cmd, **kw):
    return subprocess.run([str(c) for c in cmd], check=True, text=True, capture_output=True, **kw)


# One decode request after a long document (TUNE_LONG tokens, at most half the context): each verified token pays its
# decode attention over the whole KV and a drafter attends the context too, so the depth that wins on short text can
# lose at long context (Qwen3-8B DFlash at 32k: depth 7 13.7 tok/s, depth 3 20.1, none 19.5). TUNE_LONG=0 leaves it out.
LONG_CTX = int(os.environ.get("TUNE_LONG", "16000"))


# Rows that must not get slower: the card's headline decode, decode_32k and streams4 are prose, where an n-gram or
# low-acceptance drafter finds few matches and the multi-row verify epoch (depth + 1 tokens per span, 16-token
# matvec padding at 4 streams) costs more than a plain step (Qwen3.5-0.8B: ngram depth 3 decode 223 vs 258 without,
# 32k 125 vs 166, four streams 498 vs 787 tok/s). So every row is scored against depth 0 and a drafter that is
# slower than NO_LOSS x depth 0 on any row is not used, however well it does on the code-edit row that quotes its input.
NO_LOSS = 0.98


def measure(url, model, chat, context=0, rows=1):
    """{row: decode tok/s} of the candidate being served: the WORK requests (work0..), the decode after a long
    document ("long", at most half the context) and, with a multi-row preset, "streams4": the aggregate of `rows`
    concurrent essays (measure_c4)."""
    off = {"chat_template_kwargs": {"enable_thinking": False}} if chat.get("thinking") == "toggle" else {}
    # Warm-up: the first timed request's own shape, untimed (perf.py does the same). A 16-token request left the drafted
    # stages' first compiles inside work0: LFM3-Omni-2B scored 154 tok/s there at every n-gram depth against 169 without
    # drafting, where the same request after a full warm-up decodes 172 (card perf, depth 2) and the tuner dropped drafting.
    stream(url, model, WORK[0][0], 256, extra=off)
    work = [(f"work{i}", m, x) for i, (m, x) in enumerate(WORK)]
    n = max((d for d in DOC_SIZES if d <= min(LONG_CTX, context // 2)), default=0)
    if n >= 4000:
        work.append(("long", [{"role": "user", "content": "Write a long, detailed essay discussing this documentation.\n\n" + doc(n)}], {}))
    res = {}
    for name, messages, extra in work:
        r = stream(url, model, messages, 256, extra={**off, **extra}, timeout=900)
        if r.get("decode_tps"):
            res[name] = r["decode_tps"]
    c4 = measure_c4(url, model, chat, rows)
    if c4:
        res["streams4"] = c4
    return res


def measure_c4(url, model, chat, rows, repeats=3):
    """Median of `repeats` measure_c4_once runs: one 256-token sample on a 0.5B model is a few hundred ms of decode and
    swung +-10 % (Qwen2.5-0.5B: ngram depth 2 scored 1127 vs 1265 without drafting here, 1237 vs 1112 in same-runner
    devlab-perf), enough to drop a drafter that wins every row."""
    v = [measure_c4_once(url, model, chat, rows) for _ in range(repeats)]
    return statistics.median(v) if all(v) else 0.0   # a failing run scores the candidate 0


def measure_c4_once(url, model, chat, rows):
    """Aggregate decode tok/s of `rows` concurrent essay requests (the serving preset's rows, at most 4): a speculative
    span verifies depth + 1 tokens whatever the load, so 4 streams verify 4 x (depth + 1) tokens per step, and past 16
    the int8 matvecs give way to the GEMM (Muse-Glimmer: depth 5 gave 41.6 tok/s on 4 streams, depth 3 74.0, no
    drafting 48.6, while one stream preferred depth 5 by 7 %). Returns 0.0 with a single row."""
    import threading
    n = min(max(int(rows or 1), 1), 4)
    if n < 2:
        return 0.0
    off = {"chat_template_kwargs": {"enable_thinking": False}} if chat.get("thinking") == "toggle" else {}
    res = [None] * n

    def run(i):
        try:
            res[i] = stream(url, model, [{"role": "user", "content": f"[{time.time()} {i}] Write a very long, detailed essay about the history of lighthouses."}],
                            384, extra=off, timeout=900)
        except Exception:  # noqa: BLE001  a failing stream scores 0
            res[i] = None
    ts = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    if any(not r or not r.get("tokens") or r.get("ttft") is None for r in res):
        return 0.0
    span = max(r["total"] - r["ttft"] for r in res)
    return sum(r["tokens"] for r in res) / span if span > 0 else 0.0


def single_rate(rows):
    """The single-stream rate of a measurement: the harmonic mean of its single-stream rows (what tune.py reported
    before the per-row guard)."""
    v = [t for k, t in rows.items() if k != "streams4" and t]
    return statistics.harmonic_mean(v) if v else 0.0


def margin(rows, base_rows):
    """The candidate's worst row relative to depth 0 (min over the rows depth 0 measured; 0.0 when a row is missing)."""
    return min((rows.get(k, 0.0) / t for k, t in base_rows.items() if t), default=0.0)


def load_llama_refs(cfg):
    """{"decode", "decode_32k", "c4"} llama.cpp tok/s: TUNE_BASELINE (a JSON file path or inline JSON), else the best
    llama.cpp backend on the cfg["reference"] card (its info.json: latest session and per-backend baselines), else None."""
    v = os.environ.get("TUNE_BASELINE", "").strip()
    if v:
        d = json.loads(v) if v.startswith("{") else json.load(open(v))
        return {k: float(d[k]) for k in ("decode", "decode_32k", "c4") if d.get(k)} or None
    ref = cfg.get("reference") or {}
    if not ref.get("repo"):
        return None
    sys.path.insert(0, os.path.join(SKILLS, "loom-port-llm"))
    import find_reference
    sha = ref.get("revision") or "main"
    rc = find_reference.jfile(ref["repo"], sha, "llm_config.json") or {}
    if os.path.basename(rc.get("gguf") or "") != os.path.basename(cfg.get("gguf") or "-"):
        return None   # a reference of another model (same architecture): its llama.cpp numbers are not this card's
    perf = (find_reference.jfile(ref["repo"], sha, "info.json") or {}).get("performance") or {}
    sets = [s.get("metrics") or {} for s in (perf.get("sessions") or [])[-1:]]
    sets += [(b.get("session") or {}).get("metrics") or {} for b in (perf.get("baselines") or {}).values()]
    out = {}
    for k, m in (("decode", "decode"), ("decode_32k", "decode_32k"), ("c4", "streams4")):
        vals = [float(s[m]["baseline"]) for s in sets if (s.get(m) or {}).get("baseline")]
        if vals:
            out[k] = max(vals)
    return out or None


def llama_margin(rows, refs):
    """The worst row relative to llama.cpp's rate for that row (work* -> decode, long -> decode_32k, streams4 -> c4)."""
    ref = lambda k: refs.get("c4" if k == "streams4" else "decode_32k" if k == "long" else "decode")
    return min((t / ref(k) for k, t in rows.items() if ref(k)), default=0.0)


def concurrent_ok(url, model, chat):
    """Agentic serving runs several requests at once: ~300-token prompts two at a time 0.5 s apart (both draft right
    after nearly simultaneous prefills) and four together (consistency.py's prompts), twice each, must all complete.
    A candidate that fails is not chosen."""
    import threading
    off = {"chat_template_kwargs": {"enable_thinking": False}} if chat.get("thinking") == "toggle" else {}
    body = doc(1000)
    prompts = [f"Question {i}: summarize part {i} of this text.\n\n" + body[i * 700:(i * 700) + 700 + 157 * i] for i in range(4)]
    for n, gap in ((2, 0.5), (4, 0.0), (4, 0.0), (2, 0.5)):
        res = {}
        def run(i):
            time.sleep(i * gap)
            try:
                r = stream(url, model, [{"role": "user", "content": prompts[i]}], 60, extra=off, timeout=600)
                res[i] = 200 if r.get("tokens") else (r.get("status") or "no tokens")
            except Exception as e:  # noqa: BLE001  (the server died: connection reset)
                res[i] = type(e).__name__
        ts = [threading.Thread(target=run, args=(i,)) for i in range(n)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        if any(v != 200 for v in res.values()) or len(res) < n:
            return False, f"{n} concurrent requests ({gap} s apart): {res}"
    return True, ""


def prior_depths(prior, kind, allowed):
    """(depths to measure first, climb) for this method from a reference's tuned table: its best depth's neighbourhood;
    a method that lost clearly there (under 75 % of the reference's winner) gets one probe at its best depth, and
    climbs only if it does better here (tune() checks)."""
    rows = [r for r in prior or [] if r.get("kind") == kind and r.get("depth") and r.get("decode_tps")]
    if not rows or not allowed:
        return None, True
    top = max(rows, key=lambda r: r["decode_tps"])
    winner = max((r["decode_tps"] for r in prior if r.get("depth") and r.get("decode_tps")), default=0)
    i = min(range(len(allowed)), key=lambda j: abs(allowed[j] - top["depth"]))
    if top["decode_tps"] < 0.75 * winner:
        return [allowed[i]], False
    return [allowed[j] for j in (i - 1, i, i + 1) if 0 <= j < len(allowed)], True


def serve(name, port, mem, card, env):
    subprocess.run(bash_script(os.path.join(SETUP, "stop.sh"), name), env=env, capture_output=True)
    r = subprocess.run(bash_script(os.path.join(SETUP, "serve.sh"), name, port, "amdgpu", mem, card), env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"serve {card}: {(r.stdout + r.stderr)[-400:]}")


def main():
    work, port, mem_gib, model_id = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    T = sys.executable
    cfg_path = os.path.join(work, "cfg.json")
    cfg = json.load(open(cfg_path))
    env = dict(os.environ, LOOM_SERVE_STATE=os.environ.get("LOOM_SERVE_STATE", os.path.join(work, "state")))
    spec = os.environ.get("SPEC", "on").lower()
    kinds = []
    if spec not in ("off", "0", "no", "none"):
        if cfg.get("mtp"):
            kinds.append(("mtp", [1, 2, 3, 4]))
        if (cfg.get("drafter") or {}).get("gguf"):
            kinds.append(("dflash", [3, 5, 7]))
        if (cfg.get("assistant") or {}).get("gguf"):
            kinds.append(("assist", [1, 2, 3, 4]))   # Gemma-4 assistant (gemma4 family)
        kinds.append(("ngram", [1, 2, 3, 5]))   # every model: prompt lookup needs no drafter   # depth 1: LFM3-Omni-2B decode_32k 122 vs 108 at depth 2 (verify tokens pay 32k attention), decode_2k 206 vs 220
    sys.path.insert(0, STARTER)
    import catalog
    if catalog.find(STARTER):   # a catalog package drafts with its MTP head or not at all (catalog.py)
        kinds = [("mtp", [1, 2, 3, 4])] if spec not in ("off", "0", "no", "none") and catalog.meta(catalog.find(STARTER)).get("mtp_variant") else []
    llama_refs = load_llama_refs(cfg)
    table, best, cand_mem, chats = [], ("none", 0, 0.0), {}, {}
    # A category A reference (find_reference.py apply: cfg["reference"]) measured its own drafting table: start each
    # method at the reference's best depth and its neighbours instead of every depth. Every method is still measured
    # (a method the reference ruled out gets its best depth), and the search climbs past the neighbourhood when the
    # best depth sits on its edge. TUNE_PRIOR=0 measures every depth.
    prior = (((cfg.get("reference") or {}).get("prior") or {}).get("drafter") or {}).get("tuned") \
        if os.environ.get("TUNE_PRIOR", "1") != "0" else None
    skipped = {}
    build = [T, os.path.join(STARTER, "build.py"), cfg_path]
    # No --memory-gib: each candidate (and the final package) sizes its own budget for the full native context
    # (a drafter adds KV per position; passing the old budget shrank DFlash packages' context to fit).
    common = ["--tokenizer", os.path.join(work, "tok", "tokenizer.json"), "--model-id", model_id]
    base, base_rows = None, {}
    for kind, depths in kinds:
        pkg = os.path.join(work, f"pkg_tune_{kind}")
        out = sh(*build, pkg, *common, "--draft", kind).stdout
        mem = [l for l in out.split() if l.startswith("--memory_bytes=")][-1].split("=")[1]
        cand_mem[kind] = mem
        card = json.load(open(os.path.join(pkg, "model.json")))
        # DFlash depth is bounded by the drafter's block (block_size - 1 proposals), not the card's current depth
        block = 7
        if kind == "dflash":
            sys.path.insert(0, HERE)
            import gen_dflash
            block = gen_dflash.config(cfg["drafter"]["gguf"])["block_size"] - 1
        chat = chats[kind] = json.load(open(os.path.join(pkg, "llm_config.json")))["chat"]
        allowed = [d for d in depths if d <= block]
        first, climb = prior_depths(prior, kind, allowed)
        # With several rows every span verifies depth + 1 tokens: the deepest depth whose rows x (depth + 1) still fits
        # the 16-token int8 matvec is always a candidate, whatever a (single-stream) reference preferred.
        rows_n = int(card["presets"]["chat"]["execution"].get("rows") or 1)
        fit = 16 // max(min(rows_n, 4), 1) - 1
        if first and fit >= 1 and fit in allowed and fit not in first:
            first = sorted(first + [fit])
        measured = {}
        queue = ([0] if base is None else []) + (first or allowed)
        while queue:
            depth = queue.pop(0)
            card["presets"]["chat"]["execution"]["draft_depth"] = depth
            path = os.path.join(pkg, f"model_d{depth}.json")
            json.dump(card, open(path, "w"), indent=1)
            t0 = time.time()
            try:
                serve("tune", port, mem, path, env)
                rows = measure(f"http://127.0.0.1:{port}", model_id, chat,
                               card["presets"]["chat"]["execution"].get("context_capacity") or 0, rows_n)
                tps = single_rate(rows)
            except Exception as e:  # noqa: BLE001  a failing candidate is reported and skipped
                print(f"tune: {kind} depth {depth}: {e}", file=sys.stderr)
                tps, rows = 0.0, {}
            subprocess.run(bash_script(os.path.join(SETUP, "stop.sh"), "tune"), env=env, capture_output=True)
            row = {"kind": "none" if depth == 0 else kind, "depth": depth, "decode_tps": round(tps, 2),
                   "rows": {k: round(v, 2) for k, v in rows.items()}, "seconds": round(time.time() - t0)}
            if llama_refs:
                row["margin_llama"] = round(llama_margin(rows, llama_refs), 3)   # worst row relative to llama.cpp
            if depth == 0:
                base, base_rows = tps, rows
            else:
                row["margin"] = round(margin(rows, base_rows), 3)   # worst row relative to depth 0
            print("tune:", json.dumps(row), flush=True)
            table.append(row)
            sc = row.get("margin", 0.0)
            if depth and sc > best[2]:
                best = (kind, depth, sc)
            if depth:
                measured[depth] = sc
            # the prior is a starting point: climb while the best depth's neighbours are unmeasured (a method the
            # reference ruled out climbs only when it comes within 10 % of the best candidate so far)
            if not queue and first and measured and (climb or max(measured.values()) >= 0.9 * best[2]):
                i = allowed.index(max(measured, key=measured.get))
                queue = [allowed[j] for j in (i - 1, i + 1) if 0 <= j < len(allowed) and allowed[j] not in measured]
        if first:
            skipped[kind] = [d for d in allowed if d not in measured]
    if base is None:
        base = 0.0
    # Fastest first, each checked for exact output (vs draft_depth 0 of its own package) and under concurrent load;
    # one that fails is reported and skipped.
    import spec_check
    refs = {}

    def exact(kind, depth, pkg, mem):
        url = f"http://127.0.0.1:{port}"
        if kind not in refs:
            card0 = json.load(open(os.path.join(pkg, f"model_d{depth}.json")))
            card0["presets"]["chat"]["execution"]["draft_depth"] = 0
            p0 = os.path.join(pkg, "model_d0.json")
            json.dump(card0, open(p0, "w"), indent=1)
            serve("tune", port, mem, p0, env)
            refs[kind] = os.path.join(pkg, "spec_ref.jsonl")
            spec_check.record(url, model_id, refs[kind])
            subprocess.run(bash_script(os.path.join(SETUP, "stop.sh"), "tune"), env=env, capture_output=True)
        out = os.path.join(pkg, f"spec_d{depth}.jsonl")
        serve("tune", port, mem, os.path.join(pkg, f"model_d{depth}.json"), env)
        spec_check.record(url, model_id, out)
        return spec_check.compare(refs[kind], out) == 0
    # A drafter is used only if no row is slower than NO_LOSS x depth 0 and the single-stream rate gains at least 3 %;
    # candidates rank by their worst row relative to depth 0 (the margin).
    # With TUNE_BASELINE the margin is the worst row relative to llama.cpp and a drafter qualifies when it beats depth 0's.
    if llama_refs:
        floor = next(r["margin_llama"] for r in table if r["kind"] == "none")
        for r in table:
            if r["kind"] != "none":
                r["margin"] = r["margin_llama"]
        ok_margin = lambda r: r["margin"] > floor
    else:
        ok_margin = lambda r: r["margin"] >= NO_LOSS
    ranked = sorted([r for r in table if r["kind"] != "none" and ok_margin(r) and r["decode_tps"] >= 1.03 * base],
                    key=lambda r: -r["margin"])
    # Within NO_LOSS of the best margin (noise on these short requests; a wider band on a ratio near 1 would admit
    # every candidate), the shallower depth first: every drafted token is verified with decode attention over the
    # whole context (Qwen3.6-35B-A3B DFlash: depth 7 vs 3 was 109 vs 105 tok/s short, 31 vs 48 tok/s at 32k).
    if ranked:
        top = ranked[0]["margin"]
        ranked.sort(key=lambda r: (r["margin"] < NO_LOSS * top, r["depth"] if r["margin"] >= NO_LOSS * top else 0, -r["margin"]))
    choice = ("none", 0, base)
    for r in ranked:
        kind, depth = r["kind"], r["depth"]
        pkg = os.path.join(work, f"pkg_tune_{kind}")
        mem = cand_mem[kind]
        try:
            ok, why = (True, "") if exact(kind, depth, pkg, mem) else (False, "drafting changed the output (spec_check)")
            if ok:   # the server exact() left running serves the candidate
                ok, why = concurrent_ok(f"http://127.0.0.1:{port}", model_id, chats[kind])
        except Exception as e:  # noqa: BLE001
            ok, why = False, str(e)[-300:]
        subprocess.run(bash_script(os.path.join(SETUP, "stop.sh"), "tune"), env=env, capture_output=True)
        r["checked"] = "ok (exact, concurrent)" if ok else f"FAILED: {why}"
        print("tune:", json.dumps({"kind": kind, "depth": depth, "checked": r["checked"]}), flush=True)
        if ok:
            choice = (kind, depth, r["decode_tps"])
            break
    cfg = json.load(open(cfg_path))
    cfg.setdefault("drafter", {})
    cfg["drafter"].update(kind=choice[0], depth=choice[1], tuned=table)
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    args = ["--draft", choice[0] if choice[0] != "none" else "none"] + (["--draft-depth", str(choice[1])] if choice[1] else [])
    out = sh(*build, os.path.join(work, "pkg"), *common, *args).stdout
    final_mem = [l for l in out.split() if l.startswith("--memory_bytes=")][-1].split("=")[1]
    for kind, _ in kinds:
        shutil.rmtree(os.path.join(work, f"pkg_tune_{kind}"), ignore_errors=True)
    print("TUNE_RESULT", json.dumps({"kind": choice[0], "depth": choice[1], "decode_tps": round(choice[2], 2),
                                     "baseline_tps": round(base, 2), "memory_bytes": int(final_mem), "table": table,
                                     **({"prior": (cfg.get("reference") or {}).get("repo"), "not_measured": skipped} if prior else {})}))


if __name__ == "__main__":
    main()
