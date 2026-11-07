"""Chooses the speculative-decoding method and depth for a port by measuring decode speed (qualify.sh runs it).

usage: tune.py WORKDIR PORT MEMORY_GIB MODEL_ID (MEMORY_GIB is ignored: build.py sizes each candidate)   (WORKDIR holds cfg.json, tok/tokenizer.json and pkg/; Loom must not be
                                                 serving another model: one model is loaded at a time)
Candidates: no drafting; MTP (the GGUF's NextN block) at depths 1-4; DFlash (cfg["drafter"], found by find_drafter.py)
at depths 3, 5, 7 (capped by its block); a Gemma-4 assistant (cfg["assistant"]) at depths 1-4; n-gram prompt lookup (any model, no weights) at depths 2, 3, 5. Each method
is built once; a depth is only a model.json setting (the width-specialized stages compile on first use). Every
candidate is served alone and timed on decode-heavy requests (short essay, code, a tool-call turn, a reasoning answer,
a code edit that returns most of its input as agents' file edits do; greedy, client-side tokens/s). The fastest candidate
wins if it beats no drafting by at least 3 %; its kind/depth go to cfg["drafter"] and pkg/ is rebuilt with it
(build.py --draft auto reads them). Before it is kept, the fastest candidate must pass a concurrency check and the
spec gate's equivalence check (loom-benchmark spec_check.py: the same requests at the candidate depth and at
draft_depth 0 of the same package, identical); a candidate that fails either is recorded with the reason and the
next one is checked. Prints TUNE_RESULT {...} with the full table (the report shows it).
SPEC=off skips tuning and builds without drafting.
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
from bench_common import stream  # noqa: E402

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


def measure(url, model, chat):
    off = {"chat_template_kwargs": {"enable_thinking": False}} if chat.get("thinking") == "toggle" else {}
    stream(url, model, WORK[0][0], 16, extra=off)                       # warm-up (weights, first compiles)
    tps = []
    for messages, extra in WORK:
        r = stream(url, model, messages, 256, extra={**off, **extra})
        if r.get("decode_tps"):
            tps.append(r["decode_tps"])
    return statistics.mean(tps) if tps else 0.0


def concurrent_ok(url, model, chat):
    """Agentic serving runs several requests at once: ~300-token prompts two at a time 0.5 s apart (both draft right
    after nearly simultaneous prefills) and four together (consistency.py's prompts), twice each, must all complete.
    A candidate that fails is not chosen."""
    import threading
    off = {"chat_template_kwargs": {"enable_thinking": False}} if chat.get("thinking") == "toggle" else {}
    from bench_common import doc
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
    subprocess.run([os.path.join(SETUP, "stop.sh"), name], env=env, capture_output=True)
    r = subprocess.run([os.path.join(SETUP, "serve.sh"), name, str(port), "amdgpu", str(mem), card], env=env, capture_output=True, text=True)
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
        kinds.append(("ngram", [2, 3, 5]))   # every model: prompt lookup needs no drafter
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
    base = None
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
                tps = measure(f"http://127.0.0.1:{port}", model_id, chat)
            except Exception as e:  # noqa: BLE001  a failing candidate is reported and skipped
                print(f"tune: {kind} depth {depth}: {e}", file=sys.stderr)
                tps = 0.0
            subprocess.run([os.path.join(SETUP, "stop.sh"), "tune"], env=env, capture_output=True)
            row = {"kind": "none" if depth == 0 else kind, "depth": depth, "decode_tps": round(tps, 2), "seconds": round(time.time() - t0)}
            print("tune:", json.dumps(row), flush=True)
            table.append(row)
            if depth == 0:
                base = tps
            elif tps > best[2]:
                best = (kind, depth, tps)
            if depth:
                measured[depth] = tps
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
            subprocess.run([os.path.join(SETUP, "stop.sh"), "tune"], env=env, capture_output=True)
        out = os.path.join(pkg, f"spec_d{depth}.jsonl")
        serve("tune", port, mem, os.path.join(pkg, f"model_d{depth}.json"), env)
        spec_check.record(url, model_id, out)
        return spec_check.compare(refs[kind], out) == 0
    ranked = sorted([r for r in table if r["kind"] != "none" and r["decode_tps"] >= 1.03 * base], key=lambda r: -r["decode_tps"])
    # Within 5 % of the fastest (measurement noise on these short requests), the shallower depth first: every drafted
    # token is verified with decode attention over the whole context, so a deep draft costs at long context
    # (Qwen3.6-35B-A3B DFlash: depth 7 vs 3 was 109 vs 105 tok/s short, 31 vs 48 tok/s at 32k).
    if ranked:
        top = ranked[0]["decode_tps"]
        ranked.sort(key=lambda r: (r["decode_tps"] < 0.95 * top, r["depth"] if r["decode_tps"] >= 0.95 * top else 0, -r["decode_tps"]))
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
        subprocess.run([os.path.join(SETUP, "stop.sh"), "tune"], env=env, capture_output=True)
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
