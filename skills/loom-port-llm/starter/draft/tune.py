"""Chooses the speculative-decoding method and depth for a port by measuring decode speed (qualify.sh runs it).

usage: tune.py WORKDIR PORT MEMORY_GIB MODEL_ID   (WORKDIR holds cfg.json, tok/tokenizer.json and pkg/; Loom must not be
                                                 serving another model: one model is loaded at a time)
Candidates: no drafting; MTP (the GGUF's NextN block) at depths 1-4; DFlash (cfg["drafter"], found by find_drafter.py)
at depths 3, 5, 7 (capped by its block). Each method is built once; a depth is only a model.json setting (the
width-specialized stages compile on first use). Every candidate is served alone and timed on decode-heavy requests
(short essay, code, a tool-call turn, a reasoning answer; greedy, client-side tokens/s). The fastest candidate
wins if it beats no drafting by at least 3 %; its kind/depth go to cfg["drafter"] and pkg/ is rebuilt with it
(build.py --draft auto reads them). Prints TUNE_RESULT {...} with the full table (the report shows it).
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
SETUP = os.path.join(STARTER, "..", "..", "loom-setup", "scripts")
BENCH = os.path.join(STARTER, "..", "..", "loom-benchmark", "scripts")
sys.path.insert(0, BENCH)
from bench_common import stream  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "search", "description": "Search the web.",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]
WORK = [
    ([{"role": "user", "content": "Write a 300-word essay about the history of the printing press."}], {}),
    ([{"role": "user", "content": "Write a Python class implementing an LRU cache with get and put, with type hints and docstrings."}], {}),
    ([{"role": "user", "content": "Find recent news about fusion energy and summarize it. Use the search tool first."}], {"tools": TOOLS}),
    ([{"role": "user", "content": "Explain step by step why the sum of the first n odd numbers is n squared."}], {}),
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
    table, best = [], ("none", 0, 0.0)
    build = [T, os.path.join(STARTER, "build.py"), cfg_path]
    common = ["--tokenizer", os.path.join(work, "tok", "tokenizer.json"), "--model-id", model_id, "--memory-gib", mem_gib]
    base = None
    for kind, depths in kinds:
        pkg = os.path.join(work, f"pkg_tune_{kind}")
        out = sh(*build, pkg, *common, "--draft", kind).stdout
        mem = [l for l in out.split() if l.startswith("--memory_bytes=")][-1].split("=")[1]
        card = json.load(open(os.path.join(pkg, "model.json")))
        block = (card["presets"]["chat"]["execution"].get("draft_depth") or 7) if kind == "dflash" else 7
        chat = json.load(open(os.path.join(pkg, "llm_config.json")))["chat"]
        for depth in ([0] if base is None else []) + [d for d in depths if d <= block]:
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
    if base is None:
        base = 0.0
    choice = best if best[0] != "none" and best[2] >= 1.03 * base else ("none", 0, base)
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
                                     "baseline_tps": round(base, 2), "memory_bytes": int(final_mem), "table": table}))


if __name__ == "__main__":
    main()
