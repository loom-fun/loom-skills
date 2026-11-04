"""Chooses llama.cpp's speculative decoding for the baseline, so Loom is compared drafting vs drafting.

usage: LLAMACPP_DIR=DIR llama_spec_tune.py GGUF PORT CTX OUT.json [--cfg WORKDIR/cfg.json] [--pkg PKG]
Candidates, each served alone (llamacpp_server.sh, 4 slots, unified KV of CTX) and timed on the same decode-heavy
requests Loom's tune.py uses (loom-port-llm/starter/draft/tune.py WORK; greedy, client-side tokens/s; thinking off
when the package's chat toggles it, as tune.py does, so both sides decode the same kind of text):
  none; --spec-type draft-mtp at --spec-draft-n-max 1-4 when the GGUF has a NextN/MTP block (cfg["mtp"]);
  --spec-type draft-dflash with the DFlash drafter Loom found (cfg["drafter"]["gguf"]) at n-max 3, 5, 7;
  --spec-type draft-mtp with the Gemma-4 assistant Loom found (cfg["assistant"]["gguf"]) at n-max 1-4;
  n-gram drafting for every model (Loom's tune.py always tries its prompt-lookup drafter): --spec-type ngram-simple and
  ngram-mod at llama.cpp's defaults (they draft only after a long match, so they cost little on new text).
A candidate the build rejects (old build, unsupported arch) is reported and skipped. The fastest wins if it beats
no drafting by at least 3 % (the rule tune.py applies to Loom). OUT = {"args": "...", "kind", "n_max", "decode_tps",
"baseline_tps", "build", "table"}; qualify.sh / finalize.sh export its args as LLAMA_SPEC, which llamacpp_server.sh
appends for the performance runs. Gates and goldens never use it (references are plain greedy log-probs).
SPEC=off: OUT says {"args": ""} without measuring.
"""
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "loom-port-llm", "starter", "draft"))
sys.path.insert(0, HERE)
from bench_common import stream  # noqa: E402
from tune import WORK  # noqa: E402


def measure(url, off):
    stream(url, None, WORK[0][0], 16, extra=off)
    tps = [r["decode_tps"] for r in (stream(url, None, m, 256, extra={**off, **e}) for m, e in WORK) if r.get("decode_tps")]
    return statistics.mean(tps) if tps else 0.0   # as tune.py: a tool-call-only reply streams no text, so no rate


def main():
    gguf, port, ctx, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    cfg = json.load(open(sys.argv[sys.argv.index("--cfg") + 1])) if "--cfg" in sys.argv else {}
    srv = os.path.join(HERE, "llamacpp_server.sh")
    chat = {}
    if "--pkg" in sys.argv:
        chat = json.load(open(os.path.join(sys.argv[sys.argv.index("--pkg") + 1], "llm_config.json"))).get("chat", {})
    off = {"chat_template_kwargs": {"enable_thinking": False}} if chat.get("thinking") == "toggle" else {}
    cands = [("none", 0, "")]
    if os.environ.get("SPEC", "on").lower() not in ("off", "0", "no", "none"):
        if cfg.get("mtp"):
            cands += [("mtp", n, f"--spec-type draft-mtp --spec-draft-n-max {n}") for n in (1, 2, 3, 4)]
        d = (cfg.get("drafter") or {}).get("gguf")
        if d and os.path.exists(d):
            cands += [("dflash", n, f"--spec-type draft-dflash --spec-draft-model {d} --spec-draft-n-max {n} -ngld 99")
                      for n in (3, 5, 7)]
        asst = (cfg.get("assistant") or {}).get("gguf")
        if asst and os.path.exists(asst):   # Gemma-4 assistants run as llama.cpp's MTP drafter
            cands += [("assist", n, f"--spec-type draft-mtp --spec-draft-model {asst} --spec-draft-n-max {n} -ngld 99") for n in (1, 2, 3, 4)]
        cands += [("ngram-simple", 0, "--spec-type ngram-simple"), ("ngram-mod", 0, "--spec-type ngram-mod")]
    else:
        cands = []
    table, base, best = [], 0.0, ("none", 0, "", 0.0)
    for kind, n, args in cands:
        t0 = time.time()
        r = subprocess.run([srv, gguf, port, "4", ctx, "unified"], env=dict(os.environ, LLAMA_SPEC=args), capture_output=True, text=True)
        tps, err = 0.0, None
        if r.returncode:
            err = (r.stdout + r.stderr).strip().splitlines()[-3:]
        else:
            try:
                tps = measure(f"http://127.0.0.1:{port}", off)
            except Exception as e:  # noqa: BLE001
                err = str(e)
        subprocess.run([srv, "--stop", port], capture_output=True)
        row = {"kind": kind, "n_max": n, "decode_tps": round(tps, 2), "seconds": round(time.time() - t0)}
        if err:
            row["error"] = err
        print("llama_spec:", json.dumps(row), flush=True)
        table.append(row)
        if kind == "none":
            base = tps
        elif tps > best[3]:
            best = (kind, n, args, tps)
    choice = best if best[0] != "none" and best[3] >= 1.03 * base else ("none", 0, "", base)
    res = {"args": choice[2], "kind": choice[0], "n_max": choice[1], "decode_tps": round(choice[3], 2),
           "baseline_tps": round(base, 2), "build": os.environ.get("LLAMACPP_DIR"), "table": table}
    json.dump(res, open(out, "w"), indent=1)
    print("LLAMA_SPEC_RESULT", json.dumps(res))


if __name__ == "__main__":
    main()
