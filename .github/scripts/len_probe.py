"""Temporary: first request A of varying prompt length, B (287 tokens) arrives D ms later; A's first N tokens vs alone.
usage: len_probe.py URL --model M"""
import argparse, concurrent.futures as cf, sys, time
import requests
sys.path.insert(0, __file__.rsplit("/.github", 1)[0] + "/skills/loom-benchmark/scripts")
from bench_common import doc
ap = argparse.ArgumentParser(); ap.add_argument("url"); ap.add_argument("--model", required=True)
ap.add_argument("--reps", type=int, default=2); ap.add_argument("--n", type=int, default=12)
a = ap.parse_args()
body = doc(1000)
def run(p, delay=0.0):
    time.sleep(delay)
    r = requests.post(a.url + "/v1/chat/completions", json={"model": a.model, "messages": [{"role": "user", "content": p}], "max_tokens": a.n, "temperature": 0}).json()
    m = r["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "|" + (m.get("content") or ""), r["usage"]["prompt_tokens"]
partner = f"Question 2: summarize part 2 of this text.\n\n" + body[1400:1400 + 700 + 314]
for k in range(0, 40):
    p = f"Question 1: summarize part 1 of this text.\n\n" + body[700:700 + 700 + 9 * k]
    ref, n = run(p)
    bad = 0
    for _ in range(a.reps):
        with cf.ThreadPoolExecutor(2) as ex:
            fa = ex.submit(run, p); fb = ex.submit(run, partner, 0.02)
            ra = fa.result(); fb.result()
        bad += ra[0] != ref
    print(f"len {n} (mod16 {n % 16}): differs {bad}/{a.reps}", flush=True)
