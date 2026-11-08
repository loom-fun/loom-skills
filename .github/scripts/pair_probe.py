"""Temporary: which two-request schedules make a request's first tokens differ from running alone?
For every pair (i, j), every start offset d (j starts d ms after i, and i after j) and REPS repetitions: the first N tokens of
both requests against their alone output. usage: pair_probe.py URL --model M"""
import argparse, concurrent.futures as cf, sys, time
import requests
sys.path.insert(0, __file__.rsplit("/.github", 1)[0] + "/skills/loom-benchmark/scripts")
from bench_common import doc
ap = argparse.ArgumentParser(); ap.add_argument("url"); ap.add_argument("--model", required=True)
ap.add_argument("--reps", type=int, default=4); ap.add_argument("--n", type=int, default=12)
a = ap.parse_args()
body = doc(1000)
prompts = [f"Question {i}: summarize part {i} of this text.\n\n" + body[i * 700:(i * 700) + 700 + 157 * i] for i in range(4)]
def run(p, delay=0.0):
    time.sleep(delay)
    r = requests.post(a.url + "/v1/chat/completions", json={"model": a.model, "messages": [{"role": "user", "content": p}], "max_tokens": a.n, "temperature": 0}).json()
    m = r["choices"][0]["message"]; return (m.get("reasoning_content") or "") + "|" + (m.get("content") or "")
alone = [run(p) for p in prompts]; alone2 = [run(p) for p in prompts]
print("alone repeat identical:", [x == y for x, y in zip(alone, alone2)], flush=True)
tot = 0
for i in range(4):
    for j in range(4):
        if i == j: continue
        for d in (0, 5, 10, 20, 40, 80, 160):
            bad = [0, 0]
            for _ in range(a.reps):
                with cf.ThreadPoolExecutor(2) as ex:
                    fi = ex.submit(run, prompts[i]); fj = ex.submit(run, prompts[j], d / 1000)
                    ri, rj = fi.result(), fj.result()
                bad[0] += ri != alone[i]; bad[1] += rj != alone[j]
            tot += sum(bad)
            if sum(bad): print(f"first {i} (+{d}ms) second {j}: first differs {bad[0]}/{a.reps}, second differs {bad[1]}/{a.reps}", flush=True)
print("total differing:", tot)
