"""Temporary: which batching situation makes a greedy output differ from alone? Scenarios (consistency.py's prompts):
  C  all prompts at once, max_tokens 1  (prefill packing only; compares the first token's logprob bit for bit)
  S  all prompts at once, 40 tokens, REPS times (the gate's situation)
  M  prompt 0 decodes 120 tokens; once it streams, prompts 1.. arrive (their prefill epoch carries prompt 0's decode row)
usage: stagger_probe.py URL --model MODEL"""
import argparse, concurrent.futures as cf, json, sys, threading, time
import requests
sys.path.insert(0, __file__.rsplit("/.github", 1)[0] + "/skills/loom-benchmark/scripts")
from bench_common import doc

ap = argparse.ArgumentParser(); ap.add_argument("url"); ap.add_argument("--model", required=True); ap.add_argument("--reps", type=int, default=6)
a = ap.parse_args()
body = doc(1000)
prompts = [f"Question {i}: summarize part {i} of this text.\n\n" + body[i * 700:(i * 700) + 700 + 157 * i] for i in range(4)]


LP = [True]


def run(p, n, stream_event=None):
    j = {"model": a.model, "messages": [{"role": "user", "content": p}], "max_tokens": n, "temperature": 0, **({"logprobs": True, "top_logprobs": 2} if LP[0] else {})}
    if stream_event is None:
        r = requests.post(a.url + "/v1/chat/completions", json=j).json()
        if "choices" not in r:
            print("no logprobs:", str(r)[:200], flush=True)
            LP[0] = False
            r = requests.post(a.url + "/v1/chat/completions", json={k: v for k, v in j.items() if "logprobs" not in k}).json()
        c = r["choices"][0]
        lp = [(t["token"], t["logprob"]) for t in ((c.get("logprobs") or {}).get("content") or [])]
        m = c["message"]
        return {"text": (m.get("reasoning_content") or "") + "|" + (m.get("content") or ""), "lp": lp}
    j["stream"] = True
    j.pop("logprobs", None); j.pop("top_logprobs", None)
    text, lp = "", []
    with requests.post(a.url + "/v1/chat/completions", json=j, stream=True) as r:
        for line in r.iter_lines():
            if not line.startswith(b"data: ") or line.endswith(b"[DONE]"):
                continue
            d = json.loads(line[6:])["choices"][0]
            m = d.get("delta") or {}
            text += (m.get("reasoning_content") or "") + "|" + (m.get("content") or "")
            lp += [(t["token"], t["logprob"]) for t in ((d.get("logprobs") or {}).get("content") or [])]
            stream_event.set()
    return {"text": text, "lp": lp}


def cmp(x, y):
    n = min(len(x["lp"]), len(y["lp"]))
    k = next((j for j in range(n) if x["lp"][j][0] != y["lp"][j][0]), None)
    same = [j for j in range(n) if x["lp"][j] == y["lp"][j]]
    first_bit = next((j for j in range(n) if x["lp"][j] != y["lp"][j]), None)
    return f"text {'same' if x['text'] == y['text'] else 'DIFF'}, lp tokens {len(x['lp'])}/{len(y['lp'])}, first token mismatch {k}, first logprob bit difference {first_bit}"


alone1 = [run(p, 40) for p in prompts]
alone2 = [run(p, 40) for p in prompts]
print("logprobs returned:", all(len(x["lp"]) > 0 for x in alone1))
if not LP[0]:   # no logprobs: compare the texts (a token list from the text split into words)
    _run = run
    def run(p, n, stream_event=None):
        r = _run(p, n, stream_event); r["lp"] = [(w, 0.0) for w in r["text"].replace("|", " | ").split(" ")]; return r
    alone1 = [run(p, 40) for p in prompts]; alone2 = [run(p, 40) for p in prompts]
for i in range(4):
    print(f"alone vs alone again, prompt {i}:", cmp(alone1[i], alone2[i]))
alone120 = run(prompts[0], 120, threading.Event())
one = [run(p, 1) for p in prompts]
for rep in range(a.reps):
    with cf.ThreadPoolExecutor(4) as ex:
        t = list(ex.map(lambda p: run(p, 1), prompts))
    for i in range(4):
        print(f"C rep {rep} prompt {i}:", cmp(one[i], t[i]))
for rep in range(a.reps):
    with cf.ThreadPoolExecutor(4) as ex:
        t = list(ex.map(lambda p: run(p, 40), prompts))
    for i in range(4):
        print(f"S rep {rep} prompt {i}:", cmp(alone1[i], t[i]))
for rep in range(a.reps):
    ev = threading.Event()
    with cf.ThreadPoolExecutor(4) as ex:
        fa = ex.submit(run, prompts[0], 120, ev)
        ev.wait(60); time.sleep(0.05 * rep)
        fb = [ex.submit(run, p, 40) for p in prompts[1:]]
        ra, rb = fa.result(), [f.result() for f in fb]
    print(f"M rep {rep} prompt 0 (decoding while others prefill):", cmp(alone120, ra))
    for i in range(1, 4):
        print(f"M rep {rep} prompt {i}:", cmp(alone1[i], rb[i - 1]))


# Group scenarios: does the failure rate depend on how many tokens share the epoch?
def group(name, ps, reps):
    base = [run(p, 40) for p in ps]
    base2 = [run(p, 40) for p in ps]
    print(name, "alone twice identical:", [x["text"] == y["text"] for x, y in zip(base, base2)])
    for rep in range(reps):
        with cf.ThreadPoolExecutor(len(ps)) as ex:
            t = list(ex.map(lambda p: run(p, 40), ps))
        for i in range(len(ps)):
            print(f"{name} rep {rep} prompt {i}:", cmp(base[i], t[i]))

short = [f"Question {i}: summarize part {i} of this text.\n\n" + body[i * 300:(i * 300) + 300] for i in range(4)]
group("G2a", prompts[0:2], 30)
group("G2b", prompts[2:4], 30)
group("G4short", short, 30)
group("G3", prompts[1:4], 30)
