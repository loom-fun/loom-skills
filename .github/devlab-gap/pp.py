"""Prefill-only probe: N requests of ~TOKENS prompt tokens (max_tokens=1, unique nonce), print prompt_tokens/TTFT.
usage: pp.py URL MODEL TOKENS N [OUT.jsonl]"""
import json, sys, time, urllib.request, random
url, model, toks, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
out = open(sys.argv[5], 'a') if len(sys.argv) > 5 else None
words = ("the quick brown fox jumps over a lazy dog while seven wizards quietly hex every jovial bystander near "
         "old stone bridges and river banks under pale autumn skies full of distant thunder").split()
for i in range(n):
    rnd = random.Random(i * 7919 + toks)
    body = f"[{random.random():.12f}] " + " ".join(rnd.choice(words) for _ in range(int(toks * 0.92)))
    req = {"model": model, "messages": [{"role": "user", "content": body + "\nSummarize."}], "max_tokens": 1,
           "temperature": 0, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(req).encode(),
                                                                 {"Content-Type": "application/json"}), timeout=600))
    dt = time.time() - t; pt = r["usage"]["prompt_tokens"]
    row = {"tokens": pt, "s": round(dt, 4), "tps": round(pt / dt, 1), "t_end": t + dt, "t_start": t}
    print(json.dumps(row)); out and out.write(json.dumps(row) + "\n")
