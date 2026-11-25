"""Paired single-stream decode requests. Each round starts a new server on each SDK."""
import hashlib
import json
import os
import sys
from bench_common import doc, long_prompt, server_draft_counters, stream, thinking_extra

url, model, round_id, repeats, thinking = sys.argv[1:]
essay = "Write a detailed 300-word essay about the history of the printing press."
def request(prompt, limit):
    result = stream(url, model, [{"role": "user", "content": prompt}], limit, extra=thinking_extra(thinking), timeout=300)
    if result["status"] != 200:
        raise RuntimeError(result.get("error"))
    return result

request(f"[warm-up {round_id}] " + essay, 300)
cases = [("decode", essay, 300)]
for i in range(int(repeats)):
    tag = f"1700000000.{round_id}{i}"
    cases.append(("decode_2k", f"[{tag} ctx] Write a long, detailed essay discussing this documentation.\n\n" + doc(2000), 256))
    if os.environ["ACCEPT_LONG"] == "1":
        cases.append(("decode_32k", long_prompt("decode_tps_32k", tag), 128))
for metric, prompt, limit in cases:
    before = server_draft_counters(url)
    result = request(prompt, limit)
    after = server_draft_counters(url)
    required = {"draft_accepted_inputs", "decode_tokens"}
    if not required <= before.keys() or not required <= after.keys():
        raise ValueError((metric, before, after))
    delta = {k: v - before.get(k, 0) for k, v in after.items()}
    accepted = delta["draft_accepted_inputs"]
    passes = delta["decode_tokens"] - accepted
    if passes <= 0 or result["tokens"] <= 0:
        raise ValueError((metric, result["tokens"], delta))
    print(json.dumps(dict(metric=metric, prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
                          prompt_tokens=result["prompt_tokens"], completion_tokens=result["tokens"],
                          completion_hash=hashlib.sha256(result["text"].encode()).hexdigest(),
                          round=round_id, counters=delta, accepted_per_pass=accepted / passes)), flush=True)
