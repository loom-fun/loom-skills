#!/usr/bin/env python3
"""Replay matched Qwen3.6 decode prompts and retain timing and epoch counts."""

import hashlib
import json
from pathlib import Path
import re
import sys

import requests


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "skills/loom-benchmark/scripts"))
from bench_common import doc, long_prompt, stream, thinking_extra


COUNTERS = ("completed_epochs", "decode_tokens", "draft_accepted_inputs",
            "draft_proposed_tokens", "model_duration")
GAUGES = ("pool_capacity", "pool_resident", "pool_reserved",
          "elastic_state_committed_bytes", "elastic_state_reserved_bytes")
METRIC_FIELD = re.compile(r'field="([^"]+)"')


def sample_metrics(url, model):
    response = requests.get(url + "/metrics", headers={"X-Loom-Model": model}, timeout=15)
    response.raise_for_status()
    values = {}
    for line in response.text.splitlines():
        field = METRIC_FIELD.search(line)
        if field and field[1] in COUNTERS + GAUGES:
            values[field[1]] = int(float(line.split()[-1]))
    return values


def request(url, model, content, tokens):
    before = sample_metrics(url, model)
    result = stream(url, model, [{"role": "user", "content": content}], tokens,
                    extra={**thinking_extra("off"), "cache_prompt": False}, timeout=240)
    after = sample_metrics(url, model)
    if result["status"] != 200 or result["tokens"] < tokens // 2 or not result["decode_tps"]:
        raise RuntimeError(f"unexpected decode result: {result}")
    return {
        **{key: result[key] for key in ("decode_tps", "ttft", "total", "tokens", "prompt_tokens", "finish")},
        "prompt_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "output_sha256": hashlib.sha256(result["text"].encode()).hexdigest(),
        "deltas": {key: after[key] - before[key] for key in COUNTERS if key in before and key in after},
        "state_after": {key: after[key] for key in GAUGES if key in after},
    }


def main():
    url, model, session, output = sys.argv[1:]
    warmup = [{"role": "user", "content": "Write a detailed 300-word essay about the history of the printing press."}]
    warm = stream(url, model, warmup, 300, extra=thinking_extra("off"), timeout=120)
    if warm["status"] != 200 or not warm["decode_tps"]:
        raise RuntimeError(f"unexpected warmup result: {warm}")
    samples = []
    for repeat in range(3):
        nonce = f"179547623{session}.06706{repeat:02d}"
        short = f"[{nonce} ctx] Write a long, detailed essay discussing this documentation.\n\n" + doc(2000)
        long = long_prompt("decode_tps_32k", nonce)
        samples.append({"2k": request(url, model, short, 256), "32k": request(url, model, long, 128)})
    result = {"session": int(session), "warmup_decode_tps": warm["decode_tps"], "samples": samples}
    Path(output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
