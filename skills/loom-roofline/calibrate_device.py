#!/usr/bin/env python3
"""Measure the two device constants the roofline needs, on the machine you are on.

BW and LAYER_US cannot be read off a datasheet: the first is the rate a streaming
kernel reaches, the second is the per-layer launch/sync floor inside one decode step.
Both are fitted from one served model over a context sweep, in the units roofline.py
consumes:

    1/tps(ctx) = pass_bytes(model, ctx) / BW + L * LAYER_US

pass_bytes comes from roofline.py itself, so the fitted BW is the number that makes
that code's ceilings correct on this device. Prefill is scored against roofline.py's
own FLOP count, giving a measured lower bound on peak rate.

usage: calibrate_device.py URL MODEL_ID GGUF [--cfg llm_config.json] [--gen 64]
                           [--ctxs 1000,2000,8000,16000,32000] [--out calib.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import roofline as R  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "loom-benchmark", "scripts"))
from bench_common import doc, stream  # noqa: E402

ESSAY = "Write a short essay about the history of the abacus, with three examples."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("model")
    ap.add_argument("gguf")
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--ctxs", default="1000,2000,8000,16000,32000")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ctxs = [int(x) for x in a.ctxs.split(",")]

    cfg = json.load(open(a.cfg)) if a.cfg and os.path.exists(a.cfg) else None
    m = R.model_shape(a.gguf, cfg)
    print(f"model: L={m['geo']['L']} layers, attention layers={m['geo'].get('attn')}, "
          f"params={m['params']/1e9:.2f}B, weights/decode-token={m['wb']/2**20:.1f} MiB")

    rows = []
    prompts = [ESSAY] + [f"[{n} ctx] Summarize this documentation in one sentence.\n\n" + doc(n) for n in ctxs]
    for prompt in prompts:
        for _ in range(2):                     # first request warms allocation and caches
            s = stream(a.url, a.model, [{"role": "user", "content": prompt}], a.gen)
        rows.append((s["prompt_tokens"] + a.gen / 2, s["decode_tps"]))
        print(f"  ctx {s['prompt_tokens']:>6} tok  decode {s['decode_tps']:>7.2f} tok/s", flush=True)

    x = np.array([R.pass_bytes(m, ctx=int(c)) for c, _ in rows])
    y = np.array([1 / t for _, t in rows])
    (inv_bw, floor_s), *_ = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)
    bw = 1 / inv_bw
    print(f"\nBW = {bw/1e9:.1f} GB/s   step floor = {floor_s*1e6:.0f} us   "
          f"LAYER_US = {floor_s*1e6/m['geo']['L']:.1f} us over {m['geo']['L']} layers")
    pred = 1 / (x * inv_bw + floor_s)
    print("   measured/fit: " + " ".join(f"{t:.0f}/{p:.0f}" for (_, t), p in zip(rows, pred)))
    print(f"   the floor is {floor_s*pred[0]*100:.0f}% of the shortest-context step, "
          f"{x[0]/bw/(y[0])*100:.0f}% of it is weights")

    print("prefill (achieved FLOP/s on roofline.py's flop count):")
    best = 0.0
    for n in ctxs:
        s = stream(a.url, a.model, [{"role": "user", "content": f"[{n}] Summarize in one sentence.\n\n" + doc(n)}], 4)
        fl = 2 * (m["params"] - m["head_params"]) + R.attn_flops_per_token(m["geo"], n)
        tps = s["prompt_tokens"] / s["ttft"]
        best = max(best, fl * tps / 1e12)
        print(f"  {s['prompt_tokens']:>6} tok  {tps:>8.0f} tok/s  {fl*tps/1e12:>6.2f} TFLOP/s")
    print(f"  peak f16 WMMA >= {best:.2f} TFLOP/s measured")

    res = {"bw": bw, "layer_us": floor_s * 1e6 / m["geo"]["L"], "floor_s": floor_s,
           "max_prefill_tflops": best, "layers": m["geo"]["L"],
           "points": [[c, t] for c, t in rows], "gguf": os.path.basename(a.gguf)}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
