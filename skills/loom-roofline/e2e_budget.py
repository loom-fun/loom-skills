#!/usr/bin/env python3
"""End-to-end time budget for one model: a function of its roofline decode (TG) and prefill (PP)
speeds, not a constant. A 0.8B model and a 30B dense model can't share one wall-clock target.

usage: e2e_budget.py MODEL.gguf [--cfg cfg.json] [--context N] [--json] [--manual-min M]
  prints the budget (minutes) and its breakdown; --json prints it as JSON for scripts.

budget = FIXED_MIN (agent work: install, port, reading, decisions, report; the optimization
         phase's own share, OPT_MIN)
       + measured workload: every token the pipeline must decode and prefill on BOTH engines
         (qualify: picker, quick gates, llama.cpp + Loom baselines; finalize: final perf,
         full gates incl. the full-context probe), at the speed this model reaches here:
         decode at DEC_EFF x its decode ceiling, prefill at PRE_EFF x its prefill ceiling at that
         length (attention makes long prefill slower: the roofline counts it)
       + --manual-min (hand-port allowance for an architecture the starter doesn't cover; 0 if
         the starter or a reference example covers it)
The constants are calibrated on fresh-agent trials (loom-improve/reference/results.md): edit them
there and here together.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roofline as R  # noqa: E402

FIXED_MIN = 22      # agent overhead outside measurements (install, port.sh, decisions, report)
OPT_MIN = 15        # optimization phase (A/B rebuilds, its own measurements)
DEC_EFF = 0.95      # of the bandwidth ceiling, before the per-layer floor (roofline.LAYER_US per layer)
# Prefill efficiency grows with the dense matrix size; MoE stays low (expert gather). Fitted on eval.sh
# (llama.cpp and Loom, 5 models): 0.8B dense 0.13-0.16, ~4B dense 0.17-0.27, 30B dense 0.29-0.36,
# MoE with 2-3B active 0.10-0.18.
PRE_EFF_MOE = 0.13
PRE_EFF_DENSE = lambda active_b: min(0.33, max(0.13, 0.13 + 0.06 * math.log2(max(active_b, 0.8) / 0.8)))
C4_GAIN = 2.5       # 4-stream aggregate vs single-stream decode


def speeds(meta, tensors, cfg):
    geo = R.geometry(meta, cfg)
    geo["attn"] = R.attn_layers(tensors, geo["L"])
    wb, params, head, _ = R.weight_bytes(tensors, geo)
    bw, fl = 216e9, 59e12
    floor_s = geo["L"] * R.LAYER_US * 1e-6          # launches, norms, sync: per decode step, per layer
    dec = lambda c: 1 / ((wb + R.kv_read_bytes(geo, c)) / (bw * DEC_EFF) + floor_s)
    hb = tensors.get("output.weight", tensors.get("token_embd.weight"))[1]
    eff = PRE_EFF_MOE if geo["n_exp"] else PRE_EFF_DENSE(params / 1e9)

    def pre(n):
        f = 2 * (params - hb[0] * hb[1]) + R.attn_flops_per_token(geo, n)
        return fl / f * eff
    return dec, pre


def workload(dec, pre, context):
    """Seconds of pure measurement/gate work, from the token counts the scripts send."""
    longs = [n for n in (1000, 2000, 16000, 32000) if n + 2000 <= context]
    t = {}
    # golden (one engine): 20 short prompts x 64 tokens + one 64-token answer per long document
    golden = 20 * (40 / pre(40) + 64 / dec(600)) + sum(n / pre(n) + 64 / dec(n) for n in longs)
    agent = 6 * (150 / dec(600)) + 3500 / pre(600)
    sampling = 216 * (40 / pre(40) + 1 / dec(40)) + 16 * 16 / dec(40)
    consistency = 12 * (300 / pre(300) + 40 / dec(300))
    perf_short = 300 / dec(40) + 10 * (1500 / pre(1500)) + 2000 / pre(2000) + 256 / dec(2000)
    perf_long = sum(n / pre(n) for n in (8000, 16000, 32000, 32000) if n + 2000 <= context) + 128 / dec(32000)
    c4 = 4 * 400 / (dec(600) * C4_GAIN)
    picker = 2 * (200 / dec(40) + 1000 / pre(1000) + 30)       # two builds, each with a model load
    t["qualify"] = picker + 2 * (golden + agent + sampling) + consistency \
        + 2 * (perf_short + perf_long + c4)
    t["finalize"] = 2 * (2 * perf_short + perf_long + c4) + golden + agent + sampling + consistency
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--cfg")
    ap.add_argument("--context", type=int)
    ap.add_argument("--manual-min", type=float, default=0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(a.cfg)) if a.cfg else {}
    meta, tensors = R.load(a.gguf)
    arch = meta.get("general.architecture", "")
    context = a.context or cfg.get("context_length") or int(meta.get(f"{arch}.context_length", 32768))
    context = min(context, 262144)
    dec, pre = speeds(meta, tensors, cfg)
    w = workload(dec, pre, context)
    meas = sum(w.values()) / 60
    budget = FIXED_MIN + OPT_MIN + meas + a.manual_min
    out = {"budget_min": round(budget), "fixed_min": FIXED_MIN, "optimize_min": OPT_MIN, "manual_min": a.manual_min,
           "measurement_min": {k: round(v / 60, 1) for k, v in w.items()}, "context": context,
           "est_decode_tps": round(dec(600), 1), "est_prefill_2k_tps": round(pre(2000)),
           "est_prefill_full_ctx_tps": round(pre(context))}
    if a.json:
        print(json.dumps(out))
    else:
        print(f"end-to-end budget {out['budget_min']} min = fixed {FIXED_MIN} + optimize {OPT_MIN} + measurements "
              f"{meas:.0f} (qualify {w['qualify'] / 60:.0f}, finalize {w['finalize'] / 60:.0f})"
              + (f" + manual port {a.manual_min:.0f}" if a.manual_min else ""))
        print(f"  at est. decode {out['est_decode_tps']} tok/s, prefill 2k {out['est_prefill_2k_tps']} tok/s, "
              f"prefill at {context} {out['est_prefill_full_ctx_tps']} tok/s")


if __name__ == "__main__":
    main()
