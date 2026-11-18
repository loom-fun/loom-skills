#!/usr/bin/env python3
"""The requests of one profiled serve session (devlab_perf.sh step profile:V): the server runs with --print_device_statistics,
so its per-kernel statistics cover exactly this script's requests (plus one tiny warm-up, which loads weights and runs first-use work).

usage: profile_requests.py URL MODEL SCENARIO [--reps K] [--depth D] [--tokens N]
  probe   one tiny request (is the statistics output there at all?)
  pf2k / pf8k   K fresh-nonce prompts of ~2000 / ~8000 tokens, max_tokens 2 (prefill only)
  dec     one essay request (short prompt), N (default 400) tokens: a drafted decode, the counters give its verify steps
Prints one JSON line: scenario, requests (prompt/completion tokens, ttft, decode tok/s), draft counter deltas."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import time

from bench_common import doc, server_draft_counters, stream, thinking_extra

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("model")
ap.add_argument("scenario", choices=["probe", "pf2k", "pf8k", "dec"])
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--tokens", type=int, default=int(os.environ.get("LOOM_PROFILE_TOKENS", "400")))
ap.add_argument("--literal-decode", action="store_true", default=os.environ.get("LOOM_PROFILE_LITERAL_DECODE") == "1",
                help="profile perf.py's exact short-decode prompt without a nonce (diagnostic only)")
ap.add_argument("--perf-warmup", action="store_true", default=os.environ.get("LOOM_PROFILE_PERF_WARMUP") == "1",
                help="use perf.py's nonced 300-token essay warmup")
ap.add_argument("--output-digest", action="store_true", default=os.environ.get("LOOM_PROFILE_OUTPUT_DIGEST") == "1")
ap.add_argument("--process-io", action="store_true", default=os.environ.get("LOOM_PROFILE_PROCESS_IO") == "1")
ap.add_argument("--trace-gaps", action="store_true", default=os.environ.get("LOOM_PROFILE_TRACE_GAPS") == "1")
ap.add_argument("--timeout", type=float, default=float(os.environ.get("LOOM_PROFILE_TIMEOUT", "300")),
                help="per-request connect/read timeout in seconds")
a = ap.parse_args()
if a.perf_warmup and a.scenario != "dec":
    ap.error("--perf-warmup is a decode-only diagnostic")
TH = thinking_extra("off")
nonce = str(time.time())



def process_snapshot():
    """Best-effort counters of the one served process; diagnostic evidence, never a timing filter."""
    snapshots = {}
    state = os.environ.get("LOOM_SERVE_STATE")
    if not a.process_io or not state:
        return snapshots
    for pid_file in Path(state).glob("perf-*.pid"):
        try:
            pid = int(pid_file.read_text().strip())
            proc = Path("/proc") / str(pid)
            counters = {key: int(value) for key, value in
                        (line.split(":", 1) for line in (proc / "io").read_text().splitlines())}
            # The executable name in parentheses may contain spaces.
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            counters.update(minor_faults=int(fields[7]), major_faults=int(fields[9]))
            snapshots[str(pid)] = counters
        except (OSError, ValueError, IndexError):
            continue
    return snapshots


def counter_delta(before, after):
    return {pid: {key: value - before[pid].get(key, 0) for key, value in counters.items()}
            for pid, counters in after.items() if pid in before}


def progress(event, **fields):
    path = os.environ.get("LOOM_PROFILE_PROGRESS_LOG")
    if path:
        with open(path, "a") as out:
            out.write(json.dumps({"event": event, "monotonic_ns": time.monotonic_ns(), **fields}) + "\n")


def go(content, n, phase="timed"):
    progress("request_start", phase=phase, max_tokens=n)
    arrivals = []
    process_before = process_snapshot()
    s = stream(a.url, a.model, [{"role": "user", "content": content}], n, extra=TH, timeout=a.timeout,
               on_token=arrivals.append if a.trace_gaps else None)
    if s["status"] != 200:
        raise SystemExit(f"HTTP {s['status']}: {s.get('error')}")
    result = {"prompt_tokens": s["prompt_tokens"], "completion_tokens": s["tokens"], "ttft_s": round(s["ttft"], 4),
            "decode_tps": s["decode_tps"] and round(s["decode_tps"], 2), "total_s": round(s["total"], 3)}
    if a.output_digest:
        result["output_sha256"] = hashlib.sha256(s["text"].encode("utf-8")).hexdigest()
    if a.process_io:
        result["process_counter_delta"] = counter_delta(process_before, process_snapshot())
    if a.trace_gaps:
        gaps = [round(right - left, 6) for left, right in zip(arrivals, arrivals[1:])]
        result["stream_chunk_arrivals_monotonic_ns"] = [round(value * 1e9) for value in arrivals]
        result["stream_chunk_gaps_s"] = gaps
        result["max_stream_chunk_gap_s"] = max(gaps, default=0)
    progress("request_complete", phase=phase, result=result)
    return result


ESSAY = "Write a detailed 300-word essay about the history of the printing press."
warmup = f"[warm-up {nonce}] " + (ESSAY if a.perf_warmup else "Say hi.")
before_warmup = server_draft_counters(a.url) if a.perf_warmup else {}
out = {"scenario": a.scenario, "warmup_mode": "perf-essay" if a.perf_warmup else "tiny",
       "warmup": go(warmup, 300 if a.perf_warmup else 2, phase="warmup")}
if a.scenario != "probe":
    before = server_draft_counters(a.url)
    if a.perf_warmup:
        out["warmup_draft"] = {k: before.get(k, 0) - before_warmup.get(k, 0) for k in before}
    reqs = []
    if a.scenario in ("pf2k", "pf8k"):
        n = 2000 if a.scenario == "pf2k" else 8000
        for i in range(a.reps):
            reqs.append(go(f"[{nonce} {i}] Summarize this documentation in one sentence.\n\n" + doc(n), 2))
    else:
        content = ESSAY if a.literal_decode else f"[{nonce}] {ESSAY}"
        out["prompt_mode"] = "literal-perf" if a.literal_decode else "nonce"
        reqs.append(go(content, a.tokens))
    after = server_draft_counters(a.url)
    out["requests"] = reqs
    out["draft"] = {k: after.get(k, 0) - before.get(k, 0) for k in after}
print(json.dumps(out), flush=True)
