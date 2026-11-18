"""Low-duty fabric-latency probe: every PERIOD ms, time BURST ms of random 8-byte gathers from a 1 GiB array (latency /
memory-level-parallelism bound, so it tracks the data-fabric clock), and log ns per gathered element.
usage: python fclk_probe.py OUT.jsonl [PERIOD_MS] [BURST_MS]"""
import json
import sys
import time

import numpy as np

out = open(sys.argv[1], "w")
period = (int(sys.argv[2]) if len(sys.argv) > 2 else 1000) / 1000
burst = (int(sys.argv[3]) if len(sys.argv) > 3 else 30) / 1000
a = np.ones(2**27)                         # 1 GiB of float64
rng = np.random.default_rng(1)
idx = rng.integers(0, a.size, 2**16)
np.take(a, idx)
while True:
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < burst:
        np.take(a, idx)
        n += idx.size
    dt = time.perf_counter() - t0
    out.write(json.dumps({"t": round(time.time(), 3), "ns_per_elem": round(dt / n * 1e9, 3)}) + "\n")
    out.flush()
    time.sleep(max(0.0, period - dt))
