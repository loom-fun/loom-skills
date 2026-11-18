"""CPU-side DRAM bandwidth probe: T threads each copy their own 256 MiB numpy buffer N times (numpy releases the GIL
for large copies). Prints one JSON line: GB/s (read+write bytes) for 1 thread and for T threads, best of 3."""
import json
import os
import sys
import threading
import time

import numpy as np

MB = 256 * 2**20


def run(threads, reps=8):
    src = [np.ones(MB // 8) for _ in range(threads)]
    dst = [np.empty_like(s) for s in src]
    for s, d in zip(src, dst):
        np.copyto(d, s)
    best = 0.0
    for _ in range(3):
        bar = threading.Barrier(threads + 1)

        def work(i):
            bar.wait()
            for _ in range(reps):
                np.copyto(dst[i], src[i])

        ts = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
        [t.start() for t in ts]
        bar.wait()
        t0 = time.perf_counter()
        [t.join() for t in ts]
        dt = time.perf_counter() - t0
        best = max(best, 2 * MB * reps * threads / dt / 1e9)
    return round(best, 1)


T = int(sys.argv[1]) if len(sys.argv) > 1 else min(16, os.cpu_count() or 8)
print(json.dumps({"membw_1t_gbs": run(1), f"membw_{T}t_gbs": run(T)}))
