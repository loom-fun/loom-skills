"""Host-overhead probe: N concurrent decode streams while (a) /logz/stream is recorded (epoch records: host-observed
submission-to-readiness duration_ns) and (b) the GPU busy fraction is sampled from sysfs.
usage: host_probe.py URL MODEL N OUTPREFIX"""
import glob, json, os, statistics as st, sys, threading, time
import requests
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "skills", "loom-benchmark", "scripts"))
from bench_common import stream
url, model, n, pre = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
P = "Write a very long, detailed essay about the history of lighthouses."
stop = threading.Event(); raw = []; busy = []
def logz():
    try:
        with requests.get(url + "/logz/stream", headers={"X-Loom-Model": model}, stream=True, timeout=600) as r:
            for line in r.iter_lines():
                if stop.is_set():
                    break
                if line.startswith(b"data:"):
                    raw.append((time.perf_counter(), line[5:].decode()))
    except Exception as e:
        raw.append((0, "ERR " + str(e)))
def sampler():
    fs = glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")
    while fs and not stop.is_set():
        try:
            busy.append(int(open(fs[0]).read()))
        except Exception:
            pass
        time.sleep(0.002)
def run(i, out):
    ev = []
    s = stream(url, model, [{"role": "user", "content": f"[{time.time()} {i}] {P}"}], 300, extra={"chat_template_kwargs": {"enable_thinking": False}}, on_token=ev.append)
    out[i] = (s["tokens"], ev)
def go():
    out = {}
    ts = [threading.Thread(target=run, args=(i, out)) for i in range(n)]
    [t.start() for t in ts]; [t.join() for t in ts]
    return out
go()  # warm-up
threading.Thread(target=logz, daemon=True).start(); threading.Thread(target=sampler, daemon=True).start()
time.sleep(1); mark = len(raw); b0 = len(busy)
out = go()
time.sleep(0.5); stop.set()
start = max(o[1][0] for o in out.values()); end = min(o[1][-1] for o in out.values())
tok = sum(sum(1 for x in o[1] if start <= x <= end) for o in out.values())
wall = end - start
print(f"streams {n}: agg {tok / wall:.1f} tok/s, step {wall * 1000 / (tok / n):.3f} ms/token/stream")
if busy:
    b = busy[b0:]
    print(f"gpu_busy_percent mean {st.mean(b):.1f} over {len(b)} samples")
recs = []
for t, blob in raw[mark:]:
    for ln in blob.split("\n"):
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                recs.append((t, json.loads(ln)))
            except Exception:
                pass
open(pre + "_logz.jsonl", "w").write("\n".join(json.dumps(r) for _, r in recs))
print("logz records", len(recs)); [print("SAMPLE", json.dumps(r)[:400]) for _, r in recs[:6]]
def find(o, k):
    if isinstance(o, dict):
        if k in o: return o[k]
        for v in o.values():
            r = find(v, k)
            if r is not None: return r
ep = [(t, r) for t, r in recs if find(r, "device_epochs") is not None and find(r, "duration_ns") is not None]
if ep:
    d = [find(r, "duration_ns") / 1e6 for _, r in ep]
    ts = [t for t, _ in ep]
    per = [(b - a) * 1000 for a, b in zip(ts, ts[1:])]
    print(f"epochs {len(ep)}: duration submit->ready median {st.median(d):.3f} ms (mean {st.mean(d):.3f})")
    tn = [find(r, "time_ns") for _, r in ep]
    if all(x is not None for x in tn):
        pg = [(b - a) / 1e6 for a, b in zip(tn, tn[1:])]
        print(f"epoch period (server time_ns) median {st.median(pg):.3f} ms, mean {st.mean(pg):.3f}; gap = period - duration = {st.median(pg) - st.median(d):.3f} ms")
