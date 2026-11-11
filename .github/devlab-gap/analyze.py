"""usage: analyze.py P.pkl [samples.csv] -> per-submission span/busy/gaps and per-kernel totals for big submissions."""
import pickle, sys, collections, statistics as st
d = pickle.load(open(sys.argv[1], 'rb'))
hz = d['devices'][0][2]; tick = 1e9 / hz
fn = d['functions']
by = collections.defaultdict(list)
for (pdev, q, eid, sub, cb, ex, ci, fo, wc, ws, s, e) in d['dispatch']:
    by[(q, sub)].append((s, e, fn.get((ex, fo), f'{ex}:{fo}'), wc, ws))
subs = []
for k, v in by.items():
    v.sort(); span = (max(e for _, e, *_ in v) - v[0][0]) * tick / 1e6
    busy = sum(e - s for s, e, *_ in v) * tick / 1e6
    gaps = [(v[i + 1][0] - v[i][1]) * tick / 1e3 for i in range(len(v) - 1)]
    subs.append((k, len(v), span, busy, gaps, v))
subs.sort(key=lambda x: x[5][0][0])
print("queue sub  ndisp  span_ms  busy_ms  busy%  gap_sum_ms  gap_med_us  gap_max_us")
for k, n, span, busy, gaps, v in subs:
    if n < 20: continue
    print(f"{k[0]:5d} {k[1]:5d} {n:5d} {span:8.2f} {busy:8.2f} {100*busy/span:5.1f} {sum(gaps)/1e3:10.3f} {st.median(gaps) if gaps else 0:10.2f} {max(gaps) if gaps else 0:10.1f}")
big = [x for x in subs if x[1] >= 200]
if big:
    agg = collections.defaultdict(lambda: [0, 0.0])
    for k, n, span, busy, gaps, v in big:
        for s, e, name, wc, ws in v:
            agg[name][0] += 1; agg[name][1] += (e - s) * tick / 1e6
    tot = sum(a[1] for a in agg.values())
    print(f"\nper kernel over {len(big)} big submissions (total {tot:.1f} ms):")
    for name, (c, t) in sorted(agg.items(), key=lambda x: -x[1][1])[:25]:
        print(f"  {t:8.2f} ms {100*t/tot:5.1f}%  n={c:5d}  avg {1000*t/c:8.1f} us  {name[:90]}")
# inter-submission gaps on queue 0 (host-side time between GPU work)
q0 = [x for x in subs if x[0][0] == subs[0][0][0]]
