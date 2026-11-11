"""usage: cpuuse.py PID SECONDS -> per-thread CPU% over the window (top 6) and total."""
import os, sys, time
pid, s = sys.argv[1], float(sys.argv[2]); tk = os.sysconf('SC_CLK_TCK')
def snap():
    d = {}
    for t in os.listdir(f'/proc/{pid}/task'):
        try: st = open(f'/proc/{pid}/task/{t}/stat').read()
        except OSError: continue
        name = st[st.index('(') + 1:st.rindex(')')]; f = st[st.rindex(')') + 2:].split()
        d[t] = (name, int(f[11]) + int(f[12]))
    return d
a = snap(); time.sleep(s); b = snap()
rows = sorted(((b[t][1] - a[t][1]) / tk / s * 100, b[t][0]) for t in b if t in a)[::-1]
print(f"total {sum(r[0] for r in rows):.0f}%  top: " + ", ".join(f"{p:.0f}% {n}" for p, n in rows[:6]))
