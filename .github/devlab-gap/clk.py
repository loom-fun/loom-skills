"""usage: clk.py samples.csv [label] -> distribution of fabric/memory clocks while GPU is active (avg gfx activity > 60)."""
import csv, sys
r = list(csv.DictReader(open(sys.argv[1])))
if 'average_fclk_frequency' not in r[0]: print('no gpu_metrics'); sys.exit()
b = [x for x in r if int(x['average_gfx_activity']) > 60]
q = lambda a, f: a[min(len(a) - 1, int(len(a) * f))]
def d(k, s=1):
    a = sorted(float(x[k]) / s for x in b); return f"{q(a,.1):.0f}/{q(a,.5):.0f}/{q(a,.9):.0f}"
thr = {k: int(r[-1][k]) - int(r[0][k]) for k in r[0] if k.startswith('throttle_residency')}
print(f"{(sys.argv[2] if len(sys.argv)>2 else ''):14s} n={len(b):5d} gfxclk {d('average_gfxclk_frequency')} fclk {d('average_fclk_frequency')} uclk {d('average_uclk_frequency')} socclk {d('average_socclk_frequency')} "
      f"dramR {d('average_dram_reads')} dramW {d('average_dram_writes')} sockW {d('average_socket_power',1000)} gfxW {d('average_gfx_power',1000)} act {d('average_gfx_activity')} | thr " + " ".join(f"{k[19:]}={v}" for k, v in thr.items() if v))
