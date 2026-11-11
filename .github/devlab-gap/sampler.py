"""Sample GPU sclk/power/temp (hwmon) and CPU freqs every INTERVAL s until killed. usage: sampler.py OUT.csv [interval]"""
import glob, sys, time, os, signal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpumetrics
out = open(sys.argv[1], 'w'); iv = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
hw = [h for h in glob.glob('/sys/class/drm/card*/device/hwmon/hwmon*') if os.path.exists(h + '/freq1_input')][0]
cpus = sorted(glob.glob('/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq'), key=lambda p: int(p.split('cpu')[2].split('/')[0]))
busy = glob.glob('/sys/class/drm/card*/device/gpu_busy_percent')[0]
gm = os.path.join(os.path.dirname(busy), 'gpu_metrics')
def rd(p):
    try: return open(p).read().strip()
    except Exception: return ''
pw = hw + '/power1_input' if os.path.exists(hw + '/power1_input') else hw + '/power1_average'
out.write('t,sclk_mhz,power_w,temp_c,busy,cpu_max_mhz,cpu_mean_mhz,' + ','.join(gpumetrics.FIELDS) + '\n')
stop = False
def h(*a):
    global stop; stop = True
signal.signal(signal.SIGTERM, h); signal.signal(signal.SIGINT, h)
while not stop:
    f = [int(rd(c) or 0) / 1000 for c in cpus]
    out.write(f"{time.time():.4f},{int(rd(hw+'/freq1_input') or 0)/1e6:.0f},{int(rd(pw) or 0)/1e6:.1f},{int(rd(hw+'/temp1_input') or 0)/1000:.0f},{rd(busy)},{max(f):.0f},{sum(f)/len(f):.0f}," + ','.join(str(v) for v in gpumetrics.read(gm).values()) + "\n")
    time.sleep(iv)
out.close()
