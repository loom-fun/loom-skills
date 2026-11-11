"""Duty-cycled CPU load: busy ON_MS then sleep OFF_MS, forever (OFF_MS 0 = pure spin). usage: dutyspin.py ON_MS OFF_MS"""
import sys, time
on, off = float(sys.argv[1]) / 1e3, float(sys.argv[2]) / 1e3
while True:
    t = time.perf_counter() + on
    while time.perf_counter() < t: pass
    if off: time.sleep(off)
