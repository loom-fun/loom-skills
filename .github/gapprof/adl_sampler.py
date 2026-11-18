"""Windows GPU sensor sampler through the installed AMD driver's ADL library (atiadlxx.dll, ADL2_New_QueryPMLogData_Get):
one JSON line per sample {t, a: {adapter: {sensor_index: value}}} with every supported PMLog sensor, until killed.
usage: python adl_sampler.py OUT.jsonl [INTERVAL_MS]   (--once: print one sample and the adapter list, then exit)"""
import ctypes as C
import json
import sys
import time


class Sensor(C.Structure):
    _fields_ = [("supported", C.c_int), ("value", C.c_int)]


class PMLog(C.Structure):
    _fields_ = [("size", C.c_int), ("sensors", Sensor * 256)]


adl = C.WinDLL("atiadlxx.dll")
_malloc = C.cdll.msvcrt.malloc
_malloc.restype = C.c_void_p
MALLOC = C.WINFUNCTYPE(C.c_void_p, C.c_int)(lambda n: _malloc(n))
ctx = C.c_void_p()
rc = adl.ADL2_Main_Control_Create(MALLOC, 1, C.byref(ctx))
n = C.c_int()
adl.ADL2_Adapter_NumberOfAdapters_Get(ctx, C.byref(n))


def sample():
    out = {}
    for i in range(n.value):
        d = PMLog()
        d.size = C.sizeof(PMLog)
        if adl.ADL2_New_QueryPMLogData_Get(ctx, i, C.byref(d)) == 0:
            vals = {k: d.sensors[k].value for k in range(256) if d.sensors[k].supported}
            if vals:
                out[i] = vals
    return out


if "--once" in sys.argv:
    print(json.dumps({"create_rc": rc, "adapters": n.value, "sample": sample()}))
    sys.exit(0)
ms = int(sys.argv[2]) if len(sys.argv) > 2 else 100
with open(sys.argv[1], "w") as f:
    while True:
        f.write(json.dumps({"t": round(time.time(), 3), "a": sample()}) + "\n")
        f.flush()
        time.sleep(ms / 1000)
