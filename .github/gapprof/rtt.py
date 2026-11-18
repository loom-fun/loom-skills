"""Microbenchmarks with no model, same on Windows and Linux (run by gapprof.py when REPO is "rtt"):
  rtt   one tiny kernel (16 workgroups), --batch-size=1 --input-ring-count=1: host-wall time of one submit+complete
        round trip (loom-benchmark's measurement.operation_timing_ns) next to its device time (profile replay)
  bw    SwiGLU over 4M f32 (48 MiB moved per dispatch), --batch-size=64: device time per dispatch -> GB/s
Each under: spin on/off (completion_spin_ns) x busy CPU threads 0/1/4. Windows: ADL PMLog clocks sampled per config.
usage: python rtt.py VENV_BIN OUTDIR"""
import json
import os
import statistics
import subprocess
import sys
import time

BIN, OUT = sys.argv[1], sys.argv[2]
WIN = os.name == "nt"
HERE = os.path.dirname(os.path.abspath(__file__))
TPL = open(os.path.join(HERE, "rtt_template.loom")).read()
LB = os.path.join(BIN, "loom-benchmark" + (".exe" if WIN else ""))


def loom(n, path):
    open(path, "w").write(TPL.replace("NELEM", str(n)).replace("NMAX", str(n - 1)).replace("NGROUPS", str(n // 256)))
    return path


files = {"rtt": loom(4096, os.path.join(OUT, "rtt.loom")), "bw": loom(4194304, os.path.join(OUT, "bw.loom"))}
args = {"rtt": ["--batch-size=1", "--input-ring-count=1"], "bw": ["--batch-size=64"]}
devices = {"spin": "amdgpu://?completion_spin_ns=4294967295", "nospin": "amdgpu"}
res = []
for load in (0, 1, 4):
    ps = [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(load)]
    time.sleep(2)
    for dk, dev in devices.items():
        for kind in ("rtt", "bw"):
            tag = f"{kind}_{dk}_load{load}"
            adl = None
            if WIN:
                adl = subprocess.Popen([sys.executable, os.path.join(HERE, "adl_sampler.py"),
                                        os.path.join(OUT, f"adl_{tag}.jsonl"), "50"])
            out = os.path.join(OUT, f"lb_{tag}.json")
            t0 = time.time()
            p = subprocess.run([LB, files[kind], f"--device={dev}", "--benchmark=@swiglu_bm", "--measure=dispatch_complete",
                                "--profile-final-batch=true", f"--output={out}"] + args[kind],
                               capture_output=True, text=True, timeout=900)
            if adl:
                adl.kill()
            row = {"tag": tag, "kind": kind, "spin": dk, "load": load, "rc": p.returncode, "secs": round(time.time() - t0, 1),
                   "err": (p.stdout + p.stderr)[-600:] if p.returncode else ""}
            try:
                w = json.load(open(out))["work_items"][0]
                m = w.get("measurement", {})
                row["host_ns_p50"] = (m.get("operation_timing_ns") or m.get("timing_ns") or {}).get("p50")
                row["host_ns_p10"] = (m.get("operation_timing_ns") or m.get("timing_ns") or {}).get("p10")
                row["host_ns_p90"] = (m.get("operation_timing_ns") or m.get("timing_ns") or {}).get("p90")
                pr = w.get("profile_replay") or {}
                row["device_ns_p50"] = (((pr.get("dispatch_timing") or {}).get("dispatch_distribution") or {})
                                        .get("duration_ns") or {}).get("p50")
            except Exception as e:
                row["parse_error"] = str(e)
            if WIN:
                try:
                    vals = {}
                    for l in open(os.path.join(OUT, f"adl_{tag}.jsonl")):
                        for k, v in json.loads(l)["a"].get("0", {}).items():
                            vals.setdefault(k, []).append(v)
                    row["adl_median"] = {k: statistics.median(v) for k, v in vals.items() if k in ("1", "2", "3", "34")}
                except Exception as e:
                    row["adl_error"] = str(e)
            print(json.dumps(row), flush=True)
            res.append(row)
    for q in ps:
        q.kill()
json.dump(res, open(os.path.join(OUT, "rtt_results.json"), "w"), indent=1)
