"""Windows-vs-Linux gap profiler for one loom-fun card on one DevLab Strix Halo runner (runs on both OSes).

usage: python gapprof.py REPO OUTDIR [--variants card,nospin,nospin_cpuload] [--profile dec,pf2k] [--repeats 3]

Same card, same hrx-loom wheel (the card's Quick start pin) on both OSes. Per variant: serve, perf.py (decode,
prefill 1k/2k, decode at 2k), while sampling telemetry (loom-serve CPU, per-core load; Linux: DPM clocks
sclk/mclk/fclk/socclk + GPU busy + power; Windows: typeperf GPU engine utilization, adapter memory, CPU perf %).
Then per profile scenario: serve with HAL dispatch timestamps, profile_requests.py, graceful stop, profile_stats.py.
Variants: card = Quick start serve line verbatim; nospin = completion_spin_ns removed; nospin_cpuload = nospin plus
one busy CPU thread in another process (tests whether memory-fabric clocks follow CPU activity).
"""
import argparse
import json
import os
import platform
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("outdir")
ap.add_argument("--variants", default="card,nospin,nospin_cpuload")
ap.add_argument("--profile", default="dec,pf2k")
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--ready-timeout", type=int, default=3600)
a = ap.parse_args()

WIN = os.name == "nt"
OUT = os.path.abspath(a.outdir)
os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.abspath(os.path.join(HERE, "..", "..", "skills", "loom-benchmark", "scripts"))
WORK = os.path.join(os.environ.get("RUNNER_TEMP", OUT), "gapprof-work")
os.makedirs(WORK, exist_ok=True)
URL = "http://127.0.0.1:8080"
R = {"repo": a.repo, "os": "windows" if WIN else "linux", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
     "env": {}, "steps": {}, "variants": {}, "profiles": {}}


def save():
    json.dump(R, open(os.path.join(OUT, "result.json"), "w"), indent=1)


def step(name, ok, **kw):
    R["steps"][name] = dict(ok=ok, **kw)
    print(f"== {name}: {'OK' if ok else 'FAIL'} {json.dumps(kw)[:400]}", flush=True)
    save()


def sh(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=isinstance(cmd, str))
        return (p.stdout + p.stderr).strip()
    except Exception as e:
        return f"ERR {e}"


def pwsh(cmd, timeout=120):
    return sh(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd], timeout)


def tail(path, n=40):
    try:
        return open(path, encoding="utf-8", errors="replace").read().splitlines()[-n:]
    except OSError:
        return []


def http(path, timeout=10):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return json.loads(r.read())


# -- environment -----------------------------------------------------------------------------------------------------
E = R["env"]
E["platform"] = platform.platform()
E["python"] = sys.version.split()[0]
E["cpus"] = os.cpu_count()
if WIN:
    E["gpu"] = pwsh("(Get-CimInstance Win32_VideoController | % { $_.Name + ' / driver ' + $_.DriverVersion + ' / ' + $_.DriverDate }) -join '; '")
    E["cpu"] = pwsh("(Get-CimInstance Win32_Processor).Name")
    E["ram_gb"] = pwsh("[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)")
    E["bios"] = pwsh("$b=Get-CimInstance Win32_BIOS; $b.SMBIOSBIOSVersion + ' ' + $b.ReleaseDate")
    E["board"] = pwsh("$b=Get-CimInstance Win32_BaseBoard; $b.Manufacturer + ' ' + $b.Product")
    E["power_scheme"] = sh("powercfg /getactivescheme")
    E["dedicated_vram"] = pwsh(
        "Get-ItemProperty 'HKLM:\\SYSTEM\\ControlSet001\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0*' "
        "-ErrorAction SilentlyContinue | ? { $_.'HardwareInformation.qwMemorySize' } | "
        "% { $_.DriverDesc + ': ' + [math]::Round($_.'HardwareInformation.qwMemorySize'/1GB,1) + ' GB' }")
    E["hags"] = pwsh("(Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers' -ErrorAction SilentlyContinue).HwSchMode")
    E["timer"] = pwsh("w32tm /query /status 2>$null | Select-Object -First 3 | Out-String")
    E["power_processor"] = sh("powercfg /query SCHEME_CURRENT SUB_PROCESSOR")[:6000]
else:
    E["kernel"] = sh("uname -r")
    E["cpu"] = sh("grep -m1 'model name' /proc/cpuinfo")
    E["ram_gb"] = sh("free -g | awk '/Mem:/{print $2}'")
    E["bios"] = sh("cat /sys/class/dmi/id/bios_version /sys/class/dmi/id/bios_date 2>/dev/null")
    E["board"] = sh("cat /sys/class/dmi/id/board_vendor /sys/class/dmi/id/board_name 2>/dev/null")
    E["governor"] = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null")
    E["platform_profile"] = sh("cat /sys/firmware/acpi/platform_profile 2>/dev/null")
    E["smu_fw"] = sh("cat /sys/kernel/debug/dri/*/amdgpu_firmware_info 2>/dev/null | grep -i smc | head -2")
E["disk_free_gb"] = round(shutil.disk_usage(WORK).free / 1e9, 1)
save()

GPUDEV = None
if not WIN:
    for d in sorted(os.listdir("/sys/class/drm")):
        p = f"/sys/class/drm/{d}/device"
        if re.fullmatch(r"card\d+", d) and os.path.exists(p + "/pp_dpm_sclk"):
            GPUDEV = p
            break
    if GPUDEV:
        for f in ["mem_info_vram_total", "mem_info_gtt_total", "vbios_version", "power_dpm_force_performance_level"]:
            E[f] = sh(f"cat {GPUDEV}/{f} 2>/dev/null")
        for f in ["pp_dpm_sclk", "pp_dpm_mclk", "pp_dpm_fclk", "pp_dpm_socclk"]:
            E[f] = sh(f"cat {GPUDEV}/{f} 2>/dev/null")
save()

# -- card + venv --------------------------------------------------------------------------------------------------------
readme = urllib.request.urlopen(f"https://huggingface.co/{a.repo}/raw/main/README.md", timeout=120).read().decode()
m = re.search(r"## Quick start.*?```(?:bash|sh|shell)?\n(.*?)```", readme, re.S)
lines = [l.strip() for l in m.group(1).splitlines() if l.strip() and not l.strip().startswith("#")]
R["quickstart"] = lines
pin = next(l for l in lines if "hrx-loom" in l)
spec = shlex.split(pin)[-1]
fetch_line = next(l for l in lines if l.startswith("loom-fetch"))
serve_line = next(l for l in lines if l.startswith("loom-serve"))
venv = os.path.join(WORK, "venv")
if not os.path.exists(venv):
    subprocess.run([sys.executable, "-m", "venv", venv], check=True)
BIN = os.path.join(venv, "Scripts" if WIN else "bin")
VPY = os.path.join(BIN, "python.exe" if WIN else "python")
env = dict(os.environ, PATH=BIN + os.pathsep + os.environ["PATH"], VIRTUAL_ENV=venv, PYTHONUTF8="1")
t = time.time()
p = subprocess.run([VPY, "-m", "pip", "install", "-q", spec, "requests", "jinja2", "psutil"], env=env,
                   capture_output=True, text=True)
step("pip", p.returncode == 0, spec=spec, seconds=round(time.time() - t), tail=(p.stdout + p.stderr)[-1500:])
if p.returncode:
    sys.exit(1)
E["hrx_loom"] = sh([VPY, "-m", "pip", "show", "hrx-loom"]).splitlines()[1]
fa = shlex.split(fetch_line)
fa[0] = os.path.join(BIN, "loom-fetch" + (".exe" if WIN else ""))
t = time.time()
with open(os.path.join(OUT, "loom_fetch.log"), "w", encoding="utf-8", errors="replace") as f:
    p = subprocess.run(fa, cwd=WORK, env=env, stdout=f, stderr=subprocess.STDOUT)
step("loom_fetch", p.returncode == 0, seconds=round(time.time() - t), tail=tail(os.path.join(OUT, "loom_fetch.log"), 5))
if p.returncode:
    sys.exit(1)

sys.path.insert(0, subprocess.run([VPY, "-c", "import sysconfig; print(sysconfig.get_paths()['platlib'])"],
                                  capture_output=True, text=True).stdout.strip())
import psutil  # noqa: E402  (from the venv: same interpreter version)

# -- serving -------------------------------------------------------------------------------------------------------------
BASE_ARGS = shlex.split(serve_line)[1:]
PROFILE_FLAGS = ["--device_profiling_mode=dispatch-events,executable-metadata", "--device_profiling_flush_interval_ms=5",
                 "--print_device_statistics"]


def serve_args(variant):
    args = list(BASE_ARGS)
    if variant.startswith("nospin"):
        args = [re.sub(r"^--device=amdgpu://\?completion_spin_ns=\d+$", "--device=amdgpu", x) for x in args]
    return args


class Server:
    def __init__(self, args, log):
        self.log = log
        self.f = open(log, "w", encoding="utf-8", errors="replace")
        exe = os.path.join(BIN, "loom-serve" + (".exe" if WIN else ""))
        self.f.write("> " + " ".join([exe] + args) + "\n")
        self.f.flush()
        kw = {"creationflags": subprocess.CREATE_NEW_CONSOLE} if WIN else {}
        self.p = subprocess.Popen([exe] + args, cwd=WORK, env=env, stdout=self.f, stderr=subprocess.STDOUT, **kw)
        self.t0 = time.time()

    def ready(self):
        while time.time() - self.t0 < a.ready_timeout:
            if self.p.poll() is not None:
                return None
            try:
                return http("/v1/models")["data"][0]["id"]
            except Exception:
                time.sleep(3)
        return None

    def stop(self):
        """Graceful (the HAL prints device statistics at an orderly shutdown); -> how it stopped."""
        how = None
        if self.p.poll() is None:
            if WIN:
                helper = os.path.join(HERE, "ctrlc.py")
                for ev, name in ((0, "ctrl_c"), (1, "ctrl_break")):
                    subprocess.run([sys.executable, helper, str(self.p.pid), str(ev)], timeout=30)
                    try:
                        self.p.wait(90)
                        how = name
                        break
                    except subprocess.TimeoutExpired:
                        pass
                if how is None:
                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.p.pid)], capture_output=True)
                    how = "taskkill"
            else:
                self.p.send_signal(signal.SIGTERM)
                try:
                    self.p.wait(90)
                    how = "sigterm"
                except subprocess.TimeoutExpired:
                    self.p.kill()
                    how = "sigkill"
        try:
            self.p.wait(30)
        except subprocess.TimeoutExpired:
            pass
        self.f.close()
        time.sleep(5)
        return how


# -- telemetry -----------------------------------------------------------------------------------------------------------
def dpm_cur(name):
    try:
        for l in open(f"{GPUDEV}/{name}"):
            if l.strip().endswith("*"):
                return int(re.search(r"(\d+)\s*[Mm]hz", l).group(1))
    except Exception:
        return None


def hwmon(name):
    try:
        d = os.path.join(GPUDEV, "hwmon")
        return int(open(os.path.join(d, os.listdir(d)[0], name)).read())
    except Exception:
        return None


class Telemetry:
    def __init__(self, pid, path):
        self.pid, self.path, self.rows, self.stop_ev = pid, path, [], threading.Event()
        self.tp = None
        if WIN:
            self.tpcsv = path.replace(".jsonl", "_typeperf.csv")
            counters = [r"\GPU Engine(*)\Utilization Percentage", r"\GPU Adapter Memory(*)\Dedicated Usage",
                        r"\GPU Adapter Memory(*)\Shared Usage", r"\Processor Information(_Total)\% Processor Performance",
                        r"\Processor Information(_Total)\% Processor Utility", r"\Processor Information(_Total)\Processor Frequency"]
            self.tp = subprocess.Popen(["typeperf"] + counters + ["-si", "1", "-f", "CSV", "-o", self.tpcsv, "-y"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.th = threading.Thread(target=self.run, daemon=True)
        self.th.start()

    def run(self):
        try:
            proc = psutil.Process(self.pid)
            proc.cpu_percent(None)
        except Exception:
            proc = None
        psutil.cpu_percent(None, percpu=True)
        while not self.stop_ev.wait(0.5):
            r = {"t": round(time.time(), 2)}
            try:
                r["proc_cpu"] = proc.cpu_percent(None) if proc else None
            except Exception:
                r["proc_cpu"] = None
            pc = psutil.cpu_percent(None, percpu=True)
            r["cpu_max_core"], r["cpu_total"] = max(pc), round(sum(pc) / len(pc), 1)
            if GPUDEV:
                for n in ("sclk", "mclk", "fclk", "socclk"):
                    r[n] = dpm_cur("pp_dpm_" + n)
                try:
                    r["gpu_busy"] = int(open(GPUDEV + "/gpu_busy_percent").read())
                except Exception:
                    pass
                r["power_w"] = (hwmon("power1_average") or hwmon("power1_input") or 0) / 1e6
            self.rows.append(r)

    def finish(self, marks):
        self.stop_ev.set()
        self.th.join(5)
        if self.tp:
            self.tp.terminate()
        with open(self.path, "w") as f:
            for r in self.rows:
                f.write(json.dumps(r) + "\n")
        s = {}
        for t0, t1, label in marks:
            rows = [r for r in self.rows if t0 <= r["t"] <= t1]
            d = {}
            for k in ("proc_cpu", "cpu_max_core", "cpu_total", "sclk", "mclk", "fclk", "socclk", "gpu_busy", "power_w"):
                v = [r[k] for r in rows if r.get(k) is not None]
                if v:
                    d[k] = {"median": round(statistics.median(v), 1), "min": min(v), "max": max(v)}
            s[label] = d
        if WIN and os.path.exists(self.tpcsv):
            s["typeperf"] = summarize_typeperf(self.tpcsv, marks)
        return s


def summarize_typeperf(path, marks):
    import csv
    from datetime import datetime
    try:
        rows = list(csv.reader(open(path, encoding="utf-8", errors="replace")))
    except Exception as e:
        return {"error": str(e)}
    if len(rows) < 3:
        return {"error": "no samples"}
    hdr = rows[0]
    groups = {}
    for i, h in enumerate(hdr[1:], 1):
        if "GPU Engine" in h:
            mm = re.search(r"engtype_([A-Za-z0-9 ]+)\)", h)
            g = "gpu_engine_" + (mm.group(1).strip() if mm else "other")
            gk = "max"  # an engine's busy % is per instance: max across instances of a type
        elif "Dedicated Usage" in h:
            g, gk = "dedicated_gb", "sum"
        elif "Shared Usage" in h:
            g, gk = "shared_gb", "sum"
        else:
            g, gk = h.split("\\")[-1], "max"
        groups.setdefault(g, (gk, []))[1].append(i)
    out = {}
    for t0, t1, label in marks:
        d = {}
        sel = []
        for r in rows[1:]:
            try:
                ts = datetime.strptime(r[0], "%m/%d/%Y %H:%M:%S.%f").timestamp()
            except Exception:
                continue
            if t0 <= ts <= t1:
                sel.append(r)
        for g, (gk, idx) in groups.items():
            vals = []
            for r in sel:
                xs = []
                for i in idx:
                    try:
                        xs.append(float(r[i]))
                    except (ValueError, IndexError):
                        pass
                if xs:
                    vals.append(sum(xs) if gk == "sum" else max(xs))
            if vals:
                scale = 1e-9 if g.endswith("_gb") else 1
                d[g] = {"median": round(statistics.median(vals) * scale, 2), "max": round(max(vals) * scale, 2)}
        out[label] = d
    return out


def cpuload_proc():
    return subprocess.Popen([sys.executable, "-c", "while True: pass"])


# -- variants: perf.py under telemetry -------------------------------------------------------------------------------
for v in [x for x in a.variants.split(",") if x]:
    srv = Server(serve_args(v), os.path.join(OUT, f"serve_{v}.log"))
    mid = srv.ready()
    if not mid:
        step(f"serve_{v}", False, exited=srv.p.poll(), tail=tail(srv.log, 30))
        srv.stop()
        continue
    step(f"serve_{v}", True, seconds=round(time.time() - srv.t0), model=mid)
    load = cpuload_proc() if v.endswith("cpuload") else None
    tel = Telemetry(srv.p.pid, os.path.join(OUT, f"telemetry_{v}.jsonl"))
    time.sleep(3)
    idle = (time.time() - 2.5, time.time(), "idle")
    t0 = time.time()
    with open(os.path.join(OUT, f"perf_{v}.log"), "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.run([VPY, os.path.join(SCRIPTS, "perf.py"), URL, f"{R['os']}-{v}", "--model", mid, "--no-campaign",
                            "--repeats", str(a.repeats), "--log", os.path.join(OUT, f"perf_log_{v}.jsonl")],
                           cwd=SCRIPTS, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=5400)
    t1 = time.time()
    tsum = tel.finish([idle, (t0, t1, "perf")])
    if load:
        load.kill()
    js = [l for l in tail(os.path.join(OUT, f"perf_{v}.log"), 400) if l.startswith('{"')]
    o = json.loads(js[-1]) if js else {}
    R["variants"][v] = {"args": serve_args(v), "perf": {
        "decode": o.get("decode_tps"), "decode_2k": o.get("decode_tps_2k"),
        **{("prefill_1k" if e["prompt_tokens"] < 1500 else "prefill_2k"): e["prefill_tps"] for e in o.get("prefill", [])}},
        "acceptance": o.get("acceptance"), "telemetry": tsum, "stopped": srv.stop()}
    step(f"perf_{v}", p.returncode == 0 and bool(o), perf=R["variants"][v]["perf"])

# -- profiles: per-kernel device time ----------------------------------------------------------------------------------
for sc in [x for x in a.profile.split(",") if x]:
    srv = Server(serve_args("card") + PROFILE_FLAGS, os.path.join(OUT, f"profile_{sc}.serve.log"))
    mid = srv.ready()
    if not mid:
        step(f"profile_{sc}", False, tail=tail(srv.log, 30))
        srv.stop()
        continue
    tel = Telemetry(srv.p.pid, os.path.join(OUT, f"telemetry_profile_{sc}.jsonl"))
    t0 = time.time()
    req = os.path.join(OUT, f"profile_{sc}.req.json")
    with open(req, "w") as f:
        p = subprocess.run([VPY, os.path.join(SCRIPTS, "profile_requests.py"), URL, mid, sc], cwd=SCRIPTS, env=env,
                           stdout=f, stderr=subprocess.PIPE, text=True, timeout=3600)
    t1 = time.time()
    tsum = tel.finish([(t0, t1, "requests")])
    how = srv.stop()
    stats_present = any("dispatch" in l and "count=" in l for l in tail(srv.log, 100000))
    ps = subprocess.run([VPY, os.path.join(SCRIPTS, "profile_stats.py"), srv.log, req, "--families",
                         os.path.join(SCRIPTS, "profile_families.json"), "--md", os.path.join(OUT, f"profile_{sc}.md"),
                         "--json", os.path.join(OUT, f"profile_{sc}.json")], cwd=SCRIPTS, env=env,
                        capture_output=True, text=True)
    R["profiles"][sc] = {"requests_rc": p.returncode, "requests_err": p.stderr[-800:], "wall_s": round(t1 - t0, 2),
                         "stopped": how, "stats_present": stats_present, "stats_rc": ps.returncode,
                         "stats_err": ps.stderr[-800:], "telemetry": tsum}
    step(f"profile_{sc}", p.returncode == 0 and stats_present, stopped=how)

R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
save()
print(json.dumps({k: v["perf"] for k, v in R["variants"].items()}, indent=1))
