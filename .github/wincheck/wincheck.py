"""Windows red-team check of one loom-fun card: run its Quick start verbatim, then spot-check its Loom claims.

usage: python wincheck.py REPO OUTDIR [--skip-perf] [--ready-timeout S]
  REPO   e.g. loom-fun/unsloth-Qwen3-0.6B-loom

Steps (each recorded in OUTDIR/result.json, with the logs next to it):
  1. Read the card's README from Hugging Face and take the ```bash block under "## Quick start".
  2. Run its pip / loom-fetch lines verbatim in Windows PowerShell, inside a fresh venv (PATH puts the venv first,
     as an activated venv would), and start its loom-serve line verbatim in the background.
  3. Wait for /v1/models, then functional checks for the features the card claims: chat, tool calling, image input.
  4. Spot-check the card's Loom performance rows with the scripts that measured them (loom-benchmark perf.py,
     c4_steady.py): decode short prompt, prefill 1k/2k, decode at 2k, 4 streams. Ratios are Windows / card (Linux).
No llama.cpp is run: only the Loom claims are checked.
"""
import argparse
import base64
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
import zlib

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("outdir")
ap.add_argument("--skip-perf", action="store_true")
ap.add_argument("--ready-timeout", type=int, default=2400)
ap.add_argument("--c4-runs", type=int, default=2)
ap.add_argument("--long", action="store_true", help="also prefill 8k/16k/32k and decode at 32k (perf.py --long)")
a = ap.parse_args()

OUT = os.path.abspath(a.outdir)
os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "..", "skills", "loom-benchmark", "scripts")
WORK = os.path.join(os.environ.get("RUNNER_TEMP", OUT), "wincheck-work")
os.makedirs(WORK, exist_ok=True)
URL = "http://127.0.0.1:8080"
WIN = os.name == "nt"
BIN = "Scripts" if WIN else "bin"
R = {"repo": a.repo, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}, "checks": {}, "perf": {}, "claims": {},
     "ratios": {}, "env": {}}


def save():
    json.dump(R, open(os.path.join(OUT, "result.json"), "w"), indent=1)


def step(name, ok, **kw):
    R["steps"][name] = dict(ok=ok, **kw)
    print(f"== {name}: {'OK' if ok else 'FAIL'} {json.dumps(kw)[:400]}", flush=True)
    save()


def http(path, body=None, timeout=600):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ps(line, log, cwd, env, timeout=None):
    """Run one Quick start line verbatim in Windows PowerShell; -> (rc, seconds)."""
    t = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        f.write(f"> {line}\n")
        f.flush()
        cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", line + "; exit $LASTEXITCODE"] if WIN else ["bash", "-c", line]
        p = subprocess.run(cmd,
                           cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode, round(time.time() - t, 1)


def tail(path, n=40):
    try:
        return open(path, encoding="utf-8", errors="replace").read().splitlines()[-n:]
    except OSError:
        return []


# -- environment -------------------------------------------------------------------------------------------------
R["env"]["python"] = sys.version.split()[0]
R["env"]["platform"] = platform.platform()
try:
  if WIN:
    R["env"]["gpu"] = subprocess.run(["powershell", "-NoProfile", "-Command",
                                      "(Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + ' / driver ' + $_.DriverVersion }) -join '; '"],
                                     capture_output=True, text=True, timeout=60).stdout.strip()
    R["env"]["cpu"] = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                     capture_output=True, text=True, timeout=60).stdout.strip()
    R["env"]["ram_gb"] = subprocess.run(["powershell", "-NoProfile", "-Command",
                                         "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"],
                                        capture_output=True, text=True, timeout=60).stdout.strip()
  else:
    R["env"]["cpu"] = open("/proc/cpuinfo").read().split("model name")[1].split(":")[1].split("\n")[0].strip()
    R["env"]["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    import glob as _g
    R["env"]["vram_total_gb"] = [round(int(open(f).read()) / 2**30, 2) for f in sorted(_g.glob("/sys/class/drm/card*/device/mem_info_vram_total"))]
    ri = subprocess.run(["bash", "-c", "rocminfo | grep -E 'Marketing Name|Name: +gfx' | sort -u"], capture_output=True, text=True, timeout=60).stdout
    R["env"]["gpu"] = " | ".join(l.split(":", 1)[1].strip() for l in ri.splitlines() if ":" in l)
except Exception as e:  # environment info is best effort
    R["env"]["error"] = str(e)
R["env"]["disk_free_gb"] = round(shutil.disk_usage(WORK).free / 1e9, 1)
R["env"]["hf_home"] = os.environ.get("HF_HOME")
save()

# -- 1. the card ---------------------------------------------------------------------------------------------------
readme = urllib.request.urlopen(f"https://huggingface.co/{a.repo}/raw/main/README.md", timeout=120).read().decode("utf-8")
open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(readme)
try:
    info = json.loads(urllib.request.urlopen(f"https://huggingface.co/{a.repo}/raw/main/info.json", timeout=120).read())
except Exception:
    info = {}
m = re.search(r"## Quick start\s*\n\s*```(?:bash|sh|shell)?\n(.*?)```", readme, re.S)
if not m:
    step("quickstart_found", False, error="no ```bash block under ## Quick start")
    sys.exit(1)
lines = [l.strip() for l in m.group(1).splitlines() if l.strip() and not l.strip().startswith("#")]
R["quickstart"] = lines
step("quickstart_found", True, lines=len(lines))

# Card claims (the README table the user reads): Loom column of the performance table.
ROWS = {"decode": "decode tok/s, short prompt", "decode_2k": "decode tok/s, 2k context",
        "decode_32k": "decode tok/s, 32k context",
        "prefill_1k": "prefill tok/s, 1k document", "prefill_2k": "prefill tok/s, 2k document",
        "prefill_8k": "prefill tok/s, 8k document", "prefill_16k": "prefill tok/s, 16k document",
        "prefill_32k": "prefill tok/s, 32k document",
        "streams4": "4 streams, aggregate decode tok/s"}
for k, label in ROWS.items():
    mm = re.search(r"^\| " + re.escape(label) + r" \| [^|]* \| ([^|]*) \|", readme, re.M)
    if mm:
        v = re.match(r"\s*([\d.]+)", mm.group(1))
        R["claims"][k] = float(v.group(1)) if v else mm.group(1).strip()
feat = (info.get("card") or {}).get("features") or {}
R["claimed_features"] = {k: v.get("status") for k, v in feat.items()}
save()

# -- 2. Quick start, verbatim --------------------------------------------------------------------------------------
venv = os.path.join(WORK, "venv")
subprocess.run([sys.executable, "-m", "venv", venv], check=True)
env = dict(os.environ)
env["PATH"] = os.path.join(venv, BIN) + os.pathsep + env["PATH"]
env["VIRTUAL_ENV"] = venv
env["PYTHONUTF8"] = "1"
vpy = os.path.join(venv, BIN, "python.exe" if WIN else "python")
serve_line = None
for i, line in enumerate(lines):
    if line.startswith("loom-serve"):
        serve_line = line
        continue
    name = "pip" if line.startswith("pip") else ("loom_fetch" if line.startswith("loom-fetch") else f"line{i}")
    log = os.path.join(OUT, f"{name}.log")
    rc, secs = ps(line, log, WORK, env, timeout=6 * 3600)
    step(name, rc == 0, rc=rc, seconds=secs, tail=tail(log, 15) if rc else tail(log, 3))
    if rc:
        sys.exit(1)
if not serve_line:
    step("serve_line", False, error="no loom-serve line in the Quick start")
    sys.exit(1)

serve_log = os.path.join(OUT, "loom-serve.log")
slog = open(serve_log, "w", encoding="utf-8", errors="replace")
slog.write(f"> {serve_line}\n")
slog.flush()
t0 = time.time()
srv = subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", serve_line] if WIN else ["bash", "-c", "exec " + serve_line],
                       cwd=WORK, env=env, stdout=slog, stderr=subprocess.STDOUT, start_new_session=not WIN)


def stop_server():
    if WIN:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(srv.pid)], capture_output=True)
    else:
        import signal
        os.killpg(srv.pid, signal.SIGTERM)


model_id = None
R["serve_poll"] = []
last_print = 0
while time.time() - t0 < a.ready_timeout:
    if srv.poll() is not None:
        break
    seen = {}
    for path in ("/v1/models", "/v1/status"):
        try:
            seen[path] = http(path, timeout=10)
        except Exception as e:
            seen[path] = f"{type(e).__name__}: {e}"
    try:
        model_id = seen["/v1/models"]["data"][0]["id"]
        break
    except Exception:
        pass
    if time.time() - last_print > 30:   # what the server says while the model is not listed (it was invisible before)
        last_print = time.time()
        entry = {"s": round(time.time() - t0), **{k: (json.dumps(v)[:1500] if not isinstance(v, str) else v[:300]) for k, v in seen.items()}}
        R["serve_poll"] = (R["serve_poll"] + [entry])[-20:]
        print("== poll", json.dumps(entry)[:2000], flush=True)
        save()
    time.sleep(5)
if not model_id:
    step("serve_ready", False, exited=srv.poll(), seconds=round(time.time() - t0), tail=tail(serve_log, 40))
    stop_server()
    sys.exit(1)
step("serve_ready", True, seconds=round(time.time() - t0), model=model_id)


def vram_used_gb():
    import glob as _g
    try:
        return max(round(int(open(f).read()) / 2**30, 2) for f in _g.glob("/sys/class/drm/card*/device/mem_info_vram_used"))
    except Exception:
        return None


R["env"]["vram_used_after_load_gb"] = vram_used_gb()

# -- 3. functional checks ------------------------------------------------------------------------------------------
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}


def chat(messages, max_tokens=256, **extra):
    body = {"model": model_id, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    body.update(extra)
    return http("/v1/chat/completions", body)


def check(name, fn):
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    R["checks"][name] = {"ok": ok, "detail": detail}
    print(f"== check {name}: {'OK' if ok else 'FAIL'} {str(detail)[:300]}", flush=True)
    save()


def c_chat():
    t = time.time()
    r = chat([{"role": "user", "content": "What is the capital of France? Answer in one word."}], 512, **NOTHINK)
    msg = r["choices"][0]["message"]
    text = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")
    return "paris" in text.lower(), {"content": (msg.get("content") or "")[:200], "s": round(time.time() - t, 1),
                                     "usage": r.get("usage")}


check("chat", c_chat)

if R["claimed_features"].get("tool_calling") == "yes":
    TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

    def c_tools():
        r = chat([{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}], 1024,
                 tools=TOOLS, **NOTHINK)
        msg = r["choices"][0]["message"]
        tc = msg.get("tool_calls") or []
        ok = bool(tc) and tc[0]["function"]["name"] == "get_weather" and \
            "paris" in json.loads(tc[0]["function"]["arguments"] or "{}").get("city", "").lower()
        return ok, {"tool_calls": tc, "content": (msg.get("content") or "")[:200]}

    check("tool_calling", c_tools)


def png(w, h, rgb):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


if R["claimed_features"].get("image_input") == "yes":
    def c_image():
        url = "data:image/png;base64," + base64.b64encode(png(224, 224, (220, 20, 20))).decode()
        r = chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                               {"type": "text", "text": "What color is this image? Answer in one word."}]}],
                 512, **NOTHINK)
        msg = r["choices"][0]["message"]
        text = ((msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")).lower()
        return "red" in text, {"content": (msg.get("content") or "")[:200]}

    check("image_input", c_image)

# -- 4. performance spot-check -------------------------------------------------------------------------------------
if not a.skip_perf and R["claims"]:
    subprocess.run([vpy, "-m", "pip", "install", "-q", "requests", "jinja2"], env=env)
    perf_log = os.path.join(OUT, "perf.log")
    with open(perf_log, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.run([vpy, os.path.join(SCRIPTS, "perf.py"), URL, "windows", "--model", model_id, "--no-campaign",
                            "--repeats", "3", "--log", os.path.join(OUT, "perf_log.jsonl")] + (["--long"] if a.long else []),
                           cwd=SCRIPTS, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=7200)
    js = [l for l in tail(perf_log, 400) if l.startswith("{\"")]
    if p.returncode == 0 and js:
        o = json.loads(js[-1])
        R["perf"]["decode"] = o.get("decode_tps")
        R["perf"]["decode_2k"] = o.get("decode_tps_2k")
        R["perf"]["decode_32k"] = o.get("decode_tps_32k")
        for e in o.get("prefill", []) + (o.get("prefill_long") or []):
            n = e["prompt_tokens"]
            k = "prefill_1k" if n < 1500 else "prefill_2k" if n < 4000 else "prefill_8k" if n < 12000 else \
                "prefill_16k" if n < 24000 else "prefill_32k"
            R["perf"][k] = e["prefill_tps"]
        R["perf"]["perf_py"] = o
        step("perf", True)
    else:
        step("perf", False, rc=p.returncode, tail=tail(perf_log, 30))
    if "streams4" in R["claims"]:
        runs = []
        for i in range(a.c4_runs):
            c4 = subprocess.run([vpy, os.path.join(SCRIPTS, "c4_steady.py"), URL, "4", "--model", model_id],
                                cwd=SCRIPTS, env=env, capture_output=True, text=True, timeout=1800)
            open(os.path.join(OUT, f"c4_{i}.log"), "w", encoding="utf-8").write(c4.stdout + c4.stderr)
            jl = [l for l in c4.stdout.splitlines() if l.startswith("{")]
            if c4.returncode == 0 and jl:
                runs.append(json.loads(jl[-1])["agg_tps"])
        if runs:
            R["perf"]["streams4"] = sorted(runs)[len(runs) // 2]
            R["perf"]["streams4_runs"] = runs
        step("streams4", bool(runs), runs=runs)
    for k, claim in R["claims"].items():
        got = R["perf"].get(k)
        if isinstance(claim, float) and isinstance(got, (int, float)) and claim:
            R["ratios"][k] = round(got / claim, 3)
    save()

R["env"]["vram_used_end_gb"] = vram_used_gb()
stop_server()
R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
R["serve_log_tail"] = tail(serve_log, 25)
fail = [k for k, v in R["steps"].items() if not v["ok"]] + [k for k, v in R["checks"].items() if not v["ok"]]
R["verdict"] = "fail: " + ", ".join(fail) if fail else "pass"
save()
print(json.dumps({"verdict": R["verdict"], "ratios": R["ratios"], "perf": {k: v for k, v in R["perf"].items() if k != "perf_py"}}))
sys.exit(1 if fail else 0)
