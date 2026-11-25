#!/usr/bin/env python3
"""Release smoke of published cards on one DevLab runner (.github/workflows/devlab-smoke.yml): each card's own Quick start for
the job's GPU, run line by line as a user copies it, then one chat, one tool call (image cards: the card's own image request).

    devlab_smoke.py SMOKE.json JOB_NAME WORK_DIR OUT_DIR

SMOKE.json is written by fleet/smoke_plan.py: {"jobs": [{"name", "device", "os": "linux" | "windows", "runs_on", "stock_memlock",
"cards": [{"card": "ORG/REPO", "slug", "id" (result directory), "revision": SHA, "block": SUMMARY | null, "kind": "llm" | "image", "tools", "thinking",
"reasoning_effort", "ready_timeout"?}], "hook"?, "env"?}]}.

Per card (OUT_DIR/<id>/result.json, logs next to it):
  1. README.md at the exact revision; the Quick start block whose <summary> is "block" (or the only block). Checks: the block has
     exactly an install, a fetch and a serve line; the pin of the install line equals the package's (llm_config.json sdk.hrx_loom);
     no `--parameter_mode=preload`; the fetch line names a preset of this repo.
  2. The lines, verbatim, in a fresh shell each (Linux: bash; Windows: Windows PowerShell), with an isolated pipx (PIPX_HOME and
     PIPX_BIN_DIR under WORK_DIR, its bin first on PATH). Two edits, both recorded: the fetch selector gets `@REVISION` (the
     revision under test, not whatever main is by then), and the serve line gets `--listen` on a free port only when 8080 is taken.
     stock_memlock (Linux): every line runs after `ulimit -l 8192` (8 MiB, a stock Ubuntu host; DevLab runners raise it), as the
     runner's own non-root user; the serve process's limit is read back from /proc.
  3. GET /v1/status every 10 s until every model is ready (logged on each change), a registration failure, the server exiting, or
     ready_timeout (default 1200 s).
  4. One chat (thinking off where the card has it, the lowest reasoning effort where it has levels): the reply names Paris. One
     tool call where the card claims tool calling: get_weather with Paris. The model id of the card's own request example equals
     the served id. Image cards: the card's own /v1/images/generations body, a PNG comes back.
  5. Stop the server, delete the fetched directory (the Hub cache stays).
Then the job's hook, if any (a .py with this Python, else bash; env SMOKE_CARD SMOKE_REVISION SMOKE_DEVICE SMOKE_OUT SMOKE_WORK + the job's env). OUT_DIR/summary.json
holds every check; the exit status is 1 when any check failed.
"""
import base64
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

HF = "https://huggingface.co"
WINDOWS = os.name == "nt"
STOCK_MEMLOCK_KIB = 8192
PIPX = "pipx==1.7.1"
DEFAULT_PORT = 8080


# ---- pure helpers (unit-tested) ---------------------------------------------------------------------------------------------
def quick_start_block(readme, label):
    """Command lines of the Quick start block whose <summary> is `label` (None: the section's only block)."""
    section = readme.partition("\n## Quick start\n")[2].split("\n## ", 1)[0]
    if not section:
        raise ValueError("the card has no Quick start section")
    details = re.findall(r"<details>\s*<summary>(.*?)</summary>(.*?)</details>", section, re.S)
    if label is not None:
        hit = [b for s, b in details if s.replace("&quot;", '"').replace("&amp;", "&") == label]
        if not hit:
            raise ValueError(f"no Quick start block {label!r} (blocks: {[s for s, _ in details]})")
        section = hit[0]
    elif details:
        raise ValueError(f"the card has {len(details)} Quick start blocks; the job names none")
    blocks = re.findall(r"^```(?:bash|sh)\n(.*?)^```$", section, re.M | re.S)
    if not blocks:
        raise ValueError("the Quick start has no bash block")
    return [ln.strip() for ln in blocks[0].splitlines() if ln.strip() and not ln.strip().startswith("#")]


def classify(lines):
    """{'install', 'fetch', 'serve': line} and the lines that are none of them."""
    kinds, other = {}, []
    for ln in lines:
        k = ("install" if re.match(r"(pipx|pip|uv) ", ln) else "fetch" if ln.startswith("loom-fetch ")
             else "serve" if ln.startswith("loom-serve ") else None)
        if k and k not in kinds:
            kinds[k] = ln
        else:
            other.append(ln)
    return kinds, other


def install_pin(line):
    m = re.search(r"hrx-loom(?:\[[^\]]*\])?==([0-9][^\"' ]*)", line or "")
    return m.group(1) if m else None


def fetch_selector(line, repo):
    """(preset, revision) of the line's huggingface:REPO[:PRESET][@REV] selector, else None."""
    m = re.search(r"huggingface:" + re.escape(repo) + r"(?::([\w.\-]+))?(?:@([\w.\-]+))?(?=\s|$)", line or "")
    return (m.group(1), m.group(2)) if m else None


def pin_fetch(line, repo, revision):
    """The fetch line with @REVISION on the repo's selector (a selector already pinned is left as printed)."""
    def sub(m):
        return m.group(0) if m.group(2) else m.group(0) + "@" + revision
    return re.sub(r"huggingface:" + re.escape(repo) + r"(:[\w.\-]+)?(@[\w.\-]+)?(?=\s|$)", sub, line, count=1)


def listen_of(line):
    m = re.search(r"--listen=([\w.\-]+):(\d+)", line)
    return (m.group(1), int(m.group(2))) if m else ("127.0.0.1", DEFAULT_PORT)


def with_listen(line, port):
    if "--listen=" in line:
        return re.sub(r"--listen=\S+", f"--listen=127.0.0.1:{port}", line)
    return f"{line} --listen=127.0.0.1:{port}"


def request_model_id(readme):
    """The "model" of the card's first request example (Send a request), else None."""
    sec = readme.partition("\n## Send a request\n")[2].split("\n## ", 1)[0]
    m = re.search(r'"model":\s*"([^"]+)"', sec)
    return m.group(1) if m else None


def image_request(readme):
    m = re.search(r"/v1/images/generations.*?-d '(\{.*?\})'", readme, re.S)
    return json.loads(m.group(1)) if m else None


def memlock_of(limits_text):
    """(soft, hard) of 'Max locked memory' in /proc/PID/limits, in bytes ('unlimited' -> -1)."""
    for ln in limits_text.splitlines():
        if ln.startswith("Max locked memory"):
            parts = ln.split()
            vals = [(-1 if v == "unlimited" else int(v)) for v in parts[3:5]]
            return tuple(vals)
    return None


def chat_body(model, card, content, extra=None):
    body = {"model": model, "temperature": 0, "max_tokens": 1024, "messages": [{"role": "user", "content": content}]}
    if card.get("thinking"):
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if card.get("reasoning_effort"):
        body["reasoning_effort"] = card["reasoning_effort"][0]
    body.update(extra or {})
    return body


TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}]


def tool_ok(reply):
    calls = ((reply.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
    if not calls:
        return False
    fn = calls[0].get("function") or {}
    return fn.get("name") == "get_weather" and "paris" in str(fn.get("arguments", "")).lower()


def chat_ok(reply):
    msg = (reply.get("choices") or [{}])[0].get("message") or {}
    return "paris" in (msg.get("content") or "").lower()


# ---- running ---------------------------------------------------------------------------------------------------------------
def hf_get(url, timeout=120):
    tok = os.environ.get("HF_TOKEN")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"} if tok else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def http(url, body=None, timeout=600):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:4000]}


def port_free(port):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def shell_argv(line, stock):
    if WINDOWS:
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", line + "; exit $LASTEXITCODE"]
    pre = f"ulimit -l {STOCK_MEMLOCK_KIB} && " if stock else ""
    return ["bash", "-c", pre + "exec " + line]


def run_line(line, log, cwd, env, stock, timeout):
    t = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        f.write(f"> {line}\n")
        f.flush()
        try:
            p = subprocess.run(shell_argv(line, stock), cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = f"timeout after {timeout} s"
    return rc, round(time.time() - t, 1)


def tail(path, n=25):
    try:
        return "\n".join(open(path, encoding="utf-8", errors="replace").read().splitlines()[-n:])
    except OSError:
        return ""


def kill_tree(proc):
    if proc.poll() is not None:
        return
    if WINDOWS:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        pass


def host_facts(work):
    f = {"os": platform.platform(), "python": sys.version.split()[0], "host": socket.gethostname(),
         "disk_free_gb": round(shutil.disk_usage(work).free / 1e9, 1), "hf_token": bool(os.environ.get("HF_TOKEN"))}
    if not WINDOWS:
        f["uid"] = os.getuid()
        f["runner_memlock"] = subprocess.run(["bash", "-c", "ulimit -l"], capture_output=True, text=True).stdout.strip()
        try:
            f["mem_available_gb"] = round(int(re.search(r"MemAvailable:\s+(\d+)", open("/proc/meminfo").read()).group(1)) / 2**20, 1)
        except (OSError, AttributeError):
            pass
    return f


def bootstrap_pipx(work, env):
    """An isolated pipx: a venv with pipx, PIPX_HOME / PIPX_BIN_DIR under WORK, its bin directories first on PATH."""
    tools = os.path.join(work, "pipx-tool")
    subprocess.run([sys.executable, "-m", "venv", tools], check=True)
    tbin = os.path.join(tools, "Scripts" if WINDOWS else "bin")
    subprocess.run([os.path.join(tbin, "python"), "-m", "pip", "install", "-q", PIPX], check=True)
    env["PIPX_HOME"] = os.path.join(work, "pipx")
    env["PIPX_BIN_DIR"] = os.path.join(work, "bin")
    env["PIPX_MAN_DIR"] = os.path.join(work, "man")
    env["PATH"] = os.pathsep.join([env["PIPX_BIN_DIR"], tbin, env.get("PATH", "")])
    env.pop("VIRTUAL_ENV", None)
    return env


def installed_version(env):
    py = os.path.join(env["PIPX_HOME"], "venvs", "hrx-loom", "Scripts" if WINDOWS else "bin", "python")
    r = subprocess.run([py, "-c", "from importlib.metadata import version; print(version('hrx-loom'))"], capture_output=True, text=True)
    return r.stdout.strip() or None


def smoke_card(job, card, work, out, env):
    repo, rev, slug = card["card"], card["revision"], card.get("id") or card["slug"]
    o = os.path.join(out, slug)
    os.makedirs(o, exist_ok=True)
    cwd = os.path.join(work, "cards", slug)
    shutil.rmtree(cwd, ignore_errors=True)
    os.makedirs(cwd)
    stock = bool(job.get("stock_memlock")) and not WINDOWS
    R = {"card": repo, "revision": rev, "block": card.get("block"), "kind": card.get("kind"), "checks": {}, "seconds": {}, "edits": []}

    def check(name, ok, why=""):
        R["checks"][name] = "pass" if ok else ("fail: " + str(why))[:1500]
        print(f"[{slug}] {'PASS' if ok else 'FAIL'} {name} {'' if ok else str(why)[:600]}", flush=True)
        json.dump(R, open(os.path.join(o, "result.json"), "w"), indent=1)
        return ok

    base = f"{HF}/{repo}/resolve/{rev}"
    try:
        readme = hf_get(f"{base}/README.md").decode()
    except OSError as e:
        check("card README at the revision", False, e)
        return R
    open(os.path.join(o, "README.md"), "w", encoding="utf-8").write(readme)
    pin = None
    if card.get("kind") == "llm":
        try:
            pin = (json.loads(hf_get(f"{base}/llm_config.json")).get("sdk") or {}).get("hrx_loom")
        except (OSError, ValueError) as e:
            check("llm_config.json at the revision", False, e)
            return R
    try:
        lines = quick_start_block(readme, card.get("block"))
    except ValueError as e:
        check("Quick start block", False, e)
        return R
    open(os.path.join(o, "quick_start.sh"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    kinds, other = classify(lines)
    check("Quick start: one install, fetch and serve line, nothing else", set(kinds) == {"install", "fetch", "serve"} and not other,
          f"kinds {sorted(kinds)}, other lines {other}")
    if set(kinds) != {"install", "fetch", "serve"}:
        return R
    R["pin_printed"] = install_pin(kinds["install"])
    check("install pin = package pin", R["pin_printed"] and (pin is None or R["pin_printed"] == pin), f"{R['pin_printed']} vs {pin}")
    check("no --parameter_mode=preload", "--parameter_mode=preload" not in " ".join(lines), "preload flag printed")
    sel = fetch_selector(kinds["fetch"], repo)
    check("fetch line names a preset of this repo", bool(sel and sel[0]), kinds["fetch"])

    # 1. install, verbatim
    rc, R["seconds"]["install"] = run_line(kinds["install"], os.path.join(o, "install.log"), cwd, env, stock, 1200)
    if not check("install (Quick start line)", rc == 0, f"exit {rc}: {tail(os.path.join(o, 'install.log'))}"):
        return R
    R["installed"] = installed_version(env)
    check("installed hrx-loom = printed pin", R["installed"] == R["pin_printed"], f"{R['installed']} vs {R['pin_printed']}")
    # 2. fetch, at the revision under test
    fetch = pin_fetch(kinds["fetch"], repo, rev)
    if fetch != kinds["fetch"]:
        R["edits"].append({"line": "fetch", "printed": kinds["fetch"], "run": fetch, "why": "pin the revision under test"})
    rc, R["seconds"]["fetch"] = run_line(fetch, os.path.join(o, "fetch.log"), cwd, env, stock, 3600)
    if not check("fetch (Quick start line)", rc == 0, f"exit {rc}: {tail(os.path.join(o, 'fetch.log'))}"):
        return R
    # 3. serve, verbatim
    serve = kinds["serve"]
    host, port = listen_of(serve)
    if not port_free(port):
        port = free_port()
        serve2 = with_listen(serve, port)
        R["edits"].append({"line": "serve", "printed": serve, "run": serve2, "why": f"port {listen_of(serve)[1]} is taken on this runner"})
        serve = serve2
    url = f"http://127.0.0.1:{port}"
    slog = open(os.path.join(o, "serve.log"), "w", encoding="utf-8", errors="replace")
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WINDOWS else {"start_new_session": True}
    t0 = time.time()
    proc = subprocess.Popen(shell_argv(serve, stock), cwd=cwd, env=env, stdout=slog, stderr=subprocess.STDOUT, **kw)
    try:
        if not WINDOWS:
            time.sleep(2)
            try:
                lim = memlock_of(open(f"/proc/{proc.pid}/limits").read())
                R["serve_memlock"] = lim
                R["serve_uid"] = os.stat(f"/proc/{proc.pid}").st_uid
                if stock:
                    check("stock host: loom-serve runs with an 8 MiB memlock limit, not as root",
                          lim == (STOCK_MEMLOCK_KIB * 1024,) * 2 and R["serve_uid"] != 0, f"memlock {lim}, uid {R['serve_uid']}")
            except OSError as e:
                R["serve_memlock"] = f"unreadable: {e}"
        deadline = t0 + int(card.get("ready_timeout") or 1200)
        model, last, states = None, None, None
        with open(os.path.join(o, "status.log"), "w") as stl:
            while time.time() < deadline and proc.poll() is None:
                try:
                    code, st = http(f"{url}/v1/status", timeout=5)
                    data = (st or {}).get("data") or []
                    states = [((d.get("loom") or {}).get("registration")) for d in data]
                except (OSError, ValueError, AttributeError) as e:
                    states = [f"unreachable ({type(e).__name__})"]
                if states != last:
                    stl.write(f"{round(time.time() - t0, 1)} s {states}\n")
                    stl.flush()
                    print(f"[{slug}] {round(time.time() - t0)} s status {states}", flush=True)
                    last = states
                if states and all(s == "ready" for s in states):
                    model = data[0]["id"]
                    break
                if any(s and "fail" in str(s) for s in states):
                    break
                time.sleep(10)
        R["seconds"]["ready"] = round(time.time() - t0, 1)
        if not check("loom-serve ready (Quick start line)", model is not None,
                     f"exit {proc.poll()}, status {states}, after {R['seconds']['ready']} s: {tail(os.path.join(o, 'serve.log'))}"):
            return R
        R["model"] = model
        ex = request_model_id(readme)
        check("request example model id = served id", ex is None or ex == model, f"card example {ex!r}, served {model!r}")
        if card.get("kind") == "image":
            body = image_request(readme)
            t = time.time()
            code, r = http(f"{url}/v1/images/generations", body, timeout=900)
            R["seconds"]["image"] = round(time.time() - t, 1)
            png = b""
            try:
                png = base64.b64decode(r["data"][0]["b64_json"])
                open(os.path.join(o, "image.png"), "wb").write(png)
            except (KeyError, IndexError, TypeError, ValueError):
                json.dump(r, open(os.path.join(o, "image.json"), "w"), indent=1)
            check("image request (card example): PNG", code == 200 and png[:8] == b"\x89PNG\r\n\x1a\n", f"{code} {json.dumps(r)[:400]}")
        else:
            t = time.time()
            code, r = http(f"{url}/v1/chat/completions", chat_body(model, card, "What is the capital of France? Answer in one word."))
            R["seconds"]["chat"] = round(time.time() - t, 1)
            json.dump(r, open(os.path.join(o, "chat.json"), "w"), indent=1)
            check("chat: names Paris", code == 200 and chat_ok(r), f"{code} {json.dumps(r)[:600]}")
            if card.get("tools"):
                t = time.time()
                code, r = http(f"{url}/v1/chat/completions", chat_body(model, card, "What is the weather in Paris right now? Use the tool.",
                                                                        {"tools": TOOLS}))
                R["seconds"]["tool"] = round(time.time() - t, 1)
                json.dump(r, open(os.path.join(o, "tool.json"), "w"), indent=1)
                check("tool call: get_weather(Paris)", code == 200 and tool_ok(r), f"{code} {json.dumps(r)[:600]}")
            else:
                R["checks"]["tool call"] = "n/a: the card does not claim tool calling"
        try:
            _, st = http(f"{url}/v1/status", timeout=10)
            json.dump(st, open(os.path.join(o, "status_final.json"), "w"), indent=1)
        except OSError:
            pass
    finally:
        kill_tree(proc)
        slog.close()
        json.dump(R, open(os.path.join(o, "result.json"), "w"), indent=1)
        shutil.rmtree(cwd, ignore_errors=True)
    return R


def main(argv=None):
    spec, name, work, out = (argv or sys.argv[1:])[:4]
    job = next(j for j in json.load(open(spec))["jobs"] if j["name"] == name)
    work, out = os.path.abspath(work), os.path.abspath(out)
    os.makedirs(work, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    S = {"job": {k: v for k, v in job.items() if k != "cards"}, "host": host_facts(work), "cards": {}}
    print(json.dumps(S["host"]), flush=True)
    env = bootstrap_pipx(work, dict(os.environ))
    for card in job["cards"]:
        try:
            r = smoke_card(job, card, work, out, env)
        except Exception as e:   # one card's harness error must not hide the others
            r = {"card": card["card"], "revision": card["revision"], "checks": {"harness": f"fail: {type(e).__name__}: {e}"}}
        S["cards"][card.get("id") or card["slug"]] = r
        json.dump(S, open(os.path.join(out, "summary.json"), "w"), indent=1)
    if job.get("hook"):
        c = job["cards"][0]
        henv = dict(env, SMOKE_CARD=c["card"], SMOKE_REVISION=c["revision"], SMOKE_DEVICE=job["device"], SMOKE_OUT=out,
                    SMOKE_WORK=work, **{k: str(v) for k, v in (job.get("env") or {}).items()})
        hook = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", job["hook"])
        with open(os.path.join(out, "hook.log"), "w") as hl:
            try:
                argv = [sys.executable, hook] if hook.endswith(".py") else ["bash", hook]
                rc = subprocess.run(argv, env=henv, stdout=hl, stderr=subprocess.STDOUT, timeout=4 * 3600).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        S["hook"] = {"script": job["hook"], "result": "pass" if rc == 0 else f"fail: exit {rc}: {tail(os.path.join(out, 'hook.log'), 40)}"}
        print(f"hook {job['hook']}: {S['hook']['result'][:800]}", flush=True)
    json.dump(S, open(os.path.join(out, "summary.json"), "w"), indent=1)
    bad = [f"{s}: {k}" for s, r in S["cards"].items() for k, v in r.get("checks", {}).items() if v.startswith("fail")]
    if S.get("hook", {}).get("result", "pass") != "pass":
        bad.append("hook")
    print(f"{len(S['cards'])} cards, {len(bad)} failed checks" + ("".join(f"\n  {b}" for b in bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
