#!/usr/bin/env python3
"""devlab-smoke hook: Lemonade's Loom backend against a published card (release-checklist.md section 4, the Lemonade PR).

Builds lemonade-sdk/lemonade at LEMONADE_REVISION (its Loom contract tests and the embeddable bundle), runs its contract tests,
starts lemond with a private cache and config, registers SMOKE_CARD (pinned to SMOKE_REVISION) as a user model with the loom
recipe, then one chat and one tool call through Lemonade's OpenAI route. Checks: the tests pass, the model registers, the chat names
Paris, the tool call is get_weather(Paris), the backend Lemonade installed reports LOOM_VERSION (default v0.3.0) and the serve line
it launched carries no --parameter_mode=preload. Results: SMOKE_OUT/lemonade/lemonade.json; exit 1 on a failed check.
Env: SMOKE_CARD SMOKE_REVISION SMOKE_OUT SMOKE_WORK (devlab_smoke.py), LEMONADE_REVISION, LOOM_VERSION (job env).
"""
import glob
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import devlab_smoke as D  # noqa: E402

REV = os.environ["LEMONADE_REVISION"]
CARD, CARD_REV = os.environ["SMOKE_CARD"], os.environ.get("SMOKE_REVISION")
OUT = os.path.join(os.environ["SMOKE_OUT"], "lemonade")
WORK = os.path.join(os.environ["SMOKE_WORK"], "lemonade")
WANT = os.environ.get("LOOM_VERSION", "v0.3.0")
R = {"lemonade_revision": REV, "card": CARD, "card_revision": CARD_REV, "checks": {}}


def check(name, ok, why=""):
    R["checks"][name] = "pass" if ok else ("fail: " + str(why))[:2000]
    print(f"{'PASS' if ok else 'FAIL'} {name} {'' if ok else str(why)[:800]}", flush=True)
    json.dump(R, open(os.path.join(OUT, "lemonade.json"), "w"), indent=1)
    return ok


def run(argv, log, cwd, timeout=3600):
    with open(os.path.join(OUT, log), "w") as f:
        p = subprocess.run(argv, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode


def build(src):
    subprocess.run(["git", "init", "-q", src], check=True)
    subprocess.run(["git", "-C", src, "fetch", "-q", "--depth", "1", "https://github.com/lemonade-sdk/lemonade.git", REV], check=True)
    subprocess.run(["git", "-C", src, "-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD"], check=True)
    head = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if not check("Lemonade checkout = LEMONADE_REVISION", head == REV, head):
        return None
    rc = run(["cmake", "--preset", "default", "-DBUILD_WEB_APP=OFF"], "configure.log", src)
    if not check("configure", rc == 0, D.tail(os.path.join(OUT, "configure.log"), 40)):
        return None
    rc = run(["cmake", "--build", "--preset", "default", "--target", "test_loom_contract", "test_install_atomicity", "embeddable", "-j", "8"],
             "build.log", src)
    if not check("build (contract tests + embeddable)", rc == 0, D.tail(os.path.join(OUT, "build.log"), 60)):
        return None
    rc = run(["ctest", "--test-dir", "build", "--output-on-failure", "--no-tests=error", "-R", "^(LoomContractTest|InstallAtomicityTest)$"],
             "ctest.log", src)
    check("ctest LoomContractTest + InstallAtomicityTest", rc == 0, D.tail(os.path.join(OUT, "ctest.log"), 60))
    tars = glob.glob(os.path.join(src, "build", "lemonade-embeddable-*-ubuntu-x64.tar.gz"))
    if not check("one embeddable bundle", len(tars) == 1, tars):
        return None
    app = os.path.join(WORK, "app")
    with tarfile.open(tars[0]) as t:
        t.extractall(app, filter="data")
    found = [os.path.join(d, "lemond") for d, _, fs in os.walk(app) if "lemond" in fs]
    return found[0] if check("lemond in the bundle", bool(found), app) else None


def serve(lemond):
    cache, cfg = os.path.join(WORK, "cache"), os.path.join(WORK, "cfg")
    os.makedirs(cache)
    os.makedirs(cfg)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    log = open(os.path.join(OUT, "lemond.log"), "w")
    proc = subprocess.Popen([lemond, cache, cfg, "--port", str(port), "--host", "127.0.0.1"], stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        for _ in range(240):
            try:
                code, _ = D.http(f"{base}/api/v1/health", timeout=3)
                if code == 200:
                    break
            except OSError:
                time.sleep(0.5)
        code, r = D.http(f"{base}/api/v1/models/register", {"model_name": "user.SmokeLoom", "checkpoint": CARD, "recipe": "loom",
                                                            "source": "huggingface"}, timeout=120)
        R["register"] = r
        R["checkpoint_registered"] = CARD   # Lemonade registers a repo id (its head); the job records the revision the smoke pinned
        if not check("register the card as a user model", code == 200, f"{code} {json.dumps(r)[:500]}"):
            return
        card = {"thinking": True}
        t = time.time()
        code, r = D.http(f"{base}/api/v1/chat/completions", D.chat_body("user.SmokeLoom", card, "What is the capital of France? Answer in one word."),
                         timeout=7200)
        R["chat_s_incl_install_and_load"] = round(time.time() - t, 1)
        R["chat"] = r
        check("chat through Lemonade: names Paris", code == 200 and D.chat_ok(r), f"{code} {json.dumps(r)[:500]}")
        code, r = D.http(f"{base}/api/v1/chat/completions", D.chat_body("user.SmokeLoom", card, "What is the weather in Paris right now? Use the tool.",
                                                                         {"tools": D.TOOLS}), timeout=900)
        R["tool"] = r
        check("tool call through Lemonade: get_weather(Paris)", code == 200 and D.tool_ok(r), f"{code} {json.dumps(r)[:500]}")
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        log.close()
    inst = os.path.join(cache, "bin", "loom")
    vt = sorted(glob.glob(os.path.join(inst, "**", "version.txt"), recursive=True))
    R["backend_version"] = open(vt[0]).read().strip() if vt else None
    check(f"installed Loom backend = {WANT}", R["backend_version"] == WANT, f"{R['backend_version']} ({vt})")
    text = open(os.path.join(OUT, "lemond.log"), errors="replace").read()
    R["serve_lines"] = [ln for ln in text.splitlines() if "loom-serve" in ln][:5]
    check("no --parameter_mode=preload in the launched serve line", "--parameter_mode=preload" not in text, R["serve_lines"])


def main():
    os.makedirs(OUT, exist_ok=True)
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK)
    try:
        lemond = build(os.path.join(WORK, "src"))
        if lemond:
            serve(lemond)
    finally:
        json.dump(R, open(os.path.join(OUT, "lemonade.json"), "w"), indent=1)
        shutil.rmtree(os.path.join(WORK, "src"), ignore_errors=True)
    return 0 if R["checks"] and all(v == "pass" for v in R["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
