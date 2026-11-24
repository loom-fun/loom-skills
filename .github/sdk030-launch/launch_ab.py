"""Matched launch-profile measurements on one DevLab runner."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import runpy
import shlex
import socket
import statistics
import subprocess
import sys
import threading
import time

import requests


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def deterministic_benchmark(bench, name, seed, target, argv):
    sys.path.insert(0, str(bench))
    import bench_common

    original = bench_common.stream
    journal = {}
    ordinals = {}
    lock = threading.Lock()

    def stream(url, model, messages, max_tokens, extra=None, timeout=None, on_token=None):
        prompt = messages[0]["content"]
        match = re.match(r"^\[\d+\.\d+ (\d+)\]", prompt)
        lane = match[1] if name == "c4_steady.py" and match else "serial"
        with lock:
            ordinal = ordinals.get(lane, 0)
            ordinals[lane] = ordinal + 1
        key = f"{lane}/{ordinal}"
        if prompt.startswith("["):
            prompt = re.sub(r"^\[[^\]]*\]", f"[benchmark {seed} {key}]", prompt, count=1)
            if prompt.startswith(f"[benchmark {seed} {key}] ["):
                prompt = re.sub(r"\] \[[^\]]*\]", "]", prompt, count=1)
        messages = [{**messages[0], "content": prompt}, *messages[1:]]
        result = original(url, model, messages, max_tokens, extra, timeout or 180, on_token)
        journal[key] = {
            "request": digest([model, messages, max_tokens, extra]),
            **{key: result.get(key) for key in ("status", "prompt_tokens", "tokens", "finish")},
            "output": digest(result.get("text")),
            "visible_timing": result.get("ttft") is not None,
        }
        return result

    bench_common.stream = stream
    sys.argv = [str(bench / name), *argv]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        write(target, journal)


def measure(work, sdk, bench):
    model = work / "model"
    out = work / "out"
    plan = read(out / "plan.json")
    focus = plan.get("focus")
    selected = read(model / "model.json")
    preset = plan["preset"]
    execution = selected["presets"][preset].get("execution", {})
    rows = int(execution.get("rows", plan.get("execution_rows", 1)))
    model_id = selected["model"]["id"]
    card_features = read(out / "info.json").get("card", {}).get("features", {})
    thinking = card_features.get("thinking", {}).get("value") or {}
    forced_reasoning = isinstance(thinking, dict) and thinking.get("mode") == "always"
    expected = plan.get("model_id")
    if expected and expected != model_id:
        raise ValueError(f"Expected {expected}, fetched {model_id}")
    tuned = ["--device=" + plan["device"], *plan["args"]]
    modes = {"current": tuned, "bare": ["--device=amdgpu"]}
    package_args = read(out / "llm_config.json").get("serve", {})
    if tuned == modes["bare"]:
        write(out / "comparison.json", {"plan": plan, "launch_args": modes, "package_serve": package_args,
                                         "status": "identical argv"})
        return
    if package_args and package_args.get("args"):
        config_args = package_args["args"]
        if isinstance(config_args, str):
            config_args = shlex.split(config_args)
        if not all(arg in tuned for arg in config_args):
            raise ValueError(f"Quick Start omits package launch args: {config_args} vs {tuned}")
    relevant_env = {key: value for key, value in os.environ.items()
                    if key.startswith(("LOOM_", "HSA_", "HIP_", "ROCM_", "HRX_", "ROCR_", "GPU_"))
                    and not re.search(r"TOKEN|KEY|SECRET|PASSWORD|AUTH", key, re.IGNORECASE)}
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    target = 8 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (target if hard == resource.RLIM_INFINITY else min(target, hard), hard))
    write(out / "environment.json", {"overrides": relevant_env, "before_memlock": [soft, hard],
                                     "memlock": resource.getrlimit(resource.RLIMIT_MEMLOCK)})

    sys.path.insert(0, str(bench))
    from bench_common import server_draft_counters

    def session(number, arm):
        directory = out / f"{number}-{arm}"
        directory.mkdir()
        pair = (number + 1) // 2
        env = {**os.environ, "XDG_CACHE_HOME": str(work / "cache")}
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        argv = [str(sdk / ".venv/bin/loom-serve"), f"--model={model / 'model.json'}",
                f"--listen=127.0.0.1:{port}", *modes[arm]]
        report = {"number": number, "pair": pair, "arm": arm, "rows": rows, "metrics": {}, "errors": {}}
        started = time.monotonic()

        def phase(name, operation):
            before = time.monotonic()
            try:
                return operation()
            except (OSError, ValueError, RuntimeError, KeyError, IndexError, subprocess.SubprocessError,
                    requests.RequestException) as error:
                report["errors"][name] = f"{type(error).__name__}: {error}"
                return None
            finally:
                report.setdefault("phase_seconds", {})[name] = round(time.monotonic() - before, 2)
                write(directory / "session.json", report)

        def benchmark(label, script, args, timeout, deterministic=False):
            command = [sys.executable, str(bench / script), *args]
            if deterministic:
                command = [sys.executable, str(Path(__file__).resolve()), "bench", str(bench), script,
                           f"pair-{pair}-{label}", str(directory / f"{label}.requests.json"), *args]
            with (directory / f"{label}.log").open("w") as log:
                subprocess.run(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=timeout)

        def timed(label, script, args, timeout):
            before = server_draft_counters(url)
            try:
                benchmark(label, script, args, timeout, True)
            finally:
                after = server_draft_counters(url)
                report.setdefault("draft_work", {})[label] = {
                    key: after[key] - before.get(key, 0) for key in after
                }

        def ready(server):
            deadline = time.monotonic() + 420
            while time.monotonic() < deadline and server.poll() is None:
                try:
                    response = requests.get(url + "/v1/status", timeout=2)
                    response.raise_for_status()
                    status = response.json()
                    write(directory / "status.json", status)
                    registrations = [item["loom"]["registration"] for item in status["data"]]
                    if "failed" in registrations:
                        raise RuntimeError("model registration failed")
                    if registrations and all(value == "ready" for value in registrations):
                        actual = Path(f"/proc/{server.pid}/cmdline").read_bytes().decode().rstrip("\0").split("\0")
                        write(directory / "argv.json", actual)
                        if actual[-len(argv) + 1:] != argv[1:]:
                            raise RuntimeError(f"unexpected process argv: {actual}")
                        report["metrics"]["registration_seconds"] = round(time.monotonic() - started, 2)
                        benchmark("ready", "wait_model.py", [url, "--model", model_id, "--wait", "180"], 230)
                        report["metrics"]["ready_seconds"] = round(time.monotonic() - started, 2)
                        return True
                except (requests.RequestException, ValueError):
                    pass
                time.sleep(1)
            raise RuntimeError(f"registration timed out or server exited ({server.poll()})")

        def functional():
            results = []
            for prompt, expected in (
                ("Name the capital of France. Reply with only the city name.", "paris"),
                ("What is 6 times 7? Reply with only the number.", "42"),
                ("Repeat this word once and nothing else: violet", "violet"),
            ):
                response = requests.post(url + "/v1/chat/completions", timeout=90, json={
                    "model": model_id, "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0, "max_tokens": 512 if forced_reasoning else 32,
                    "chat_template_kwargs": {"enable_thinking": forced_reasoning},
                })
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                answer = choice["message"].get("content") or ""
                reasoning = choice["message"].get("reasoning_content") or ""
                results.append({"prompt": prompt, "expected": expected, "answer": answer,
                                "reasoning_only": bool(reasoning and not answer.strip()),
                                "reasoning_sha256": digest(reasoning) if reasoning else None,
                                "completion_tokens": body.get("usage", {}).get("completion_tokens"),
                                "finish": choice.get("finish_reason"),
                                "exact": answer.strip().lower().rstrip(".") == expected})
            write(directory / "answers.json", results)
            if any(not row["finish"] or not row["answer"].strip() for row in results):
                raise RuntimeError("empty answer or missing finish reason")
            return sum(result["exact"] for result in results)

        def capabilities():
            from spec_check import image_requests, TOOLS

            features = plan.get("features", {})
            results = {}
            cases = {}
            if features.get("image_input") == "yes":
                cases["vision"] = {**image_requests(model_id)[0], "max_tokens": 96}
            if features.get("tool_calling") == "yes":
                cases["tools"] = {
                    "model": model_id, "tools": TOOLS,
                    "messages": [{"role": "user", "content": "Use the get_weather tool for Paris."}],
                    "temperature": 0, "max_tokens": 128, "chat_template_kwargs": {"enable_thinking": False},
                }
            for name, body in cases.items():
                response = requests.post(url + "/v1/chat/completions", json=body, timeout=180)
                response.raise_for_status()
                choice = response.json()["choices"][0]
                message = choice["message"]
                calls = [call["function"] for call in message.get("tool_calls") or []]
                passed = any(call["name"] == "get_weather" and "paris" in call["arguments"].lower()
                             for call in calls) if name == "tools" else bool(message.get("content"))
                results[name] = {"request": digest(body), "content": message.get("content"),
                                 "calls": calls, "finish": choice.get("finish_reason"), "passed": passed}
            write(directory / "capabilities.json", results)
            return results

        def perf(label, args):
            common = [url, arm, "--model", model_id, "--no-campaign", "--thinking", "off",
                      "--repeat-rows", "2", "--repeats", "3", "--log", f"{label}.jsonl"]
            timed(label, "perf.py", common + args, 900)
            result = json.loads((directory / f"{label}.jsonl").read_text().splitlines()[-1])
            if label == "short":
                report["metrics"].update({key: result.get(key) for key in ("decode_tps", "decode_tps_2k")})
                report["metrics"].update({f"prefill_{row.get('size', index * 1000) // 1000}k": row["prefill_tps"]
                                          for index, row in enumerate(result["prefill"], 1)})
            else:
                report["metrics"]["decode_tps_32k"] = result.get("decode_tps_32k")
                lengths = [length for length in (8, 16, 32)
                           if f"prefill_{length * 1000}" not in result.get("over_context", [])]
                report["metrics"].update({f"prefill_{row.get('size', length * 1000) // 1000}k": row["prefill_tps"]
                                          for length, row in zip(lengths, result["prefill_long"])})
                report["over_context"] = result.get("over_context", [])
            if result.get("prefill_untimed"):
                report.setdefault("prefill_untimed", {})[label] = result["prefill_untimed"]
            return True

        def concurrent():
            label = f"c{min(rows, 4)}"
            timed(label, "c4_steady.py", [url, str(min(rows, 4)), "--model", model_id,
                  "--thinking", "off", "--warmup-min", "3", "--warmup-passes", "3"], 720)
            result = json.loads((directory / f"{label}.log").read_text().splitlines()[-1])
            report["metrics"][f"{label}_agg_tps"] = result["agg_tps"]
            report["draft_acceptance"] = result.get("acceptance")

        def admission():
            benchmark("admission", "ctx_admit.py", [str(model), url, "--preset", preset,
                      "--wait", "120", "--out", "admission.json"], 210)
            return read(directory / "admission.json")

        print(f"Session {number} {arm}: {argv}", flush=True)
        write(directory / "expected-argv.json", argv)
        with (directory / "serve.log").open("w") as log:
            server = subprocess.Popen(argv, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                if phase("cold_launch", lambda: ready(server)):
                    if focus is None or "functional" in focus:
                        report["functional_exact"] = phase("functional", functional)
                    if focus is None or "capabilities" in focus:
                        report["capabilities"] = phase("capabilities", capabilities)
                    if focus is None or "short" in focus:
                        phase("short", lambda: perf("short", []))
                    if focus is None or "long" in focus:
                        phase("long", lambda: perf("long", ["--long"]))
                    if rows >= 2 and (focus is None or "concurrency" in focus):
                        phase("concurrency", concurrent)
                    if focus is None or "native_admission" in focus:
                        report["admission"] = phase("native_admission", admission)
                        if report["admission"] and not report["admission"].get("admitted"):
                            report["errors"]["native_admission"] = "native context refused"
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
                report["server_exit"] = server.returncode
                write(directory / "session.json", report)
        return report

    reports = []
    for number, arm in enumerate(("current", "bare", "bare", "current"), 1):
        reports.append(session(number, arm))
        write(out / "sessions.json", reports)
    pairs = []
    for pair, (current, bare) in enumerate(((reports[0], reports[1]), (reports[3], reports[2])), 1):
        a = out / f"{current['number']}-current"
        b = out / f"{bare['number']}-bare"
        journals = {}
        for filename in sorted({p.name for side in (a, b) for p in side.glob("*.requests.json")}):
            old, new = [read(side / filename) if (side / filename).exists() else {} for side in (a, b)]
            common = old.keys() & new.keys()
            journals[filename] = {
                "request_counts": [len(old), len(new)], "matched_requests": sum(
                    old[key]["request"] == new[key]["request"] for key in common),
                "matched_outputs": sum(old[key] == new[key] for key in common),
                "mismatched_keys": sorted(key for key in common if old[key] != new[key]),
            }
        delta = {metric: round(100 * (bare["metrics"][metric] / value - 1), 2)
                 for metric, value in current["metrics"].items() if value and bare["metrics"].get(metric)}
        pairs.append({"pair": pair, "bare_delta_pct": delta, "journals": journals,
                      "draft_work_match": current.get("draft_work") == bare.get("draft_work"),
                      "capabilities_match": current.get("capabilities") == bare.get("capabilities"),
                      "functional_match": (a / "answers.json").exists() and (b / "answers.json").exists()
                      and read(a / "answers.json") == read(b / "answers.json")})
    medians = {metric: round(statistics.median(pair["bare_delta_pct"][metric] for pair in pairs), 2)
               for metric in pairs[0]["bare_delta_pct"].keys() & pairs[1]["bare_delta_pct"].keys()}
    write(out / "comparison.json", {"plan": plan, "launch_args": modes, "package_serve": package_args,
                                     "median_paired_bare_delta_pct": medians, "pairs": pairs, "sessions": reports})
    if any(report["errors"] for report in reports) or any(
        ((focus is None or "functional" in focus) and not pair["functional_match"])
        or not pair["capabilities_match"] for pair in pairs
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    if sys.argv[1] == "bench":
        _, _, bench, name, seed, target, *args = sys.argv
        deterministic_benchmark(Path(bench), name, seed, Path(target), args)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("work", type=Path)
        parser.add_argument("sdk", type=Path)
        parser.add_argument("bench", type=Path)
        args = parser.parse_args()
        measure(args.work, args.sdk, args.bench)
