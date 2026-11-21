#!/usr/bin/env python3
"""Run card performance workloads on a leased Windows gfx1152 host with pre-staged inputs."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
from zipfile import ZipFile

from machine_info import detect_device, hrx_loom_identity

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "card"))
from card_lib import programs_sha256  # noqa: E402
from matched_spec import classification, violations  # noqa: E402

ROCM_PIN = "918f16a4657b83925df5a9a1e10485f177b46756a8b17d8c6e2deae58dce0d2a"
VULKAN_ASSET = "llama-b12211-bin-win-vulkan-x64.zip"
DEVICE = "amdgpu://?completion_spin_ns=4294967295"
KV_CONTEXT_LIMITS = {"rocm": 65536, "vulkan": 36864}
EXTERNAL_DRAFT_PINS = {
    "AtomicChat/Qwen3.5-4B-DFlash-GGUF/Qwen3.5-4B-DFlash.Q8_0.gguf": {
        "repository": "AtomicChat/Qwen3.5-4B-DFlash-GGUF",
        "revision": "56b80828981c35dff23affb49820e9ef816048f0",
        "file": "Qwen3.5-4B-DFlash.Q8_0.gguf",
        "sha256": "f0689eadab0d46468bc2629bb2706f4aba7f794b63e1c9f6eb291269d572a52b",
    },
}
LOGS = {
    "loom": ("perf_loom.jsonl", "c4_loom.txt"),
    "rocm": ("perf_llamacpp-rocm.np1.jsonl", "c4_llamacpp-rocm.nodraft.txt"),
    "vulkan": ("perf_llamacpp-vulkan.np1.jsonl", "c4_llamacpp-vulkan.nodraft.txt"),
}
HARDWARE_PROBE = """
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
[ordered]@{
  computer = (Get-CimInstance Win32_ComputerSystem | Select-Object Name,Manufacturer,Model,TotalPhysicalMemory)
  system = (Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,TotalVisibleMemorySize)
  cpu = @(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors)
  video = @(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,PNPDeviceID)
  power_scheme = (powercfg.exe /GETACTIVESCHEME | Out-String).Trim()
} | ConvertTo-Json -Depth 5 -Compress
"""
HTTP = build_opener(ProxyHandler({}))


class RunError(RuntimeError):
    pass


class CleanupError(RunError):
    """A subsequent model must not load while the previous listener still exists."""

@dataclass
class Server:
    process: subprocess.Popen
    child_listener: dict | None = None



def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def append_json(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value) + "\n")


def digest(stream):
    sha = hashlib.sha256()
    for chunk in iter(lambda: stream.read(2**20), b""):
        sha.update(chunk)
    return sha.hexdigest()


def sha256(path):
    with path.open("rb") as stream:
        return digest(stream)


def identity(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def baseline_identity(backend, exe, archive):
    pin = ROCM_PIN
    if backend == "vulkan":
        pins = (HERE.parent / "data" / "llamacpp-vulkan.sha256").read_text(encoding="utf-8").splitlines()
        pin = next(line.split()[0] for line in pins if line.endswith(" " + VULKAN_ASSET))
    release = identity(archive)
    if release["sha256"] != pin:
        raise RunError(f"{backend} archive SHA-256 differs from the Windows release pin")
    binary = identity(exe)
    libraries = []
    with ZipFile(archive) as zipped:
        candidates = [name for name in zipped.namelist() if PurePosixPath(name).name.lower() == "llama-server.exe"]
        if len(candidates) != 1:
            raise RunError(f"{backend} pinned archive must contain exactly one llama-server.exe")
        with zipped.open(candidates[0]) as member:
            if digest(member) != binary["sha256"]:
                raise RunError(f"{backend} executable differs from the pinned archive")
        sibling = PurePosixPath(candidates[0]).parent
        for name in zipped.namelist():
            file = PurePosixPath(name)
            if file.parent == sibling and file.suffix.lower() == ".dll":
                library = identity(exe.parent / file.name)
                with zipped.open(name) as member:
                    if digest(member) != library["sha256"]:
                        raise RunError(f"{backend} {file.name} differs from the pinned archive")
                libraries.append(library)
    return {"build": "b1382" if backend == "rocm" else "b12211", "archive": release,
            "executable": binary, "release_dlls": libraries}


def package_identity(args):
    path = args.package / "model.json"
    model = read_json(path)
    cfg = read_json(args.package / "llm_config.json")
    preset = model["presets"][args.preset]
    execution = preset["execution"]
    parameters = model["variants"][preset["variant"]]["services"][preset["service"]]["parameters"]
    files = parameters["model"]["files"]
    if len(files) != 1 or files[0]["format"] != "gguf":
        raise RunError("Expected the gfx1152 preset to load one model GGUF")
    target = (args.package / files[0]["path"]).resolve()
    expected = classification(args.repo)
    problems = violations(args.repo, cfg)
    if problems:
        raise RunError("; ".join(problems))
    drafts = execution.get("draft_depth", 0) > 0
    if drafts != (expected["choice"] != "none"):
        raise RunError("Preset draft depth disagrees with the model's drafting classification")
    program_sha = programs_sha256(args.package)
    if not program_sha:
        raise RunError("No packaged .loom programs to hash")
    base = identity(args.gguf) if args.gguf else identity(target)
    if args.gguf and target != args.gguf and sha256(target) != base["sha256"]:
        raise RunError("The baseline GGUF differs from the selected Loom preset's model GGUF")
    package = {
        "repo": args.repo,
        "model_id": model["model"]["id"],
        "model_json": identity(path),
        "llm_config": identity(args.package / "llm_config.json"),
        "programs_sha256": program_sha,
        "gguf": base,
        "preset": args.preset,
        "execution": execution,
        "draft_method": expected["choice"],
    }
    if expected["choice"] == "dflash":
        drafter = (args.package / parameters["draft"]["files"][0]["path"]).resolve()
        package["drafter"] = identity(drafter)
        if "{drafter}" in expected["llama_spec"] and args.draft and args.draft != drafter:
            if sha256(args.draft) != package["drafter"]["sha256"]:
                raise RunError("The baseline drafter differs from the selected Loom preset's drafter")
    return model, package


def spec_args(repo, draft):
    classified = classification(repo)
    spec = shlex.split(classified["llama_spec"])
    if not spec:
        if draft:
            raise RunError("The model classification calls for no drafter")
        return [], None
    drafter = None
    for index, arg in enumerate(spec):
        if arg == "{drafter}" or arg.startswith("{hf:"):
            if not draft:
                raise RunError(f"The classified baseline needs --draft ({arg})")
            pin = None
            if arg.startswith("{hf:"):
                pin = EXTERNAL_DRAFT_PINS.get(arg.removeprefix("{hf:").removesuffix("}"))
                if pin is None:
                    raise RunError(f"No immutable input pin for the classified external drafter {arg}")
            expected = pin["file"] if pin else classified["drafter_file"]
            if draft.name != expected:
                raise RunError(f"The drafting classification requires {expected}, got {draft.name}")
            drafter = identity(draft)
            if pin:
                if drafter["sha256"] != pin["sha256"]:
                    raise RunError(f"The external drafter SHA-256 differs from the pinned {pin['repository']}@{pin['revision']}")
                drafter["source"] = pin
            spec[index] = str(draft)
    return spec, drafter


def server_argv(args, engine, mode, model_path, context, spec, device=None):
    if engine == "loom":
        if args.memory_bytes is None:
            pool_args = ["--pool_backing=fixed"]
        else:
            pool_args = ["--pool_backing=elastic", f"--memory_bytes={args.memory_bytes}"]
        return [str(args.loom), f"--device={DEVICE}", f"--listen=127.0.0.1:{args.port}",
                f"--model={model_path}", f"--preset={args.preset}", *pool_args, *args.loom_arg]
    streams = 4 if mode == "nodraft" else 1
    unified = engine == "rocm" and streams == 4
    context = min(context, KV_CONTEXT_LIMITS[engine])
    total_context = context if unified else context * streams
    draft = spec if mode == "single" else []
    return [str(getattr(args, engine)), "-m", str(args.gguf), "-ngl", "99", "-cram", "0",
            "--device", device, *draft, "-c", str(total_context), *(["-kvu"] if unified else []),
            "-np", str(streams), "--port", str(args.port), "--host", "127.0.0.1"]


class Evidence:
    def __init__(self, out):
        self.out = out
        self.env = os.environ.copy()
        run_dir = self.env.get("XSJ_RUN_DIR")
        cache = Path(tempfile.mkdtemp(prefix="xsj-perf-cache-", dir=run_dir)) if run_dir else out / "cache"
        for name in ("HF_HOME", "IREE_CACHE_DIR", "LOOM_CACHE_DIR", "XDG_CACHE_HOME", "TEMP", "TMP"):
            path = cache / name.lower()
            path.mkdir(parents=True, exist_ok=True)
            self.env[name] = str(path)
        self.env["PYTHONUNBUFFERED"] = "1"
        self.env["PYTHONIOENCODING"] = "utf-8"
        self.env["NO_PROXY"] = "localhost,127.0.0.1"
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"

    def event(self, **record):
        append_json(self.out / "commands.jsonl", {"utc": now(), **record})

    def command(self, name, argv, *, cwd=None, timeout=300):
        cwd = cwd or self.out
        stdout, stderr = f"{name}.stdout.log", f"{name}.stderr.log"
        self.event(event="start", name=name, argv=argv, cwd=str(cwd), stdout=stdout, stderr=stderr)
        code = None
        try:
            with (self.out / stdout).open("wb") as out, (self.out / stderr).open("wb") as err:
                code = subprocess.run(argv, cwd=cwd, env=self.env, stdout=out, stderr=err, timeout=timeout).returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RunError(f"{name}: {exc}") from exc
        finally:
            self.event(event="exit", name=name, code=code)
        if code:
            raise RunError(f"{name}: exit {code}; see {stderr} and {stdout}")
        return (self.out / stdout).read_text(encoding="utf-8-sig", errors="replace")

    @contextmanager
    def server(self, name, argv, port):
        if listener_pids(self, f"{name}.port_before", port):
            raise CleanupError(f"Port {port} already has a listener; refusing to start {name}")
        cwd = Path(argv[0]).parent
        log = f"{name}.server.log"
        self.event(event="start", name=name, argv=argv, cwd=str(cwd), stdout=log, stderr=log)
        with (self.out / log).open("wb") as out:
            proc = subprocess.Popen(argv, cwd=cwd, env=self.env, stdout=out, stderr=subprocess.STDOUT)
            server = Server(proc)
            try:
                self.event(event="pid", name=name, pid=proc.pid)
                yield server
            finally:
                if proc.poll() is None:
                    try:
                        self.command(f"{name}.stop", ["taskkill.exe", "/PID", str(proc.pid), "/T", "/F"], timeout=30)
                    except RunError:
                        if proc.poll() is None:
                            self.event(event="kill", name=name, pid=proc.pid)
                            try:
                                proc.kill()
                            except OSError as exc:
                                raise CleanupError(f"Could not stop owned server PID {proc.pid}: {exc}") from exc
                try:
                    proc.wait(timeout=30)
                    listeners = listener_pids(self, f"{name}.port_after", port)
                    child = server.child_listener
                    if child and str(child["pid"]) in listeners:
                        current = windows_process(self, f"{name}.child_after", child["pid"])
                        if current == child:
                            try:
                                self.command(f"{name}.stop_child", ["taskkill.exe", "/PID", str(child["pid"]), "/T", "/F"],
                                             timeout=30)
                            except RunError:
                                # The child can exit on its own between the identity check and taskkill.
                                pass
                            listeners = listener_pids(self, f"{name}.port_after_child", port)
                    if listeners:
                        raise RunError(f"Port {port} still has a listener after PID {proc.pid} exited")
                except (subprocess.TimeoutExpired, RunError) as exc:
                    raise CleanupError(f"Server {name} failed to release port {port}: {exc}") from exc
                self.event(event="exit", name=name, code=proc.returncode)


def listener_pids(evidence, name, port):
    listeners = evidence.command(name, ["netstat.exe", "-ano", "-p", "TCP"], timeout=20)
    return set(re.findall(rf"(?m)^\s*TCP\s+\S+:{port}\s+\S+\s+LISTENING\s+(\d+)\s*$", listeners))


def windows_process(evidence, name, pid):
    cmd = (f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}' -ErrorAction Stop; "
           "if ($p) { @{pid=$p.ProcessId; parent=$p.ParentProcessId; "
           "created=$p.CreationDate.ToUniversalTime().ToString('o')} | ConvertTo-Json -Compress } else { 'null' }")
    return json.loads(evidence.command(name, ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", cmd],
                                       timeout=30))


def verify_listener(evidence, name, server, port):
    proc = server.process
    listeners = listener_pids(evidence, f"{name}.listener", port)
    if listeners == {str(proc.pid)} and proc.poll() is None:
        return
    if len(listeners) == 1:
        root = windows_process(evidence, f"{name}.root_process", proc.pid)
        owner = int(next(iter(listeners)))
        child = windows_process(evidence, f"{name}.listener_process", owner)
        ancestor = child
        visited = set()
        while root and ancestor and ancestor["pid"] not in visited and ancestor["created"] >= root["created"]:
            visited.add(ancestor["pid"])
            if ancestor["parent"] == proc.pid and proc.poll() is None:
                server.child_listener = child
                evidence.event(event="listener_child", name=name, **child)
                return
            parent = windows_process(evidence, f"{name}.ancestor_{ancestor['parent']}", ancestor["parent"])
            # A reused PID can never be the parent of an older process.
            if not parent or parent["created"] > ancestor["created"]:
                break
            ancestor = parent
    raise RunError(f"{name}: responding listener is outside the live launched process tree (root PID {proc.pid})")


def wait_ready(evidence, name, server, engine, url, port, model_id, timeout):
    proc = server.process
    endpoint = "/v1/status" if engine == "loom" else "/health"
    deadline = time.monotonic() + timeout
    last = "server has not answered"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            lines = (evidence.out / f"{name}.server.log").read_text(encoding="utf-8", errors="replace").splitlines()
            raise RunError(f"{name}: server exited {proc.returncode}; " + " / ".join(lines[-4:])[-800:])
        try:
            with HTTP.open(url + endpoint, timeout=2) as response:
                status = json.load(response)
        except (URLError, OSError, ValueError) as exc:
            last = str(exc)
        else:
            write_json(evidence.out / f"{name}.readiness.json", status)
            if engine == "loom":
                last = next((x["loom"]["registration"] for x in status.get("data", [])
                             if x.get("id") == model_id), "model not registered")
                if last == "failed":
                    raise RunError(f"{name}: Loom registration failed; see {name}.readiness.json")
            else:
                last = status.get("status")
            if last in ("ready", "ok"):
                verify_listener(evidence, name, server, port)
                return
        time.sleep(1)
    raise RunError(f"{name}: not ready after {timeout:g}s ({last}); see {name}.server.log")


@contextmanager
def undrafted_model(args, model, out):
    variant = copy.deepcopy(model)
    variant["presets"][args.preset]["execution"]["draft_depth"] = 0
    write_json(out / "model.nodraft.json", variant)
    # Manifest asset paths are relative to model.json. Write an owned sibling, since the
    # staged assets might be large. Leave the measured model.json and its lock alone.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.package,
                                     prefix="xsj-nodraft-", suffix=".json", delete=False) as manifest:
        path = Path(manifest.name)
        json.dump(variant, manifest)
    try:
        yield path
    finally:
        path.unlink()


def workloads(evidence, args, engine, mode, name, model_id, url):
    model = ["--model", model_id] if engine == "loom" else []
    common = ["--thinking", "off", *model]
    if mode == "single":
        log = evidence.out / f"{name}.raw.jsonl"
        cmd = [sys.executable, str(HERE / "perf.py"), url, engine, "--no-campaign",
               "--kind", "loom" if engine == "loom" else "llama", "--log", str(log), *common,
               "--repeat-rows", str(args.repeat_rows)]
        evidence.command(f"{name}.short", cmd, timeout=args.workload_timeout)
        evidence.command(f"{name}.long", [*cmd, "--long"], timeout=args.workload_timeout)
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        if len(rows) != 2 or "prefill" not in rows[0] or "prefill_long" not in rows[1]:
            raise RunError(f"{name}: expected one short and one long row")
        return rows
    cmd = [sys.executable, str(HERE / "c4_steady.py"), url, "4", *common]
    raw = evidence.command(f"{name}.c4", cmd, timeout=args.workload_timeout)
    if not raw.strip():
        raise RunError(f"{name}: c4_steady.py produced no output")
    row = json.loads(raw.strip().splitlines()[-1])
    if row.get("streams") != 4 or row.get("agg_tps", 0) <= 0 or row.get("window_s", 0) <= 0:
        raise RunError(f"{name}: c4_steady.py produced no valid four-stream measurement")
    return [row]


def measure(evidence, args, engine, mode, name, model_path, model_id, context, spec, device):
    url = f"http://127.0.0.1:{args.port}"
    argv = server_argv(args, engine, mode, model_path, context, spec, device)
    with evidence.server(name, argv, args.port) as server:
        wait_ready(evidence, name, server, engine, url, args.port, model_id, args.startup_timeout)
        if engine == "loom":
            evidence.command(f"{name}.load", [sys.executable, str(HERE / "wait_model.py"), url,
                                             "--model", model_id, "--wait", str(args.startup_timeout)],
                             timeout=2 * args.startup_timeout + 10)
        rows = workloads(evidence, args, engine, mode, name, model_id, url)
    log = evidence.out / LOGS[engine][0 if mode == "single" else 1]
    with log.open("a", encoding="utf-8") as stream:
        stream.write("".join(json.dumps(row) + "\n" for row in [*rows, {"session_end": name}]))
    return rows


def probe_backend(evidence, engine, exe):
    evidence.command(f"{engine}.version", [str(exe), "--version"], cwd=exe.parent, timeout=60)
    report = evidence.command(f"{engine}.devices", [str(exe), "--list-devices"], cwd=exe.parent, timeout=60)
    report += "\n" + (evidence.out / f"{engine}.devices.stderr.log").read_text(encoding="utf-8", errors="replace")
    prefix = "ROCm" if engine == "rocm" else "Vulkan"
    matches = re.findall(rf"(?mi)^\s*({prefix}\d+)\s*:\s*(.*(?:gfx1152|Radeon.*860M).*)$", report)
    if len(matches) != 1:
        raise RunError(f"Expected exactly one gfx1152/Radeon 860M {prefix} device; see {engine}.devices logs")
    return matches[0][0]


def arguments(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "out"):
        ap.add_argument(f"--{name}", required=True, type=Path)
    for name in ("loom", "gguf", "draft", "rocm", "rocm-archive", "vulkan", "vulkan-archive"):
        ap.add_argument(f"--{name}", type=Path)
    ap.add_argument("--preset", default="chat-gfx1152")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run-id", default=os.environ.get("XSJ_RUN") or Path(os.environ.get("XSJ_RUN_DIR", "")).name)
    ap.add_argument("--host-alias", required=True, help="Lease pool alias (amd-krk-1 or amd-krk-2)")
    ap.add_argument("--vulkan-unavailable")
    ap.add_argument("--engine", action="append", choices=LOGS)
    ap.add_argument("--loom-arg", action="append", default=[], choices=["--parameter_mode=preload"])
    ap.add_argument("--memory-bytes", type=int, help="Loom's tested elastic pool budget in bytes (default: fixed backing)")
    ap.add_argument("--port", type=int, default=os.environ.get("XSJ_PORT"))
    ap.add_argument("--sessions", type=int)
    ap.add_argument("--fast", action="store_true", help="Methodology rule 5: six sessions and three in-row samples")
    ap.add_argument("--startup-timeout", type=int, default=300)
    ap.add_argument("--workload-timeout", type=int, default=1800)
    args = ap.parse_args(argv)
    args.engine = list(dict.fromkeys(args.engine or LOGS))
    args.sessions = args.sessions if args.sessions is not None else (6 if args.fast else 3)
    args.repeat_rows = 3 if args.fast else 1
    if not 1 <= (args.port or 0) <= 65535:
        ap.error("--port or XSJ_PORT must name a valid run-owned loopback port")
    if args.memory_bytes is not None and not 1 <= args.memory_bytes < 2**64:
        ap.error("--memory-bytes must be a nonzero unsigned 64-bit byte count")
    if min(args.sessions, args.startup_timeout, args.workload_timeout) < 1:
        ap.error("sessions and timeouts must be positive")
    if not re.fullmatch(r"\d{8}-\d{6}-[a-z0-9]{4}", args.run_id):
        ap.error("identify the lease-backed run with --run-id or XSJ_RUN")
    if os.environ.get("XSJ_RUN") and args.run_id != os.environ["XSJ_RUN"]:
        ap.error("--run-id must match the current XSJ_RUN")
    if not re.fullmatch(r"amd-krk-\d+", args.host_alias):
        ap.error("--host-alias must name the leased amd-krk host")
    if "loom" in args.engine and args.loom is None:
        ap.error("--loom is required to measure Loom")
    if "vulkan" in args.engine and args.vulkan_unavailable:
        if args.vulkan or args.vulkan_archive or not args.vulkan_unavailable.strip():
            ap.error("--vulkan-unavailable needs a reason and cannot accompany Vulkan inputs")
    for engine in ("rocm", "vulkan"):
        if engine in args.engine and not (engine == "vulkan" and args.vulkan_unavailable):
            if not all((getattr(args, engine), getattr(args, engine + "_archive"), args.gguf)):
                ap.error(f"{engine} requires --{engine}, --{engine}-archive and --gguf")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    return args


def run(args, evidence, manifest):
    model, package = package_identity(args)
    manifest["package"] = package
    (evidence.out / "programs_sha256.txt").write_text(package["programs_sha256"] + "\n", encoding="ascii")
    write_json(evidence.out / "model.measured.json", model)
    manifest["hardware"] = json.loads(evidence.command("hardware", [
        "powershell.exe", "-NoProfile", "-NonInteractive", "-Command", HARDWARE_PROBE]))
    adapters = [row for row in manifest["hardware"]["video"]
                if re.search(r"Radeon.*860M", row["Name"], re.I) and row["DriverVersion"]]
    if len(adapters) != 1:
        raise RunError("Hardware inventory must identify one Radeon 860M and its installed driver")
    device = detect_device()
    device.update(host=args.host_alias, arch="gfx1152",
                  gpu=adapters[0]["Name"], driver=adapters[0]["DriverVersion"])
    write_json(evidence.out / "device.json", device)
    sdk = hrx_loom_identity(loom_root=sys.prefix, venv=sys.prefix)
    if not sdk:
        raise RunError("Run with the Python environment where hrx-loom is installed")
    write_json(evidence.out / "hrx-loom.json", sdk)
    context = package["execution"]["context_capacity"]
    if "loom" in args.engine and package["execution"].get("rows", 0) < 4:
        raise RunError("Selected Loom preset needs at least four rows for c4")
    engine_ids = manifest["engines"]
    if "loom" in args.engine:
        if args.loom != Path(sys.executable).with_name("loom-serve.exe").resolve():
            raise RunError("--loom must point to the SDK's loom-serve.exe next to the running Python")
        engine_ids["loom"] = {"sdk": sdk, "executable": identity(args.loom), "wait": DEVICE}
        evidence.command("loom.version", [str(args.loom), "--version"], cwd=args.loom.parent, timeout=60)
    baselines = [eng for eng in args.engine if eng != "loom" and not (eng == "vulkan" and args.vulkan_unavailable)]
    spec, drafter = spec_args(args.repo, args.draft) if baselines else ([], None)
    if drafter:
        manifest["baseline_drafter"] = drafter
    failed = False
    for engine in baselines:
        exe = getattr(args, engine)
        try:
            engine_ids[engine] = baseline_identity(engine, exe, getattr(args, engine + "_archive"))
            engine_ids[engine]["device"] = probe_backend(evidence, engine, exe)
            limit = min(context, KV_CONTEXT_LIMITS[engine])
            engine_ids[engine]["context"] = {"single": limit, "four_stream_total": limit if engine == "rocm" else 4 * limit,
                                             "four_stream_shared": engine == "rocm"}
        except (RunError, OSError, ValueError) as exc:
            engine_ids.setdefault(engine, {})["error"] = str(exc)
            failed = True
    if "vulkan" in args.engine and args.vulkan_unavailable:
        engine_ids["vulkan"] = {"status": "unavailable", "reason": args.vulkan_unavailable}

    model_path = args.package / "model.json"
    if "loom" in args.engine:
        with undrafted_model(args, model, evidence.out) as nodraft:
            failed = sessions(args, evidence, manifest, context, spec, model_path, nodraft) or failed
    else:
        failed = sessions(args, evidence, manifest, context, spec, model_path, None) or failed
    if programs_sha256(args.package) != package["programs_sha256"] or sha256(model_path) != package["model_json"]["sha256"]:
        raise RunError("The measured package's programs or model manifest changed during this job")
    return int(failed)


def sessions(args, evidence, manifest, context, spec, model_path, nodraft):
    failed = False
    for index in range(1, args.sessions + 1):
        for engine in args.engine:
            details = manifest["engines"][engine]
            if "error" in details or details.get("status") == "unavailable":
                continue
            for mode in ("single", "nodraft"):
                name = f"{engine}.{mode}.{index:02d}"
                entry = {"id": name, "engine": engine, "mode": mode, "round": index, "status": "started"}
                manifest["sessions"].append(entry)
                write_json(evidence.out / "manifest.json", manifest)
                print(name, flush=True)
                try:
                    rows = measure(evidence, args, engine, mode, name, nodraft if mode == "nodraft" else model_path,
                                   manifest["package"]["model_id"], context, spec, details.get("device"))
                except CleanupError as exc:
                    entry.update(status="cleanup_failed", reason=str(exc))
                    raise
                except (RunError, OSError, ValueError) as exc:
                    entry.update(status="failed", reason=str(exc))
                    print(f"{name}: {exc}", file=sys.stderr, flush=True)
                    failed = True
                else:
                    entry["status"] = "completed"
                    if mode == "single":
                        entry["over_context"] = rows[1].get("over_context", [])
                        entry["short_completion"] = [k for row in rows for k, count in row.get("completion_tokens", {}).items()
                                                     if count and row.get(k) is None]
                finally:
                    write_json(evidence.out / "manifest.json", manifest)
    return failed


def main(argv=None):
    args = arguments(argv)
    if os.name != "nt":
        raise SystemExit("xsj_perf.py requires native Windows Python")
    args.out.mkdir(parents=True, exist_ok=False)
    evidence = Evidence(args.out)
    manifest = {
        "schema": "loom-xsj-perf/1", "source_kind": "xsj", "run_id": args.run_id, "started_utc": now(),
        "host": args.host_alias, "computer_name": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version,
        "runner_sha256": sha256(Path(__file__)),
        "workload_sha256": {name: sha256(HERE / name) for name in
                            ("perf.py", "c4_steady.py", "bench_common.py", "wait_model.py",
                             "../data/doc_2000.txt", "../data/doc_32000.txt")},
        "argv": [*sys.argv] if argv is None else [str(Path(__file__)), *argv],
        "environment": {k: v for k, v in evidence.env.items() if k in
                        ("HF_HOME", "IREE_CACHE_DIR", "LOOM_CACHE_DIR", "XDG_CACHE_HOME", "TEMP", "TMP", "NO_PROXY",
                         "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION",
                         "VK_DRIVER_FILES", "AMD_VULKAN_ICD")},
        "sampling": {"requested_sessions": args.sessions, "repeat_rows": args.repeat_rows,
                     "engines": args.engine, "thinking": "off",
                     "diagnostic": set(args.engine) != set(LOGS) or args.sessions != (6 if args.fast else 3)},
        "engines": {}, "sessions": [], "status": "started",
    }
    write_json(args.out / "manifest.json", manifest)
    try:
        result = run(args, evidence, manifest)
        manifest["status"] = "failed" if result else "completed"
    except (RunError, KeyError, OSError, ValueError) as exc:
        manifest.update(status="failed", error=str(exc))
        print(exc, file=sys.stderr)
        result = 1
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        result = 130
    finally:
        manifest["ended_utc"] = now()
        manifest["sampling"]["completed_sessions"] = {
            engine: {mode: sum(entry["engine"] == engine and entry["mode"] == mode and entry["status"] == "completed"
                               for entry in manifest["sessions"])
                     for mode in ("single", "nodraft")}
            for engine in args.engine
        }
        write_json(args.out / "manifest.json", manifest)
    return result


if __name__ == "__main__":
    sys.exit(main())
