#!/usr/bin/env python3
"""Image-server benchmark, request in -> PNG out over HTTP: loom-serve vs Lemonade (stable-diffusion.cpp, thenoise) vs sd-server.

usage: bench_images.py --target LABEL=SPEC [--target ...] --out DIR [--alternate] [--repeats 3]
                       [--steps N] [--cfg-scale G] [--size 1024x1024] [--prompt-file prompts.json]
                       [--prompts fox,sign] [--seeds 0,42] [--loom-serve PATH] [--device amdgpu] [--port 8211]
                       [--lemonade http://localhost:13305] [--loom-card LABEL=PATH] [--loom-guidance LABEL=G]
                       [--thenoise-cfg-free LABEL]

SPEC is one of
  loom:PKG/model.json              loom-serve on that card; started and stopped by this script (needs --alternate)
  loom-url:URL[@MODEL_ID]          a loom-serve already running; supply its served document via --loom-card
  sdserver:URL                     a stable-diffusion.cpp sd-server started by hand (e.g. a GGUF quant with --diffusion-fa)
  lemonade:MODEL:RECIPE:BACKEND    a Lemonade model, e.g. Flux-2-Klein-4B:sd-cpp:rocm or Flux-2-Klein-4B-TheNoise:thenoise:rocm
The first target is the baseline the ratios are reported against.

Modes:
  --alternate  block-alternating sessions, the mode for any row you publish. One model is resident at a time: per repeat,
               targets run in a rotated order; each block brings its target up (starts loom-serve / Lemonade /v1/load),
               warms up once, times every case once, then takes it down (stops loom-serve / Lemonade /v1/unload).
               A resident image server holds tens of GB, and a second resident model in a shared GPU carve-out
               spills to shared memory and slows both. Rotation spreads thermal drift over every target.
               Multiple targets require script-managed Loom or Lemonade servers.
  default      every target already resident; each case visits all targets in a rotating order. Only valid when
               all targets fit in device memory together.
Cases are ordered seed-major, prompt-minor so consecutive requests to one target never share a prompt: thenoise
answers a repeat of its previous request from a stage cache. Loom takes steps and guidance from the deployed card.
Supply --loom-guidance LABEL=G from the pinned program or card documentation when the card omits guidance metadata.
Lemonade takes steps and cfg_scale per request. A direct sd-server accepts them under sd_cpp_extra_args.sample_params.

Output: OUT/<label>/<prompt>_s<seed>_<WxH>/image.png (first timed result) and meta.json (every timing, median, min,
whether repeats were bit-identical, server info), the layout metrics.py reads. Prints, per target, the median over
cases of each case's median, and the median per-case ratio to the baseline. Standard library only.
"""
import argparse
import base64
import hashlib
import json
import math
import platform
import re
import shutil
import statistics
import struct
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

WARMUP_PROMPT = "a plain grey test card"


def http(method, url, body=None, timeout=1800):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {url} -> {e.code}: {e.read().decode(errors='replace')[:2000]}")


def loom_settings(package, guidance):
    manifest = package.read_bytes()
    card = json.loads(manifest)
    services = [s for s in card["services"].values() if s["kind"] == "image"]
    if len(services) != 1:
        raise ValueError(f"{package}: expected exactly one image service")
    interface = services[0]["interface"]
    steps = interface.get("denoising_steps")
    fixed_guidance = interface.get("guidance_scale")
    if guidance is not None and fixed_guidance is not None and guidance != fixed_guidance:
        raise ValueError(f"{package}: --loom-guidance differs from the card's guidance")
    cfg = fixed_guidance if fixed_guidance is not None else guidance
    if not isinstance(steps, int) or steps <= 0 or cfg is None or not math.isfinite(cfg):
        raise ValueError(f"{package}: expected card denoising_steps and an explicit effective guidance scale")
    return card["model"]["id"], steps, cfg, hashlib.sha256(manifest).hexdigest()


class Target:
    def __init__(self, spec, args):
        self.label, rest = spec.split("=", 1)
        self.kind, rest = rest.split(":", 1)
        self.args, self.proc, self.log = args, None, None
        self.steps, self.cfg_scale, self.card_sha256 = args.steps, args.cfg_scale, None
        if self.kind == "loom":
            self.package = Path(rest)
            self.url = f"http://127.0.0.1:{args.port}"
        elif self.kind == "loom-url":
            url, _, model = rest.rpartition("@") if "@" in rest else (rest, "", "")
            self.url = url.rstrip("/")
            if self.label not in args.loom_card:
                raise ValueError(f"{self.label}: provide the deployed image card via --loom-card")
            self.package = Path(args.loom_card[self.label])
        elif self.kind == "sdserver":
            self.url, self.model = rest.rstrip("/"), "sd-server"
        elif self.kind == "lemonade":
            self.model, self.recipe, self.backend = rest.split(":")
            self.url = args.lemonade.rstrip("/")
            if self.recipe == "thenoise" and self.cfg_scale == 0:
                if self.label not in args.thenoise_cfg_free:
                    raise ValueError(f"{self.label}: Lemonade omits zero guidance for TheNoise. "
                                     "Use --thenoise-cfg-free LABEL only for a checkpoint where 1.0 disables CFG.")
                self.cfg_scale = 1.0
        else:
            raise ValueError(f"unknown target kind {self.kind!r} in {spec!r}")
        if self.kind in ("loom", "loom-url"):
            self.model, self.steps, self.cfg_scale, self.card_sha256 = loom_settings(
                self.package, args.loom_guidance.get(self.label))
            if self.kind == "loom-url" and model and model != self.model:
                raise ValueError(f"{self.label}: requested model differs from {self.package}")
            if (self.steps, self.cfg_scale) != (args.steps, args.cfg_scale):
                raise ValueError(f"{self.label}: deployed steps/guidance {self.steps}/{self.cfg_scale} "
                                 f"differ from requested {args.steps}/{args.cfg_scale}")

    def body(self, prompt, seed, size):
        body = {"prompt": prompt, "size": size, "seed": seed, "response_format": "b64_json"}
        if self.model:
            body["model"] = self.model
        if self.kind == "lemonade":
            body.update(n=1, steps=self.steps, cfg_scale=self.cfg_scale)
        elif self.kind == "sdserver":
            # sd-server reads sampler settings from a tag in the prompt, nested as Lemonade forwards them;
            # flat sample_steps / cfg_scale fields are ignored.
            extra = {"seed": seed, "sample_params": {"sample_steps": self.steps,
                                                      "guidance": {"txt_cfg": self.cfg_scale}}}
            body["prompt"] = f"{prompt}<sd_cpp_extra_args>{json.dumps(extra)}</sd_cpp_extra_args>"
        return body

    def generate(self, prompt, seed, size):
        t = time.perf_counter()
        resp = http("POST", f"{self.url}/v1/images/generations", self.body(prompt, seed, size))
        png = base64.b64decode(resp["data"][0]["b64_json"], validate=True)
        expected = tuple(int(dim) for dim in size.split("x"))
        if (len(png) < 24 or png[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
                or struct.unpack(">II", png[16:24]) != expected):
            raise ValueError(f"{self.label}: expected a {size} PNG")
        return png, time.perf_counter() - t

    def up(self):
        if self.kind == "lemonade":
            http("POST", f"{self.url}/v1/load", {"model_name": self.model, "steps": self.steps,
                                                 "cfg_scale": self.cfg_scale, f"{self.recipe}_backend": self.backend,
                                                 "save_options": False})
        elif self.kind == "loom":
            self.log = Path(self.args.out) / f"{self.label}_serve.log"
            self.log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [self.args.loom_serve, f"--device={self.args.device}", f"--model={self.package}",
                   f"--listen=127.0.0.1:{self.args.port}"]
            with self.log.open("a") as output:
                self.proc = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT)
            try:
                self.wait_ready()
            except BaseException:
                self.down()
                raise

    def wait_ready(self):
        deadline = time.monotonic() + self.args.ready_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                break
            try:
                if self.info()["registration"].get("registration") == "ready":
                    return
            except (urllib.error.URLError, TimeoutError, LookupError):
                pass
            time.sleep(1)
        raise SystemExit(f"{self.label}: loom-serve did not become ready, see {self.log}")

    def down(self):
        if self.kind == "lemonade":
            http("POST", f"{self.url}/v1/unload", {"model_name": self.model})
        elif self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
            time.sleep(self.args.cooldown)

    def info(self):
        if self.kind == "lemonade":
            health = http("GET", f"{self.url}/v1/health")
            loaded = [m for m in health.get("all_models_loaded", []) if m.get("model_name") == self.model]
            keep = ("model_name", "checkpoint", "device", "recipe", "recipe_options", "launch_command")
            if len(loaded) != 1 or loaded[0].get("recipe") != self.recipe:
                raise ValueError(f"{self.label}: Lemonade did not load the requested recipe")
            options = loaded[0].get("recipe_options", {})
            for key, value in {"steps": self.steps, "cfg_scale": self.cfg_scale,
                               f"{self.recipe}_backend": self.backend}.items():
                if options.get(key) != value:
                    raise ValueError(f"{self.label}: Lemonade reported {key}={options.get(key)!r}, requested {value!r}")
            return {"lemonade_version": health.get("version"), "loaded": {k: loaded[0].get(k) for k in keep}}
        if self.kind in ("loom", "loom-url"):
            status = http("GET", f"{self.url}/v1/status", timeout=3)
            record = next((item for item in status.get("data", []) if item.get("id") == self.model), None)
            if record is None:
                raise LookupError(f"{self.label}: missing registration for {self.model}")
            registration = record.get("loom", {})
            if registration.get("registration") == "failed":
                raise ValueError(f"{self.label}: registration failed: {record}")
            return {"loom_status": status, "registration": registration}
        return {}


def default_loom_serve():
    venv = Path.home() / "loom" / ".venv"
    for p in (venv / "Scripts" / "loom-serve.exe", venv / "bin" / "loom-serve"):
        if p.exists():
            return str(p)
    return shutil.which("loom-serve") or "loom-serve"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alternate", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--steps", type=int, required=True, help="the checkpoint's step count; loom-serve takes it from the card")
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--prompt-file", default=str(Path(__file__).resolve().parent.parent / "reference/prompts.json"))
    ap.add_argument("--prompts", help="comma-separated prompt ids (default: all)")
    ap.add_argument("--seeds", help="comma-separated seeds (default: the prompt file's)")
    ap.add_argument("--loom-serve", default=default_loom_serve())
    ap.add_argument("--device", default="amdgpu")
    ap.add_argument("--port", type=int, default=8211)
    ap.add_argument("--ready-timeout", type=float, default=600)
    ap.add_argument("--cooldown", type=float, default=8, help="seconds after stopping loom-serve before the next block")
    ap.add_argument("--lemonade", default="http://localhost:13305")
    ap.add_argument("--loom-card", action="append", default=[], metavar="LABEL=PATH")
    ap.add_argument("--loom-guidance", action="append", default=[], metavar="LABEL=G")
    ap.add_argument("--thenoise-cfg-free", action="append", default=[], metavar="LABEL")
    args = ap.parse_args()

    try:
        args.loom_card = dict(item.split("=", 1) for item in args.loom_card)
        args.loom_guidance = {label: float(value) for label, value in
                              (item.split("=", 1) for item in args.loom_guidance)}
        targets = [Target(t, args) for t in args.target]
        if set(args.loom_card) - {t.label for t in targets if t.kind == "loom-url"}:
            raise ValueError("--loom-card labels must name loom-url targets")
        if set(args.loom_guidance) - {t.label for t in targets if t.kind in ("loom", "loom-url")}:
            raise ValueError("--loom-guidance labels must name Loom targets")
        if set(args.thenoise_cfg_free) - {t.label for t in targets if getattr(t, "recipe", None) == "thenoise"}:
            raise ValueError("--thenoise-cfg-free labels must name TheNoise targets")
        if args.thenoise_cfg_free and args.cfg_scale != 0:
            raise ValueError("--thenoise-cfg-free requires --cfg-scale 0 for the other targets")
        if len({t.label.lower() for t in targets}) != len(targets):
            raise ValueError("target labels must be unique")
        if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", t.label) for t in targets):
            raise ValueError("target labels must start with a letter or digit and use only letters, digits, '-' or '_'")
        output = Path(args.out)
        for target in targets:
            owned_paths = (output / target.label, output / f"{target.label}_serve.log")
            if any(path.exists() or path.is_symlink() for path in owned_paths):
                raise ValueError(f"{target.label}: use a new benchmark output path for this label")
        if args.steps < 1 or args.repeats < 1 or not math.isfinite(args.cfg_scale):
            raise ValueError("provide positive steps and repeats and finite guidance")
    except (KeyError, OSError, TypeError, ValueError) as error:
        ap.error(str(error))
    if not args.alternate and any(t.kind == "loom" for t in targets):
        ap.error("loom:PKG targets are started by the script and need --alternate; use loom-url:URL for a running server")
    if args.alternate and len(targets) > 1 and any(t.kind in ("loom-url", "sdserver") for t in targets):
        ap.error("--alternate with multiple targets requires the script to manage every target's residency")
    cfg = json.loads(Path(args.prompt_file).read_text(encoding="utf-8"))
    prompts = cfg["prompts"]
    if args.prompts:
        keep = set(args.prompts.split(","))
        prompts = [q for q in prompts if q["id"] in keep]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["seeds"]
    if not prompts or not seeds:
        ap.error("select at least one prompt and seed")
    prompt_ids = [q["id"] for q in prompts]
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", item) for item in prompt_ids):
        ap.error("prompt IDs must start with a letter or digit and use only letters, digits, '-' or '_'")
    if len({item.lower() for item in prompt_ids}) != len(prompt_ids) or len(set(seeds)) != len(seeds):
        ap.error("provide unique prompt IDs and seeds")
    if len(prompts) < 2:
        print("warning: one prompt means consecutive requests share it; thenoise answers those from its stage cache")
    cases = [(q, s) for s in seeds for q in prompts]
    width, height = (int(v) for v in args.size.lower().split("x"))
    if width <= 0 or height <= 0:
        ap.error("--size requires positive dimensions")
    args.size = f"{width}x{height}"
    runs = {(t.label, i): [] for t in targets for i in range(len(cases))}
    digests = {k: set() for k in runs}
    first, info, warm = {}, {}, {}

    def timed(t, i):
        q, seed = cases[i]
        png, elapsed = t.generate(q["text"], seed, args.size)
        runs[(t.label, i)].append(elapsed)
        digests[(t.label, i)].add(hashlib.sha256(png).hexdigest())
        first.setdefault((t.label, i), png)
        return elapsed

    k = len(targets)
    if args.alternate:
        for rep in range(args.repeats):
            for j in range(k):
                t = targets[(rep + j) % k]
                t.up()
                try:
                    info.setdefault(t.label, []).append(t.info())
                    _, w = t.generate(WARMUP_PROMPT, 0, args.size)
                    warm.setdefault(t.label, []).append(w)
                    times = [timed(t, i) for i in range(len(cases))]
                finally:
                    t.down()
                print(f"repeat {rep + 1}/{args.repeats} {t.label}: warm-up {w:.2f}s (excluded), "
                      f"median {statistics.median(times):.3f}s over {len(times)} cases", flush=True)
    else:
        for t in targets:
            t.up()
            info[t.label] = [t.info()]
            _, w = t.generate(WARMUP_PROMPT, 0, args.size)
            warm[t.label] = [w]
        n = 0
        for rep in range(args.repeats):
            for i, (q, seed) in enumerate(cases):
                line = " ".join(f"{t.label}={timed(t, i):.2f}" for t in (targets[(n + j) % k] for j in range(k)))
                n += 1
                print(f"pass {rep + 1}/{args.repeats} {q['id']}_s{seed}: {line}", flush=True)

    order = (f"block-alternating across {[t.label for t in targets]}, rotating start, one resident model"
             if args.alternate else f"interleaved per case across {[t.label for t in targets]}, rotating start")
    base = targets[0].label
    for t in targets:
        for i, (q, seed) in enumerate(cases):
            case = Path(args.out) / t.label / f"{q['id']}_s{seed}_{width}x{height}"
            case.mkdir(parents=True, exist_ok=True)
            r = runs[(t.label, i)]
            (case / "image.png").write_bytes(first[(t.label, i)])
            (case / "meta.json").write_text(json.dumps({
                "label": t.label, "kind": t.kind, "url": t.url, "model": t.model,
                "recipe": getattr(t, "recipe", None), "backend": getattr(t, "backend", None),
                "prompt_id": q["id"], "prompt": q["text"], "seed": seed, "width": width, "height": height,
                "steps": t.steps, "cfg_scale": t.cfg_scale, "comparison_cfg_scale": args.cfg_scale,
                "loom_card_sha256": t.card_sha256,
                "seconds": {"runs": r, "median": statistics.median(r), "min": min(r), "warmup": warm.get(t.label)},
                "order": order, "deterministic_across_repeats": len(digests[(t.label, i)]) == 1,
                "server_info": info.get(t.label), "host": platform.node(),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }, indent=2), encoding="utf-8")
        med = [statistics.median(runs[(t.label, i)]) for i in range(len(cases))]
        ratio = [statistics.median(runs[(t.label, i)]) / statistics.median(runs[(base, i)]) for i in range(len(cases))]
        nondet = sum(len(digests[(t.label, i)]) > 1 for i in range(len(cases)))
        print(f"{t.label}: median {statistics.median(med):.3f}s mean {statistics.mean(med):.3f}s "
              f"ratio_to_{base} {statistics.median(ratio):.4f} non-deterministic cases {nondet}/{len(cases)}")


if __name__ == "__main__":
    main()
