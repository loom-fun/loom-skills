#!/usr/bin/env python3
"""Image-server benchmark, request in -> PNG out over HTTP: loom-serve vs Lemonade (stable-diffusion.cpp, thenoise) vs sd-server.

usage: bench_images.py --target LABEL=SPEC [--target ...] --out DIR [--alternate] [--repeats 3]
                       [--steps N] [--cfg-scale G] [--size 1024x1024] [--prompt-file prompts.json]
                       [--prompts fox,sign] [--seeds 0,42] [--loom-serve PATH] [--device amdgpu] [--port 8211]
                       [--lemonade http://localhost:13305]

SPEC is one of
  loom:PKG/model.json              loom-serve on that card; started and stopped by this script (needs --alternate)
  loom-url:URL[@MODEL_ID]          a loom-serve already running; MODEL_ID is the card's model.id (omitted: no model field)
  sdserver:URL                     a stable-diffusion.cpp sd-server started by hand (e.g. a GGUF quant with --diffusion-fa)
  lemonade:MODEL:RECIPE:BACKEND    a Lemonade model, e.g. Flux-2-Klein-4B:sd-cpp:rocm or Flux-2-Klein-4B-TheNoise:thenoise:rocm
The first target is the baseline the ratios are reported against.

Modes:
  --alternate  block-alternating sessions, the mode for any row you publish. One model is resident at a time: per repeat,
               targets run in a rotated order; each block brings its target up (starts loom-serve / Lemonade /v1/load),
               warms up once, times every case once, then takes it down (stops loom-serve / Lemonade /v1/unload).
               A resident image server holds tens of GB, and a second resident model in a shared GPU carve-out
               spills to shared memory and slows both. Rotation spreads thermal drift over every target.
  default      every target already resident; each case visits all targets in a rotating order. Only valid when
               all targets fit in device memory together.
Cases are ordered seed-major, prompt-minor so consecutive requests to one target never share a prompt: thenoise
answers a repeat of its previous request from a stage cache. Send the same steps and cfg everywhere: loom-serve takes
both from the card and rejects unknown fields, Lemonade takes them per request (sd-server under
sd_cpp_extra_args.sample_params), and a registry default can differ from the checkpoint's.

Output: OUT/<label>/<prompt>_s<seed>_<WxH>/image.png (first timed result) and meta.json (every timing, median, min,
whether repeats were bit-identical, server info), the layout metrics.py reads. Prints, per target, the median over
cases of each case's median, and the median per-case ratio to the baseline. Standard library only.
"""
import argparse
import base64
import hashlib
import json
import platform
import shutil
import statistics
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


class Target:
    def __init__(self, spec, args):
        self.label, rest = spec.split("=", 1)
        self.kind, rest = rest.split(":", 1)
        self.args, self.proc, self.log = args, None, None
        if self.kind == "loom":
            self.package = Path(rest)
            card = json.loads(self.package.read_text(encoding="utf-8"))
            self.model, self.url = card["model"]["id"], f"http://127.0.0.1:{args.port}"
        elif self.kind == "loom-url":
            url, _, model = rest.partition("@")
            self.url, self.model = url.rstrip("/"), model or None
        elif self.kind == "sdserver":
            self.url, self.model = rest.rstrip("/"), "sd-server"
        elif self.kind == "lemonade":
            self.model, self.recipe, self.backend = rest.split(":")
            self.url = args.lemonade.rstrip("/")
        else:
            raise SystemExit(f"unknown target kind {self.kind!r} in {spec!r}")

    def body(self, prompt, seed, size):
        a = self.args
        body = {"prompt": prompt, "size": size, "seed": seed, "response_format": "b64_json"}
        if self.model:
            body["model"] = self.model
        if self.kind == "lemonade":
            body.update(n=1, steps=a.steps, cfg_scale=a.cfg_scale)
        elif self.kind == "sdserver":
            # sd-server reads sampler settings from a tag in the prompt, nested as Lemonade forwards them;
            # flat sample_steps / cfg_scale fields are ignored.
            extra = {"seed": seed, "sample_params": {"sample_steps": a.steps, "guidance": {"txt_cfg": a.cfg_scale}}}
            body["prompt"] = f"{prompt}<sd_cpp_extra_args>{json.dumps(extra)}</sd_cpp_extra_args>"
        return body

    def generate(self, prompt, seed, size):
        t = time.perf_counter()
        resp = http("POST", f"{self.url}/v1/images/generations", self.body(prompt, seed, size))
        elapsed = time.perf_counter() - t
        return base64.b64decode(resp["data"][0]["b64_json"]), elapsed

    def up(self):
        if self.kind == "lemonade":
            http("POST", f"{self.url}/v1/load", {"model_name": self.model, "steps": self.args.steps,
                                                 "cfg_scale": self.args.cfg_scale, f"{self.recipe}_backend": self.backend})
        elif self.kind == "loom":
            self.log = Path(self.args.out) / f"{self.label}_serve.log"
            self.log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [self.args.loom_serve, f"--device={self.args.device}", f"--model={self.package}",
                   f"--listen=127.0.0.1:{self.args.port}", "--heartbeat_ms=0"]
            self.proc = subprocess.Popen(cmd, stdout=self.log.open("w"), stderr=subprocess.STDOUT)
            deadline = time.time() + self.args.ready_timeout
            while '"ready"' not in self.log.read_text(errors="replace"):
                if self.proc.poll() is not None or time.time() > deadline:
                    self.down()
                    raise SystemExit(f"{self.label}: loom-serve did not become ready, see {self.log}")
                time.sleep(2)

    def down(self):
        if self.kind == "lemonade":
            try:
                http("POST", f"{self.url}/v1/unload", {})
            except SystemExit:
                pass
        elif self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
            time.sleep(self.args.cooldown)

    def info(self):
        if self.kind == "lemonade":
            health = http("GET", f"{self.url}/v1/health")
            loaded = [m for m in health.get("all_models_loaded", []) if m.get("model_name") == self.model]
            keep = ("model_name", "checkpoint", "device", "recipe", "recipe_options", "launch_command")
            return {"lemonade_version": health.get("version"), "loaded": {k: loaded[0].get(k) for k in keep} if loaded else None}
        if self.kind in ("loom", "loom-url"):
            return {"loom_status": http("GET", f"{self.url}/v1/status")}
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
    args = ap.parse_args()

    targets = [Target(t, args) for t in args.target]
    if not args.alternate and any(t.kind == "loom" for t in targets):
        ap.error("loom:PKG targets are started by the script and need --alternate; use loom-url:URL for a running server")
    cfg = json.loads(Path(args.prompt_file).read_text(encoding="utf-8"))
    prompts = cfg["prompts"]
    if args.prompts:
        keep = set(args.prompts.split(","))
        prompts = [q for q in prompts if q["id"] in keep]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["seeds"]
    if len(prompts) < 2:
        print("warning: one prompt means consecutive requests share it; thenoise answers those from its stage cache")
    cases = [(q, s) for s in seeds for q in prompts]
    width, height = (int(v) for v in args.size.lower().split("x"))
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
                    info.setdefault(t.label, t.info())
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
            info[t.label] = t.info()
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
                "steps": "card" if t.kind.startswith("loom") else args.steps,
                "cfg_scale": "card" if t.kind.startswith("loom") else args.cfg_scale,
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
