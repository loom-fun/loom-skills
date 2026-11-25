#!/usr/bin/env python3
"""The SDK comparison of an image package (devlab_perf.sh job "kind": "image", fleet/regen.py --programs production): the frozen package
on the old and on the new SDK, same runner, same request.

usage: image_sdk_compare.py EXPERIMENT.json JOB --old-root ROOT --new-root ROOT --work DIR --out DIR [--port 8161] [--requests 2]

The job: {"name": ..., "kind": "image", "card": "ORG/REPO@REV", "image": {"old_sdk": "0.2.0", "new_sdk": "0.3.0", "programs_sha256": H,
"preset": null}} (preset null = the package's first preset). Each SDK fetches the package with its own loom-fetch (the published model.json at
REV, nothing rebuilt) and must hold programs hashing to H (card_lib.programs_sha256) before anything is served. Sessions alternate ABBA
(old new new old): each serves the package alone, waits for registration, sends REQUESTS timed requests of the preset's own request (its
"request" block with the model id; an upscale service gets a fixed synthetic RGB input at a quarter of the preset's geometry). Every output must be a
PNG of the expected size with visual variation; the RGB bytes of every output of one SDK must be identical (deterministic), and the old and new
SDK's must be identical (pixels_match). Writes OUT/image_result.json, OUT/image_{old,new}.png, OUT/serve_{old,new}_N.log and OUT/summary.json
({"job", "runner", "results": {"image:old": ..., "image:new": ..., "pixels": PASS|FAIL ...}}): the job's verdict for fleet/regen.py collect.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card"))


def upscale_input(service):
    """True when the image service takes a base64 raw RGB input image in the prompt (an upscaler) instead of a text prompt"""
    return "base64" in str(((service or {}).get("interface") or {}).get("input", "")).lower()


def request_body(model, preset):
    """(body, expected (width, height)) of a preset's deterministic request"""
    p = model["presets"][preset]
    ex = p.get("execution") or {}
    body = dict(p.get("request") or {}, model=model["model"]["id"], response_format="b64_json")
    service = (model.get("services") or {}).get(p.get("service"))
    if upscale_input(service):
        w, h = 64, 16   # a 4x upscaler: the output geometry is four times the input's
        raw = bytes((x * 5 + y * 9 + c * 47) % 256 for y in range(h) for x in range(w) for c in range(3))
        body.update(prompt=base64.b64encode(raw).decode(), size=f"{4 * w}x{4 * h}")
        return body, (4 * w, 4 * h)
    body.setdefault("seed", 42)
    return body, (int(ex.get("width") or 1024), int(ex.get("height") or 1024))


def abba(n=2):
    out = []
    for i in range(n):
        out += ["old", "new"] if i % 2 == 0 else ["new", "old"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("experiment")
    ap.add_argument("job")
    ap.add_argument("--old-root", required=True)
    ap.add_argument("--new-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8161)
    ap.add_argument("--requests", type=int, default=2)
    a = ap.parse_args(argv)
    import numpy as np
    import requests
    from PIL import Image
    import card_lib

    e = json.load(open(a.experiment))
    j = next(x for x in e["jobs"] if x["name"] == a.job)
    spec = j["image"]
    repo, _, rev = j["card"].partition("@")
    roots = {"old": a.old_root, "new": a.new_root}
    sdks = {"old": spec["old_sdk"], "new": spec["new_sdk"]}
    base = f"http://127.0.0.1:{a.port}"
    res = {"repo": repo, "revision": rev, "sdk": sdks, "runs": {"old": {"sessions": []}, "new": {"sessions": []}}}
    results = {}
    pkgs, model, preset = {}, None, spec.get("preset")

    def save():
        json.dump(res, open(os.path.join(a.out, "image_result.json"), "w"), indent=1)

    try:
        for v in ("old", "new"):   # fetch: each SDK's own loom-fetch, the published model.json at the pinned revision
            pkg = os.path.join(a.work, f"pkg_{v}")
            shutil.rmtree(pkg, ignore_errors=True)
            m = json.load(open(_download(f"https://huggingface.co/{repo}/resolve/{rev}/model.json", os.path.join(a.work, "model_card.json"))))
            preset = preset or next(iter(m["presets"]))
            env = dict(os.environ, HF_HOME=os.path.join(a.work, "hf-home"))
            with open(os.path.join(a.out, f"fetch_{v}.log"), "w") as log:
                subprocess.run([os.path.join(roots[v], ".venv", "bin", "loom-fetch"), f"https://huggingface.co/{repo}/resolve/{rev}/model.json",
                                f"--preset={preset}", "--no-input", "--materialize=share", "--cache", os.path.join(a.work, "fetch-cache"),
                                "--output", pkg], env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
            got = card_lib.programs_sha256(pkg)
            res["runs"][v]["programs_sha256"] = got
            if got != spec["programs_sha256"]:
                raise RuntimeError(f"{v}: fetched programs {got}, not the expected {spec['programs_sha256']}: nothing served")
            pkgs[v] = pkg
            model = json.load(open(os.path.join(pkg, "model.json")))
        body, size = request_body(model, preset)
        res.update(preset=preset, request_sha256=hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(), size=list(size))
        for n, v in enumerate(abba()):
            sess = {"seconds": [], "rgb_sha256": []}
            res["runs"][v]["sessions"].append(sess)
            log = open(os.path.join(a.out, f"serve_{v}_{n}.log"), "w")
            t0 = time.monotonic()
            server = subprocess.Popen([os.path.join(roots[v], ".venv", "bin", "loom-serve"), "--device=amdgpu", f"--listen=127.0.0.1:{a.port}",
                                       f"--model={pkgs[v]}/model.json"], stdout=log, stderr=subprocess.STDOUT)
            try:
                _wait_ready(requests, base, server, 900)
                sess["registration_s"] = round(time.monotonic() - t0, 2)
                for i in range(a.requests):
                    t = time.monotonic()
                    while True:
                        r = requests.post(base + "/v1/images/generations", json=body, timeout=900)
                        if r.status_code != 503 or time.monotonic() - t > 900:
                            break
                        time.sleep(2)
                    if not r.ok:
                        raise RuntimeError(f"{v} session {n}: HTTP {r.status_code}: {r.text[:400]}")
                    img = Image.open(io.BytesIO(base64.b64decode(r.json()["data"][0]["b64_json"], validate=True)))
                    if img.format != "PNG" or img.size != size:
                        raise RuntimeError(f"{v}: output {img.format} {img.size}, expected PNG {size}")
                    rgb = img.convert("RGB")
                    if np.asarray(rgb).std() <= 1:
                        raise RuntimeError(f"{v}: the output has no visual variation")
                    sess["seconds"].append(round(time.monotonic() - t, 3))
                    sess["rgb_sha256"].append(hashlib.sha256(rgb.tobytes()).hexdigest())
                    png = os.path.join(a.out, f"image_{v}.png")
                    if not os.path.exists(png):
                        rgb.save(png)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
                log.close()
            save()
        for v in ("old", "new"):
            run = res["runs"][v]
            hashes = {h for s in run["sessions"] for h in s["rgb_sha256"]}
            secs = [x for s in run["sessions"] for x in s["seconds"]]
            run.update(deterministic=len(hashes) == 1, median_s=round(statistics.median(secs), 3))
            results[f"image:{v}"] = (f"PASS hrx-loom {sdks[v]} median {run['median_s']} s over {len(secs)} requests"
                                     + ("" if run["deterministic"] else f"; NOT deterministic ({len(hashes)} distinct outputs)"))
        a_ = np.asarray(Image.open(os.path.join(a.out, "image_old.png"))).astype(np.int16)
        b_ = np.asarray(Image.open(os.path.join(a.out, "image_new.png"))).astype(np.int16)
        res.update(pixels_match=bool(np.array_equal(a_, b_)), max_abs_channel_diff=int(np.abs(a_ - b_).max()),
                   programs_match=res["runs"]["old"]["programs_sha256"] == res["runs"]["new"]["programs_sha256"] == spec["programs_sha256"],
                   time_ratio=round(res["runs"]["new"]["median_s"] / res["runs"]["old"]["median_s"], 4))
        ok = res["pixels_match"] and res["programs_match"] and all(res["runs"][v]["deterministic"] for v in ("old", "new"))
        results["pixels"] = ("PASS" if ok else "FAIL") + (f" pixels {'identical' if res['pixels_match'] else 'differ (max channel diff ' + str(res['max_abs_channel_diff']) + ')'}"
                                                          f", new/old time {res['time_ratio']}")
    except Exception as x:   # every failure is the job's verdict (summary.json), never silent
        results.setdefault("pixels", f"FAIL: {type(x).__name__}: {str(x)[:300]}")
        res["error"] = f"{type(x).__name__}: {x}"
    save()
    json.dump({"job": a.job, "runner": open(os.path.join(a.out, "runner.txt")).read().strip() if os.path.exists(os.path.join(a.out, "runner.txt")) else "",
               "results": results}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    for k, v in results.items():
        print(f"RESULT {a.job} {k}: {v}")
    shutil.rmtree(os.path.join(a.work, "hf-home"), ignore_errors=True)
    shutil.rmtree(os.path.join(a.work, "fetch-cache"), ignore_errors=True)
    return 0 if results.get("pixels", "").startswith("PASS") else 1


def _download(url, path):
    import urllib.request
    urllib.request.urlretrieve(url, path)
    return path


def _wait_ready(requests, base, server, timeout):
    """until /v1/status reports every record's registration ready (a failed registration or an exited server raises)"""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if server.poll() is not None:
            raise RuntimeError(f"loom-serve exited {server.returncode}")
        try:
            r = requests.get(base + "/v1/status", timeout=3)
        except requests.ConnectionError:
            time.sleep(1)
            continue
        if r.ok:
            recs = r.json().get("data") or []
            if any((x.get("loom") or {}).get("registration") == "failed" for x in recs):
                raise RuntimeError(str(r.json())[:800])
            if recs and all((x.get("loom") or {}).get("registration") == "ready" for x in recs):
                return
        time.sleep(2)
    raise RuntimeError("registration timed out")


if __name__ == "__main__":
    sys.exit(main())
