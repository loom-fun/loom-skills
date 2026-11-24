"""Compare complete, immutable staging image presets with the stock SDK 0.3 CLI."""

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from PIL import Image, ImageStat
import requests


CASES = (
    ("xinntao-Real-ESRGAN-x4plus-loom", "543db9362057e38a5fb5feb00fb59bc023fec351", "upscale"),
    ("black-forest-labs-FLUX.2-klein-4B-loom", "b54f2586335be9a8139813954002847edf80711e", "image"),
    ("Tongyi-MAI-Z-Image-Turbo-loom", "c99a723ec213d767afae2251bc6a93cbba6157b2", "image"),
)
WHEEL_SHA256 = "6252bb0df262351175bdb2c4aaca22ffa63d80d791659aa4d2fa5c92a0af426f"
ESRGAN_RGB_SHA256 = "dc4487c53c518b0e6a796d03e46dd040cbcb64b732af013ad5ab316a877df93a"


def save(path, record):
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def manifest(package):
    files = {}
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        with path.open("rb") as stream:
            sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        link = "symlink" if path.is_symlink() else "hardlink" if stat.st_nlink > 1 else "copy"
        files[path.relative_to(package).as_posix()] = {"size": stat.st_size, "sha256": sha256, "link": link}
    return files


def generate(sdk, package, preset, output, mode):
    card = json.loads((package / "model.json").read_text(encoding="utf-8"))
    body = dict(card["presets"][preset].get("request") or {}, model=card["model"]["id"], response_format="b64_json")
    expected_size = (1024, 1024)
    if preset == "upscale":
        width, height = 64, 16
        expected_size = (4 * width, 4 * height)
        raw = bytes((x * 5 + y * 9 + c * 47) % 256 for y in range(height) for x in range(width) for c in range(3))
        body.update(prompt=base64.b64encode(raw).decode(), size=f"{4 * width}x{4 * height}")
    else:
        body["seed"] = 42
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{reserve.getsockname()[1]}"
    base = "http://" + address
    start = time.monotonic()
    with (output / f"{mode}-serve.log").open("wb") as log:
        server = subprocess.Popen([sdk / "loom-serve", "--device=amdgpu", f"--listen={address}",
                                   f"--model={package / 'model.json'}"], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = start + 900
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"loom-serve exited {server.returncode}")
                try:
                    response = requests.get(base + "/v1/status", timeout=5)
                except requests.exceptions.ConnectionError:
                    time.sleep(1)
                    continue
                records = (response.json().get("data") or []) if response.ok else []
                states = [r.get("loom", {}).get("registration") for r in records]
                if "failed" in states:
                    raise RuntimeError(f"image registration failed: {response.text[:1200]}")
                if states and all(state == "ready" for state in states):
                    break
                time.sleep(1)
            else:
                raise TimeoutError("image registration timed out")
            started = time.monotonic()
            while True:
                response = requests.post(base + "/v1/images/generations", json=body, timeout=900)
                if response.status_code != 503 or time.monotonic() - started > 900:
                    break
                time.sleep(1)
            if not response.ok:
                raise RuntimeError(f"{response.status_code}: {response.text[:500]}")
            data = base64.b64decode(response.json()["data"][0]["b64_json"], validate=True)
            with Image.open(io.BytesIO(data)) as image:
                assert image.format == "PNG" and image.size == expected_size, (image.format, image.size)
                if preset == "upscale":
                    assert image.mode == "RGBA" and image.getchannel("A").getextrema() == (255, 255)
                rgb = image.convert("RGB")
                assert max(ImageStat.Stat(rgb).stddev) > 1, "image has no visual variation"
                digest = hashlib.sha256(rgb.tobytes()).hexdigest()
                (output / f"{mode}.png").write_bytes(data)
                return {"size": image.size, "rgb_sha256": digest}
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10)


def inspect_case(work, sdk, slug, revision, preset):
    root = work / slug
    output = work / "results" / slug
    output.mkdir()
    env = dict(os.environ, HF_HOME=str(root / "hub"), HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    for key in ("HF_HUB_CACHE", "HF_ENDPOINT", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        env.pop(key, None)
    repo = "loom-fun-staging/" + slug
    source = f"huggingface:{repo}:{preset}@{revision}"
    result = {"repo": repo, "revision": revision, "preset": preset, "source": source, "modes": {}}
    try:
        for mode in ("auto", "copy"):
            package = root / mode
            options = [] if mode == "auto" else ["--materialize=copy"]
            with (output / f"{mode}-fetch.log").open("wb") as log:
                subprocess.run([sdk / "loom-fetch", source, f"--output={package}", "--no-input",
                                f"--cache={root / 'fetch-cache'}", *options], env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=2400)
            result["modes"][mode] = {"manifest": manifest(package)}
            save(output / "result.json", result)
        auto, copy = (result["modes"][mode] for mode in ("auto", "copy"))
        identities = [
            {name: (entry["size"], entry["sha256"]) for name, entry in record["manifest"].items()}
            for record in (auto, copy)
        ]
        assert identities[0] == identities[1], "preset closure differs"
        assert "model.json" in auto["manifest"] and any(name.endswith(".loom") for name in auto["manifest"])
        assert all(entry["link"] == "copy" for entry in copy["manifest"].values())
        assert all(entry["link"] == "copy" for name, entry in auto["manifest"].items() if name.endswith(".loom"))
        weights = [entry for name, entry in auto["manifest"].items() if name.endswith(".safetensors")]
        assert weights and all(entry["link"] in ("hardlink", "symlink") for entry in weights)
        result["weight_bytes"] = sum(entry["size"] for entry in weights)
        result["manifests_match"] = True
        save(output / "result.json", result)
        for mode, record in result["modes"].items():
            record["image"] = generate(sdk, root / mode, preset, output, mode)
        assert auto["image"] == copy["image"], "pixel output differs"
        if preset == "upscale":
            assert auto["image"]["rgb_sha256"] == ESRGAN_RGB_SHA256, "archived independent output changed"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        save(output / "result.json", result)
        shutil.rmtree(root, ignore_errors=True)
    print(slug, result.get("error", "matched manifests and output"), flush=True)
    return result


def main():
    work = Path(os.environ["IMAGE_WORK"])
    sdk = work / "sdk" / ".venv" / "bin"
    out = work / "results"
    settings = json.loads((out / "sdk.json").read_text(encoding="utf-8"))
    assert settings["hrx_loom"] == "0.3.0" and settings["wheel_sha256"] == WHEEL_SHA256
    results = [inspect_case(work, sdk, *case) for case in CASES]
    save(out / "summary.json", results)
    return int(any("error" in row for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
