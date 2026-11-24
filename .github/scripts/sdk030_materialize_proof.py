"""DevLab-only proof for the pinned Loom 0.3.0 wheel and Qwen card."""
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from unittest import mock


WHEELS = {
    "nt": "06afbc88370fde741037ebd38dba91c2184cb4d2fbd5cee337720cd494a06752",
    "posix": "6252bb0df262351175bdb2c4aaca22ffa63d80d791659aa4d2fa5c92a0af426f",
}
CARD = "huggingface:loom-fun-staging/unsloth-Qwen3-0.6B-loom:chat@f3ffda86d7bd1fa98d615f4972a7d3f82345294c"


def executable(directory, name):
    return str(directory / (f"{name}.exe" if os.name == "nt" else name))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def request(url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    with urllib.request.urlopen(urllib.request.Request(url, data, {"Content-Type": "application/json"}), timeout=10) as response:
        return json.load(response)


def serve(bin_dir, package, log):
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    with log.open("wb") as stream:
        child = subprocess.Popen([executable(bin_dir, "loom-serve"), f"--model={package}", "--device=amdgpu",
                                  "--pool_backing=fixed", f"--listen=127.0.0.1:{port}"],
                                 stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline and child.poll() is None:
                try:
                    name = request(url + "/v1/models")["data"][0]["id"]
                    answer = request(url + "/v1/chat/completions", {
                        "model": name, "messages": [{"role": "user", "content": "What is the capital of France? Reply with one word."}],
                        "temperature": 0, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False},
                    })["choices"][0]
                    if answer["message"]["content"]:
                        return {"model": name, "answer": answer["message"]["content"].strip(), "finish": answer["finish_reason"]}
                except (OSError, urllib.error.URLError, IndexError, KeyError):
                    pass
                time.sleep(1)
            raise RuntimeError(f"{package.name} did not serve a chat reply. See {log.name}")
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)


def inspect_links(work):
    from loom.model import fetch, fetch_filesystem

    source = work / "link-source"
    source.write_bytes(b"materialization probe")
    same = work / "same-filesystem"
    assert fetch_filesystem.share_file(source, same) == "hardlink"
    assert os.path.samefile(source, same) and same.stat().st_nlink > 1
    result = {"same_filesystem": "hardlink"}
    if os.name == "nt":
        result["administrator"] = bool(ctypes.windll.shell32.IsUserAnAdmin())
    cross = Path("/dev/shm") if os.name == "posix" else Path(os.environ.get("LOCALAPPDATA", str(work))) / "Temp"
    if cross.is_dir() and cross.stat().st_dev != work.stat().st_dev:
        with tempfile.TemporaryDirectory(prefix="loom-materialize-", dir=cross) as directory:
            destination = Path(directory) / "cross-filesystem"
            try:
                outcome = fetch_filesystem.share_file(source, destination)
            except fetch_filesystem.SharingUnavailable:
                outcome = "unavailable"
            else:
                assert outcome == "symlink" and destination.read_bytes() == source.read_bytes()
            result["cross_filesystem"] = outcome
    else:
        result["cross_filesystem"] = "unavailable for probe"
    denied = work / "links-denied"
    progress = mock.Mock()
    if os.name == "nt" and not result["administrator"]:
        with mock.patch.object(fetch_filesystem.os, "link", side_effect=OSError(errno.EXDEV, "cross-filesystem")):
            shared = fetch._try_share(source, work / "stock-user-link", automatic=True, progress=progress)
        result["restricted_user_after_exdev"] = "readable symlink" if shared else "warn and copy"
    with mock.patch.object(fetch_filesystem.os, "link", side_effect=OSError(errno.EXDEV, "cross-filesystem")), \
            mock.patch.object(fetch_filesystem.os, "symlink", side_effect=OSError(errno.EPERM, "no symlink privilege")):
        assert not fetch._try_share(source, denied, automatic=True, progress=progress)
        assert not denied.exists()
        assert "independent copies" in progress.emit.call_args.kwargs["message"]
        try:
            fetch._try_share(source, denied, automatic=False, progress=progress)
        except fetch_filesystem.SharingUnavailable:
            pass
        else:
            raise AssertionError("explicit share unexpectedly accepted a copy fallback")
    result["both_link_types_denied"] = "automatic copy fallback and warning"
    return result


def main(workspace):
    output = workspace / "materialize-proof"
    output.mkdir(exist_ok=True)
    scratch = workspace.parent
    with tempfile.TemporaryDirectory(prefix="sdk030-materialize-", dir=scratch) as tmp:
        work = Path(tmp)
        bin_dir = work / "venv" / ("Scripts" if os.name == "nt" else "bin")
        python = executable(bin_dir, "python")
        subprocess.run([sys.executable, "-m", "venv", str(bin_dir.parent)], check=True)
        subprocess.run([str(python), "-m", "pip", "download", "-q", "--only-binary=:all:", "--no-deps",
                        "--dest", str(work), "hrx-loom==0.3.0"], check=True)
        wheel, = work.glob("hrx_loom-*.whl")
        assert digest(wheel) == WHEELS[os.name]
        subprocess.run([str(python), "-m", "pip", "install", "-q", str(wheel) + "[hf]"], check=True)
        result = {"wheel": wheel.name, "sha256": digest(wheel), "modes": {}}
        env = dict(os.environ, HF_HOME=str(work / "hf-home"), HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
        for key in ("HF_HUB_CACHE", "HF_ENDPOINT", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            env.pop(key, None)
        errors = []
        for mode in ("auto", "copy"):
            package = work / mode
            options = [] if mode == "auto" else ["--materialize=copy"]
            try:
                with (output / f"{mode}-fetch.log").open("wb") as log:
                    subprocess.run([executable(bin_dir, "loom-fetch"), CARD, f"--output={package}", "--no-input",
                                    f"--cache={work / 'fetch-cache'}", *options], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                weight, = package.rglob("*.gguf")
                programs = sorted(package.rglob("*.loom"))
                assert programs and all(not path.is_symlink() and path.stat().st_nlink == 1 for path in programs)
                result["modes"][mode] = {"weight": weight.relative_to(package).as_posix(), "weight_sha256": digest(weight),
                                          "weight_link": "symlink" if weight.is_symlink() else "hardlink" if weight.stat().st_nlink > 1 else "copy",
                                          "programs": {p.relative_to(package).as_posix(): digest(p) for p in programs},
                                          "model_json_sha256": digest(package / "model.json")}
                result["modes"][mode]["chat"] = serve(bin_dir, package, output / f"{mode}-serve.log")
            except (AssertionError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
                errors.append(f"{mode}: {error}")
        probe = subprocess.run([str(python), __file__, "--link-probes", str(work)], check=True, capture_output=True, text=True)
        result["link_probes"] = json.loads(probe.stdout)
        result["errors"] = errors
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        assert not errors, errors
        auto, copied = (result["modes"][mode] for mode in ("auto", "copy"))
        assert auto["weight_link"] in ("hardlink", "symlink") and copied["weight_link"] == "copy", result
        assert all(auto[k] == copied[k] for k in ("weight_sha256", "model_json_sha256", "programs", "chat")), result
        assert "paris" in auto["chat"]["answer"].lower() and auto["chat"]["finish"] == "stop", result
        (output / "passed.txt").write_text("full fetch and serve parity passed\n", encoding="utf-8")
        print(json.dumps({"auto_weight": auto["weight_link"], "copy_weight": copied["weight_link"],
                          "chat": auto["chat"], "link_probes": result["link_probes"]}))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--link-probes":
        print(json.dumps(inspect_links(Path(sys.argv[2]))))
    elif len(sys.argv) == 3 and sys.argv[1] == "--stock-workspace":
        assert os.name == "nt" and not ctypes.windll.shell32.IsUserAnAdmin(), "expected a non-administrator Windows token"
        main(Path(sys.argv[2]).resolve())
    else:
        main(Path(os.environ["GITHUB_WORKSPACE"]).resolve())
