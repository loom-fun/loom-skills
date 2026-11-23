"""Reuse one exact SDK 0.3 fleet correctness artifact in a performance confirmation."""

import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


DEVLAB_HEAD = "3adfe7b7b6e68f753042e9f89b318e29a6088579"
SOURCE_RUN = 45275710722
SOURCE_HEAD = "577b3ed2f0d8cf7d5e6ede2085fd2949c0419757"
STAGING = ("staging", "stagingnd")
FULL_STEPS = ("func:staging", "ktest:staging", "gates:staging:full")
SMOKE_STEPS = ("smoke:fun", "smoke:staging")
WHEEL = "6252bb0df262351175bdb2c4aaca22ffa63d80d791659aa4d2fa5c92a0af426f"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gh_download(url, path):
    config = f'header = "Authorization: Bearer {os.environ["GH_TOKEN"]}"\n'
    result = subprocess.run(["curl", "-fsSL", "--retry", "2", "--max-time", "120", "--config", "-",
                             url, "--output", str(path)], input=config, text=True, capture_output=True)
    if result.returncode:
        raise ValueError("Could not read the immutable GitHub qualification artifact")


def job(plan):
    (item,) = plan["jobs"]
    return item


def stable_job(plan):
    item = job(plan)
    return {
        "runner": plan["runner"],
        "memory": plan.get("memory_bytes"),
        "job": {key: value for key, value in item.items() if key not in {"steps", "variants"}},
        "variants": {row["name"]: row for row in item["variants"] if row["name"] in STAGING},
    }


def reconstruct_launch(plan, out):
    """Derive launch arguments for the pinned Radeon harness's standard 0.3.0 path."""
    identity = stable_job(plan)
    item, variants = identity["job"], identity["variants"]
    mem = item.get("memory_bytes_r9700") or plan.get("memory_bytes_r9700") or min(
        item.get("memory_bytes") or identity["memory"], 34_000_000_000
    )
    assert isinstance(mem, int), "Cannot reconstruct the initial memory bound"
    expected = {}
    for variant in STAGING:
        settings = variants[variant]
        assert not any(settings.get(key) for key in ("serve_args", "memory_bytes", "elastic"))
        assert settings.get("active_wait") is not False
        model = read(out / f"model_{variant}.json")
        first, *other = model["presets"]
        specific = first + "-r9700"
        preset = specific if specific in other else first if other else None
        argv = ["--device=amdgpu://?completion_spin_ns=4294967295", "--listen=<loopback>",
                "--workers=8", "--model=<package>", "--pool_backing=fixed"]
        if preset:
            argv.append(f"--preset={preset}")
        expected[variant] = {
            "argv": argv,
            "env": {"LOOM_ACTIVE_WAIT": "1", "LOOM_MEMNEED": str(mem), "LOOM_MEMORY_CAP_GIB": "34"},
        }
    return expected


def full_checks(out):
    summary = read(out / "summary.json")["results"]
    assert read(out / "func_staging.json")["verdict"] == "pass"
    assert summary["ktest:staging"].startswith("PASS (")
    assert "FAIL" not in (out / "ktest_staging.log").read_text()
    gates = read(out / "gates_staging/gates.json")
    assert (gates["verdict"], gates["level"]) == ("PASS", "full")


def seconds(out, steps):
    rows = [line.split("\t") for line in (out / "step_times.tsv").read_text().splitlines()]
    return round(sum(float(end) - float(start) for step, start, end in rows if step in steps), 2)


def package_config(value):
    if isinstance(value, dict):
        return {key: package_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [package_config(item) for item in value]
    if isinstance(value, str) and "/_card/" in value:
        return value[value.index("/_card/"):]
    return value


def initial_package(root):
    with tarfile.open(root / "package-without-weights.tar.gz") as package:
        config = package.extractfile("./llm_config.json")
        assert config, "Missing runtime package configuration"
        result = json.load(config)
        weights = package.extractfile("./weights.loom")
        assert weights, "Missing weight program"
        return package_config(result), hashlib.sha256(weights.read()).hexdigest()


def prepare(work, proof, checkout):
    source = work / "initial-correctness"
    source.mkdir()
    receipt = work / "receipt"
    try:
        assert proof["run"] == SOURCE_RUN and proof["generator"] == DEVLAB_HEAD
        assert re.fullmatch(r"sha256:[\da-f]{64}", proof["digest"])
        assert all(isinstance(proof[x], int) for x in ("artifact", "job"))
        repository = os.environ["GITHUB_REPOSITORY"]
        assert repository == "loom-fun/loom-skills"
        api = f"https://api.github.com/repos/{repository}/actions"
        url = f"{api}/artifacts/{proof['artifact']}"
        gh_download(url, source / "artifact.json")
        artifact = read(source / "artifact.json")
        assert artifact["id"] == proof["artifact"] and artifact["digest"] == proof["digest"] and not artifact["expired"]
        assert artifact["workflow_run"]["id"] == SOURCE_RUN and artifact["workflow_run"]["head_sha"] == SOURCE_HEAD
        assert artifact["name"] == f"sdk030-radeon-r9700-{os.environ['SLUG']}"
        gh_download(f"{api}/jobs/{proof['job']}", source / "job.json")
        run_job = read(source / "job.json")
        assert run_job["run_id"] == SOURCE_RUN and run_job["id"] == proof["job"] and run_job["conclusion"] == "success"
        assert run_job["name"] == f"radeon-r9700 {os.environ['SLUG']}"
        assert "radeon-r9700" in run_job["labels"]
        gh_download(f"{url}/zip", source / "artifact.zip")
        assert "sha256:" + sha(source / "artifact.zip") == proof["digest"]
        with zipfile.ZipFile(source / "artifact.zip") as archive:
            assert all(not PurePosixPath(path).is_absolute() and ".." not in PurePosixPath(path).parts for path in archive.namelist())
            archive.extractall(source)
        original = read(source / "experiment.json")
        candidate = read(receipt / "experiment.json")
        assert stable_job(original) == stable_job(candidate), "The planned launch settings changed"
        out = source / "out"
        full_checks(out)
        assert read(out / "summary.json")["runner"] == run_job["runner_name"]
        assert "-r9700s-" in run_job["runner_name"]
        config, _ = initial_package(source)
        assert not config.get("serve"), "The initial package used additional launch overrides"
        original_ref = (out / "card_revision.txt").read_text().strip()
        assert original_ref == job(candidate)["card"] == proof["production"]
        slug, rev = original_ref.split("@")
        assert re.fullmatch(r"loom-fun/[A-Za-z0-9._-]+", slug) and re.fullmatch(r"[\da-f]{40}", rev)
        with urllib.request.urlopen(f"https://huggingface.co/{slug}/resolve/{rev}/model.json.lock", timeout=30) as response:
            (source / "weights.lock").write_bytes(response.read())
        original_launch = reconstruct_launch(original, out)
        for name in STAGING:
            assert read(out / f"sdk_{name}.json")["wheel_sha256"] == WHEEL
        full = receipt / "experiment.json"
        (work / "full-experiment.json").write_bytes(full.read_bytes())
        item = job(candidate)
        item["steps"] = list(SMOKE_STEPS) + [step for step in item["steps"] if step not in FULL_STEPS]
        scripts = checkout / ".github/scripts"
        original_script = (scripts / "devlab_perf.sh").read_text(encoding="utf-8")
        anchor = '  grep -q \'^FAILED\' "$out/serve_$1.out"'
        assert original_script.count(anchor) == 1, "Launch capture hook could not attach to the reviewed harness"
        capture = ('  if [ -n "${SDK030_LAUNCH_PROOF:-}" ]; then\n'
                   '    "$T" "$SDK030_LAUNCH_PROOF" capture "$1" "$LOOM_SERVE_STATE/perf-$1.pid" "$out/launches.jsonl" || return 1\n'
                   '  fi\n')
        (scripts / "sdk030_record_perf.sh").write_text(original_script.replace(anchor, capture + anchor, 1), encoding="utf-8")
        write(work / "reuse-plan.json", {
            **proof, "original_launch": original_launch, "original_launch_source": "reconstructed from pinned harness and immutable inputs",
            "original_correctness_seconds": seconds(out, FULL_STEPS), "original_runner": run_job["runner_name"],
            "production_weight_lock_sha256": sha(source / "weights.lock"),
        })
        write(full, candidate)
    except (AssertionError, OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, tarfile.TarError) as error:
        write(receipt / "correctness-fallback.json", {"reason": str(error) or "Initial evidence is incomplete or differs from the planned job"})


def capture(variant, pid_file, output):
    proc = Path("/proc") / pid_file.read_text().strip()
    argv = [item.decode() for item in (proc / "cmdline").read_bytes().split(b"\0") if item]
    env = {}
    for entry in (proc / "environ").read_bytes().split(b"\0"):
        key, _, value = entry.partition(b"=")
        if key in {b"LOOM_ACTIVE_WAIT", b"LOOM_MEMNEED", b"LOOM_MEMORY_CAP_GIB"}:
            env[key.decode()] = value.decode()
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"variant": variant, "argv": argv, "env": env}) + "\n")


def check_weights(lock, package, model, known):
    entries = {}
    for entry in lock["files"]:
        name = PurePosixPath(entry["origin"]).name
        assert name not in entries, "A weight basename is ambiguous in the frozen lock"
        entries[name] = entry
    referenced = set()
    for variant in model["variants"].values():
        for service in variant["services"].values():
            for parameter in service.get("parameters", {}).values():
                referenced.update(item["path"] for item in parameter.get("files", []))
    assert referenced, "No locked model weights were found"
    for path in referenced:
        relative = PurePosixPath(path)
        assert relative.parts[0] == "parameter" and ".." not in relative.parts
        weight = package.joinpath(*relative.parts).resolve(strict=True)
        entry = entries[relative.name]
        if weight not in known:
            known[weight] = {"size": weight.stat().st_size, "sha256": sha(weight)}
        assert known[weight] == {"size": entry["size"], "sha256": entry["sha256"]}, f"Weight differs from frozen lock: {path}"
    return sorted(referenced)


def validate(work, job_dir, elapsed):
    proof = read(work / "reuse-plan.json")
    source = work / "initial-correctness"
    first, new = source / "out", job_dir / "out"
    assert (new / "card_revision.txt").read_bytes() == (first / "card_revision.txt").read_bytes()
    config, weights = initial_package(source)
    locker = read(source / "weights.lock")
    weighted = {}
    for variant in STAGING:
        for file in (f"programs_sha256_{variant}.txt", f"sdk_{variant}.json", f"model_{variant}.json"):
            assert (new / file).read_bytes() == (first / file).read_bytes(), f"Measured {file} changed"
        package = job_dir / variant / "pkg"
        assert package_config(read(package / "llm_config.json")) == config, "Runtime configuration changed"
        assert sha(package / "weights.loom") == weights, "The program that loads weights changed"
        proof.setdefault("verified_weight_paths", {})[variant] = check_weights(locker, package, read(package / "model.json"), weighted)
    launches = [json.loads(line) for line in (new / "launches.jsonl").read_text().splitlines()]
    assert {entry["variant"] for entry in launches} == {"fun", "funnd", *STAGING}, "Incomplete live launch evidence"
    controls = {}
    for entry in launches:
        argv = entry["argv"]
        assert PurePosixPath(argv[0]).name == "loom-serve", "Unexpected comparator executable"
        normalized = ["--listen=<loopback>" if re.fullmatch(r"--listen=127\.0\.0\.1:\d+", arg)
                      else "--model=<package>" if arg.startswith("--model=") and arg.endswith("/pkg/model.json") else arg
                      for arg in argv[1:]]
        observed = {"argv": normalized, "env": entry["env"]}
        if entry["variant"] in STAGING:
            assert observed == proof["original_launch"][entry["variant"]], "Live launch differs from initial reconstructed launch"
        else:
            assert normalized[:4] == ["--device=amdgpu://?completion_spin_ns=4294967295",
                                      "--listen=<loopback>", "--workers=8", "--model=<package>"]
            assert "LOOM_ACTIVE_WAIT" in entry["env"] and entry["env"]["LOOM_ACTIVE_WAIT"] == "1"
            assert controls.setdefault(entry["variant"], observed) == observed, "Control launch changed between sessions"
    summary = read(new / "summary.json")
    assert "-r9700s-" in summary["runner"], "Comparator used a different device"
    results = summary["results"]
    assert all(results.get(step, "").startswith("SMOKE PASS") for step in SMOKE_STEPS)
    old, current = (read(new / f"text_{variant}.json")["texts"] for variant in ("fun", "staging"))
    assert old and old == current, "Live greedy text did not match"
    write(work / "receipt/correctness-reused.json", {
        **proof, "comparator_runner": summary["runner"], "comparator_seconds": int(elapsed),
        "comparator_smoke_seconds": seconds(new, SMOKE_STEPS), "launch_records": len(launches),
    })


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "prepare":
        prepare(Path(args[0]), json.loads(args[1]), Path(args[2]))
    elif mode == "capture":
        capture(args[0], Path(args[1]), Path(args[2]))
    elif mode == "validate":
        validate(Path(args[0]), Path(args[1]), args[2])
    else:
        raise SystemExit(f"Unknown reuse operation {mode}")
