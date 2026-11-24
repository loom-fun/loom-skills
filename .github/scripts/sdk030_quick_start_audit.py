"""Resolve staging Quick start recipes at reviewed commits, without acquiring weights."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tempfile
from urllib.request import Request, urlopen

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile


REFERENCE = re.compile(r"huggingface:(loom-fun-staging/[\w.-]+)(?::([\w.-]+))?(?:@([a-f0-9]{40}))?\Z")
REVISION = re.compile(r"[a-f0-9]{40}\Z")
STALE_COPY_NOTE = re.compile(r"--materialize(?:\s*=\s*|\s+)[\"']?copy\b|copies of the fetched package files", re.I)


def option(argv, name):
    for index, arg in enumerate(argv):
        if arg == name:
            return argv[index + 1]
        if arg.startswith(name + "="):
            return arg[len(name) + 1 :]
    return None


def relative_path(value):
    if not value:
        raise ValueError("missing output or model path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in value:
        raise ValueError(f"invalid Quick start path {value!r}")
    return path


def recipes(readme, repo, revision):
    section = re.split(r"^##\s+Quick start\s*$", readme, maxsplit=1, flags=re.I | re.M)
    if len(section) != 2:
        raise ValueError("README has no Quick start section")
    block = re.split(r"^##\s+", section[1], maxsplit=1, flags=re.M)[0]
    rows = []
    for line in block.splitlines():
        if not line.lstrip().startswith(("loom-fetch ", "loom-serve ")):
            continue
        argv = shlex.split(line)
        if argv[0] == "loom-fetch":
            ref = REFERENCE.fullmatch(argv[1])
            if not ref or ref[1] != repo or ref[3] not in (None, revision):
                raise ValueError(f"fetch must target the reviewed staging card: {line.strip()}")
            preset = ref[2] or option(argv, "--preset")
            if not preset:
                raise ValueError(f"fetch must name a preset: {line.strip()}")
            output = relative_path(option(argv, "--output"))
            argv[1] = argv[1].split("@", 1)[0] + "@" + revision
            rows.append({"fetch": line.strip(), "argv": argv, "preset": preset, "output": output})
        else:
            if not rows or "serve" in rows[-1]:
                raise ValueError(f"serve command is missing a corresponding fetch: {line.strip()}")
            output = rows[-1]["output"]
            model = relative_path(option(argv, "--model"))
            if model not in (output, output / "model.json"):
                raise ValueError(f"serve model path {model} does not match fetch output {output}")
            rows[-1]["serve"] = line.strip()
    if not rows or any("serve" not in row for row in rows):
        raise ValueError("each Quick start fetch must have a matching serve command")
    return rows


def check_sources(api, sources):
    groups = defaultdict(set)
    for source in sources:
        kind = source["kind"]
        if kind == "hf":
            groups[source["repository"], source["revision"]].add(source["path"])
        elif kind == "url":
            with urlopen(Request(source["url"], method="HEAD"), timeout=30):
                pass
        elif kind != "inline":
            raise ValueError(f"unexpected remote package source type {kind!r}")
    missing = []
    for (repo, revision), names in sorted(groups.items()):
        paths = api.get_paths_info(repo, sorted(names), revision=revision)
        found = {item.path for item in paths if isinstance(item, RepoFile)}
        missing.extend(f"{repo}@{revision}/{name}" for name in sorted(names - found))
    return missing


def audit(fetch, repo, revision, require_default_materialization):
    result = {"repo": repo, "revision": revision, "recipes": []}
    try:
        api = HfApi()
        card = api.model_info(repo, revision=revision, expand=["sha"])
        if card.id != repo or card.sha != revision:
            raise ValueError("Hub did not resolve the exact staging revision")
        readme = Path(hf_hub_download(repo, "README.md", revision=revision)).read_text(encoding="utf-8-sig")
        if require_default_materialization and STALE_COPY_NOTE.search(readme):
            raise ValueError("README still contains the explicit copy flag or its obsolete explanation")
        commands = recipes(readme, repo, revision)
        sources = {}
        with tempfile.TemporaryDirectory(prefix="loom-quick-start-") as tmp:
            for command in commands:
                outcome = {key: command[key] for key in ("fetch", "serve", "preset")}
                materialize = option(command["argv"], "--materialize")
                # The verified SDK 0.3.0 CLI defaults to auto. Leave the printed arguments unchanged.
                outcome["materialize"] = materialize or "auto"
                outcome["materialize_explicit"] = materialize is not None
                result["recipes"].append(outcome)
                try:
                    if require_default_materialization and materialize is not None:
                        raise ValueError("Quick start should let SDK 0.3.0 select its default materialization mode")
                    proc = subprocess.run(
                        [fetch, *command["argv"][1:], "--json", "--no-input", "--dry-run"],
                        cwd=tmp, capture_output=True, text=True, timeout=90,
                    )
                    if proc.returncode:
                        raise ValueError(f"loom-fetch exited {proc.returncode}: {proc.stderr.strip()[-1800:]}")
                    plan = json.loads(proc.stdout)
                    if plan["preset"] != command["preset"] or not plan["service"] or not plan["files"]:
                        raise ValueError("SDK plan did not select the printed preset and a service with files")
                    if plan["lock"] != "verify" or plan["unverified_parameters"]:
                        raise ValueError(f"unexpected lock {plan['lock']}, unverified parameters: {plan['unverified_parameters']}")
                    if Path(plan["output"]) != Path(tmp) / command["output"]:
                        raise ValueError("SDK resolved a different output path")
                    outcome.update({key: plan[key] for key in ("model", "service", "variant", "partition", "programs")})
                    outcome["files"] = len(plan["files"])
                    for member in plan["files"]:
                        transport = member["transport"]
                        sources[json.dumps(transport, sort_keys=True)] = transport
                except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
                    outcome["error"] = str(error)
        result["unique_sources"] = len(sources)
        result["missing_sources"] = check_sources(api, sources.values())
    except Exception as error:
        result["error"] = str(error)
    result["passed"] = not (result.get("error") or result.get("missing_sources") or
                            any("error" in item for item in result["recipes"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--fetch", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--require-default-materialization", action="store_true")
    args = parser.parse_args()
    heads = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if not heads or any(not REFERENCE.fullmatch(f"huggingface:{repo}") or not REVISION.fullmatch(rev)
                        for repo, rev in heads.items()):
        parser.error("manifest must map staging repositories to full immutable commits")
    with ThreadPoolExecutor(max_workers=4) as pool:
        fetch = str(args.fetch.resolve())
        jobs = [pool.submit(audit, fetch, repo, rev, args.require_default_materialization) for repo, rev in sorted(heads.items())]
        results = [job.result() for job in jobs]
    args.report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for row in results:
        print(f"{'PASS' if row['passed'] else 'FAIL'} {row['repo']}@{row['revision'][:12]}: {len(row['recipes'])} recipes")
        if not row["passed"]:
            print(row.get("error") or row.get("missing_sources") or [item.get("error") for item in row["recipes"]])
    return 0 if all(row["passed"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
