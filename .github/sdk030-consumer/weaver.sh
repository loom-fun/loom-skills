#!/usr/bin/env bash
set -euo pipefail
exec > >(tee "$WORK/out/weaver.log") 2>&1
cd "$WORK"
revision=7d4525cb34e89d161dc8864a1456d19152ca262a
skills_revision=18d4cc22afc8e5211813a32f073db90d5c42aced
git init -q weaver
git -C weaver fetch -q --depth 1 https://github.com/loom-fun/weaver.git "$revision"
git -C weaver -c advice.detachedHead=false checkout -q FETCH_HEAD
cd weaver
git submodule update --init --depth 1
git -C vendor/loom-skills fetch -q --depth 1 origin "$skills_revision"
git -C vendor/loom-skills -c advice.detachedHead=false checkout -q FETCH_HEAD
git add vendor/loom-skills
python3 - <<'PY'
from pathlib import Path

def update(path, old, new):
    file = Path(path)
    text = file.read_text()
    assert text.count(old) == 1, (path, old, text.count(old))
    file.write_text(text.replace(old, new))

update("pyproject.toml", "hrx-loom[hf]==0.2.0", "hrx-loom[hf]==0.3.0")
update("packaging/versions.env", "HRX_LOOM_VERSION=0.2.0", "HRX_LOOM_VERSION=0.3.0")
update("weaver/templates/model_card.md", "hrx-loom[hf]==0.2.0", "hrx-loom[hf]==0.3.0")
update("weaver/runtime.py", 'SKILL_VAR =', 'LOOM_SDK_VERSION = "0.3.0"\nSKILL_VAR =')
update("weaver/commands/upload.py", "'pipx install \"hrx-loom[hf]==0.2.0\"\\n'",
                                  "f'pipx install \"hrx-loom[hf]=={LOOM_SDK_VERSION}\"\\n'")
update("weaver/commands/upload.py", "from weaver.runtime import Runtime",
                                  "from weaver.runtime import LOOM_SDK_VERSION, Runtime")

runtime_path = Path("weaver/runtime.py")
runtime = runtime_path.read_text()
start = runtime.index('    b = paths.bundle_root()', runtime.index('def ensure_library('))
stop = runtime.index('\n\ndef build_bin(', start)
runtime_path.write_text(runtime[:start] + '''    if lib.exists() and not lib.is_symlink():
        return lib

    current = lib.resolve() if lib.is_symlink() else None
    b = paths.bundle_root()
    bundled = b / "library" if b and (b / "library").is_dir() else None
    if bundled:
        target = bundled
    else:
        home = paths.home().resolve()
        if current and lib.exists():
            cached = current.parent == home and current.name.startswith("library-")
            old_bundle = (current.name == "library" and current.parent.parent == home
                          and (current.parent / "BUNDLE").is_file())
            if not (cached or old_bundle):
                return lib
        rev = catalog_revision()
        if not rev:
            return lib if lib.exists() else None
        target = home / f"library-{rev[:12]}"
        if not target.is_dir():
            url = f"https://codeload.github.com/ROCm/hrx-loom-library/tar.gz/{rev}"
            data = urllib.request.urlopen(url, timeout=120).read()
            tmp = target.with_name(target.name + ".tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                archive.extractall(tmp, filter="data")
            (next(tmp.iterdir())).rename(target)
            shutil.rmtree(tmp, ignore_errors=True)

    if current != target.resolve():
        if lib.is_symlink():
            lib.unlink()
        lib.symlink_to(target)
    return lib
''' + runtime[stop:])

update("weaver/commands/doctor.py", "import os\n", "import importlib.metadata as md\nimport os\n")
update("weaver/commands/doctor.py", "from weaver import jobs, paths, pi, servers",
                                   "from weaver import jobs, paths, pi, runtime, servers")
update("weaver/commands/doctor.py", '    kfd = Path("/dev/kfd")',
       '''    try:
        sdk = md.version("hrx-loom")
    except md.PackageNotFoundError:
        sdk = None
    r.append(("Loom SDK", sdk == runtime.LOOM_SDK_VERSION, sdk or "missing",
              f"reinstall weaver with its pinned hrx-loom {runtime.LOOM_SDK_VERSION}"))
    kfd = Path("/dev/kfd")''')

update("packaging/build_bundle.sh",
       'echo "== pi $PI_VERSION"',
       '''installed_sdk=$("$py" -c 'from importlib.metadata import version; print(version("hrx-loom"))')
[ "$installed_sdk" = "$HRX_LOOM_VERSION" ] || { echo "expected hrx-loom $HRX_LOOM_VERSION, installed $installed_sdk"; exit 1; }
echo "== pi $PI_VERSION"''')
update("README.md", "hrx-loom[hf]BKT 0.2.0;".replace("BKT", chr(96)), "hrx-loom[hf]BKT 0.3.0;".replace("BKT", chr(96)))
update("README.md",
       "- Verified on hrx-loom 0.1.0 (the evidence above); the 0.2.0 upgrade is checked by the unit tests and a serve/chat smoke test (BKTdocs/release-notes.mdBKT). Model packages built for 0.1.0 (card field BKTpresets.*.execution.pool_capacityBKT) do not load on 0.2.0: serve the current loom-fun package or BKTweaver portBKT the GGUF again.".replace("BKT", chr(96)),
       "- Weaver uses hrx-loom 0.3.0. Published Loom 0.2 packages, including Weaver's pinned Qwen3.8 agent card, continue to work through controller ABI 1. See BKTdocs/release-notes.mdBKT for the upgrade details and earlier measurements.".replace("BKT", chr(96)))
update("weaver/config.py", "# the loom-fun repo revision of the 0.2.0 card (promoted 2026-11-17)",
                          "# the published 0.2.0 card uses controller ABI 1, also supported by Loom 0.3.0")
update("weaver/agent_server.py", '''    """A fetched card of this SDK. A package fetched before hrx-loom 0.2.0 declares presets.*.execution.pool_capacity,
    which 0.2.0 rejects: it counts as not installed, so BKTweaver initBKT fetches the current card over it."""'''.replace("BKT", chr(96)),
                               '''    """A fetched compatible card. The Loom 0.1 pool_capacity field requires a fresh fetch."""''')
notes = Path("docs/release-notes.md")
notes.write_text("""## hrx-loom 0.3.0

- Weaver and generated model instructions pin Loom SDK 0.3.0, and the bundle checks the installed wheel version before recording it.
- The installed SDK selects the catalog revision on upgrades, including existing pip environments. BKTweaver doctorBKT reports the installed Loom SDK.
- The pinned Qwen3.8-27B agent card continues to use the published controller ABI 1. Its model revision and earlier performance results remain attributable to the corresponding release.

""".replace("BKT", chr(96)) + notes.read_text())

test = Path("tests/test_core.py")
test.write_text(test.read_text() + '''

def test_ensure_library_updates_a_stale_sdk_catalog(tmp_path, monkeypatch):
    home = tmp_path / "weaver"
    root = home / "loom"
    root.mkdir(parents=True)
    old = home / "library-aaaaaaaaaaaa"
    new = home / "library-bbbbbbbbbbbb"
    old.mkdir()
    new.mkdir()
    (root / "library").symlink_to(old)
    monkeypatch.setattr(runtime.paths, "home", lambda: home)
    monkeypatch.setattr(runtime.paths, "bundle_root", lambda: None)
    monkeypatch.setattr(runtime, "catalog_revision", lambda: "b" * 40)
    assert runtime.ensure_library(root).resolve() == new
    assert old.is_dir()

    custom = tmp_path / "my-library"
    custom.mkdir()
    (root / "library").unlink()
    (root / "library").symlink_to(custom)
    assert runtime.ensure_library(root).resolve() == custom
''')
PY
git rev-parse HEAD > "$WORK/out/weaver.base"
git diff HEAD --check
git diff HEAD --binary --submodule=short > "$WORK/out/weaver.patch"
git diff HEAD --submodule=short -- vendor/loom-skills

python3 -m venv "$WORK/weaver-env"
source "$WORK/weaver-env/bin/activate"
wheel=hrx_loom-0.3.0-py3-none-manylinux_2_39_x86_64.whl
python -m pip download --no-deps --dest "$WORK" hrx-loom==0.3.0
echo "6252bb0df262351175bdb2c4aaca22ffa63d80d791659aa4d2fa5c92a0af426f  $WORK/$wheel" | sha256sum -c -
python -m pip install "$WORK/$wheel" '.[test]'
export WEAVER_HOME="$WORK/weaver-state"
python -m pytest -q
python -m weaver doctor
python - <<'PY'
from weaver.runtime import Runtime, LOOM_SDK_VERSION
import subprocess
rt = Runtime()
version = subprocess.check_output([rt.loom_root / ".venv/bin/loom-serve", "--version"], text=True)
assert f"loom-serve {LOOM_SDK_VERSION} " in version, version
assert rt.library and rt.library.is_dir()
print(version.strip())
PY
packaging/build_bundle.sh "$WORK/weaver-release"
mkdir "$WORK/weaver-unpacked"
tar -C "$WORK/weaver-unpacked" -xzf "$WORK/weaver-release/weaver-linux-x64.tar.gz"
cat "$WORK/weaver-unpacked/BUNDLE"
"$WORK/weaver-unpacked/bin/weaver" --version
"$WORK/weaver-unpacked/python/bin/loom-serve" --version
