#!/usr/bin/env bash
set -euo pipefail
exec > >(tee "$WORK/out/gufo.log") 2>&1
recipes=$(cd "$(dirname "$0")" && pwd)
cd "$WORK"
cmake --version
ninja --version
python3 --version
clang --version || true
rustc --version || true
command -v nix || true
command -v docker || true
revision=e2c41a720363911c04606ce0f4417e2d55630352
previous=0cd370f2baaa0cfef47cc23007f1a109906f8a52
for commit in "$revision" "$previous"; do
  mkdir "$commit"
  curl -fsSL --retry 3 "https://codeload.github.com/ROCm/hrx-system/tar.gz/$commit" -o "$commit.tgz"
  tar -C "$commit" --strip-components=1 -xzf "$commit.tgz"
done
echo "2bdf75bc99d530054dc7457ec68df0269f7fd23675115faa361702981cc51fa9  $previous.tgz" | sha256sum -c -
checkout() {
  git init -q "$1"
  git -C "$1" fetch -q --depth 1 "https://github.com/$2.git" "$3"
  git -C "$1" -c advice.detachedHead=false checkout -q FETCH_HEAD
}
checkout gufo gufo-org/gufo 60c48924a040edc0f025e8d27e6236fcaecc22fd
checkout extension loom-fun/loom-gufo-extension e98bed995cbc50b09f18c21cd1a1c11cda587113
python3 - "$revision" "$previous" <<'PY'
import base64
import hashlib
import os
from pathlib import Path
import re
import stat
import struct
import sys

revision, previous = sys.argv[1:]
old_nar = "sha256-t/R1Qq00lpeS/elqLIltGJgHstNsWdgQWmpBMS3axkw="

def nar_hash(path):
    digest = hashlib.sha256()
    def write(value):
        if isinstance(value, str):
            value = os.fsencode(value)
        digest.update(struct.pack("<Q", len(value)))
        digest.update(value)
        digest.update(b"\0" * (-len(value) % 8))
    def node(path):
        mode = path.lstat().st_mode
        write("(")
        write("type")
        if stat.S_ISDIR(mode):
            write("directory")
            for child in sorted(path.iterdir(), key=lambda p: os.fsencode(p.name)):
                write("entry")
                write("(")
                write("name")
                write(child.name)
                write("node")
                node(child)
                write(")")
        elif stat.S_ISLNK(mode):
            write("symlink")
            write("target")
            write(os.readlink(path))
        else:
            assert stat.S_ISREG(mode)
            write("regular")
            if mode & stat.S_IXUSR:
                write("executable")
                write("")
            write("contents")
            size = path.stat().st_size
            digest.update(struct.pack("<Q", size))
            with path.open("rb") as src:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    digest.update(chunk)
            digest.update(b"\0" * (-size % 8))
        write(")")
    write("nix-archive-1")
    node(path)
    return "sha256-" + base64.b64encode(digest.digest()).decode()

control = nar_hash(Path(previous))
assert control == old_nar, (control, old_nar)
new_nar = nar_hash(Path(revision))
print(f"Verified source NAR control {control}; v0.3.0 {new_nar}")
archive = hashlib.file_digest(Path(f"{revision}.tgz").open("rb"), "sha256").hexdigest()
print(f"v0.3.0 source archive SHA-256 {archive}")

def edit(path, old, new, expected=None):
    file = Path(path)
    text = file.read_text()
    if expected is not None:
        assert text.count(old) == expected, (path, old, text.count(old))
    else:
        assert old in text, (path, old)
    file.write_text(text.replace(old, new))

gufo_runtime = "gufo/.devops/nix/hrx/loom-runtime.nix"
ext_runtime = "extension/nix/loom-runtime.nix"
for path in (gufo_runtime, ext_runtime):
    edit(path, "0.2.0", "0.3.0")
    edit(path, previous, revision, 1)
    edit(path, old_nar, new_nar, 1)
dep_nar = "sha256-eeC2UsCLc/YnFjla0q2FXIM0uXOLPkjBdrbkxG5HU1Q="
edit(gufo_runtime, "sha256-kUc4b7acjbrnrG6KHIpJPTRkRvZHJKeRX7WnyAUR7ug=", dep_nar, 1)
cmake = "gufo/src/models/loom/runtime.cmake"
edit(cmake, "0.2.0", "0.3.0")
edit(cmake, previous, revision, 1)
edit(cmake, "2bdf75bc99d530054dc7457ec68df0269f7fd23675115faa361702981cc51fa9", archive, 1)
edit(cmake, "pkg_check_modules(GUFO_LOOM_PC QUIET loom-text-embed)",
     "pkg_check_modules(GUFO_LOOM_PC QUIET loom-text-embed=${GUFO_LOOM_HRX_VERSION})", 1)
notice = "gufo/THIRD_PARTY_NOTICES.md"
edit(notice, previous, revision, 2)
edit(notice, "v0.2.0", "v0.3.0", 1)

for path in ("gufo/src/models/loom/embed/loom_text_embed.h",
             "extension/loom-embed/loom_text_embed.h"):
    edit(path, '#define LOOM_TEXT_EMBED_HRX_LOOM "0.2.0"',
         '#define LOOM_TEXT_EMBED_HRX_LOOM "0.3.0"', 1)
edit("gufo/src/models/loom/embed/loom_text_embed.h",
     '// The Loom SDK release (hrx-loom, ROCm/hrx-system) this library is built from:\n'
     '// packages must be built for it (their llm_config.json "sdk": {"hrx_loom":\n'
     '// ...}). ../runtime.cmake (which checks this) and',
     '// The Loom SDK release (hrx-loom, ROCm/hrx-system) this library uses.\n'
     '// loom_package.hpp defines the compatible package SDK versions.\n'
     '// ../runtime.cmake (which checks this) and', 1)
edit("extension/loom-embed/loom_text_embed.h",
     '// The Loom SDK release (hrx-loom, ROCm/hrx-system) this library is built from:\n'
     '// packages must be built for it (their llm_config.json "sdk": {"hrx_loom": ...}).',
     '// The Loom SDK release (hrx-loom, ROCm/hrx-system) this library uses.\n'
     '// src/loom_package.hpp defines the compatible package SDK versions.', 1)
for path in ("gufo/src/models/loom/loom_package.hpp", "extension/src/loom_package.hpp"):
    old = "SdkSeries(built) == SdkSeries(runtime)"
    file = Path(path)
    text = file.read_text()
    marker = "// A package built for another Loom SDK release fails inside Loom"
    assert text.count(marker) == 1 and text.count(old) == 1
    helper = '''// SDK 0.3 retains the controller ABI used by the published 0.2 cards.
inline bool SdkCompatible(std::string_view built, std::string_view runtime) {
  const auto built_series = SdkSeries(built);
  const auto runtime_series = SdkSeries(runtime);
  return built_series == runtime_series ||
         (built_series == "0.2" && runtime_series == "0.3");
}

'''
    file.write_text(text.replace(old, "SdkCompatible(built, runtime)").replace(marker, helper + marker))
edit("gufo/src/models/loom/loom_package.hpp",
     "  if (built.empty() || runtime.empty() ||\n      SdkCompatible(built, runtime))",
     "  if (built.empty() || runtime.empty() || SdkCompatible(built, runtime))", 1)
edit("gufo/src/models/loom/loom_source.hpp",
     "// A package for another SDK series fails inside Loom: refuse it here.",
     "// Check that the installed revision is supported by this runtime.", 1)
edit("gufo/src/models/loom/loom_source.hpp",
     "package::SdkSeries(built) != package::SdkSeries(sdk)",
     "!package::SdkCompatible(built, sdk)", 1)
for path in ("gufo/src/models/loom/tests/package_test.cpp", "extension/tests/package_test.cpp"):
    file = Path(path)
    text = file.read_text()
    text, n = re.subn(r'CHECK\(refused\("0\.3\.0"\), "[^"]*"\);',
                      'CHECK(!refused("0.3.0"), "SDK 0.3 still serves published 0.2 packages");', text)
    assert n == 1, (path, n)
    file.write_text(text)
edit("extension/README.md", "pipx install 'hrx-loom[hf]==0.2.0'", "pipx install 'hrx-loom[hf]==0.3.0'", 1)
edit("extension/DESIGN.md", "v0.2.0 (the release loom-fun targets)", "v0.3.0", 1)
edit("extension/DESIGN.md",
     '  default branch, and refuses a revision whose `llm_config.json` targets\n'
     '  another SDK; `--check` verifies the current pins.\n'
     '- At load, a package whose `llm_config.json` names another SDK series (none:\n'
     '  0.1.0) is refused with what to fetch instead, rather than failing inside\n'
     '  Loom.',
     '  default branch, then checks the SDK named in `llm_config.json`.\n'
     '  `--check` verifies the current pins.\n'
     '- The loader and model lock checker accept packages from the runtime SDK\n'
     '  series. Loom 0.3 also accepts published Loom 0.2 packages through\n'
     '  controller ABI 1. A missing SDK pin implies 0.1.0.', 1)
edit("extension/tools/models.py",
     "one), else the default branch; the revision's llm_config.json must say it is\n"
     "built for that SDK. Then it reads its",
     "one), else the default branch. Its llm_config.json must name an SDK the\n"
     "runtime supports. Then it reads its", 1)
edit("extension/tools/models.py",
     "# every pinned package is built for the runtime's SDK",
     "# every pinned package can be used by the runtime", 1)
edit("extension/tools/models.py",
     "    if series(built) != series(sdk):\n"
     '        raise SystemExit(f"{repo}@{revision[:8]} is built for Loom SDK {built}, the runtime is {sdk}: "\n'
     '                         f"it has no v{sdk} tag and its default branch targets another SDK")',
     "    built_series, runtime_series = series(built), series(sdk)\n"
     "    # Keep this rule in step with src/loom_package.hpp.\n"
     '    if built_series != runtime_series and (built_series, runtime_series) != ("0.2", "0.3"):\n'
     '        raise SystemExit(f"{repo}@{revision[:8]} is built for Loom SDK {built}, the runtime is {sdk}")', 1)
edit("extension/tools/models.py",
     "{len(manifest['models'])} packages built for Loom SDK {sdk}",
     "{len(manifest['models'])} packages compatible with Loom SDK {sdk}", 1)
PY

python3 extension/nix/hrx-deps.py generate "$revision" extension/nix/hrx-deps.nix
python3 "$recipes/embed-v030.py"
for repo in gufo extension; do
  git -C "$repo" rev-parse HEAD > "$WORK/out/$repo.base"
  git -C "$repo" diff --check
  git -C "$repo" diff --binary > "$WORK/out/$repo.patch"
done

if [ "${SDK030_COMPAT_ONLY:-0}" = 1 ]; then
  python3 extension/tools/models.py --check
  exit
fi

extra=()
if command -v clang >/dev/null && command -v clang++ >/dev/null; then
  extra+=(-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++)
fi
cmake -S "$revision" -B hrx-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DIREE_ENABLE_POSITION_INDEPENDENT_CODE=ON -DIREE_BUILD_TESTS=OFF \
  -DIREE_ENABLE_WERROR_FLAG=OFF -DLIBHRX_BUILD=OFF \
  -DIREE_HAL_DRIVER_AMD_XDNA=OFF -DLOOM_TARGET_ALL_AMD_GPU=ON \
  -DIREE_CMAKE_TRY_FILE="$WORK/gufo/src/models/loom/embed/loom_text_embed.cmake" \
  "${extra[@]}"
cmake --build hrx-build --target loom_text_embed
ldd hrx-build/libloom_text_embed.so | tee "$WORK/out/gufo-embed-ldd.txt"
mv hrx-build/libloom_text_embed.so "$WORK/libloom_text_embed-gufo.so"

cmake -S "$revision" -B hrx-build \
  -DIREE_CMAKE_TRY_FILE="$WORK/extension/loom-embed/loom_text_embed.cmake"
cmake --build hrx-build --target loom_text_embed
ldd hrx-build/libloom_text_embed.so | tee "$WORK/out/extension-embed-ldd.txt"
