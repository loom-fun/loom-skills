#!/usr/bin/env bash
set -euo pipefail
exec > >(tee "$WORK/out/lemonade.log") 2>&1
cd "$WORK"
url=https://github.com/ROCm/hrx-system/releases/download/v0.3.0
curl -fsSL --retry 3 "$url/SHA256SUMS" -o SHA256SUMS
echo '9e1269b2b094f786f338e9050c8fcfe359ea537ccc40af878b48a6458c09b19c  SHA256SUMS' | sha256sum -c -
for platform in manylinux_2_39_x86_64 win_amd64; do
  name="loom-serve-0.3.0-$platform.zip"
  curl -fsSL --retry 3 "$url/$name" -o "$name"
  awk -v name="$name" '$2 == name' SHA256SUMS | sha256sum -c -
done
python3 - <<'PY'
from pathlib import Path
from zipfile import ZipFile

for platform, suffix in (("manylinux_2_39_x86_64", ""), ("win_amd64", ".exe")):
    with ZipFile(f"loom-serve-0.3.0-{platform}.zip") as archive:
        assert archive.namelist() == [f"loom-serve{suffix}"], archive.namelist()
        path = archive.extract(f"loom-serve{suffix}", f"serve-{platform}")
        Path(path).chmod(0o755)
PY
./serve-manylinux_2_39_x86_64/loom-serve --version

git clone -q --filter=blob:none --depth 1 --branch dholanda/loom_engine https://github.com/lemonade-sdk/lemonade.git lemonade
cd lemonade
test "$(git rev-parse HEAD)" = c288d4b3a5b430ba716bac8650e26302c53ade46
python3 - <<'PY'
import json
from pathlib import Path

def update(path, old, new, count=1):
    file = Path(path)
    text = file.read_text()
    assert text.count(old) == count, (path, old, text.count(old))
    file.write_text(text.replace(old, new))

resource = Path("src/cpp/resources/backend_versions.json")
text = resource.read_text()
version = "v0.3.0"
release_hashes = {
    name: "sha256:" + digest
    for digest, name in [line.split(maxsplit=1) for line in Path("../SHA256SUMS").read_text().splitlines()]
    if name.startswith("loom-serve-0.3.0-")
}
names = ["loom-serve-0.3.0-win_amd64.zip",
         "loom-serve-0.3.0-manylinux_2_39_x86_64.zip"]
registry = '        "v0.3.0": {\n' + ",\n".join(
    f'          "{name}": "{release_hashes[name]}"' for name in names
) + '\n        },\n'
update(resource, '        "v0.2.0": {', registry + '        "v0.2.0": {')
update(resource, '"loom": {\n    "hrx": "v0.2.0"', '"loom": {\n    "hrx": "v0.3.0"')
data = json.loads(resource.read_text())
assert data["loom"]["hrx"] == version

header = "src/cpp/include/lemon/backends/loom/loom_bundle.h"
update(header, 'kBundleRevision = "v0.2.0"', 'kBundleRevision = "v0.3.0"')
source = "src/cpp/server/backends/loom/loom_server.cpp"
update(source,
       '// A snapshot Lemonade fetched records its Loom release; one fetched for another\n'
       '// release holds a card this loom-serve rejects (0.2.0 refuses 0.1.0\'s\n'
       '// ', 
       '// A snapshot Lemonade fetched records its Loom release. The v0.3.0 runtime\n'
       '// also accepts the published v0.2.0 controller ABI. The v0.1.0 card schema\n'
       '// still contains the unsupported ')
update(source,
       'return read_json_file(dir / loom::kCompleteMarker).value("loom", "") ==\n'
       '                   loom::kBundleRevision;',
       'const auto release = read_json_file(dir / loom::kCompleteMarker).value("loom", "");\n'
       '            return release == loom::kBundleRevision || release == "v0.2.0";')
update(source, 'const std::string wheel_version =', 'const std::string release_version =')
update(source, '    params.repo = "ROCm/hrx-system";\n#if',
       '    params.repo = "ROCm/hrx-system";\n    const char* platform;\n#if')
update(source, '    params.filename = "hrx_loom-" + wheel_version + "-py3-none-win_amd64.whl";',
       '    platform = "win_amd64";')
update(source, '    params.filename = "hrx_loom-" + wheel_version + "-py3-none-manylinux_2_39_x86_64.whl";',
       '    platform = "manylinux_2_39_x86_64";')
update(source,
       '    params.archive_subdir = "hrx_loom-" + wheel_version + ".data/scripts";',
       '    // Older releases supplied only wheels. The v0.3.0 standalone archives\n'
       '    // contain the executable and its runtime dependencies in one file.\n'
       '    if (release_version == "0.2.0") {\n'
       '        params.filename = "hrx_loom-" + release_version + "-py3-none-" + platform + ".whl";\n'
       '        params.archive_subdir = "hrx_loom-" + release_version + ".data/scripts";\n'
       '    } else {\n'
       '        params.filename = "loom-serve-" + release_version + "-" + platform + ".zip";\n'
       '    }')
update(header,
       '// loom-serve reads, and the backend_versions.json pin. A snapshot Lemonade recorded\n'
       '// for another release (an older card loom-serve rejects) is fetched again.',
       '// loom-serve reads, and the backend_versions.json pin. A snapshot recorded\n'
       '// for a supported older release can also be reused offline.')
test = Path("test/cpp/test_loom_contract.cpp")
text = test.read_text()
# Insert near the existing per-platform installer assertions.
first = text.index('    const auto params = LoomServer::get_install_params("hrx", "v0.2.0");')
start = text.rindex('\n#if', 0, first)
new_test = '''
#if defined(_WIN32) || (defined(__linux__) && (defined(__x86_64__) || defined(__amd64__)))
    const auto standalone = LoomServer::get_install_params("hrx", "v0.3.0");
#if defined(_WIN32)
    check("Loom 0.3.0 uses the standalone Windows archive",
          standalone.filename == "loom-serve-0.3.0-win_amd64.zip");
#else
    check("Loom 0.3.0 uses the standalone Linux archive",
          standalone.filename == "loom-serve-0.3.0-manylinux_2_39_x86_64.zip");
#endif
    check("The standalone executable is at the root of the archive",
          standalone.archive_subdir.empty());
#endif
'''
test.write_text(text[:start] + new_test + text[start:])
update(test, '        check("A snapshot Lemonade fetched for another Loom release is not downloaded",',
       '        check("The v0.1.0 bundle schema still requires a fresh download",')
update(test,
       '        FixtureHub::write(hub.snapshot() / loom::kCompleteMarker,\n'
       '                          json{{"loom", loom::kBundleRevision}}.dump());',
       '        FixtureHub::write(hub.snapshot() / loom::kCompleteMarker, json{{"loom", "v0.2.0"}}.dump());\n'
       '        check("A complete v0.2.0 bundle remains usable offline", FixtureHub::downloaded());\n'
       '        FixtureHub::write(hub.snapshot() / loom::kCompleteMarker,\n'
       '                          json{{"loom", loom::kBundleRevision}}.dump());')
PY
git rev-parse HEAD > "$WORK/out/lemonade.base"
git diff --check
git diff --binary > "$WORK/out/lemonade.patch"
cmake --preset default -DBUILD_WEB_APP=OFF
cmake --build --preset default --target test_loom_contract
ctest --test-dir build --output-on-failure --no-tests=error -R '^LoomContractTest$'
