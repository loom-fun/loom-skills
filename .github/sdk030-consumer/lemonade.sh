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
    with file.open(newline="") as source_file:
        text = source_file.read()
    assert text.count(old) == count, (path, old, text.count(old))
    file.write_text(text.replace(old, new), newline="")

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
       '// `pool_capacity`). A snapshot without that record (loom-fetch, `hf download`)\n'
       '// is ready once every file its card reads is here or, for an origin file, in\n'
       '// its own repository\'s cache.',
       '// A snapshot Lemonade fetched records its Loom release. The v0.3.0 runtime\n'
       '// also accepts the published v0.2.0 controller ABI. The v0.1.0 schema still\n'
       '// contains the unsupported pool_capacity field. A snapshot without a marker\n'
       '// is ready once its files are present alongside the card or in the\n'
       '// originating repository\'s cache.')
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
update(header,
       '                                           const std::vector<std::string>& serve_args,\n'
       '                                           const std::string& custom_args);',
       '                                           const std::vector<std::string>& serve_args,\n'
       '                                           const std::string& custom_args,\n'
       '                                           bool legacy_parameter_modes = false);')
update(header,
       '// --parameter_mode=preload: reads the weights into host memory first, which',
       '// Legacy Loom 0.2 --parameter_mode=preload reads weights into host memory first, which')
bundle = "src/cpp/server/backends/loom/loom_bundle.cpp"
update(bundle,
       '    std::string repository;\n    std::string revision;\n};',
       '    std::string repository;\n    std::string revision;\n    bool bundle_local = false;\n};')
update(bundle,
       '        if (kind != "hf") {',
       '        // Loom 0.3 loom-fetch gives materialized bundle files a document origin.\n'
       '        // Those files are checked locally and never downloaded from that origin.\n'
       '        if ((kind == "url" || kind == "local") && name.compare(0, 8, "document") == 0 &&\n'
       '            name.find_first_not_of(\'_\', 8) == std::string::npos) {\n'
       '            origins.emplace(name, Origin{"", "", true});\n'
       '            continue;\n'
       '        }\n'
       '        if (kind != "hf") {')
update(bundle,
       '        plan.origin_files.push_back(OriginFile{path, source->second.repository,',
       '        if (source->second.bundle_local) {\n'
       '            plan.locked_files.push_back(LockedFile{path, digest->second.size,\n'
       '                                                   digest->second.sha256});\n'
       '            return;\n'
       '        }\n'
       '        plan.origin_files.push_back(OriginFile{path, source->second.repository,')
update(bundle,
       '                                           const std::vector<std::string>& serve_args,\n'
       '                                           const std::string& custom_args) {',
       '                                           const std::vector<std::string>& serve_args,\n'
       '                                           const std::string& custom_args,\n'
       '                                           bool legacy_parameter_modes) {')
update(bundle,
       '        argv.insert(argv.end(), parsed.begin(), parsed.end());\n'
       '    }\n'
       '    return argv;',
       '        argv.insert(argv.end(), parsed.begin(), parsed.end());\n'
       '    }\n'
       '    if (!legacy_parameter_modes) {\n'
       '        // Loom 0.3 reads buffered files by default. Old cards and saved arguments\n'
       '        // may still contain these former parameter modes.\n'
       '        argv.erase(std::remove_if(argv.begin(), argv.end(), [](const std::string& arg) {\n'
       '                       return arg == kPreloadFlag || arg == "--parameter_mode=mmap";\n'
       '                   }), argv.end());\n'
       '    }\n'
       '    return argv;')
update(source,
       '    constexpr bool kPreload = true;\r\n',
       '    const std::string sdk_version =\n'
       '        backend_manager_->get_install_params(loom::descriptor.recipe, kBackend).version;\n'
       '    const bool legacy_parameter_modes = sdk_version == "v0.2.0" || sdk_version == "0.2.0";\n')
update(source,
       'loom::serve_args(target, info, loom::parameter_bytes(plan), kPreload, loom_args);',
       'loom::serve_args(target, info, loom::parameter_bytes(plan), legacy_parameter_modes, loom_args);')
update(source, '                                           serve_args, loom_args);',
               '                                           serve_args, loom_args, legacy_parameter_modes);')
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
update(test,
       '    json missing_digest = kLock;',
       '    json materialized = kCard;\n'
       '    materialized["origins"]["document"] = {{"kind", "url"}};\n'
       '    materialized["programs"]["source"]["inputs"][0]["origin"] = "@document/program/prepare.loom";\n'
       '    json materialized_lock = kLock;\n'
       '    materialized_lock["files"].push_back({{"origin", "@document/program/prepare.loom"},\n'
       '                                           {"size", 123u}, {"sha256", "digest"}});\n'
       '    const auto installed = loom::plan_bundle(materialized, materialized_lock);\n'
       '    check("Installed document files are verified locally",\n'
       '          installed.origin_files.size() == 2 && installed.locked_files.size() == 2 &&\n'
       '              installed.locked_files[0].path == "program/prepare.loom");\n'
       '    check("Installed document files still require a locked digest",\n'
       '          throws_exception([&] { (void)loom::plan_bundle(materialized, kLock); }));\n'
       '\n'
       '    json missing_digest = kLock;')
update(test,
       '    check("Serve flags are passed through in order",\n'
       '          loom::build_server_argv("/run", 1, "", {"--pool_backing=fixed", loom::kPreloadFlag}, "") ==',
       '    check("The legacy runtime still receives its parameter mode",\n'
       '          loom::build_server_argv("/run", 1, "", {"--pool_backing=fixed", loom::kPreloadFlag}, "", true) ==')
update(test, '    check("A custom device and memory size follow the defaults, so they win",',
       '    check("The 0.3 launch drops obsolete parameter modes from both input sources",\n'
       '          loom::build_server_argv("/run", 1, "", {"--pool_backing=fixed", loom::kPreloadFlag},\n'
       '                                  "--parameter_mode=mmap --parameter_mode=direct") ==\n'
       '              std::vector<std::string>({"--model=/run/model.json", "--listen=127.0.0.1:1", device,\n'
       '                                        "--pool_backing=fixed", "--parameter_mode=direct"}));\n'
       '    check("A custom device and memory size follow the defaults, so they win",')
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
if [ "${SDK030_GPU_SMOKE:-0}" = 1 ]; then
  cmake --build --preset default --target lemond
fi
