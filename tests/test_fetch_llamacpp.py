"""Downloader destination precedence, using cached stub releases and no network."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "skills/loom-benchmark/scripts/fetch_llamacpp.sh"


class FetchLlamacppTests(unittest.TestCase):
    def run_cached(self, destination, override=None, args=(), tag="test-release", gfx="gfx1151"):
        with tempfile.TemporaryDirectory(prefix="llamacpp paths ") as tmp:
            root = Path(tmp)
            home = root / "home"
            cache = root / "persistent cache"
            explicit = root / "explicit cache"
            selected = {"home": home / "llamacpp-rocm", "env": cache, "explicit": explicit}[destination]
            release = selected / tag
            release.mkdir(parents=True)
            server = release / "llama-server"
            server.write_text('#!/bin/sh\necho "ROCm device"\n')
            server.chmod(0o755)
            bindir = root / "bin"
            bindir.mkdir()
            metadata = {"tag_name": tag, "assets": [{
                "name": f"llama-ubuntu-rocm-{gfx}-x64.zip",
                "browser_download_url": "https://example.invalid/release.zip",
            }]}
            # Fail rather than access the network or download an archive.
            curl = bindir / "curl"
            curl.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CURL_LOG"\n'
                            'case "$*" in *" -o "*) exit 99;; esac\n'
                            'cat "$RELEASE_META"\n')
            curl.chmod(0o755)
            rocminfo = bindir / "rocminfo"
            rocminfo.write_text('#!/bin/sh\necho gfx1151\n')
            rocminfo.chmod(0o755)
            meta = root / "meta.json"
            meta.write_text(json.dumps(metadata))
            env = os.environ.copy()
            env.pop("LLAMACPP_HOME", None)
            env.update(HOME=str(home), PATH=f"{bindir}:{env['PATH']}",
                       LLAMACPP_BACKEND="rocm", RELEASE_META=str(meta),
                       CURL_LOG=str(root / "curl.log"))
            if override is not None:
                env["LLAMACPP_HOME"] = str(cache) if override else ""
            argv = [str(explicit) if arg == "EXPLICIT" else arg for arg in args]
            result = subprocess.run(["bash", str(SCRIPT), *argv], env=env,
                                    text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), str(release))
            self.assertEqual(result.stderr, "")
            api = (root / "curl.log").read_text().strip().split()[-1]
            expected = "latest" if len(args) < 2 else f"tags/{tag}"
            self.assertEqual(api, f"https://api.github.com/repos/lemonade-sdk/llamacpp-rocm/releases/{expected}")
            self.assertEqual(list(selected.iterdir()), [release])

    def test_explicit_destination_overrides_environment(self):
        self.run_cached("explicit", override=True, args=("EXPLICIT", "custom-tag", "gfx1201"),
                        tag="custom-tag", gfx="gfx1201")

    def test_environment_destination(self):
        self.run_cached("env", override=True)

    def test_home_default(self):
        self.run_cached("home")

    def test_empty_environment_uses_home(self):
        self.run_cached("home", override=False)

    def test_empty_destination_uses_environment_and_keeps_tag_gfx(self):
        self.run_cached("env", override=True, args=("", "custom-tag", "gfx1201"),
                        tag="custom-tag", gfx="gfx1201")
