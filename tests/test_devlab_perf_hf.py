"""devlab_perf.sh hf_sub: {hf:OWNER/REPO/FILE} in build_args / llama_spec resolves to the per-runner cached Hub file."""
import hashlib, os, shlex, subprocess, tempfile, unittest
from pathlib import Path

SCRIPT = os.path.join(os.path.dirname(__file__), "..", ".github", "scripts", "devlab_perf.sh")


def _fn():
    lines, on = [], False
    for ln in Path(SCRIPT).read_text().splitlines(keepends=True):
        if ln.startswith("hf_sub()"):
            on = True
        if on:
            lines.append(ln)
            if ln.startswith("}"):
                break
    return "".join(lines)


@unittest.skipIf(os.name == "nt", "devlab_perf.sh runs on the Linux DevLab runners only")
class HfSub(unittest.TestCase):
    REV = "a" * 40
    DATA = b"checkpoint"
    SHA = hashlib.sha256(DATA).hexdigest()

    def pinned(self, work, marker=None, transport=None):
        marker = marker or f"{{hf:o/r@{self.REV}/d.gguf#{self.SHA}}}"
        # Mock only transport; the real parser, cache path and sha256sum run.
        curl = transport or 'curl() { printf "%s\\n" "$@" > "$WORK/curl.args"; printf checkpoint > "${@: -1}"; }'
        return subprocess.run(["bash", "-c", f'WORK={shlex.quote(str(work))}\n{curl}\n{_fn()}\nhf_sub {shlex.quote(marker)}'],
                              capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k != "HF_TOKEN"})

    def test_pinned_download_uses_revision_and_checksum(self):
        with tempfile.TemporaryDirectory() as w:
            out = self.pinned(w)
            self.assertEqual(out.returncode, 0, out.stderr)
            path = Path(w) / "hf-pinned/o/r" / self.REV / "d.gguf"
            self.assertEqual(out.stdout.strip(), str(path))
            self.assertEqual(path.read_bytes(), self.DATA)
            self.assertIn(f"https://huggingface.co/o/r/resolve/{self.REV}/d.gguf", (Path(w) / "curl.args").read_text())
            self.assertEqual(list(path.parent.glob("*.part.*")), [])

    def test_pinned_cached_bytes_verified_without_fetch(self):
        with tempfile.TemporaryDirectory() as w:
            path = Path(w) / "hf-pinned/o/r" / self.REV / "d.gguf"
            path.parent.mkdir(parents=True); path.write_bytes(self.DATA)
            out = self.pinned(w)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(out.stdout.strip(), str(path))
            self.assertFalse((Path(w) / "curl.args").exists())
            path.write_bytes(b"corrupt")
            out = self.pinned(w)
            self.assertNotEqual(out.returncode, 0)
            self.assertIn("checksum mismatch", out.stderr)
            self.assertEqual(path.read_bytes(), b"corrupt")
            self.assertFalse((Path(w) / "curl.args").exists())

    def test_pinned_download_mismatch_never_becomes_cache(self):
        with tempfile.TemporaryDirectory() as w:
            out = self.pinned(w, f"{{hf:o/r@{self.REV}/d.gguf#{'0' * 64}}}")
            self.assertNotEqual(out.returncode, 0)
            self.assertIn("checksum mismatch", out.stderr)
            self.assertEqual(list((Path(w) / "hf-pinned").rglob("*.gguf*")), [])

    def test_failed_download_removes_only_its_temporary_file(self):
        with tempfile.TemporaryDirectory() as w:
            out = self.pinned(w, transport='curl() { printf partial > "${@: -1}"; return 22; }')
            self.assertNotEqual(out.returncode, 0)
            self.assertIn("FAIL: fetch", out.stderr)
            self.assertEqual(list((Path(w) / "hf-pinned").rglob("*.gguf*")), [])

    def test_invalid_pin_fails_before_transport(self):
        for value in (f"o/r@main/d.gguf#{self.SHA}", f"o/r@{self.REV}/d.gguf",
                      f"o/r@{self.REV}/d.gguf#bad", f"o/r@{self.REV}/../d.gguf#{self.SHA}"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as w:
                out = self.pinned(w, "{hf:" + value + "}")
                self.assertNotEqual(out.returncode, 0)
                self.assertIn("invalid pinned HF marker", out.stderr)
                self.assertFalse((Path(w) / "curl.args").exists())

    def test_pinned_revision_does_not_reuse_legacy_or_other_revision(self):
        with tempfile.TemporaryDirectory() as w:
            for rel in ("hf/o/r/d.gguf", f"hf-pinned/o/r/{'b' * 40}/d.gguf"):
                path = Path(w) / rel; path.parent.mkdir(parents=True); path.write_bytes(b"old")
            out = self.pinned(w)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(Path(out.stdout.strip()).read_bytes(), self.DATA)
            self.assertTrue((Path(w) / "curl.args").exists())

    def test_cached_file_substituted(self):
        with tempfile.TemporaryDirectory() as w:
            f = os.path.join(w, "hf", "o", "r", "d.gguf")
            os.makedirs(os.path.dirname(f))
            open(f, "w").close()
            out = subprocess.run(["bash", "-c", f'WORK={w}\n{_fn()}\nhf_sub "--drafter {{hf:o/r/d.gguf}} --draft-depth 7"'],
                                 capture_output=True, text=True, check=True).stdout.strip()
            self.assertEqual(out, f"--drafter {f} --draft-depth 7")

    def test_no_marker_unchanged(self):
        out = subprocess.run(["bash", "-c", f'WORK=/nonexistent\n{_fn()}\nhf_sub "--draft mtp"'], capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(out, "--draft mtp")


if __name__ == "__main__":
    unittest.main()
