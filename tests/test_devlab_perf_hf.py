"""devlab_perf.sh hf_sub: {hf:OWNER/REPO/FILE} in build_args / llama_spec resolves to the per-runner cached Hub file."""
import os, subprocess, tempfile, unittest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", ".github", "scripts", "devlab_perf.sh")


def _fn():
    lines, on = [], False
    for ln in open(SCRIPT):
        if ln.startswith("hf_sub()"):
            on = True
        if on:
            lines.append(ln)
            if ln.startswith("}"):
                break
    return "".join(lines)


@unittest.skipIf(os.name == "nt", "devlab_perf.sh runs on the Linux DevLab runners only")
class HfSub(unittest.TestCase):
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
