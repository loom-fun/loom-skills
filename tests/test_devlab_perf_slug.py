"""devlab_perf.sh slug_arg: a DevLab build passes the card's slug to build.py, as fleet/build_fleet.py's canonical build does.

Without it the knowledge base's slug-matched model nodes do not apply and the job builds other programs than the published ones
(found on the 0.2.0 sweep: Qwen3.8-27B lost draft.propose.head_rows = Strix decode -10 %, Qwen3.6-35B-A3B lost decode.attn.kv_once.d256 on rdna4)."""
import os, subprocess, tempfile, unittest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", ".github", "scripts", "devlab_perf.sh")


def _fn(name):
    lines, on = [], False
    for ln in open(SCRIPT):
        if ln.startswith(name + "()"):
            on = True
        if on:
            lines.append(ln)
            if ln.startswith("}"):
                break
    return "".join(lines)


def _run(build_py, repo):
    return subprocess.run(["bash", "-c", f'{_fn("slug_arg")}\nslug_arg "{build_py}" "{repo}"'], capture_output=True, text=True, check=True).stdout.strip()


@unittest.skipIf(os.name == "nt", "devlab_perf.sh runs on the Linux DevLab runners only")
class SlugArg(unittest.TestCase):
    def test_kit_with_slug_flag(self):
        with tempfile.TemporaryDirectory() as w:
            b = os.path.join(w, "build.py")
            open(b, "w").write('ap.add_argument("--slug", default=None)\n')
            self.assertEqual(_run(b, "loom-fun-staging/bartowski-Qwen3.8-27B-loom"), "--slug bartowski-Qwen3.8-27B-loom")

    def test_kit_without_slug_flag(self):
        with tempfile.TemporaryDirectory() as w:
            b = os.path.join(w, "build.py")
            open(b, "w").write('ap.add_argument("--context")\n')
            self.assertEqual(_run(b, "loom-fun/x-loom"), "")

    def test_build_uses_it(self):
        text = open(SCRIPT).read()
        self.assertIn('slugarg=$(slug_arg "${KIT[$v]}/build.py" "$repo")', text)
        self.assertIn('build.py "$D/$v/cfg.json" "$pkg" $slugarg', text)


if __name__ == "__main__":
    unittest.main()
