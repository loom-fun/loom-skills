"""fleet/replays/: the replay inputs are well formed (meta, scope blocks, patches) and the committed evidence agrees with itself."""
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RP = ROOT / "fleet" / "replays"
sys.path.insert(0, str(ROOT / "fleet"))
sys.path.insert(0, str(RP))
import scope_lint as SL  # noqa: E402
import summarize  # noqa: E402

NAMES = sorted(p.name for p in RP.iterdir() if (p / "meta.json").is_file())


class Replays(unittest.TestCase):
    def test_every_replay_has_inputs(self):
        self.assertGreaterEqual(len(NAMES), 9)
        for n in NAMES:
            meta = json.loads((RP / n / "meta.json").read_text())
            self.assertTrue(meta["pr"] and meta["title"] and meta["intent"], n)
            if meta.get("commit"):
                self.assertTrue((RP / n / "A.scope").is_file(), n)
                continue
            for v in meta["variants"]:
                self.assertTrue((RP / n / (v + ".scope")).is_file(), f"{n}/{v}.scope")
                self.assertTrue((RP / n / (v + ".patch")).is_file() or (RP / n / (v[0] + ".patch")).is_file(), f"{n}/{v}.patch")

    def test_scope_blocks_parse(self):
        for n in NAMES:
            for sf in (RP / n).glob("*.scope"):
                block, errs = SL.parse_block("```loom-scope\n" + sf.read_text() + "\n```\n")
                self.assertEqual(errs, [], str(sf))
                self.assertIn(block["kind"], SL.KINDS, str(sf))

    def test_patches_are_diffs(self):
        for n in NAMES:
            for pf in (RP / n).glob("*.patch"):
                t = pf.read_text()
                self.assertTrue(t.startswith("diff --git a/"), str(pf))
                self.assertEqual(len(re.findall(r"^diff --git ", t, re.M)), len(re.findall(r"^\+\+\+ ", t, re.M)), str(pf))

    def test_evidence_matches_the_scope_claims(self):
        """B variants (opt-in scope rules) touch exactly their declared scope: no leak, lint passes."""
        d = summarize.collect()
        for n, e in d.items():
            if e["meta"].get("commit"):
                self.assertEqual(e["variants"]["A"]["touched"], 0, n)   # tooling / card PRs: empty program impact
                continue
            b = e["variants"]["B"]
            self.assertIsNotNone(b["lint"], n)
            if n == "r573":   # the guard reads the target tag a twin adds to every gfx11 kernel as a gfx11 change (REPLAYS.md, finding 1)
                self.assertGreater(b["lint"]["leaked"], 0, n)
                continue
            self.assertEqual(b["lint"]["leaked"], 0, n)
            self.assertEqual(b["lint"]["exit"], 0, n)

    def test_summary_tables_render(self):
        t = summarize.tables(summarize.collect())
        self.assertIn("| PR |", t)
        self.assertGreaterEqual(t.count("\n"), 2 * len(NAMES))


if __name__ == "__main__":
    unittest.main()
