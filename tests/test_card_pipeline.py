"""The safetensors MXFP4 card's pipeline data agree: nightly entry, fit row, drafting classification, fleet build, org_status flags."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/loom-fleet/scripts"))
sys.path.insert(0, str(ROOT / "skills/loom-benchmark/scripts/card"))
import org_status  # noqa: E402
import matched_spec  # noqa: E402

SLUG = "amd-Qwen3.8-27B-Quark-AWQ-MXFP4-loom"
REPO = "loom-fun/" + SLUG


def load(p):
    return json.loads((ROOT / p).read_text())


class CardPipelineTests(unittest.TestCase):
    def test_nightly_entry_has_no_llama_build(self):
        e = {m["repo"]: m for m in load(".github/nightly-models.json")["models"]}[REPO]
        self.assertIsNone(e["llamacpp"])   # llama.cpp cannot read a safetensors checkpoint
        self.assertEqual(e["memory_bytes"], 51000000000)   # build.py's value at the native context

    def test_fit_row_from_the_built_package(self):
        r = {x["slug"]: x for x in load(".github/devlab/r9700_fit.json")}[SLUG]
        self.assertEqual(r["ctx"], 262144)
        self.assertGreater(r["w"], 19.8e9)   # weights incl. the vision tower + workspace + recurrent state
        self.assertEqual(r["verdict"], "fits (reduced ctx)")

    def test_classification_has_no_llama_spec(self):
        e = matched_spec.classification(REPO)
        self.assertEqual(e["choice"], "mtp")
        self.assertEqual(e["llama_spec"], "")
        self.assertEqual(e["baseline_engine"], "none")

    def test_fleet_builds_the_native_context(self):
        b = load("fleet/manifest.json")["models"][SLUG]["build"]
        self.assertEqual((b["context"], b["pool"]), (262144, 327680))

    def test_org_status_has_nothing_to_flag(self):
        nightly = {m["repo"]: m for m in load(".github/nightly-models.json")["models"]}
        r = {"repo": REPO, "category": "A", "memory_bytes": 51000000000, "llamacpp": None, "generator": None}
        self.assertEqual(org_status.consistency_checks(r, nightly, None), [])


if __name__ == "__main__":
    unittest.main()
