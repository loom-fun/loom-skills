"""build_fleet.py: the default preset (Strix Halo's) takes the ref's exec.prefill_capacity scope rule as the build's --prefill."""
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "fleet"))
import build_fleet  # noqa: E402


class DefaultPresetPrefill(unittest.TestCase):
    def test_a_strix_scope_rule_is_the_build_prefill(self):
        self.assertEqual(build_fleet.default_preset_prefill("ggml-org-Qwen3.6-35B-A3B-loom", ROOT), 8192)
        self.assertEqual(build_fleet.default_preset_prefill("unsloth-Qwen3.5-35B-A3B-loom", ROOT), 8192)

    def test_no_rule_keeps_the_manifest_value(self):
        self.assertIsNone(build_fleet.default_preset_prefill("unsloth-Qwen3-0.6B-loom", ROOT))
        self.assertIsNone(build_fleet.default_preset_prefill("not-a-fleet-model", ROOT))


if __name__ == "__main__":
    unittest.main()
