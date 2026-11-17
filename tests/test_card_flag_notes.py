"""Per-GPU quickstart flag explanations belong to source cards, alongside their commands."""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "skills/loom-benchmark/scripts/card"
sys.path.insert(0, str(CARD))
import render_card as R


class QuickstartFlagNotes(unittest.TestCase):
    def test_only_emitted_flags_are_explained(self):
        fetch = 'loom-fetch source --materialize=copy --output=X'
        fixed = "loom-serve --model=X '--device=amdgpu://?completion_spin_ns=4294967295' --pool_backing=fixed"
        text = '\n'.join(R.quickstart_flag_notes(fetch, fixed))
        self.assertIn('copies of the fetched package files', text)
        self.assertIn("performance measurements use this setting", text)
        self.assertIn('uses CPU while GPU work is in flight', text)
        self.assertIn('commits the preset', text)
        self.assertNotIn('--memory_bytes', text)
        self.assertNotIn('--parameter_mode=preload', text)
        self.assertEqual(R.quickstart_flag_notes('loom-fetch source', 'loom-serve --model=X --device=amdgpu'), [])

    def test_budget_preload_and_slab_notes_do_not_claim_fixed_backing(self):
        line = 'loom-serve --memory_bytes=15000000000 --state_slab_bytes=4mib --weight_slab_bytes=64mib --parameter_mode=preload'
        text = '\n'.join(R.quickstart_flag_notes('loom-fetch source', line))
        for flag in ['--memory_bytes', '--state_slab_bytes', '--weight_slab_bytes', '--parameter_mode=preload']:
            self.assertIn(flag, text)
        self.assertNotIn('--pool_backing=fixed', text)
        self.assertNotIn('completion_spin_ns', text)
        self.assertNotIn('--materialize=copy', text)

    def test_rendered_single_and_multi_gpu_blocks(self):
        for fixture in ['sdk02', 'r9700_two', 'gpus4']:
            text = (CARD / 'test' / fixture / 'expected/README.md').read_text()
            quick = text.split('## Quick start', 1)[1].split('\n## ', 1)[0]
            details = re.findall(r'<details>.*?<summary>.*?</summary>(.*?)</details>', quick, re.S)
            for block in details or [quick]:
                fetch = re.search(r'^loom-fetch .+$', block, re.M).group()
                serve = re.search(r'^loom-serve .+$', block, re.M).group()
                for note in R.quickstart_flag_notes(fetch, serve):
                    self.assertIn(note, block, fixture)
                if '--pool_backing=fixed' not in serve:
                    self.assertNotIn('commits the preset', block)
                if '--parameter_mode=preload' not in serve:
                    self.assertNotIn('only changes how the weights', block)


if __name__ == '__main__':
    unittest.main()
