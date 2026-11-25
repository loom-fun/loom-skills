"""Choice telemetry.proposal_counts (starter/proposal_counts.py): publish_epoch takes the SDK 0.3.0 proposal_counts buffer."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STARTER = os.path.join(ROOT, "skills", "loom-port-llm", "starter")
sys.path.insert(0, STARTER)
import proposal_counts  # noqa: E402


def read(*p):
    with open(os.path.join(STARTER, *p), encoding="utf-8") as f:
        return f.read()


class ProposalCounts(unittest.TestCase):
    def test_service_wrappers_take_the_buffer(self):
        out = proposal_counts.patch_service(read("templates", "service.loom"))
        heads = [l for l in out.splitlines() if "publish_epoch" in l and l.startswith(("func.def", "func.decl"))]
        self.assertGreaterEqual(len(heads), 3)   # the declaration, the text and sequence version-1 wrappers
        for h in heads:
            self.assertIn("%progress: buffer, %proposal_counts: buffer)", h)
        self.assertNotIn("%next_results, %progress) :", out)

    def test_plain_control_writes_zero(self):
        out = proposal_counts.patch_control(read("templates", "control.loom"))
        body = out[out.index("func.def public @publish_epoch("):]
        body = body[:body.index("\n}\n")]
        self.assertIn("view.store %c0_i32, %proposals[%i]", body)

    def test_drafted_control_counts_candidates(self):
        sys.path.insert(0, os.path.join(STARTER, "draft"))
        text = read("templates", "control.loom")
        # the draft patch's verified publish, as build_draft.py emits it (the f-string's doubled braces undone)
        src = read("draft", "build_draft.py")
        a = src.index("func.def public @publish_epoch(")
        b = src.index("// propose (when any span proposes)", a)
        draft = src[a:b].replace("{{", "{").replace("}}", "}").replace("{source_step}", "2")
        text = text.replace("func.def public @publish_epoch(", "func.def @plain_publish_epoch(") + "\n" + draft
        out = proposal_counts.patch_control(text)
        self.assertIn("%presented = scf.select %speculative, %candidates, %c0_i32 : i32", out)
        self.assertIn("%next_results, %progress, %proposal_counts) :", out)
        self.assertNotIn("%progress: buffer)", out)

    def test_choice_registered_off(self):
        sys.path.insert(0, os.path.join(ROOT, "skills", "loom-kb"))
        import choices
        c = choices.CHOICES["telemetry.proposal_counts"]
        self.assertEqual((c.env, c.legacy, c.rows), ("LLM_PROPOSAL_COUNTS", False, []))


if __name__ == "__main__":
    unittest.main()
