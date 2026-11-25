"""Choice chat.input_measure (starter/input_measure.py): prepare_input on the SDK 0.3.0 complete-conversation count contract."""
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STARTER = os.path.join(ROOT, "skills", "loom-port-llm", "starter")
sys.path.insert(0, STARTER)
import input_measure  # noqa: E402


def read(*p):
    with open(os.path.join(STARTER, *p), encoding="utf-8") as f:
        return f.read()


def vision_block():
    """@vision_prepare_input as gen_chat_vision.py emits it (its f-string braces undoubled)."""
    src = read("vision", "gen_chat_vision.py")
    start = src.index("func.def @vision_prepare_input(")
    end = src.index("func.return %tokens, %total, %spans : buffer, i64, buffer\n}}", start)
    return src[start:end + len("func.return %tokens, %total, %spans : buffer, i64, buffer\n}}")].replace("{{", "{").replace("}}", "}")


def body(text, name):
    start = text.index(f"@{name}(")
    return text[start:text.index("\n}\n", start)]


class InputMeasure(unittest.TestCase):
    def test_plain(self):
        out = input_measure.patch_control(read("templates", "control.loom"))
        b = body(out, "prepare_input")
        self.assertNotIn("@require", b)
        self.assertIn("%probe = scf.select %measuring, %measure, %bounded_probe : i64", b)
        self.assertIn("func.return %output, %count, %spans", b)

    def test_vision(self):
        out = input_measure.patch_control(read("templates", "control.loom") + "\n" + vision_block() + "\n")
        b = body(out, "vision_prepare_input")
        self.assertNotIn("@require(%fits32", b)
        self.assertEqual(b.count("scf.if %keep_ids {"), 2)   # the id copies are skipped while measuring (encode returns no IDs)
        self.assertIn("func.return %output, %total, %output_spans", b)
        self.assertIn("%token_bytes = scf.select %measuring, %origin, %all_bytes : offset", b)
        # every SSA value the patch uses is defined in the function
        for v in ("measuring", "keep_ids", "fits", "origin", "tokens", "spans"):
            self.assertRegex(b, rf"%{v}(, %\w+)* = |%\w+, %{v} = ")

    def test_patched_once(self):
        out = input_measure.patch_control(read("templates", "control.loom"))
        with self.assertRaises(AssertionError):   # the old body is gone: a second application fails loudly instead of double-patching
            input_measure.patch_control(out)

    def test_choice_registered_off(self):
        sys.path.insert(0, os.path.join(ROOT, "skills", "loom-kb"))
        import choices
        c = choices.CHOICES["chat.input_measure"]
        self.assertEqual((c.env, c.legacy, c.rows), ("LLM_INPUT_MEASURE", False, []))


if __name__ == "__main__":
    unittest.main()
