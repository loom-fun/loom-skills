"""draft/tune.py: the 4-stream row is not a drafting criterion (the card measures 4 streams undrafted on both engines)."""
import importlib.util
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("tune", os.path.join(ROOT, "skills", "loom-port-llm", "starter", "draft", "tune.py"))
tune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tune)


class TuneRows(unittest.TestCase):
    def test_streams4_dip_keeps_the_drafter(self):   # Ternary Bonsai 27B run 45535242750: MTP depth 3 vs no drafting
        base = {"work0": 26.23, "work1": 26.22, "long": 19.82, "streams4": 83.79}
        mtp3 = {"work0": 40.39, "work1": 59.32, "long": 28.75, "streams4": 79.09}
        self.assertGreater(tune.margin(mtp3, base), 1.4)
        refs = {"decode": 34.7, "decode_32k": 28.4, "c4": 62.7}
        self.assertGreater(tune.llama_margin(mtp3, refs), tune.llama_margin(base, refs))

    def test_single_stream_loss_still_drops_it(self):
        base = {"work0": 30.0, "long": 20.0, "streams4": 80.0}
        self.assertLess(tune.margin({"work0": 31.0, "long": 18.0, "streams4": 90.0}, base), tune.NO_LOSS)


if __name__ == "__main__":
    unittest.main()
