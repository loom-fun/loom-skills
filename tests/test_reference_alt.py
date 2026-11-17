"""The reference store's second recording (bench_common.record_alt / ALT_SALT_*): taken only for a request whose first recording was in the store
when the job started (a different job's answer), under its own salt, with the same request body."""
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts"))
try:
    import bench_common as bc  # noqa: E402
except ImportError:   # the CI python has no requests / numpy: the tests skip
    bc = None


class Pkg:
    cfg = {"gguf": None, "source": {"kind": "safetensors", "repo": "org/model", "revision": "abc"}}


@unittest.skipIf(bc is None, "bench_common needs requests")
class Alt(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {"REFERENCE_ALT": "record", "REFERENCE_IDENTITY": "x", "LLAMA_CACHE_BUILD": "vllm-0.18.0",
                                                "REFERENCE_STORED_KEYS": os.path.join(self.d, "stored_keys.txt")})
        self.env.start()
        bc._STORED = None
        self.pkg = None
        self.calls = []

    def tearDown(self):
        self.env.stop()
        bc._STORED = None

    def test_salts_make_distinct_keys_for_one_request(self):
        b = bc.greedy_body([1, 2, 3], 64)
        self.assertEqual(len({bc.ref_key(b, None, s) for s in ("", bc.ALT_SALT_TEXT, bc.ALT_SALT_VISION)}), 3)

    def test_alt_only_for_requests_stored_at_start(self):
        b1, b2 = bc.greedy_body([1, 2, 3], 64), bc.greedy_body([4, 5], 64)
        open(os.environ["REFERENCE_STORED_KEYS"], "w").write(bc.ref_key(b1, None, "") + "\n")
        with mock.patch.object(bc, "llama_completion", lambda url, body, pkg=None, salt="", **k: self.calls.append((body, salt))):
            self.assertTrue(bc.record_alt_greedy("u", [1, 2, 3], 64, None))
            self.assertFalse(bc.record_alt_greedy("u", [4, 5], 64, None))   # first recorded by this very job: not independent
        self.assertEqual(self.calls, [(b1, bc.ALT_SALT_TEXT)])

    def test_cached_peek(self):
        b = bc.greedy_body([9], 8)
        with mock.patch.dict(os.environ, {"LLAMA_CACHE": self.d}):
            self.assertFalse(bc.cached(b, None, bc.ALT_SALT_TEXT))
            k = bc.ref_key(b, None, bc.ALT_SALT_TEXT)
            os.makedirs(os.path.join(self.d, k[:2]))
            open(os.path.join(self.d, k[:2], k + ".json"), "w").write("{}")
            self.assertTrue(bc.cached(b, None, bc.ALT_SALT_TEXT))
            self.assertFalse(bc.cached(b, None, ""))

    def test_off_without_the_env(self):
        with mock.patch.dict(os.environ, {"REFERENCE_ALT": ""}), mock.patch.object(bc, "llama_completion") as lc:
            self.assertFalse(bc.record_alt_greedy("u", [1], 8, None))
            lc.assert_not_called()

    def test_no_stored_list_records_nothing(self):
        with mock.patch.object(bc, "llama_completion") as lc:
            self.assertFalse(bc.record_alt_greedy("u", [1], 8, None))
            lc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
