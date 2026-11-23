"""vLLM as the gates' reference engine for safetensors packages (vllm_shim.py): request/response mapping, a mock server
answering bench_common's llama_greedy, and the cache identity. No GPU, no vLLM."""
import base64
import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "skills/loom-benchmark/scripts"
sys.path.insert(0, str(SCRIPTS))
import vllm_shim as vs  # noqa: E402

try:   # bench_common needs requests (+ tokenizers through chat_probe): not on every CI image
    import bench_common as bc  # noqa: E402
except ImportError:
    bc = None
need_bc = unittest.skipIf(bc is None, "bench_common needs requests")


class Pkg:
    def __init__(self, cfg):
        self.cfg = cfg


ST = {"source": {"kind": "safetensors", "repo": "amd/X", "revision": "abc"}}


class Mapping(unittest.TestCase):
    def test_response_orders_top_and_truncates(self):
        r = vs.completion_response([5, 7], [{5: (-0.1, "a"), 6: (-2.0, "b"), 9: (-1.0, "c")}, {1: (-3.0, "x"), 7: (-0.5, "y")}],
                                   "ab", 2, 11)
        self.assertEqual(r["tokens"], [5, 7])
        self.assertEqual([t["id"] for t in r["completion_probabilities"][0]["top_logprobs"]], [5, 9])
        self.assertEqual(r["completion_probabilities"][1]["logprob"], -0.5)
        self.assertEqual(r["timings"]["prompt_n"], 11)
        self.assertEqual(r["tokens_evaluated"], 11)

    def test_sampled_token_outside_top_k_is_kept_as_chosen(self):
        r = vs.completion_response([3], [{1: (-0.2, "a"), 2: (-0.3, "b"), 3: (-9.0, "z")}], "z", 2, 1)
        self.assertEqual(r["completion_probabilities"][0]["id"], 3)

    def test_parse_ids_and_string(self):
        eng, imgs, n, p = vs.parse_request({"prompt": [1, 2], "n_predict": 4, "n_probs": 5, "temperature": 0})
        self.assertEqual((eng, imgs, n, p), ({"prompt_token_ids": [1, 2]}, [], 4, 5))

    def test_parse_vision_replaces_marker(self):
        img = base64.b64encode(b"png").decode()
        eng, imgs, _, _ = vs.parse_request({"prompt": {"prompt_string": "a<__media__>b", "multimodal_data": [img]}})
        self.assertEqual(eng["prompt"], "a" + vs.IMAGE_PLACEHOLDER + "b")
        self.assertEqual(imgs, [b"png"])
        with self.assertRaises(ValueError):
            vs.parse_request({"prompt": {"prompt_string": "no marker", "multimodal_data": [img]}})

    def test_non_greedy_refused(self):
        with self.assertRaises(ValueError):
            vs.parse_request({"prompt": [1], "temperature": 0.7})


@need_bc
class Server(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = vs.make_server(vs.MockEngine(), 0)
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        os.environ["LLAMA_CACHE"] = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_gates_helper_roundtrip_and_cache(self):
        r, gen, cp = bc.llama_greedy(self.url, [10, 20, 30], 3, Pkg(ST))
        self.assertEqual(gen, [31, 32, 33])
        self.assertEqual(len(cp), 3)
        self.assertEqual(cp[0]["top_logprobs"][0]["id"], 31)
        srv_down = self.srv.shutdown  # a second call is answered from the cache, the server is not needed
        r2 = bc.llama_completion("http://127.0.0.1:1", dict(bc.GREEDY, prompt=[10, 20, 30], n_predict=3, n_probs=5, return_tokens=True), Pkg(ST))
        self.assertEqual(r2["tokens"], gen)
        del srv_down

    def test_micro_batch_view_is_empty_and_health_props(self):
        import requests
        self.assertTrue(requests.get(self.url + "/health").ok)
        self.assertEqual(requests.get(self.url + "/props").json()["engine"], "mock")
        r = requests.post(self.url + "/completion", json={"prompt": [1, 2], "_micro_batch": 8}).json()
        self.assertEqual(r["completion_probabilities"], [])
        self.assertEqual(requests.post(self.url + "/completion", json={"prompt": [1], "temperature": 1}).status_code, 400)

    def test_vision_request(self):
        import requests
        img = base64.b64encode(b"x").decode()
        r = requests.post(self.url + "/completion", json={"prompt": {"prompt_string": "q<__media__>", "multimodal_data": [img]},
                                                          "n_predict": 2, "n_probs": 2}).json()
        self.assertEqual(len(r["tokens"]), 2)


@need_bc
class Identity(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(bc.reference_identity(ST), "safetensors:amd/X@abc")
        self.assertIsNone(bc.reference_identity({"gguf": "x.gguf"}))

    def test_revision_keys_the_cache(self):
        os.environ["LLAMA_CACHE"] = tempfile.mkdtemp()
        os.environ["LLAMA_CACHE_BUILD"] = "vllm-0.18.0"
        srv = vs.make_server(vs.MockEngine(), 0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            body = dict(bc.GREEDY, prompt=[1], n_predict=1, n_probs=1, return_tokens=True)
            bc.llama_completion(url, body, Pkg(ST))
            other = {"source": dict(ST["source"], revision="def")}
            r = bc.llama_completion("http://127.0.0.1:1", body, Pkg(ST))   # cached
            self.assertIn("tokens", r)
            with self.assertRaises(SystemExit):   # another revision is a miss: it needs a server
                bc.llama_completion("http://127.0.0.1:1", body, Pkg(other))
        finally:
            srv.shutdown()
            os.environ.pop("LLAMA_CACHE_BUILD")


@need_bc
class VisionReference(unittest.TestCase):
    def measure(self, cfg, has_independent_alt=False, passes_on=None):
        import vision_golden as vision

        def read_reference(*args, salt="", server=None):
            return {"tokens_evaluated": 10, "content": server or salt}

        def judge(pkg, ref, got, views, engine):
            views([])
            status = "PASS" if ref["content"] == passes_on else "FAIL"
            return status, f"{engine} and Loom diverged"

        with tempfile.TemporaryDirectory() as temp, contextlib.chdir(temp), contextlib.redirect_stdout(io.StringIO()) as stdout, \
                mock.patch.dict(os.environ, {"REFERENCE_ALT": "", "REFERENCE_CHECKPOINT": ""}), \
                mock.patch.object(sys, "argv", ["vision_golden.py", "pkg"]), \
                mock.patch.object(bc, "Package", return_value=Pkg(cfg)), \
                mock.patch.object(bc, "cached", return_value=has_independent_alt) as cached, \
                mock.patch.object(vision, "cases", return_value={"chart": ("data:,fake", "question", (1, 1))}), \
                mock.patch.object(vision, "reference_body", return_value={}), \
                mock.patch.object(vision, "reference", side_effect=read_reference) as references, \
                mock.patch.object(vision, "prefill_view", return_value=None) as prefill, \
                mock.patch.object(vision, "judge", side_effect=judge), \
                mock.patch.object(vision, "request", return_value={}), \
                mock.patch.object(vision.requests, "get") as models:
            models.return_value.json.return_value = {"data": [{"id": "loom"}]}
            with self.assertRaises(SystemExit) as exit_status:
                vision.main()
            row = json.loads(Path("vision_golden.json").read_text())[0]
        summary = json.loads(stdout.getvalue().split("VISION_RESULT ")[-1])
        return exit_status.exception.code, summary, row, references.call_args_list, prefill.call_args_list, cached.call_args.args[-1]

    def test_vllm_failure_stays_visible_and_never_reuses_an_encoder_alias(self):
        rc, summary, row, references, prefill, salt = self.measure(ST)
        self.assertEqual((rc, summary["verdict"], summary["reference_engine"]), (1, "FAIL", "vLLM"))
        self.assertIn("vLLM", row["detail"])
        self.assertEqual(len(references), 1)
        self.assertEqual([call.args[-1] for call in prefill], [None])
        self.assertEqual(salt, "vllm-independent")

    def test_independent_vllm_and_llamacpp_encoder_alternates_are_distinct(self):
        rc, summary, row, refs, prefill, _ = self.measure(ST, has_independent_alt=True, passes_on="vllm-independent")
        self.assertEqual((rc, summary["alt_decided"], refs[-1].kwargs), (0, ["chart"], {"salt": "vllm-independent"}))
        self.assertTrue(all(call.args[-1] is None for call in prefill))
        self.assertIn("vLLM's re-recording", row["detail"])
        rc, summary, row, refs, prefill, _ = self.measure({"gguf": "model.gguf"}, passes_on="cpu-encoder")
        self.assertEqual((rc, summary["reference_engine"], refs[-1].kwargs), (0, "llama.cpp", {"server": "cpu-encoder"}))
        self.assertIn("cpu-encoder", [call.args[-1] for call in prefill])
        self.assertIn("image encoder on the CPU", row["detail"])


if __name__ == "__main__":
    unittest.main()
