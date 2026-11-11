import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

STARTER = Path(__file__).resolve().parents[1] / "skills/loom-port-llm/starter"
GIB = 2**30


def card():
    def files(*names):
        return {"files": [{"path": f"parameter/model-{i}.gguf", "origin": f"@quantized/UD/{n}", "format": "gguf"}
                          for i, n in enumerate(names)]}
    tok = {"text": {"path": "tokenizer.json", "origin": "@checkpoint/tokenizer.json"}}
    svc = lambda **p: {"chat": {"programs": ["source"], "parameters": p, "tokenizers": tok}}
    ex = {"rows": 4, "context_capacity": 16384, "draft_depth": 0}
    return {"origins": {"checkpoint": {"kind": "hf", "repository": "o/Q", "revision": "a" * 40},
                        "quantized": {"kind": "hf", "repository": "o/Q-GGUF", "revision": "b" * 40}},
            "programs": {"source": {"inputs": ["program/a.loom"]}}, "services": {"chat": {"kind": "text"}, "other": {"kind": "text"}},
            "variants": {"t": {"services": svc(model=files("q-1-of-2.gguf", "q-2-of-2.gguf"))},
                         "tm": {"services": svc(model=files("q-1-of-2.gguf", "q-2-of-2.gguf"), draft=files("mtp.gguf"))}},
            "presets": {"chat": {"service": "chat", "variant": "t", "execution": dict(ex),
                                 "memory": {"weights_bytes": 99 * GIB, "workload": "everything"}},
                        "mtp": {"service": "chat", "variant": "tm", "execution": {**ex, "draft_depth": 3}}}}


class CatalogBuild(unittest.TestCase):
    def build(self, memory, *args, **env_extra):
        tmp = Path(tempfile.mkdtemp())
        kit = tmp / "kit"
        subprocess.run(["cp", "-r", str(STARTER), str(kit)], check=True)
        cat = kit / "catalog"
        (cat / "program").mkdir(parents=True)
        (cat / "program/a.loom").write_text("// x\n")
        (cat / "model.json").write_text(json.dumps(card()))
        (cat / "BUILD.bazel").write_text("")
        (cat / "catalog.json").write_text(json.dumps({
            "variant": "t", "preset": "chat", "mtp_variant": "tm", "mtp_preset": "mtp", "context": 65536,
            "memory": {"resident_weights_bytes": 10 * GIB, "mtp_weights_bytes": GIB, "kv_bytes_per_position": 25344,
                       "recurrent_bytes_per_row": 100 * 2**20, "workspace_gib": 2, **memory},
            "sdk": {"hrx_loom": "0.2.0.dev0", "source": "ci", "artifact_id": 1}}))
        gguf = tmp / "gguf"
        gguf.mkdir()
        for n in ("q-1-of-2.gguf", "q-2-of-2.gguf", "mtp.gguf"):
            (gguf / n).write_bytes(b"x")
        (tmp / "cfg.json").write_text(json.dumps({"name": "Fake Model", "gguf": str(gguf / "q-1-of-2.gguf"), "context_length": 262144,
                                                  "chat": {"generation_prompt_plain": "", "call_format": "x",
                                                           "tool_item_prefix": "", "history_think_scope": "all"}}))
        (tmp / "tok.json").write_text('{"added_tokens": []}')
        env = {k: v for k, v in os.environ.items() if k not in ("SPEC", "DRAFT", "CATALOG", "LOOM_MEMORY_CAP_GIB")} | env_extra
        r = subprocess.run([sys.executable, str(kit / "build.py"), str(tmp / "cfg.json"), str(tmp / "pkg"), "--tokenizer",
                            str(tmp / "tok.json"), *args], capture_output=True, text=True, env=env)
        return tmp / "pkg", r

    def test_mtp_package_and_files(self):
        pkg, r = self.build({"over_cap": True, "note": "table read on demand"})
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        m = json.loads((pkg / "model.json").read_text())
        self.assertEqual(list(m["variants"]), ["tm"])
        self.assertEqual(list(m["presets"]), ["mtp"])
        self.assertEqual(list(m["services"]), ["chat"])
        ex = m["presets"]["mtp"]["execution"]
        self.assertEqual((ex["context_capacity"], ex["draft_depth"], ex["max_tokens"]), (65536, 3, 4096))
        names = [e["path"] for e in m["variants"]["tm"]["services"]["chat"]["parameters"]["model"]["files"]]
        self.assertEqual(names, ["parameter/q-1-of-2.gguf", "parameter/q-2-of-2.gguf"])   # their own names, in order
        self.assertTrue((pkg / "parameter/mtp.gguf").is_symlink())
        self.assertEqual(m["variants"]["tm"]["services"]["chat"]["parameters"]["model"]["files"][0]["origin"], "@quantized/UD/q-1-of-2.gguf")
        self.assertFalse((pkg / "BUILD.bazel").exists())
        self.assertTrue((pkg / "model_target.json").exists())
        c = json.loads((pkg / "llm_config.json").read_text())
        self.assertEqual((c["draft"], c["sdk"]["artifact_id"], c["model_id"]), ({"kind": "mtp", "depth": 3}, 1, "Fake-Model"))
        self.assertTrue(c["mtp"]["gguf"].endswith("mtp.gguf"))
        self.assertIn("--memory_bytes=", r.stdout)

    def test_draft_none_serves_the_target_variant(self):
        pkg, r = self.build({}, "--draft", "none", LOOM_MEMORY_CAP_GIB="64")   # the default cap is 60 % of this machine's RAM
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        m = json.loads((pkg / "model.json").read_text())
        self.assertEqual(list(m["variants"]), ["t"])
        self.assertIsNone(json.loads((pkg / "llm_config.json").read_text()).get("draft"))
        self.assertFalse((pkg / "model_target.json").exists())
        self.assertEqual(m["presets"]["chat"]["memory"]["weights_bytes"], 10 * GIB)   # the resident figure, not the catalog's

    def test_budget_above_the_cap_needs_the_explicit_flag(self):
        cap = {"LOOM_MEMORY_CAP_GIB": "8"}
        _, r = self.build({}, "--draft", "none", **cap)
        self.assertEqual(r.returncode, 1)   # 10 GiB resident > the 8 GiB cap, no over_cap
        self.assertIn("does not hold the package's resident set", r.stderr)
        _, r = self.build({"over_cap": True}, "--draft", "none", **cap)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("explicit per-model budget above the default cap", r.stdout)


if __name__ == "__main__":
    unittest.main()
