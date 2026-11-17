"""Router test: safetensors / Quark / mxfp4s nodes load for a safetensors source and never for a GGUF config (CPU only)."""
import importlib.util
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KB = REPO / "skills/loom-kb"
spec = importlib.util.spec_from_file_location("kb", KB / "kb.py")
kb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kb)

FIXTURE = REPO / "tests/data/kb_source/mxfp4_safetensors_llm_config.json"
SOURCE_NODES = ("facet/safetensors/", "facet/quark/", "quant/mxfp4s/")
TOPICS = ["", "porting,kernels-ir,program-abi,starter", "prefill", "decode", "benchmark", "speculative,kb-authoring"]


def route(config=None, arch=None, device="radeon-r9700", topic="", extra=()):
    ns = type("A", (), dict(config=config, gguf=None, arch=arch, facets=None, device=device, quant=None, topic=topic,
                            relative=True, model=None))()
    buf = io.StringIO()
    with redirect_stdout(buf):
        kb.route(ns, root=KB)
    return [l for l in buf.getvalue().splitlines() if l and not l.startswith("#")]


class SourceNodeRoutingTests(unittest.TestCase):
    def test_safetensors_mxfp4_config_loads_the_source_nodes(self):
        out = route(str(FIXTURE), topic="porting,benchmark")
        for f in ("facet/safetensors/README.md", "facet/safetensors/porting.md", "facet/safetensors/benchmark.md",
                  "facet/quark/README.md", "quant/mxfp4s/README.md"):
            self.assertIn(f, out)
        self.assertEqual(len(out), len(set(out)))

    def test_task_files_follow_the_topic(self):
        out = route(str(FIXTURE), topic="decode")
        self.assertIn("facet/safetensors/README.md", out)
        self.assertNotIn("facet/safetensors/porting.md", out)
        self.assertNotIn("facet/safetensors/benchmark.md", out)

    def test_every_fleet_config_loads_none_of_them(self):
        manifest = json.loads((REPO / "fleet/manifest.json").read_text())["models"]
        for slug in sorted(manifest):
            cfg = REPO / "fleet" / slug / "llm_config.json"
            data = json.loads(cfg.read_text())
            if any(f.get("kind") == "safetensors" for f in manifest[slug]["files"].values()):   # a safetensors card loads them (below), GGUF cards never do
                self.assertIn("source", data, slug)
                out = route(str(cfg), device="radeon-r9700", topic="porting")
                self.assertIn("facet/safetensors/README.md", out, slug)
                continue
            self.assertNotIn("source", data, slug)
            self.assertNotIn("quark", data, slug)
            for dev in ("strix-halo", "radeon-r9700"):
                for topic in TOPICS:
                    out = route(str(cfg), device=dev, topic=topic)
                    self.assertFalse([f for f in out if f.startswith(SOURCE_NODES)], (slug, dev, topic))

    def test_arch_only_route_never_loads_them(self):
        for arch in ("qwen35", "qwen3", "gemma4"):
            out = route(arch=arch, topic="porting,benchmark")
            self.assertFalse([f for f in out if f.startswith(SOURCE_NODES)], arch)

    def test_gguf_inspector_output_has_no_source_key(self):
        text = (REPO / "skills/loom-port-llm/starter/inspect_gguf.py").read_text()
        self.assertNotIn('"source"', text)
        self.assertNotIn('"quark"', text)


if __name__ == "__main__":
    unittest.main()
