"""JoyAI checkpoint tuning, supported MLA geometry, and explicit decode-context rows."""

import contextlib
import importlib.util
import io
import json
import re
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PORT = ROOT / "skills/loom-port-llm/port.sh"
EXAMPLE = PORT.parent / "reference/deepseek2-example"
PERF = ROOT / "skills/loom-benchmark/scripts/perf.py"
SLUG = "jdopensource-JoyAI-LLM-Flash-loom"
REPO = "jdopensource/JoyAI-LLM-Flash-GGUF"
CHECKPOINT = "JoyAI-LLM-Flash-IQ4_XS.gguf"


class Port(unittest.TestCase):
    def port_config(self, cfg, repo, override=""):
        text = PORT.read_text(encoding="utf-8")
        script = text.split('"$T" - "${SLUG:-}" "$repo" <<\'PY\'\n', 1)[1].split("\nPY\n", 1)[0]
        with tempfile.TemporaryDirectory() as root:
            config = Path(root, "cfg.json")
            original = json.dumps(cfg, separators=(",", ":")).encode()
            config.write_bytes(original)
            subprocess.run([sys.executable, "-", override, repo], input=script, text=True,
                           cwd=root, capture_output=True, check=True)
            return json.loads(config.read_bytes()), config.read_bytes() == original

    def test_only_the_measured_joyai_checkpoint_opts_in_automatically(self):
        cfg = {"arch": "deepseek2", "gguf": f"gguf/{CHECKPOINT}"}
        saved, unchanged = self.port_config(cfg, REPO)
        self.assertFalse(unchanged)
        self.assertEqual(saved, {**cfg, "slug": SLUG, "prefill_capacity": 8192})
        self.assertEqual(self.port_config({**cfg, "prefill_capacity": 4096}, REPO)[0]["prefill_capacity"], 4096)
        alternative = {**cfg, "gguf": "JoyAI-LLM-Flash-Q4_K_M.gguf"}
        self.assertEqual(self.port_config(alternative, REPO), (alternative, True))

    def test_non_target_configuration_changes_only_with_explicit_override(self):
        cfg = {"arch": "qwen3", "gguf": "Qwen3-0.6B-Q4_K_M.gguf"}
        repo = "unsloth/Qwen3-0.6B-GGUF"
        self.assertEqual(self.port_config(cfg, repo), (cfg, True))
        saved = {**cfg, "slug": "existing"}
        self.assertEqual(self.port_config(saved, repo), (saved, True))
        self.assertEqual(self.port_config(cfg, repo, "explicit")[0], {**cfg, "slug": "explicit"})

    def test_unsupported_geometry_fails_before_lowering(self):
        spec = importlib.util.spec_from_file_location("joyai_gen_mla", EXAMPLE / "gen_mla.py")
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        cfg = json.loads((EXAMPLE / "cfg.json").read_text(encoding="utf-8"))
        self.assertEqual(gen.geometry(cfg)["D"], 576)
        for changed in ({**cfg, "heads": 64}, {**cfg, "mla": {**cfg["mla"], "q_lora": 0}},
                        {**cfg, "mla": {**cfg["mla"], "kv_lora": 256}}, {**cfg, "sliding_layers": [3]}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "unsupported deepseek2 MLA geometry"):
                gen.geometry(changed)


class ContextRows(unittest.TestCase):
    def run_perf(self, value, log):
        bench = types.ModuleType("bench_common")
        bench.doc = lambda size: "x" * size
        bench.thinking_extra = lambda setting: {}
        bench.server_draft_counters = lambda url: {}
        bench.acceptance_between = bench.long_prompt = bench.perf_metrics = lambda *args: None
        bench.accepted_between = bench.per_pass_between = bench.proposals_unreported = lambda *args: None

        def stream(url, model, messages, max_tokens, extra):
            nominal = re.search(r"dctx(\d+)\]", messages[0]["content"])
            prompt = int(nominal[1]) + 23 if nominal else 34
            return {"status": 200, "prompt_tokens": prompt, "ttft": 2, "decode_tps": 42, "tokens": max_tokens}

        bench.stream = stream
        argv = [str(PERF), "http://127.0.0.1:1", "test", "--decode-ctx", value, "--log", str(log), "--no-campaign"]
        with patch.dict(sys.modules, bench_common=bench, requests=types.ModuleType("requests")), patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            runpy.run_path(str(PERF), run_name="__main__")

    def test_same_thousand_has_distinct_keys_and_reports_actual_tokens(self):
        with tempfile.TemporaryDirectory() as root:
            log = Path(root, "perf.jsonl")
            self.run_perf("16000,16500", log)
            row = json.loads(log.read_text(encoding="utf-8"))
            self.assertEqual((row["decode_tps_16k"], row["decode_tps_16500"]), (42, 42))
            self.assertEqual(row["decode_ctx_prompt_tokens"], {"decode_tps_16k": [16023], "decode_tps_16500": [16523]})

    def test_invalid_sizes_are_rejected_without_results(self):
        with tempfile.TemporaryDirectory() as root:
            log = Path(root, "perf.jsonl")
            for value in ("0", "-1", "16,16", "16000,", ""):
                with self.subTest(value=value), self.assertRaises(SystemExit) as error:
                    self.run_perf(value, log)
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(log.exists())
