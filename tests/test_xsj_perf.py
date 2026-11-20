"""Offline tests of the native Windows gfx1152 benchmark runner."""

import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills/loom-benchmark/scripts"))
import xsj_perf as runner  # noqa: E402


class RunnerTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name)
        self.args = SimpleNamespace(package=self.out, preset="chat-gfx1152", loom="loom-serve.exe",
                                    gguf=self.out / "weights.gguf", rocm="llama-server.exe",
                                    vulkan="llama-server.exe", loom_arg=[], port=8160,
                                    repeat_rows=1, startup_timeout=1, workload_timeout=1)

    def test_engine_modes_match_context_and_load_drafting_only_for_single_stream(self):
        model = self.out / "model.json"
        spec = runner.spec_args("loom-fun/unsloth-Qwen3.5-4B-loom", self.out / "Qwen3.5-4B-DFlash.Q8_0.gguf")
        self.assertEqual(spec[:2], ["--spec-type", "draft-dflash"])
        self.assertEqual(spec[spec.index("--spec-draft-n-max") + 1], "7")
        for engine, device in (("rocm", "ROCm0"), ("vulkan", "Vulkan0")):
            with self.subTest(engine=engine):
                single = runner.server_argv(self.args, engine, "single", model, 32768, spec, device)
                self.assertEqual(single[single.index("--device") + 1], device)
                self.assertEqual(single[single.index("-c") + 1], "32768")
                self.assertEqual(single[single.index("-np") + 1], "1")
                self.assertIn("--spec-type", single)
                four = runner.server_argv(self.args, engine, "nodraft", model, 32768, spec, device)
                self.assertEqual(four[four.index("-np") + 1], "4")
                self.assertEqual(four[four.index("-c") + 1], "32768" if engine == "rocm" else "131072")
                self.assertEqual("-kvu" in four, engine == "rocm")
                self.assertFalse(any(flag.startswith("--spec-") for flag in four))
        loom = runner.server_argv(self.args, "loom", "single", model, 32768, spec)
        self.assertIn("--device=amdgpu://?completion_spin_ns=4294967295", loom)
        self.assertFalse(any(flag.startswith("--workers") for flag in loom))

    def test_baseline_context_is_bounded_on_native_context_models(self):
        for engine, slot, four in (("rocm", "65536", "65536"), ("vulkan", "36864", "147456")):
            with self.subTest(engine=engine):
                for mode, limit in (("single", slot), ("nodraft", four)):
                    argv = runner.server_argv(self.args, engine, mode, None, 262144, [], "GPU0")
                    self.assertEqual(argv[argv.index("-c") + 1], limit)

    def test_alternate_drafter_is_selected_only_by_the_classification(self):
        self.assertEqual(runner.spec_args("loom-fun/unsloth-Qwen3.5-2B-loom", None), [])
        with self.assertRaisesRegex(runner.RunError, "requires Qwen3.5-4B-DFlash.Q8_0"):
            runner.spec_args("loom-fun/unsloth-Qwen3.5-4B-loom", self.out / "Qwen3.5-4B-DFlash-Q8_0.gguf")
        with self.assertRaisesRegex(runner.RunError, "needs --draft"):
            runner.spec_args("loom-fun/unsloth-Qwen3.5-9B-loom", None)

    def test_release_matches_executable_and_sibling_dlls(self):
        archive = self.out / "release.zip"
        exe = self.out / "llama-server.exe"
        dll = self.out / "ggml-hip.dll"
        with ZipFile(archive, "w") as zipped:
            zipped.writestr("some-folder/llama-server.exe", b"server")
            zipped.writestr("some-folder/ggml-hip.dll", b"dll")
        exe.write_bytes(b"server")
        dll.write_bytes(b"dll")
        with patch.object(runner, "ROCM_PIN", hashlib.sha256(archive.read_bytes()).hexdigest()):
            provenance = runner.baseline_identity("rocm", exe, archive)
            self.assertEqual(provenance["build"], "b1382")
            self.assertEqual(provenance["release_dlls"][0]["sha256"], hashlib.sha256(b"dll").hexdigest())
            dll.write_bytes(b"different")
            with self.assertRaisesRegex(runner.RunError, "ggml-hip.dll differs"):
                runner.baseline_identity("rocm", exe, archive)

    def test_nodraft_variant_leaves_original_model_and_programs_alone(self):
        original = self.out / "model.json"
        model = {"presets": {self.args.preset: {"execution": {"draft_depth": 3, "rows": 4}}}}
        original.write_text(json.dumps(model), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "benchmark failed"):
            with runner.undrafted_model(self.args, model, self.out) as variant:
                self.assertEqual(variant.parent, original.parent)
                self.assertEqual(runner.read_json(variant)["presets"][self.args.preset]["execution"],
                                 {"draft_depth": 0, "rows": 4})
                raise ValueError("benchmark failed")
        self.assertFalse(variant.exists())
        self.assertEqual(runner.read_json(original), model)
        self.assertEqual(model["presets"][self.args.preset]["execution"]["draft_depth"], 3)

    def test_sequential_sessions_stop_server_even_on_a_workload_error(self):
        evidence = runner.Evidence(self.out)
        process = Mock(pid=42, returncode=1)
        process.poll.return_value = None
        with patch.object(runner, "listener_pids", return_value=set()), \
                patch.object(runner.subprocess, "Popen", return_value=process), \
                patch.object(evidence, "command") as command:
            with self.assertRaisesRegex(ValueError, "benchmark failed"):
                with evidence.server("loom.single.01", [str(self.out / self.args.loom)], self.args.port):
                    raise ValueError("benchmark failed")
        command.assert_called_once_with("loom.single.01.stop", ["taskkill.exe", "/PID", "42", "/T", "/F"], timeout=30)
        process.wait.assert_called_once_with(timeout=30)
        events = [json.loads(line) for line in (self.out / "commands.jsonl").read_text().splitlines()]
        self.assertEqual([row["event"] for row in events], ["start", "pid", "exit"])

    def test_preexisting_listener_blocks_engine_launch(self):
        evidence = runner.Evidence(self.out)
        with patch.object(runner, "listener_pids", return_value={"9"}), patch.object(runner.subprocess, "Popen") as popen:
            with self.assertRaises(runner.CleanupError):
                with evidence.server("loom.single.01", [str(self.out / self.args.loom)], self.args.port):
                    self.fail("Foreign listener was already present")
        popen.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "exercises native Windows process ownership")
    def test_native_windows_listener_is_owned_and_released(self):
        handler = """
import http.server
import sys
import threading

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

with http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler) as server:
    timer = threading.Timer(15, server.shutdown)
    timer.daemon = True
    timer.start()
    server.serve_forever()
"""
        evidence = runner.Evidence(self.out)
        wrapper = ("import subprocess, sys; raise SystemExit(subprocess.call("
                   f"[sys.executable, '-c', {handler!r}, sys.argv[1]]))")
        for name, code in (("direct", handler), ("child", wrapper)):
            with self.subTest(name=name):
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                with evidence.server(name, [sys.executable, "-c", code, str(port)], port) as server:
                    runner.wait_ready(evidence, name, server, "rocm", f"http://127.0.0.1:{port}", port, None, 10)
                    self.assertIsNone(server.process.poll())
                    self.assertEqual(server.child_listener is not None, name == "child")
                self.assertIsNotNone(server.process.poll())
                self.assertFalse(runner.listener_pids(evidence, f"{name}.verify", port))

    def test_only_complete_session_reaches_ingest_file(self):
        evidence = runner.Evidence(self.out)
        server = Mock()
        server.__enter__ = Mock(return_value=object())
        server.__exit__ = Mock(return_value=False)
        model = self.out / "model.json"
        with patch.object(evidence, "server", return_value=server), \
                patch.object(runner, "wait_ready"), \
                patch.object(runner, "workloads", side_effect=[ValueError("long failed"),
                                                                [{"prefill": []}, {"prefill_long": []}]]):
            with self.assertRaisesRegex(ValueError, "long failed"):
                runner.measure(evidence, self.args, "rocm", "single", "rocm.single.01", model, "id", 32768, [], "ROCm0")
            path = self.out / runner.LOGS["rocm"][0]
            self.assertFalse(path.exists())
            runner.measure(evidence, self.args, "rocm", "single", "rocm.single.02", model, "id", 32768, [], "ROCm0")
        self.assertEqual([json.loads(line) for line in path.read_text().splitlines()],
                         [{"prefill": []}, {"prefill_long": []}, {"session_end": "rocm.single.02"}])
        self.assertEqual(server.__exit__.call_count, 2)

    def test_each_round_walks_engines_and_modes_in_order(self):
        self.args.engine = ["loom", "rocm", "vulkan"]
        self.args.sessions = 2
        manifest = {"engines": {"loom": {}, "rocm": {}, "vulkan": {"status": "unavailable", "reason": "probe"}},
                    "sessions": [], "package": {"model_id": "id"}}
        evidence = runner.Evidence(self.out)
        with patch.object(runner, "measure", return_value=[{}, {}]) as measure:
            self.assertFalse(runner.sessions(self.args, evidence, manifest, 32768, [], "model.json", "nodraft.json"))
        self.assertEqual([c.args[4] for c in measure.call_args_list],
                         [f"{engine}.{mode}.{index:02d}" for index in (1, 2) for engine in ("loom", "rocm")
                          for mode in ("single", "nodraft")])
        self.assertTrue(all(row["status"] == "completed" for row in manifest["sessions"]))


if __name__ == "__main__":
    unittest.main()
