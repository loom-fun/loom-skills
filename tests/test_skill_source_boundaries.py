"""Regression coverage for the supported public skill input boundaries."""
from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest
from urllib.request import Request


ROOT = Path(__file__).resolve().parents[1] / "skills"


def load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuildDeclarationTests(unittest.TestCase):
    def setUp(self):
        self.runner = load("skill_kernel_tests", "loom-benchmark/scripts/kernel_tests.py")

    def test_expands_library_and_declared_test_variants(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            (root / "kernel.loom").touch()
            (root / "BUILD.bazel").write_text('''
load("//tools:defs.bzl", "loom_test", "loom_kernel_library")
filegroup(name="source", srcs=["kernel.loom"])
loom_kernel_library(name="kernel", srcs=[":source"])
VARIANTS = {"case" + str(i): {"configs": {"n": i}} for i in range(2)}
loom_test(name="shape", srcs=["kernel.loom"], deps=[":kernel"], variants=VARIANTS,
          execution_profiles=[VM_REFERENCE_PROFILE])
''', encoding="utf-8")
            build = self.runner.Build(work)
            declarations = build.package("")
            self.assertEqual(build.files(declarations["lib"]["kernel"]["srcs"], ""),
                             [(str((root / "kernel.loom").resolve()), "kernel.loom")])
            self.assertEqual(declarations["test"]["shape"]["profiles"], ["VM_REFERENCE_PROFILE"])
            self.assertEqual(declarations["test"]["shape"]["variants"],
                             {"case0": {"configs": {"n": 0}}, "case1": {"configs": {"n": 1}}})

    def test_reports_unsupported_declarations(self):
        with self.assertRaisesRegex(ValueError, "unsupported BUILD declaration Assert"):
            self.runner.declarations("assert True", "BUILD.bazel", {}, {})


class HubBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.hub = load("skill_hf_hub", "loom-fleet/scripts/hub.py")

    def test_authorization_is_retained_only_on_the_hub_origin(self):
        req = Request("https://huggingface.co/loom-fun/demo/resolve/main/info.json")
        req.add_unredirected_header("Authorization", "Bearer fixture-only")
        handler = self.hub.HubRedirects()
        same = handler.redirect_request(req, None, 302, "Found", {}, "https://huggingface.co/api/resolve-cache/example")
        cdn = handler.redirect_request(req, None, 302, "Found", {}, "https://cdn-lfs.huggingface.co/example")
        self.assertEqual(same.get_header("Authorization"), "Bearer fixture-only")
        self.assertIsNone(cdn.get_header("Authorization"))


class PromotionPathTests(unittest.TestCase):
    def test_accepts_package_slug_and_rejects_nested_repo_id(self):
        with tempfile.TemporaryDirectory() as work:
            source = ROOT / "loom-benchmark/scripts/card"
            sys.path.insert(0, str(source))
            try:
                promote = load("skill_promotion", "loom-benchmark/scripts/card/promote_staging.py")
            finally:
                sys.path.remove(str(source))
            self.assertEqual(Path(promote.package_work_dir(work, "Model-1.0")), Path(work).resolve() / "Model-1.0")
            with self.assertRaisesRegex(ValueError, "invalid package slug"):
                promote.package_work_dir(work, "owner/model")
