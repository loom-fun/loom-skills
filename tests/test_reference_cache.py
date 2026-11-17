"""reference_cache.py: restore / merge / count and committed reference-store invariants."""
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts"))
import reference_cache as rc  # noqa: E402

REPO, VER = "org/model", "vllm-0.18.0"


def make_tgz(path, entries, extra=()):
    with tarfile.open(path, "w:gz") as t:
        for n, data in {**entries, **dict(extra)}.items():
            ti = tarfile.TarInfo(n)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))


class Store(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.store = os.path.join(self.d, "store")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.d, True)

    def job(self, name, keys, ref=None):
        p = os.path.join(self.d, name)
        ents = {f"llama_cache/{k[:2]}/{k}.json": json.dumps({"k": k}).encode() for k in keys}
        if ref:
            ents["reference.json"] = ref
        make_tgz(p, ents, extra={"../evil.json": b"x", "llama_cache/ab/../../x.json": b"x", "other.txt": b"x"})
        return p

    def test_merge_restore_count(self):
        a = self.job("a.tgz", ["aa11", "bb22"], b'{"v": 1}')
        b = self.job("b.tgz", ["bb22", "cc33"], b'{"v": 2}')
        self.assertEqual(rc.merge(REPO, VER, [a], self.store)[1], 3)   # 2 entries + reference.json
        rc.merge(REPO, VER, [b], self.store)
        bench = os.path.join(self.d, "bench")
        self.assertEqual(rc.restore(REPO, VER, bench, self.store), 3)
        self.assertEqual(rc.count(bench), 3)
        self.assertEqual(json.loads(open(os.path.join(bench, "reference.json")).read()), {"v": 2})
        self.assertFalse(os.path.exists(os.path.join(self.d, "evil.json")))   # only llama_cache/xx/key.json + reference.json are taken
        self.assertEqual(rc.count(rc.store_file(REPO, VER, self.store)), 3)

    def test_miss_is_zero_and_version_keyed(self):
        self.assertEqual(rc.restore(REPO, VER, os.path.join(self.d, "b"), self.store), 0)
        rc.merge(REPO, VER, [self.job("a.tgz", ["aa11"])], self.store)
        self.assertEqual(rc.restore(REPO, "vllm-0.19.0", os.path.join(self.d, "b"), self.store), 0)

    def test_llama_engine_key(self):
        rc.merge(REPO, "llama-b1375", [self.job("a.tgz", ["aa11"])], self.store)
        self.assertTrue(rc.store_file(REPO, "llama-b1375", self.store).endswith("org__model__llama-b1375.tgz"))
        self.assertEqual(rc.restore(REPO, "llama-b1375", os.path.join(self.d, "b"), self.store), 1)
        self.assertEqual(rc.restore(REPO, "llama-b1376", os.path.join(self.d, "b2"), self.store), 0)

    def test_merge_is_deterministic(self):
        a = self.job("a.tgz", ["aa11", "bb22"])
        rc.merge(REPO, VER, [a], self.store)
        first = open(rc.store_file(REPO, VER, self.store), "rb").read()
        rc.merge(REPO, VER, [a], self.store)
        self.assertEqual(open(rc.store_file(REPO, VER, self.store), "rb").read(), first)


class Committed(unittest.TestCase):
    def test_committed_store_is_well_formed(self):
        fs = [f for f in os.listdir(rc.STORE) if f.endswith(".tgz")] if os.path.isdir(rc.STORE) else []
        self.assertTrue(fs, "the MXFP4 checkpoint's references are committed")
        for f in fs:
            self.assertRegex(f, r"^[^_].*__.*__(vllm-[0-9][0-9.]*|llama-b[0-9]+)\.tgz$")
            p = os.path.join(rc.STORE, f)
            self.assertLess(os.path.getsize(p), 5_000_000)   # a small tgz, not a model
            ents = rc._entries(p)
            self.assertGreater(rc.count(p), 10)
            if "__vllm-" in f:
                ref = json.loads(ents["reference.json"])
                self.assertEqual(ref["engine"], "vllm")
                self.assertEqual(f.split("__vllm-")[1][:-4], ref["version"])
            else:
                # GGUF gates cache llama.cpp responses directly; only the
                # vLLM shim emits a separate reference.json identity record.
                for name, data in ents.items():
                    if name.startswith("llama_cache/"):
                        record = json.loads(data)
                        self.assertIsInstance(record["tokens"], list)
                        self.assertTrue(all(type(token) is int for token in record["tokens"]))


if __name__ == "__main__":
    unittest.main()
