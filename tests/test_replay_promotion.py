"""Guard the promotion bundle and receipt behavior without writing to the Hub."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "skills/loom-migrate/scripts/replay_promotion.py"
SPEC = importlib.util.spec_from_file_location("replay_promotion", SCRIPT)
promotion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(promotion)
SLUG = "example-loom-r9700"
STAGE, TARGET, COMMIT = "a" * 40, "b" * 40, "c" * 40
PINS = {"staging_revision": STAGE, "target_revision": TARGET}


def package(root):
    root.mkdir()
    receipt = {"staging_repo": f"loom-fun-staging/{SLUG}", "staging_parent": STAGE}
    files = {
        "model.json": json.dumps({"presets": {"vision-mtp": {}}}),
        "model.json.lock": "{}",
        "info.json": '{\r\n  "repo": "loom-fun-staging/' + SLUG + '"\r\n}\r\n',
        "REPORT.md": "Keep the native-context caveat verbatim.\n",
        ".gitattributes": "*.loom filter=lfs\n",
        "README.md": 'pipx install "hrx-loom[hf]==0.3.0"\n\n```sh\n'
                     f"loom-fetch huggingface:loom-fun-staging/{SLUG}:vision-mtp --output=example\n"
                     "loom-serve --model=example '--device=amdgpu://?completion_spin_ns=4294967295' --workers=8\n"
                     "```\n\nThe published report contains the active caveat.\n",
        "quick-start-qualification.json": json.dumps(receipt),
    }
    for name, content in files.items():
        (root / name).write_bytes(content.encode())
    return {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()}


def bundle(tmp_path, deletion=False):
    source = tmp_path / "source"
    source_hashes = package(source)
    root = tmp_path / "release"
    out = root / "packages" / SLUG
    target_root = tmp_path / "production"
    target_root.mkdir()
    (target_root / ".gitattributes").write_bytes(b"*.bin filter=lfs\n")
    target = {"README.md": "e" * 64, ".gitattributes": promotion.digest(target_root / ".gitattributes")}
    pins = PINS.copy()
    if deletion:
        target["program/retired.loom"] = "f" * 64
        pins["delete_target_files"] = {"program/retired.loom": target["program/retired.loom"]}
        pins["rollback_tag"] = "v0.2.0"
    row = promotion.compose(SLUG, pins, source, source_hashes, target_root, target, out)
    manifest = root / "manifest.json"
    manifest.write_bytes(promotion.canonical({"schema": 1, "packages": {SLUG: row}}))
    return source, root, out, row, promotion.digest(manifest)


def check_composition(case, tmp_path):
    source, root, out, row, sha = bundle(tmp_path)
    assert (out / "README.md").read_bytes() == (source / "README.md").read_bytes().replace(
        b"loom-fun-staging/", b"loom-fun/")
    assert (out / "info.json").read_bytes() == (source / "info.json").read_bytes().replace(
        b"loom-fun-staging/", b"loom-fun/")
    for name in ("model.json", "model.json.lock", "REPORT.md", "quick-start-qualification.json"):
        assert row["source"][name] == row["output"][name]
    assert row["source"][".gitattributes"] != row["output"][".gitattributes"] == row["target"][".gitattributes"]
    assert ".gitattributes" not in row["operations"]
    assert promotion.inspect_bundle(root, sha) == [(SLUG, row)]
    (out / "REPORT.md").write_text("tampered")
    with case.assertRaisesRegex(ValueError, "incorrect bundle file"):
        promotion.inspect_bundle(root, sha)


def check_replay(case, tmp_path):
    _, root, _, row, sha = bundle(tmp_path, deletion=True)
    receipt = tmp_path / "receipts.jsonl"
    head = TARGET
    rollback = "d" * 40
    writes = []

    class Api:
        def model_info(self, repo, revision, token):
            assert token is False
            if revision == "v0.2.0":
                return SimpleNamespace(sha=rollback)
            assert revision == "main"
            return SimpleNamespace(sha=head)

        def create_commit(self, repo, **kwargs):
            nonlocal head
            assert kwargs["parent_commit"] == TARGET
            assert kwargs["revision"] == "main"
            assert sorted(op.path_in_repo for op in kwargs["operations"]) == sorted(row["operations"] + row["deletions"])
            assert [op.path_in_repo for op in kwargs["operations"] if getattr(op, "delete", False)] == row["deletions"]
            writes.append(repo)
            head = COMMIT
            return SimpleNamespace(oid=COMMIT)

    def snapshot(api, repo, revision):
        return root, row["target"] if revision == TARGET else row["output"]

    hub = SimpleNamespace(
        CommitOperationAdd=lambda name, path: SimpleNamespace(path_in_repo=name),
        CommitOperationDelete=lambda name, is_folder: SimpleNamespace(path_in_repo=name, delete=True),
    )
    with patch.object(promotion, "snapshot", snapshot), patch.dict(sys.modules, {"huggingface_hub": hub}), \
            patch.dict("os.environ", {"PROMOTE_CONFIRM": "loom-fun"}):
        promotion.replay(Api(), root, sha, False, None)
        assert not writes
        with case.assertRaisesRegex(ValueError, "rollback tag"):
            promotion.replay(Api(), root, sha, True, receipt)
        assert not writes
        rollback = TARGET
        promotion.replay(Api(), root, sha, True, receipt)
        assert len(writes) == 1
        first = receipt.read_text().splitlines()[0]  # Simulate interruption before the verified receipt.
        receipt.write_text(first + "\n")
        promotion.replay(Api(), root, sha, True, receipt)
        assert len(writes) == 1
        head = "d" * 40
        with case.assertRaisesRegex(ValueError, "main moved"):
            promotion.replay(Api(), root, sha, True, receipt)


def check_provenance(case):
    promotion.inspect_refs("README.md", f"https://huggingface.co/loom-fun-staging/{SLUG}/commit/{STAGE}".encode(), SLUG)
    with case.assertRaisesRegex(ValueError, "unhandled staging reference"):
        promotion.inspect_refs("model.json", f"https://huggingface.co/loom-fun-staging/{SLUG}/resolve/{STAGE}/x".encode(), SLUG)
    with case.assertRaisesRegex(ValueError, "unhandled staging reference"):
        promotion.inspect_refs("REPORT.md", f"https://huggingface.co/loom-fun-staging/{SLUG}/blob/main/REPORT.md".encode(), SLUG)
    proof = {"production_revision": TARGET, "fixed_prompt_replays": {"decode_2k": {
        "arms": {"staging": {"package": f"loom-fun-staging/{SLUG}@{STAGE}"}}}}}
    promotion.inspect_refs("sdk-comparison-strix.json", promotion.canonical(proof), SLUG)
    proof["extra"] = f"loom-fun-staging/{SLUG}@{STAGE}"
    with case.assertRaisesRegex(ValueError, "unhandled staging reference"):
        promotion.inspect_refs("sdk-comparison-strix.json", promotion.canonical(proof), SLUG)
    retired = {"program/kernel/continuation.loom": "f" * 64}
    promotion.check_deleted_refs("model.json", b'["program/motif/continuation.loom"]', retired, SLUG)
    with case.assertRaisesRegex(ValueError, "references deleted target file"):
        promotion.check_deleted_refs("program/entry.loom", b'"kernel/continuation.loom"', retired, SLUG)


class ReplayPromotionTests(unittest.TestCase):
    def test_production_composition_preserves_bespoke_bytes(self):
        with TemporaryDirectory() as directory:
            check_composition(self, Path(directory))

    def test_readonly_and_interrupted_replay(self):
        with TemporaryDirectory() as directory:
            check_replay(self, Path(directory))

    def test_only_immutable_provenance_is_accepted(self):
        check_provenance(self)
