"""Guard the promotion bundle and receipt behavior without writing to the Hub."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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
        "README.md": 'pipx install "hrx-loom[hf]==0.3.0"\n\n```sh\n'
                     f"loom-fetch huggingface:loom-fun-staging/{SLUG}:vision-mtp --output=example\n"
                     "loom-serve --model=example '--device=amdgpu://?completion_spin_ns=4294967295' --workers=8\n"
                     "```\n\nThe published report contains the active caveat.\n",
        "quick-start-qualification.json": json.dumps(receipt),
    }
    for name, content in files.items():
        (root / name).write_bytes(content.encode())
    return {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()}


def bundle(tmp_path):
    source = tmp_path / "source"
    source_hashes = package(source)
    root = tmp_path / "release"
    out = root / "packages" / SLUG
    target = {"README.md": "e" * 64}
    row = promotion.compose(SLUG, PINS, source, source_hashes, source, target, out)
    manifest = root / "manifest.json"
    manifest.write_bytes(promotion.canonical({"schema": 1, "packages": {SLUG: row}}))
    return source, root, out, row, promotion.digest(manifest)


def test_production_composition_preserves_bespoke_serving_and_receipts(tmp_path):
    source, root, out, row, sha = bundle(tmp_path)
    assert (out / "README.md").read_bytes() == (source / "README.md").read_bytes().replace(
        b"loom-fun-staging/", b"loom-fun/")
    assert (out / "info.json").read_bytes() == (source / "info.json").read_bytes().replace(
        b"loom-fun-staging/", b"loom-fun/")
    for name in ("model.json", "model.json.lock", "REPORT.md", "quick-start-qualification.json"):
        assert row["source"][name] == row["output"][name]
    assert promotion.inspect_bundle(root, sha) == [(SLUG, row)]
    (out / "REPORT.md").write_text("tampered")
    with pytest.raises(ValueError, match="incorrect bundle file"):
        promotion.inspect_bundle(root, sha)


def test_readonly_and_interrupted_replay(tmp_path, monkeypatch):
    _, root, _, row, sha = bundle(tmp_path)
    receipt = tmp_path / "receipts.jsonl"
    head = TARGET
    writes = []

    class Api:
        def model_info(self, repo, revision, token):
            assert revision == "main"
            assert token is False
            return SimpleNamespace(sha=head)

        def create_commit(self, repo, **kwargs):
            nonlocal head
            assert kwargs["parent_commit"] == TARGET
            assert kwargs["revision"] == "main"
            assert sorted(op.path_in_repo for op in kwargs["operations"]) == row["operations"]
            writes.append(repo)
            head = COMMIT
            return SimpleNamespace(oid=COMMIT)

    monkeypatch.setattr(promotion, "snapshot", lambda api, repo, revision:
                        (root, row["target"] if revision == TARGET else row["output"]))
    promotion.replay(Api(), root, sha, False, None)
    assert not writes
    monkeypatch.setenv("PROMOTE_CONFIRM", "loom-fun")
    promotion.replay(Api(), root, sha, True, receipt)
    assert len(writes) == 1
    first = receipt.read_text().splitlines()[0]  # Simulate interruption before the verified receipt.
    receipt.write_text(first + "\n")
    promotion.replay(Api(), root, sha, True, receipt)
    assert len(writes) == 1
    head = "d" * 40
    with pytest.raises(ValueError, match="main moved"):
        promotion.replay(Api(), root, sha, True, receipt)


def test_provenance_accepts_only_immutable_document_references():
    promotion.inspect_refs("README.md", f"https://huggingface.co/loom-fun-staging/{SLUG}/commit/{STAGE}".encode(), SLUG)
    with pytest.raises(ValueError, match="unhandled staging reference"):
        promotion.inspect_refs("model.json", f"https://huggingface.co/loom-fun-staging/{SLUG}/resolve/{STAGE}/x".encode(), SLUG)
    with pytest.raises(ValueError, match="unhandled staging reference"):
        promotion.inspect_refs("REPORT.md", f"https://huggingface.co/loom-fun-staging/{SLUG}/blob/main/REPORT.md".encode(), SLUG)
