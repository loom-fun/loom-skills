# Prepare a reviewed promotion bundle

Use `scripts/replay_promotion.py` for an explicitly scoped release. Prepare the production form on DevLab before requesting promotion approval. The preparation preserves the staged README and report prose, serving arguments and qualification files. It changes the staging namespace in the card's fetch selectors and generated `info.json` repository field. Image packages keep their own card layout and are checked against the published image qualification.

From the release tool checkout with Python 3.11 or later and `huggingface_hub==0.36.0`, run:

```bash
python skills/loom-migrate/scripts/replay_promotion.py prepare \
  --pins pins.json --bundle release
```

The pin file maps each approved slug to its exact `staging_revision` and `target_revision`. Use `retain_target_files` to preserve a reviewed file that exists only in production. If the release explicitly retires a production-only file, list its path and current SHA256 in `delete_target_files` and supply the outgoing `rollback_tag`. Preparation checks for references to retired files. Verify every advertised preset's released SDK fetch plan against those deletions as part of the release receipt. The tool verifies both pinned inventories and refuses unaccounted production-only files. Include the prepared directory, manifest SHA256, tool commit and exact outgoing tag plan in the release handoff. The model repos remain unchanged during preparation.

Hugging Face's `.gitattributes` is included in the source, target and output checksums wherever present. The prepared output preserves the production bytes, with no upload operation for that path.

An operator in a fresh session uses the downloaded bundle and the **reviewed** manifest SHA256 supplied by the release handoff:

```bash
python skills/loom-migrate/scripts/replay_promotion.py replay \
  --bundle release --manifest-sha256 "$REVIEWED_MANIFEST_SHA256"
```

Replay checks every prepared file and production parent before any upload. After explicit approval and verification that the outgoing tags resolve to the reviewed parents, the same command accepts `--execute --receipt production-commits.jsonl` with `PROMOTE_CONFIRM=loom-fun` and an authenticated Hub session. It submits the reviewed operations with the pinned parent as a compare-and-swap condition. A removal also requires the rollback tag to resolve to that exact parent. Each returned commit is recorded before full readback. Successful readback requires the exact prepared inventory and SHA256 at that commit and the same commit on main.

Keep the receipt in the operator's durable release artifacts. The same command and same receipt can resume an interrupted replay after verifying previously recorded commits. An unexpected head with no matching receipt stops the release.

If the Hub accepted a commit before its response was recorded, the release owner can recover the receipt through read-only checks. Fetch the candidate commit in a scratch Git repository and confirm that its sole parent equals the manifest's `target_revision` and its commit description contains the exact reviewed manifest SHA256. After the release owner confirms that identity, append one JSON line to the receipt with the exact `repo`, candidate `commit`, `manifest_sha256` and `"verified": false`. Run the read-only replay with that receipt. It checks that the candidate is still on main and that its entire file map matches the approved bundle before any remaining upload is allowed. If any check differs, leave the release paused for a new explicit decision.
