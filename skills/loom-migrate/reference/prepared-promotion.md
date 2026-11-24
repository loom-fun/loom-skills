# Prepare a reviewed promotion bundle

Use `scripts/replay_promotion.py` for an explicitly scoped release. Prepare the production form on DevLab before requesting promotion approval. The preparation preserves the staged README and report prose, serving arguments and qualification files. It changes the staging namespace in the card's fetch selectors and generated `info.json` repository field. Image packages keep their own card layout and are checked against the published image qualification.

From the release tool checkout with Python 3.11 or later and `huggingface_hub==0.36.0`, run:

```bash
python skills/loom-migrate/scripts/replay_promotion.py prepare \
  --pins pins.json --bundle release
```

The pin file maps each approved slug to its exact `staging_revision` and `target_revision`. If a reviewed artifact exists only in production, identify it explicitly with a `retain_target_files` map of path to SHA256. The tool verifies both pinned inventories, rejects unaccounted production-only files and records the input and output hashes and exact upload operations for every package. Include the prepared directory, manifest SHA256, tool commit and separate exact outgoing tag plan in the release handoff. The model repos remain unchanged during preparation.

An operator in a fresh session uses the downloaded bundle and the **reviewed** manifest SHA256 supplied by the release handoff:

```bash
python skills/loom-migrate/scripts/replay_promotion.py replay \
  --bundle release --manifest-sha256 "$REVIEWED_MANIFEST_SHA256"
```

Replay checks every prepared file and production parent before any upload. After explicit approval and verification that the outgoing tags resolve to the reviewed parents, the same command accepts `--execute --receipt production-commits.jsonl` with `PROMOTE_CONFIRM=loom-fun` and an authenticated Hub session. It submits only the reviewed additive or replacement operations with the pinned parent as a compare-and-swap condition. Each returned commit is recorded before full readback. Successful readback requires the exact prepared inventory and SHA256 at that commit and the same commit on main.

Keep the receipt in the operator's durable release artifacts. The same command and same receipt can resume an interrupted replay after verifying previously recorded commits. An unexpected head with no matching receipt stops the release.

If the Hub accepted a commit before its response was recorded, the release owner can recover the receipt through read-only checks. Fetch the candidate commit in a scratch Git repository and confirm that its sole parent equals the manifest's `target_revision` and its commit description contains the exact reviewed manifest SHA256. After the release owner confirms that identity, append one JSON line to the receipt with the exact `repo`, candidate `commit`, `manifest_sha256` and `"verified": false`. Run the read-only replay with that receipt. It checks that the candidate is still on main and that its entire file map matches the approved bundle before any remaining upload is allowed. If any check differs, leave the release paused for a new explicit decision.
