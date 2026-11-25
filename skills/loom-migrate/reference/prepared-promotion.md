# Prepare and replay a reviewed promotion bundle

An SDK release moves every eligible staging card to production in one reviewed step. The writer is
`scripts/replay_promotion.py` (package bytes) plus `scripts/tag_org.py --bundle` (outgoing tags); both are general tools
driven only by the release's data files. Release-specific facts (which slugs, which exceptions) never go into code.
Targeted single-card promotions outside an SDK release use `loom-benchmark/scripts/card/promote_staging.py` instead
(`loom-fleet` reference/publish.md).

Requirements: Python 3.11 or later and `huggingface_hub==0.36.0`. Preparation and every check use anonymous Hub reads;
only the owner-approved steps 1 and 3 of the runbook write.

## Inputs (data, frozen before approval)

- `eligible.json`: `{SLUG: {"staging_revision": SHA, "target_revision": SHA, ...}}`, generated from the qualification evidence
  (`fleet/regen.py eligible`). Optional per-package exceptions, each reviewed with the row:
  - `programs_sha256`: the programs hash the qualifying job served (`card_lib.programs_sha256`); checked against the prepared
    package when present, required for image packages (they have no `llm_config.json` or generated `info.json`).
  - `retain_target_files` / `delete_target_files` (`{path: sha256}`): a production-only file kept, or retired. A deletion also
    needs `rollback_tag`, which must be the release's outgoing tag; preparation refuses any remaining reference to a retired file.
  - `image_readme_refresh`: an image card whose production README changed after the reference revision (the four identities).
  Other fields of the evidence row (devices, runs) are informational and do not enter the manifest.
- `held.json`: `{SLUG: {"reason": TEXT, "tag_revision": SHA?}}`: every production repo that is not promoted, with the owner's
  reason. `tag_revision` (the repo's exact current head) still gives it the outgoing tag.

Every production `*-loom` repo (including suffixed names such as `...-loom-r9700`) must be in exactly one of the two files.
A repo published after the snapshot therefore stops the release instead of being left behind; so does a repo that is in the
inputs but no longer in production.

## Prepare (DevLab or any host)

```bash
python skills/loom-migrate/scripts/replay_promotion.py prepare \
  --eligible eligible.json --held held.json --sdk 0.3.0 --outgoing-tag v0.2.0 --bundle release \
  [--loom-fetch "$(command -v loom-fetch)"]
```

Per package it reads the exact staging and production commits (verifying every blob), composes the production form, and
plans the outgoing tag:

- The production form is the staged bytes with exactly two rewrites: the staging repo in the README's `loom-fetch` selectors
  and in the generated `info.json` `repo`. The README must carry the release's exact install pin.
- No file of the production form may name the staging org, and the card's own file-set check
  (`loom-benchmark/scripts/card/card_files.py`, `check_package`) must pass: cards ship only what the card format defines
  (no qualification or provenance JSONs, rehearsal leftovers or process text). There is no per-file stripping: a card that
  fails is fixed on staging and qualified again. A missing check fails closed.
- Hugging Face's `.gitattributes` keeps the production bytes (no upload operation for it).
- The tag plan reads the production parent's SDK pin (`llm_config.json`, else the README install pin for image packages):
  `create` or `already-at-parent` for the outgoing SDK, `already-on-target-sdk` for a parent that already pins the release
  SDK; any other pin, or the outgoing tag on another commit, is refused.
- `--loom-fetch` also runs the SDK's own fetch dry run of every preset of the prepared form (no download).

The bundle holds `manifest.json` (schema 2: release SDK and outgoing tag, inventory, packages with source / target / output
SHA256 maps and operations, held rows, tag plan), the prepared `packages/`, and a generated `RELEASE.md` runbook with the
section 5 order and the manifest SHA256. Hand over the bundle, the manifest SHA256 and the tool commit for review. The
manifest SHA256 is the single identity the owner approves; it covers the tag plan.

## Dry run (read-only, any time)

```bash
python skills/loom-migrate/scripts/replay_promotion.py dry-run --eligible eligible.json --held held.json \
  --sdk 0.3.0 --outgoing-tag v0.2.0 --bundle /disk/scratch/dry [--only SLUG ...] [--skip-card-check]
```

Prepares, prints the tag plan and the read-only replay plan, and lists every failing package instead of stopping at the first.
`--only` (a sample) and `--skip-card-check` (before the card check exists on the branch) produce a bundle that replay refuses
to execute.

## Release (owner's go, in this order: release-checklist.md section 5)

1. Outgoing tags: `tag_org.py --bundle release --manifest-sha256 SHA --dry-run`, then without `--dry-run`. Each repo is tagged
   at its reviewed parent, never at a newer head (`moved` means the parent changed: prepare again); text, image and held repos
   alike; created tags are read back. Rerun until every row is `exists`, `created` or `skip`.
2. Merge the integration branch into loom-skills `main` and tag the merge commit `sdk<new>-uN`.
3. `PROMOTE_CONFIRM=loom-fun python skills/loom-migrate/scripts/replay_promotion.py replay --bundle release --manifest-sha256 SHA --execute --receipt production-commits.jsonl`

Replay verifies before any write: every bundle file against the manifest, the production inventory, each production parent
and its bytes, each staging head (a moved staging head means the reviewed staging card is no longer the newest: review again),
and that every planned outgoing tag resolves to its parent (step 1 first). It then writes one commit per pending repo with the
parent as compare-and-swap, title `Promote reviewed package from staging <staging sha>` and description
`Reviewed promotion manifest SHA256: <sha>`, records the returned commit in the receipt, reads the full inventory back and
records it as verified. Without `--execute` the same command is the read-only check.

## Interruptions and recovery

Keep the receipt with the release artifacts and rerun the same command with the same receipt after any interruption:

- A commit in the receipt is verified again (exact inventory and still the head) and skipped.
- A crash after the Hub accepted a commit but before its receipt line: the live head differs from the reviewed parent, and replay
  recovers it automatically only when that head is this release's commit for this repo: the exact title, the exact manifest
  marker in its description, the reviewed parent as its parent, and the exact prepared byte inventory. It appends a
  `"recovered": true` receipt line and does not commit again.
- Anything else on `main` (a foreign commit, a look-alike with another parent or other bytes) stops the release with
  `reconcile before retry`. Reconcile by hand with the release owner; a changed production parent needs a newly prepared and
  reviewed bundle.
