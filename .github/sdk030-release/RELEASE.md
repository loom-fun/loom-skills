# Loom SDK 0.3.0 promotion handoff

The approval scope is the 58 exact package rows in `inputs/eligible.json`. Each row binds an exact staging commit to the reviewed production parent. The three excluded packages and reasons are recorded in `inputs/held.json`. A bundle with any missing or failed row is a rehearsal receipt and cannot be applied.

Use Python 3.11 or newer, `huggingface_hub==0.36.0`, and the GitHub CLI for the public entry points. Download the archive from the reviewed DevLab artifact, compare its SHA256 with the issue, extract it, and run these commands in the extracted directory.

```sh
sha256sum --check archive-files.sha256
manifest="$(sha256sum release/manifest.json | cut -d' ' -f1)"
tags="$(sha256sum tag-plan.json | cut -d' ' -f1)"
python promote-reviewed.py --manifest-sha256 "$manifest" --tag-plan-sha256 "$tags"
```

The default invocation checks the reviewed scope, current staging heads, outgoing tag identities, live production parents and bundle content. Compare its reported parent and tag state with the reviewed evidence. A changed staging head, production parent, tag or reviewed file stops the operation and needs a new review of the affected row.

After the owner explicitly approves this exact manifest and tag plan, use an authorized Hugging Face account and run one guarded package command.

```sh
PROMOTE_CONFIRM=loom-fun python promote-reviewed.py --manifest-sha256 "$manifest" --tag-plan-sha256 "$tags" --apply --receipt production-commits.jsonl
```

The command rechecks all parents before the first write, creates only missing outgoing `v0.2.0` tags at named immutable parents, verifies tag readback, and publishes the reviewed changes with parent guards. Keep `production-commits.jsonl` outside any throwaway scratch directory. Use the same manifest, tag plan and receipt after an interruption. The replay only recovers a missing receipt when the live commit has the expected release marker, reviewed parent and exact byte inventory. Existing release tags are immutable. The production package archive contains no weight downloads.

Follow `github-release-tag.json` for the GitHub release tag. The expected `sdk0.3.0-u0` candidate follows the approved integration merge in [#1366](https://github.com/loom-fun/loom-skills/pull/1366). Verify the actual merge SHA and current remote tags again at release time. The package command does not create a GitHub tag.

When the approved production package reads and outgoing tags are complete, apply the reviewed public entry points using the instructions and checksums in `release-surfaces/RELEASE.md`. The entry points have their own parent guards and post-commit readbacks. Allow the summary and Pages jobs to finish before checking public links.

Read `review.md` for the staging-lane rehearsal results and `review.json` for SHA256, path, Git blob and LFS identities. The reviewed run checked the SDK's actual fetch dry run for each printed preset after transforming the package to production form. It used exact staged payload bytes and changed production references only where required. Historical references with immutable revisions remain in provenance. The package run requires separate owner approval for all write steps.
