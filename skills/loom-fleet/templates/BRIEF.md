# @CAMPAIGN@: team brief (every agent reads this fully before acting)

Campaign directory: @DIR@. Integration branch: `@INTEGRATION@` (from `@BASE@`). Started @DATE@.
How a campaign runs, and why each rule exists: `skills/loom-fleet/reference/campaign.md` in the loom-skills repo.

## Goal
TODO one paragraph: the owner's goal in their words (new GPU, new models, a refactor ...), then the numbered deliverables.

## How success is measured (each needs written evidence in the final report)
TODO numbered measures. Always include: results equal or better than the published ones on a supermajority of metrics, and the
effort of this campaign against the previous one (velocity, `status.json` "velocity").

## Roles
- **PM** (main session): talks to the owner only; relays instructions into inbox files. Not a reviewer.
- **Tech lead**: owns `@INTEGRATION@`, designs, splits work, spawns engineers, reviews and merges their PRs, merges the base in when
  needed, runs regeneration and publishing, writes the final report, opens the final PR into `@BASE@`.
- **Engineers**: one task at a time, own worktree, PRs against `@INTEGRATION@` only.
- **Dashboarder**: keeps the tracker artifact current from `status.json` + `log.md`; never edits code or cards.

## Communication (strict)
- Mailboxes: `@DIR@/inbox/<name>.md` (techlead, dashboard, pm, eng-<id>): append a dated line to write. To wait for mail run ONE
  background command `@DIR@/waitmail.sh <name> @DIR@`.
- Never message another agent directly (no peer SendMessage) and never use remote-session tools: they prompt the owner.
- State: `status.json` (tech lead writes, `status.py` locks), `log.md` (append-only), `TECHLEAD.md`, `DECISIONS.md`.
- Only the owner can decide/do something: append to `inbox/pm.md`; ask only when it is truly their call. Decide technical details yourselves.

## Binding rules
Copy the rules of `reference/campaign.md` that apply (GPU only on DevLab, labels, baselines, quant, publishing target, disk, gh quirks)
and add this campaign's own. TODO.

## Prior campaign state to reuse
TODO paths (previous status.json, recipes, A/B files for noise calibration).
