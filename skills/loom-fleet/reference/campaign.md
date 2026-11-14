# Running a campaign (a new GPU, many models, a refactor)

A campaign is a goal too big for one agent: a team under one tech lead, state in files, GPU work on DevLab. Start it with
`scripts/campaign/new_campaign.sh DIR NAME INTEGRATION_BRANCH BASE_BRANCH` (next to this skill's `SKILL.md`), then fill `BRIEF.md`.
Narrow changes inside a campaign are [targeted fixes](targeted-fix.md).

## Contents
- Roles
- State files
- Communication
- Branches, worktrees, PRs, review
- The invariant
- DevLab
- Baselines, quant, models, gates
- Publishing and reporting
- Hygiene: disk, memory, gh
- Velocity and the next-campaign dry run
- Working style

## Roles

| Role | Does | Never |
|---|---|---|
| PM (main session) | talks to the owner; relays instructions into inbox files; the only agent that asks the owner | review code, merge, spawn engineers |
| Tech lead | owns the integration branch; designs, splits work, writes task files, spawns engineers, reviews and merges their PRs, merges the base in, runs regeneration and publishing, writes the final report, opens the final PR | wait for the owner's go-ahead on technical details |
| Engineers | one task at a time in their own worktree; PR against the integration branch; read `notes/ENG_CONTRACT.md` and their task file first | open PRs into the base or another branch, publish outside the task |
| Dashboarder | keeps the tracker artifact current from `status.json` + `log.md` (counts, log entries, ETA, scope changes together, not counts alone) | edit code or cards |

Launch every subagent on the cheaper model tier (engineers, reviewers, dashboarder); the tech lead uses the strongest. Briefs for
engineers are explicit and self-contained: they point at the files to read, never assume the conversation.

## State files (all in the campaign directory, all on disk, none in `/tmp`)

| File | Owner | Holds |
|---|---|---|
| `BRIEF.md` | PM / tech lead | goal verbatim, measures, deliverables, roles, binding rules; read fully by every agent |
| `DECISIONS.md` | tech lead | design decisions and deviations from the plan, each with the reason (plans are not gospel: replace a structurally wrong part and record why) |
| `TECHLEAD.md` | tech lead | its own handoff: branches, engineers in flight, next waves; read first after a restart or compaction |
| `status.json` | tech lead (`status.py set`, locked) | see below |
| `log.md` | everyone (`status.py log WHO TEXT`) | append-only, one line per event: `YYYY-MM-DD HH:MM who: what` |
| `inbox/<name>.md` | the writers | mailboxes: `techlead`, `pm`, `dashboard`, `eng-<id>` |
| `notes/` | tech lead | `ENG_CONTRACT.md`, `task_<id>.md` per engineer, recipes |
| `eng/<id>/`, `wt/<id>/` | engineers | big outputs and worktrees (see Hygiene) |

`status.json`:
```json
{"workstreams": {"<id>": {"title": "", "owner": "", "stage": "", "pr": 0, "note": "", "updated": ""}},
 "models": {"<slug>": {"<device>": {"stage": "", "verdict": "", "run": ""}, "staging_url": "", "note": ""}},
 "measures": {"m1": {"status": "", "evidence": ""}},
 "velocity": {"stages": {"<stage>": {"start": "", "end": "", "agent_hours": 0, "runner_hours": 0, "note": ""}}}}
```
`python3 status.py --dir DIR set workstreams.guards.stage '"review"' workstreams.guards.pr 716`. The tracker artifact is refreshed with
counts, the log, an ETA and scope changes after every merged PR or when the done count moves by 2 or more.

## Communication

- Write to a mailbox by appending a dated line to `inbox/<name>.md`. To wait for mail run ONE background command
  `DIR/waitmail.sh <name> DIR` and read its output when it exits; read your own inbox between steps and obey `HALT`.
- **Agents never message each other directly and never use remote-session or messaging tools**: they raise a permission prompt on the
  owner's phone for trivial coordination. Mailbox files only; only the PM talks to the owner.
- Anything only the owner can decide or do (spending, scope, external commitments, an action outside the brief): one line in
  `inbox/pm.md`. Technical details are decided by the agent that has the context; ask only when blocked.
- Do not retry a denied action: log it and move on (repeated denials pause the whole session). Read files outside the worktree
  with the shell. Briefs forbid likely-denied actions up front.
- Under a stated goal, never pause for a go-ahead, never end a turn with "should I proceed?". Answer a question briefly (a question
  is not a request: start no new work because someone asked "why") and keep going. Bring the owner finished results or decisions only
  they can make; keep replies short, the answer first.

## Branches, worktrees, PRs, review

- Integration branch `<campaign>` from the base. Engineer branch `<campaign>-<name>` in `wt/<name>`
  (`git worktree add -b <campaign>-<name> DIR/wt/<name> origin/<campaign>`). Never work in the main checkout.
- PRs only into the integration branch, never the base or another feature branch; the tech lead opens the one final PR into the base.
  PR body: what, why, evidence, a `loom-scope` block ([architecture.md](architecture.md) "Guards"), the closing credit line. Commit
  trailer: the author's own credit line; no model ids in commit or PR text.
- Each PR is focused; the tech lead squash-merges; never stack PRs: branch off the latest integration branch and merge it back in when
  it moves. A PR with merge conflicts gets no CI runs.
- **Review, then merge without waiting for the owner**: an in-scope PR that passes review (DRY, meets its task's "Done when", scoped,
  tests, no duplicated package contents) is merged by the tech lead; a failing one goes back to its author. Gate-rule changes get a
  strict review and never loosen a gate. Closing superseded drafts and temporary evidence PRs is part of the job; at the end every PR
  opened is merged or closed. Changes to external repos (for example loom-dev): open the PR, do not merge it, tell the PM.
- Task file = `templates/task.md`, one per engineer: read-first list, branch, scope block, numbered build items, proof, "Done when".
  Engineer contract = `templates/ENG_CONTRACT.md` (copied into `notes/` by `new_campaign.sh`). Waves: engineers whose work does not
  overlap run concurrently, each with a module ownership list; dependent work launches when its inputs merge.

## The invariant

Before any deliberate behaviour change, the **fleet program diff** (all published models, programs + `model.json`, both targets)
between the integration branch and its base is empty. Every deliberate change is then (1) a scope rule, (2) shown by the fleet diff to
touch exactly the models it declares, (3) A/B'd on DevLab. Refactors prove byte identity first; behaviour changes come later and
separately. Tools: `fleet/README.md`; process: [targeted-fix.md](targeted-fix.md).

## DevLab

- **All GPU work runs on DevLab Dispatch (GitHub Actions self-hosted runners), including every performance number. Never the local
  GPU** (it is scarce, sequential and shared). Where it must be decided locally, keep the local GPU busy only with agent-in-the-loop
  work, never idle waiting on DevLab; local and DevLab work are decoupled.
- Runner labels: Strix Halo `[strix-halo, devlab-dispatch, Linux, self-hosted]` (always include `Linux`); Radeon R9700
  `[self-hosted, devlab-dispatch, radeon-r9700]` (about 3 concurrent runners, 31 GB host RAM, no HIP headers). Other devices: the
  `runner.labels` field of `skills/loom-kb/device/<family>/<device>/device.json`.
- Iterate by labelling a draft PR: `devlab-perf` (workflow `devlab-perf.yml`, jobs from `.github/devlab/perf.json`), `devlab-debug`,
  `nightly-gates`. Labels and base changes go through the REST API (`gh pr edit` fails in this repo; re-adding a present label
  fires no event, so remove then add): `gh api -X POST repos/loom-fun/loom-skills/issues/N/labels -f 'labels[]=devlab-perf'`.
  `workflow_dispatch` works only once the workflow file is on the default branch.
- **Compare only within one job on one runner** (runners differ from each other and from any dev machine). Old and new variants
  interleave ABBA in the job. Loom perf runs serve with the active-wait device URI (`completion_spin_ns`, the default of `loom-setup/scripts/serve.sh`):
  the runners' memory fabric clocks otherwise stay low while Loom parks its CPU.
- Wait with ONE background until-loop on `gh run view RUN --json status`, **no faster than every 120 s**; never `gh run watch`;
  one central poller, not one per agent. Queue wait on a runner pool (10+ min on the R9700) is normal: do not re-trigger.
  Download with `gh run download RUN -D FRESH_DIR` (a fresh directory each time; unzip collides).
- Smoke tests use tight timeouts and log the server status endpoints while waiting, so a hang shows its cause.

## Baselines, quant, models, gates

- **Baselines are apples to apples and out of the box**: llama.cpp at build defaults plus the standard flags, in the same job as Loom,
  with the package's drafting config on every row (4 streams: both off); no per-model sweeps, no best-of-configs mixing. Loom tuning
  counts only when baked into the package preset. Full rules: `skills/loom-benchmark/reference/methodology.md` and
  `skills/loom-kb/method/methodology.md`.
- **Quant**: plain Q4_K_M from the HF cache; never switch to another quant on your own, never download a GGUF that is cached under any
  repo, no big downloads without need. If the kit lacks a format Q4_K_M needs, add the format.
- **Models**: nothing above 35B parameters on a local machine (MoE counts total); the shared box has room for one small build at a time.
- **Gates**: no needle or other very long retrieval gates in the critical path (the cheap native-context admission probe stays);
  speculative decoding is on by default whenever the GGUF supports it (MTP head or matching drafter), the optimizer picks method and
  settings by decode speed, users opt out. Expensive gates run nightly (`nightly-gates.yml`), cards show them as not measured.

## Publishing and reporting

- **Publishing**: only to the staging org unless the task names the production org; never edit production repos in a refactor.
  Verify after every upload that the live card carries the new session and programs hash (`kb.py route --topic org`, "Publishing: verify the live card").
  **Publish partial progress**: a card with failing gates or a perf flag is published with the evidence as it is, because the record
  matters and the card's fields say what works; keep fixing and republish. Report outcomes against the real targets (native context,
  all gates, target perf), not the stopgaps passed on the way.

## Hygiene: disk, memory, gh

- Big files never go to `/tmp` or an agent scratchpad (usually tmpfs = RAM; 28 GB of GGUF copies once crashed a whole campaign):
  use `DIR/eng/<id>/` on disk, and delete build outputs and kernel-test dumps when done.
- Watch `MemAvailable` (and process anonymous RSS), not only GPU memory: unified memory and CPU-side tables (llama.cpp copies large
  per-layer tables into anonymous RAM) OOM the session. Watchdogs kill below ~16 GiB available; one build at a time per engineer
  unless the task says otherwise; big runs in a memory-capped `systemd-run --user --scope -p MemoryMax=...`.
- Cache work that is reproducible (programs-only fleet builds, header stubs) instead of re-downloading weights.
- `gh` quirks here: labels and base changes via REST; `gh issue view` fails (use `gh api repos/OWNER/REPO/issues/N`); PRs with merge
  conflicts get no CI runs; do not use `gh run watch`.
- Regenerable outputs (anything a generator or build reproduces) do not belong in commits: verify regeneration byte for byte first.

## Velocity and the next-campaign dry run

Every campaign records `status.json` "velocity": per stage start, end, agent-hours, runner-hours and the note on what was automated or
reused. The goal is that campaign N+1 costs clearly less than campaign N, so each campaign ends with a **dry run** of a realistic
next one (a new device row such as gfx1100, or a new model in an existing family): perform it on a branch using only the repo's
files, measure the effort per stage against the recorded one, and list every manual step as a defect to automate. A campaign whose
dry run is not cheaper than its predecessor is not finished.

## Working style

- Decide, don't escalate: pick the option you would recommend, finish, then report the outcome in a line or two.
- When the owner cites an external project, engine or performance claim, find it (clone the repo into an on-disk work directory, read
  its benchmark docs and kernels) and answer with specifics before asking them for details.
- Report outcomes, not workarounds; no justifications of stopgaps in status updates.
- When a practice is learned, land it in this repo (a reference file, a lint, a script) in the same campaign: the next agent starts
  from a fresh clone with no notes. [practices-checklist.md](practices-checklist.md) maps every rule to where it lives.
