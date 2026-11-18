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
- Regenerating many cards: `fleet/regen.py`
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
| Dashboarder | keeps the tracker artifact current from `status.json` + `log.md` (counts, log entries, ETA, scope changes together, not counts alone); republishes only on an important change, at most hourly ("State files") | edit code or cards; republish on every change |

## Model selection and dispatch

Use the owner's exact model-family mapping; catalog descriptions such as "strongest available," "complex coding," or "fast and affordable" do
not change it:

| Work | Codex | Claude |
|---|---|---|
| Routine coding, docs, bounded research/review/packaging (default) | Luna | Sonnet |
| Hard coding (for example difficult kernel or architecture design) | Sol | Opus |

These are the allowed families, not examples. Never use Astra or Fable for any role, nested delegation, resume, or fallback. Do not substitute
another family because a catalog describes it as stronger, cheaper, or better suited. Classify the work itself: a large campaign does not make
every bounded task hard coding. Reasoning effort is a separate control; do not infer `ultra` from hard coding, and use it only when the owner
explicitly requests it.

At startup and resume, inspect current host controls/catalog only to choose a currently supported concrete model ID within the mapped family
and to learn which arguments the dispatch API accepts. Record the selected family and ID, task tier and reason, owner preference, and actual
dispatch arguments/result. A changing catalog or model version does not authorize changing families. If the allowed family or required override
is unavailable, disclose that dispatch limitation and continue unaffected work; do not silently substitute. Inherit only when the parent's
concrete family is known, allowed, and appropriate to the task. An unknown inherited identity is unverified and does not count as compliance.

Before every spawn, resume, or nested delegation, check the actual model argument against this policy. Each task payload has its own tier,
reason, selected family/concrete ID, requested supported controls, and verification status; a parent's fields do not cover a child. Pass the
selected model and any separately chosen reasoning value on the actual call. When an override requires a fresh/no-history fork, use it and
provide a self-contained brief with the goal, scope, read-first files, constraints, and deliverables. Capture the accepted agent ID and actual
call result. Report the requested model separately from runtime identity; mark runtime identity `unverified` when the host does not expose it.
On resume, record continuing workers separately from new launches and do not relaunch or interrupt useful in-flight work solely to retier it.

Every task brief includes `Tier`, `Reason`, `Host model`, `Dispatch`, and `Verification`. `Dispatch` records requested controls and the actual
call/result, including unavailable controls. `Verification` records host confirmation or says `unverified` with the limitation. Engineer briefs
are self-contained and point to files to read; never assume the conversation.

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
| `notes/techniques.md` | everyone (append) | optimization campaigns: one row per technique tried, win or loss (`loom-optimize` step 8); input to the skill and KB updates at the end |
| `eng/<id>/`, `wt/<id>/` | engineers | big outputs and worktrees (see Hygiene) |

`status.json`:
```json
{"workstreams": {"<id>": {"title": "", "owner": "", "stage": "", "pr": 0, "note": "", "updated": ""}},
 "models": {"<slug>": {"<device>": {"stage": "", "verdict": "", "run": ""}, "staging_url": "", "note": ""}},
 "measures": {"m1": {"status": "", "evidence": ""}},
 "velocity": {"stages": {"<stage>": {"start": "", "end": "", "agent_hours": 0, "runner_hours": 0, "note": ""}}}}
```
`python3 status.py --dir DIR set workstreams.guards.stage '"review"' workstreams.guards.pr 716`. The tracker artifact is republished only when
something important happens (a model or workstream changes stage or verdict, a PR merges, a regression or blocker appears, a scope change or an
owner decision) and at most once per hour: diff against the last published state first and batch everything else into the next publish; the
local HTML copy may update freely. Each publish carries counts, the log, an ETA and scope changes together. (The artifact service allows about
200 publishes a day; a dashboarder that republished on every change burned that in under an hour.)

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
  trailer exactly `Co-Authored-By: Claude <noreply@anthropic.com>`; no model names or versions anywhere in commit or PR text.
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
  `[self-hosted, devlab-dispatch, radeon-r9700]` (31 GB host RAM, no HIP headers). The runner count of a device is
  `runner.concurrent_runners` in its `device.json` (the R9700: 8, measured 2026-11-16 over 156 jobs: 5.6 busy on average, queue median
  1 min, p90 20 min; the runners are not the bottleneck, a job's own length is). Other devices: the `runner.labels` field of
  `skills/loom-kb/device/<family>/<device>/device.json`.
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
- Cancelling: a plain `gh run cancel` of a queued self-hosted matrix run can leave it queued or in progress for hours, holding the queue
  (sdk020: 30 cancelled runs blocked the sweep) -> `gh api -X POST repos/OWNER/REPO/actions/runs/RUN/force-cancel`, then confirm with
  `gh run list --status queued` and `--status in_progress` that nothing of yours is left.
- Smoke tests use tight timeouts and log the server status endpoints while waiting, so a hang shows its cause.
- **Job anatomy and the iteration rule.** Measured on 156 R9700 jobs (2026-11-16), a typical 95-minute experiment job is: setup + the 3
  builds 2.5 min (3 %), func 15 min (16 %), ktest 1 min (1 %), perf 9 min (10 %), gates against the vLLM reference about 70 min (72 %),
  including a gate of the unchanged base every time. Cost-wise a round is almost all gates, so:
  - **Iterate on perf only.** An optimization round runs the `iterate` preset (`ab_jobs.py` / `device_jobs.py --preset iterate`,
    `.github/scripts/job_presets.py`): build + ktest + perf + c4 (+ the A/B text match), no gates, and `smoke:V` (one served chat request:
    the model loads and answers) instead of func. About 15 minutes instead of 95.
  - **Gates once, on the winner.** Only the change that is going to merge gets the `final` preset: func + ktest + `gates:V:dev` once on the
    new variant, no perf (the iterations measured it) and the unchanged base variant is not built or gated (a base that already passed is
    never gated again). Hand-written experiment files follow the same two shapes. `full` (the generators' default; `regen.py` always uses it,
    a card needs every step) stays for card regeneration.
  - **Reference engines are invariant inputs: cached, not loaded** (methodology rule 16; the vLLM baseline rows are measured at campaign
    start and in the final card job only, iteration rounds never run vLLM or llama.cpp). A package's gates restore the references recorded
    once per engine version + checkpoint revision + request from the committed `.github/devlab/reference_cache/<org>__<repo>__<ENGINE>.tgz`
    (`vllm-<VERSION>` for a safetensors package, `llama-<BUILD>` for a GGUF card)
    (`skills/loom-benchmark/scripts/reference_cache.py`), so a gate run whose requests are all cached never starts vLLM (the job line
    says `references: N cached before gates, 0 recorded in this job, vLLM loaded: no`). A request that is not cached (a new prompt set,
    a new checkpoint revision, a changed vision case) is recorded by that job as before; add its `gates_V/llama_cache.tgz` to the store
    with `reference_cache.py merge --repo ORG/REPO --engine ENGINE TGZ` in the same PR. Job field `"reference_cache": "none"` forces a cold
    recording.
    The store is one coherent first recording plus a second, independent one: vLLM's answer to the same request varies between jobs (28 of 101
    keys recorded in different jobs of this campaign differ in their tokens, mostly late near-tie flips), so `merge` only ever adds keys, never
    replaces one, and the store is not a union of every job's tgz. The second recording is the store's form of the `--alt` / golden2 rule: the same
    request under another salt (`bench_common.ALT_SALT_TEXT` "rec2" for golden/compare, `ALT_SALT_VISION` "2" for vision), taken by a DIFFERENT job
    (job `env` `"REFERENCE_ALT": "record"`: gates.sh then runs only golden + vision, requests a second recording only for keys that were in the store
    when the job started, and the job's `llama_cache.tgz` is merged). `compare.py` and `vision_golden.py` score a prompt/case the first recording
    fails against the second one (a cache hit, no engine load), no threshold or near-tie change; `compare.json` rows carry `alt_decided`, the
    summary counts them, and the vision result lists the cases. A fixed pair of recordings also makes the gate deterministic. Agent-turn requests contain Loom's own earlier replies: a change that alters
    Loom's text creates new keys (a miss, recorded by that job); a change that keeps the text identical never misses. The consistency gate asks the reference only to diagnose a concurrent-vs-alone mismatch (scheduling-dependent), so a mismatch
    is the one case in which a gate run on a cached store still loads vLLM.

## Regenerating many cards: `fleet/regen.py`

One command per step, state in one JSON file, resumable; no per-engineer helper scripts. `plan --device D [--mode ab|card] --all --state S.json`
(`--mode card` for a NEW device: one card measurement per model, no old package, no A/B, about 0.6-0.75x the runner time of an A/B job; `ab` regenerates a device
that has one) -> `submit` (experiment branch + `perf.json` + PR + label by REST; the device with the fewest runners first) -> `poll` in a >= 120 s
loop -> `collect` (fresh dir per job, `strix_ab_compare.py`, ratchet verdict) -> `rerun` + `submit` (ONE confirming `-r2` job for a model whose
verdict needs it) -> `ingest` (card from the job; Strix keeps the published table plus the re-measure line) -> `publish --dry-run` / `publish`
(card-only, `loom-fun-staging`). A confirmed regression, failed gate or text difference stops that model in `needs_decision`; the rest go on
(`accept` records a person's near-tie call). `status --status-json` writes `models.<slug>.<gpu>` for the dashboard. Time model: `fleet/README.md`.

**Escalation ladder for a new check or a fleet-wide change** (logits campaign: a 1-model pilot went straight to a 99-job sweep, and the sweep
mostly found bugs in the checks themselves). (1) Offline: `fleet/stage_equiv.py` and `fleet/link_check.py` over the whole fleet build (the fold
proof alone missed a package that did not link). (2) Pilot: one model, both GPUs, every new step. (3) Stratified sample across the axes the change
or check depends on (tool dialect, vocab size, draft kind, dense/MoE, model size vs VRAM, GPU): `fleet/sample_plan.py`, one job per stratum.
(4) The full per-model sweep only once the sample is clean. Each rung fixes what it finds before the next one starts.

## Baselines, quant, models, gates

- **Baselines are apples to apples and out of the box**: llama.cpp at build defaults plus the standard flags,
  with the package's drafting config on every row (4 streams: both off); no per-model sweeps, no best-of-configs mixing. Loom tuning
  counts only when baked into the package preset. Full rules: `skills/loom-benchmark/reference/methodology.md` and
  `skills/loom-kb/method/methodology.md`.
- **Reuse stored measurements when nothing they measured changed** (methodology rule 17): same model files and same engine binary
  (llama.cpp build, hrx-loom version) on the same GPU type means the stored rows stand. An SDK migration re-measures Loom only; llama.cpp
  and the old card are never re-run except a targeted ABBA confirmation of a row outside its band.
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
files, measure the cost per stage (agent tokens, runner-hours) against the recorded one, and list every manual step as a defect to automate. A campaign whose
dry run is not cheaper than its predecessor is not finished.

## Adding a device row (no new kernels)

For a GPU whose ISA family exists (a discrete RDNA3 card shares gfx11-generic): copy `skills/loom-kb/device/<family>/<device>/device.json` + `README.md`
of a sibling, mark every value not measured as spec, add the device to the family's `family.json` `devices`, and add it to `info_schema.json`
(`devices` enum; a test checks it against the registry). Then: `build_fleet.py` + `diff_fleet.py` against the base must report 0 models touched;
`fit.py --device DEVICE --approx` gives which models fit (no builds; `fleet/fit_device.py` is the same); `new_device_impact.py DEVICE > impact.json; sample_plan.py impact.json --gpus DEVICE` gives the
first measurement wave; `roofline.py --device DEVICE` gives ceilings. The device row is the whole change for the tooling: the workflows read
the runner labels from the registry (`registry.runner_labels()`), `devlab_perf.sh` finds the device of a runner key (`runner_device.py`),
`ab_jobs.py` / `device_jobs.py --device DEVICE` write the measurement jobs (`r9700_jobs.py`, `strix_ab_jobs.py` are wrappers), `fit.py --device DEVICE` writes the
fit table (path: `fit_table`, default `.github/devlab/<device>_fit.json`), `make_info.py build --fit auto` and `build.py --prefill-device DEVICE=N` use the device's
preset suffix. Once a runner exists: fill `runner.labels` (include Linux) and `runner.key` in `device.json`, add its llama.cpp builds under the device in
`method/methodology.json` `baselines` (replacing the explicit `"none"`), then measure. A device with no runner gets no Quick start on a card: it reads "not measured".

## Working style

- **Cost model: agent tokens and DevLab runner-hours are the only currencies.** Never estimate or weigh human implementation effort or
  complexity in briefs, task files, reports or decisions: agents write the code. Default to the most aggressive design (full fusion,
  alternative designs in parallel); spend runner time carefully (kernel-level checks first, one batched end-to-end job per round).
- Decide, don't escalate: pick the option you would recommend, finish, then report the outcome in a line or two.
- When the owner cites an external project, engine or performance claim, find it (clone the repo into an on-disk work directory, read
  its benchmark docs and kernels) and answer with specifics before asking them for details.
- Report outcomes, not workarounds; no justifications of stopgaps in status updates.
- When a practice is learned, land it in this repo (a reference file, a lint, a script) in the same campaign: the next agent starts
  from a fresh clone with no notes. [practices-checklist.md](practices-checklist.md) maps every rule to where it lives.
