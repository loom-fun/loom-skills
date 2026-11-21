# Practices checklist: every campaign rule and where it lives in the repo

The proof that nothing the team learned depends on notes outside the repo. "Source" names the origin of the rule (campaign brief,
architecture decisions, standing owner rules); "Where" is the file and section an agent on a fresh clone reads. Paths are relative to
the repo root; `campaign.md`, `targeted-fix.md` and `architecture.md` are in `skills/loom-fleet/reference/`.

## Contents
- Goals and measures
- Architecture decisions
- Team and communication
- Branches, PRs, review
- DevLab and GPU
- Methodology, baselines, quant, models, gates
- Publishing and reporting
- Hygiene
- Optimization
- SDK migrations
- Image ports
- Rules that are project history, not practice

## Goals and measures

| # | Rule | Source | Where |
|---|---|---|---|
| 1 | Separation of concerns across GPU, quant, model type, MoE/dense, family | goal 1 | `architecture.md` "Goals", "Axes" |
| 2 | Sampling-based testing: confidence in a targeted change without a fleet-wide regression | goal 2 | `architecture.md` "Sampled testing"; `targeted-fix.md` section 4 |
| 3 | Scale to every AMD GPU, model and quant | goal 3 | `architecture.md` "Registries" |
| 4 | Efficient runs: instructions as a matrix, only the relevant ones loaded | goal 4 | `skills/loom-kb/SKILL.md` (route, then read); `architecture.md` "Layout" |
| 5 | Velocity: campaign N+1 must be much faster than N, measured and dry-run | goal 5 | `campaign.md` "Velocity and the next-campaign dry run"; `scripts/campaign/status.py` |
| 6 | Guard the architecture against off-topic changes by targeted work | goal 6 | `architecture.md` "Guards"; `skills/loom-kb/SKILL.md` "Mandate rules" |
| 7 | Measures need written evidence: regenerated results equal or better on a supermajority, replayed PRs narrowly scoped, skill structure per best practice, DRY, sampled verdict = full verdict | measures 1-5 | `templates/BRIEF.md` "How success is measured"; `architecture.md` "Sampled testing" (replay check) |
| 8 | Deliverables: one PR into the base from the integration branch, regenerated cards for every model and GPU, final report with evidence | deliverables | `campaign.md` "Roles", "Branches, worktrees, PRs, review"; `templates/BRIEF.md` |
| 9 | The plan is not gospel: replace a structurally wrong part and record why | brief | `campaign.md` "State files" (`DECISIONS.md`) |

## Architecture decisions

| # | Rule | Source | Where |
|---|---|---|---|
| 10 | Invariant: empty fleet program diff before any deliberate change | decisions 0 | `campaign.md` "The invariant"; `architecture.md` "Goals and the invariant" |
| 11 | Layout: fleet tools, one KB skill, device/quant/arch/facet/topic/method nodes; workflow skills hold process only | decisions 1 | `architecture.md` "Layout" |
| 12 | Selectors (device, quant, arch, facet) vs topics (tasks); only selectors carry scope.json | DEV-1 | `architecture.md` "Axes" |
| 13 | One router skill with flat nodes, SKILL.md < 500 lines, nodes < 300 lines + TOC | DEV-2 | `architecture.md` "Layout"; `skills/loom-kb/SKILL.md`; `kb.py lint` |
| 14 | Choice registry with legacy values; resolver precedence; resolved block in the package; hand-set env marks a build dirty | decisions 2 | `architecture.md` "Choice registry and resolver"; `skills/loom-kb/choices.py`, `resolve.py` |
| 15 | New choices are opt-in by scope; per-target and shared-kernel limits | decisions 2 | `targeted-fix.md` section 2; `architecture.md` "Choice registry" |
| 16 | Fleet manifest and programs-only build, byte-identical to a full build; program diff with kernels, targets, rows, strata | decisions 3 | `architecture.md` "Fleet manifest"; `fleet/README.md` |
| 17 | Impact check on every PR (CI) | decisions 3 | `architecture.md` "Guards"; `.github/workflows/impact.yml` |
| 18 | T1 sampled canary: strata, representatives, ABBA, affected rows only, 6+6 near a threshold | decisions 4 | `targeted-fix.md` section 4; `fleet/sample_plan.py` |
| 19 | Per-row verdicts against methodology bands, no geomean trading, bimodal handling | decisions 4 | `targeted-fix.md` section 4; `fleet/verdict.py`; `skills/loom-kb/method/methodology.json` |
| 20 | Promotion ratchet and baseline store; improvements need two confirming runs | decisions 4 | `targeted-fix.md` section 5; `fleet/promote_check.py`, `fleet/baselines.json` |
| 21 | Device, quant and methodology registries; behaviour-preserving | decisions 5 | `architecture.md` "Registries" |
| 22 | loom-scope block in every PR; scope lint rules 1-5; protected core files | decisions 6 | `architecture.md` "Guards"; `fleet/scope_lint.py`, `fleet/ownership.json` |
| 23 | Mandate rules in workflow skills: declare scope first, work inside it, opt in, lint before the PR, stop on out-of-mandate needs | decisions 6 | `skills/loom-kb/SKILL.md` "Mandate rules"; `targeted-fix.md` |
| 24 | Deferred: topology split and capability emitters (revisit before CDNA/XDNA) | decisions 7 | `architecture.md` "Deferred" |
| 25 | Deliberate changes by scope rule, each measured; regressions fixed by scope not global revert | decisions 8 | `architecture.md` "Deliberate behaviour changes"; `targeted-fix.md` section 2 |

## Team and communication

| # | Rule | Source | Where |
|---|---|---|---|
| 26 | Roles: PM, tech lead, engineers, dashboarder | brief | `campaign.md` "Roles" |
| 26a | Tracker artifact: republish only on an important change, at most hourly, diff first, batch the rest | sdk020 (owner) | `campaign.md` "State files"; `templates/BRIEF.md` |
| 27 | Mailbox files and one background waiter; no direct agent messaging, no remote-session tools (they prompt the owner) | brief; owner rule | `campaign.md` "Communication"; `scripts/campaign/waitmail.sh` |
| 28 | Shared state: status.json (locked), log.md, TECHLEAD.md, DECISIONS.md; read TECHLEAD first after a restart | brief; owner rule | `campaign.md` "State files"; `scripts/campaign/status.py`, `new_campaign.sh` |
| 29 | Only the owner's own decisions go to the owner; decide technical details yourself; finish before reporting | brief; owner rule | `campaign.md` "Communication", "Working style" |
| 30 | Under a goal never pause for a go-ahead; a question is not a request | owner rule | `campaign.md` "Communication" |
| 31 | Do not retry a denied action; log and move on; forbid likely-denied actions in briefs | brief; owner rule | `campaign.md` "Communication"; `templates/ENG_CONTRACT.md` |
| 32 | Engineers: task file template, engineer contract, module ownership per wave | brief | `templates/task.md`, `templates/ENG_CONTRACT.md`; `campaign.md` "Branches" |
| 33 | Every spawn/resume/nested delegation follows the exact owner model-family mapping; record actual dispatch and verification | owner rule | `campaign.md` "Model selection and dispatch"; `templates/task.md`, `templates/ENG_CONTRACT.md` |
| 34 | Status page refresh = counts + log + ETA + scope changes | owner rule | `campaign.md` "State files" |
| 35 | Look up an external project or claim yourself before asking | owner rule | `campaign.md` "Working style" |
| 36 | Short replies, answer first | owner rule | `campaign.md` "Communication" |
| 37 | Every methodology improvement lives in the repo, verified by a fresh-clone trial | owner rule | `campaign.md` "Working style"; this file |

## Branches, PRs, review

| # | Rule | Source | Where |
|---|---|---|---|
| 38 | Integration branch; engineer branches `<campaign>-<name>` in own worktrees; PRs only into the integration branch | brief | `campaign.md` "Branches"; `templates/ENG_CONTRACT.md` |
| 39 | PR body: what, why, evidence, loom-scope block, credit line; commit trailer exactly `Co-Authored-By: Claude <noreply@anthropic.com>`; no model names or versions | brief | `campaign.md` "Branches"; `templates/ENG_CONTRACT.md`; `tests/test_campaign_scripts.py` |
| 40 | Merge in-scope PRs that pass review without waiting; failing ones go back; close superseded PRs | owner rule | `campaign.md` "Branches" |
| 41 | External repo changes (loom-dev): open the PR, do not merge, tell the PM | brief | `campaign.md` "Branches" |
| 42 | Tests pass, byte identity proven unless the task says otherwise | brief | `templates/ENG_CONTRACT.md`; `targeted-fix.md` section 3 |

## DevLab and GPU

| # | Rule | Source | Where |
|---|---|---|---|
| 43 | All GPU work on DevLab, never the local GPU, including perf numbers; local GPU never idle on DevLab | brief; owner rules | `campaign.md` "DevLab" |
| 44 | Runner labels (Strix always with `Linux`; R9700 about 3 concurrent) | brief | `campaign.md` "DevLab"; `skills/loom-kb/device/*/*/device.json` |
| 45 | Compare only within one job on one runner; ABBA | brief; owner rule | `campaign.md` "DevLab"; `targeted-fix.md` section 4 |
| 46 | Poll with one background loop >= 120 s; never `gh run watch`; do not re-trigger queued runs | brief | `campaign.md` "DevLab" |
| 47 | Labels and base changes via REST; `gh issue view` fails; conflicted PRs get no CI | brief; owner rule | `campaign.md` "DevLab", "Hygiene" |
| 48 | Active-wait device URI for perf runs (memory fabric clocks) | owner rule | `campaign.md` "DevLab"; `skills/loom-setup/scripts/serve.sh` |
| 49 | Smoke tests with tight timeouts and status logging | owner rule | `campaign.md` "DevLab" |

## Methodology, baselines, quant, models, gates

| # | Rule | Source | Where |
|---|---|---|---|
| 50 | Baselines apples to apples: llama.cpp at defaults, the package's drafting config, no best-of mixing | brief; owner rules | `campaign.md` "Baselines, quant, models, gates"; `skills/loom-benchmark/reference/methodology.md` |
| 51 | No per-model sweeps; Loom tuning counts only when in the package preset | owner rule | `campaign.md` "Baselines, quant, models, gates" |
| 52 | Median of sessions per metric | brief | `skills/loom-benchmark/reference/methodology.md`; `skills/loom-kb/method/methodology.json` |
| 53 | Plain Q4_K_M from the HF cache; no quant changes, no cached-model downloads | brief; owner rule | `campaign.md` "Baselines, quant, models, gates"; `skills/loom/SKILL.md` "Quant" |
| 54 | No models above 35B on a local machine | brief; owner rule | `campaign.md` "Baselines, quant, models, gates" |
| 55 | No needle gates; keep the cheap admission probe; expensive gates nightly | brief; owner rules | `campaign.md` "Baselines, quant, models, gates"; `skills/loom-benchmark/SKILL.md` |
| 56 | Speculative decoding on by default when the GGUF supports it; optimizer picks method and depth | owner rule | `campaign.md` "Baselines, quant, models, gates"; `skills/loom-kb/topic/spec-tuning.md` |
| 57 | Reuse stored measurements when model files and engine binary are unchanged; re-measure only what changed | owner rule (2026-11-16) | `skills/loom-kb/method/methodology.md` rule 17; `campaign.md` "Baselines, quant, models, gates" |

## Publishing and reporting

| # | Rule | Source | Where |
|---|---|---|---|
| 57 | Publish to the staging org only; never edit production repos | brief | `campaign.md` "Publishing and reporting"; `targeted-fix.md` section 5 |
| 58 | Publishing is the job: no extra approval for staging/org cards; verify the live card after upload | owner rule | `campaign.md` "Publishing and reporting"; `skills/loom-kb/topic/org.md` |
| 59 | Publish partial progress: cards show status, never hold a model back | owner rule | `campaign.md` "Publishing and reporting"; `targeted-fix.md` section 5 |
| 60 | Report outcomes against real targets, not workarounds | brief; owner rule | `campaign.md` "Publishing and reporting", "Working style" |
| 61 | Record velocity per stage; run a next-campaign dry run | brief | `campaign.md` "Velocity"; `templates/BRIEF.md` |

## Hygiene

| # | Rule | Source | Where |
|---|---|---|---|
| 62 | Big files never in `/tmp` or a scratchpad (RAM); work dirs on disk; delete outputs | brief; owner rule | `campaign.md` "Hygiene"; `skills/loom/SKILL.md` "Hard rules" |
| 63 | Watch MemAvailable and anonymous RSS; one build at a time; memory-capped scopes | brief; owner rule | `campaign.md` "Hygiene"; `skills/loom/SKILL.md` "Hard rules" |
| 64 | Regenerable outputs stay out of commits; verify regeneration byte for byte | owner rule | `campaign.md` "Hygiene"; `targeted-fix.md` section 2 |

## Optimization

| # | Rule | Source | Where |
|---|---|---|---|
| 65 | Real profilers first (HAL per-dispatch device timestamps); doubling/ablation only as a cross-check, with its known distortions | owner rule | `skills/loom-optimize/SKILL.md` step 2; `skills/loom-kb/topic/optimize-loop.md` "Profiling", "Ablation: known distortions" |
| 66 | Bucket budgets from the leader's number; judge the bucket, not each kernel | owner rule | `skills/loom-optimize/SKILL.md` steps 1 and 3 |
| 67 | Fuse by default, un-fuse only where measured; price a design against its family's profiled share | owner rule | `skills/loom-optimize/SKILL.md` step 4; `skills/loom-kb/topic/optimize-loop.md` "Pricing a design" |
| 68 | Perf-only iteration rounds, gates once on the winner, cached reference engines, A/B base pinned to a sha | owner rule | `skills/loom-optimize/SKILL.md` steps 6 and 7; `campaign.md` "DevLab" |
| 69 | Cost = agent tokens + runner time only, never human effort; model family follows the exact campaign mapping, alternative designs in parallel | owner rule | `skills/loom-optimize/SKILL.md` "Cost model"; `campaign.md` "Model selection and dispatch", "Working style" |
| 70 | Keep going past a losing design: a lane ends when every lever has a measured stop reason; stop rules with numbers | owner rule | `skills/loom-optimize/SKILL.md` "Keep going", "When to stop a metric" |
| 71 | Every technique tried (win or loss) is a row of the technique log; facts go to the KB node; SDK limits to topic `sdk` | owner rule | `skills/loom-optimize/SKILL.md` step 8; `skills/loom-kb/topic/sdk.md` |

## SDK migrations

| # | Rule | Source | Where |
|---|---|---|---|
| 72 | A new SDK release is migrated by one procedure with a version entry per adjacent pair: research first (release, migration table, upstream diff, the wheel), mechanical steps by idempotent script, fleet parse + link sweep, explained program diff, DevLab SDK A/B (old published card on the old SDK vs new package, ABBA, one job), staging publish | owner rule (sdk020) | `skills/loom-migrate/SKILL.md`; `skills/loom-migrate/versions/`; `scripts/sweep.py` |
| 73 | Simplifications the release allows go in a ledger (landed or listed, measured when behavioural); optimizations it opens become issues with evidence, not work in the migration campaign | brief (sdk020) | `skills/loom-migrate/SKILL.md` steps 8 and 9 |
| 74 | Every gotcha found is one line (symptom, cause, fix) in the version entry; a fresh agent migrates one card from the repo alone before the entry is final | brief (sdk020) | `skills/loom-migrate/versions/v0.1.0-to-v0.2.0.md` "Gotchas"; `skills/loom-migrate/SKILL.md` step 10 |

## Image ports

| # | Rule | Source | Where |
|---|---|---|---|
| 75 | An image port is judged against diffusers started from Loom's own Philox noise: PSNR and LPIPS within the bf16 reference's spread of fp32, before any digest is pinned | owner rule | `skills/loom-port-image/SKILL.md` steps 1, 5 and 6; `scripts/philox_noise.py`, `scripts/oracle_diffusers.py` |
| 76 | Image backends are timed in block-alternating sessions (one resident model, block order rotated per repeat, cases seed-major), with the same size, steps and guidance sent to every backend | owner rule | `skills/loom-port-image/SKILL.md` "Timing fairly"; `scripts/bench_images.py --alternate` |
| 77 | Backends that draw their own noise are compared on CLIPScore and PickScore with a paired bootstrap interval, never on PSNR or LPIPS | owner rule | `skills/loom-port-image/SKILL.md` step 6; `scripts/metrics.py scores --baseline` |
| 78 | Image timings stay in the porter's local logs; library cards, Hugging Face READMEs and pull requests carry accuracy only | owner rule | `skills/loom-port-image/SKILL.md` steps 7 and 9, "Timing fairly"; `reference/card-template.md` |

## Rules that are project history, not practice

Campaign-specific state (which models, which PRs, the Windows red-team harness branch, a particular GPU's status) is not a rule and is
deliberately not ported; a new campaign records its own in its `status.json` and `DECISIONS.md`.
