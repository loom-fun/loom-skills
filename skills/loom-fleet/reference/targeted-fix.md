# Targeted fix: a narrow change to one family, device, quant or model

Use this for "make X faster / fix Y on the R9700 / handle quant Z for the gemma4 family". A change that must reach every model is
`kind: architecture` or `horizontal` and follows [architecture.md](architecture.md) instead; a many-model program of work is a
[campaign](campaign.md). Everything here is process; the facts it points to live in `loom-kb` (`kb.py route` prints them).

## Contents
- The checklist
- 1 Route and declare
- 2 Make the change opt-in by scope
- 3 Prove which models it touches (fleet diff + scope lint)
- 4 Test a sample on DevLab (plan, A/B, verdict)
- 5 Promote, publish
- When to stop

## The checklist (copy it into the task notes; tick as you go)

```
- [ ] 1. kb.py route --model <cfg|slug> --device <device> --topic <topic>; read exactly the printed files
- [ ] 2. Write the loom-scope block (kind, scopes, kernels) FIRST; it is the mandate for every edit
- [ ] 3. Change = a scope.json choice (new key opt-in, or a rule on the node) + the code behind it; legacy defaults untouched
- [ ] 4. Fleet build at base and head, diff_fleet: the touched models are exactly the declared scope
- [ ] 5. scope_lint --local passes (leak, protected files, registry, kb lint); python -m unittest discover -s tests passes
- [ ] 6. sample_plan: representatives per touched stratum and GPU family; DevLab A/B of them (ab_jobs.py --replicas 2 --sequential), the touched GPU only
- [ ] 7. verdict.py per row against methodology.json bands; the job extends itself to 6+6 near a threshold; the pair is the confirming run
- [ ] 8. Text match / gates pass on the representatives; near-ties reviewed
- [ ] 9. promote_check.py before any republish; improvements wait for a second confirming run
- [ ] 10. PR with the loom-scope block, the impact table, the plan and the verdicts; publish to the staging org only
```

## The shortest path for a one-model, one-device change

What dominates is the DevLab queue (R9700: 3 runners, median wait 24-30 min, 100+ min behind a fleet sweep, `eng/regenauto/velocity.md`), not
the measurement, and the ratchet's second run used to queue a second time. The path pays the queue once:

| step | tool | wall |
|---|---|---|
| 1 route + scope rule (+ code) | `kb.py route`, `scope.json` rule | engineer time, as before |
| 2 fleet diff + scope lint | `build_fleet.py` x2, `diff_fleet.py`, `scope_lint.py --local` | ~5 min CPU |
| 3 submit the touched model on the touched GPU only | `regen.py plan --device D --replicas 2 --sequential --no-baselines --priority SLUG --state S.json` then `regen.py submit S.json` (or by hand: `ab_jobs.py --device D --replicas 2 --sequential SLUG`) | 1 min |
| 4 wait: queue + ONE job (two replica jobs in parallel on two runners) | `regen.py poll` loop, >= 120 s | queue (see below) + ~15 min with 2+2 sessions and no llama.cpp re-measure (an A/B compares Loom with Loom; the baselines are 40 % of a small model's job) |
| 5 `regen.py collect` (the pair is read together: first = replica 1, confirming = replica 2) then `promote_check.py` | `collect` runs `strix_ab_compare.py` per job and the ratchet on the pair | 1 min |
| 6 `regen.py ingest` / `publish` | | ~2 min |

Expected: queue 25-30 min (median; one wave, at most one job's length, when the targeted batch is submitted during a fleet sweep that went out
with `--max-inflight`) + 15-25 min job = **about 45-55 min** to a verdict, against ~100 min (median; 200+ behind a sweep) when the first run, a
look at its result and a second queued run were sequential.

- **Replicas.** `--replicas 2` puts the two ratchet runs in the same `perf.json`: the second job (`...-r2`, ABBA order rotated) starts on a
  second runner while the first runs, so the confirming run no longer waits for the first and queues again. With a single free runner they
  run one after the other (the verdict then still saves the second submission). `promote_check.py` prints the two runner names; the same
  runner twice is reported as such.
- **Sequential.** `--sequential` starts at 2+2 sessions; a job whose row lands within 2 points of its reject threshold extends itself to 6+6
  (step `extend:6:...`), so a borderline row never costs a third queued run. Fast (bimodal) models keep their fixed 6 sessions.
- **Priority lane.** A fleet sweep is submitted with `regen.py submit S.json --max-inflight auto` (the device's runner count; call it again as jobs
  finish): only one wave is ahead of anything submitted meanwhile. A targeted plan uses `--priority`: its batch is submitted first and
  ignores the cap, so it waits for at most the wave already queued. No runner admin is involved. Without `--max-inflight` a sweep submits every batch at once, as
  before, and a targeted job queues behind all of it.

## 1 Route and declare

```bash
python3 "${LOOM_SKILL_DIR}/../loom-kb/kb.py" route --model <slug or llm_config.json> --device <device> --topic decode,speculative
```
Read what it prints, nothing else from the knowledge base (the family, device, quant and facet nodes already say what is special and
what was measured; a fact you learn goes into the most specific node, see `loom-kb` "Where a fact goes"). Then write the PR's
`loom-scope` block before touching a file:

````
```loom-scope
kind: targeted
scopes: arch/llm/moe/qwen35moe+drafted
kernels: attention_decode
```
````
`kind`s, what each may touch and what a leak is: [architecture.md](architecture.md) section "Guards". A targeted change has one
or a few selector nodes (`device/...`, `quant/...`, `arch/...`, `facet/...`); if you cannot name them, the task is not targeted.

## 2 Make the change opt-in by scope

- New behaviour is a new **choice** (`skills/loom-kb/choices.py`: key, type, `legacy` = today's behaviour, env, kernels, rows,
  devices, `from_pr`) set by a rule in the node's `scope.json`: `{"when": {...}, "set": {key: value}, "evidence": "<run or PR>"}`.
  The resolver applies it to the models that node matches and nowhere else; a model, GPU or quant it was not measured on keeps its
  old program byte for byte.
- Never change a `legacy` value, a default in a generator, or a protected core file (`fleet/ownership.json` lists them) from a
  targeted PR: that is an architecture PR. A default lives twice (the generator's and the registry's `legacy`); `tests/test_choices.py::test_legacy_equals_the_generators_default` fails when they diverge. If the fix needs one, stop and report instead of working around it.
- A key that changes a kernel shared by both targets of a dual-target package cannot differ per device inside one package: scope it by
  family or facet and say so, or implement it as a twin kernel (`K` / `K_rdna4`).
- Prefer a regression fix by scope over a global revert: restrict the offending path to where it was measured good, keep it where it won.
- Programs (`.loom`) are generated: change the generator and the scope rule, never edit a committed program by hand, and leave
  regenerable outputs out of the commit when a build reproduces them byte for byte.

## 3 Prove which models it touches

```bash
W=<work dir on disk, not /tmp>; PY=python3 (with gguf installed)
$PY fleet/build_fleet.py --ref origin/<integration branch> --out $W/base --jobs 12     # CPU only, both targets, ~2 min
$PY fleet/build_fleet.py --tree .                          --out $W/head --jobs 12
$PY fleet/diff_fleet.py $W/base $W/head --json $W/impact.json --md $W/impact.md
$PY fleet/scope_lint.py --local --pr-body $W/pr_body.md --impact $W/impact.json   # checks files vs kind, leak, protected, registry, kb lint
```
The impact report lists per model the changed files, kernels, GPU target (gfx11 / gfx12 / both), card rows and the stratum
signature. It must match the declared scope: a model outside it is a leak. Fix the scope rule (tighter `when`), not the lint. A
change meant to alter nothing (refactor, docs, tooling) must show an empty diff: that is the invariant, see [campaign.md](campaign.md).
Details of the tools: `fleet/README.md`.

## 4 Test a sample on DevLab

All GPU work runs on DevLab, never the local GPU (rules and labels: [campaign.md](campaign.md) section "DevLab").

```bash
python3 fleet/sample_plan.py $W/impact.json --gpus strix-halo,radeon-r9700 --json $W/plan.json --emit-jobs $W/jobs
```
It groups the touched models into strata and picks the representatives (>= 1 per stratum per touched GPU family, 2 when the stratum
spans a wide shape range, no sub-1B model as a decode canary when a larger member exists, least recently measured first) and the
rows to measure. Run the A/B of the representatives only:

```bash
python3 .github/scripts/ab_jobs.py --device radeon-r9700 --replicas 2 --sequential --gates dev --old-org loom-fun-staging --new-org loom-fun-staging \
    --refs MAP.json SLUG... > .github/devlab/perf.json            # or --device strix-halo; strix_ab_jobs.py / r9700_jobs.py = wrappers; add --baselines only for a card
# experiment PR off your branch; label it by REST (gh pr edit fails): gh api -X POST repos/loom-fun/loom-skills/issues/N/labels -f 'labels[]=devlab-perf'
python3 .github/scripts/strix_ab_compare.py JOBDIR --json cmp.json         # one compare per job (the replicas: one each, the second with --prev cmp.json)
python3 fleet/verdict.py cmp.json --model SLUG --rows <affected rows>      # per-row verdicts against skills/loom-kb/method/methodology.json
```
Rules: old and new variants run in ONE job on ONE runner, ABBA ordered; the runner labels are in [campaign.md](campaign.md); each
row is judged against its own band (`methodology.json`: bands, T1 reject thresholds, sessions, bimodal models), never a geomean
that lets a gain pay for a loss; a row within 0.02 of a threshold is extended to 6+6 sessions; a bimodal model is compared within a
mode or flagged. The benchmark rules themselves (drafting, baselines, medians over sessions, gates) are
`skills/loom-benchmark/reference/methodology.md`, not restated here. The sampled verdict of a stratum is its worst member's.
Poll the run with ONE background until-loop on `gh run view RUN --json status`, no faster than every 120 s.

## 5 Promote, publish

```bash
python3 fleet/promote_check.py --model SLUG --gpu radeon-r9700 cmp.json [cmp_confirming.json]    # promote | improved-pending | rerun | hold
                                                                                                  # the pair = replica 1, replica 2 (concurrent runs)
```
A (model, GPU) is republished only on `promote`: every row >= published - band, gates pass, greedy text matches or is a reviewed
near-tie. An improvement is recorded, not trusted: it reaches the card after two confirming runs. `hold` keeps the published
package serving and files a regression ticket against the change. Publish only to the staging org (`loom-fun-staging`); production
repos change when the owner promotes. Partial progress is published too: a card shows gate and row status, so a model that is not
100 % still gets its card with the evidence as it is (never hidden or loosened). Report outcomes (native context, gates, targets met),
not the workarounds you passed through.

## When to stop
- The fix needs a protected file or a legacy value: report, do not work around it.
- The fleet diff touches models outside the scope and tightening the rule removes the benefit: split the PR or declare `leak:` with the
  reason on a horizontal PR.
- A representative regresses a row beyond its band: the stratum fails; fix the rule's scope, do not retest until it passes.
