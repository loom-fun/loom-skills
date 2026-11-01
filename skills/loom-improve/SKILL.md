---
name: loom-improve
description: Improves the loom skills themselves (port, optimize, qualify, report) without regressing them - the KPIs (end-to-end wall clock, correctness vs llama.cpp in agentic use, decode/prefill speed), how to measure them with fresh-agent trials on a reference model suite, the improve-retest loop, subagent handling, and the rules that keep changes general. Use for "improve the loom skill at X", "make the loom skill faster/more reliable", "why did the skill take so long on model Y", or any change to the skills in this repo.
---

# Improve the loom skills

The job: make `loom` (and its sub-skills) better at an end-to-end request, "get MODEL working on
loom-serve, optimize it, and report vs llama.cpp", on models it has never seen, without making any
KPI worse. This skill is evaluation-driven: measure a baseline first, change one thing, re-measure on
the same models, and keep the change only if no KPI regressed (Anthropic's skill-authoring and agent
eval guidance; sources at the end).

## 1. KPIs (all three, every time)

| KPI | Target | Measured by |
|---|---|---|
| **Wall clock** | ≤ 60 min end to end (install -> published report), default prompt, no hand-holding | trial `phases.log` (`scripts/kpi.py summarize`) |
| **Quality** (agentic correctness vs llama.cpp) | final `LEVEL=full` gates PASS and `AUDIT PASS`: golden compare (all divergences near-ties), consistency, agent tool loop (rendering, history reuse, valid calls), sampling, needle at half and full native context, clean 400 over capacity | trial `bench/gates.json`, `audit.json`; re-checked by `scripts/eval.sh` with a frozen skill tree |
| **Performance** | decode (TG) and prefill (PP 1k/2k, long) at or above llama.cpp ROCm on the same GGUF; never lower than the previous skill revision | `scripts/eval.sh` on a quiet GPU (interleaved, medians). In-trial numbers are informational only |

Secondary (diagnostic, not goals): manual minutes (hand-work phases), friction notes (each `#` line a
trial agent logs is a skill gap), time per phase, tool calls.

**A change ships only if:** wall clock doesn't rise > 10 % on any reference model, no gate goes PASS ->
FAIL, compare max gap doesn't rise > 0.3, and quiet-GPU perf margins don't drop beyond noise
(`kpi.py compare` checks the first three; `eval.sh` the last). Never "fix" time by dropping a check
that carries weight: a gate may only be removed if another gate provably covers the same failure.

## 2. Reference models (the eval suite)

Run the cheapest model on every change; the challenge set before a PR. Each one stresses something
different; don't tune to one.

| Model (HF repo) | Role | What it stresses |
|---|---|---|
| `unsloth/Qwen3.5-0.8B-GGUF` | smoke, every change (~45 min) | starter path end to end, gated delta-net hybrid, 262k context (the full-context probe is the slow step) |
| `unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF` | challenge | gemma4: PLE, KV sharing, head_dim 256/512, bf16, Gemma-4 tool/channel chat format (has `reference/gemma4-example`) |
| `LiquidAI/LFM2.5-24B-A2B-GGUF` | challenge | lfm2moe: short-conv + MoE, head_dim 64, pythonic tool calls, `{% generation %}` template |
| `unsloth/Muse-Glimmer-30B-GGUF` | challenge | 30B dense at ~9 tok/s: gate cost on big models, channel (ATEM) chat format, norm RoPE, NoPE/SWA layers |
| `unsloth/flint-GGUF` | challenge | 40B MoE with sinks + sliding window, arch newer than installed llama.cpp builds, GGUF picking under memory limits |

Add a model when a user hits a failure none of these reproduce (a new architecture family, chat
format, quant mix). Before trusting a new eval model, confirm a passing run is possible at all.
Baselines to beat: `reference/results.md` (append every trial).

## 3. The loop

```
Progress:
- [ ] 0. State the goal (which KPI, which models) and freeze a baseline snapshot
- [ ] 1. Baseline trial(s) on the current skills (or reuse a row of reference/results.md for the same rev)
- [ ] 2. Profile: kpi.py summarize; read phases.log friction notes AND the agent's final report
- [ ] 3. Rank causes by minutes (or by gate failures); pick the biggest general one
- [ ] 4. Change the skills on a branch (one cause per change; prefer scripts over prose)
- [ ] 5. Verify the change locally before any trial (bash -n, a dry run, a regression model)
- [ ] 6. Re-run the same model on a NEW snapshot; kpi.py compare; then a different model
- [ ] 7. Fold the trial's own discoveries back (bugs it fixed in its kit, tools it wrote)
- [ ] 8. Repeat 3-7 until the goal is met or the top remaining cause is not fixable in the skill
- [ ] 9. Quiet-GPU eval.sh on every finished package; results.md rows; PR
```

**0. Snapshots.** Trials read a frozen copy, never the working tree: you will edit files while trials
run, and bash reads scripts as it executes them (an edit can corrupt a running `gates.sh`).
```bash
git worktree add ~/lt/skills-vN <branch> --detach      # one per trial wave; never edit it
```
**1-2. Run a trial** (section 4). Profile with
```bash
scripts/kpi.py summarize TRIAL --label NAME --skill-rev vN --log kpis.jsonl
scripts/kpi.py compare --log kpis.jsonl BASE NEW        # exit 1 on a regression
```
Then read the trial's report and every `#` note. Time estimates in notes are unreliable (an agent
wrote "45 min" for a 6-minute step): use the timestamps.

**3. Rank by minutes, and classify each cause:**

| Cause type | Fix it with | Example from past campaigns |
|---|---|---|
| A step the agent re-invents every run | a script or template in the skill | A/B, ablation and report scripts (agents wrote their own 3 times) |
| A step whose result never changes inside the loop | float it out / cache it | llama.cpp goldens re-recorded every gate run -> reference cache |
| Repeated identical work | memoize on an input hash | gates re-run on an unchanged build -> skip on same package hash |
| A wrong or missing instruction | fix the doc where the agent was when it needed it | port.sh positional args, "don't use the GPU while qualify measures" |
| A brittle script or silent failure | make it fail loudly with the reason | picker "failed to start" with no reason; FLASH2_KT=16 silently empty |
| A policy that forces waste | change the policy | 1 h optimization on top of everything else -> 60 min for the whole job |
| Architecture or chat format not covered | a reference example or a starter feature, from the trial that solved it | gemma4-example cut a 45-min hand port to 5 min |
| Shared-GPU noise | interleave, re-measure on quiet GPU | finalize interleaves llama.cpp and Loom |

Prefer, in order: deleting a step, scripting it, caching it, documenting it. Put deterministic work
in scripts (low freedom); leave judgment (kernel design, debugging) as guidance (high freedom).

**4. Change.** One cause per commit, message with the measured reason. Keep SKILL.md files concise:
the agent already knows how to code; tell it what it can't know (this hardware, this SDK, the gates,
past failures). A new reference example goes under `loom-port-llm/reference/<family>-example/` with
a README stating what it covers and what was left open.

**5. Verify before spending a trial.** Syntax (`bash -n`, `python -c "import ast..."`), the changed
script on a finished trial's package, and a regression model for anything touching the starter or
chat glue (both Qwen chat blocks still derive 12/12 and pass `agent_golden.py`). A fallback you add
must be tested on a model that takes it (a split-K prefill fallback once produced garbage).

**6. Re-test.** Same model first (did the change work?), then a different family (does it
generalize?). A re-test that benefits from a reference example of the same model measures the
template, not generalization: say so, and also run a model the example doesn't cover.

**7. Fold back.** Each trial's kit diff is free engineering: generator fixes, missing helpers, a chat
format. Diff `TRIAL/work/kit` (or `skills-local`) against the snapshot; merge what is general and
verified (gates passed on it), with the trial's measurement in the commit message.

**8. Stop** when all reference models meet the target, or the top remaining cause is outside the
skill (hardware speed, model size, upstream SDK/llama.cpp bugs): report it with the minutes.

## 4. Running trials with subagents

**Brief.** Fill `trial_prompt.md` exactly; it states the objective, the output format (phases,
numbers, gates, friction list, deliverables), the tools and sources allowed, and the boundaries. A
vague brief produced duplicated work and contaminated results.
- Fresh general-purpose agent, skills from a frozen snapshot, its own install root, work dir, ports
  and state dirs. Isolated trials: no shared caches beyond the HF and pip caches.
- **Contamination:** forbid reading the skills repo, other trials and old work dirs, and require the
  same of any subagent it spawns (a trial's chat subagent once read the working branch).
- Instrumentation: `phases.log` with epoch timestamps per phase and `#` friction notes.

**Supervise, don't wait.** Start a watchdog for every trial, in the background, right after launch:
```bash
scripts/watch_trial.sh TRIAL 25 300    # exits DONE, STALLED (nothing changed for 25 min) or TIMEOUT
```
A completion notification is not a liveness signal: an agent blocked on its own background loop never
completes (one sat idle 16 h). On STALLED: look at the last phase and processes; message the agent or
stop it, record the result so far, move on. Check every running trial at least every 30 min.

**Concurrency.** One GPU: at most two trials at once, and only if their models fit together (each
gate run holds llama.cpp + Loom: 2x weights per trial; GPU queues run out at ~6 processes). Contention
inflates every trial's wall clock and makes in-trial perf meaningless: compare wall clock only
between runs with similar contention, and take perf from `eval.sh`. Stop finished trials' servers
promptly (they hold GPU queues and memory).

**Effort scaling.** Smoke (Qwen3.5-0.8B) on every change; one challenge model to validate a fix for
it; the full challenge set before a PR. Don't run the full suite for a doc tweak.

**Reading results.** Grade outcomes, not paths: the gates and the wall clock, not whether the agent
followed your expected steps. Read the transcript summary for failures; a failure must be clearly the
skill's or the model port's fault, otherwise fix the eval (e.g. a sampling check that fails because
the prompt is near-deterministic for that model, or llama.cpp disagreeing with itself).

## 5. Measuring correctly

- In-trial perf is informational. Comparable perf comes from `scripts/eval.sh OUT PKG GGUF` on a quiet
  GPU (no other loom-serve/llama-server), same llama.cpp build for every revision, interleaved runs,
  medians. Re-serve a finished package from its trial dir to evaluate it later.
- Quality is re-checked by `eval.sh` with a frozen skill tree (`EVAL_SKILLS=`), so a skill change
  can't relax the gates it is graded by.
- The llama.cpp reference must be the fastest available build (latest lemonade release vs installed);
  a weak baseline inflates every margin.
- Two llama.cpp runs can disagree at near-ties; the gates handle it (on-demand re-record). Before
  blaming Loom for a divergence, check llama.cpp against itself.

## 6. Pitfalls that cost real time (don't repeat them)

- Editing a script while a trial executes it (use snapshots). `pkill -f`/`pgrep -f` matching your own
  shell (kill by PID file). Waiting on notifications instead of watching liveness.
- Running perf or A/B while another measurement runs (qualify stages 4/6, finalize stage 1).
- Measuring a change on a contended GPU and keeping or reverting it on that number.
- Adding a fallback or generalization without a model that exercises it.
- Treating a template or reference win on the same model as generalization.
- Writing docs for imagined problems: every addition should trace to a trial note or measurement.

## Files

- `trial_prompt.md`: the brief for a trial agent (fill the placeholders).
- `scripts/kpi.py`: trial KPIs (`summarize`) and regression check between two trials (`compare`).
- `scripts/watch_trial.sh`: trial watchdog (DONE / STALLED / TIMEOUT).
- `scripts/eval.sh`: fixed quiet-GPU evaluation of a package (frozen gates + interleaved perf).
- `reference/results.md`: every trial so far: the baselines to beat, and the lessons behind each rule.

## Sources

- Anthropic, [Skill authoring best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices):
  evaluations before documentation, baseline then iterate; Claude A refines, Claude B tests on real
  tasks; concise SKILL.md; scripts for deterministic steps; feedback loops; test across models.
- Anthropic, [Equipping agents for the real world with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills):
  start from observed gaps on representative tasks; have Claude capture successes and mistakes back into the skill.
- Anthropic, [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents):
  tasks, trials, graders; capability vs regression suites; grade outcomes not paths; isolate trials; read
  transcripts; suspect the grader when scores look wrong.
- Anthropic, [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system):
  subagent briefs with objective, output format, tools and boundaries; explicit effort scaling; resumable
  long runs; observability of agent behavior.
