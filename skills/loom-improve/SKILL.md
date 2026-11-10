---
name: loom-improve
description: Improves the loom skills themselves (port, optimize, qualify, report) without regressing them - the KPIs (end-to-end wall clock, correctness vs llama.cpp in agentic use, decode/prefill speed), how to measure them with fresh-agent trials on a reference model suite, the improve-retest loop, subagent handling, and the rules that keep changes general. Use for "improve the loom skill at X", "make the loom skill faster/more reliable", "why did the skill take so long on model Y", or any change to the skills in this repo.
---

# Improve the loom skills

**Bundled paths ? first shell action.** Before running commands, set `LOOM_SKILL_DIR` in the shell to the
absolute directory containing **this** `SKILL.md`, using its path from the skill listing or read tool.
In Claude Code, run `export LOOM_SKILL_DIR="${CLAUDE_SKILL_DIR}"` (the host substitutes the path).
In Codex, Pi or another host, export `LOOM_SKILL_DIR` to the loaded file's directory. Use a
shell-compatible absolute path (Git Bash on Windows). Rebind before using another skill's commands
and in every new shell; it is not the working directory or SDK root. On Windows, use native `C:/...`
paths for file tools; `/c/...` paths belong to Git Bash. Never run an unresolved placeholder. Run bundled commands by quoted absolute path; do not run `scripts/...` from the working directory
or change into the installed skill tree. Install the complete Loom suite: sibling paths are required.

The job: make `loom` (and its sub-skills) better at an end-to-end request, "get MODEL working on
loom-serve, optimize it, and report vs llama.cpp", on models it has never seen, without making any
KPI worse. This skill is evaluation-driven: measure a baseline first, change one thing, re-measure on
the same models, and keep the change only if no KPI regressed (Anthropic's skill-authoring and agent
eval guidance; sources at the end).

## 1. KPIs (all three, every time)

| KPI | Target | Measured by |
|---|---|---|
| **Wall clock** | within **the model's roofline budget** end to end (install -> published report), default prompt, no hand-holding: `loom-roofline/e2e_budget.py` = fixed agent work + optimization + the pipeline's token workload at this model's decode/prefill speed (0.8B ≈ 46 min, 30B dense ≈ 106). A constant target would punish big models for physics and let small ones be sloppy | trial `phases.log` (`"${LOOM_SKILL_DIR}/scripts/kpi.py" summarize`: `wall_min` vs `budget_min`) |
| **Quality** (agentic correctness vs llama.cpp) | final `LEVEL=full` gates PASS and `AUDIT PASS`: golden compare (all divergences near-ties), consistency, agent tool loop (rendering, history reuse, valid calls), sampling, needle at half and full native context, clean 400 over capacity | trial `bench/gates.json`, `audit.json`; re-checked by `"${LOOM_SKILL_DIR}/scripts/eval.sh"` with a frozen skill tree |
| **Performance** | decode (TG) and prefill (PP 1k/2k, long) at or above llama.cpp ROCm on the same GGUF; never lower than the previous skill revision | `"${LOOM_SKILL_DIR}/scripts/eval.sh"` on a quiet GPU (interleaved, medians). In-trial numbers are informational only |

Secondary (diagnostic, not goals): manual minutes (hand-work phases), friction notes (each `#` line a
trial agent logs is a skill gap), time per phase, tool calls.

**A change ships only if,** on every model its checks covered (section 3, step 5): it stays within its budget, wall clock doesn't rise > 10 %, no gate goes PASS ->
FAIL, compare max gap doesn't rise > 0.3, and quiet-GPU perf margins don't drop beyond noise
(`kpi.py compare` checks the first three; `eval.sh` the last). Never "fix" time by dropping a check
that carries weight: a gate may only be removed if another gate provably covers the same failure.

## 2. Reference models (the eval suite)

No model runs by default and there is no full-suite run before a PR: each change gets the checks it
needs (section 3, step 5). Each model stresses something different; don't tune to one.

| Model (HF repo) | Role | What it stresses |
|---|---|---|
| `unsloth/Qwen3.5-0.8B-GGUF` | cheapest trial (~20 min); the only starter-path model | starter path end to end, gated delta-net hybrid, 262k context (the full-context probe is the slow step); vision: `qwen3vl_merger` + M-RoPE source positions; drafting: MTP (`*-MTP-GGUF`) and n-gram with GDN replay (launch-bound: tune.py keeps none) |
| `unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF` | challenge | gemma4: PLE, KV sharing, head_dim 256/512, bf16, Gemma-4 tool/channel chat format (has `reference/gemma4-example`); vision: `gemma4v` (clamped linears, unscaled rows, PLE token 0), whole-conversation chat kit -> generic image rendering; drafting: NVIDIA's BF16-only DFlash drafter (vendor repo search, Q8_0 requantization, `embedding_scale`) and the Gemma-4 E4B assistant (`assist`: attends the target's K/V) |
| `LiquidAI/LFM2.5-24B-A2B-GGUF` | challenge | lfm2moe: short-conv + MoE, head_dim 64, pythonic tool calls, `{% generation %}` template; drafting: no drafter exists, so n-gram only, with the short-conv state replay |
| `unsloth/Muse-Glimmer-30B-GGUF` | challenge | 30B dense at ~9 tok/s: gate cost on big models, channel (ATEM) chat format, norm RoPE, NoPE/SWA layers; vision: `muse-glimmer` (50-layer encoder, windowed attention, Lanczos resize, rows through the embedding norm); drafting: z-lab DFlash2 with a 33280-wide feature projection (`mv_wide`) and the drafter's own block embedding (the target norms its embedding) |
| `unsloth/flint-GGUF` | challenge | 40B MoE with sinks + sliding window, arch newer than installed llama.cpp builds, GGUF picking under memory limits; drafting: incoai DFlash2 (block 8), MoE verify near-ties at depth > 3 |

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
- [ ] 5. Plan the checks (what the change can affect, the cheapest evidence for each), write the plan down, run the deterministic ones
- [ ] 6. Run the planned agent trials, if any, on a NEW snapshot; kpi.py compare
- [ ] 7. Fold the trial's own discoveries back (bugs it fixed in its kit, tools it wrote)
- [ ] 8. Repeat 3-7 until the goal is met or the top remaining cause is not fixable in the skill
- [ ] 9. Quiet-GPU eval.sh on the packages the checks produced; results.md rows (with the plans); PR
```

**0. Snapshots.** Trials read a frozen copy, never the working tree: you will edit files while trials
run, and bash reads scripts as it executes them (an edit can corrupt a running `gates.sh`).
```bash
git worktree add ~/lt/skills-vN <branch> --detach      # one per trial wave; never edit it
```
**1-2. Run a trial** (section 4). Profile with
```bash
"${LOOM_SKILL_DIR}/scripts/kpi.py" summarize TRIAL --label NAME --skill-rev vN --log kpis.jsonl
"${LOOM_SKILL_DIR}/scripts/kpi.py" compare --log kpis.jsonl BASE NEW        # exit 1 on a regression
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
| A policy that forces waste | change the policy | 1 h optimization on top of everything else -> one budget for the whole job |
| Architecture or chat format not covered | a reference example or a starter feature, from the trial that solved it | gemma4-example cut a 45-min hand port to 5 min |
| Shared-GPU noise | re-measure on a quiet GPU, medians | finalize and eval.sh take medians of repeated runs, one model at a time |

Prefer, in order: deleting a step, scripting it, caching it, documenting it. Put deterministic work
in scripts (low freedom); leave judgment (kernel design, debugging) as guidance (high freedom).

**4. Change.** One cause per commit, message with the measured reason. Keep SKILL.md files concise:
the agent already knows how to code; tell it what it can't know (this hardware, this SDK, the gates,
past failures). A new reference example goes under `loom-port-llm/reference/<family>-example/`: its
new files, a `starter.patch` for the starter files it changes (`diff -u` against the current starter),
the reviewed `cfg.json` (or, once the port is published as a category A loom-fun package, an `examples.json`
entry pinning that package instead: its `llm_config.json` is the reviewed config) and a README stating what it
covers and what was left open. Don't ship what a category A package already carries. Move features that
any model could use into the starter instead (default off, so the Qwen programs stay byte-identical).
After any starter change, `reference/rebase_examples.sh OLD_STARTER` and `check_examples.sh`.

**5. Plan the checks.** An agent trial costs $2-6 in API tokens and 20-85 min; a deterministic check
costs GPU time only. Choose by judgment, not a fixed suite. Before running anything, write down what
the change can affect, which models exercise that path, the cheapest evidence for each, and what result
would count as a regression; it goes in the results.md row. Reason from what each model's trial actually
uses: an example's files only by its model; the starter path only by models without an example (Qwen3.5
today); a doc only by trials that read it (past trial transcripts show which: only flint read
`porting-notes.md` and `loom-ir-cookbook.md`). The maker of a change wants it to pass: plan before
seeing results, and don't shrink the plan afterwards. Cheapest evidence first:
- Syntax: `bash -n`, `python3 -m py_compile`.
- Benchmark, gate or perf scripts: replay the changed script (or `LEVEL=dev gates.sh`, or `eval.sh`) on
  finished trial packages of every model the change can reach; `qualify.sh` + `finalize.sh` only when the
  change is in those wrappers. No agent needed (the one-model-at-a-time change was validated this way on
  all five).
- Kernels, generators, the starter: kernel tests for every new or changed kernel, `chat_probe --verify`
  when the chat glue changes, then `LEVEL=dev gates.sh` on each affected model's rebuilt package after each
  kept change. Not `quick`/`full`: they run nightly on the published cards (`nightly-gates.yml`). In the
  issue #26–#29 sessions, 24+ `quick` runs (6.6 GPU-hours) caught 0 bugs; every real bug was caught by
  consistency (3), agent (3–4) or compare at first bring-up (2), which `dev` keeps (`loom-benchmark`,
  "Which level when").
- Starter or a reference example: rebuild every affected model's package (`apply_example.sh` + `port.sh`,
  or `port.sh` alone on the starter path), diff it against the last trial's package, and run that
  example's own checks (`chat_probe --verify`, kernel tests, gates). An identical package is not enough:
  glimmer's `chat_probe --verify` CLI broke while its package stayed byte-identical, and an example's
  stale `build.py` broke the chat placeholders at run time. Chat glue: both Qwen chat blocks still derive
  12/12 and pass `agent_golden.py`. A fallback you add must be tested on a model that takes it (a split-K
  prefill fallback once produced garbage).
- **Where checks run.** The dev machine's GPU is scarce and sequential; AMD DevLab Dispatch is a large
  elastic pool. The dev machine runs only what the inner loop needs (kernel tests, `chat_probe --verify`,
  `LEVEL=dev` on the package you just changed). Long or deep checks, and anything that runs against the
  published cards (`quick`/`full`, gate-script or workflow changes, timing a gate level across models),
  run in DevLab CI: label the PR `nightly-gates` (runs `dev` and `full` on every card against the PR branch)
  or dispatch `nightly-gates.yml`.
- An agent trial only where the change can alter what the agent does: instructions, workflow, hints, the
  interface or output of a script the agent reads. Use the cheapest model that takes the changed path;
  add another only for a different path (starter vs example) or a risk specific to a model (memory and
  context changes: the largest-context model; the auto memory budget ran the machine out of RAM only on
  flint). A doc tweak usually needs no trial.

Misses surface on their own: every trial runs the whole current skill tree, so a regression on a model
you didn't trial shows at its next trial against its results.md baseline. Trace it to the change and add
what the plan missed to this step.

**6. Run the agent trials.** Same model first (did the change work?); a different family only if the
plan says the change should generalize there. A re-test that benefits from a reference example of the
same model measures the template, not generalization: say so.

**7. Fold back.** Each trial's kit diff is free engineering: generator fixes, missing helpers, a chat
format. Diff `TRIAL/work/kit` (or `skills-local`) against the snapshot; merge what is general and
verified (gates passed on it), with the trial's measurement in the commit message.

**8. Stop** when the goal's models finish within their roofline budgets with all gates passing, or the top remaining cause is outside the
skill (hardware speed, model size, upstream SDK/llama.cpp bugs): report it with the minutes.

## 4. Running trials with fresh agents

Use a fresh subagent if the host supports one. Otherwise launch a separate clean
session of the same agent with the filled `trial_prompt.md` and frozen snapshot.
Do not run an "independent" trial in the session that edited the skills. If a clean
session cannot be launched, finish the deterministic checks and report the agent
trial as pending. Record the agent, model and version alongside every trial result.

**Brief.** Fill `trial_prompt.md` exactly; it states the objective, the output format (phases,
numbers, gates, friction list, deliverables), the tools and sources allowed, and the boundaries. A
vague brief produced duplicated work and contaminated results.
- Fresh general-purpose agent, skills from a frozen snapshot, its own install root, work dir, ports
  and state dirs. Isolated trials: no shared caches beyond the HF and pip caches. TRIAL dirs live on
  disk (e.g. `~/lt/trials/NAME`), never under `/tmp` (tmpfs), and launch inside a memory-capped
  `systemd-run --user --scope`.
- **Contamination:** forbid reading the skills repo, other trials and old work dirs, and require the
  same of any subagent it spawns (a trial's chat subagent once read the working branch).
- Instrumentation: `phases.log` with epoch timestamps per phase and `#` friction notes.

**Supervise, don't wait.** Start a watchdog for every trial, in the background, right after launch:
```bash
"${LOOM_SKILL_DIR}/scripts/watch_trial.sh" TRIAL 25 300    # exits DONE, STALLED (nothing changed for 25 min) or TIMEOUT
```
A completion notification is not a liveness signal: an agent blocked on its own background loop never
completes (one sat idle 16 h). On STALLED: look at the last phase and processes; message the agent or
stop it, record the result so far, move on. Check every running trial at least every 30 min.

**Concurrency.** One GPU: at most two trials at once, and only if their models fit together (each
gate run holds llama.cpp + Loom: 2x weights per trial; GPU queues run out at ~6 processes). Contention
inflates every trial's wall clock and makes in-trial perf meaningless: compare wall clock only
between runs with similar contention, and take perf from `eval.sh`. Stop finished trials' servers
promptly (they hold GPU queues and memory).

**Reading results.** Grade outcomes, not paths: the gates and the wall clock, not whether the agent
followed your expected steps. Read the transcript summary for failures; a failure must be clearly the
skill's or the model port's fault, otherwise fix the eval (e.g. a sampling check that fails because
the prompt is near-deterministic for that model, or llama.cpp disagreeing with itself).

## 5. Measuring correctly

- In-trial perf is informational. Comparable perf comes from `"${LOOM_SKILL_DIR}/scripts/eval.sh" OUT PKG GGUF` on a quiet
  GPU (no other loom-serve/llama-server), same llama.cpp build for every revision, one model at a time,
  medians. Re-serve a finished package from its trial dir to evaluate it later.
- Quality is re-checked by `eval.sh` with a frozen skill tree (`EVAL_SKILLS=`), so a skill change
  can't relax the gates it is graded by.
- The llama.cpp reference must be the fastest available build (latest lemonade release vs installed);
  a weak baseline inflates every margin.
- Two llama.cpp runs can disagree at near-ties; the gates handle it (on-demand re-record). Before
  blaming Loom for a divergence, check llama.cpp against itself.

## 6. Pitfalls that cost real time (don't repeat them)

- Big files in `/tmp` (tmpfs: RAM) and model jobs without a memory-capped scope (`loom` hard rules):
  leftover kernel-test dumps plus a 27B tune got systemd-oomd to kill the editor and every session in
  it. Trial work dirs go on disk; clean dumps and ablation packages as you go.
- Editing a script while a trial executes it (use snapshots). `pkill -f`/`pgrep -f` matching your own
  shell (kill by PID file). Waiting on notifications instead of watching liveness.
- Running perf or A/B while another measurement runs (qualify stages 4/6, finalize stage 1).
- Measuring a change on a contended GPU and keeping or reverting it on that number.
- Adding a fallback or generalization without a model that exercises it.
- Treating a template or reference win on the same model as generalization.
- Writing docs for imagined problems: every addition should trace to a trial note or measurement.
- Declaring a root cause before the gate confirms it (flint: "llama.cpp nondeterminism" was written up,
  then re-recording showed llama.cpp stable on those ids; the real cause was a sink mismatch between
  two attention kernels). Prove it with the failing gate, then write it down.
- Loosening a gate rule to make a model pass. Bisect first (config variants that switch one feature
  off, comparing Loom with Loom), then fix the kernel or document the evidence.

## Files

- `trial_prompt.md`: the brief for a trial agent (fill the placeholders).
- `"${LOOM_SKILL_DIR}/scripts/kpi.py"`: trial KPIs (`summarize`) and regression check between two trials (`compare`).
- `"${LOOM_SKILL_DIR}/scripts/watch_trial.sh"`: trial watchdog (DONE / STALLED / TIMEOUT).
- `"${LOOM_SKILL_DIR}/scripts/eval.sh"`: fixed quiet-GPU evaluation of a package (frozen gates + perf, one model at a time);
  `LEVEL=dev` re-checks in minutes when the trial's own finalize already ran `full`.
- `reference/results.md`: every trial so far: the baselines to beat, and the lessons behind each rule.

## Sources

Anthropic: [Skill authoring best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices),
[Equipping agents with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills),
[Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents),
[How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system).
