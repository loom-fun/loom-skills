---
name: loom-optimize
description: Make a Loom LLM port faster (decode, prefill, long context, concurrency) while proving it stays correct — targets from the best published numbers and the roofline, real per-kernel profiles (HAL dispatch timestamps), bucket budgets, fuse-by-default designs tried in parallel, perf-only iteration rounds with gates once on the winner, and a lane that ends only when every lever has a measured stop reason. Use for "speed up", "beat llama.cpp / vLLM / leader X", "SOTA", "optimize prefill/decode" on a Loom model, or an optimization lane of a campaign.
---

# Optimize a Loom port without breaking it

**Bundled paths ? first shell action.** Before running commands, set `LOOM_SKILL_DIR` in the shell to the
absolute directory containing **this** `SKILL.md`, using its path from the skill listing or read tool.
In Claude Code, run `export LOOM_SKILL_DIR="${CLAUDE_SKILL_DIR}"` (the host substitutes the path).
In Codex, Pi or another host, export `LOOM_SKILL_DIR` to the loaded file's directory. Use a
shell-compatible absolute path (Git Bash on Windows). Rebind before using another skill's commands
and in every new shell; it is not the working directory or SDK root. On Windows, use native `C:/...`
paths for file tools; `/c/...` paths belong to Git Bash. Never run an unresolved placeholder. Run bundled commands by quoted absolute path; do not run `scripts/...` from the working directory
or change into the installed skill tree. Install the complete Loom suite: sibling paths are required.

This skill is the process. Every fact (a lever, its numbers, what did not help, tool recipes, SDK limits) is in the knowledge base
at the most specific node; route before you design and write what you learn back there.

## Contents
- Before you start
- Cost model and who does the work
- The loop (targets, profile, budgets, design, kernel level, perf rounds, gates, record)
- Keep going: a lane ends when every lever has a measured stop reason
- When to stop a metric
- Mandate
- Reporting

## Before you start

- A package that passes the golden gate (`loom-port-llm` + `loom-benchmark`), a baseline row in `perf_log.jsonl` (`perf.py`), the
  baseline engine on the same checkpoint and workloads (llama.cpp ROCm for a GGUF card, vLLM ROCm for a safetensors card; methodology
  in `loom-benchmark`), and the ceilings from `loom-roofline` (`bench/roofline.json`; `qualify.sh` writes it). No target given: run
  `loom-roofline` first, announce targets and budget, then loop here; don't ask for a number.
- Route the knowledge (`loom-kb`):
  `python3 "${LOOM_SKILL_DIR}/../loom-kb/kb.py" route --config PKG/llm_config.json --device DEVICE --topic optimize-loop,prefill,decode`
  (add `speculative,spec-tuning,long-context,concurrency,roofline` for those levers, `kernels-ir,sdk` before writing kernels). Read
  what it prints; never re-run an A/B a node already settled. Start from a published package's settled choices: topic `reference`.
- Mechanics (fast iteration, profiling recipe, ablation, kernel A/B, gates, recording, pitfalls) are topic `optimize-loop`; the
  sections named below are its headings.

## Cost model and who does the work

- **Currencies: agent tokens and GPU/runner time, nothing else.** Never estimate, weigh or mention human implementation effort or
  "complexity": agents write the code. The most aggressive design (whole-layer fusion, epilogue/prologue fusion, persistent kernels)
  is a normal candidate, not a last resort. Spend runner time carefully instead: CPU checks, then kernel benches, then one batched
  end-to-end job per round.
- **Model tiers** (team work, `loom-fleet` reference `campaign.md`, "Roles"): the strongest tier for design-heavy lanes (fused or
  persistent kernels, register/LDS budgets across fused stages, cross-workgroup sync, open-ended optimization); the cheaper tier for
  sweeps, plumbing and parallel variants of a design. Run **alternative designs of one lever in parallel lanes**, each with its own
  worktree, switch and A/B; the bucket's measurement decides, not the first design to land.
- On DevLab (campaigns, no local GPU): job presets, runner rules and the reference store are `campaign.md`, "DevLab".

## The loop

### 1. Targets: the leader and the roofline, per row

For every card row (prefill 1k-32k, decode by prompt class, long-context decode, 4 streams), write down: the best published number on
the same device class and a comparable workload (engine, weight format, drafting on or off; find and read the leader's own
benchmark docs and code, do not guess), Loom's number, and the `loom-roofline` ceiling. The row's target is the leader where the
ceiling allows it, the roofline target otherwise. Work on the row with the most milliseconds of gap, not the most percent.

### 2. Profile with the real profiler first

- **Per-kernel device time from the HAL** (per-dispatch device timestamps; recipe, flags, flush interval and the families file:
  topic `optimize-loop`, "Profiling"; DevLab step `profile:V`). One session per scenario (pf2k, pf8k, drafted decode); two prefill
  lengths separate per-token from per-request cost; decode in ms per verify step and tokens per step.
- Check the profile is whole: no `dropped_records`, every launch in a family (an unnamed family hid 12 % of a prefill), device busy
  vs request wall (the difference is launch/idle overhead; it decides whether launch count is a lever at all).
- **Doubling/ablation is a cross-check**, and the tool for counter-like questions (what does this kernel wait for?), never the primary
  profile: it distorts through caches, power and non-idempotent kernels (topic `optimize-loop`, "Ablation: known distortions").
  Hardware counters are not available on gfx11/gfx12 with the current SDK (topic `sdk`).
- No GPU needed: the compile report and the generated ISA answer codegen questions (spills, waits, instruction mix) before a runner is
  spent (topic `optimize-loop`, "Local pre-flight").

### 3. Bucket budgets from the leader

- **Decompose the leader's number into bucket budgets.** Prefill: `1000 / leader tok/s` = the ms/token budget; the GEMM bucket's share
  is the GEMM FLOPs per token at the best GEMM rate measured on this device (yours or the leader's); the non-GEMM budget is the rest.
  Drafted decode: the step's weight bytes at the measured bandwidth are the floor; the leader's ms per accepted token, times the
  tokens per step, minus the floor is the budget of everything else (proposals, recurrent state, producers, attention, head).
- **Judge the bucket, not each kernel.** A bucket of kernels that are 1-2 % each looked "at diminishing returns" kernel by kernel and was
  ~40 % of prefill. Sum the families of a bucket, compare with its budget, rank buckets by ms over budget, attack the largest.
- Worked example (MXFP4 27B card on the R9700): its model node, "Budgets".

### 4. Design: fuse by default, price first, several designs

- **Kernel boundaries are ours.** The default design for an over-budget bucket is the most fused one the constraints allow (producer work
  in the producing kernel's epilogue, front + body + epilogue of a layer stage in one kernel, whole-layer kernels). Un-fuse only where a
  measurement says so, and keep that measurement with the decision.
- **Price a design before building it:** expected gain = the family's share of the profiled step or token x the fraction the design can
  remove, counting what the fused kernel takes on (judge the sum of both kernels, not the kernel fused away). Measured conversions from
  kernel gains to end-to-end gains: topic `optimize-loop`, "Pricing a design".
- **Read the source of the idea** (the leader's code, the paper) and its measured net on comparable hardware before porting it.
- Two or three structurally different designs per lever, in parallel lanes where possible. Each is a choice or env switch over the old
  path (the old variant stays selectable); generated by the generators, never hand-pasted IR; kernel tests for every new kernel.

### 5. Kernel level first

- CPU: build every variant's programs and run `loom-opt` over them (catches type/symbol errors in about a minute); read the ISA of a
  schedule or load change before timing it.
- Correctness: `ab_kernel.py` (old kernel beside new, negative control) with serving-shaped cases (topic `optimize-loop`, "Step 4").
- Timing: the kernel alone, interleaved with the old one (`devlab-kbench`, or one local GPU). A kernel that does not beat its noise in
  isolation does not go end to end; one that does still has to convert (step 6).

### 6. Perf-only iteration rounds

- End-to-end A/B of base vs variant(s), interleaved ABBA in one session/job: locally `ab_perf.py` on one loom-serve (topic
  `optimize-loop`, "Fast iteration"); on DevLab the `iterate` preset (build + ktest + perf + c4 + smoke, no gates, no reference engine).
- **The base is a sha** (the branch's merge base), never a moving branch name: a branch that moves under a running A/B confounds it.
- Keep a change only if it wins its target row and regresses no other by more than noise (+-2-3 %; a second confirming round when
  it is close). Profile the winner (step 2) so the next design starts from the new budget, not the old one.

### 7. Gates once, on the winner

- Only the build that is going to be kept runs the gates: `LEVEL=dev gates.sh` locally, the `final` preset on DevLab, both against the
  cached references (reference engines are invariant inputs; never reload them per round).
- **A numerics-moving kernel is gated before its speed counts**, including an exact kernel that only reorders a sum: it can flip a
  near-tie past the threshold.
- A gate failure: run the base's gates in the same job before blaming the change. Never loosen a gate; a real failure keeps the change
  opt-in and recorded.
- A change that moves device memory (scratch, prepared copies) re-runs the served-context check with every claimed feature (image input
  included); an admission probe alone proves nothing, a request at that context must finish.

### 8. Record every technique, wins and losses

- Locally: `perf.py` with a label (`--reverted` for a discarded build) and `bench/decisions.jsonl` (topic `optimize-loop`, "Step 6").
- In a campaign: one row per technique tried in the campaign's technique log (date, technique, source, how measured, numbers + run
  id, verdict, lesson). Losses are as valuable as wins.
- Facts go to the KB node they were measured at (topic `kb-authoring`), opt-in by scope; the lesson, not the diary.

## Keep going: a lane ends when every lever has a measured stop reason

List the levers of each over-budget bucket. A lever is finished only with one of these stop reasons, each with its numbers:

| Stop reason | Evidence required |
|---|---|
| At the roof | bandwidth-bound kernels at >= 90 % of the measured device bandwidth; compute kernels where zero-operand / no-op ablations show the rest is power or issue, and 3+ restructurings moved < 3 % |
| Ceiling below 1 % | the family's profiled share x the most any design could remove < 1 % of the row (e.g. launch gaps 2.5 % of a step cap persistent kernels at ~1 %) |
| Diminishing returns | after 2+ structurally different designs, the last three each within end-to-end noise (+-2 %) or < 3 % on the lever's kernels in isolation |
| Leader's net is zero | the source measured ~0 for the technique on comparable hardware, and your A/B confirms |
| Correctness | no exact or gate-passing form found; the variant stays opt-in with the failing evidence |
| Costs another row more | measured: context, memory, another prompt class, or the 4-stream row |

Not stop reasons: the first design lost (ablate the loser, write what bound it, design the next one from that: a fused kernel that lost
because its serial loop waited on global loads won once it prefetched); a kernel win that ties end to end (re-price; the share was too
small); "diminishing returns" judged per kernel instead of per bucket; implementation effort.

## When to stop a metric

A row is done when it meets its target (>= the leader on the same kind of workload and the roofline target), or when every lever of
every bucket over budget has a stop reason from the table. Locally, act on the verdict `perf.py` prints (CONTINUE, REVIEW: re-profile,
then pivot or stop with `loom-roofline`'s plateau review, BUDGET, DONE). Stopping short of the target is a result: report the
remaining gap per bucket with the stop reasons.

## Mandate

Declare the scope first, work only inside it, opt in by scope, run `fleet/scope_lint.py` before the PR, stop on out-of-mandate
needs: `loom-kb/SKILL.md` ("Mandate rules").

## Reporting

Per row: Loom, the leader (source, date, workload), the baseline engine measured in the same job, the roofline ceiling and % of it,
met or not, and the stop reason when not; numbers only from one machine, one checkpoint, the same workloads and session (`loom-benchmark`),
with the gate summary. Add the calibration row to the device's roofline node (`kb.py route --topic roofline`).
