---
name: loom-optimize
description: Make a Loom LLM port faster (decode, prefill, long context, concurrency) while proving it stays correct — the measure/ablate/change/verify loop, correctness gates (kernel A/B tests, llama.cpp golden, concurrency consistency), and a catalog of optimizations that worked (and ones that didn't) on RDNA3/Strix Halo. Use for "speed up", "beat llama.cpp", "optimize prefill/decode" on a Loom model.
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

Before you start you need a package that passes the golden gate (`loom-port-llm` + `loom-benchmark`), a baseline row in
`perf_log.jsonl`, llama.cpp ROCm numbers on the same GGUF and workloads, and **targets from `loom-roofline`**
(`bench/roofline.json`; `qualify.sh` writes it). If the user gave no target or time, run `loom-roofline` first, announce the
targets and the budget (its section 0), then loop here under its rate controller: don't ask for a number, and don't stop
after the first win.

**Start from the reference** (a published package's settled choices): loom-kb topic `reference`, "Optimizing".

**Mechanics** (fast iteration, kernel A/B, gates, measuring and recording, pitfalls) are the loom-kb topic `optimize-loop`; levers: "Levers" below. Build every variant in its own package dir behind an env switch and A/B it interleaved on one server.

## The loop (one change at a time)

1. **Measure.** Run `perf.py` (short) and `perf.py --long` (8k/16k/32k), plus
   `c4_steady.py 4`. Work on the headline metric furthest below its
   `loom-roofline` target, measured in milliseconds per token of gap, not in
   percent.
2. **Attribute by ablation** (no hardware counters on gfx1151; `ablate.py` launches a kernel twice, the end-to-end delta is its cost): topic `optimize-loop`, "Steps 2 and 3".
3. **Change one thing** in the generator, the old variant behind an env switch.
4. **Kernel A/B test** (`ab_kernel.py`, with a negative control; DevLab `devlab-kbench` when no GPU is free): topic `optimize-loop`, "Step 4".
5. **End-to-end gates** on a build you intend to keep (`LEVEL=dev gates.sh`; never `quick`/`full` in the loop): topic `optimize-loop`, "Step 5".
6. **Measure again** with `perf.py` from `bench/` (label, `--reverted` for a discarded build); keep a change only if it wins the target metric
   and regresses no other by more than noise. Act on the verdict it prints (CONTINUE, REVIEW, BUDGET, DONE) and record decisions in
   `bench/decisions.jsonl`: topic `optimize-loop`, "Step 6".

Pitfalls (a busy GPU, microbenchmarks, prefix caching): topic `optimize-loop`; Strix-specific ones are in the device node.

## Levers and what is known (loom-kb)

Every lever, its numbers and the "did not help" lists are in the knowledge base, at the most specific scope. Route
`--config PKG/llm_config.json --device DEVICE --topic optimize-loop,decode,prefill` (`kb.py route`, `loom-kb`; add
`speculative,spec-tuning,long-context,concurrency,roofline` for those levers) and read what it prints: do not re-run an A/B a
node already settled. A lever you add goes to the node it was measured at, opt-in by scope.

## Mandate

Declare the scope first, work only inside it, opt in by scope, run `fleet/scope_lint.py` before the PR, stop on
out-of-mandate needs: `loom-kb/SKILL.md` ("Mandate rules").

## Reporting a win

Report only numbers from the same machine, the same GGUF, the same workloads and
the same session, with llama.cpp ROCm beside them, plus the correctness gate
summary. See `loom-benchmark`. For each metric, include its target, its % of
ceiling and whether it was met. Add the result to
the loom-kb `roofline` topic (`kb.py route --topic roofline`).
