---
name: loom-optimize
description: Make a Loom LLM port faster (decode, prefill, long context, concurrency) while proving it stays correct — the measure/ablate/change/verify loop, correctness gates (kernel A/B tests, llama.cpp golden, concurrency consistency), and a catalog of optimizations that worked (and ones that didn't) on RDNA3/Strix Halo. Use for "speed up", "beat llama.cpp", "optimize prefill/decode" on a Loom model.
---

# Optimize a Loom port without breaking it

Before you start you need:
- a package that passes the golden gate (`loom-port-llm` + `loom-benchmark`);
- a baseline row in `perf_log.jsonl`;
- llama.cpp ROCm numbers on the same GGUF and workloads;
- **targets from `loom-roofline`** (`bench/roofline.json`; `qualify.sh` writes it in an
  end-to-end job). If the user said "optimize this" with no target or time, run
  `loom-roofline` first, announce the targets and the budget (section 0 there), then loop
  here under its rate controller. Don't ask the user to pick a number, and don't stop
  after the first win.

**Fast iteration (the budget is short).**
- Build each variant into its own package dir with its own `--model-id`, and serve the
  baseline and the variant in ONE loom-serve (`serve.sh NAME PORT amdgpu MEM a/model.json
  b/model.json`), then alternate requests: an interleaved A/B is both faster and more
  reliable than restarting servers between runs. Use `--context 8192` for A/B packages
  (faster to serve, same kernels) and rebuild the winner at full context for the gates.
  Give that server headroom: `--memory_bytes` well above the sum of build.py's per-package
  figures (a tight budget skewed an A/B by model order).
- Generator switches (env vars) make a variant a rebuild, not an edit: try those first.
- `perf.py` takes 1–3 min; for a decision on one metric, time just that metric
  (e.g. `bench_common.stream()` on the decode prompt, 3–5 interleaved rounds) and run the
  full `perf.py` only on the build you keep.

## The loop (one change at a time)

1. **Measure.** Run `perf.py` (short) and `perf.py --long` (8k/16k/32k), plus
   `c4_steady.py 4`. Work on the headline metric furthest below its
   `loom-roofline` target, measured in milliseconds per token of gap, not in
   percent.
2. **Attribute by ablation.** Hardware counters are unavailable (`loom-benchmark`
   counters are UNIMPLEMENTED on gfx1151). Instead, launch a kernel twice (or skip it)
   and measure end to end: `${CLAUDE_SKILL_DIR}/ablate.py PKG pkg_abl '@attn_decode'`
   writes the copy with its own model id; serve both in one loom-serve and run
   `ab_perf.py URL <id> <id>-abl-<name>`. The delta is that kernel's cost. Wrong outputs
   are fine for an ablation build; never ship one.
3. **Change one thing** in the generator: kernel shape, fusion, tiling or dispatch.
   Keep the old variant behind an env switch (like `GEMM_SKIP_PADDING`, `GEMM_N_FASTEST`,
   `FLASH2_KT`; both GEMM switches are on by default, `=0` turns one off).
4. **Kernel A/B test.** `python ${CLAUDE_SKILL_DIR}/ab_kernel.py OLD/program/llm.loom
   NEW/program/llm.loom KERNEL ab.loom-test` puts the old kernel next to the new one
   as `KERNEL_old`; its docstring has a case skeleton. Launch both on identical
   inputs in a `check.case` and compare with `check.expect.approximate`. Include a
   **negative control**: an assertion that the output is not the initial fill,
   because two kernels that both write nothing agree. Put it in a separate case that
   is expected to fail and run `loom-test ab.loom-test --device=amdgpu
   --xfail=@ab_negative_control=EXPECT/003`; the skeleton in ab_kernel.py shows it. Use the real shapes, including edge counts
   (t=1, t=4, t=63/64, a partial last tile, mixed-sequence tiles, the full context).
   Run it with `loom-test` (see the cookbook in `loom-port-llm/reference/`).
5. **End-to-end gates.** `LEVEL=fast gates.sh PKG URL GGUF BENCH_DIR` (`loom-benchmark`):
   - `compare.py` on short prompts and the 1k/2k/16k/32k goldens. Every divergence
     must be a near-tie (the compare rule, `bench_common.near_tie`). A new
     divergence with a large gap is a bug even if the text "looks fine".
   - `consistency.py`: concurrent outputs must equal sequential outputs. This
     catches mixed-tile and padding bugs that single requests never hit.
   The llama.cpp goldens come from the cache `qualify.sh` filled, so this costs only the
   Loom side (≈1–3 min). Use `LEVEL=quick` if the change touched the chat glue, the
   recurrent state or the sampling head. Run the gates only on a build you intend to keep
   (it won on perf): a reverted build needs no gates.
6. **Measure again** on the same workloads. Run `perf.py` from `bench/` with a
   label for the build (add `--reverted` if you discard it). It logs the result
   and prints the controller verdict by itself. Keep the change only if it wins on the
   target metric and doesn't regress the others by more than noise (±2–3 %; run
   twice when it's close). Commit it with the numbers in the message. Keep the work dir (cfg.json, any
   generator copies you modified, the bench/ logs) under git: run `git init` once
   in the work dir, and gitignore everything big or regenerated: `pkg*/ ktest*/ gguf/ tok/ llamacpp*/ state/
bench/llama_cache/` (a trial staged 5 GB of ktest/ and llama.cpp builds and `git add` hung). If the user doesn't
   want commits, skip git and rely on `bench/decisions.jsonl` plus a copy of each kept
   generator (`starter_<label>/`); `perf.py` then logs commit `-`.
   When you change the starter's generators, copy `starter/` into the work dir
   first and edit the copy. Act on the verdict that `perf.py` prints:
   - CONTINUE: keep going on this path;
   - REVIEW: re-ablate, then pivot or stop. Forking from an earlier decision is an
     option;
   - BUDGET: wrap up;
   - DONE: move to the next metric.

   Record design choices, including the rejected options and their numbers, in
   `bench/decisions.jsonl`, one JSON object per line (report.py renders it):
   `{"ts": "...", "decision": "prefill chunk", "chosen": "2048", "rejected": [{"option": "4096", "measured": "1k -12 %", "why": "slower short prefill"}]}`.

## Measurement pitfalls

- **A busy GPU makes single runs meaningless** (check `pgrep -a 'loom-serve|llama-server'`):
  the same build swings 30–50 %. A/B interleaved (Fast iteration, above) with
  `${CLAUDE_SKILL_DIR}/ab_perf.py URL MODEL_A MODEL_B --metrics decode,prefill_1k,prefill_2k --rounds 5`:
  medians, B/A and a WINS/LOSES/TIE verdict against the noise band, logged to `ab_log.jsonl`. Always
  include the shortest size you report (a chunk size that won at 2k/8k lost 12 % at 1k on flint).
  Measure llama.cpp right next to the final Loom run, and say in the report that the GPU was shared.

- **Microbenchmarks lie through the 32 MB MALL.** A kernel benchmark whose weights
  or KV fit in 32 MB shows 2–5× the DRAM bandwidth. Use full-size tensors (all
  layers) or end-to-end numbers.
- **Prefix caching.** Start every timed prompt with a nonce, as `perf.py` does.
- Lazy weight load, stale servers on the port, server- vs client-side timing and near-tie
  noise: see the `loom` hard rules and `loom-benchmark`.

## Catalog: what worked (Strix Halo gfx1151, measured end to end)

`reference/optimization-catalog.md` has each lever with its rationale and numbers. Decode (bandwidth-
bound): multi-token matvec, adaptive rows per workgroup, Q4_K scale/min folding, fused QKV and paired-row
gate/up, argmax head, adaptive split-K attention, F16 KV. Prefill (compute/attention-bound): WMMA GEMMs on
row-scaled f16 (N-fastest order: A/B it per port), padding-free tiles, flash2, listed fallback, per-token
RoPE + KV store, fused norm + row scale, prefill chunk size. Apply them in catalog order.

**Settled for starter ports (dense / Qwen3.5 hybrid, ≤ 4B): don't re-run these A/Bs.** Fresh-agent
trials re-tested them on Qwen3.5-0.8B and the defaults won or tied every time:
`GEMM_N_FASTEST=0` (slower prefill, 3 trials), `MV_WG1=1024` (no decode win), `--prefill 4096`
(short prefill −20–37 %; 2048 ≈ 1024 within noise). On a starter port, spend the budget on fusion
(fewer launches per layer: fused QKV or norm fusion, which you write in a copy of the generator; pairing Qwen3.5's tiny GDN alpha/beta matvecs
measured neutral) or on the long
prefill path. Bigger or MoE models: the chunk size is worth one A/B measured at 1k too (4096 helped
one MoE port; 2048 and 4096 both cost 12-18 % at 1k on flint).

**Folded into the starter from trials** (already on; don't redo): the gated delta-net conv and
history-commit kernels spread channels over `GDN_CB` workgroups per token when fewer than 64
tokens are in flight (Qwen3.5-0.8B decode +18 %, identical output; `GDN_CB=1` restores the old grid).

**Tried and did not help** (don't repeat without a new reason): see the catalog's "Did not help".

**Speculative decoding** (on by default when the GGUF has an MTP block or a DFlash drafter exists;
`loom-port-llm/starter/draft/README.md`). `qualify.sh` already chose the method and depth by measurement
(`bench/draft_tune.json`); re-run `draft/tune.py WORKDIR PORT MEMORY_GIB MODEL_ID` after a change that moves
the decode/verify balance (it rebuilds `pkg`). On a bandwidth-bound model it is the biggest decode lever
(Qwen3.8-27B Q8_0: 7.9 -> 14.6 tok/s with MTP depth 3); on small launch-bound models it loses (Qwen3.5-0.8B:
262 -> 208) and tune.py keeps it off. Levers inside it: verify cost for more than 4 tokens per epoch (the
4-token matvec re-reads weights per 4 tokens: a 16-row WMMA GEMM is the fix; streamed / f16 8-token matvecs
were slower), the head over verify and draft rows (per 4 rows), drafter catch-up cost at prefill. Every change
here must keep the `spec` gate exact.

## Where the remaining gap usually is

- **Prefill GEMM efficiency.** We got 25–31 TFLOP/s of ~59 peak. The next steps are
  a double-buffered LDS pipeline for B (weights) and dequantizing Q4_K straight
  into WMMA fragments.
- **Small models** (≤ 1B), for example Qwen3-0.6B: decode at 2k context ran 179
  vs 193 tok/s for llama.cpp. There, the attention and launch overhead per layer
  dominates rather than weight bandwidth. Fewer, fused launches per layer, and
  fewer splits when context is short, are the levers.
- **MoE decode** is bound by the expert gather: batch the active experts' matvecs
  per token group.

## Reporting a win

Report only numbers from the same machine, the same GGUF, the same workloads and
the same session, with llama.cpp ROCm beside them, plus the correctness gate
summary. See `loom-benchmark`. For each metric, include its target, its % of
ceiling and whether it was met. Add the result to
`loom-roofline/reference/calibration.md`.
