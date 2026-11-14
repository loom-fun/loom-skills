# Topic: optimize-loop (mechanics of the measure / ablate / change / verify loop)

## Contents

- Fast iteration (the budget is short)
- Steps 2 and 3: ablation and one change
- Step 4: kernel A/B test
- Step 5: end-to-end gates
- Step 6: measure again and record
- Pitfalls: busy GPU, prefix caching

## Fast iteration (the budget is short)

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

## Steps 2 and 3: ablation and one change

- **Ablation.** Hardware counters are unavailable (`loom-benchmark`
   counters are UNIMPLEMENTED on gfx1151). Instead, launch a kernel twice (or skip it)
   and measure end to end: `"${LOOM_SKILL_DIR}/../loom-optimize/ablate.py" PKG pkg_abl '@attn_decode'`
   writes the copy with its own model id; serve both in one loom-serve and run
   `ab_perf.py URL <id> <id>-abl-<name>`. The delta is that kernel's cost. Wrong outputs
   are fine for an ablation build; never ship one.
- **Change one thing** in the generator: kernel shape, fusion, tiling or dispatch.
   Keep the old variant behind an env switch (like `GEMM_SKIP_PADDING`, `GEMM_N_FASTEST`,
   `FLASH2_KT`; both GEMM switches are on by default, `=0` turns one off).

## Step 4: kernel A/B test

4. **Kernel A/B test.** `python "${LOOM_SKILL_DIR}/../loom-optimize/ab_kernel.py" OLD/program/llm.loom
   NEW/program/llm.loom KERNEL ab.loom-test` puts the old kernel next to the new one
   as `KERNEL_old`; its docstring has a case skeleton. Launch both on identical
   inputs in a `check.case` and compare with `check.expect.approximate`. Include a
   **negative control**: an assertion that the output is not the initial fill,
   because two kernels that both write nothing agree. Put it in a separate case that
   is expected to fail and run `loom-test ab.loom-test --device=amdgpu
   --xfail=@ab_negative_control=EXPECT/003`; the skeleton in ab_kernel.py shows it. Use the real shapes, including edge counts
   (t=1, t=4, t=63/64, a partial last tile, mixed-sequence tiles, the full context).
   Run it with `loom-test` (see the cookbook in `loom-port-llm/reference/`).
   **No local GPU free?** A/B kernel timings on DevLab with the `devlab-kbench` workflow: put the
   self-contained bench files (kernel plus a `check.case` and `check.benchmark`, kernels named `k_*`) in
   `.github/devlab/kbench/<suite>/`, list them in `.github/devlab/kbench.json` (first file is the baseline), push a
   branch and add the PR label `devlab-kbench` (`gh api -X POST repos/OWNER/REPO/issues/N/labels -f 'labels[]=devlab-kbench'`).
   Each suite runs its variants interleaved on one runner and publishes a median-ms table with % vs base as a step
   summary and artifact. Timing only: keep correctness checks in `loom-test`. See `.github/devlab/kbench/example/`.

## Step 5: end-to-end gates

5. **End-to-end gates.** `LEVEL=dev gates.sh PKG URL GGUF BENCH_DIR` (`loom-benchmark`):
   - `consistency.py`: concurrent outputs must equal sequential outputs, and a prompt repeated
     alone must repeat exactly. This catches mixed-tile, padding and race bugs that single
     requests never hit (3 of the 8 real bugs the gates ever caught).
   - `agent_golden.py`: the multi-turn tool loop (rendering, history reuse, stop tokens; thinking
     on and off). 3–4 of the 8.
   - `compare.py` on 12 short prompts and the 1k/2k goldens. Every divergence must be a near-tie
     (the compare rule, `bench_common.near_tie`). A new divergence with a large gap is a bug even
     if the text "looks fine".
   - a light touch of the rest: 2 images, 5–6 short drafted-vs-undrafted requests, sampling
     semantics.
   The llama.cpp goldens come from the cache `qualify.sh` filled, so this costs only the Loom
   side (a few minutes; `loom-benchmark` has the measured times). Kernel tests (step 4) for every new or changed
   kernel and `chat_probe --verify` for a chat-glue change come first: they are cheaper and
   more specific. Run the gates only on a build you intend to keep (it won on perf): a reverted
   build needs no gates. Don't run `LEVEL=quick`/`full` in the loop: they run nightly on the
   published cards (a 24-run `quick` loop cost 6.6 GPU-hours and caught nothing `dev` misses).

## Step 6: measure again and record

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

## Pitfalls: busy GPU, prefix caching

- **A busy GPU makes single runs meaningless** (check `pgrep -a 'loom-serve|llama-server'`):
  the same build swings 30–50 %. A/B interleaved (Fast iteration, above) with
  `"${LOOM_SKILL_DIR}/../loom-optimize/ab_perf.py" URL MODEL_A MODEL_B --metrics decode,prefill_1k,prefill_2k --rounds 5`:
  medians, B/A and a WINS/LOSES/TIE verdict against the noise band, logged to `ab_log.jsonl`. Always
  include the shortest size you report (a chunk size that won at 2k/8k lost 12 % at 1k on flint).
  Measure llama.cpp right next to the final Loom run, and say in the report that the GPU was shared.

- **Prefix caching.** Start every timed prompt with a nonce, as `perf.py` does.
- Lazy weight load, stale servers on the port, server- vs client-side timing and near-tie
  noise: see the `loom` hard rules and `loom-benchmark`.
