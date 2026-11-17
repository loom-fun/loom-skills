# Topic: optimize-loop (mechanics of the measure / ablate / change / verify loop)

## Contents

- Fast iteration (the budget is short)
- Steps 2 and 3: ablation and one change
- Profiling: per-kernel device time of a whole model, counters
- Ablation: known distortions
- Local pre-flight (no GPU): programs, compile report, ISA
- Pricing a design: measured kernel-to-model conversions
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

- **Per-kernel time first, ablation to confirm.** The HAL's dispatch statistics give every kernel's device time in one
   session (next section); hardware counters, executable traces and device metrics are UNIMPLEMENTED on gfx1151 and gfx1201.
   For a counter-like question (what does this kernel wait for?), or to cross-check a profile row, launch a kernel twice
   (or skip it) and measure end to end (read "Ablation: known distortions" first): `"${LOOM_SKILL_DIR}/../loom-optimize/ablate.py" PKG pkg_abl '@attn_decode'`
   writes the copy with its own model id; serve both in one loom-serve and run
   `ab_perf.py URL <id> <id>-abl-<name>`. The delta is that kernel's cost. Wrong outputs
   are fine for an ablation build; never ship one.
- **Change one thing** in the generator: kernel shape, fusion, tiling or dispatch.
   Keep the old variant behind an env switch (like `GEMM_SKIP_PADDING`, `GEMM_N_FASTEST`,
   `FLASH2_KT`; both GEMM switches are on by default, `=0` turns one off).

## Profiling: per-kernel device time of a whole model (devlab-perf step `profile:V`)

`loom-serve --device_profiling_mode=dispatch-events,executable-metadata --device_profiling_flush_interval_ms=5 --print_device_statistics`
makes the HAL time every dispatch on the device and print, at a graceful shutdown (`stop.sh`, SIGTERM), one row per kernel:
`dispatch NAME count=N total=T avg=A` (stderr, the serve log). Without `--device_profiling_mode` the flag prints queue-level rows only
(`dispatch_function_total=0`). **The flush interval matters**: each queue keeps 4096 records between flushes, a prefill dispatches
thousands of kernels per second and the rest is dropped (`dropped_records=N` in the output; a first run lost 80 % of a 2k prefill).
`devlab_perf.sh` step `profile:V[:probe,pf2k,pf8k,dec]` serves V once per scenario (so each table holds only that scenario's
requests), drives it with `profile_requests.py` and writes `out/profile_V_<scenario>.{md,json,serve.log}` through `profile_stats.py`
(families: `profile_families.json`, first regex that matches a kernel name wins; ms/token = total over the tokens the scenario
processed; decode adds ms per verify step from the draft counters). Compare the table's `busy` with the request `wall` for launch/idle overhead (2 % on the MXFP4 card).
Two sessions with the same number of requests but different prompt lengths separate per-token cost from per-request fixed cost
(a request that asks for 2 tokens also runs a decode step). One-time kernels
(`g_prep*`, first-use) are in the totals of the first request: they have their own family. A standalone `rocprofv3` does not apply:
the `amdgpu` device is the native libamdf driver, not ROCr.

Per-kernel timing of one kernel in isolation: `loom-benchmark --profile-final-batch=true` (`profile_replay` rows; devlab-kbench).
Compile-side facts without a GPU: `--compile-report=details` (spills, register pressure, instruction mix, `target_capability_rows`:
gfx1201 `matrix_fp8_native_kind` / `matrix_bf8_native_kind` = `unscaled`, `matrix_fp4_native_kind` = `none`).
`--profile-data=counters|counter-ranges|device-metrics|executable-traces` on gfx1201 (kbench suite `prof_counters_r9700`, run
43803638485): every one returns `UNIMPLEMENTED; unsupported GPU profiling data families 0x10 / 0x400 / 0x100 / 0x40`
(`runtime/src/iree/hal/drivers/amd/gpu/profile.c:636`); the run does not fail, the decoded rows are `state: unavailable`. A named
counter (`--profile-counter=SQ_WAVES`) with `counters` exits non-zero with the same error. Not a flag problem: the HAL (hrx-loom 0.1.0) lacks these families.

## Ablation: known distortions

Doubling or skipping a kernel answers "what does this kernel cost" only when the second launch behaves like the first. Measured traps
(q38r9700 and mxfp4st campaigns, R9700, 2026-11-15/16):
- **Caches.** The second launch of a weight matvec reads its weights from the 64 MB MALL / L2 the first one filled: doubling under-states
  a bandwidth-bound kernel (run 43745471701). Compare a doubling row with the HAL profile of the same build; they agreed within ~10 % on
  the prefill producers (run 43803217338) and disagreed on the matvecs.
- **Power.** A variant that computes on zeros or skips its math draws less power and lets the rest of the GPU clock higher: zero
  operands made the fp8 WMMA GEMM 14-17 % faster on a power-capped card (kbench 43865260671). An ablation that removes work is an upper bound.
- **State.** Doubling a non-idempotent launch (in-place updates such as `silu_mul`, the speculative-accept kernels, the head select of a
  verify, an in-place RoPE) corrupts the state: drafted acceptance fell to 0.00-0.27 and the row was void (runs 43800856774 ..
  43804664636). Normalise a drafted row by tokens per step (`step ms = 1000 (1 + depth x acceptance) / tok/s`) and drop any variant
  whose acceptance moved.
- **Matching.** A dual-target build launches through `template.apply<@selN_...>` lines: a regex on the bare kernel launch matched nothing
  (runs 43803038899, 43805820597). `devlab_perf.sh` prints `ablation: doubled N launches`; check N before reading the row.
- **Coverage.** Double every launch family, not the suspects: a first pass that named GEMM, GDN, producers and attention left 12 % of a
  prefill unexplained. Only the sum over families says what is left.

## Local pre-flight (no GPU): programs, compile report, ISA

- **Programs.** `loom-opt PKG/program/<file>.loom --pass=canonicalize > /dev/null` over every generated program file of every variant
  parses and type-checks the launches in about a minute on the CPU (TYPE / SYMBOL errors that otherwise surface on the runner; one wasted
  DevLab run, 43813833801). Command-program lowering itself is first seen on a GPU (topic `sdk`).
- **Compile report.** `loom-compile ... --compile-report=details`: spills, register pressure, LDS, instruction mix, target capability rows.
  Use it to price a hand-written conversion or a prologue before wiring it (an integer-bit f32 -> e4m3 conversion measured 54 VALU
  against 21 for the SDK's per 4 values, and was dropped without a run).
- **ISA.** `loom-compile --format=amdgpu-hsaco` writes the code object; ROCm's (or TheRock's) `llvm-objdump -d --mcpu=gfx1201` (the target
  of the device) disassembles it and `llvm-readelf --notes` gives VGPR, LDS and scratch. Count WMMA / SALU / VALU / waits per loop step
  and read where loads are waited on before spending a kbench round; a schedule annotation (`scf.for pipeline(N)`, split barriers)
  is checked here first (both were rejected from the ISA on the fp8 GEMM). Fewer instructions is not speed by itself: a kernel that is
  not issue-bound does not get faster (device node, prefill).

## Pricing a design: measured kernel-to-model conversions

What a kernel-level gain became end to end (R9700, MXFP4 27B card, 2026-11-16): the rule is gain ~ the family's profiled share x the
kernel's reduction, minus what the fused kernel takes on.
- A fused verify FFN producer, -4 % in kbench on a family that is ~4 % of the drafted step: tie end to end (43812485595).
- A fused GDN prefill, -31 % on its family in kbench, -37 % in the model, on a family that is ~9 % of prefill: +2-3 % prefill (43811667696).
- A store GEMM -5..-7 % in kbench on about half of the GEMM time: +1-2 % prefill (43871204257).
- Fusing a residual add into a GEMM epilogue removed 0.0084 ms/token from the producer and added 0.0072 to the GEMM: net +0.2 %
  (43825101805). Judge a fusion by the sum of both kernels in the profile.
- Removing a write no kernel reads (an operand copy the consumer never loads) gave +8-10 % for a flag argument (43799119336): audit every
  store of a hot path against its readers before designing new kernels.
- A technique ported from a leader keeps the leader's measured net: a deferred recurrent-state fold that the source measured at ~0
  tied here as well (#971).

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
   **Serving-shaped cases.** The runtime packs prompt rows, decode anchors and speculative verify spans into one epoch, out of packet
   order, with gaps and stale packets and with more configured spans than live ones; a kernel tested only on one clean span passed its
   kbench and failed the consistency gate (fused GDN, run 43824481645). Give every new kernel a case in that shape. Reference
   quantizers mirror the hardware's sign of zero (e4m3 -0 is 0x80), and each new case draws from its own random generator (a shared
   stream shifted the data of every later case and failed an unrelated test). When a generator rewrites another program by text
   pattern (the verify program derived from advance), count the rewrite's anchors per branch in the derived program.
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
   - REVIEW: re-profile, then pivot or stop. Forking from an earlier decision is an
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
