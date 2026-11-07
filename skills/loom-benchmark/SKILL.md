---
name: loom-benchmark
description: Correctness and performance methodology for Loom models vs llama.cpp ROCm — golden token-by-token comparison with log-prob near-tie analysis, a multi-turn tool-calling (agent) gate, a full-context needle gate, batching consistency, decode/prefill/long-context/concurrency throughput on real-text workloads, and how to report results honestly (tables, executive artifact). Use for "benchmark", "compare with llama.cpp", "is the port correct", "performance report".
---

# Benchmark and verify Loom models

Scripts are in `${CLAUDE_SKILL_DIR}/scripts/`; run them with the tools
venv (`~/loom/.tools/bin/python`). Workload documents: `bench_common.doc(n)` for n in 1000, 2000, 8000, 16000, 32000 (stored as `data/doc_2000.txt` and `doc_32000.txt`; the others are byte prefixes).
They are real text (Loom's own markdown docs) and their sizes are approximate token counts.
Scripts that take `PACKAGE` read `llm_config.json`, `tokenizer.json` and the model
id from a package built by `loom-port-llm/starter/build.py`.

## 0. Reference: llama.cpp ROCm (not Vulkan, not CPU)

**Get the latest lemonade build, then pick the fastest of it and what is installed** (`qualify.sh`
does both):
```bash
S=${CLAUDE_SKILL_DIR}/scripts
$S/fetch_llamacpp.sh                 # latest lemonade-sdk/llamacpp-rocm for this GPU -> ~/llamacpp-rocm/<tag> (pre-approved, ~25 s)
$S/pick_llamacpp.sh MODEL.gguf 8139 "$HOME"   # every llama-server with libggml-hip.so, benchmarked and ranked
```
The baseline is the top-ranked build. Don't choose by version string: the lemonade
nightly prints `version: 1`, and the tag is usually only in the directory name.
Builds differ up to 2× on the same GGUF (a stale build once inflated a margin from +25 % to 2.5×),
and a new architecture can need the newest release (no installed build knew `flint`).

A weaker baseline inflates every margin you report. Report the chosen build's path and tag.
lemonade-sdk/llamacpp-rocm releases are pre-approved: download them without asking. Check that
`llama-server --list-devices` shows `ROCm0` (`llamacpp_server.sh` does), because a build missing its
libraries falls back to CPU silently; `llamacpp_server.sh` also adds lemonade's TheRock runtime libraries
when a build needs them. Pick the zip that matches the GPU (`gfx1151` = Strix
Halo). Then:
```bash
LLAMACPP_DIR=<dir> $S/llamacpp_server.sh MODEL.gguf 8130 1 34000   # goldens / long context (1 slot, 34k)
LLAMACPP_DIR=<dir> $S/llamacpp_server.sh MODEL.gguf 8130 4 8192    # perf incl. 4 streams
$S/llamacpp_server.sh --stop 8130
```
Always use the **same GGUF file** as the Loom package. Never run llama.cpp and a
big Loom model at the same time if together they exceed GPU memory.

## 1. Correctness gates (before any performance claim)

**End-to-end jobs use two wrappers** (see the `loom` skill): `qualify.sh` (after the port: llama.cpp
pick, llama.cpp + Loom baselines, `LEVEL=quick` gates, roofline and budget)
and `finalize.sh` (on the final build: final perf, `LEVEL=full` gates, audit, `report.html`).

**One command for the gates:** `[LEVEL=dev|fast|quick|full] gates.sh PKG LOOM_URL GGUF BENCH_DIR [LLAMA_PORT]`
(needs `LLAMACPP_DIR`, or `LLAMA_URL` for a running llama-server) runs the gates below in order
and writes `BENCH_DIR/gates.json` (verdict, level, seconds per gate; logs in `gates_*.log`).

| LEVEL | When | Gates | Cost |
|---|---|---|---|
| `dev` | **after each kept change** (loom-optimize, loom-improve) | consistency; agent tool loop (thinking on, and off when the template toggles it); goldens (cached) + compare on 12 short prompts + 1k/2k documents; vision on 2 images (with a projector); `spec` on 5–6 short requests (drafting packages); sampling (temperature 0 = greedy, seed reproduces); needle at 16k vs llama.cpp | Qwen3.8-27B (drafting, vision): 3.3 min cached, 5.5 min with the first reference fill; Bolt-E4B: 1.1 / 2.2 min |
| `fast` | (kept for scripts that call it) | goldens + compare (20 short prompts + 1k–32k documents), consistency, vision on all 6 images | 1–3 min |
| `quick` | nightly CI; a new port's first qualification (`qualify.sh`) | fast + `spec` (full request set), agent tool loop (thinking on, and off when the template toggles it), 200-draw sampling, needle at half the context (≤ 60k) vs llama.cpp | median 11 min, up to 42 min (audit) |
| `full` | nightly CI; once on a new port's final build before its first card (`finalize.sh`) | quick + one Loom-only probe at the full context | + one full-context prefill (up to 1 h on a 27B) |

**Which level when** (from an audit of every gate run in the issue #26–#29 sessions):
- Real Loom bugs caught: consistency 3 (an MoE router race, batch-variant matvec rows, a DFlash hang under
  concurrency), agent 3–4 (stop tokens, a missing `<eos>`, history reuse), compare 2 (both at first bring-up),
  needle_full 1 class (a KV pool < 1.5x the context; `build.py` now warns statically), spec 0–1. Vision,
  sampling and the half-context needle: 0 (every failure was the reference or the harness). About 85 % of
  compare failures were noise or harness. One `LEVEL=quick` loop of 24+ runs cost 6.6 GPU-hours and found 0 bugs.
- So the development loop runs: kernel tests for new or changed kernels (`run_kernel_tests.sh`), `chat_probe
  --verify` when the chat glue changes (seconds, CPU; it caught the template derivation errors), and
  `LEVEL=dev` after each kept change. `dev` keeps the gates that found bugs at full strength and touches every
  other feature lightly, so a feature that breaks outright still fails in minutes.
- **Where it runs:** the dev machine's GPU is scarce and sequential, DevLab Dispatch is a large elastic pool.
  Locally: only what the inner loop needs. Long or deep checks and anything against published cards: DevLab CI.
  Performance for a card: a local session (the headline numbers) and a DevLab session, both kept and labeled by
  source; a gap between them is noted, never debugged (`loom-org`, "Local and DevLab performance").
- `quick` and `full` run **nightly** on the published cards (`.github/workflows/nightly-gates.yml`, below), and
  by hand only once on a new port that has no card yet (qualify.sh / finalize.sh).
- A `dev` pass is never reused by `quick`/`full` (gates.sh only skips a gate that passed at the same or a
  stronger level on the same package hash).

**Nightly CI** (`.github/workflows/nightly-gates.yml`): every night, one AMD DevLab Dispatch Strix Halo runner
per card in `.github/nightly-models.json`, in parallel, runs `.github/scripts/nightly_gates.sh CARD_REPO`:
install hrx-loom, `loom-fetch` the latest card from huggingface.co/loom-fun, build the reference example's kit
when the card's chat format needs it (`LOOM_KIT`), fetch the card's llama.cpp release, serve, `LEVEL=full`
gates. No performance is measured there (see "Where it runs"). No LLM and no skill run: a FAIL is a
regression of the published package (or of the gate scripts). llama.cpp references are cached between nights
(`actions/cache`). Run it on demand from the Actions tab (pick cards and a level), or on a pull request by adding
the `nightly-gates` label (the PR branch's scripts run). The same script runs by hand on any Strix Halo:
```bash
LEVEL=quick WORK=~/loom-nightly LOOM_PORT=8160 LLAMA_PORT=8130 MATERIALIZE=share \
  .github/scripts/nightly_gates.sh loom-fun/nvidia-NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-loom   # -> WORK/<slug>/out/
```
Adding a card: an entry in `nightly-models.json` (its `--memory_bytes`, the reference example its chat
format needs or null, the llama.cpp release it was measured against).

`NEEDLE_FULL=0` (gates.sh, finalize.sh) runs `full` without the full-context probe and records it as deferred:
the audit WARNs instead of failing. Across the #26–#29 sessions it caught one class of bug (a KV pool smaller than
the context, which build.py now warns about) at a median of ~8 min and up to an hour per run. A harness can run it
separately: `LEVEL=full gates.sh` on the same build re-uses every gate that passed and runs only the probe.

- **Speculative-decoding gate `spec`** (packages that draft; `spec_check.py`): the package must produce exactly
  what the same package produces with `draft_depth 0` (`PKG/model_target.json`, same compiled program): greedy
  and seeded-sampling chat, a 1k-token prompt, a tool call, a reasoning answer. Any difference fails (drafting
  is lossless by construction: a difference is a verification, acceptance or state-replay bug). The log also
  prints the acceptance rate. The agent and needle gates run with drafting on, so they cover multi-turn reuse
  and the full context with speculation.
- **Vision gate** (packages with image input, every level; `dev` sends 2 of them): `vision_golden.py` sends six built-in images
  (shapes + text, chart, document, photo-like, tall, wide; `--images` adds files) to Loom as chat
  requests and to llama-server `--mmproj` (gates.sh sets `MMPROJ` from the package, so the lazily
  started server loads it) as the package's own rendering + image. Prompt token counts must be equal
  (resize rule, token limits, image markers, the image request's chat rendering); replies must match
  token by token up to a near-tie. Verified: Bolt-E4B, Gemma-4-E2B, Qwen3.5-0.8B, Muse-Glimmer-30B.
- **One model at a time** (GPU memory is system RAM): the gates run with only Loom loaded; every
  llama.cpp reference missing from the cache is deferred, then gates.sh stops Loom (remembering its
  command line), lets `llama_fill.py` answer them with only llama.cpp loaded, restarts Loom and re-runs
  from the cache (a few rounds: re-recordings appear only once the first references exist). qualify.sh
  and finalize.sh measure llama.cpp with Loom stopped too. `LLAMA_URL` (a running llama-server) skips
  this and keeps both loaded: only when memory clearly allows.
- **The llama.cpp side is cached** (`BENCH_DIR/llama_cache`, keyed on the GGUF, the build and the
  request): it never depends on the Loom build, so only the first run pays for it. Re-running the
  gates on an optimized build costs Loom time only.
- `dev` is enough inside the loop: consistency and agent are the gates that found real bugs; compare keeps
  the short prompts and the 1k/2k documents (prefill, KV paging, attention); the 16k needle covers long
  context; the rest catch a feature that breaks outright. When you change the chat glue (`control.loom`), run
  `chat_probe --verify` first; `dev` then checks rendering, reuse and both thinking modes in the agent gate.
- The full-context probe is the most expensive step (a 261k prompt: ~8 min for a 0.8B hybrid, up to an hour on
  a 27B) and only proves the size is served: nightly, and once on a new port's final build.

```bash
S=${CLAUDE_SKILL_DIR}/scripts; P=~/work/mymodel/pkg; T=~/loom/.tools/bin/python; mkdir -p ~/work/mymodel/bench && cd ~/work/mymodel/bench   # outputs land here, not in the skill
$T $S/golden.py $P golden.json --llama http://127.0.0.1:8130 --long 1000,2000,16000,32000    # once per model/GGUF
$T $S/golden.py $P golden2.json --llama http://127.0.0.1:8130 --long 1000,2000,16000,32000   # 2nd recording: llama.cpp is not run-to-run deterministic
$T $S/compare.py $P golden.json --alt golden2.json --url http://127.0.0.1:8160   # -> per-prompt lines + {"verdict": "PASS"|"FAIL", ...}
$T $S/consistency.py http://127.0.0.1:8160 --model <model id>         # concurrent == sequential
$T $S/agent_golden.py $P http://127.0.0.1:8160 --llama http://127.0.0.1:8130   # tool loop: rendering, reuse, calls, golden per turn
$T $S/needle.py $P http://127.0.0.1:8160 --llama http://127.0.0.1:8130         # retrieval at 25-100% of context_capacity + over-limit 400
$T $S/sampling_check.py $P http://127.0.0.1:8160 --llama http://127.0.0.1:8130 # temperature/top_k/top_p/seed semantics + first-token distribution
```
For `needle.py` sizes above the llama.cpp slot's context, drop `--llama` (Loom-only:
it then must find the needle) or give `--sizes`. A 262k-token prefill takes minutes.
- **How golden.py works.** It renders each prompt with the package's chat block
  and sends the *token ids* to llama.cpp `/completion` (greedy, `n_probs: 5`). Both
  engines therefore see identical prompts: 20 short prompts × 64 tokens, plus 1
  prompt per long document × 64 tokens (the two question kinds alternate across sizes).
- **How compare.py works.** It asks Loom for the same completion through chat,
  rebuilds the raw text from the reasoning/answer channels, re-tokenizes it, and
  finds the first divergence. It reports the **golden log-prob gap**
  logp(golden) − logp(loom token) at that point.
- **Interpretation.** Greedy decoding through different summation orders
  *always* diverges eventually at near-ties.
  - **PASS**: every divergence is a near-tie (`bench_common.near_tie`): gap ≤ 1.0 with the token in the
    golden top-5, or Loom picked llama.cpp's second choice where llama.cpp's first has p < 0.5
    (llama.cpp itself is unsure there: on flint its own re-recording moved by gap 1.05).
    Typical healthy ports have max gap 0.1–0.6, and full matches on a quarter or
    more of the prompts.
  - **The bar includes llama.cpp's own rounding noise at that position.** A divergence the rule above
    rejects is judged again: llama.cpp re-scores the same position (prompt + its own tokens before it)
    under two numerics it uses itself, the context in one batch (MMQ matmuls) and in micro-batches of 8
    (MMVQ: another q8_1 dot-product formula and summation order, and the decode path a reused history
    took; `bench_common.llama_views`). The spread of the gap across the golden and these views is
    llama.cpp's own noise there. The divergence passes when the smallest of these gaps is ≤ 1.0 + the
    spread (`near_tie(..., views)`): the exact gap lies within llama.cpp's spread of its views, and Loom,
    rounding as coarsely, within as much again (at most 1.0 + twice the spread from the golden). Loom's
    token must appear in at least one view; a view that does not list it bounds its gap from below.
    Why: both engines round activations to 8 bits at every matmul, so a single rounding-level change
    (f16 vs f32 KV, MMVQ vs MMQ, ties-to-even vs away) can move a gap by about 1 nat where the model is
    truly tied (gemma-3-1b, first token: an f64 forward puts the two candidates 0.004 apart, the golden
    gap is 1.36; MoE routing flips move more). A fixed larger constant would let real bugs through
    everywhere; the measured noise widens the bar only where llama.cpp disagrees with itself.
    Only the first divergence of a prompt (turn, probe) is judged, as before. compare.py, agent_golden.py,
    needle.py and spec_check.py use it (`near_tie_at`); vision_golden.py (image prompts) does not. No
    micro-batch view where llama.cpp cannot extend its cache in small steps (recurrent state).
  - **FAIL**: a large gap, or a token outside every view's top list. That is a real bug, even
    if the text reads fine.
  - Two llama.cpp builds (Vulkan vs ROCm) disagree with each other in the same
    near-tie way, which is useful context for readers.
- llama.cpp ROCm itself changes greedy output between identical runs at near-ties
  (seen at 1k and 16k on Qwen3-4B). That is why there are two recordings: `--alt`
  scores each prompt against whichever recording Loom follows further. Before
  declaring FAIL on a single prompt, re-record it; a gap that disappears was noise.
  - A plain re-run of a *text* prompt at `-np 1` is often bit-identical, so it proves little.
    The informative check is llama.cpp's sensitivity to its own summation order. The gates measure
    it at the divergence already (one batch vs micro-batches of 8, above). For a FAIL that remains,
    re-record the one prompt with the server started under other legitimate numerics: `LLAMA_SPEC="-ub 128"`,
    `"-ub 2048 -b 2048"` and `"-fa off"` (llamacpp_server.sh appends LLAMA_SPEC). Then compare
    the top-5 at the divergence.
  - If llama.cpp's own top-1 changes across these settings, or Loom's token falls within the
    1.0 gap under one of them, the FAIL is a near-tie. Document it with the per-setting top-5;
    don't change the kernels to chase it. Example, Flint 15.5k summary at token 15: llama.cpp
    picked ` across` (default), ` combine` (`-ub 128`, `-ub 2048`) and ` summarize` (`-fa off`).
    Loom's ` final` had gap 1.39 (default), 1.37 (`-ub 128`), 1.06 (`-ub 2048`) and 0.72 (`-fa off`).
  - If llama.cpp stays on one token under every setting and Loom's gap stays above 1.0, it is
    *not* proven noise. Before changing kernels, check that both sides compute the same model:
    - Near-exact reference: dequantize the GGUF to F32 (`llama-quantize --allow-requantize IN OUT F32`)
      and run it on the CPU (`LLAMA_SPEC="-ngl 0 --no-op-offload -nkvo --no-mmproj-offload -fa off
      -ctk f32 -ctv f32"`). If it agrees with llama.cpp's GPU run, Loom is the outlier.
    - Vision: dump llama.cpp's image embeddings (`MTMD_DEBUG_EMBEDDINGS=out.bin llama-mtmd-cli ...
      --image X --jinja`: int32 rows, int32 width, then f32 data) and compare them with Loom's
      encoder. Row cosines far below 0.999 are a preprocessing or encoder difference, not numerics.
      Feeding llama.cpp the image already resized to the processed size isolates the resize.
    - Example, Bolt Q4_K_M vision `photo` (1024x768) at token 7: 17 recordings with llama.cpp b1355
      all chose ` grid` (gap 1.02–1.54 to Loom's `,`), and the F32 CPU run agreed (gap 1.40). Loom's
      kernels were not the cause: f32 activations, matvec-only prefill and attention variants all
      gave the same reply. b1355 letterboxed Gemma 4 images (row cosine to Loom's embeddings 0.32 at
      worst); upstream llama.cpp 7bd18083 (lemonade b1368) resizes without padding, as Loom does.
      Against b1368 all six images, `photo` included, match token by token (57 tokens).
- Golden ids come from llama.cpp's `tokens`. `completion_probabilities` omits
  byte-fallback tokens (emoji, partial UTF-8), so top-5 lists are kept only while
  they line up.
- Document sizes are approximate. With a 262k-vocab tokenizer (Gemma),
  `doc_32000` is about 35k tokens, which is over a 32768 context. Either build the
  package (and llama-server) with a 40960 context, or use `--long 1000,2000,16000`
  and say so.
- Long goldens (16k/32k) are what catch KV paging, prefill chunking and attention
  masking bugs.
- `consistency.py` catches mixed-sequence epoch bugs (padding, tile uniformity, fallback lists) and
  races (each prompt also runs alone twice and must be identical on the same path). A concurrent-vs-alone
  difference passes only when every prompt is deterministic alone and every divergence is a near-tie by the
  compare rule (gap ≤ 1 nat, measured with `--pkg/--llama`): decode tokens that share an epoch with a
  prefill go through the f16 GEMM path instead of the f32 matvec path. Anything else is a bug.
  "Same path" means equal `cached_tokens`: a repeat that hits the prefix cache (its history retained from
  the concurrent phase) recomputes only the uncached tail with the decode-shaped kernels, so it can flip a
  near-tie against the first, uncached run without any race (issue #39); such a prompt runs alone again
  until two runs share a path, and the hit-vs-miss difference takes the near-tie rule. To hunt a suspected
  race, `determinism.py URL --model ID --conc 1 --reps 20` (alone, never a cache hit: any difference is a
  race), `--pairs` (cache miss vs hit) and `--conc 4` (mixed epochs) separate the three causes.
- 503 errors on long prompts mean `--memory_bytes` is too small for the KV pool plus the prefill scratch
  (loom-port-llm troubleshooting 5; even on a fresh server, whatever the message says about retained state),
  not a model bug.
- **agent_golden.py** runs three tool-using tasks (weather lookups, a calculator)
  with deterministic tool results, greedy, and checks every assistant turn:
  - `render_ok`: Loom's `/v1/chat/completions/input_tokens` equals the token count
    of the template-exact rendering (`chat_probe.simulate`, verified against the
    Jinja template). A mismatch is a control-program bug.
  - `reuse_ok`: inside a tool loop, `cached_tokens` covers ≥ 80 % of the previous
    prompt + completion (history reuse works, including recurrent state). Required when the template
    keeps that history: it is a prefix of this prompt, or (output not verified by the golden) agrees with
    it for ≥ 80 % of its tokens. Output that matched the golden and that the template re-renders (Gemma 4
    adds a `\n` before `<channel|>`) is not required: loom-serve reuses only exact extensions of a
    retained history (a note prints where the template changed it). Neither is a turn after a reply whose
    parse dropped a second call where the template takes one (Llama 3.1: the reply generated more tokens than
    its parse holds, all of them matched the golden, and the golden's continuation opens another call): the
    retained history holds tokens no rebuilt prompt has.
  - `calls_ok`: tool calls name a declared function with JSON arguments.
  - golden: Loom's turn vs llama.cpp `/completion` on the same prompt ids; a
    divergence must be a near-tie (the compare rule). Loom's turn is its parsed reply re-tokenized, so
    the comparison is on text: the same text split into other tokens (the model's `.\n` `\n` `\n` where
    the tokenizer merges `.\n\n\n`) is no divergence (`bench_common.text_divergence`; it once read as
    "llama.cpp ends at `<tool_call>`" because the 12-token windows end at different text). What the parsed
    reply cannot tell is taken from the golden: a reply with nothing after its reasoning is also rebuilt
    unclosed (EOS inside the reasoning), and whitespace between or after tool calls (the rebuild prints
    `call_separator`) follows the golden's. A turn that would fail is re-recorded on
    llama.cpp (up to twice) and scored against the recording Loom follows further (llama.cpp's
    run-to-run nondeterminism, above).
  "The model ended inside its reasoning" notes are a greedy-decoding behavior of
  small thinking models, confirmed when llama.cpp's own distribution at that
  point prefers EOS; not a Loom bug by itself.
- **sampling_check.py**: temperature 0 and `top_k` 1 equal greedy, a fixed seed
  reproduces, different seeds vary, and 200 temperature-1 draws of the first token
  match llama.cpp's probabilities on the same ids (total-variation distance within
  sampling noise; Qwen3-0.6B: 0.011).
- **needle.py** hides access codes in a real-text haystack and asks for them (thinking off
  when the template allows), then sends a prompt over the capacity (must be a clean HTTP
  400, server still healthy). **Retrieval is not the gate**: a small model finds or misses a
  needle on a near-tie first token (Qwen3.5-0.8B at 130k: llama.cpp's top two first tokens
  0.013 nats apart), so a handful of probes says nothing statistically. Up to `--llama-max`
  each probe is compared with llama.cpp on the same ids and passes when Loom found the
  needle, llama.cpp missed it too, or the first divergence is a near-tie (the compare rule, in
  the top-5: the compare.py rule). The full-context probe (Loom only; llama.cpp would take
  many minutes) proves the size is served. Found counts are reported as information.

## 2. Performance

```bash
$T $S/perf.py http://127.0.0.1:8160 loom-baseline --model <id> --log perf_log.jsonl
$T $S/perf.py http://127.0.0.1:8160 loom-baseline --model <id> --long --log perf_log.jsonl
$T $S/perf.py http://127.0.0.1:8130 llama-rocm --kind llama --log perf_log.jsonl     (and --long, with the 1-slot 34k server)
$T $S/c4_steady.py http://127.0.0.1:8160 4 --model <id>;  $T $S/c4_steady.py http://127.0.0.1:8130 4
```

| Metric | Definition |
|---|---|
| decode_tps | 300-token essay from a short prompt. Client-side: (tokens−1)/(total−TTFT). |
| prefill[] | 1k and 2k-token documents with max_tokens=4. prompt_tokens / TTFT, the median of 5 runs (`--repeats`); single runs varied 1.5–2.9k tok/s on one build. |
| decode_tps_2k / _32k | Decode with that much context in the KV cache. |
| prefill_long[] | 8k / 16k / 32k documents. |
| c4_steady agg_tps | Tokens from all 4 streams inside the window where all 4 are decoding. Use this for "N requests at once". A run in which a stream failed exits 1 and records nothing (`make_info.py` also skips a rate <= 0). |

Rules:
- A card's headline numbers come from one local session: Loom and llama.cpp back to back on a quiet local GPU, the
  programs being published. Its DevLab columns come from a DevLab session on the same programs (`loom-org` step 3);
  the local ÷ DevLab ratio is recorded, never investigated. A DevLab job's variant comparisons (`devlab-perf`,
  `devlab-experiment`) compare within that job only.
- Warm up after every restart. `perf.py` first sends its first timed request's own shape, untimed (the 300-token
  essay with another nonce: prefill, decode, drafting and first-use kernels all run once; a trivial "Say hi."
  left DevLab's short-prompt decode 51-60 % below the local session's while 2k decode agreed), and `c4_steady.py`
  one short request. Every caller (qualify, finalize, devlab_perf) gets the same warm-up. When timing by hand:
  `curl -s URL/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"<id>","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'`.
- A decode rate counts only if its request generated at least half its max_tokens. `perf.py` logs each timed
  decode request's `completion_tokens` (both engines); a short one is re-requested once with another nonce, then
  logged as null. A request that stops after a few tokens reads as a fast rate (drafted tokens arrive in bursts):
  one 32k decode read 2.6x its KV-bandwidth ceiling that way.
- With thinking models, llama.cpp's chat endpoint renders 2 fewer prompt tokens
  (no `<think>\n`) than Loom. That is negligible for throughput, and goldens are
  unaffected.
- Every prompt carries a nonce, so prefix caching can't inflate prefill.
- Loom packages need `execution.rows ≥ 4` for the 4-stream numbers, and llama-server
  needs `-np 4`.
- llama.cpp numbers use its server-side `timings`, which exclude HTTP and
  tokenization. That slightly favors llama.cpp; say so.
- Measure on an idle machine: stop the demo servers and other models first
  (`pgrep -af "loom-serve|llama-server"`; Windows: `ps -W | grep -E 'loom-serve|llama-server'`).
- Noise levels:
  - decode and c4 vary by about ±2–3 %;
  - short prefill varies a lot, so use the median;
  - long prefill (8k–32k) single runs vary ±5–10 %, so repeat any metric you base
    a decision on.
- Don't use kernel microbenchmarks as headline numbers. The 32 MB MALL inflates
  them.

## 3. Reporting

**Speculative decoding in reports.** Say which drafter the package uses (`llm_config.json` `draft`), show the
tuning table (`bench/draft_tune.json`: decode tok/s per method and depth, including no drafting) and the
acceptance rate from the `spec` gate log. Decode numbers with drafting depend on the text: report them on the
standard workloads and say so.

**The baseline drafts too (drafting vs drafting).** `qualify.sh` runs `scripts/llama_spec_tune.py` before the
llama.cpp baseline: the same drafters Loom has (`--spec-type draft-mtp` for a GGUF with an MTP block,
`--spec-type draft-dflash --spec-draft-model DRAFTER` for the DFlash drafter `find_drafter.py` found) at
`--spec-draft-n-max` 1-4 / 3, 5, 7, its n-gram drafters (`ngram-simple`, `ngram-mod`) for every model, plus none,
each timed on the same requests as Loom's `tune.py`; the fastest
(>= 3 % over none) goes to `bench/llamacpp_spec.json`, and `llamacpp_server.sh` adds it (`LLAMA_SPEC`) to every
llama.cpp performance run in qualify and finalize. References (goldens, gates) never draft. A build that rejects
a method is recorded as failed in the table and that method is skipped; the report shows the table.

**Run the audit first** (from `bench/`):
```bash
$T $S/audit.py .     # -> headline table + checks; exit 1 on FAIL; writes audit.json
```
It checks:
- **Baseline.** `llamacpp_pick.json` exists, and the measured baseline is the
  fastest build (decode and prefill).
- **Plausibility.** Loom is not above ~0.97 of a ceiling, and llama.cpp is not
  implausibly far below one.
- **Tracking.** `targets.jsonl` has a start row and later measurements.
- **Gates.** `compare.py` and `consistency.py` log to `gates.jsonl` by themselves.
  The latest compare must be PASS, and run on the final commit. Consistency must
  be PASS, or a documented near-tie: write the token, the gap and the cause in
  `consistency_note.md`.

Fix every FAIL before reporting, and report every WARN. The headline table is the
source for the report's numbers.

- **Lead with the bottom line**: a table with Loom and llama.cpp ROCm side by side
  for each metric, followed by one line on the correctness gate (e.g. "24 prompts
  up to 32k tokens; all divergences near-ties (max gap 0.97); concurrent ==
  sequential").
- **State the setup**:
  - hardware;
  - SDK wheel (CI artifact id or commit);
  - llama.cpp build tag and backend;
  - GGUF file and quant;
  - the package preset (context, pool, prefill chunk, rows).
- **Be honest about losses and limits.** Report the metrics where Loom is slower
  and any targets you missed (e.g. "prefill target 3,700 tok/s not reached:
  GEMM at ~50 % of WMMA peak"), plus what is unsupported (sampling, tools,
  images).
- **No "before this session" columns** in executive tables. Give current Loom
  against the current baseline.
- **Generate it.** `scripts/report.py BENCH_DIR --pkg PKG [--notes notes.txt]` (`finalize.sh` runs it)
  writes `report.html` (single page, light/dark) and `report.md` from the logs: bottom line,
  headline table, gates, setup, decisions, audit, timeline. Write only the notes yourself (what
  was optimized, what was skipped and why, losses), rerun `report.py --notes`, and publish
  `report.html` as is. Extend the generated page only if the user asked for more.
- A hand-extended report (only on request): load the artifact-design skill first, and generate it from
  the results JSON with a small script, so no number is typed by hand.
- Keep raw logs (`perf_log.jsonl`, `compare.json`, `golden.json`) next to the
  report, so every number can be traced.

### Hugging Face model card (loom-fun repos)

A published card's `README.md`, `info.json` and `REPORT.md` are **generated, never hand-written**: the skill
supplies data, `scripts/card/` turns it into the card. Every loom-fun card has the same sections, rows and
footnotes (lead-in, quick start, performance vs llama.cpp, features, correctness gates, files); only the values
differ. No prose of your own on a card: no "what's new", no explanations, no caveats. A value without evidence
renders as not measured / no evidence / unknown; never fill one in by hand.

`info.json` (schema `card/info_schema.json`, `"schema": "loom-card/2"`) holds only what no package file
records: measurement sessions (per-run values, commands, the baseline's version and exact server arguments),
the spec gate's acceptance, gate evidence, the device and Loom SDK of each source, dates and versions. The card
reads everything else from the repo's own files: sources, revisions, sha256 and sizes from `model.json.lock`;
context, rows, presets and drafting depth from `model.json`; features, the drafter and the generating
loom-skills commit and SDK from `llm_config.json` (`"generator"`, stamped by `build.py`).

1. Evidence files (`loom-card-evidence/2`) from the run's own outputs, made **on the machine that measured**
   (each records `scripts/machine_info.py`'s detected device and the SDK under `$LOOM_ROOT`):
   - `card/make_info.py perf --perf-log bench/perf_log.jsonl --baseline-version llamacpp-rocm/bNNNN
     --baseline-args "$(cat bench/llamacpp_args.txt)" --package PKG -o ev_perf.json` (finalize.sh's `final` /
     `llama-final` rows and the llama-server arguments it recorded; or `--loom-jsonl`/`--llama-jsonl` +
     `--c4-loom`/`--c4-llama` for separate same-session logs, `--command` for each measuring command, and
     `--metrics` for a later session that re-measured some rows);
   - `card/make_info.py gates bench/gates.json [ktest.out --time ...] --package PKG -o ev_gates.json` (also the
     spec gate's `acceptance N% of M proposals` from `gates_spec.log`);
   - `card/pull_devlab.py loom-fun/REPO --package DIR -o ev_devlab.json`: the latest nightly-gates run on AMD
     DevLab Dispatch: gates of every finished level (job URL as evidence), and the spec gate's acceptance from the
     job's `gates_spec.log` as the card's `spec_acceptance`, stamped with the job's programs (a drafting statistic of
     model, drafter and text, not a timing: it counts from any source). A perf session recorded with
     `--source-kind devlab` (a scripted DevLab port, a `devlab-perf` job: `loom-org` step 3) fills the card's DevLab
     columns and stays next to the local session (`card_lib.card_metrics`);
   - `card/make_info.py roofline --package DIR -o ev_roofline.json` (tools venv; CPU only, reads the GGUF from the
     Hugging Face cache): the hardware ceiling of every card metric (`loom-roofline` `roofline.py`) with the
     device constants and the roofline.py sha256 and commit. Drafting packages: decode and 4-stream ceilings are
     verify passes per second x (1 + acceptance x depth) tokens per pass (the drafter's cost not counted: an upper
     bound). Per row the acceptance is the one measured in the same session as the card's value (perf.py records
     each decode request's, c4_steady.py its streams', from loom-serve's `/metrics` counters); a row whose session
     recorded none falls back to the spec gate's, and the card says so (the gate's text can accept far more or
     less than perf.py's essay: 41 % vs ~17 % on a small drafted model, a 1.8x too-high ceiling). Without either
     they have none. A first card has no `DIR/info.json` yet: pass the run's evidence (`--evidence ev_perf.json
     ev_gates.json`), or build once, run roofline, and build again.
2. `card/make_info.py build DIR --repo loom-fun/REPO --memory-bytes N --output OUT --report generated
   --evidence ev_*.json -o DIR/info.json` (`--generator-commit` defaults to this checkout's HEAD; to regenerate a
   published card with only new evidence: `--from-info DIR/info.json`, re-read from the Hub right before, which
   also supplies `--repo`). DIR holds the repo's `model.json`,
   `model.json.lock`, `llm_config.json` (and `program/`). Per gate the latest evidence wins, so a later failing
   run shows as a failure, except that on the same programs a later pass at a lower level (dev < fast < quick <
   full) keeps the earlier higher-level pass (a published date-only gate and a same-day timestamped job: the new
   evidence is the later); per metric the last local session that measured it, and separately the last DevLab
   one. `--report attached` instead keeps a
   REPORT.md written by the run (the factory's) or an older card's narrative, as is.
   A new or replaced package: `card/publish_package.py prepare PKG DIR --source ORG/GGUF-REPO` first, on the machine
   that holds the GGUFs (copies the package without weights, pins each file's origin at the revision it was validated
   on, writes `model.json.lock` with `loom-fetch --update-lock`), then `make_info.py build DIR ...` on that DIR.
3. `card/render_card.py DIR` writes `DIR/README.md` (and `DIR/REPORT.md` for `--report generated`: setup,
   commands, every run's value, the roofline, gate evidence). The performance table shows per metric llama.cpp,
   Loom, the margin, the roofline and Loom as % of it (local session), then DevLab's llama.cpp and Loom and the
   local ÷ DevLab ratio of each engine; footnotes say what the roofline and the ratio are. Loom-only DevLab
   sweeps that older cards recorded in `info.json` (`devlab`) stay valid and are carried over by `--from-info`, but
   are not rendered. When republishing a card, upload only those three files: `publish_package.py upload DIR --repo
   REPO --card-only --parent-commit SHA` (SHA: the repo commit whose info.json the card was built from; the upload
   fails instead of overwriting a card another worker published since). Never re-upload a repo's programs, `model.json`, lock or weights for a card change.
   A whole package: `publish_package.py upload DIR --repo REPO` (removes files an in-place update leaves stale).

**Performance guardrails** (`card_lib.perf_flags`, applied by `loom-org`'s `org_status.py`): a metric whose Loom
value is below half the efficiency Loom ports have demonstrated (far from roofline) sends the model back to a local
GPU worker with `loom-optimize`, and to a deep debugging session if it stays far after that (`loom-org` step 5). A
local value measured on other programs (stale), or a metric only DevLab measured, is re-measured in a local session
first; a card without a DevLab value on its programs gets a DevLab perf session (low priority). The local ÷ DevLab
ratio is never a guardrail.

The card's Versions line shows the schema, the loom-skills commit and hrx-loom build that generated the
programs, and the card generator commit. `loom-setup/scripts/install_loom.sh` writes `$LOOM_ROOT/loom-install.json`
(version, hrx-system commit, CI artifact or PyPI release, wheel sha256), which `build.py` stamps into the package
and the evidence records for every measurement. A schema change that only adds keys, metrics or gates keeps
`loom-card/2` (readers ignore unknown names); renaming, removing or retyping a key bumps it.
To change what every card says, change `render_card.py` (and `card/test/check.sh`'s golden files with
`UPDATE=1`, CI `check-cards`), then re-render all cards. `card/import_card.py` maps cards published before
loom-card/1 to evidence (one time; their versions are recovered from the factory run and the report they
name, and their narrative report becomes REPORT.md verbatim). The loom-factory workflow's publish step runs
steps 1-3 itself and uploads the run's REPORT.md.
