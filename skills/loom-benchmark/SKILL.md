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

**One command for the gates:** `[LEVEL=fast|quick|full] gates.sh PKG LOOM_URL GGUF BENCH_DIR [LLAMA_PORT]`
(needs `LLAMACPP_DIR`, or `LLAMA_URL` for a running llama-server) runs the gates below in order
and writes `BENCH_DIR/gates.json` (verdict, level, seconds per gate; logs in `gates_*.log`).

| LEVEL | When | Gates | Cost |
|---|---|---|---|
| `fast` | after each kept optimization | goldens (cached) + compare, consistency (+ vision, for a package with image input) | 1–3 min |
| `quick` | after the port (`qualify.sh` runs it) | fast + agent tool loop, sampling, needle at half the context (≤ 60k) vs llama.cpp | 5–15 min the first time |
| `full` | once, on the final build (`finalize.sh` runs it) | quick + one Loom-only probe at the full context | + one full-context prefill |

- **Vision gate** (packages with image input, every level): `vision_golden.py` sends six built-in images
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
- `fast` is enough inside the loop because the kernel paths an optimization touches are what the
  goldens exercise (short prompts, 1k–32k documents: prefill, KV paging, attention) and consistency
  covers mixed batches. Run `quick` instead when you changed the chat glue (`control.loom`), the
  storage/recurrent state, or the sampling head.
- The full-context probe is the most expensive step (a 261k prompt: ~8 min for a 0.8B hybrid) and
  only proves the size is served: run it once, on the final build.

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
  - **FAIL**: a large gap, or a token outside the top-5. That is a real bug, even
    if the text reads fine.
  - Two llama.cpp builds (Vulkan vs ROCm) disagree with each other in the same
    near-tie way, which is useful context for readers.
- llama.cpp ROCm itself changes greedy output between identical runs at near-ties
  (seen at 1k and 16k on Qwen3-4B). That is why there are two recordings: `--alt`
  scores each prompt against whichever recording Loom follows further. Before
  declaring FAIL on a single prompt, re-record it; a gap that disappears was noise.
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
  races (each prompt also runs alone twice and must be identical). A concurrent-vs-alone difference
  passes only when every prompt is deterministic alone and every divergence is a near-tie by the
  compare rule (gap ≤ 1 nat, measured with `--pkg/--llama`): decode tokens that share an epoch with a
  prefill go through the f16 GEMM path instead of the f32 matvec path. Anything else is a bug.
- 503 errors on long prompts mean `--memory_bytes` is too small for the KV pool,
  not a model bug.
- **agent_golden.py** runs three tool-using tasks (weather lookups, a calculator)
  with deterministic tool results, greedy, and checks every assistant turn:
  - `render_ok`: Loom's `/v1/chat/completions/input_tokens` equals the token count
    of the template-exact rendering (`chat_probe.simulate`, verified against the
    Jinja template). A mismatch is a control-program bug.
  - `reuse_ok`: inside a tool loop, `cached_tokens` covers ≥ 80 % of the previous
    prompt + completion (history reuse works, including recurrent state).
  - `calls_ok`: tool calls name a declared function with JSON arguments.
  - golden: Loom's turn vs llama.cpp `/completion` on the same prompt ids; a
    divergence must be a near-tie (the compare rule). A turn that would fail is re-recorded on
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
| c4_steady agg_tps | Tokens from all 4 streams inside the window where all 4 are decoding. Use this for "N requests at once". |

Rules:
- Warm up after every restart. `perf.py` and `c4_steady.py` send one short request
  first by themselves. When timing by hand:
  `curl -s URL/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"<id>","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'`.
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
