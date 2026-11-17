---
name: loom-benchmark
description: Correctness and performance methodology for Loom models vs llama.cpp ROCm — golden token-by-token comparison with log-prob near-tie analysis, a multi-turn tool-calling (agent) gate, a native-context admission check, batching consistency, decode/prefill/long-context/concurrency throughput on real-text workloads, and how to report results honestly (tables, executive artifact). Use for "benchmark", "compare with llama.cpp", "is the port correct", "performance report".
---

# Benchmark and verify Loom models

There are no needle-in-a-haystack gates: long-context retrieval is not part of the correctness gates (the 1k-32k goldens
cover long prefill and KV paging). One cheap check remains, the **native-context admission probe** (`ctx_admit.py`, run
by `gates.sh` at `quick` and `full`): one request of about the package's native context is sent, must be admitted by
loom-serve (not refused with a 4xx/503) and is then cancelled. It is a check, not a gate: no gate row, no card row;
`gates.json` records it under `context_admit` and a refusal fails the run and the audit.

**Bundled paths ? first shell action.** Before running commands, set `LOOM_SKILL_DIR` in the shell to the
absolute directory containing **this** `SKILL.md`, using its path from the skill listing or read tool.
In Claude Code, run `export LOOM_SKILL_DIR="${CLAUDE_SKILL_DIR}"` (the host substitutes the path).
In Codex, Pi or another host, export `LOOM_SKILL_DIR` to the loaded file's directory. Use a
shell-compatible absolute path (Git Bash on Windows). Rebind before using another skill's commands
and in every new shell; it is not the working directory or SDK root. On Windows, use native `C:/...`
paths for file tools; `/c/...` paths belong to Git Bash. Never run an unresolved placeholder. Run bundled commands by quoted absolute path; do not run `scripts/...` from the working directory
or change into the installed skill tree. Install the complete Loom suite: sibling paths are required.

Scripts are in `"${LOOM_SKILL_DIR}/scripts/"`; run them with the tools
venv (`~/loom/.tools/bin/python`). Workload documents: `bench_common.doc(n)` for n in 1000, 2000, 8000, 16000, 32000 (stored as `data/doc_2000.txt` and `doc_32000.txt`; the others are byte prefixes).
They are real text (Loom's own markdown docs) and their sizes are approximate token counts.
Scripts that take `PACKAGE` read `llm_config.json`, `tokenizer.json` and the model
id from a package built by `loom-port-llm/starter/build.py`.

## 0. Reference: llama.cpp ROCm (not Vulkan, not CPU)

**Get the latest lemonade build, then pick the fastest of it and what is installed** (`qualify.sh`
does both):
```bash
S="${LOOM_SKILL_DIR}/scripts"
"$S/fetch_llamacpp.sh"                 # latest lemonade-sdk/llamacpp-rocm for this GPU -> ~/llamacpp-rocm/<tag> (pre-approved, ~25 s)
"$S/pick_llamacpp.sh" MODEL.gguf 8139 "$HOME"   # every llama-server with libggml-hip.so, benchmarked and ranked
```
The baseline is the top-ranked build (a build choice for local qualification; cards pin one build per backend, and
configurations are never picked by speed: `reference/methodology.md`). Don't choose by version string: the lemonade
nightly prints `version: 1`, and the tag is usually only in the directory name.
Builds differ up to 2× on the same GGUF (a stale build once inflated a margin from +25 % to 2.5×),
and a new architecture can need the newest release (no installed build knew `flint`).

**Vulkan, the second baseline.** Every card also carries the official ggml-org llama.cpp Vulkan build (`b12211`,
sha256-pinned in `data/llamacpp-vulkan.sha256`) next to the ROCm one: `LLAMACPP_BACKEND=vulkan fetch_llamacpp.sh DIR
b12211` and `llamacpp_server.sh` (same variable) run the same server arguments, drafting settings and workloads as the
ROCm baseline. It is measured only in `devlab-perf` jobs (`llama@vulkan:N`, `llamac4@vulkan` steps, `loom-fleet` publish.md step 3)
and recorded with `make_info.py perf --backend vulkan` under `performance.baselines.vulkan`; the card renders it as one
more column. It never feeds a card value, a guardrail or an `org_status` action. A setting Vulkan cannot run
is a "not supported" cell with the reason (`baseline.differs`, rendered on the card), never swapped silently
(`reference/methodology.md` rules 1-6).

A weaker baseline inflates every margin you report. Report the chosen build's path and tag.
lemonade-sdk/llamacpp-rocm releases are pre-approved: download them without asking. Check that
`llama-server --list-devices` shows `ROCm0` (`llamacpp_server.sh` does), because a build missing its
libraries falls back to CPU silently; `llamacpp_server.sh` also adds lemonade's TheRock runtime libraries
when a build needs them. Pick the zip that matches the GPU (`gfx1151` = Strix
Halo). Then:
```bash
LLAMACPP_DIR=<dir> "$S/llamacpp_server.sh" MODEL.gguf 8130 1 34000   # goldens / long context (1 slot, 34k)
LLAMACPP_DIR=<dir> "$S/llamacpp_server.sh" MODEL.gguf 8130 4 8192    # perf incl. 4 streams
"$S/llamacpp_server.sh" --stop 8130
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
| `dev` | **after each kept change** (loom-optimize, loom-fleet) | consistency; agent tool loop (thinking on, and off when the template toggles it); goldens (cached) + compare on 12 short prompts + 1k/2k documents; vision on 2 images (with a projector); `spec` on 5–6 short requests (drafting packages); sampling (temperature 0 = greedy, seed reproduces) | Qwen3.8-27B (drafting, vision): 3.3 min cached, 5.5 min with the first reference fill; Bolt-E4B: 1.1 / 2.2 min |
| `fast` | (kept for scripts that call it) | goldens + compare (20 short prompts + 1k–32k documents), consistency, vision on all 6 images | 1–3 min |
| `quick` | nightly CI; a new port's first qualification (`qualify.sh`) | fast + `spec` (full request set), agent tool loop (thinking on, and off when the template toggles it), 200-draw sampling, native-context admission probe (seconds) | median 11 min, up to 42 min (audit) |
| `full` | nightly CI; once on a new port's final build before its first card (`finalize.sh`) | quick (the same gates; the admission probe is in both) | as quick |

**Which level when** (from an audit of every gate run in the issue #26–#29 sessions):
- Real Loom bugs caught: consistency 3 (an MoE router race, batch-variant matvec rows, a DFlash hang under
  concurrency), agent 3–4 (stop tokens, a missing `<eos>`, history reuse), compare 2 (both at first bring-up),
  a KV pool < 1.5x the context 1 class (found by the former full-context needle; `build.py` now warns statically and
  the admission probe checks it), spec 0–1. Vision and sampling: 0 (every failure was the reference or the harness). About 85 % of
  compare failures were noise or harness. One `LEVEL=quick` loop of 24+ runs cost 6.6 GPU-hours and found 0 bugs.
- So the development loop runs: kernel tests for new or changed kernels (`run_kernel_tests.sh`), `chat_probe
  --verify` when the chat glue changes (seconds, CPU; it caught the template derivation errors), and
  `LEVEL=dev` after each kept change. `dev` keeps the gates that found bugs at full strength and touches every
  other feature lightly, so a feature that breaks outright still fails in minutes.
- **Where it runs:** the dev machine's GPU is scarce and sequential, DevLab Dispatch is a large elastic pool.
  Locally: only what the inner loop needs. Long or deep checks and anything against published cards: DevLab CI.
  Performance for a card comes only from a DevLab `devlab-perf` job (`loom-fleet` publish.md, "Performance: DevLab is the
  source"); local runs are for development.
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

The native-context admission probe (`ctx_admit.py PKG URL [--tokens N]`, in `gates.sh` at `quick`/`full`, and the
`admit[:N]` step of `devlab_experiment.sh`) catches a KV pool or memory budget below the context in seconds: it sends one
request of about the native context, treats "the server accepts it and starts" as admitted and a 4xx/503 as refused, then
closes the connection (no long prefill, no generation).

- **Speculative-decoding gate `spec`** (packages that draft; `spec_check.py`): the package must produce exactly
  what the same package produces with `draft_depth 0` (`PKG/model_target.json`, same compiled program): greedy
  and seeded-sampling chat, a 1k-token prompt, a tool call, a reasoning answer. Any difference fails (drafting
  is lossless by construction: a difference is a verification, acceptance or state-replay bug). The log also
  prints the acceptance rate. The agent gate runs with drafting on, so it covers multi-turn reuse with speculation.
- **Vision gate** (packages with image input, every level; `dev` sends 2 of them): `vision_golden.py` sends six built-in images
  (shapes + text, chart, document, photo-like, tall, wide; `--images` adds files) to Loom as chat
  requests and to llama-server `--mmproj` (gates.sh sets `MMPROJ` from the package, so the lazily
  started server loads it) as the package's own rendering + image. Prompt token counts must be equal
  (resize rule, token limits, image markers, the image request's chat rendering); replies must match
  token by token up to a near-tie. A reply that fails is judged against two more llama.cpp recordings, the
  re-recording and one with the image encoder on the CPU (`--no-mmproj-offload`; `llama_fill.py` restarts the
  server for it, `bench_common.SERVER_VARIANTS`): the encoder's rounding alone moved the first token of a text
  image by 5 nats (Ornith-1.5-35B-A3B `wide`: 'THE' by 2.4 nats on the GPU encoder, 'The' by 2.8 on the CPU one).
  The two are llama.cpp's own numerics, so following either one passes; a vision bug diverges from both.
  At each of those divergences the near-tie rule also gets llama.cpp's prefill views, as the text gates do
  (`llama_views`): the image and the reference's tokens before the divergence as one prompt, with the default encoder
  and with the CPU encoder (`vision_golden.prefill_view`; a view whose evaluated token count is not prompt + prefix
  does not count). Decoding one token at a time and prefilling the same context differ by whole nats (Ornith-1.5-122B
  `chart`, 'Thu' vs 'Thursday': decode 2.75 nats for 'Thu', GPU-encoder prefill a 0.08 tie, CPU-encoder prefill 2.59 for
  'Thursday'). Requested only after a failure.
  Below llama.cpp's own encoder noise a bug is invisible to this gate. Verified: Bolt-E4B, Gemma-4-E2B, Qwen3.5-0.8B, Muse-Glimmer-30B.
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
  the short prompts and the 1k/2k documents (prefill, KV paging, attention); the 16k golden document covers long
  context; the rest catch a feature that breaks outright. When you change the chat glue (`control.loom`), run
  `chat_probe --verify` first; `dev` then checks rendering, reuse and both thinking modes in the agent gate.
- The native-context admission probe costs seconds (it never prefills the whole prompt): it runs at `quick` and `full`.

```bash
S="${LOOM_SKILL_DIR}/scripts"; P=~/work/mymodel/pkg; T=~/loom/.tools/bin/python; mkdir -p ~/work/mymodel/bench && cd ~/work/mymodel/bench   # outputs land here, not in the skill
$T "$S/golden.py" $P golden.json --llama http://127.0.0.1:8130 --long 1000,2000,16000,32000    # once per model/GGUF
$T "$S/golden.py" $P golden2.json --llama http://127.0.0.1:8130 --long 1000,2000,16000,32000   # 2nd recording: llama.cpp is not run-to-run deterministic
$T "$S/compare.py" $P golden.json --alt golden2.json --url http://127.0.0.1:8160   # -> per-prompt lines + {"verdict": "PASS"|"FAIL", ...}
$T "$S/consistency.py" http://127.0.0.1:8160 --model <model id>         # concurrent == sequential
$T "$S/agent_golden.py" $P http://127.0.0.1:8160 --llama http://127.0.0.1:8130   # tool loop: rendering, reuse, calls, golden per turn
$T "$S/ctx_admit.py" $P http://127.0.0.1:8160                                    # the native context is admitted (then cancelled)
$T "$S/sampling_check.py" $P http://127.0.0.1:8160 --llama http://127.0.0.1:8130 # temperature/top_k/top_p/seed semantics + first-token distribution
```
- **How golden.py and compare.py work, and how to read a divergence** (near-tie rule, PASS/FAIL, causes): `kb.py route --topic benchmark` (loom-kb).
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
    retained history holds tokens no rebuilt prompt has. Nor is a turn that reused nothing (`cached` 0) after a
    reply whose whitespace between calls, or before the stop token, the parsed reply cannot show: the
    retained tokens must equal the template's rendering byte for byte, so a model that wrote `\n\n` where
    the template prints `\n` leaves nothing exact to reuse. Waived only with `--llama`, and only where
    llama.cpp, given Loom's text before that position, puts a whitespace-only token within the near-tie rule of the
    template's token (`whitespace_ties`; a model that is sure of the template's whitespace is no excuse, a
    note and `reuse_waived` record the positions).
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
- **ctx_admit.py** (the native-context admission check, not a gate): sends one streaming request whose prompt is about
  `context_capacity` tokens of real text (`max_tokens` 1; thinking off when the template allows) and watches
  `/v1/status`: admitted when the server answers 200 or shows an active request; refused on a 4xx/503. It then closes the
  connection and waits for the server to go idle (a server that keeps prefilling a cancelled request is flagged). Exit 1 on a
  refusal or no answer; the verdict goes to `context_admit.json` and `gates.json`.
- **test_9070xt.py** (device data, not a gate): on a machine with a 16 GB Radeon RX 9070 XT (the R9700's gfx1201 chip with half the
  memory; no DevLab runner), `$T "$S/test_9070xt.py" PKG --loom-root ROOT --weight-slab-bytes 64mib -o result.json` serves the package
  elastic with the 9070 XT flags (`--memory_bytes=15000000000 --state_slab_bytes=4mib --parameter_mode=preload`; `--prefill-capacity N`
  for a package whose prefill workspace does not fit) through a deployment document, finds the largest `context_capacity` that registers,
  then the largest at which a fresh server completes one request of that full length and answers a sanity prompt correctly after it
  (a wrong answer's text is recorded). The fleet's results: `fleet/devices/gfx1201-16gb.json` (`make_info.py devices fleet/devices` validates).

## 2. Performance

```bash
$T "$S/perf.py" http://127.0.0.1:8160 loom-baseline --model <id> --log perf_log.jsonl
$T "$S/perf.py" http://127.0.0.1:8160 loom-baseline --model <id> --long --log perf_log.jsonl
$T "$S/perf.py" http://127.0.0.1:8130 llama-rocm --kind llama --log perf_log.jsonl     (and --long, with the 1-slot 34k server)
$T "$S/c4_steady.py" http://127.0.0.1:8160 4 --model <id>;  $T "$S/c4_steady.py" http://127.0.0.1:8130 4
```

| Metric | Definition |
|---|---|
| decode_tps | 300-token essay from a short prompt. Client-side: (tokens−1)/(total−TTFT). |
| prefill[] | 1k and 2k-token documents with max_tokens=4. prompt_tokens / TTFT, the median of 5 runs (`--repeats`); single runs varied 1.5–2.9k tok/s on one build. |
| decode_tps_2k / _32k | Decode with that much context in the KV cache. |
| prefill_long[] | 8k / 16k / 32k documents. |
| c4_steady agg_tps | Tokens from all 4 streams inside the window where all 4 are decoding, after the untimed warm-up passes (rates in `warmup_agg_tps`). Use this for "N requests at once". A run in which a stream failed exits 1 and records nothing (`make_info.py` also skips a rate <= 0). |

Rules:
- A card's numbers come from `devlab-perf` jobs: Loom (card programs, active wait: `serve.sh` serves
  `--device=amdgpu` as `amdgpu://?completion_spin_ns=4294967295`, `LOOM_ACTIVE_WAIT=0` opts out for A/B only) and
  llama.cpp on the same runner type (`loom-fleet` publish.md step 3).
- Stored rows are reused while their inputs are unchanged (methodology rule 15): a Loom-only change re-measures Loom only,
  and the card names each side's run.
- Local runs are for development, never card numbers, and a Loom
  number measured without the active wait is not comparable. A job's variant comparisons (`devlab-perf`,
  `devlab-experiment`) compare within that job only.
- Sampling (`reference/methodology.md` rule 5): every metric is the median over sessions (rounds) of each session's
  median sample, identical for Loom and llama.cpp (ROCm, Vulkan). A fast model (decode > ~100 tok/s) gets 6 sessions and
  `perf.py --repeat-rows 3` (decode_2k, decode_32k, prefill_8k three times per session); the others 3 sessions of one
  sample. `devlab_perf.sh` marks each session end in the log (`{"session_end": V}`) and the job's `repeat_rows` passes the
  option; `make_info.py perf` aggregates and records `aggregate: "median"` on the session (older sessions are the mean of N
  rounds and say so on the card).
- Warm up after every restart. `perf.py` first sends its first timed request's own shape, untimed (the 300-token
  essay with another nonce: prefill, decode, drafting and first-use kernels all run once; a trivial "Say hi."
  left DevLab's short-prompt decode 51-60 % below another session's while 2k decode agreed), and `c4_steady.py`
  one short request and then untimed full N-stream passes (same shape as the timed one) until two successive passes
  agree within 10 % (at least 3, at most 6; `--warmup-min`, `--warmup-passes`, `--warmup-tol`; two equal slow passes do not end it early). Without them the median of three timed runs
  reported a cold number: a model that reads weights on demand (Flash-Next's per-layer-embedding table) decoded 4
  streams at 4 tok/s for its first two passes on a DevLab runner with a slow disk (cold page cache, IO pressure 93 %),
  then 65. Every warm-up pass's rate is in the JSON line (`warmup_agg_tps`) and nothing is dropped: a server that stays
  slow after the warm-up reports the slow number, and its slow warm-up passes show why. Every caller (qualify, finalize,
  devlab_perf) gets the same warm-up. When timing by hand:
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

**Benchmark methodology (normative: [`reference/methodology.md`](reference/methodology.md)).** One drafting choice per model, the same on
Loom and llama.cpp ROCm + Vulkan (MTP head in the card's GGUF > exact-model DFlash/assistant drafter > none; never n-gram; from
`data/drafting_classification.json`); each engine loaded once for the 1-stream rows, both undrafted for streams4; win/tie/loss at +-3 %
vs the better backend. `card/matched_spec.py` derives the llama.cpp arguments and checks the rule (`rule_violations`). To rebuild a
package whose drafting differs from its classification: `TUNE_KIND=<choice> TUNE_REQUIRE_DRAFT=1` (or `SPEC=off`). References never draft.

**Run the audit first** (from `bench/`):
```bash
$T "$S/audit.py" .     # -> headline table + checks; exit 1 on FAIL; writes audit.json
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

The generated card sections, rows, publishing and republishing rules: `kb.py route --topic benchmark` (loom-kb).
