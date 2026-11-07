---
name: loom-port-llm
description: Convert an arbitrary LLM (GGUF checkpoint) into a loom-serve package — generate Loom IR kernels + command program from the GGUF's metadata, derive the chat/tool-calling glue from the model's own chat template, build model.json at the model's native context length, bring it up, unit-test kernels, and validate token-by-token against llama.cpp (short prompts, agent tool loops, full context). Includes a config-driven starter kit for dense transformers (Llama/Qwen2/Qwen3/Mistral-style, GQA, QK-norm, sliding window, output gates, post-norms, head_dim 128/256, partial RoPE) and Qwen3.5/Qwen3-Next-style hybrids (gated delta-net layers with recurrent state), with fast decode and WMMA prefill. Speculative decoding is on by default for every model family (the GGUF's MTP block, a Gemma-4 assistant, a DFlash/DFlash2 drafter, or weight-free n-gram prompt lookup, chosen by measured decode speed, output identical to the target). Adds image input automatically when the repo ships a vision projector (mmproj) of a supported family (Gemma 4, Qwen3-VL/Qwen3.5+, Muse-Glimmer), and documents how to add a family. Use for "port model X to Loom", "get X working on loom-serve", custom/unsupported models.
---

# Port an LLM to Loom

The goal is a package directory that `loom-serve --model=PKG/model.json` serves as a
complete chat model:
- greedy output that matches llama.cpp up to near-ties;
- the model's **native context length** (e.g. 40,960 for Qwen3-0.6B, 262,144 for
  Qwen3.5), with over-length prompts refused cleanly;
- **tool calling** in the model's own format, tool results, reasoning in history,
  the thinking on/off switch, and **history reuse** between agent turns.

The starter kit (`starter/`) gets a dense GGUF LLM, a Qwen3.5-style hybrid, or either with a
MoE FFN (Qwen3 / Qwen3.5 / Qwen3.6 MoE: softmax router, shared experts with a sigmoid gate) there
in minutes. Other architectures (MLA, other recurrent layers) need new kernels;
see "Beyond the starter" below.

Prerequisites:
- `loom-setup` is done: `~/loom/.venv` has the SDK (`hrx-loom` 0.1.0 from PyPI, which
  the templates target) and `~/loom/.tools` has gguf, numpy, tokenizers, jinja2 and
  requests.
- The llama.cpp ROCm reference is available (`loom-benchmark`), for the golden
  comparison.

## Fast path: one command (≈1 min for a 0.6–0.8B model)

```bash
LOOM_ROOT=~/loom ${CLAUDE_SKILL_DIR}/port.sh <org>/<repo-GGUF>[:file.gguf] ~/work/mymodel [PORT] [MEMORY_GIB] [MODEL_ID]
#   PORT, MEMORY_GIB, MODEL_ID are positional (defaults 8160, 56, derived): e.g. ... ~/work/m 8260 40
# then, in an end-to-end job, loom-benchmark/scripts/qualify.sh (gates + baselines + roofline);
# or, while fixing the port, the lean gates (a few minutes once the llama.cpp references are cached;
# LEVEL=quick/full run once before the first card via qualify.sh/finalize.sh, then nightly in CI):
LEVEL=dev LLAMACPP_DIR=<llama.cpp ROCm dir> ${CLAUDE_SKILL_DIR}/../loom-benchmark/scripts/gates.sh \
  ~/work/mymodel/pkg http://127.0.0.1:8160 ~/work/mymodel/gguf/<file.gguf> ~/work/mymodel/bench [LLAMA_PORT=8130]
```
`port.sh` runs steps 1–6 below deterministically:
- **starts from a published reference** (stage 1b, `find_reference.py`; `REFERENCE=off` skips it,
  `REFERENCE=loom-fun/REPO` picks one): ranks the loom-fun org's **category A** packages for this GGUF and starts
  from the best one (section "Start from a reference" below);
- picks the GGUF (`starter/pick_gguf.py`): **Q4_K_M** (then UD-Q4_K_XL / Q4_K_S only when the repo has no Q4_K_M; a bigger
  tier only with `QUANT_TIER=Q5|Q6|Q8`), then a file **already in the HF cache** (any revision: nothing is downloaded
  twice), formats the starter decodes (incl. unsloth UD's IQ4_XS/IQ4_NL/Q3_K/IQ3_S and Q4_0), and a file that
  fits twice in GPU memory. It never leaves the tier on its own: with no supported Q4 file it stops and names
  the missing formats. Only an uncached file is downloaded (into the HF cache); it is symlinked as
  `WORKDIR/gguf/`, and the base model's tokenizer is fetched;
- inspects, stops if the chat block isn't verified, builds at the native context, serves (state
  in `WORKDIR/state`), runs the kernel tests and a smoke test (chat + a tool call), and prints
  `PORT_RESULT {...}`;
- starts the end-to-end clock (`WORKDIR/.e2e_start`);
- **image input** (stage 2b): when the repo ships a vision projector (`mmproj-*.gguf`) whose family has
  an encoder plan (`gemma4v`, `qwen3vl_merger`, `muse-glimmer`), adds the image component and tests it
  (stage 5: vision kernels vs NumPy, the whole encoder vs a reference checked against llama.cpp;
  stage 6: one image). `VISION=0` skips it, `MMPROJ=file` picks the projector. An unsupported family
  leaves a text-only port; `reference/vision.md` says how to add one. (A recognized family can still fail
  to build an encoder: Qwen3.8-27B's projector has head_dim 72; `VISION=0` gives a text-only port.)
- **speculative decoding, by default**: an MTP/NextN block in the GGUF, or a DFlash drafter found for the base
  model (stage 2c, `starter/draft/find_drafter.py`; vendor repos and BF16-only drafters included), is built in;
  the smoke stage proves drafting changes no token (`spec_check.py`). `qualify.sh` later keeps whichever drafter
  (n-gram prompt lookup is a candidate for every model) and depth decode fastest, or none
  (`starter/draft/README.md`). `SPEC=off` opts out.

`DRY_RUN=1 port.sh ...` plans a port on the CPU in under a minute: stages 1-2 only, an uncached GGUF read as a
header-only sparse stub (nothing downloaded), then `DRY_RUN_RESULT {...}` (architecture, kit, reference, inspect).

Its exit code says where it stopped (2 download, 3 chat template or uncovered architecture,
4 build/serve, 5 kernel tests, 6 smoke); only then do you need the manual steps below for that
stage. Verified on Qwen3-0.6B (Q8_0/Q4_K_M) and Qwen3.5-0.8B (Q4_K_M).

## Start from a reference (loom-fun, category A only)

Every port and every optimization starts by looking for a published package that already solved most of it
(port.sh does this itself; by hand: `find_reference.py rank GGUF --repo ORG/REPO -o W/reference.json`).
- **Only category A counts**: a card made by the current skill, `info.json` with `"schema": "loom-card/2"` and
  `"report": "generated"`. Imported cards (`"report": "attached"`, old factory runs) are listed as skipped, never used.
- **Ranking:** same architecture and family (GGUF name, base-model chain), then same architecture, then shared
  building blocks (gated delta-net hybrid, MoE routing, vision tower, tool-call format, thinking switch, drafter
  type), then nearest size. The `generator` stamp of its `llm_config.json` says how stale it is.
- **What is taken:** the reference's example kit (when it was built from one: `reference/examples.json` maps the
  packages to `*-example` kits; with no category A package of the architecture, the example kit of that architecture alone); the cfg keys `inspect_gguf.py` left open (geometry, formats and eos ids always stay
  this GGUF's); its chat block when this template's derivation fails, accepted only if `chat_probe.py --verify`
  passes on THIS template; and its measured drafting table, from which `draft/tune.py` starts each method at the
  reference's best depth and climbs from there, and gives a method the reference ruled out (under 75 % of its
  winner) one probe (`TUNE_PRIOR=0` measures all). The llama.cpp baseline's own tuning never takes a prior.
- **Where it pays:** an architecture the starter doesn't cover (the reference's kit is applied automatically; its
  chat block replaces a failed derivation when it verifies). On the starter path (Qwen3.5 sizes) port.sh is already
  automatic: the reference trims the drafting search and pre-answers the cfg review (only geometry differs).
- **What is recorded:** `cfg["reference"]` (repo, revision, score, reasons, what was taken, what differs), which
  `build.py` copies into the package's `llm_config.json`.
- A reference saves time; it never lowers the bar: the derived port runs the same kernel tests and gates.
- `find_reference.py fetch REPO DIR` gets its package (no weights) to read or diff; `show` prints what an
  optimization should start from.

## New architecture (port.sh exit 3 with "no kernels", or a model the starter doesn't cover)

The expensive path. Keep it short:
1. **Get the spec from llama.cpp itself.** `starter/llama_trace.py LLAMACPP_DIR MODEL.gguf "Hello world" --grep '-0$'`
   prints layer 0's compute graph as llama.cpp runs it (op, input names, shapes, values). Use
   `--grep '-N$'` for a layer of each kind (sliding/global, attention/recurrent, dense/MoE). This
   is faster and more exact than reading C++; the GGUF's `arch.*` metadata gives the constants.
2. **Diff it against what the starter does** for a dense layer (Qwen3: `attn_norm → Qcur/Kcur/Vcur
   (+ q/k norm, RoPE) → attention → attn_output → ffn_inp → ffn_norm → gate/up → swiglu → down →
   l_out`). Every extra node or different op is a feature to add; write the list down first.
3. **Copy the starter into the work dir** (`${CLAUDE_SKILL_DIR}/reference/apply_example.sh starter W/kit`,
   which also records the loom-skills commit in `W/kit/KIT_SOURCE`: `build.py` stamps it and the Loom SDK's
   identity into `llm_config.json` `"generator"`, and the model card shows them) and extend
   the copy: a cfg flag per feature in `inspect_gguf.py`, the op in `gen_llm.py` (decode first:
   matvec path, prefill later), and a kernel test case in `make_kernel_tests.py`. If you change
   `chat_probe.py` (a chat format it can't model), `export LOOM_KIT=W/kit`: the benchmark
   scripts then render prompts with your copy, and use its `raw_from_chat(chat, message, finished,
   thinking)` if you define one (how a reply's channels map back to raw text, for a format the
   default doesn't cover). No need to copy the skills tree. Reuse before
   writing: **check `reference/` for the architecture family first** and make the kit with
   `${CLAUDE_SKILL_DIR}/reference/apply_example.sh EXAMPLE W/kit W/cfg.json` (the starter + the example's
   files + its `starter.patch`, and its reviewed config merged into yours: the example's category A loom-fun
   package's `llm_config.json`, pinned in `reference/examples.json`, or the example's own `cfg.json` when none is
   published):

   | Family | Example |
   |---|---|
   | Gemma-3n / Gemma-4 (PLE, KV sharing, head_dim 512, Gemma chat format) | `gemma4` |
   | LFM2 short-conv hybrids (+ its own MoE kernels) | `lfm2moe` (top-k router, expert matvecs, grouped WMMA prefill) |
   | attention sinks, SWA, NoPE (MoE from the starter) | `flint` (built on lfm2moe in ~12 min; its MoE is now `starter/gen_moe.py`) |
   | channel chat formats (ATEM), norm RoPE, NoPE | `glimmer` |

   `../loom-optimize/reference/flint-moe-example/` holds one more MoE idea (bucketed grouped GEMM), not
   a base. The starter's `chat_probe.py` covers hermes, qwen_xml, glm and pythonic (LFM2) tool calls.
4. **Test every path that implements an op.** Attention runs in three kernels (decode split-K + combine,
   prefill `flash2`, the listed fallback for mixed tiles); a feature added to one (sinks, softcap, a gate,
   a window) must be added and unit-tested in all of them. A mismatch passes the goldens and shows up only
   when a turn reuses history the model decoded itself (flint: sinks differed between flash2 and split-K).
5. **Check numerically, node by node.** A short numpy forward of layers 0–1 on the trace's tokens
   (gguf `dequantize` for weights) compared with the trace's sums and edge values names the first
   misunderstood op in minutes. Then the Loom kernels against numpy (kernel tests), `chat_probe --verify`
   for the chat glue, then the gates (`LEVEL=dev` while iterating).
6. **Resume the scripted path** once the kit builds: `FROM=build KIT=W/kit port.sh ORG/REPO W PORT
   MEMORY_GIB MODEL_ID` (all three positionals) reruns build, serve, kernel tests and the smoke test from your `cfg.json`
   (no need to replay those stages by hand), then go on with `qualify.sh`.
7. **Prefill can come second.** A decode-only port (prefill through the matvec path, chunk 1) is
   correct but slow: get the gates passing first, then add GEMM prefill as the first optimization.
8. **The chat block** for templates `chat_probe.py` can't model (channels, role tags without a
   suffix, tools inside a message): see the checklist below and
   `reference/glimmer-example` (a channel format end to end: `chat/atem.loom` + its `chat_probe.py`).
9. **Work in parallel when you can spawn agents.** The kernels (steps 1–6) and the chat block
   (step 8: `chat_probe.py` + `control.loom`, verified against the Jinja template) are
   independent until `build.py`. Give the chat work to a subagent on its own copy of the kit and
   its own port; merge when both pass. Without subagents (other harnesses, e.g. Pi under weaver), do the
   kernels first and the chat block second.

## Workflow, step by step (what port.sh does; use it to debug a stage)

```bash
K=${CLAUDE_SKILL_DIR}/starter; T=~/loom/.tools/bin/python
W=~/work/mymodel && mkdir -p $W && cd $W
# 1. Inspect the GGUF: geometry, tensor formats, eos ids, native context, hybrid layers, and the
#    chat block DERIVED from the GGUF's Jinja chat template (chat_probe.py) and VERIFIED against it:
#    it prints "chat OK <conversation>" per probe (chat, multi-turn, tools, system+tools, agent loop,
#    parallel calls, with thinking on/off). Every line must be OK; a FAIL or "not derivable" means
#    fix cfg.json["chat"] by hand (fields: chat_probe.py) and rerun `chat_probe.py MODEL.gguf`.
$T -I $K/inspect_gguf.py /path/model-Q4_K_M.gguf cfg.json
# 2. REVIEW cfg.json against llama.cpp's src/models/<arch>.cpp - see the checklist below
# 3. Tokenizer: the HF tokenizer.json of the *same* model (base repo; GGUF repos often lack it)
~/loom/.venv/bin/hf download <org>/<base-model> tokenizer.json --local-dir tok
# 4. Build: context defaults to the model's native context, the KV pool to context + max(64k, context/4), both fitted
#    to --memory-gib (default auto: that pool, capped at 60% of RAM; one model is loaded at a time). Prints the model id and --memory_bytes.
$T $K/build.py cfg.json pkg --tokenizer tok/tokenizer.json --model-id <org>/<name-without-spaces>
# 5. Serve + smoke test
VENV=~/loom/.venv ${CLAUDE_SKILL_DIR}/../loom-setup/scripts/serve.sh mine 8160 amdgpu <memory_bytes> pkg/model.json
curl -s localhost:8160/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<model id>","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":400}'
#    thinking models spend 100-400 tokens reasoning: check finish_reason and that content is non-empty;
#    "chat_template_kwargs": {"enable_thinking": false} (or "reasoning_effort": "none") turns thinking off
# 6. Kernel unit tests vs numpy dequant references (embed, matvecs, RoPE, the whole attention chain,
#    the output head, and for hybrids the gated delta-net chain). Run them even if the output looks fine.
$T -I $K/make_kernel_tests.py cfg.json ktest && VENV=~/loom/.venv $K/run_kernel_tests.sh pkg ktest
# 7. Gates vs llama.cpp - loom-benchmark skill: golden.py/compare.py (short prompts + 1k-32k documents),
#    agent_golden.py (multi-turn tool calling, rendering, history reuse) and needle.py (full context).
```
After every rebuild, restart the server with `stop.sh` and then `serve.sh`; the
server does not reload the package (`serve.sh` stops a server whose registration failed).

**A port is done when** the kernel tests pass, the golden gate shows only near-tie
divergences, `agent_golden.py` passes (exact rendering, reuse inside tool loops,
valid calls), and `needle.py` passes up to the context capacity with a clean 400 above it.

## Speculative decoding (automatic)

Nothing to configure; read `starter/draft/README.md` only to debug it.
- Kinds: `mtp` (the GGUF's NextN block; Qwen3.5/3.6/3.8, unsloth `*-MTP-GGUF` repos), `dflash` (DFlash and
  DFlash2 drafter GGUFs from z-lab or the model's vendor, a separate `draft` parameter scope), `ngram` (prompt
  lookup in the row's own history, any model, no weights), `assist` (a Gemma-4 assistant GGUF for gemma4-family
  targets: Gemma 4's MTP head, reading the target's K/V). `build.py --draft auto|mtp|dflash|assist|ngram|none`
  `[--drafter GGUF] [--draft-depth N]`; auto takes `cfg["drafter"]` (tune.py's measured choice), else MTP, else DFlash.
- Across families: the verifier is the trunk itself; DFlash embeds its block the way llama.cpp does (the target's
  raw row times the drafter's `embedding_scale`), so Gemma-style scaled or Glimmer-style normed embeddings work.
  A kit with a recurrent layer other than gated delta-net (LFM2 short-conv) registers a state replay
  (`starter/draft/README.md` "Recurrent layers"); without one the `spec` gate fails on the first rejected draft.
- Output is exactly the target's: verification, device acceptance and gated delta-net replay commit only target
  tokens, and seeded sampling draws at the true position. The `spec` gate compares with the same package at
  `draft_depth 0` (`pkg/model_target.json`); for a single request (up to 16 verified tokens: the int8 matvecs'
  1-/4-/8-/16-token variants round like 1-token decode) a difference is a bug, not a near-tie. Wider verification
  (many drafting streams, 17+ tokens) takes the f16 WMMA GEMM for speed and can flip near-ties (`SPEC_EXACT=1`
  builds trade that speed for decode's rounding); tune.py runs the equivalence check before it keeps a drafter and
  depth, so a depth that changes a request's output is never chosen.
- Drafting works to the full context and with history reuse and checkpoints: the draft K/V shares the pool's pages.
- Depth is a `model.json` setting (`execution.draft_depth`, 1-7); the method needs a rebuild.
- Small models (< ~2B) decode faster without it (launch-bound); `tune.py` measures and decides.

## What the starter generates

| File | Role |
|---|---|
| `inspect_gguf.py` | GGUF metadata → `cfg.json`: hidden, layers, heads/kv_heads, head_dim, ffn, vocab, **context_length**, rms_eps, rope_theta/type, **rope_dims** (partial RoPE), qk_norm, attn_gate (`true`: separate tensor; `"fused_q"`: packed in attn_q), post_norms, sliding window/layers, NoPE layers, tied embeddings, eos ids, **recurrent_layers / gdn** (hybrids), the chat block (via `chat_probe.py`), and every tensor's format and shape. |
| `chat_probe.py` | Renders the model's Jinja chat template on probe conversations and cuts out what the control program needs: role prefixes/suffix, BOS, default system prompt, generation prompts with thinking on/off, the tools block and its order relative to the system message, the tool-call format (`hermes` JSON, `qwen_xml`, `glm`, `pythonic`) and separators, tool-response framing, reasoning-in-history policy, trimming. A template that ignores `tools` (plain ChatML chat models) gives a chat-only block (`call_format` `none`): a request's tools are ignored as the template does, no call is parsed, the card says tool calling "no" and gates.sh skips the agent gates (not applicable); probes with tool calls or results print SKIP. `verify()` re-renders a dozen conversations exactly as `control.loom` will (`simulate()`) and diffs them against the template. `simulate()` is also what the benchmark scripts use to build golden prompts. |
| `gen_llm.py` | `cfg.json` → `llm.loom`: all kernels plus the `@llm_advance` command program. The command program is a **static per-layer unroll**, so each layer's kind (attention or gated delta-net) and tensor formats come from the GGUF (Q4_K_M mixes Q4_K/Q5_K/Q6_K/Q8_0 per tensor). Decode-shaped linears (< 64 tokens) and the head use int8 activations (`g_quant_q8` + `_q8` matvecs with integer dot4, llama.cpp's Q8_1 numerics; 1-, 4-, 8- and 16-token variants that round alike) for Q8_0/Q4_K/Q5_K/Q6_K weights; `ACT_Q8=0` keeps f32 activations. |
| `gen_moe.py` | MoE FFN kernels (router + top-k, expert matvecs, shared-expert gate, combine; bucketing + grouped WMMA GEMMs for prefill), for `cfg["moe"]` models. |
| `gen_gdn.py` | Gated delta-net kernels (conv1d + SiLU, history commit, q/k L2 norms, register recurrence, gated RMSNorm) and the fused q/gate split, for hybrids. Recurrent state lives in the runtime's recurrent slots (see porting-notes). |
| `gen_gemm.py` | WMMA prefill GEMMs per (format, mode). Formats: q4_k, q5_k, q6_k, q8_0. Modes: store, acc, swiglu. |
| `gen_flash2.py` | Prefill flash attention (transposed, K/V staged once per KV head), head_dim 128 (and 256 with `FLASH2_SPLIT=0`). |
| `gen_flash2s.py` | Prefill flash attention for head_dim 256 (used automatically): two waves per query head own 128 dims each, so Q stays in registers for the whole key loop; the pair sums S partials through workgroup memory. |
| `tile_uniform.loom` | Helper that classifies 16-token tiles for flash2. |
| `templates/*.loom` | Serving glue for the hrx-loom 0.1.0 text ABI (`text_*_v1` exports). `prepare` stages one program per packing shape. `storage` sets the KV geometry (2 planes per attention layer, F16) and the recurrent slot size. `control` is the chat policy: rendering, tools, reasoning, thinking switch, tokenization, output channels, tool-call parsing, canonical history for reuse. `service` maps the ABI exports. `weights.loom` is the parameter policy. |
| `build.py` | Fills the templates from `cfg.json["chat"]`, sizes context/pool/memory, then writes `model.json`, `tokenizer.json`, the GGUF symlink and `llm_config.json` (read by the benchmark scripts). |
| `pick_gguf.py` | Chooses the GGUF in a repo from the headers only (HTTP range read): quant preference, tensor formats the kit supports, and the memory rule (the file must fit twice: a conservative margin, kept until quant choice accounts for the KV pool at full context). |
| `llama_trace.py` | llama.cpp's intermediate tensors and compute graph for a prompt (`llama-eval-callback`): the spec and the node-by-node reference for a new architecture. |
| `draft/` | Speculative decoding: `gen_draft.py` (verify stage, device acceptance, GDN capture/replay and the generic replay hook, MTP drafter), `gen_dflash.py` (DFlash/DFlash2 drafters), `gen_assist.py` (Gemma-4 assistant), `gen_ngram.py` (prompt lookup), `build_draft.py` (prepare/storage/control glue), `find_drafter.py`, `tune.py`; see its README. |
| `make_kernel_tests.py`, `run_kernel_tests.sh` | `check.case` unit tests with numpy references (gguf dequantize) on the real tensor shapes: embed, matvecs (every format/K in use), QK-norm + RoPE, the prefill GEMMs (every format/mode in use, 64-, 128- and 256-row kernels), the f16 row scale, the attention chain (q/gate split → RoPE + paged KV store → split-K attention → gated combine), the output head argmax, and the gated delta-net chain on a span that continues a stored state. |

Kernel inventory:
- **Decode:**
  - one-lane-per-32-value-group matvecs, 1-token and 4-token variants, with fused SwiGLU;
  - an argmax output head that writes no logits to memory;
  - split-K decode attention with up to 64 splits and early exit.
- **Prefill:**
  - f16 row-scaled activations feeding WMMA GEMMs;
  - transposed flash attention;
  - a split-K fallback for tokens in mixed (multi-sequence) tiles.
- **Shared:** a per-token QK-norm + RoPE + F16 KV store kernel, and fused add + RMSNorm.

**Fixed assumptions:**
- head_dim 128 or 256 (attention); gated delta-net head dims multiples of 32; query-head groups that are not a
  power of two (Qwen3.8: 6) use flash prefill with staging sized for the power of two below (`gen_flash2.py`);
  `NO_FLASH2=1` forces the listed split-K path for every prefill token (diagnostics; quadratic at long prompts);
- K-quant / Q8_0 / Q4_0 / IQ4_NL / IQ4_XS / Q3_K / IQ3_S weights (`starter/quant_values.loom`; add dequant helpers
  for other formats); f32 norms and conv weights;
- hidden ≤ 8192; context ≤ 262,144 (kernel bounds; raise the `range(..., 262144)` facts for 1M-context models);
- **sampling**: `temperature`, `top_p`, `top_k` and `seed` per request. Temperature 0 (the default
  unless `build.py --temperature/--top-p/--top-k` set other defaults) is exact greedy. Otherwise the
  head draws from the top-k candidates (k ≤ 64; `top_k` 0 = 64) after temperature softmax and top-p,
  with a counter-based RNG keyed by (seed, position): a fixed `seed` reproduces a completion, and a
  request without one draws from its own nonce. Thinking models want sampling (Qwen: temperature
  0.6, top_p 0.95, top_k 20 for thinking; 0.7/0.8/20 without); greedy can end reasoning with EOS or
  loop, which `agent_golden.py` flags. Gates and goldens always send temperature 0.
- text only (image content parts are rejected; text parts are joined).

**Chat policy facts (control.loom):**
- Tools follow the template exactly (verified by `chat_probe.py`). Tool definitions are printed as
  the request sent them; send them serialized like Jinja `tojson` (`json.dumps` defaults) for
  byte parity with the template.
- Hermes-JSON calls are parsed in source without a JSON parser, so a malformed call never fails
  the request: its arguments go back as generated for the harness to handle. Qwen-XML and GLM
  calls go through the host `tools.parse_xml` codec (a malformed XML frame fails the request; a
  frame cut off by `max_tokens` is dropped). Text after the last complete call is dropped.
  Content before a call ends without `content_call_separator`: the template prints it between content and
  calls, so returning it doubled it in the next turn's rendering and broke history reuse (`@content_until`;
  streaming withholds trailing whitespace until it is known not to precede a call).
- Reasoning is rendered for assistant turns after the last user message, as the Qwen templates do.
  A new user turn therefore re-renders earlier turns without reasoning; inside a tool loop the previous
  turn is reused entirely (`prompt_tokens_details.cached_tokens`). loom-serve resumes a retained row only
  when the new prompt extends its history (`complete_text`) exactly: it cannot cut a row back to a common
  prefix, so a turn the template re-renders differently (dropped reasoning, an added newline) is prefilled
  again (the prefix cache only keeps intermediate frontiers of long prefills).
- loom-serve rejects `"reasoning_content": null` in a message: omit the field instead.
- `usage.prompt_tokens` of a turn that reuses history is one lower than `/v1/chat/completions/input_tokens`
  (the previous turn's EOS is already in the row); compare renderings with `input_tokens`.

## cfg.json review checklist (step 2)

Compare with llama.cpp's graph builder for the arch (`src/models/<arch>.cpp` in
recent llama.cpp; older trees have it in `src/llama-model.cpp`). Check these fields:
- `rope_type`:
  - `neox` pairs (d, d+64). Used by qwen2/qwen3/gemma/phi3/…; llama.cpp's
    `LLAMA_ROPE_TYPE_NEOX`.
  - `norm` pairs (2d, 2d+1). Used by llama/mistral; `LLAMA_ROPE_TYPE_NORM`.
  - A wrong value still gives plausible-looking text for a few tokens, then degrades.
- `rope_theta`; rope scaling (YaRN/linear) is **not** implemented, so add it in
  `qk_rope()` if the model uses it.
- `qk_norm` (per-head RMSNorm on q/k before RoPE), `post_norms` (Gemma-style
  post-attention/FFN norms), `attn_gate` (sigmoid output gate), `embed_scale`
  (Gemma multiplies embeddings by √hidden), `embed_norm`.
- Sliding-window layers: list them in `sliding_layers` and set `sliding_window`.
  `nope_layers` are layers without RoPE.
- `extra_eos_ids`: every id that should stop generation. For ChatML models that is
  `<|im_end|>` and `<|endoftext|>`.
- `chat`: derived and verified automatically (all `chat OK` lines). It matches the
  template byte for byte with **one deliberate exception**: when the model emits
  `<think>` itself (Qwen3 hybrid), `<think>\n` is appended to the thinking generation
  prompt so output starts in the reasoning channel. llama.cpp's *chat* endpoint then
  sees fewer prompt tokens than Loom; the goldens render with Loom's prompt, so they
  are exact. Check the printed summary against the model card:
  - `thinking`: `toggle` (Qwen3, Qwen3.5: `enable_thinking`), `always`, or `never`
    (`*-Instruct-2507`); `thinking_default` (Qwen3: on; Qwen3.5 small models: **off**);
  - `call_format`: `hermes` (Qwen2.5/Qwen3), `qwen_xml` (Qwen3.5/Qwen3-Coder), `glm`, `pythonic`
    (LFM2: `<|tool_call_start|>[f(a='x', n=3)]<|tool_call_end|>`, parsed in control.loom);
  - `default_system` (Qwen2.5 inserts "You are Qwen, created by Alibaba Cloud…").
  - `reasoning_instructions` (Qwen3.8: "Reasoning effort is set to xhigh. …" per `reasoning_effort`, only
    with thinking on, `reasoning_separator` before the tools text / system message): build.py advertises
    those efforts (plus none = thinking off) and `chat_begin` renders the request effort's text.

  **History reuse depends on the canonical rendering.** `complete_text` must store the turn as the
  template re-renders it, not as the model generated it. Two trials failed `reuse_ok` here: a model
  that omits `<|eom|>` before a tool call (the template renders it), and one that writes newlines
  after a call that the template drops. If `reuse_ok` fails after a tool call, diff Loom's
  `input_tokens` rendering of the next turn against `chat_probe.simulate()` and canonicalize those bytes.
  GLM-style templates: the model ends a call turn with `<|observation|>` and an answer with `<|user|>`;
  those stop tokens stay in the KV row, so set `chat.stop_after_calls` / `chat.stop_after_answer`:
  `complete_text` publishes them, and `g_embed` maps the service terminal loom-serve feeds back at the
  next turn to the real stop token (`stop_restore()`; flint reused turns diverged at gaps up to 2.7 without it).
  Templates the probe can't model (e.g. role tags with no closing suffix, tools
  inside the first user message, channel formats): fill the block by hand, keep
  `simulate()` in step, and run `chat_probe.py MODEL.gguf --verify cfg.json` (your kit's copy) until every line is OK. If the
  control program must parse structural tokens out of the generated text (e.g.
  channel tags), list them in `chat.nonspecial_tokens`: loom-serve strips special
  tokens from decoded text.
- Hybrids (`recurrent_layers` present): check `gdn` against llama.cpp's
  `src/models/qwen35.cpp` (k/v heads, head dims, conv kernel), that `kv_layers`
  counts only the attention layers, `rope_dims` (Qwen3.5: 64 of 256), `attn_gate:
  "fused_q"`, and `ffn_norm_tensor: "post_attention_norm"`.

## Bring-up debugging order

1. **The server fails to register.** `serve.sh` prints diagnostics. See
   `reference/loom-ir-cookbook.md` for each diagnostic family (PARSE, TYPE,
   SUBRANGE, TARGET, SYMBOL).
2. **Garbage from the first token.** Run the kernel unit tests. They found the
   one-wave matvec writer bug (K=1024 gives 32 lanes for 64 outputs) in seconds.
   Then suspect, in order:
   - rope pairing;
   - the norm eps;
   - a tied head that is missing `output.weight` (the head must use `token_embd.weight`);
   - a tensor format with no dequant helper.
3. **Correct short answers but wrong long ones.** The prefill path (≥64 tokens:
   GEMM + flash2 + fallback) is wrong, or KV paging is. Compare 1k/2k goldens
   against short goldens, and run `consistency.py` for mixed batches.
4. **Correct but stops early or never stops.** Check the EOS ids and the chat
   `terminal`.
5. **503 "insufficient completion capacity".** `--memory_bytes` is too small for
   the KV pool (pool × 2·kv_layers × kv_heads·head_dim·2 bytes) plus recurrent state.
6. **Registration fails with a VGPR spill (`spill-traffic-register-exhausted`).** A
   kernel holds too much in registers for this model's shape. Split the work over more waves
   (`gen_flash2s.py`: two waves per head at head_dim 256) or reload operands per iteration.
7. **Coherent but wrong from the first token, kernel tests pass.** A wiring or
   config bug. Compare against `starter/llama_trace.py` (llama.cpp's intermediate tensors,
   node by node) with a numpy forward of layers 0–1; then read the generated command program for one layer of each kind
   (`awk '/---- layer N$/,/---- layer N+1/' pkg/program/llm.loom`). A real example:
   a stale cfg.json with `qk_norm: false` left Qwen3.5's Q/K norms out (`%no_norm`).
   An independent numpy forward of a short prompt that matches llama.cpp's top
   logits narrows "my understanding of the model" vs "my kernels" quickly; it is a
   diagnostic, never the golden.

## Beyond the starter (MoE, other formats, new ops)

- **MoE** is in the starter (`gen_moe.py`, wired by `gen_llm.py` from `cfg["moe"]`): router top-k (softmax or
  sigmoid + selection bias, weights kept in registers), per-assignment expert matvecs for decode, tokens bucketed
  by expert and grouped WMMA GEMMs for prefill, shared experts (with the Qwen sigmoid gate), combine; MoE layers
  after `leading_dense` dense ones (an MoE MTP block generates too; not yet validated on the GPU). `inspect_gguf.py` fills `cfg["moe"]` from the GGUF;
  routing choices llama.cpp hard-codes per architecture instead of writing them (`expert_gating_func`,
  `expert_weights_norm`) come from `MOE_DEFAULTS`: add a row for a new MoE architecture from its
  `src/models/<arch>.cpp` (`build_moe_ffn` arguments). Expert formats: Q4_K/Q5_K/Q6_K/Q8_0 (`gen_moe.FMT`; an MTP
  block in another format is left out). Kernel tests cover the decode and prefill chains on the real tensors.
  Ported this way: Qwen3.6-35B-A3B (`qwen35moe`, hybrid + MoE, 256 experts top-8).
- **Gemma-3n / Gemma-4 family (arch `gemma3n`, `gemma4`).** `apply_example.sh gemma4`: a complete,
  gate-passing port whose README lists every feature the starter lacks (PLE, KV sharing, mixed
  head_dim 256/512, proportional partial RoPE, V norm without weights, GEGLU, softcap, bf16); each
  one silently breaks output if missed. Read llama.cpp's `src/models/gemma*.cpp` for a new variant.
  Sliding layers with a small window (<= ~1k keys) don't need flash prefill: split-K is fine there
  (a trial spent 15 min on it for no gain).
- **New quant format.** Add a `@g_<fmt>_values` func in `COMMON` (gen_llm.py; `mv()` calls it),
  add the entries to `FMT`/`SB` in `starter/formats.py` (inspect_gguf.py's "no kernels" check and pick_gguf.py read
  the same table), and add a `dequant16` case in `gen_gemm.py`. bf16 needs a
  value helper (`view<32xbf16>` -> `vector.extf`). Unit-test against `gguf.quants.dequantize`.
- **Other recurrent layers (Mamba, RWKV, other delta-net variants).** Follow
  `gen_gdn.py`: per-span kernels read the reader slot at the fork position (else the
  row's writer slot), start from zeros at position 0, and publish the final state to
  the writer slot; `build.py` sizes the slot (`gdn_geometry()["slot_bytes"]`). Test the
  chain like `make_kernel_tests.py`'s `gdn_chain` (a span that continues a stored state).
- **Library programs are the canonical reference** for the text-service ABI (prepare,
  control, storage, service entry points). `templates/` follow the upstream tiny-text
  fixture (`loom/src/loom/serve/fixture/text` at tag v0.1.0 of ROCm/hrx-system) and
  `library/program/text/qwen3_8/chat.loom`. The upstream doc is
  `loom/docs/src/serving/model-programs.md`. When a new hrx-loom release changes the
  ABI, diff that fixture between the old and new tags; the registration error names
  the missing export (e.g. `root symbol '@text_chat_reasoning_v1' was not found`).

## Reference

- `reference/loom-ir-cookbook.md`: IR idioms, proof facts, and compile-error fixes.
- `reference/vision.md`: image input: what port.sh adds, how it works (image component, chat markers,
  media input records, M-RoPE), and **how to add a projector family** (llama.cpp graph -> NumPy
  reference vs `llama-mtmd-debug` -> kernels -> plan -> gate). Code: `starter/vision/`.
- `reference/porting-notes.md`: how the text service drives the program (epochs,
  spans, metadata packets, KV paging), and what each glue file does.
- `reference/*-example/`: worked ports (table in step 3), applied with `apply_example.sh`; `examples.json` names
  each one's category A loom-fun package (its reviewed config). `find_reference.py`: the reference lookup.
  Maintainers: after changing the starter, `reference/rebase_examples.sh OLD_STARTER` rebuilds the
  examples' patches and `check_examples.sh` must pass.
