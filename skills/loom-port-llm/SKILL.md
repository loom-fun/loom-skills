---
name: loom-port-llm
description: 'Port GGUF LLMs to AMD Loom and loom-serve: inspect metadata, generate and test kernels, derive chat/tool/reasoning glue, build at native context, and validate against llama.cpp. Covers dense and MoE transformers, gated delta-net hybrids, speculative decoding and supported vision projectors, with a playbook for new architectures. Use for "port model X to Loom", "get X working on loom-serve", and custom or unsupported models.'
---

# Port an LLM to Loom

**Bundled paths ? first shell action.** Before running commands, set `LOOM_SKILL_DIR` in the shell to the
absolute directory containing **this** `SKILL.md`, using its path from the skill listing or read tool.
In Claude Code, run `export LOOM_SKILL_DIR="${CLAUDE_SKILL_DIR}"` (the host substitutes the path).
In Codex, Pi or another host, export `LOOM_SKILL_DIR` to the loaded file's directory. Use a
shell-compatible absolute path (Git Bash on Windows). Rebind before using another skill's commands
and in every new shell; it is not the working directory or SDK root. On Windows, use native `C:/...`
paths for file tools; `/c/...` paths belong to Git Bash. Never run an unresolved placeholder. Run bundled commands by quoted absolute path; do not run `scripts/...` from the working directory
or change into the installed skill tree. Install the complete Loom suite: sibling paths are required.

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
LOOM_ROOT=~/loom "${LOOM_SKILL_DIR}/port.sh" <org>/<repo-GGUF>[:file.gguf] ~/work/mymodel [PORT] [MEMORY_GIB] [MODEL_ID]
#   PORT, MEMORY_GIB, MODEL_ID are positional (defaults 8160, 56, derived): e.g. ... ~/work/m 8260 40
# then, in an end-to-end job, loom-benchmark/scripts/qualify.sh (gates + baselines + roofline);
# or, while fixing the port, the lean gates (a few minutes once the llama.cpp references are cached;
# LEVEL=quick/full run once before the first card via qualify.sh/finalize.sh, then nightly in CI):
LEVEL=dev LLAMACPP_DIR=<llama.cpp ROCm dir> "${LOOM_SKILL_DIR}/../loom-benchmark/scripts/gates.sh" \
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
  an encoder plan (`gemma4v`, `qwen3vl_merger`, `qwen2.5vl_merger`, `muse-glimmer`), adds the image component and tests it
  (stage 5: vision kernels vs NumPy, the whole encoder vs a reference checked against llama.cpp;
  stage 6: one image). `VISION=0` skips it, `MMPROJ=file` picks the projector. An unsupported family
  leaves a text-only port; the loom-kb vision node (`kb.py route --facets vision`) says how to add one. (A recognized family can still fail
  to build an encoder: Qwen3.8-27B's projector has head_dim 72; `VISION=0` gives a text-only port.)
- **speculative decoding, by default**: an MTP/NextN block in the GGUF, or a DFlash drafter found for the base
  model (stage 2c, `starter/draft/find_drafter.py`; vendor repos and BF16-only drafters included), is built in;
  the smoke stage proves drafting changes no token (`spec_check.py`). `qualify.sh` later keeps whichever drafter
  (n-gram is opt-in, never on a card) and depth decode fastest, or none
  (the loom-kb `speculative` topic (`kb.py route --topic speculative`)). `SPEC=off` opts out.

**Split GGUFs** (`NAME-00001-of-0000N.gguf`): pass the first part; port.sh fetches and links every part; inspect, build
and the kernel tests read them all, and the package lists each as a parameter file. A model whose weights need more
than 60 % of RAM: `LOOM_MEMORY_CAP_GIB=N` sets the memory cap (build.py, pick_gguf.py) (a DevLab port job: `"env": {"LOOM_MEMORY_CAP_GIB": N}`);
a load above the cap must also pass `loom-setup/scripts/memcheck.sh` (MemAvailable ≥ budget + 16 GiB) when it is served.
Programs hand-written in the hrx-loom catalog: "Catalog packages" below.

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

## Catalog packages (hand-written programs: `CATALOG=`)

When ROCm/hrx-loom-library already has a package for an architecture the starter cannot generate (qwen4exp:
hyper-connections, per-layer-embedding hash table, sparse indexed attention), port it from that package instead of
writing a kit. `reference/qwen4exp-example` is one: `catalog/catalog.json` + the (adapted) package files.
```bash
CATALOG=qwen4exp "${LOOM_SKILL_DIR}/port.sh" catalog ~/work/fn 8160 auto <MODEL_ID>   # or CATALOG=DIR (catalog.json + the package)
export LOOM_ROOT=$(cat ~/work/fn/loom_root)   # the pinned SDK's root, for qualify.sh / gates.sh / finalize.sh
```
Same stages and outputs as any port (`pkg/` with `model.json` cut to one variant and preset, `llm_config.json`, `cfg.json`,
`model_target.json` for the draft-off twin), so qualify, gates, perf, card, `publish_package.py prepare` and the
nightly list work unchanged. What differs (`starter/catalog.py`, `catalog.json`'s keys are documented there):
- **Weights** (repo, revision, split parts, the MTP head) and the tokenizer repo come from `catalog.json`; the package keeps
  the catalog's `@origin` names, which `prepare` preserves (it only pins the revisions).
- **SDK pin**: `catalog.json` `sdk` (hrx-loom wheel) goes to `llm_config.json`; `port.sh`, `port_e2e.sh`, the nightly and
  DevLab scripts install it with `loom-setup/scripts/ensure_sdk.sh` next to the default root (loom-setup "SDK pins").
- **Memory**: the budget is the package's *resident* set (`memory.resident_weights_bytes`, KV, recurrent rows, workspace),
  not the GGUF size; above the 60 % default it needs `memory.over_cap` (explicit, with a note) and `memcheck.sh` headroom.
  llama.cpp keeps `per_layer_token_embd` on the CPU by itself (`llamacpp_server.sh` forces nothing onto the GPU).
- **Kernel tests are the package's own** `loom_test` targets, run by `skills/loom-benchmark/scripts/kernel_tests.py` (no Bazel;
  `devlab_experiment` runs them on the GPU for catalog kits). Drafting is the MTP head or none (`tune.py` measures both); no
  reference lookup or DFlash/n-gram drafter. Vision comes from the package's vision variants (`VISION=0` keeps text only).
- **Rebuilding variants** (`devlab-perf` `build:`/`ref:` variants, `devlab-experiment`) cannot regenerate programs: use the `card` variant.
- `inspect_gguf.py` prints starter-coverage warnings for qwen4exp (arch unknown, `q5_1`, head_dim); a NOTE says they do not
  apply. Use it when the first thing needed is a package, not a kernel set; a new kit (below) only when none exists.

## New architecture (port.sh exit 3 with "no kernels", or a model the starter doesn't cover)

The expensive path. Keep it short:
1. **Get the spec from llama.cpp itself.** `starter/llama_trace.py LLAMACPP_DIR MODEL.gguf "Hello world" --grep '-0$'`
   prints layer 0's compute graph as llama.cpp runs it (op, input names, shapes, values). Use
   `--grep '-N$'` for a layer of each kind (sliding/global, attention/recurrent, dense/MoE). This
   is faster and more exact than reading C++; the GGUF's `arch.*` metadata gives the constants.
2. **Diff it against what the starter does** for a dense layer (Qwen3: `attn_norm → Qcur/Kcur/Vcur
   (+ q/k norm, RoPE) → attention → attn_output → ffn_inp → ffn_norm → gate/up → swiglu → down →
   l_out`). Every extra node or different op is a feature to add; write the list down first.
3. **Copy the starter into the work dir** (`"${LOOM_SKILL_DIR}/reference/apply_example.sh" starter W/kit`,
   which also records the loom-skills commit in `W/kit/KIT_SOURCE`: `build.py` stamps it and the Loom SDK's
   identity into `llm_config.json` `"generator"`, and the model card shows them) and extend
   the copy: a cfg flag per feature in `inspect_gguf.py`, the op in `gen_llm.py` (decode first:
   matvec path, prefill later), and a kernel test case in `make_kernel_tests.py`. If you change
   `chat_probe.py` (a chat format it can't model), `export LOOM_KIT=W/kit`: the benchmark
   scripts then render prompts with your copy, and use its `raw_from_chat(chat, message, finished,
   thinking)` if you define one (how a reply's channels map back to raw text, for a format the
   default doesn't cover). No need to copy the skills tree. Reuse before
   writing: **check `reference/` for the architecture family first** and make the kit with
   `"${LOOM_SKILL_DIR}/reference/apply_example.sh" EXAMPLE W/kit W/cfg.json` (the starter + the example's
   files + its `starter.patch`, and its reviewed config merged into yours: the example's category A loom-fun
   package's `llm_config.json`, pinned in `reference/examples.json`, or the example's own `cfg.json` when none is
   published):

   | Family | Example |
   |---|---|
   | Gemma-3n / Gemma-4 (PLE, KV sharing, head_dim 512, Gemma chat format) | `gemma4` |
   | LFM2 / LFM3 short-conv hybrids (+ its own MoE kernels) | `lfm2moe` (top-k router, expert matvecs, grouped WMMA prefill; lfm3: kernel-7 conv, sandwich norms; its per-head gate and SWA rope base are the starter's `per_head` and `rope_theta_swa`) |
   | NoPE pattern, GLM-style stop tokens (sinks, SWA and MoE from the starter) | `flint` (built on lfm2moe in ~12 min; its MoE and sinks are now the starter's) |
   | OpenAI harmony channel format, reasoning effort (sinks, SWA, YaRN, biases, MXFP4 MoE from the starter) | `gpt-oss` |
   | channel chat formats (ATEM), norm RoPE, NoPE | `glimmer` |

   `../loom-optimize/reference/flint-moe-example/` holds one more MoE idea (bucketed grouped GEMM), not
   a base. The starter's `chat_probe.py` covers hermes, qwen_xml, glm and pythonic (LFM2) tool calls, and tool
   declarations printed as JSON or as XML elements (Qwen3-Coder: `chat.tools_format` `"xml"`).
4. **Test every path that implements an op.** Attention runs in three kernels (decode split-K + combine,
   prefill `flash2`, the listed fallback for mixed tiles); a feature added to one (sinks, softcap, a gate,
   a window) must be added and unit-tested in all of them. A mismatch passes the goldens and shows up only
   when a turn reuses history the model decoded itself (flint: sinks differed between flash2 and split-K).
   Prefill output must not depend on where the scheduler cuts a request: runs of 9 or more tokens go through flash2
   (whole 16-position tiles) or its partial kernel (head and tail blocks), the same bits either way; runs of 1 to 8 tokens
   (decode, verify) use split-K and are not cut-invariant.
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
   suffix, tools inside a message): see the loom-kb `cfg-review` topic and
   `reference/glimmer-example` (a channel format end to end: `chat/atem.loom` + its `chat_probe.py`; another
   channel format reuses `starter/templates/control_channels.loom`, as `reference/gpt-oss-example` does).
9. **Work in parallel when you can spawn agents.** The kernels (steps 1–6) and the chat block
   (step 8: `chat_probe.py` + `control.loom`, verified against the Jinja template) are
   independent until `build.py`. Give the chat work to a subagent on its own copy of the kit and
   its own port; merge when both pass. Without subagents (other harnesses, e.g. Pi under weaver), do the
   kernels first and the chat block second.

## Workflow, step by step (what port.sh does; use it to debug a stage)

```bash
K="${LOOM_SKILL_DIR}/starter"; T=~/loom/.tools/bin/python
W=~/work/mymodel && mkdir -p $W && cd $W
# 1. Inspect the GGUF: geometry, tensor formats, eos ids, native context, hybrid layers, and the
#    chat block DERIVED from the GGUF's Jinja chat template (chat_probe.py) and VERIFIED against it:
#    it prints "chat OK <conversation>" per probe (chat, multi-turn, tools, system+tools, agent loop,
#    parallel calls, with thinking on/off). Every line must be OK; a FAIL or "not derivable" means
#    fix cfg.json["chat"] by hand (fields: chat_probe.py) and rerun `chat_probe.py MODEL.gguf`.
$T -I "$K/inspect_gguf.py" /path/model-Q4_K_M.gguf cfg.json
# 2. REVIEW cfg.json against llama.cpp's src/models/<arch>.cpp - see the loom-kb `cfg-review` topic
# 3. Tokenizer: the HF tokenizer.json of the *same* model (base repo; GGUF repos often lack it)
~/loom/.venv/bin/hf download <org>/<base-model> tokenizer.json --local-dir tok
# 4. Build: context defaults to the model's native context, the KV pool to context + max(64k, context/4), both fitted
#    to --memory-gib (default auto: that pool, capped at 60% of RAM; one model is loaded at a time). Prints the model id and --memory_bytes
#    (weights + KV pool + the largest program's prefill scratch + 3 GiB + recurrent state: a full-context request must fit).
$T "$K/build.py" cfg.json pkg --tokenizer tok/tokenizer.json --model-id <org>/<name-without-spaces>
# 5. Serve + smoke test
VENV=~/loom/.venv "${LOOM_SKILL_DIR}/../loom-setup/scripts/serve.sh" mine 8160 amdgpu <memory_bytes> pkg/model.json
curl -s localhost:8160/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<model id>","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":400}'
#    thinking models spend 100-400 tokens reasoning: check finish_reason and that content is non-empty;
#    "chat_template_kwargs": {"enable_thinking": false} (or "reasoning_effort": "none") turns thinking off
# 6. Kernel unit tests vs numpy dequant references (embed, matvecs, RoPE, the whole attention chain,
#    the output head, and for hybrids the gated delta-net chain). Run them even if the output looks fine.
$T -I "$K/make_kernel_tests.py" cfg.json ktest && VENV=~/loom/.venv "$K/run_kernel_tests.sh" pkg ktest
# 7. Gates vs llama.cpp - loom-benchmark skill: golden.py/compare.py (short prompts + 1k-32k documents),
#    agent_golden.py (multi-turn tool calling, rendering, history reuse) and `ctx_admit.py` (the full native context is admitted).
```
After every rebuild, restart the server with `stop.sh` and then `serve.sh`; the
server does not reload the package (`serve.sh` stops a server whose registration failed).

**A port is done when** the kernel tests pass, the golden gate shows only near-tie
divergences, `agent_golden.py` passes (exact rendering, reuse inside tool loops,
valid calls), and `ctx_admit.py` shows the full context capacity admitted.

## Domain knowledge (run the router, read what it prints)

Facts about families, facets, devices and tasks are in `loom-kb`, not here. For this port run
`python3 "${LOOM_SKILL_DIR}/../loom-kb/kb.py" route --config cfg.json --device strix-halo --topic porting,cfg-review,starter`
(add `speculative`, `kernels-ir` as needed) and read the files it prints:

| Need | Topic or node |
|---|---|
| cfg.json review checklist (step 2), chat template fields | `cfg-review` |
| What the starter generates, fixed assumptions, chat policy | `starter` |
| How loom-serve drives a program, bring-up debugging order, new quant format, ABI | `porting` |
| Loom IR idioms and compile-error fixes | `kernels-ir` |
| Speculative decoding (drafters, gates, tuning) | `speculative`, `spec-tuning` |
| Image input and adding a projector family | facet `vision` and `arch/vision` |
| MoE, hybrid gated-delta-net, sliding window, drafted, kv-le2 | the matching facet |

`reference/*-example/`: worked ports (table in step 3), applied with `apply_example.sh`; `examples.json` names
each one's category A loom-fun package (its reviewed config). `find_reference.py`: the reference lookup.
Maintainers: after changing the starter, `reference/rebase_examples.sh OLD_STARTER` rebuilds the
examples' patches and `check_examples.sh` must pass.

## Mandate

Declare the scope first, work inside it, opt in by scope, run `fleet/scope_lint.py` before a PR, stop on
out-of-mandate needs: the rules are in `loom-kb/SKILL.md` ("Mandate rules").
