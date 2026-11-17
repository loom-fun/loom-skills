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
see the loom-kb topic `new-arch`.

Prerequisites:
- `loom-setup` is done: `~/loom/.venv` has the SDK (`hrx-loom` 0.2.0 from PyPI, which the
  templates target) and `~/loom/.tools` has gguf, numpy, tokenizers, jinja2 and
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
  from the best one (loom-kb topic `reference`);
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
Programs hand-written in the hrx-loom catalog: loom-kb topic `catalog-ports`.

`DRY_RUN=1 port.sh ...` plans a port on the CPU in under a minute: stages 1-2 only, an uncached GGUF read as a
header-only sparse stub (nothing downloaded), then `DRY_RUN_RESULT {...}` (architecture, kit, reference, inspect).

Its exit code says where it stopped (2 download, 3 chat template or uncovered architecture,
4 build/serve, 5 kernel tests, 6 smoke); only then do you need the manual steps below for that
stage. Verified on Qwen3-0.6B (Q8_0/Q4_K_M) and Qwen3.5-0.8B (Q4_K_M).

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

Facts about families, facets, devices and tasks are in `loom-kb`, not here. Route the topics this port needs:
`python3 "${LOOM_SKILL_DIR}/../loom-kb/kb.py" route --config cfg.json --device strix-halo --topic porting,kernels-ir`
(cfg.json is the `inspect_gguf.py` output, or `inspect_safetensors.py` for a safetensors source: the router then adds its nodes; before it exists use `--gguf MODEL.gguf`). The router prints a node's facts for each topic you name.

| Need | Topic |
|---|---|
| Bring-up debugging order, package card, tokenizer gotchas | `porting` |
| Writing kernels: Loom IR shape and fast idioms; compile errors, proof facts, kernel tests | `kernels-ir`; `ir-debug` |
| cfg.json review checklist (step 2), chat template fields; what the starter generates | `cfg-review`; `starter` |
| The program ABI, lifecycle, a new quant format | `program-abi` |
| An architecture the starter cannot generate; its example kits (`reference/*-example/`, applied with `apply_example.sh`) | `new-arch` |
| A published package to start from (`find_reference.py`); a hand-written catalog package | `reference`; `catalog-ports` |
| Speculative decoding (drafters, gates, tuning) | `speculative`, `spec-tuning` |

`examples.json` names each example's category A loom-fun package (its reviewed config). Maintainers: after changing the starter,
`reference/rebase_examples.sh OLD_STARTER` rebuilds the examples' patches and `check_examples.sh` must pass. Image input and
adding a projector family: facet `vision` (printed for `--topic porting`).

## Mandate

Declare the scope first, work inside it, opt in by scope, run `fleet/scope_lint.py` before a PR, stop on
out-of-mandate needs: the rules are in `loom-kb/SKILL.md` ("Mandate rules").
