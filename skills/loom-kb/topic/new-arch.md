# Topic: new-arch (an architecture the starter cannot generate: the expensive path)

The expensive path. Keep it short:
1. **Get the spec from llama.cpp itself.** `starter/llama_trace.py LLAMACPP_DIR MODEL.gguf "Hello world" --grep '-0$'`
   prints layer 0's compute graph as llama.cpp runs it (op, input names, shapes, values). Use
   `--grep '-N$'` for a layer of each kind (sliding/global, attention/recurrent, dense/MoE). This
   is faster and more exact than reading C++; the GGUF's `arch.*` metadata gives the constants.
2. **Diff it against what the starter does** for a dense layer (Qwen3: `attn_norm → Qcur/Kcur/Vcur
   (+ q/k norm, RoPE) → attention → attn_output → ffn_inp → ffn_norm → gate/up → swiglu → down →
   l_out`). Every extra node or different op is a feature to add; write the list down first.
3. **Copy the starter into the work dir** (`"${LOOM_SKILL_DIR}/../loom-port-llm/reference/apply_example.sh" starter W/kit`,
   which also records the loom-skills commit in `W/kit/KIT_SOURCE`: `build.py` stamps it and the Loom SDK's
   identity into `llm_config.json` `"generator"`, and the model card shows them) and extend
   the copy: a cfg flag per feature in `inspect_gguf.py`, the op in `gen_llm.py` (decode first:
   matvec path, prefill later), and a kernel test case in `make_kernel_tests.py`. If you change
   `chat_probe.py` (a chat format it can't model), `export LOOM_KIT=W/kit`: the benchmark
   scripts then render prompts with your copy, and use its `raw_from_chat(chat, message, finished,
   thinking)` if you define one (how a reply's channels map back to raw text, for a format the
   default doesn't cover). No need to copy the skills tree. Reuse before
   writing: **check `reference/` for the architecture family first** and make the kit with
   `"${LOOM_SKILL_DIR}/../loom-port-llm/reference/apply_example.sh" EXAMPLE W/kit W/cfg.json` (the starter + the example's
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
   | JoyAI's 32-head MLA layout (absorbed latent attention, one 576-value cache row), sigmoid+bias MoE, IQ4_XS experts | `deepseek2` |

   `loom-optimize/reference/flint-moe-example/` holds one more MoE idea (bucketed grouped GEMM), not
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
