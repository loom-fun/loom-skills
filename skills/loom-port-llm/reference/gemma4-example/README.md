# Gemma 4 (arch `gemma4`) port: worked example

A complete, gate-passing port of `unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF` (UD-Q5_K_XL), written by
a fresh agent from these skills in October 2026: LEVEL=full gates PASS (28 golden prompts up to 35k
tokens, max gap 0.59; consistency; agent tool loop 6/6 with exact rendering and history reuse;
sampling; needle at 61k and the full 131k context). Latest run: 52 min end to end, audit PASS.
Make a kit for a Gemma-3n/4-family model with `../apply_example.sh gemma4 W/kit`.

`starter.patch` changes these starter files:

| File | What it adds over the starter |
|---|---|
| `gen_llm.py` | Mixed head_dim 256 (sliding) / 512 (global) attention; KV sharing (layers >= `n_layer_kv_from_start` reuse the last owning layer's KV of the same kind; KV planes only for owners); V RMSNorm without weights; attention scale 1.0; proportional partial RoPE on global layers (`rope_freqs` finite pairs); GEGLU (gelu-tanh) FFN; per-layer embeddings (gathered row + projection + norm + gate + post-norm, on the starter's BF16 matvec and WMMA GEMM (prefill +11 % at 1-2k); count only the gathered row for bandwidth); `layer_output_scale`; final logit softcap applied to the sampling logits (`final_softcap`, argmax unaffected) |
| `gen_flash2.py` | Prefill flash attention for head_dim 512 in two 256-wide output passes (`FLASH2_DO`) with V^T staged per pass (fits the VGPR/LDS budget): 61k-token TTFT 335 s -> 131 s. Sliding layers (window 512) stay on the split-K path: flash there gave nothing |
| `make_kernel_tests.py` | Test cases adapted to the layout (embedding scale, per-tensor K, no k/v tensors on shared-KV layers) |
| `build.py` | the `gemma4` call format and its placeholders |
| `chat_probe.py`, `templates/control.loom` | Gemma 4's chat format, which `chat_probe.derive()` can't model: `<|turn>role` / `<turn|>` turns, the `<|channel>thought ... <channel|>` reasoning channel (closed with a bare `<channel|>`), its tool-declaration and tool-call syntax, tool responses, the thinking switch. `simulate()` mirrors control.loom and is verified against the Jinja template; `raw_from_chat()` is the benchmark hook (`export LOOM_KIT=W/kit`). Needs `extra_eos_ids` [1, 106, 50] |

The reviewed config (geometry, sliding/KV-sharing layers, chat block) is the published package's `llm_config.json`
([loom-fun/nvidia-NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-loom](https://huggingface.co/loom-fun/nvidia-NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-loom),
category A, pinned in `../examples.json`); `apply_example.sh gemma4 KIT CFG` merges it.

Lessons the agent paid for:
- `llama_trace.py ... --grep '-0$|-23$|-24$'` (a sliding, a global and a KV-shared layer) gave the whole
  spec in one run; the kernels were right the first time.
- Bugs in a hand-written `output_next` show up only on real outputs: an empty span must not change
  state (see `porting-notes.md`), and the reasoning close is `<channel|>` with no newline.
- Sliding layers with a window of <= ~1k keys don't need flash prefill (split-K is fine; a trial wasted
  15 min on it). head_dim > 256 needs split-K attention generalized to D/32 values per lane and flash2's
  D/16 fragment loop.
- Known gap: with thinking ON, history after a tool result is not reused (the template re-renders the
  reasoning with an extra newline). Not covered by the gates (they run the template default).
