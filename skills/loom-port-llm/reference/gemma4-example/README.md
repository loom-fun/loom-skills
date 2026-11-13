# Gemma 4 (arch `gemma4`) port: worked example

A complete, gate-passing port of `unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF` (UD-Q5_K_XL), written by
a fresh agent from these skills in October 2026: LEVEL=full gates PASS (28 golden prompts up to 35k
tokens, max gap 0.59; consistency; agent tool loop 6/6 with exact rendering and history reuse;
sampling; needle at 61k and the full 131k context). Latest run: 52 min end to end, audit PASS.
Make a kit for a Gemma-3n/4-family model with `../apply_example.sh gemma4 W/kit`.

`starter.patch` changes these starter files:

| File | What it adds over the starter |
|---|---|
| `gen_llm.py` | Mixed head_dim 256 (sliding) / 512 (global) attention; KV sharing (layers >= `n_layer_kv_from_start` reuse the last owning layer's KV of the same kind; KV planes only for owners); V RMSNorm without weights; attention scale 1.0; proportional partial RoPE on global layers (`rope_freqs` finite pairs); GEGLU (gelu-tanh) FFN; per-layer embeddings (gathered row + projection + norm + gate + post-norm, on the starter's BF16 matvec and WMMA GEMM (prefill +11 % at 1-2k); count only the gathered row for bandwidth); `layer_output_scale`; MoE layers (26B-A4B, `gemma_moe_layer`): the starter's routed experts (`moe_routed`: fused gate_up, per-expert down scale, GEGLU) beside the dense FFN on `pre_ffw_norm_2(x)`, routed on `rmsnorm(x) * ffn_gate_inp.scale / sqrt(hidden)`, the two branches summed after `post_ffw_norm_1/2`; final logit softcap applied to the sampling logits (`final_softcap`, argmax unaffected) |
| `gen_flash2.py`, `gen_llm.py` | Prefill flash attention for head_dim 512 (global layers): the starter's `gen_flash2s.py`, four waves of 128 dims per query head (`FLASH512_DO=256` keeps the earlier two 256-wide output passes, 61k-token TTFT 335 s -> 131 s over split-K; 26B-A4B prefill 8k/16k/32k 1.35x/1.64x/2.17x over that), tested by `flash_prefill*_d512`. Sliding layers (window 512) stay on the split-K path: flash there gave nothing |
| `gen_llm.py` | **Prune KV-shared layers' non-output rows on prefill** (`prune_shared_rows`, `PRUNE_SHARED=0` disables): layers >= `kv_shared_from` of `@llm_advance` read a token info in which only the last live row of each span is live, as llama.cpp's gemma4.cpp `prune_shared` does (E2B prefill 1k-32k +55..106 %, E4B +27..51 %, Bolt +28..32 %; decode, outputs and `@llm_verify` unchanged). Models without shared layers are unchanged |
| `gen_llm.py`, `gen_flash2.py`, `gen_flash2s.py` | **Dense KV pages for narrow layer kinds.** A layer kind whose KV position is narrower than the widest kind (26B/12B/31B global layers: 2 / 1 KV heads x 512 = 2 KB / 1 KB of a 4 KB position slot) packs the 16 positions of a page densely at its own width inside the page's 16 x `POS_BYTES` block (`g_kv_offset_<bytes>`; block, plane and pool sizes, hence the runtime's block copies, are unchanged). Reading half-empty 4 KB slots halved the global-layer scan rate (same bytes, 1.15 vs 0.62 ms in a kbench scan; 7 of the 23 ms/token of 26B decode at 32k). `make_kernel_tests.py` builds the flash arena in that layout and adds `flash_prefill_d512_negcontrol` (old layout, must fail; `run_kernel_tests.sh` inverts `*_negcontrol` cases). **Only the full-attention (global) kind packs densely**: the sliding kind (window 512) gains nothing at 32k and Bolt-E4B, whose narrow kind is the sliding one (2 x 256 next to 2 x 512), measured 61.1 -> 59.2 tok/s decode with it. Every KV reader of a kind nevertheless takes its offset function and position stride from `at_dim` (`KVF`, `KVW`) or `FLASH2_KV_FN` (split-K, multi-token and K/V-once bodies, segment body, KV store, both flash2 kernels): #413 left the sliding flash2 kernel on the plain layout while a narrow sliding kind stored densely, so a cached-prefix prefill read the wrong bytes (Bolt consistency "The" vs "This"). `flash_prefill_swa*` tests the sliding flash2 kernel of a two-kind model (position-major arena, no negative control) |
| `make_kernel_tests.py` | Test cases adapted to the layout (embedding scale, per-tensor K, no k/v tensors on shared-KV layers) |
| `build.py` | the `gemma4` call format and its placeholders |
| `chat_probe.py`, `templates/control.loom` | Gemma 4's chat format, which `chat_probe.derive()` can't model: `<|turn>role` / `<turn|>` turns, the `<|channel>thought ... <channel|>` reasoning channel (closed with a bare `<channel|>`), its tool-declaration and tool-call syntax, tool responses, the thinking switch. `simulate()` mirrors control.loom and is verified against the Jinja template; `raw_from_chat()` is the benchmark hook (`export LOOM_KIT=W/kit`). Needs `extra_eos_ids` [1, 106, 50]. Two template layouts are derived: Google's own Gemma 4 templates render an assistant turn as thought, calls, their results, then the content (`chat.content_after_calls`), and the 12B/26B ones open the model turn with an empty thought channel when thinking is off (`generation_prompt_plain` differs from `generation_prompt`); this port's template puts the content before the calls and has one generation prompt. `chat/gemma4.loom` holds both as `// @@IF_CHAT:...@@` variants (build.py `chat_variants`), so this port's programs are unchanged |

The reviewed config (geometry, sliding/KV-sharing layers, chat block) is the published package's `llm_config.json`
([loom-fun/nvidia-NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-loom](https://huggingface.co/loom-fun/nvidia-NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-loom),
category A, pinned in `../examples.json`); `apply_example.sh gemma4 KIT CFG` merges it.
The model-specific geometry (KV-sharing start and KV-owning layer count, global layers, embedding scale, per-layer embeddings, rope cut, SWA head_dim and theta, softcap, stop tokens) is read from each GGUF by the starter's `inspect_gguf.py`, so a Gemma 4 of another size (E2B, 12B) does not inherit this E4B port's values from the reference. The 12B/26B/31B layouts are covered too: a KV-head count per layer kind (`kv_heads` on the sliding layers,
`kv_heads_global` on the global ones, e.g. 8 and 1; each kind's attention, RoPE + KV-store and flash kernels are generated
with its own count, and a KV position is as wide as the widest kind, `POS_BYTES` = build.py's `pos_bytes`), and global
layers without `attn_v`, where V is the raw K projection (llama.cpp `Vcur = Kcur`: `g_qk_rope_store` reads V before K's
norm and RoPE, then V-norms it). A layout with several KV-head counts within one kind still stops at inspect with
`WARNING: unsupported layout`.

Lessons the agent paid for:
- `llama_trace.py ... --grep '-0$|-23$|-24$'` (a sliding, a global and a KV-shared layer) gave the whole
  spec in one run; the kernels were right the first time.
- Bugs in a hand-written `output_next` show up only on real outputs: an empty span must not change
  state (see `porting-notes.md`), and the reasoning close is `<channel|>` with no newline.
- Sliding layers with a window of <= ~1k keys don't need flash prefill (split-K is fine; a trial wasted
  15 min on it). head_dim > 256 needs split-K attention generalized to D/32 values per lane and flash2's
  D/16 fragment loop.
- Known gap: with thinking ON, history after a tool result is not reused (the template re-renders the
  reasoning with a `\n` before `<channel|>` that the model does not generate). loom-serve resumes a retained
  row only when the next prompt extends its history exactly (hrx-system `input.c`: the transcript must be a
  prefix; an idle row cannot be cut back to the common prefix, and the prefix cache only captures
  intermediate frontiers of long prefills), so neither the chat program nor storage can reuse the common
  prefix: the turn is prefilled again (~200 tokens per tool step in the agent gate). The agent gate requires
  reuse only where the runtime can give it (`agent_golden.py`: history the model verifiably generated and
  the template rewrote is not required to be reused).
  Likewise for templates whose thinking-off generation prompt adds an empty thought channel (Gemma 4
  12B/26B): the template does not render it in history, so the next turn re-prefills from there.
