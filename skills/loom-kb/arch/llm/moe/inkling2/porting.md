# Family inkling2: porting

## Chat template, tokenizer, drafter, projector

- Tokenizer: `thinkingmachines/Inkling-2-Mini` (`tokenizer.json`). The chat format is a channel format (`<|message_*|>`,
  `<|content_*|>`) with `reasoning_effort` none / minimal / low / medium / high / xhigh / max (default high), single and
  parallel tool calls, and history reuse; `chat_probe --verify` 99/99 exact and 44 VM cases.
- The vocabulary is laid out in bands, not frequency order: on 10,812 real generated tokens, `[0, 8192)` holds 45 %,
  space-prefixed English words `[147456, 184320)` 33 %, the chat/tool special tokens `[196608, 200064)` 11 %, numbers and
  `\n\n` `[65536, 73728)` 5 %. A `[0, N)` prefix shortlist (`prep_weights.subset_rows`) covers only 47 % at N = 32768.
- Drafter: DSpark (`dspark-Inkling-2-Mini-Q8_0.gguf`, same repository; facts in this node's `speculative.md`).
  Projector: `mmproj-F16.gguf` (`inkling2v` ViT, 23 layers, 2x2 merge; size rule ceil-align to 40 with -1.0 pre-norm padding
  on the right/bottom, bicubic downscale only above 4096 tokens). Image features bypass `token_embd_norm` in `g_embed`.
- `thinkingmachines/Inkling-2-Mini` also ships `mtp.safetensors` (3-layer MTP, block 8, markov rank 256), not in the GGUF.

## Reference for golden and compare

- llama.cpp ROCm b1383 (= ggml b12600) predates `inkling2_loop`: the reference GGUF was the user's file remapped to the
  `inkling2_loop` schema with bit-exact tensor bytes, plus 48 x 512-byte `.scale` copies for `blk.24..47` (the loader read
  `.scale` after the loop-layer copy). Mainline >= b12611 loads Unsloth's root file directly.
- Without `LOOM_KIT` pointing at the kit, `compare.py` cannot render this chat format and every prompt reads "0/64 BAD".

## Memory and context (Strix Halo, FP16 KV)

- Only the 8 full-attention logical layers need paged KV: 32 KiB per token in f16, so 1,048,576 tokens is 32 GiB. The 40
  sliding layers fit in a per-row f16 ring of 1024 positions in the recurrent slot (160 MiB per slot), copied once on a
  prefix-cache fork. 1M context fits under the 62.5 GiB GTT of a 128 GB box (34 GiB KV + 16.4 GiB weights + 1.3 GiB rings
  + workspace).
- Workspace decides admission. At `prefill_capacity` 4096 the round-1 workspace was 6.7 GiB and `context_admit` refused
  1M under `--memory_bytes=64e9`. Sizing decode-only scratch by the decode token count, prefill-only scratch by the
  prefill count, dropping dead buffers and capping verify rows took it from 11.25 to 1.84 GiB.
- Shipped preset: 262,144 context, prefill 4096. Fixed backing commits 30.45 GiB (admits 262,075); elastic
  `--memory_bytes=32e9` commits 21.1 GiB, so it fits a 64 GB Strix Halo. Kernel `index.assume` caps on pool arguments
  must allow the largest pool (`2097152` caused `TARGET/059` compile failures on bigger boxes; `2147483584` fixes it).

## Kit structure gotchas

- `@llm_advance` has three token-count paths: `%gemm` (t >= 64, prefill), `%single` (t == 1) and small t (2-63,
  decode/verify). Some post-conv launch sites had only `scf.if %single`: a per-span serial kernel there ran prefill on
  9 workgroups for 4096 tokens. Always write all three branches.
- `gen_llm.quant_wiring` asserts every int8 activation read by `g_mv_*_q8*` / `g_moe_gu_*_q8_k*` has a known producer;
  add a new producer kernel (a fused norm+quant, a fused combine) to that regex.
- ShortConv state: `hist[0]` is t-3 ... `hist[2]` is t-1; commit writes the last K-1 inputs of hist ++ span. Speculative
  spans do not commit; they capture <= 8 inputs that the verify program commits (`gen_draft.py` rewrites the
  `commit` / `span` kernels into `commit_cap` / `spancap`). The kernels assume one span per row per epoch.
- Bit-identical fusions must keep the original sum order: `rms_norm_plane` reduces per-thread vec4 dots at tid, tid+256,
  tid+512 then `workgroup.reduce`; the decode bf16 projection is 72 per-group dots plus a 96-thread workgroup reduce.
  `g_quant_q8`'s math (amax/127, divf, roundf, f16-rounded d) reproduces exactly in registers with xor shuffles over the
  8 lanes of a 32-value group, which is what lets norm+quant fuse bit-identically.
