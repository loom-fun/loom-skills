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

- ROCm b1383 uses upstream build 12600 (`29b52d398`). It includes `inkling2_loop`, but copies the shared layers before
  installing their per-expert NVFP4 scales. The historical Strix experiment worked around this with an explicit GGUF
  remap and 48 copies of 512-byte scale tensors for the extra visits. Use the pinned `Mainline_Support/` asset with
  ROCm b1384 (`44c20b1c0`) for stock reference runs. b1384 contains the upstream fix and matched native chat on the
  R9700S at both quantizations. [Upstream loop-scale fix](https://github.com/ggml-org/llama.cpp/pull/32704)
  [Stock ROCm b1384](https://github.com/lemonade-sdk/llamacpp-rocm/releases/tag/b1384)
  [Reference proof](https://github.com/loom-fun/loom-skills/actions/runs/45285206955)
- The gates do this by themselves for the Strix package (checkpoint `ea0c79b2`, compact layout): `loom-benchmark`
  `scripts/reference_models.json` maps its checkpoint to the sha256-pinned `Mainline_Support` file and refuses a llama.cpp
  older than b1384, and `_lib.sh` renders its `call_format` `inkling2` with `scripts/renderers/inkling2/chat_probe.py`.
  Without them stock llama.cpp cannot load the compact file (`blk.24.attn_q.weight not found`): every reference stays
  pending, and the starter renderer turns each reply into zero tokens, so consistency and spec report a divergence "at
  token 0" with no near-tie verdict possible. Those were the SDK 0.3.0 sweep's Strix gate failures, on 0.2.0 too.
  [Fixed gates](https://github.com/loom-fun/loom-skills/actions/runs/45624503314/job/139241089705)
- Point `LOOM_KIT` at the package's chat renderer for compare and agent. Select the renderer for its layout and chat
  policy. The Radeon kit ships `dev/kit/chat_probe.py`. Send `reasoning_effort` inside llama.cpp's
  `chat_template_kwargs` and confirm prompt counts. The Radeon Loom policy requires the tool message's `name` as well
  as `tool_call_id`. Use explicit temperature zero for the greedy check and fixed seeds separately for the service
  defaults. [Sampling proof](https://github.com/loom-fun/loom-skills/actions/runs/45288818385/job/138091995682)
- The frozen Radeon package reports six `STRUCTURE/034` purity-contract diagnostics if the expert modules are parsed
  in isolation on either SDK. Use the linked source package to validate runtime behaviour. Both pinned quantizations
  served on the R9700S. [Offline sweep](https://github.com/loom-fun/loom-skills/actions/runs/45252435739)

## Memory and context

The numbers below are for an f16 KV cache; a quantized KV cache scales the KV terms down by its bytes per value.

- Only the 8 full-attention logical layers need paged KV: 32 KiB per token in f16, so the native 1,048,576-token context
  is 32 GiB. The 40 sliding layers fit in a per-row ring of 1024 positions in the recurrent slot (160 MiB per slot in
  f16), copied once on a prefix-cache fork. At f16, 1M context fits under the 62.5 GiB GTT of a 128 GB Strix Halo (34 GiB
  KV + 16.4 GiB weights + 1.3 GiB rings + workspace).
- Workspace can decide admission at long contexts. At `prefill_capacity` 4096 the first workspace was 6.7 GiB and
  `context_admit` refused 1M under `--memory_bytes=64e9`. Sizing decode-only scratch by the decode token count, prefill-only scratch by the
  prefill count, dropping dead buffers and capping verify rows took it from 11.25 to 1.84 GiB.
- Published preset: 262,144 context, prefill 4096. Fixed backing commits 30.45 GiB (admits 262,075); elastic
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
- A fusion that must stay bit-identical keeps the original sum order: `rms_norm_plane` reduces per-thread vec4 dots at tid, tid+256,
  tid+512 then `workgroup.reduce`; the decode bf16 projection is 72 per-group dots plus a 96-thread workgroup reduce.
  `g_quant_q8`'s math (amax/127, divf, roundf, f16-rounded d) reproduces exactly in registers with xor shuffles over the
  8 lanes of a 32-value group, which is what lets norm+quant fuse bit-identically.
