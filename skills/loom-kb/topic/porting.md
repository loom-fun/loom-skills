# Topic: porting (package card, tokenizer gotchas, bring-up debugging)

The program ABI, lifecycle, decode-versus-prefill shapes and how to extend the starter (new quant format) are topic `program-abi`; Loom IR diagnostics are topic `ir-debug`.

## Package card (`model.json`)

- `programs.source.inputs`: `prepare.loom`, with `weights.loom` as its data input,
  plus `control.loom`, `storage.loom` and `service.loom`.
- `programs.source.libraries`: the kernel library and command program (`llm.loom`).
  Libraries compile per stage with stage constants applied.
- `variants.<v>.services.chat.parameters.model.files`: `[{path, format: "gguf"}]`.
  Weights are read in place (a symlink is fine); tensors are addressed by GGUF
  name in `command.parameter`.
- `variants.<v>.services.chat.config`: `{"llm.card": true}` selects scoped parameter
  sourcing in `prepare`.
- `presets.<p>.execution`:
  - `rows`: concurrent sequences;
  - `context_capacity`: maximum tokens per sequence;
  - `pool_capacity`: total KV positions, shared by all rows and the prefix cache;
  - `prefill_capacity`: tokens per prefill chunk. Shapes are powers of two up to
    this value, and the chunk equals prefill_capacity;
  - `max_tokens`: default completion length.

## Tokenizer gotchas

- Use the HF `tokenizer.json` of the same model. GGUF-only repos usually ship
  without one, so download it from the base repo.
- loom-serve strips *special* added tokens from decoded text. If your output
  parser needs to see a structural token (e.g. Glimmer/ATEM `<|start|>`,
  `<|message|>`), mark it `"special": false` in the package tokenizer
  (`chat.nonspecial_tokens`). `<think>`/`</think>` are already non-special in
  Qwen3.
- `prepare_input` encodes with special-token matching, so special tokens in the
  rendered prompt become single ids, matching HF `apply_chat_template` + encode.

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
5. **503 "insufficient completion capacity".** `--memory_bytes` is too small. The message blames retained
   state, but it also comes on a fresh server: every admission charges the weights, the request's KV pages and
   the largest command program's scratch at a prefill-sized invocation (a drafting hybrid's `@llm_verify` keeps
   every token's gated delta-net inputs: 7 GB at a 4096-token prefill of a 30-layer MoE). build.py's printed
   value covers the weights (projector included), the KV pool (pool × 2·kv_layers × kv_heads·head_dim·2 bytes),
   that scratch + 3 GiB, and the recurrent state.
6. **Registration fails with a VGPR spill (`spill-traffic-register-exhausted`).** A
   kernel holds too much in registers for this model's shape. Split the work over more waves
   (`gen_flash2s.py`: two waves per head at head_dim 256, four at 512) or reload operands per iteration.
7. **Coherent but wrong from the first token, kernel tests pass.** A wiring or
   config bug. Compare against `starter/llama_trace.py` (llama.cpp's intermediate tensors,
   node by node) with a numpy forward of layers 0–1; then read the generated command program for one layer of each kind
   (`awk '/---- layer N$/,/---- layer N+1/' pkg/program/llm.loom`). A real example:
   a stale cfg.json with `qk_norm: false` left Qwen3.5's Q/K norms out (`%no_norm`).
   An independent numpy forward of a short prompt that matches llama.cpp's top
   logits narrows "my understanding of the model" vs "my kernels" quickly; it is a
   diagnostic, never the golden.
