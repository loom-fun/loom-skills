# Topic: program-abi (how loom-serve drives a text program: ABI, lifecycle, extending the starter)

The upstream reference is hrx-system `loom/docs/src/serving/model-programs.md`; library examples live in `hrx-loom-library/program/text/qwen3_8/`.

## hrx-loom 0.1.0 (PyPI) text ABI

The service calls `text_*_v1` exports (service.loom maps them to the control
functions). New in 0.1.0 versus the earlier CI wheels:
- `chat_reasoning() -> (supported, efforts bitmap, default effort)`: efforts none=0 …
  high=4; a toggleable thinking model advertises none|high (bitmap 17).
- `chat_message` takes a `frames` buffer after `parts` and returns a 4th result (the
  image projection count; 0 for text).
- `parse_tools(schemas, response, truncated)` and `tools.parse_xml(schemas, text,
  truncated)` take `truncated` (1 when generation hit `max_tokens`). Records are
  `[{"type":"function","function":{"name":…,"arguments":"<JSON string>"}}]`.
- Model ids must be visible ASCII without spaces (`--model-id`).
- `check.expect.close` for floats is now `check.expect.approximate … : T, T`.

## Lifecycle

1. **prepare** (`@prepare`) runs once at load. It receives the packing shapes,
   each a `(tokens, spans)` pair, and creates one *stage* per shape. Each stage
   compiles `@llm_advance` with `llm.tokens`, `llm.spans`, `llm.context` and
   `llm.pool` set as constants, then binds the GGUF as parameter 0. It returns
   the effective prefill capacity, the stage count, the control state, the
   terminal marker string, and the storage records from `@llm_storage`.
2. **storage** (`@llm_storage`) declares the allocations:
   - slot 0: residual;
   - slot 1: arena, holding the KV pool followed by per-row private regions;
   - slot 2: metadata;
   - slot 3: origins/table;
   - slot 4: tokens;
   - slot 5: outputs.

   It also describes the KV geometry: planes = 2·layers, plane_bytes = pool ×
   POS_BYTES, block = 16 positions. The runtime pages KV in 16-position blocks
   and keeps a per-row block map in `origins`; `g_kv_offset()` resolves
   (row, position, plane) to an arena byte offset.
3. **control** functions turn requests into tokens and tokens back into text:
   - `chat_begin`, `chat_message` and `chat_end` render the prompt text.
     `chat_end` returns the generation prompt and the initial output state.
   - `prepare_input` tokenizes with special-token matching.
   - `output_next` splits the growing response into channels:
     - kind 0: consumed marker;
     - kind 1: answer;
     - kind 2: reasoning.

     State 1 means inside `<think>`, and state 2 skips the whitespace after
     `</think>`.
     **Host rule:** a span of zero length must not change the state. Returning an empty span with
     a new state (e.g. at a channel/tool header whose end isn't generated yet) fails the request
     with "source returned an invalid output span". Wait (return the previous position, same state)
     until the whole marker is in the response, then consume it as a kind-0 span.
   - `encode_epoch` and `publish_epoch` translate the runtime's span plan into the
     **metadata packet**:
     - a 16-byte header, whose word 0 is the span count;
     - then 9 i32 per span: `[0]` length, `[1]` start position, `[3]` row,
       `[4]` first input token index, `[6]` flags, where bit 0 means "emit an
       output token for this span".
   - `epoch` selects the stage compiled for this packing shape.
4. **advance** (`@llm_advance`, the command program) runs once per epoch with all
   spans packed into `t = llm.tokens` token slots:
   - `g_token_info` maps each token slot to (row, position, live, span) from the
     packet, writing 16 bytes per token into `%info`.
   - Embed, the layers, and the head follow. The head computes the argmax for
     each selected span's last token and writes `outputs[span]`.
     `g_head_select` maps any `extra_eos_ids` to the EOS id.
   - Padding slots have `live = 0`. Kernels must not write KV for them, and
     GEMMs can skip tiles whose row scale is 0 (`GEMM_SKIP_PADDING`).

## Decode vs prefill inside one program

The same `@llm_advance` is specialized per shape, and `t` is a constant in each
stage:
- `t < 64` (decode, small batches): matvec kernels, using the 1-token kernel when
  `t == 1` and the 4-token one otherwise, plus split-K attention.
- `t ≥ 64` (prefill): f16 row-scaled WMMA GEMMs, with 128-row tiles from 128
  tokens. Attention uses flash2 on uniform 16-token tiles (one sequence,
  consecutive positions, starting at a position multiple of 16), and the listed split-K fallback handles the other
  tokens. `g_attention_fallback_list` builds both lists, so a token's path depends on its own sequence only.

## Extending the starter

- **New quant format.** Add a `@g_<fmt>_values` func in `COMMON` (gen_llm.py; `mv()` calls it),
  add the entries to `FMT`/`SB` in `starter/formats.py` (inspect_gguf.py's "no kernels" check and pick_gguf.py read
  the same table), and add a `dequant16` case in `gen_gemm.py` (or list it in `gen_gemm.HELPER`: the GEMM then
  dequantizes through the value helper). For int8 decode matvecs, also an `@g_<fmt>_icodes` (integer codes, scale,
  minimum) and the format in `formats.ICODES`; `gen_llm.Q5_0` is a worked example of both helpers. bf16 needs a
  value helper (`view<32xbf16>` -> `vector.extf`). Unit-test against `gguf.quants.dequantize` (make_kernel_tests.py
  does, for every format and K a model uses).

- **Library programs are the canonical reference** for the text-service ABI (prepare,
  control, storage, service entry points). `templates/` follow the upstream tiny-text
  fixture (`loom/src/loom/serve/fixture/text` at tag v0.1.0 of ROCm/hrx-system) and
  `library/program/text/qwen3_8/chat.loom`. The upstream doc is
  `loom/docs/src/serving/model-programs.md`. When a new hrx-loom release changes the
  ABI, diff that fixture between the old and new tags; the registration error names
  the missing export (e.g. `root symbol '@text_chat_reasoning_v1' was not found`).
