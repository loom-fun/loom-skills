# Image input (vision) for ported LLMs

A GGUF LLM whose repo ships a vision projector (`mmproj-*.gguf`) gets image input from `port.sh` with no
hand-work when the projector family has an encoder plan in `starter/vision/`:

| `clip.projector_type` | Models (checked) | Encoder | Text side |
|---|---|---|---|
| `gemma4v` | Gemma-4-E2B, NVIDIA-Nemotron-3.5-Bolt-E4B | 16 x 768 ViT, RMSNorm, QK/V norm, 2-D NeoX RoPE (theta 100), x/y position tables, gated GELU, sandwich norms, clamped linears; 3x3 average pool, RMSNorm, linear | rows unscaled (no sqrt(H)), per-layer embedding of token 0 |
| `qwen3vl_merger` | Qwen3.5 / Qwen3.6 / Qwen3.8 | ViT with fused QKV + biases, LayerNorm, vision M-RoPE, interpolated (align-corners) position grid, 2x2 merge MLP | M-RoPE: image token (t, h, w) = (p, p+row, p+col); text after an image continues at p + max(rows, cols) |
| `muse-glimmer` | Muse-Glimmer-30B | 50 x 1536 ViT, pre/post LayerNorm, 2-D RoPE (1-indexed, adjacent pairs), 32x32-window attention in 3 of 4 layers, pixel shuffle, 3-layer erf-GELU adapter | rows go through the model's embedding RMSNorm like token rows |

What happens (all generated, nothing per model):
- `port.sh` stage **2b** (`vision/add_vision.py`) picks the projector (F16 > BF16 > F32 > Q8_0), writes
  `cfg["vision"]` (and `cfg["mrope"]` for Qwen-VL); `VISION=0` skips it, `MMPROJ=file` chooses one.
- `build.py` adds the image component: `program/vision_{prepare,control,stages,encoder}.loom`, the
  `vision` parameter scope (the mmproj) and `components.image` in `model.json`. Text-only builds are
  byte-identical to builds without this code.
- Stage 5 runs `vision/tests/run_tests.sh`: every encoder kernel vs NumPy, then the whole encoder on the
  real weights vs `vision/ref_vit.py` (per-row cosine >= 0.999). Stage 6 asks about a test image.
- `loom-benchmark/scripts/gates.sh` adds the **vision** gate (`vision_golden.py`): same request to
  llama-server `--mmproj` and Loom on six images (shapes+text, chart, document, photo-like, tall, wide).
  Prompt token counts must be equal (resize rule, token limits, markers, chat rendering), replies must
  match token by token up to a near-tie.

## How it works

```
chat request ──► control.loom (text)                       image component (vision_*.loom)
  image part      vision_content: size rule -> rows N        prepare_request: tag = processed size
                  renders BEGIN + sentinel + END             prepare_batch: device header + plan
                  plan record {rows N, metadata [w, h]}      vision_encode (one stage per size):
  prepare_input   text segments encoded, sentinel ->           resize (Pillow, as llama.cpp) -> normalize
                  N placeholder ids (0), span per image        -> patch embed -> ViT -> merge -> projector
  encode_epoch    media epochs: 8-word input records         rows -> host copies them into the epoch packet
                  [token, feature offset, ...(t, h, w)]
  device (llm)    g_embed: a live record with a feature offset copies its row instead of the embedding
```

- **Geometry**: `vision/preprocess.py` is llama.cpp's mtmd rule per family (align = patch x merge, min/max
  tokens, half-even rounding for Qwen, Glimmer's nearest-aspect grid); `gen_chat_vision.geometry()` is the
  same rule in the control program (it decides N before any pixels are decoded), and `vision_control.loom`
  re-checks it. Resize is Pillow's separable fixed-point filter (bicubic a=-0.5, Lanczos-3) computed on the
  device; F32 coefficients put about 0.2% of pixels 1 off (no visible effect, prompt tokens unchanged).
- **Like llama.cpp's stb_image**: EXIF orientation is ignored and alpha is dropped (not composited).
- **Chat**: llama-server joins text parts with "\n" and none next to an image, then expands each image to
  `img_begin` + embeddings + `img_end` (`add_vision.MARKERS`, from `tools/mtmd/mtmd.cpp`). Kits whose chat
  program renders the whole conversation in `chat_begin` (gemma4, atem) switch image requests to the
  generic per-message renderer (the host only accepts image markers inside `chat_message` fragments);
  `add_vision.END_STATES` maps the generic `chat_end` state to the kit's output parser state.
- **Device records**: with media, the host's packet holds one 8-word record per input followed by the
  feature rows; metadata word 2 = 1 tells the kernels. Padding inputs of the epoch shape read stale
  records: features are taken only for live inputs (`g_token_info`). Without media nothing changes.
- **M-RoPE** (Qwen-VL): the host's per-row *source position* is the rotary position. Text advances it by 1
  per input, an image row only when it extends its grid frontier, so an image occupies max(rows, cols)
  positions. `encode_epoch` writes (t, h, w) into media records and a per-span offset (source - KV
  position) after the plan; `publish_epoch` reports the new source position; the attention RoPE kernel
  picks t/h/w per frequency pair (`cfg.mrope.sections`, interleaved for Qwen3-VL).

## Adding a projector family

`add_vision.py` exits 3 (`unsupported projector family`) and the port stays text-only. To add one (Gemma
4 took about a day the first time; with these tools a ViT family is a few hours):

1. **Read llama.cpp, not papers.** `tools/mtmd/models/<family>.cpp` (the graph), `clip.cpp` (hparams per
   `PROJECTOR_TYPE_*`: `n_merge`, resize algorithm, `set_limit_image_tokens`, rope theta, `ffn_op`, which
   tensors are optional, host-side inputs in `set_input`), `mtmd.cpp` (`img_beg`/`img_end`, the preprocessor
   class, `mtmd_decode_use_mrope` / `use_non_causal`), and the text model's `src/models/<arch>.cpp` (is the
   embedding scaled or normed for raw embeddings? `build_inp_embd` scales only token rows).
2. **NumPy reference** in `vision/ref_vit.py` (one function, the graph op by op), checked against
   `llama-mtmd-debug -m MODEL --mmproj MMPROJ -p encode -n SIZE --image rainbow`: it prints every named
   tensor's sum and edges. `vision/trace_compare.py MMPROJ LOG rainbow SIZE name...` compares sums layer by
   layer and names the first wrong op. Use a size that exercises every path (Glimmer: > 32 patches a side
   for the windows; an interpolated position grid). Deep encoders drift in the last layers where
   activations reach 1e5 (llama.cpp rounds there too): judge by per-row cosine, not elementwise.
3. **Preprocessing**: a `preprocess.FAMILY` entry (merge, token limits, filter, rounding) or a custom size
   rule (Glimmer), mirrored in `gen_chat_vision.geometry()`.
4. **Kernels**: compose `gen_vision.py` blocks (norms: ln/rmsw/rms; GEMM via the starter's row-scaled F16
   WMMA with F16/BF16/F32 weights; patch embedding with raster or 2x2-block token order; x/y tables or
   bilinear position grids; three 2-D RoPE layouts; full or windowed attention; avgpool / pixel shuffle).
   A new op gets a kernel in `gen_vision.py` and a case in `vision/tests/` against NumPy.
5. **Plan**: a family branch in `encoder.build()` (and `config()` for its constants), then
   `python vision/test_encoder.py MMPROJ H W` (whole encoder on the real weights vs the reference).
6. **Text side**: `add_vision.MARKERS[family]`; positions (normal, or M-RoPE: `MROPE_FAMILIES`); raw-row
   scaling rules in `gen_llm.py` (`FEATURE_PICK`, `EMBED_PRE`). Not supported yet: bidirectional
   attention over image tokens (llama.cpp `use_non_causal`: Gemma 3, Gemma 4 above E4B), several images
   in one encoder run, video, audio.
7. **Gate**: `gates.sh` (vision gate) or `vision_golden.py record` with `MMPROJ=` llama-server, then
   `compare`. Equal prompt tokens first; a mismatch is the size rule, the markers or the chat rendering.

## Pitfalls that cost time

- **Every stage binds the same parameters.** A size-specialized encoder stage that drops a launch (dead
  branch folded away) fails with "checkpoint domains differ from the shared layout". The image
  component's errors are in `/v1/status` (`data[].loom.parameters.image.error`), not in the log.
- **`gpu.event_wait` DATA_LOSS on the second image request** was a feature offset read from a stale
  record of a padding input: only live inputs carry features.
- **`F16` matches `BF16`**: pick projector files by whole token.
- **Massive activations**: Glimmer's last layer reaches 5e6; F16 activations need the row-scaled GEMM.
- Run llama.cpp and Loom one at a time; `vision_golden.py` records twice (llama.cpp flips near-ties).
