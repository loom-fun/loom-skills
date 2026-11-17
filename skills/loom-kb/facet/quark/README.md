# Facet quark: checkpoints produced by AMD Quark

Selected by the llm_config key `quark` (the checkpoint's `quantization_config`, copied in by `inspect_safetensors.py`). Read together with
`facet/safetensors` (the container) and `quant/mxfp4s` (the tensor format).

## What the producer writes (config keys)

- `quant_method: "quark"`; `export.pack_method` ("reorder" in the AWQ MXFP4 export) says how the nibbles are ordered inside a byte; the verified element
  order is in `quant/mxfp4s`.
- `global_quant_config.weight`: fp4 e2m1, per_group, group_size 32, scale e8m0 (`scale_calculation_mode: even`, round half_even);
  `global_quant_config.input_tensors`: fp4 dynamic per_group 32. The producer's semantics are therefore **W4A4**; vLLM on a GPU without native fp4 emulates
  the activation quantization (quantize-dequantize), AMD-class hardware with fp4 matrix cores runs it natively.
- `exclude`: the module names that stay bf16 (here: the vision tower, the MTP module, lm_head, embeddings; linear attention gates and norms are not linears).
  Every other linear is quantized, including attention q/k/v/o and gated delta-net in_proj_a/b. The exclude list, not the tensor dtype alone, is the
  authority for which tensors are bf16.
- `layer_quant_config` is empty for the checkpoints seen; a non-empty one (per-layer override) is a finding.
- `algo_config`: null in the files seen; the calibration (AWQ) happened before export and shows only in the values.

## Tensor naming

A quantized linear `X` has `X.weight` (U8, `[rows, K/2]`, two e2m1 nibbles per byte) and `X.weight_scale` (U8 e8m0, `[rows, K/32]`). A bf16 tensor has
only `X.weight`. The inspector derives the format from the U8 dtype plus the scale tensor, then cross-checks it against `exclude`.

## What it means for Loom

- AWQ calibration: the scales were chosen with activation statistics, which is why this checkpoint is the reference quality (an uncalibrated round-to-nearest
  MXFP4 GGUF of the same model is lower quality). The weights are used exactly as stored, never requantized.
- Loom does not quantize activations to fp4: matvec activations are int8/f16, prefill activations may be fp8 e4m3 (`quant/mxfp4s`). The golden reference
  emulates A4, so exact token agreement is not expected on near-ties (see `facet/safetensors` benchmark task file).

## Known checkpoints

- amd/Qwen3.8-27B-Quark-AWQ-MXFP4 (qwen35 hybrid, 19.8 GB, one safetensors file, repo sha 93245f7): MXFP4 linears 12.9 GB, bf16 embed and lm_head 5.1 GB,
  bf16 MTP 0.85 GB, bf16 vision 0.92 GB. Used by Paiton, Radiance and saber.

- `export.pack_method: "reorder"` is Quark's int4 nibble reorder (`quark/torch/utils/pack.py` `Pack_4_bits`: [0 2 4 6 1 3 5 7]); the fp4 planes of this export are in
  element order (element `2j` low nibble), verified on the real checkpoint (`quant/mxfp4s`). No reorder is applied.
- Edge cases: `e = 0` decodes as 2^-127 (denormal, never seen) and `0xFF` is NaN (never seen); the AMD file has no all-zero blocks that matter (exponents 117..123 per
  row group). Other exports seen: none yet (MoE experts and a fused qkv would be new tensor shapes: add them to the family map, not to the producer file).
- The producer's convention is data, `starter/hf_names/quark.json` (scale suffix `_scale`, U8 nibbles + U8 E8M0, format `mxfp4s`, `exclude_key`), applied by
  `inspect_safetensors.py` (`tensor_entries`): a `<name>.weight` U8 with a `<name>.weight_scale` beside it is `mxfp4s`; a tensor of the exclude list has no scale and
  stays bf16 (`bf16` matrix, or a small prepared f32 tensor); a quantized tensor that is also in the exclude list, or a scale of a shape other than `[rows, K/32]`, stops the inspector.
  The llm_config key `quark` (weight / input dtypes, group size, pack_method, excluded count) is what routes this node.
