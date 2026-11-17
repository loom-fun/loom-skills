# MXFP4, split planes (`mxfp4s`)

OCP MXFP4 as a safetensors producer stores it: blocks of 32 e2m1 values sharing one E8M0 scale (4.25 bits per weight), but the nibbles and the scales live
in two separate tensors, `X.weight` (U8 `[rows, K/2]`) and `X.weight_scale` (U8 `[rows, K/32]`), instead of ggml's interleaved 17-byte blocks (`quant/mxfp4`).
Why planar and unrepacked: a repacked copy next to the original would need 2 x 12.9 GB on a 32 GB card, so the kernels read both planes as stored.
Format data: `format.json` (owned by the port PR); `starter/formats.py` stays the code source. The generator's side: `gen_llm.py` (`MXFP4S`, `@g_mxfp4s_values`),
`gen_gemm.py` (`dequant16`), source tooling `starter/inspect_safetensors.py` + `starter/hf_names/`.

## Value model

- Value = fp4(nibble) x 2^(e - 127); fp4 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6 with the sign in bit 3.
- **Exact in bf8 (e5m2)**: every fp4 value times a power of two is an e5m2 number while the exponent fits, so a weight loses nothing in an fp8-class matrix
  instruction. Held at 2^8 x its value, a block with E8M0 byte e in 107..132 decodes with one table lookup per nibble; the row factor 2^-8 goes to the epilogue.
  Rows differ by up to 12 exponents and 6 within a row on the GGUF checkpoint measured (e = 107..125, 99.99 % at 119/120); check the range of a new
  checkpoint (a range check tool exists in the kernel tests) before relying on one global shift.
- Activations are the lossy side of an fp8 prefill GEMM: e4m3 with a per-row power-of-two scale, about 2 % relative rounding per element, allowed
  by the campaign decision and gated by the golden gate.

## Element order and scale convention

- Element `2j` of a row is the **low** nibble of byte `j`, element `2j + 1` the high one (vLLM `csrc/rocm/mxfp4_w4a8_gemv_rdna3.cu`: "low nibble = even k").
  Not ggml's `MXFP4` order (value `j` low, `j + 16` high). Checked on the AMD checkpoint against the unquantized Qwen3.8-27B rows (layer 0 `gate_proj`,
  layer 3 `q_proj`, layer 10 `in_proj_qkv`, 32 rows each): correlation of the dequantized rows 0.990 / 0.994 / 0.993 with the low-nibble order, 0.06 with
  ggml's order, 0.00 with swapped nibbles. `tests/test_safetensors_source.py` compares the starter's reader with a scalar reimplementation of vLLM's decode
  on real rows (exact).
- Scale: one E8M0 byte per 32 consecutive elements of a row, value `2^(e - 127)` (`e = 0` is 2^-127; `0xFF` is NaN and never occurs). Rows of the AMD file use
  exponents 117..123 (2^-10 .. 2^-4), weights of std 0.009-0.014.
- `pack_method: "reorder"` in Quark's config is the int4 packing's nibble reorder (`quark/torch/utils/pack.py` `Pack_4_bits`); the fp4 planes are stored in element order,
  so no reorder is applied.
- The tensors the format key covers: every `<name>.weight` that has a `<name>.weight_scale` beside it (dtype U8); the producer's exclude list (lm_head, MTP,
  vision tower, embeddings) stays bf16 and is read as `bf16`.

## Kernels

Matvec, f16 GEMM and the gfx12 fp8 WMMA GEMM read the two planes (a weight-operand abstraction at the generator's call sites gives one view per GGUF
format and two for `mxfp4s`); the fp8 prefill path is choice-selected per device and its measured results go to `device/rdna4`.

Kernels (generator `starter/gen_mxfp4s.py`; choices `prefill.gemm.mxfp4s_f8`, `decode.mxfp4s.act_q8`; kernel tests in the `gemm_mxfp4s` kbench suite and in
make_kernel_tests):
- `g_gemm_mxfp4s_{store,swiglu}_f8_m256_rdna4`: gfx12 prefill GEMM on `v_wmma_f32_16x16x16_fp8_bf8`. Weights: the thread that stages a weight row-half loads 2 x 16
  nibble bytes (two vector loads) and 2 scale bytes of its row per 128-value step, decodes to bf8 exactly in LDS (`device/rdna4/prefill.md`, "MXFP4 split planes on fp8
  WMMA"). The fused swiglu kernel takes gate and up as two operand pairs (the checkpoint stores them as two tensors). 256 x 128 tile (swiglu: 64 columns of each),
  activation fragments straight from global memory.
- `g_mv_mxfp4s_k*_q8*`: decode and verify matvecs on int8 activations (dot4); `@g_mxfp4s_icodes(%w, %ws, %origin, %scale_origin, %group)` decodes 32 values to the signed codes 2 x fp4
  and the block scale 2^(e - 128). The f32 matvec of the simple kernels is ALU-bound at the 4-row verify width.
- Exactness: `scan_mxfp4s_exponents.py` (local or HF range read) reports the E8M0 range of a checkpoint; the AMD Qwen3.8-27B checkpoint has e = 107..128 in all 760,954,880
  blocks, so the bf8 decode (exact for 107..132) loses nothing. A checkpoint outside the range needs a wider shift or the f16 path (not built).
- Bytes: nibble plane 16 B + scale plane 1 B per 32 weights = 4.25 bits per weight, the same as ggml's 17-byte blocks; the planes are two streams per row.
