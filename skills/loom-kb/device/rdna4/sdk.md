# Device family rdna4: SDK code generation limits (gfx1201)

Code generation of hrx-loom 0.1.0 for gfx1201 that bounded kernels of the mxfp4st campaign (2026-11-15/16, DevLab R9700). Same format as
topic `sdk`: the limit, where it bit, the workaround, the unlock. Profiling limits on this target: the family README.

- **f32 -> fp8 e4m3 is software-emulated.** `vector.fptrunc` / `vector.encode` (finite_only) to f8E4M3FN lowers to 91 low ops per 16 values
  (~5.7 VALU per value); other rounding modes have no gfx1201 target-low contract, although gfx12 has packed converts (`v_cvt_pk_fp8_f32`,
  `v_cvt_sr_fp8_f32`). Bit: every e4m3 activation producer and epilogue quantizer of the MXFP4 prefill (producers ~0.031 ms/token at 2k),
  and a GEMM-prologue quantizer design (~+80 % loop issue). Workaround: quantize once per element; an integer-bit conversion through an
  f16 intermediate measured ~3.7 VALU per value (another hand version measured 54 VALU vs 21 per 4 values and lost: price each in the
  compile report). Unlock: lower fptrunc f32 -> f8E4M3FN to `v_cvt_pk_fp8_f32`.
- **f32 -> f16 vectors are not packed.** `vector.fptrunc` of `vector<8xf32>` to f16 emits cvt + bfe + lshl_add per pair, not `v_cvt_pk`.
  Unlock: packed conversion.
- **Uniform addresses are rebuilt per load.** Every `vector.fragment.load` (`global_load_tr_b128`) and `vector.load` rebuilds its 64-bit
  uniform base with ~8 SALU (`s_lshl_b64` + `s_add_co` pairs) instead of reusing one: ~500 SALU per 32 WMMA in a flash kernel, ~409 SALU per
  128-K step of the fp8 GEMM. Making the base opaque (`subgroup.broadcast` of the offset) fixes `vector.load`, but `fragment.load` then
  fails with `fragment_memory.dynamic_stride`. Workaround: lane-vector loads with immediate offsets where the operand layout allows
  (device node `prefill`). Unlock: base reuse across loads of one view.
- **Rescaled loop-carried accumulators are copied at branch edges.** Accumulators rescaled (`vector.mulf`) or merged through `scf.if` get
  branch-edge copies of the whole set (compile report `move_causes branch_edge` 229-515 units per kernel, 100-250 `v_mov` per iteration)
  and spill at ~202 VGPRs. Bit: a one-head-per-wave flash design with 16 loop-carried 8-VGPR accumulators (repro: `flash2_g6.py`,
  `FLASH2_B=6`, branch `mxfp4st-oattn`). Workaround: keep the rescale outside branches (the fixed-max fast path). Unlock: in-place phi
  allocation for accumulators.
- **Workgroup LDS is capped at 64 KB** (`TARGET/051` at 86 KB). Bit: a double-buffered weight stage with a 256-value K step does not fit
  (the 128-step double buffer measured no win anyway).
- **Global fragment loads work but are undocumented.** `vector.fragment.load<lhs>` takes a global-memory view (`buffer.view` of the
  activation buffer with a storage encoding): it was the largest single GEMM win (-11 %), and works only with the encoding-typed view.
