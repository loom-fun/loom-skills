# Device family rdna3: prefill (WMMA GEMM m256 and why it helps)

## WMMA GEMM m256 (`gemm_big256`)

- **m256: dequant per MMA halved** (why: the RDNA3 VALU/WMMA serialization, in `device/rdna3`).
  m256 is the m128 layout with twice the
  rows (512 threads, 256 × 128 tile; swiglu interleaves 64 gate + 64 up rows in one weight stage, picking the
  buffer per wave with `scf.select` on `buffer`), so each dequantized weight slice feeds 256 rows (~115 VALU per
  32 WMMAs). 55 KB LDS (padded rows), 2 workgroups per WGP. Real-data microbenchmark (interleaved
  `loom-benchmark --compare`), Qwen3.8-27B shapes at 1024 rows: down +9.5 %, gate +5.4 %, gate/up swiglu
  +4.5 %; prefill 1k/2k/8k +5.8/+5.0/+6.0 % (this includes the GDN input prefetch below; the recurrence
  ablation puts that share near zero).
  - Tried: gufo's 256 × 256 tile with 1024 threads. LDS can't hold both padded operands (73 KB), so the
    stages were K-slice-major unpadded (`[4 x 256 rows] x 16`): 22.8 TFLOP/s vs 30.8 (one workgroup per WGP,
    2-way LDS read conflicts). A half-swap swizzle to remove the conflicts made the compiler keep one address
    register per fragment half (VGPR 184 → >192, did not fit 1024 threads; the 512-thread swizzled form ran
    256 VGPRs and 19-22 TFLOP/s). Moving m128's 8 KB inverse-scale tile into the activation stage (3 instead
    of 2 workgroups per WGP) was mixed: down -4 %, swiglu/gate +1.5/+2 %; not kept.
  - Tile order (`GEMM_N_FASTEST`) makes no difference for m256 on the 27B shapes.
- **m256 with 64 × 64 waves.** `gemm_big256` now runs 256 threads (8 waves, 4 × 2) of 64 rows × 64 columns
  instead of 16 waves of 32 × 64: each A fragment feeds 4 B fragments and each B fragment 4 A fragments (0.5
  LDS fragment reads per MMA instead of 0.75), VALU per WMMA 4.5 → 3.0 (store loop), 232-248 VGPRs (two
  workgroups per WGP still fit). Each thread stages two activation rows and one 32-value K group of a weight
  row. Real-data microbenchmark, 27B shapes at 1024 rows (REPS=7 interleaved): swiglu/gate/down
  +2.5/+2.1/+4.3 %. Qwen3.8-27B same-server A/B (two sessions): prefill 2k +2.0/+1.7 % (484.5 → 494.2,
  482.7 → 490.9), 1k +1.3 %, 8k +1.9 %; Bolt (gemma4 kit) 1k/2k +1.3/+1.1 %. All inside the noise band, but
  the sign held in every run.
  - Where the m256 time goes (store kernel, 27B shapes, same session): full kernel 32 TFLOP/s; no LDS
    fragment reads (fragments held in registers) 36; no staging (no global loads, dequant, barriers) 37-38;
    neither (WMMA from registers only) 40-41. Single ablations were each small: no dequant +6-8 %, no global
    loads +4.5 %, no barriers 0-5 %, 1 of 4 B fragments +1-3 %. The overheads are latency, not bandwidth.
  - Tried: loads of fragment step kk+1 placed before step kk's MMAs: the compiler reschedules anyway (tie).
    Forcing that order with `scf.schedule.fence`: 200-208 VGPRs, -15 %. Fences around each step's loads and
    MMAs: +2 % once, -6/-10 % on reruns (noise). The 64 × 64 layout with fences: swiglu -8/-10 %. A per-wave
    `buffer.view` at the B-fragment row so all B reads use immediate offsets (swiglu VGPR 208 → 160): tie,
    so occupancy is not the limit either. Sharing one Q4_K scale/min decode between a thread's two 16-value
    slices: tie (not kept). Load-time weight repack: capped at the dequant ablation's 6-8 %; not tried.

- **Epilogue and dequant levers (q38pf campaign, Qwen3.8-27B Q4_K_M, Strix Halo, 2026-11-20).** All opt-in switches:
  - `GEMM_M256_F16O` (choice `prefill.gemm.m256_f16o`): the m256 swiglu stores f16 + scales (1, 1) / (0, 0) (gen_gemm.f16out_m256,
    f16o_fuse.fuse gfx11=True), so the f row-scale pass goes (row_scale_f16_live_lv .047 -> .021 ms/token). DevLab +0.8..1.4 % prefill.
  - `GEMM_M256_G32` (`prefill.gemm.m256_g32`, q4_k only): one 32-weight group dequant per thread, uniform header math; loop VALU
    184 -> 129 per 64 WMMA, kernel -2..-3 %. With F16O: DevLab +2.0..2.7 % all rows; local 123 W (with MIXED) 8k 590.7 -> 612.0 (+3.6 %),
    16k +3.6 %, 32k +3.4 %. q6_k G32 loses 5-10 % (an exposed vmcnt wait after the barrier).
  - `GEMM_M256_MIXED_F16O` (`prefill.gemm.m256_mixed`): q4_k gate + q6_k up layers in one f16o swiglu m256 (every thread loads both
    rows, wave-uniform branch picks the dequant; 256 VGPR, no spills): kernel -7.4 % vs the two store GEMMs, silu_mul and the f
    scale go; DevLab +0.4..1.0 % on top of F16O+G32.
  - Lost / stopped: `GEMM_M256_XCLAMP` (unconditional clamped next-step activation loads; removes q6_k's post-barrier vmcnt(0) +
    32 v_mov but the scheduler then loses 2 % on q4_k swiglu and q8_0; DevLab -1 %); `GEMM_M256_DB` (32-K double buffer, -13..-30 %);
    `GEMM_M256_SWZ` (tile order already settled above); `s_setprio` is not expressible in Low asm.
  - Where the time goes at 123 W (swiglu, 1024 rows): g32 35.8 TFLOP/s; no barriers 37.5 (+4.6 %), no LDS fragment reads 36.4
    (+1.8 %), both 39.0 (+8.9 %). The barriers are the largest in-kernel cost left.

## WMMA runs on the VALU (why m256 helps)

On RDNA3 WMMA runs on the SIMD's VALU, so the weight dequant serializes with it: the m128 GEMM spends ~190 VALU instructions per 32
WMMAs; m256 halves that per MMA (~115; see `topic/prefill.md`, "WMMA GEMM"). The gfx12 twins have the same property (below in `device/rdna4`).
