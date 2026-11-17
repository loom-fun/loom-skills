# Family gemma4: prefill

## Gemma-4-E2B prefill on the R9700 (gfx12): where the time is and what moved it

Share of the 2k / 8k / 16k / 32k prefill by kernel group (one extra copy of each group's launches, same job; before the changes below):
q4_k/q6_k GEMMs 63 / 55 / 48 / 38 %, flash attention d512 (7 global layers) 7.7 / 18.4 / 29.8 / 44.8 %, flash d256 (28 sliding layers) 9.8 / 8.2 /
6.4 / 4.9 %, row scale 9.3 / 6.7 / 5.9 / 4.6 % (the f32 FFN tensor alone is ~60 % of it), norms ~4 %, f32 PLE GEMMs 1-3 %. A doubled launch
site must be matched on `apply<@selN_<kernel>` (a plain `g_gemm_` regex also hits the template providers and runs the kernel four times).
- **Swiglu GEMM + row scale as one gfx12 launch group (+4.2 / +5.4 / +4.6 / +4.1 / +3.1 % at 1k / 2k / 8k / 16k / 32k).** `g_row_scale_f16` reads the
  f32 FFN tensor twice (it is 50-100 MB per layer at 2k rows: DRAM bound, ~5.6 % of the prefill) to write an f16 copy. The gfx12 twin
  `<K>_f16o_rdna4` of the swiglu GEMM stores the result as f16 itself (an f32 fragment stored into an f16 view narrows) and writes the scales its
  consumer reads, (1, 1) per live row and (0, 0) per dead one (a row whose input scale is 0), from n-tile 0. The row scale only matters at |f| >= 8192
  (f16 overflow protection; its scale is 1 below that), so the unscaled f16 result is the same numbers except for rows that large (a value above
  65504 becomes inf). `f16o_fuse.py` rewrites the program with `// dual-group:` launch groups (dual_target.py): the gfx11 provider launches the
  swiglu GEMM and the row scale, the gfx12 provider the one twin. Opt-in per model: llm_config `rdna4_f16_ffn` (env `GEMM_F16O=swiglu`); `LOOM_RDNA4=0`
  and gfx11 are unchanged. The ktest cases (`gm_*_sw_*_f16o`) launch the twin directly on gfx12 and are skipped elsewhere.
- **Sliding flash (d256) from `gen_flash2s.py` (+4.7 / +2.9 / +2.3 / +1.4 / +1.0 %).** gemma4's sliding kernel is the generic `gen_flash2.py` (Q reloaded
  every key tile at head_dim 256); its gfx12 twin is now the split-dims kernel (same launch signature; `FLASH2_SWA_TWIN=s`, set by the kit).
- **Lazy O^T rescale (+1.0 / +1.7 / +2.0 % at 8k / 16k / 32k; bit-identical).** The 64 multiplies per wave and key tile that rescale the output
  accumulators by exp(m_old - m_new) are skipped when no query column of the wave changed its running max (`kernel.subgroup.vote.any`); a column whose
  max did not move has alpha == 1 exactly.
- **What the d512 global-layer flash kernel is bound by.** Ablations (`FLASH2_ABL`, outputs garbage) as share of the kernel at 32k: barriers 30 % (three per
  16-key tile and 16 waves), V^T staging stores 20 %, S MMAs 22 %, PV MMAs 19 %, exp 17 %, global K/V loads 12 %: no single hotspot. The hot loop is
  ~650 instructions per wave and tile for 16 WMMAs and WMMA/VALU do not overlap, so the kernel is issue bound (~3200 cycles per tile and workgroup measured,
  ~2500 of them VALU/LDS issue of 4 waves per SIMD). Tried and not worth it: prefetching the next K/V tile into registers (-6..-13 %: the loads sit behind
  an `scf.if` phi and are waited for at once), eight head groups per KV head / 4 waves per workgroup (`FLASH2_NG512=4`: -5..-11 %: the K/V staging is
  per workgroup), two keys per staging lane with one 32-bit V^T store (-2..-4 %), S partials exchanged as lane vectors, mask-free hot loop, flat V^T store
  indices, hoisting the page-map pointer (each ~0 to +0.5 %), 16-key tiles for the sliding kernel (`FLASH2_KT256=16`: one workgroup holds all eight heads, -0.5..-1.8 %),
  and a two-barrier pipelined loop (`FLASH2_PIPE=1`, flash2_pipe.py: K_{t+1} stored after barrier B, V^T_{t+1} after C, loads of tile t+1 issued
  before the S MMAs, page entry of t+2 carried; correct, ktest 72/72, but -0.8..-3 %). The disassembly shows why a restructure at IR level does not
  pay: the backend copies loaded vectors into other registers right away (`s_wait_loadcnt` before the first S MMA, so the prefetch is waited for at
  once; it also waits for in-flight loads when it reuses their address registers) and issues the K fragment loads one at a time with `s_wait_dscnt 0`
  before each WMMA (216 VGPRs). The knobs stay opt-in. A workgroup may reserve at most 65536 bytes of workgroup memory on gfx1201 (TARGET/051).
  Per-token elementwise kernels (norm, row scale) are bandwidth bound, not dispatch bound: 64 / 128 / 256 threads per row make no difference in the
  `e2b_small` kbench (norm 33 us, row scale 10 us at 2048 x 1536 f32).
- **d512 global-layer flash: two heads per wave, key-split loop (`FLASH2_B=2`, `flash2_b2.py`; +7.3 / +11.7 / +18.8 % at 8k / 16k / 32k, +2.6 % at 2k,
  +0.8 % at 1k over the stack above; func pass, ktest 72/72).** The kernel is bound by the K / V fragment loads, not by the instruction count. What
  did not move it (same job, same stack; 8k / 16k / 32k tok/s vs 17.1k / 15.2k / 11.9k): K and V^T fragments straight from the arena with
  `global_load_tr_b128` and no staging (`flash2_direct.py`: ~598 -> 471 instructions per tile, 17.4k / 15.5k / 12.3k), the same with the
  fragments shared through workgroup memory (391 instructions, 17.2k / 15.2k / 12.0k), 256 dims per wave and two waves per head (spills 256 VGPRs,
  14.7k / 12.0k / 8.4k), eight waves per workgroup instead of 16 (+2 / +3 / +4 %), and a key-split loop with ONE softmax per S tile and one head per
  wave (19 % fewer instructions, ties). What did: giving each wave TWO query heads, so a K fragment (one 128-byte load per lane per 8 fragments)
  and a V^T fragment feed two WMMAs and a workgroup requests half the K / V bytes per query row (128 WMMAs per wave and 64-key block, 2 barriers,
  single-buffered exchange: 40 KB of workgroup memory, three workgroups per WGP; the double-buffered 50 KB version was 18.2k / 16.6k / 13.4k).
  Q^T of the two heads (32 KB, f16) is staged once per workgroup; four heads per wave would need 64 KB. A wave's per-load scalar address math
  (64-bit base + constant per fragment, rematerialised for every load: ~17 SALU per V^T load) and the loop-carried accumulator copies (~150 v_mov
  per block) remain; making the base a divergent value is rejected by the fragment-load lowering (`fragment_memory.dynamic_stride`).
- **Prefill chunk with the stack:** 4096 vs 8192 vs 2048: 16.5k vs 16.4k vs 16.3k (1k), 18.1k vs 18.1k vs 17.6k (2k), 17.2k vs 16.8k vs 17.1k (8k).

## KV-shared E2B/E4B/Bolt pruning

`prefill.prune_shared` opts shared-KV Gemma4 models into the kit's `prune_shared_rows` pass. Only `@llm_advance` changes: after the first shared layer and any required DFlash feature captures, a new token-info buffer marks every non-final live row of each span as padding, and attention lists are rebuilt from it. KV owners, verification and drafting stay unchanged. Epochs below 64 rows copy the metadata unchanged. `PRUNE_SHARED=0` builds the unpruned A/B variant.

The pass requires the SDK 0.2 `command.def` spelling and fails if the expected advance prologue or first shared-layer marker disappears. Reusing a pass anchored to the old `command.program.def` spelling silently disabled the optimization after migration. Keep a generated-program assertion, not just a patch-application check.

DFlash is an additional consumer of non-output rows: Bolt captures target features after layers 5/11/17/23/35. Pruning at the first shared layer (24) corrupts the later tap's non-final rows even though verification can preserve target text. The pass therefore starts after the last actual `g_dfl_tap` capture (layer 36 for Bolt), or skips pruning when the final layer is tapped. Assistant catch-up consumes only the final hidden row per span, so its pruning boundary stays at the first shared layer. Check feature consumers before calling rows unused. The llama.cpp Gemma4 graph guards `prune_shared` with `!cparams.embeddings_layer_inp[il]` across shared layers (source `src/models/gemma4.cpp`, inspected at `32bdc3a`); delaying until the last required capture preserves those inputs while still pruning the remaining layers.

E4B remains disabled in the default pruning scope. Its Strix final passed standard full gates, genuine cached-prefix pairs and native-context text/image requests, but one of five strict concurrent-output trials diverged. A later same-job baseline/candidate/ungated diagnostic passed all trials; candidate and ungated programs were byte-identical, so this identified no fix and did not invalidate the earlier failure. Keep its canonical 1024-token chunk and existing 31 GB nightly budget until strict consistency is resolved. Experimental builds remain available with `PRUNE_SHARED=1` and `--prefill 2048`.

E2B is also held despite passing Strix validation. R9700 strict same-job comparisons fail on both original base and candidate, and a verified ungated assistant candidate still fails all five trials with the same prompt 0/token 6 divergence. Removing the occupancy gate is not a fix. Keep its canonical 1024-token chunk, existing R9700 4096-token execution override and 16 GB nightly budget. The default scope enables only the corrected Bolt candidate; E2B/E4B require explicit experimental opt-in. Preserve the observed failures without weakening exactness or broadening into an SDK rewrite.
