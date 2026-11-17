# Device family rdna4: decode and drafted verify (R9700, Qwen3.8-27B)

What bounds a drafted step on the R9700 and what was tried (q38r9700 campaign, engineer "verify"; `devlab-kbench` suite `mv_q38` and `devlab-perf`, all runner r9700;
the suite and the generator knobs are at commit b82a5cab of branch `q38r9700-verify-exp`, not merged).

- **The int8 matvecs are at the bandwidth ceiling.** Kernels exactly as the model builds them, weights beyond the 64 MB MALL (launches cycle over distinct tensors),
  per launch: gate+up q4_k swiglu 5120 -> 17408 (100 MB) 0.160 ms 1 token, 0.162 ms 4 tokens (620 GB/s); down 17408 -> 5120 (50 MB) 0.080 / 0.082 ms;
  qkv 5120 -> 10240 0.048 / 0.049; q6_k / q8_0 6144 -> 5120 0.049-0.056 ms. The 4-token variants cost 1-2 % over the 1-token ones: the ~4 ms by which the
  4-row step (34.1 ms) exceeds the 1-row step (30.1 ms) is gdn 0.8, head 0.3 and attention 0.3 ms plus small kernels, not the weight stream.
- **Decoding U rows per loop iteration (U x the weight loads in flight per wave) is flat or worse.** U = 2/4 with 1-8 rows per workgroup on every shape above:
  -0.5 .. +4 % in kbench (the loads of one row already saturate the memory system), and in the model (all t4 / t1 int8 matvecs, U = 4 / 2) decode 65.6 -> 62.8
  (-4.3 %). The ISA shows the loads batched (6 loads in flight for 2 rows), 64-68 VGPRs, no occupancy limit.
- **GDN capture / restore of a drafted step.** Doubling the idempotent launches in `llm_verify` (48 launches each, MTP depth 3, 40 ms step):
  capture +0.45 ms, restore +0.54 ms (~10 us per launch: a scalar copy loop of ~41 dependent iterations on one workgroup per token), conv +0.32 ms
  and qk_norm +0.28 ms (96 launches each, ~3 us), head +2.1 ms (1.04 GB Q6_K at the bandwidth). Skipping the whole replay (timing only, state wrong)
  saves about 1.3 ms: the ceiling of any replay fusion. `decode.spec.gdn_copy_split` (8 workgroups per token, gfx12 twin) takes ~0.8 ms: decode +2.0 %.
- Not built: restore-free replay (the replay kernels reading the capture slots), fusing conv / conv_commit / qk_norm into the recurrence (~240 launches,
  ceiling ~1 ms = +2.5 % decode), delayed commit into the next step's front kernels (ninfer / Lucebox): each needs rewriting the GDN kernels to one bit-exact order
  and lifts only drafted decode.
- Few-kernel / persistent verify step (MXFP4 27B, R9700, lane ostep; kbench `pstep_r9700`, e2e 43812485595): a software grid barrier (device-scope
  atomics + `buffer.fence scope(device)` release/acquire, correct only with both fences) costs 1.3-1.9 us per phase, a back-to-back launch 0.6-0.7 us:
  persistence buys nothing by itself. Persistent int8 matvecs tie at best (G=1024 workgroups); consumer prologues (rmsnorm / silu + q8 in the matvec)
  lose 28-250 % because every workgroup re-reads the activation. Producer-in-epilogue (`decode.verify.fuse=swq`: gate + up of 32 rows per workgroup,
  silu + q8 codes in the epilogue) is bit-exact and -4 % in kbench but ties end to end (157 us vs 2 x 79). Producers are 1.5 ms of a 35.7 ms step:
  launch fusion is not the decode lever on gfx1201; the GDN state traffic (capture/restore + replay ~2.2 ms) and the proposals (3.3 ms) are.
