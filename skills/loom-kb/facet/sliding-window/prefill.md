# Facet: sliding-window: prefill

## flash2 prefill after the batch-invariant tile lists (#193, #213): what the loss was and how it was recovered

- Spark-X2.6 (head_dim 256, 4 query heads per KV head, 27 of 36 layers sliding, nodraft), prefill 8k/17k/35k tok/s on one
  DevLab runner: pre-#193 2738/2511/2144, #193 2299/1993/1530, main (+#213) 2418/2213/1830. Deleting the flash2 launches
  (timing only) gave the same time for all three, so the whole loss was flash2.
- #193: the tile list's entry (`first_token`) is a *loaded* value, so the compiler no longer knows it is wave-uniform and plans
  the kernel differently (hot loop 1071 vs 1113 instructions, another structure). `first_token = tile * 16` (or pre-#193's
  prologue, or `kernel.subgroup.broadcast` of the entry from lane 0) gave back 2638/2483/2117. A microbenchmark of the
  kernel (loom-benchmark, one launch) did NOT show it (+3..8 %): compare kernels end to end with `devlab-perf` (`double`
  and `drop` variants attribute time), not only in kbench.
- #213: a 16-key block with masked keys runs once per query column with V zeroed at that column's masked keys: about 0.5 ms
  per such block on a 0.6 ms sliding-window kernel (three per tile: window lower edge, diagonal, and the block past the tile's
  last position). With the masked code inside the key loop the loop itself was also slower (nomask ablation 2660/2484/2121).
- Fix: the key loop runs every tile but the last without the masked path and the last tile follows it with the masked path
  (`flash2_peel.py`; only the tile holding the last key can hold the diagonal block); a block that starts past the tile's last
  position skips the masked path (P = 0 and V staged as zeros: exact); a sliding window's lower-edge block is no longer
  zeroed per column (`FLASH2_EDGE_ZERO=1` restores it: its masked keys are older keys of the run, the same whatever the
  cut). `FLASH2_PEEL=0` keeps the loop as it was. Peeling is off for 16 output fragments per wave (head_dim 256 in gen_flash2.py):
  the straight-line loop there needs more registers than the budget (peak 323 of 128 + spills).
- Tried and not adopted: key masks from a workgroup-memory table with a prefetched row (about 1 % at 8k on Spark, LDS and
  code for it), grouping 2-4 fragments per masked loop (spills), fully unrolling the 16 columns (peak 1329 registers), a VALU
  FMA path for the diagonal block (spills in the key loop), splitting the diagonal block into dyadic rectangles (the same 15
  MMAs).
