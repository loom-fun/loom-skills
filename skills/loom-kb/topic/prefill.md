# Topic: prefill (compute- and attention-bound prompt processing)

Levers for the WMMA GEMMs, flash attention, chunk size and fusions. Numbers come from the Flint (40B MoE) and Muse Glimmer (30B dense) ports on Strix Halo (RDNA3.5, gfx1151) against llama.cpp ROCm b1355 unless a lever says otherwise. Kernels are in `loom-port-llm/starter/gen_llm.py` / `gen_gemm.py` / `gen_flash2*.py`.

## WMMA GEMM (`gen_gemm.py`)

- **Row-scaled f16 inputs.** `g_row_scale_f16` writes x/scale as f16 together with
  the per-row scale. The GEMM dequantizes weights to f16 fragments and accumulates
  in f32. The output is multiplied by the row scale (store, acc or swiglu modes).
- **Tiles.** 64, 128 or 256 rows × N tile: m128 from 128 tokens, m256 (`gemm_big256`) when the epoch's rows
  are a multiple of 256 (`GEMM_M256=0` turns it off).
- **Register prefetch** of the next K-step, guarded by `scf.if step+1 < k_steps`
  (without the guard the prover rejects the access).
- `GEMM_N_FASTEST=1` puts output tiles on grid x, so consecutive workgroups reuse
  the same A rows from L2. This is model dependent. It won on Flint and Glimmer
  (large N, many experts), but on dense Qwen3-4B turning it off was 4–12 % faster
  at 8k–16k prefill. A/B it per port; `GEMM_N_FASTEST=0` disables it.
- `GEMM_SKIP_PADDING=1`: padding rows get scale 0 (`g_row_scale_f16_live`), and a tile whose
  rows all have scale 0 exits. Packed epochs have up to 50 % padding at odd lengths.
- **Ceiling.** We reached 25–33 TFLOP/s of a ~59 peak (~40 from registers with real data). Unexplored: LDS double
  buffering of B, and direct Q4_K→fragment dequant.

## flash2 (`gen_flash2.py`)

- The workgroup is one 16-token tile × one KV head, with G waves (one per query
  head). K and Vᵀ for 32 keys are staged once in LDS for all G heads.
- **Transposed form.** Sᵀ = K·Qᵀ. In the WMMA result, each lane's column is one
  query and its 8 elements are 8 keys, so the row max and row sum are in-lane plus
  one xor-16 shuffle. Pᵀ is assembled from the lane pair, then Oᵀ += Vᵀ·Pᵀ.
- Q stays in registers (loaded straight from global memory), so no LDS is needed
  for Q or for softmax scratch.
- Gains: 2.2× on Glimmer attention. 64-key tiles (`FLASH2_KT=64`) were *slower*.
- It only handles *uniform* tiles: all 16 tokens live, one row, consecutive
  positions (`g_tile_uniform`) that start at a position multiple of 16, so a tile depends on its own span, never on
  where the epoch packs it (epoch-aligned tiles made a token's path, and its rounding, depend on its neighbours: the
  Coder consistency failure). Every other token goes to the listed fallback.
- **Token subtiles at small G** (`FLASH2_TS`): each staged key feeds 16·G queries, so at G = 2 long prefill re-streams
  the KV cache per 32 queries and is bandwidth-bound (Qwen3-0.6B: 25 % of roofline at 1k, 13 % at 32k). Two tiles per
  workgroup share each staged K/V tile: Qwen3-0.6B / 1.7B 32k prefill 2.6x / 2.1x, 1k tie (default at G <= 2). At
  G = 3 / 4 it lost 3-16 %; 4 or 8 tiles lost to 2 (too few workgroups per chunk).

## Last-layer output rows (`LAST_OUT`, llama.cpp `inp_out_ids`)

- A prefill epoch only needs the last layer's post-attention rows that feed the head (one per span). gen_llm.py
  rewrites the last layer: gather those rows of the attention output and residual (`g_rows_gather`), run the
  output projection, residual, FFN and final norm on them (the decode matvecs, with their own int8
  activations), scatter the final rows back (`g_rows_scatter`). K/V are still written for every token.
- Off for drafters that read the trunk's final hidden rows (MTP, Gemma-4 assistant) or a DFlash tap after the
  last layer; the verify program switches it off. Qwen3.8-27B: greedy outputs identical; prefill +1 % (the last
  layer is ~1.5 % of the GEMM work).

## Listed fallback

- `g_attention_fallback_list` lists the tiles flash2 computes (their first tokens, which its workgroups read) and
  compacts every other token, padding included, into a list, using an atomic counter. The split-K attention then runs over
  that list with an indirect grid (`[buffer]` launch). Before compaction, padding
  tokens did full attention work.

## Per-token RoPE + KV store (`qk_rope()`)

- One 256-lane workgroup per token, with waves looping over all q and k heads. The
  angles are computed once per lane and reused for every head. The optional
  per-head RMSNorm uses a subgroup reduce. This was 6× faster than the per-head
  version.

## Fusions

- RMSNorm + f16 row scale in one pass (`g_row_scale_f16`).
- Residual add + next RMSNorm (`g_post_add_norm`), with optional Gemma-style post
  norms.

## Tile order per port (`cfg["gemm_n_fastest"]`)

- Columns fastest (the default) reuses each activation tile; rows fastest reuses each weight tile across
  the chunk's row tiles (gufo's activation-panel grouping is the same idea). It is shape dependent, so the
  reviewed cfg carries the winner: Muse-Glimmer-30B rows fastest +4 to +12 % (three server sessions: two wins,
  one tie; never a loss); Qwen3.8-27B and Bolt tie; Flint -4 % at 8k. Repeat the A/B across sessions.

## Settled for starter ports (do not re-run these A/Bs)

**Settled for starter ports (dense / Qwen3.5 hybrid, ≤ 4B): don't re-run these A/Bs.** Fresh-agent
trials re-tested them on Qwen3.5-0.8B and the defaults won or tied every time:
`GEMM_N_FASTEST=0` (slower prefill, 3 trials), `MV_WG1=1024` (no decode win), `--prefill 4096`
(short prefill −20–37 %; 2048 ≈ 1024 within noise). On the R9700 (64 CUs) the larger chunk wins for small models (hidden size ≤ 2048: Qwen3-0.6B +7..+24 %, Gemma-4-E2B +5..+15 %, Qwen3-1.7B +11..+18 % from 8k) and loses at 1k/2k on big ones (Qwen3-14B −8/−5 %): `build.py --prefill-r9700 N` (or `prefill_capacity_r9700` in llm_config.json) adds a second preset `<preset>-r9700` that only the R9700 Quick start fetches. On a starter port, spend the budget on fusion
(fewer launches per layer: fused QKV or norm fusion, which you write in a copy of the generator; pairing Qwen3.5's tiny GDN alpha/beta matvecs
measured neutral) or on the long
prefill path. Bigger or MoE models: the chunk size is worth one A/B measured at 1k too (4096 helped
one MoE port; 2048 and 4096 both cost 12-18 % at 1k on flint).

## Did not help (prefill)

- **64-key flash tiles.** Register pressure; occupancy dropped.
