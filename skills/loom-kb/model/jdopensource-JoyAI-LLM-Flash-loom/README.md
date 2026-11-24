# Model: jdopensource-JoyAI-LLM-Flash-loom

Strix Halo prefill generator choices of the JoyAI-LLM-Flash card (deepseek2 example kit, IQ4_XS): the IQ4_XS inline dequant
GEMMs, the column-wave grouped MoE GEMMs and the MLA k_b / v_b prefill products; resolved by `resolve.py` at build time.
Measurements and stop reasons: the JoyAI pfgemm lane (PR #1361) in `scope.json` evidence and the `choices.py` docs.

## Long-context prefill (lc lane, PR #1379; DevLab Strix Halo, deepseek2 example)
- **Prefill chunk 8192** (example cfg `prefill_capacity`), made to fit by `PACC_CAP` (gen_llm.py, default on with flash): the
  prefill-shape split-K partials of the listed fallback (1.18 MB per token for MLA, 4.6 GB at 4096) are sized per live listed
  token (runs < 9 tokens: <= 8 per span), the slot carried in bits 16.. of the fallback list entry. Workspace 11.9 -> 9.6 GiB
  at 8192. vs 13a30433 (run 45327051907, same job ABBA): prefill 8k 2249 -> 2386, 16k 1685 -> 1777, 32k 1150 -> 1208,
  131072 324 -> 308 s; 1k/2k, decode, c4 unchanged; gates dev PASS. Why: half the per-chunk re-expansion of earlier keys
  (xexp 0.019 -> 0.013 ms/token at 8k) and 256 rows per expert in the MoE GEMMs. 16384 does not fit: kernels assume
  tokens <= 8192 (161 sites) and the workspace leaves no 131072-position pool at 47 GB.
- **g_mla_xflash is at its structural roof** (kbench mla_xpf, runs 45300767627 / 45312788272 / 45316793511 / 45324027053):
  27 TFLOP/s at 32k; every overhead ablated (no LDS fragment reads, barriers, softmax, P exchange) 32 TFLOP/s at 32k and
  only 5 % faster at 131k (the K/V stream bounds it). Shares at 32k: LDS fragment reads 7 %, softmax + exchange 4 %,
  barriers 3 %. Lost or tied (opt-in switches, gen_mla_x.py): MLA_XF_T fma / s4 / s4,fma (end-to-end tie, run 45320932518),
  diag / lazy (allocator spills), u2; MLA_XF_QL=4 (+9 %); MLA_XF_SRC fragments from global (k_pe tie, K +11 %, V +29 %);
  MLA_XQ 8 / 16 query tiles (+11 / +20 % at 131k). VALU trims are capped by the 4 % softmax share on RDNA3 here.
- Expansion fused into xflash: priced at ~6x the attention FLOPs per 64-query workgroup (re-expansion per head per WG);
  not built. Absorbed residue at 8k (flash2 + partial + kbm/vbm) 0.3 % of busy: crossover rules (MLA_X_MIN_KEYS) settled.
