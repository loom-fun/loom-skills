# Model: amd-Qwen3.8-27B-Quark-AWQ-MXFP4-loom

`amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (AWQ-calibrated OCP MXFP4, Quark export) served from its safetensors file: 12.9 GB of MXFP4 linears, bf16 embedding and head (5.1 GB), a bf16
MTP block (0.85 GB, the default drafter), a bf16 vision tower (0.92 GB, image input: read from the checkpoint, see facet/safetensors "Image tower"). Weights 19.8 GB: on the R9700 (32 GB) the served context and pool are limited by the weights plus the
verify scratch of the MTP drafter; the measured serving settings (`exec.*`, devices radeon-r9700) are written to `scope.json` when a DevLab run has measured them.
Source, name map and producer: loom-kb `facet/safetensors`, `facet/quark`, `quant/mxfp4s`.

MTP proposals (depth 3) read load-time copies, never the bf16 tensors: the NextN block as Q8_0 (`draft.propose.block`, 0.85 -> 0.45 GB) and a Q4_0 copy of head rows [0, 49152) + the added-token rows (`draft.propose.head_rows`, 0.15 GB instead of the 2.54 GB bf16 head); `llm_advance` / `llm_verify` keep the checkpoint's tensors (byte-identical programs with the choices off). R9700: decode 53.1 -> 69.2 tok/s, decode_2k 52.3 -> 68.8, acceptance .50 -> .46. The head copy is staged through four load-time lanes of the staged rows (4 x N x 10 KB of device budget, hrx-loom): more rows buy acceptance (N 32768 .41, 49152 .46, 81920 .49; decode 41 / 69 / 70) but cost served context (N = 81920 does not serve 81920 context with the image component), so N = 49152 and context 98304. Now the copy is 16384-row prepared tensors in the main weight policy (`draft.propose.head_pieces`): staging is one piece wide and there is one preparation plan (a separate `head_weights.loom` plan failed the first load of every fresh R9700 VM, RESOURCE_EXHAUSTED), so N = 65536 (acceptance .47, decode +1-1.5 %) and every fresh-VM first load passes at 98304 and 106496. Opt-in, off: `decode.head.verify=q8_0` (the verified head from a load-time Q8_0 copy: decode +2 % over the pieces, 1.2 GB less VRAM, changes the verified output; dev gates vs vLLM pass). Facet notes: `facet/safetensors`.


## Budgets (R9700, leader decomposition and HAL profiles, 2026-11-15/16)

Leader: Paiton (MXFP4 + DFlash2, single R9700): prefill 2k / 8k 3,689 / 3,834 tok/s, drafted decode chat 121 tok/s. Profiles: HAL dispatch
statistics (runs 43804613925, 43847625186), marginal 8k-minus-2k prefill cost.
- **Prefill 8k.** The leader's 0.261 ms/token is the whole budget. The fp8 GEMMs take 0.218 ms/token (store 0.119, SwiGLU 0.098) at
  ~76 % of the random-data fp8 WMMA peak (stop reason: power-bound, device node `prefill`), which leaves 0.043 ms/token for everything
  else; the campaign set <= 0.06 (>= 3.5k tok/s). Non-GEMM bucket 0.149 ms/token at the first profile (GDN 0.073, producers 0.031,
  attention 0.016-0.034, alpha/beta 0.010, head 0.003), ~0.091 after the unread f16 operand copy was dropped, the fused GDN prefill and the
  fp8 alpha/beta GEMM; whole 8k prefill 0.374 -> 0.320 ms/token.
- **Drafted decode (MTP depth 3).** Step 35.5 ms device at 2.5 tokens per step; the verified weights (12.94 GB MXFP4 + 2.54 GB head) are
  26.0 ms of it at 594-600 GB/s, against a 25.3 ms floor at 613 GB/s. At this acceptance the leader's 121 tok/s would need a 20.7 ms step,
  below the weight floor: the decode gap is tokens per step (drafter, acceptance) and verified bytes (the Q8_0 head copy, -1.19 GB), not
  the 9.5 ms of everything else (proposals 3.3, GDN state 2.8, producers 1.5, attention 0.6 ms). Launch gaps are 2.5 % of the step.
