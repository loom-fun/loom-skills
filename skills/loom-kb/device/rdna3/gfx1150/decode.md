# gfx1150 (Strix Point): decode

## Matvec grid size (`decode.mv.wg1`) on 16 CUs (Qwen3.5-4B, hrx-loom 0.2.0, 2026-11-16, local, 2 sessions per variant)

| `MV_WG1` | decode | decode @2k | prefill 1k / 2k |
|---|---|---|---|
| 4096 (legacy, tuned on 40-CU Strix Halo) | 31.5 | 28.9 | 1006 / 1029 |
| 2048 | 32.1 | 28.3 | 1006 / 1026 |
| **1024** | **32.3 (+2.4 %)** | **30.4 (+5.2 %)** | 1007 / 1029 |
| 8192 | 28.9 | 27.1 | 1005 / 1026 |

**Opportunity, not pursued (PM decision 2026-11-16).** A grid of a quarter the workgroups suits a GPU with 16 CUs: the only device-sensitive lever found. Not applied: the key is not
`per_target` and gfx1150 shares the gfx11-generic programs with Strix Halo, so a gfx1150-only value needs its own program set
(or a per-device twin), and Strix Halo has not been measured at 1024. `MV_WG_LONG=256` and `GEMM_M256=0` (tie / prefill -7 %) do not help.

## Drafting depth and prefill chunk are device-neutral here

DFlash depth 3 (the card's) beats 2/4/5 (-2 / -9 / -10 % decode); prefill chunk 1024 beats 512 and 2048, and 4096 does not fit a 14e9
budget (HTTP 503 `insufficient completion capacity`, server healthy). The Strix Halo preset is the right preset here except the context cap.
