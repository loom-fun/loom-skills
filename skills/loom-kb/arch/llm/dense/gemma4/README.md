# Family: gemma4

GGUF arch `gemma4` (Gemma-4 E2B/E4B/12B/31B dense, 26B-A4B with MoE beside the dense FFN, Nemotron-3.5-Bolt-E4B). Kit example: `skills/loom-port-llm/reference/gemma4-example/` (kit docs stay there). Facets: sliding-window, drafted, vision, moe (26B-A4B only).

## Contents

- Porting: features the starter lacks (Gemma-3n / Gemma-4)
- Gemma-4 assistant drafter (`gen_assist.py`)
- Nemotron-3.5-Bolt-E4B drafting: BF16 drafter as published, Q8_0 at load, Q4_0 proposal head
- Gemma-4-E2B prefill on the R9700 (gfx12): where the time is and what moved it

## Porting: features the starter lacks (Gemma-3n / Gemma-4)

- **Gemma-3n / Gemma-4 family (arch `gemma3n`, `gemma4`).** `apply_example.sh gemma4`: a complete,
  gate-passing port whose README lists every feature the starter lacks (PLE, KV sharing, mixed
  head_dim 256/512, KV heads per layer kind and V = K on global layers without `attn_v` (12B/26B/31B), MoE
  layers beside the dense FFN (26B-A4B), proportional partial RoPE, V norm without weights, GEGLU, softcap, bf16); each
  one silently breaks output if missed. Read llama.cpp's `src/models/gemma*.cpp` for a new variant.
  Sliding layers with a small window (<= ~1k keys) don't need flash prefill: split-K is fine there
  (a trial spent 15 min on it for no gain).

## Gemma-4 assistant drafter (`gen_assist.py`)

llama.cpp src/models/gemma4-assistant.cpp, mapped onto gen_draft.py's MTP machinery (per-row carry of the target's
final normalized hidden, `g_mtp_prepare` / `g_mtp_gather` / `g_mtp_advance`). Each round: target embedding *
sqrt(H) concatenated with the hidden -> `nextn.pre_projection` -> 4 layers (query-only attention, GELU FFN, post
norms, `layer_output_scale`) -> the assistant's tied head (argmax) and `nextn.post_projection` (the next round's
hidden). The attention reuses the trunk's own decode kernels for the two target layers (parsed from the program text,
`target_attention`): the assistant's QA heads are placed in the target's QH-head layout so head a reads KV head
a // (QA/KVH), and `g_as_visible` clamps every round's queries to the anchor's context (positions < P). The propose
stage therefore binds the target's origins table and arena (build_draft.py); warm only saves the carry.
Nemotron-3.5-Bolt-E4B with the E4B assistant: acceptance equals llama.cpp's (depth 1: 75 % vs 77 %, depth 3: 57 % vs
57 %), spec gate 16/16 exact including images, 46.9 -> 72.7 tok/s at depth 3.

## Nemotron-3.5-Bolt-E4B drafting: BF16 drafter as published, Q8_0 at load, Q4_0 proposal head

BF16 drafters as published (Nemotron-3.5-Bolt-E4B Q4_K_M, NVIDIA's BF16 DFlash, same-session perf.py, two runs each,
c4_steady three runs; llama.cpp b1368 drafting from the same BF16 GGUF):

| | decode | @2k | @32k | prefill 1k / 8k / 32k | c4 (median) |
|---|---|---|---|---|---|
| Loom, BF16 drafter, depth 3 (package choice) | 81.9 | 70.6 | 51.3 | 2223 / 1835 / 929 | 197.4 |
| Loom, BF16 drafter, depth 7 | 81.8 | 61.0 | 45.1 | 2179 / 1811 / 907 | 154.1 |
| Loom, Q8_0 requantized drafter, depth 7 | 90.1 | 65.3 | 45.9 | 2208 / 1830 / 928 | 164.9 |
| Loom, no drafting | 58.7 | 56.2 | 44.9 | 2314 / 1872 / 937 | 211.3 |
| llama.cpp, DFlash n-max 7 (BF16) | 89.3 | 58.4 | 37.2 | 1312 / 1291 / 732 | 169.0 |
| llama.cpp, no drafting | 51.9 | 44.5 | 33.2 | 1350 / 1304 / 739 | 155.9 |

- The BF16 drafter costs ~9 % of drafted single-stream speed against the Q8_0 copy (twice the bytes per proposal on
  the f32-activation 4-token matvec); the copy cannot be referenced by a published card.
- Depth 3 is the package setting: same short decode as depth 7 on perf.py's essay, better with KV in use and with 4
  streams (verification of 4 x 8 tokens takes the GEMM). Four drafting streams still trail no drafting (197 vs 211):
  a 4 ms model step leaves little to win at 39 % acceptance.
- **Now (PR #42): the BF16 drafter is converted to Q8_0 at load and the proposals read a Q4_0 head copy** (both through
  weight policies, `prep_weights.py`; the card still references NVIDIA's BF16 file). DevLab Strix Halo, one runner,
  perf.py means of 4 runs, c4 median of 3, llama.cpp b1368 with the same BF16 drafter at n-max 7:

  | | decode | @2k | @32k | prefill 1k / 2k / 8k / 32k | c4 |
  |---|---|---|---|---|---|
  | Loom, depth 3 (this package) | **98.0** | **81.4** | **63.6** | 2310 / 2430 / 2073 / 1017 | **223.9** |
  | Loom main (BF16 drafter as published), depth 3 | 82.6 | 70.6 | 53.4 | 2289 / 2416 / 2078 / 1013 | 189.5 |
  | llama.cpp, DFlash n-max 7 | 79.9 | 60.4 | 37.7 | 1407 / 1400 / 1625 / 782 | 186.6 |

  Per step on the essay (perf.py's decode request): main 25.1 ms and 2.09 tokens; Q8_0 drafter 23.5 ms (same
  tokens); plus the Q4_0 proposal head 22.1 ms and 2.16 tokens. llama.cpp takes ~48 ms per step for 3.5-3.9 tokens
  (it decodes a different text: llama-server turns thinking on by default, the package follows the template's
  default off). llama.cpp's own decode is bimodal across runs on DevLab (~97 on the first request of a fresh
  server, 73-81 after), so compare means of several runs. The table above used each engine's default thinking.
  With perf.py's `--thinking` (now default off, the same `enable_thinking` sent to both engines; text_match.py
  shows the same text up to near-ties), one runner each, llama.cpp at its best n-max (3):
  thinking off 97.6 vs 61.4 tok/s (+59 %; n-max 7: 50.1), @32k 63.4 vs 49.6, c4 219 vs 165;
  thinking on 113.1 vs 83.5 (+35 %; n-max 7: 82.5), @32k 72.9 vs 51.9, c4 270 vs 198.
- Tried (Bolt, DevLab, no gain): depth 7 (93-94 tok/s short, but @2k 62-70 vs 81: 8-token verification costs ~4.5 ms
  more per step than 4 for +0.4 accepted tokens on this essay); fewer rows per workgroup for the 8/16-token int8
  matvecs on small outputs (MV_WG8 128-1024: flat to -5 %); the drafter's own widths in the R picker (flat); a Q4_0
  drafter (+0.5 %, not worth its own kernel path).
- Tried: proposals of 5+ tokens on the BF16 WMMA GEMM instead of the 4-token matvec (one weight pass): depth 7
  146.3 -> 137.2 tok/s, slower (64-row padded tiles for 8 tokens). Drafter FFN with SiLU instead of the gemma4 kit's
  GELU (llama.cpp's dflash graph uses SiLU unless `hidden_act` says otherwise): no change (145.0 vs 145.7).

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

## Lessons from the E4B port (from the gemma4 kit)

Lessons the agent paid for:
- `llama_trace.py ... --grep '-0$|-23$|-24$'` (a sliding, a global and a KV-shared layer) gave the whole
  spec in one run; the kernels were right the first time.
- Bugs in a hand-written `output_next` show up only on real outputs: an empty span must not change
  state (see the loom-kb `porting` topic), and the reasoning close is `<channel|>` with no newline.
- Sliding layers with a window of <= ~1k keys don't need flash prefill (split-K is fine; a trial wasted
  15 min on it). head_dim > 256 needs split-K attention generalized to D/32 values per lane and flash2's
  D/16 fragment loop.
- Known gap: with thinking ON, history after a tool result is not reused (the template re-renders the
  reasoning with a `\n` before `<channel|>` that the model does not generate). loom-serve resumes a retained
  row only when the next prompt extends its history exactly (hrx-system `input.c`: the transcript must be a
  prefix; an idle row cannot be cut back to the common prefix, and the prefix cache only captures
  intermediate frontiers of long prefills), so neither the chat program nor storage can reuse the common
  prefix: the turn is prefilled again (~200 tokens per tool step in the agent gate). The agent gate requires
  reuse only where the runtime can give it (`agent_golden.py`: history the model verifiably generated and
  the template rewrote is not required to be reused).
  Likewise for templates whose thinking-off generation prompt adds an empty thought channel (Gemma 4
  12B/26B): the template does not render it in history, so the next turn re-prefills from there.

## Fleet models (published packages, llm_config.json)

- `lmstudio-community-gemma-4-26B-A4B-it-loom` (gemma4): 30L h2816, 16h/8kv hd512, moe 128e top-8, swa 1024, drafter dflash d1, ctx 262144
- `nvidia-NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-loom` (gemma4): 42L h2560, 8h/2kv hd512, kv_layers 24, swa 512, vision, drafter dflash d3, ctx 131072
- `unsloth-gemma-4-12B-it-loom` (gemma4): 48L h3840, 16h/8kv hd512, swa 1024, drafter dflash d3, ctx 262144
- `unsloth-gemma-4-31B-it-loom` (gemma4): 60L h5376, 32h/16kv hd512, swa 1024, drafter dflash d3, ctx 262144
- `unsloth-gemma-4-E2B-it-loom` (gemma4): 35L h1536, 8h/1kv hd512, kv_layers 15, swa 512, vision, drafter assist d2, ctx 131072
- `unsloth-gemma-4-E4B-it-loom` (gemma4): 42L h2560, 8h/2kv hd512, kv_layers 24, swa 512, vision, drafter assist d3, ctx 131072
