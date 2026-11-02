# Trial results: the baselines to beat

One row per end-to-end trial (fresh agent, `trial_prompt.md`, Strix Halo gfx1151). Append new rows;
never edit old ones. Wall clock is install -> done; the target is the model's roofline budget
(`loom-roofline/e2e_budget.py`: Qwen3.5-0.8B 46, LFM2.5 53, Bolt-E4B 53, flint 75, Muse-Glimmer-30B 106 min). "Contended" = another trial shared the GPU
(perf numbers then only valid from `scripts/eval.sh` on a quiet GPU, see the table at the end).

## End-to-end trials

| Date | Model | Skill rev | Wall (min) | Manual port (min) | Final gates | Max gap | Agent loop | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-31 | Qwen3.5-0.8B | main ea305b0 | 62 | 0 | full PASS | 0.37 | 6/6 | optimization stopped at 16 of its 60 min on noise; full gates 31 min; contended |
| 2026-10-31 | Qwen3.5-0.8B | faster-e2e v1 | 45.5 | 0 | full PASS | 0.26 | 6/6 | ~13 min of it a self-inflicted baseline re-measure (fixed since); full gates 9.5 min; contended |
| 2026-10-31 | Bolt-E4B (gemma4) | main ea305b0 | 127 | 9 + chat subagent | full FAIL (agent reuse) | 0.20 | 5/6 | hand port; final gates 32 min; contended |
| 2026-10-31 | Bolt-E4B (gemma4) | faster-e2e v4 | 108 | 27 (chat ~30 in parallel) | full PASS | 0.59 | 6/6 | qualify 27 + finalize 26 min, contended |
| 2026-10-31 | Bolt-E4B (gemma4) | faster-e2e v6 | 76 | 3 | full PASS | 0.31 | 6/6 | with reference/gemma4-example; gates 21 + 18 min, contended |
| 2026-10-31 | LFM2.5-24B-A2B (lfm2moe) | faster-e2e v3 | 48 | 17 | full FAIL (compare 1 prompt, consistency near-tie) | 1.42 | 6/6 | new MoE + short-conv kernels written in the run; contended |
| 2026-10-31 | Muse-Glimmer-30B | faster-e2e v5 | 182 | 37 | full FAIL (agent reuse, sampling) | 0.35 | 6/7 | qualify 72 min (llama.cpp goldens 22 min on a 30B at ~9 tok/s); contended |
| 2026-11-01 | flint (40B MoE, sinks, SWA) | faster-e2e v7 | 69 | 7 | full FAIL (1 agent turn at gap 0.70; rule since aligned to 1.0) | 0.50 | 6/6 | ~19 min lost: no installed llama.cpp knew `flint` (fetch_llamacpp.sh since); router race fixed in-trial; uncontended GPU |
| 2026-11-01 | LFM2.5-24B-A2B | faster-e2e v8 | 32 | 2 | full: all PASS but consistency (0.085 near-tie, deterministic alone; passes under the aligned rule since) | 0.63 | 6/6 | quiet GPU; beat llama.cpp on every metric; reference from the same model |
| 2026-11-01 | flint | faster-e2e v9 | 50.6 | 8 | full FAIL: compare 1 prompt at 1.03 (llama.cpp self 1.05), agent reuse turns 1.22/2.74 | 1.03 | 6/6 | quiet GPU; fresh prompts match llama.cpp: decode-vs-prefill numerics via MoE routing most likely (flint-example README) |
| 2026-11-01 | Muse-Glimmer-30B | faster-e2e v10 | 76 | 4 | **full PASS** (agent 7/7) | 0.41 | 7/7 | quiet GPU; ~25 min were skill bugs fixed since (stale reference signature, sampling crash, raw ambiguity) |
| 2026-11-01 | Muse-Glimmer-30B | faster-e2e v11 | 70.6 | 2 | **full PASS, AUDIT PASS** | 0.41 | 7/7 | quiet GPU; within its 106-min budget |
| 2026-11-01 | Bolt-E4B | faster-e2e v12 | **52** (budget 53) | 2 | **full PASS, AUDIT PASS** | 0.35 | 6/6 | quiet GPU; prefill 1k/2k +44/+58 %, decode @2k +12 % vs llama.cpp b1360 |
| 2026-11-01 | flint (gate re-check) | faster-e2e v12 | – | – | agent 6/6 near-ties, compare all real prompts PASS | 0.50 | 6/6 | stop-token restore fix on the v9 build |
| 2026-11-01 | Qwen3.5-0.8B | faster-e2e v13 | **24.1** (budget 46) | 0 | **full PASS, AUDIT PASS** | 0.36 | 6/6 | quiet GPU; decode +27 %, prefill 1k/2k +22/+28 %, c4 +57 %, 32k prefill -20 % vs llama.cpp b1360 |
| 2026-11-01 | flint | faster-e2e v13 | **58.2** (budget 75) | 2 | **full PASS, AUDIT PASS** (agent 6/6) | 0.50 | 6/6 | quiet GPU; starter stop-token restore working from scratch; decode +25 %, c4 +14 %, prefill 1k -35 % |
| 2026-11-01 | LFM2.5-24B-A2B (consistency re-check) | faster-e2e v13 rule | – | – | consistency PASS 3/3 (deterministic alone; one near-tie at gap 0.25 when concurrent) | – | – | v8 package, current consistency.py |
| 2026-11-01 | LFM2.5-24B-A2B (rebuilt from reference) | main fc680fc | – | – | **current full gates: all PASS** (fresh, nothing reused; max gap 0.63); previous skill's gates: golden/compare/needle/sampling PASS, consistency + agent FAIL on the old rules (strict concurrent equality; no byte-fallback alignment) | 0.63 | 6/6 | answers the open LFM2.5 gate question |
| 2026-11-01 | Qwen3.5-0.8B | streamline v1 | **20.4** (budget 46) | 0 | **full PASS, AUDIT PASS** | 0.36 | 6/6 | streamlined repo (-54 % lines); decode +27 %, prefill 1k/2k +23/+27 %, c4 +42 % vs b1361; qualify's llama.cpp baseline read low (finalize re-measures) |
| 2026-11-01 | Bolt-E4B (gemma4) | streamline v1 | **43.5** (budget 53) | 2 | **full PASS, AUDIT PASS** | 0.35 | 6/6 | gemma4 example as starter.patch + chat/gemma4.loom splice; prefill 1k/2k +46/+59 %, decode @2k +13 %, c4 -4 % |
| 2026-11-01 | LFM2.5-24B-A2B | streamline v1 | **25** (budget 53) | 2 | **full PASS, AUDIT PASS** | 0.63 | 6/6 | apply_example.sh lfm2moe; beat b1361 on every metric (decode +12 %, prefill 2k/32k +27/+54 %, c4 +17 %); MOE_R=4 won here |
| 2026-11-01 | flint | streamline v1 | 68.8 (budget 75) | 3 | full: all PASS but compare (1 prompt, 2k doc, gap 1.027 vs b1361); AUDIT FAIL on it | 1.03 | 6/6 | **same divergence with main's skills and kit** (main compare.py, main-kit package, same golden: token 30, gap 1.027): a near-tie of the new reference build, not the branch. +15 min: full-context 503 at the default 56 GiB pool, rebuilt at 72 GiB; +12 min near-tie diagnosis (f32/split-K/matvec-MoE variants each match 64/64) |

## Lessons with numbers (why the skill looks the way it does)
- The default "1 h of optimization" made every end-to-end job > 1 h by construction: the budget is now
  the whole job (the model's roofline budget, `e2e_budget.py`), sized by `qualify.sh`.
- Final gates re-recorded both llama.cpp goldens (6 min for golden2 on a 0.8B): the reference cache
  made a re-run cost only the Loom side (31 -> 9.5 min on Qwen3.5-0.8B).
- A reference example for the architecture family cut the Bolt hand port from ~45 to ~5 min.
- Gate cost on a 30B dense model is dominated by llama.cpp generating ~2,400 golden tokens at 9 tok/s:
  one question per long document, 64 tokens per short prompt (halves it).
- In-trial perf under contention swung 2-3x between back-to-back runs; only interleaved or quiet-GPU
  numbers decide.
- A trial agent sat idle for 16 h waiting on its own background loop while the orchestrator waited for
  a completion message: `scripts/watch_trial.sh` exists because of it.

## Quiet-GPU evaluations (`scripts/eval.sh`), the only perf numbers to compare across revisions

Measured 2026-11-01 with `GATES=0 scripts/eval.sh` (perf only; the trials' own full gates are in the
table above), one package at a time on an idle GPU, llama.cpp b1360 (latest lemonade release), medians of
3 interleaved rounds. tok/s, Loom / llama.cpp (margin). Each package against its own GGUF (base Bolt used
UD-Q6_K_XL, the others the quant `pick_gguf.py` chose).

| Model | Skill rev | Decode | Prefill 1k | Prefill 2k | Prefill 32k |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | main ea305b0 | 223.9 / 181.4 (+23%) | 9111 / 7558 (+21%) | 9078 / 7425 (+22%) | 3566 / 4667 (-24%) |
| Qwen3.5-0.8B | faster-e2e v1 | 265.1 / 180.7 (+47%) | 9071 / 7494 (+21%) | 8991 / 7446 (+21%) | 3553 / 4606 (-23%) |
| Bolt-E4B | main ea305b0 | 40.7 / 33.9 (+20%) | 1863 / 1504 (+24%) | 1867 / 1470 (+27%) | 768 / - |
| Bolt-E4B | faster-e2e v6 | 47.0 / 40.0 (+18%) | 1855 / 1239 (+50%) | 1932 / 1214 (+59%) | 855 / - |
| LFM2.5-24B-A2B | faster-e2e v3 | 100.9 / 84.0 (+20%) | 1732 / 2000 (-13%) | 1882 / 2076 (-9%) | 1832 / 1647 (+11%) |
| flint | faster-e2e v7 | 61.0 / 46.9 (+30%) | 765 / 1212 (-37%) | 1226 / 1187 (+3%) | 1108 / 871 (+27%) |
| Muse-Glimmer-30B | faster-e2e v5 | 12.2 / 11.3 (+8%) | 388 / 335 (+16%) | 420 / 333 (+26%) | 408 / 307 (+33%) |

(Bolt 32k: the 32k document is ~35k tokens with Gemma's tokenizer, over eval.sh's 34k llama.cpp context.)
Open perf gaps worth a campaign: long prefill on Qwen3.5 (-23% at 32k: gated delta-net prefill is
sequential), short prefill on the MoE ports (LFM2.5, flint at 1k: flint's 4096 chunk regressed 1k).
