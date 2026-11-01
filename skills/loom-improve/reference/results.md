# Trial results: the baselines to beat

One row per end-to-end trial (fresh agent, `trial_prompt.md`, Strix Halo gfx1151). Append new rows;
never edit old ones. Wall clock is install -> done. "Contended" = another trial shared the GPU
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

## Lessons with numbers (why the skill looks the way it does)
- The default "1 h of optimization" made every end-to-end job > 1 h by construction: the budget is now
  the whole job (60 min), sized by `qualify.sh`.
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
