# LFM2 MoE hybrid (arch `lfm2moe`) port: worked example

The kernel side of a port of `LiquidAI/LFM2.5-24B-A2B-GGUF` (Q4_K_M), written by a fresh agent from
these skills in October 2026, in about 48 minutes end to end. LFM2.5 mixes **short-conv layers**
(gated causal conv, kernel 3, recurrent state in the runtime's recurrent slots) with GQA attention
(head_dim 64) every few layers, and a **top-4-of-64 MoE** FFN.

Make a kit with `../apply_example.sh lfm2moe W/kit`.

| File | What it adds over the starter |
|---|---|
| `gen_sconv.py` | Short-conv layers (in-proj -> B*x gate -> causal conv over the conv state -> C* gate -> out-proj), reusing the recurrent-slot protocol of `gen_gdn.py` (read the fork position's slot, publish the final state; the slot sizing is in the starter's `build.py`) |
| `gen_moe_lfm.py` | Router + top-k (sigmoid scores, normalized; the whole wave picks the top-k: decode +3.9 %), per-expert decode matvecs (`MOE_R` rows per workgroup: 8 won), and a grouped 64x64 WMMA prefill GEMM over tokens bucketed by expert, output tiles fastest (`MOE_N_FASTEST`: prefill +24-30 %) |
| `starter.patch` | `gen_llm.py`: wiring (layer kinds from the GGUF: `shortconv_layers`, attention layers; head_dim 64), `GEMM_MIN_TOKENS`. `inspect_gguf.py`, `make_kernel_tests.py`: config extraction for `lfm2moe`, kernel tests for the short-conv and MoE chains (13/13) |
| `cfg.json` | the reviewed config without the tensor table |

The chat side is in the starter: `chat_probe.py` derives LFM2's template (`{% generation %}` tags, a
JSON-list tools block, pythonic tool calls `<|tool_call_start|>[f(a='x')]<|tool_call_end|>`), and
`control.loom` parses those calls.

A second fresh-agent run (Nov 2026, 32 min end to end) built on this example and beat llama.cpp b1361 on
every metric (decode +7 %, prefill 1k/2k/32k +8/+27/+55 %, 4 streams +18 %; prefill chunk 2048). Rebuilt
from this example, the current full gates all PASS (max gap 0.63, agent loop 6/6, consistency 3/3).

Routing must keep the top-k weights in registers: a global write + workgroup barrier + re-read races
through the per-WGP cache when neighbouring tokens' workgroups share a cache line (a flint port built on
this file saw run-to-run different prefill output). Test: one >= 64-token prompt sent alone 4 times must
give identical greedy text.
