# LFM2 MoE hybrid (arch `lfm2moe`) port: worked example

The kernel side of a port of `LiquidAI/LFM2.5-24B-A2B-GGUF` (Q4_K_M), written by a fresh agent from
these skills in October 2026, in about 48 minutes end to end. LFM2.5 mixes **short-conv layers**
(gated causal conv, kernel 3, recurrent state in the runtime's recurrent slots) with GQA attention
(head_dim 64) every few layers, and a **top-4-of-64 MoE** FFN.

| File | What it adds over the starter |
|---|---|
| `gen_sconv.py` | Short-conv layers (in-proj -> B*x gate -> causal conv over the conv state -> C* gate -> out-proj), reusing the recurrent-slot protocol of `gen_gdn.py` (read the fork position's slot, publish the final state) |
| `gen_moe_lfm.py` | Router + top-k (sigmoid scores, normalized), per-expert decode matvecs (`MOE_R` rows per workgroup: 8 won by +4.5 %), and a grouped 64x64 WMMA prefill GEMM over tokens bucketed by expert (prefill 330 -> ~1,900 tok/s) |
| `gen_llm.py` | Wiring: layer kinds from the GGUF (`shortconv_layers`, attention layers), head_dim 64 accepted by the attention and flash2 generators |
| `inspect_gguf.py`, `make_kernel_tests.py`, `cfg.json` | Config extraction for `lfm2moe`, kernel tests for the short-conv chain and the MoE chain (13/13) |

The chat side is in the starter: `chat_probe.py` derives LFM2's template (`{% generation %}` tags, a
JSON-list tools block, pythonic tool calls `<|tool_call_start|>[f(a='x')]<|tool_call_end|>`), and
`control.loom` parses those calls.

Gates on that build: golden 27/28 prompts near-ties; agent tool loop 6/6; sampling; needle at 61k and
the full 131k. Open at the time (both in near-ties of the MoE router, neither proven harmless):
- one prompt diverged at gap 1.4 (llama.cpp itself moved 0.44 nats there between runs);
- consistency failed intermittently (gap 0.33): a decode token sharing an epoch with a prefill takes
  the f16 GEMM path, and the difference flips a router near-tie.
Start there if you port an LFM2 MoE: give the router logits an f32 path in mixed epochs.
