# flint (arch `flint`) port: worked example

A port of `unsloth/flint-GGUF` (UD-Q5_K_XL, 40B MoE) written by a fresh agent from these skills in
November 2026 (69 min end to end, ~19 of them lost to a missing llama.cpp build). Built on
`../lfm2moe-example` (MoE) + the starter. Final gates: golden 24 prompts up to 31k, max gap 0.50;
consistency; sampling; needle at 130k and the full 262k (with the KV pool at 2x context); agent loop
6/6 rendering/reuse/calls (one turn diverged at gap 0.70, a near-tie under the compare rule).

| File | Adds |
|---|---|
| `gen_moe.py` | 256-expert top-6 router (bf16 router weights), shared experts, expert decode matvecs, grouped WMMA prefill GEMM. **The routing keeps the top-k weights in registers**: writing them to global memory and reading them back after a barrier made prefill nondeterministic (neighbouring tokens' workgroups share cache lines) |
| `gen_llm.py` | attention sinks (a learned per-head logit added to the softmax denominator), sliding-window (513) and NoPE layers per the GGUF pattern, MoE wiring |
| `gen_flash2.py`, `inspect_gguf.py`, `make_kernel_tests.py` | sinks in prefill flash attention; flint config extraction; MoE + sinks test cases (12/12) |
| `chat_probe.py` | the flint template: the tools block in its own system message, no message suffix (the stop marker is the end-of-text token: use the GGUF's end-of-generation ids in `extra_eos_ids`) |
| `cfg.json` | the reviewed config without the tensor table |

Notes: the tokenizer is in `RecursiveSI/flint` (the GGUF doesn't name its base model; port.sh now
searches the Hub). Prefill chunk 4096 helped 2k+ but cost 12 % at 1k: A/B the 1k size too. Build with
`--memory-gib` large enough for a pool of 2x the context, or the full-context gate gets a 503.

**Second run (Nov 2026, 50.6 min end to end, built on this example):** the agent loop's reuse turns
diverged from llama.cpp at gaps 1.22 and 2.74 while the same prompts sent fresh matched llama.cpp for 40
tokens, after a stop-token publishing bug was fixed (`chat.stop_after_calls` = `<|observation|>`,
`stop_after_answer` = `<|user|>`, now in the starter). Open, not root-caused: suspect the reuse path with
sliding window 513 + sinks (KV of the window written by decode vs prefill, or the window origin on a
resumed row). Probe: run turn N with reuse and fresh (new conversation with the same rendered prompt)
and diff the first logits. One golden prompt (2k document) sits at gap 1.03 (llama.cpp's own re-record
1.05, top-1 p 0.37).

**Diagnosis so far (orchestrator, same build):** reproduced (turns at 541 and 721 tokens diverge after
4-5 tokens; turns at 262 and 390 match). Sending the same message prefix to a freshly started server
(cache empty) matches llama.cpp for 42 tokens and fully, respectively. So the forward pass is right and
the rendering/reuse mechanics check out (byte-exact rendering, identical token counts, full cache
coverage). What differs is how the reused history's KV was produced: decoded token by token (f32 matvec
path) vs prefilled (f16 GEMM path, as llama.cpp does for a fresh prompt); with 256 experts, borderline
routing flips over ~500 decoded tokens can move a later decision. Not proven: Loom reuses only a prompt
that extends a retained prompt+completion, so "reused but prefilled" can't be forced from the API. To
settle it, build a variant whose decode routing uses the GEMM path's f16 activations (or prefill in f32)
and rerun `agent_golden.py`.
