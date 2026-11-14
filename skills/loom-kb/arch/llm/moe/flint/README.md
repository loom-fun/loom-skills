# Family: flint

GGUF arch values: flint.
Kit example: `skills/loom-port-llm/reference/flint-example/` (kit docs stay there; model-family knowledge lives here).

## Port notes (from the flint kit): sinks, chat template, chunk size, memory, agent-gate history reuse

Flash2 sinks and the chat template (the tools block in its own system message, no message suffix: the
stop marker is the end-of-text token, so put the GGUF's end-of-generation ids in `extra_eos_ids`) are in
the starter.

Notes: the tokenizer is in `RecursiveSI/flint` (the GGUF doesn't name its base model; port.sh searches
the Hub). Prefill chunk 4096 (the MoE default) is +15-27 % at 1k-32k over the PR #25 build since the coalesced prefill router `g_moe_route_tr` and padding-free GEMMs and per-row kernels (loom-optimize catalog, "Chunk size"); A/B the 1k size too. Build with
`--memory-gib` large enough for a pool of >= 1.5x the context, or the full-context gate gets a 503 (`auto`,
the default, gives ~73 GiB on a 128 GB Strix Halo: pool ~450k for the 262k context).

**Agent-gate history reuse (root-caused, fixed in the starter):** reused turns diverged (gaps 1.22, 2.74)
while the same prompts sent fresh matched llama.cpp. The cause was the stop token, not numerics or sinks:
the head reports any end-of-turn id as the service terminal, and loom-serve feeds that prediction where
the canonical history has the real stop token (`<|observation|>` after calls, `<|user|>` after an answer),
so a reused row held the wrong token's KV. `g_embed` maps a fed terminal back (`stop_restore()`, driven by
`chat.stop_after_calls/answer`). Remaining reused-vs-fresh differences are f32-decode vs f16-prefill
numerics, all near-ties. Lesson: an earlier bisect ("sinks off -> reused == fresh") misled because that
variant produced garbage, which agrees with itself.

## Fleet models (published packages, llm_config.json)

- `ggml-org-Flint-loom` (flint): 49L h2048, 32h/4kv hd128, moe 256e top-6, swa 513, drafter dflash d3, ctx 262144
