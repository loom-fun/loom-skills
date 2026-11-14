# Family: lfm2moe

GGUF arch values: lfm2moe.
Kit example: `skills/loom-port-llm/reference/lfm2moe-example/` (kit docs stay there; model-family knowledge lives here).

## Chat side, later runs and the routing race (from the lfm2moe kit README)

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

## Fleet models (published packages, llm_config.json)

- `ggml-org-LFM2.5-24B-A2B-loom` (lfm2moe): 40L h2048, 32h/8kv hd64, moe 64e top-4, kv_layers 10, shortconv, ctx 131072
