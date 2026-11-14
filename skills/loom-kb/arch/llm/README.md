# LLM ports: facts that hold for every text model

Read first for any LLM port; the topology node (dense, hybrid, moe), the family node and the facets follow.

- A port is a GGUF -> `cfg.json` -> `llm.loom` (kernels + the static per-layer `@llm_advance` program) -> package
  (`model.json`, tokenizer, `llm_config.json`). The starter (`loom-port-llm/starter/`) generates it for dense
  transformers, gated-delta-net hybrids and MoE; a new architecture adds kernels beside it (topic `porting`).
- Decode is weight-bandwidth bound (read every weight byte once per token group); prefill is WMMA compute and
  attention bound (topics `decode`, `prefill`); long contexts and several rows change the balance (topics
  `long-context`, `concurrency`). Greedy decoding through different summation orders diverges at near-ties
  only; a port is judged by the golden/consistency/agent gates (topic `benchmark`).
- Each model is checked against llama.cpp's graph builder for its arch (`src/models/<arch>.cpp`): the cfg review
  checklist is topic `cfg-review`.
- Where facts live: one family only -> its family node; a property shared across families (MoE, drafter, vision,
  sliding window, hybrid, <= 2 KV heads) -> the facet; a task -> the topic. `kb.py route` prints exactly the files for
  a model, device and task.
