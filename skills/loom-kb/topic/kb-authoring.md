# Topic: kb-authoring (where a fact goes, node layout, scope rules, lint)

## Where a fact goes

1. One model family only: `arch/llm/<topology>/<family>/README.md`.
2. One GPU or GPU family: `device/...`. One quant format: `quant/...`.
3. Every model with a property (MoE, drafter, vision, sliding window, hybrid, few KV heads): the facet.
4. Everything else about a task: the topic file. Never copy a fact into two nodes; the duplicate-paragraph lint fails.
5. Keep the number and its evidence (PR, run, date) with the fact. Node README <= 300 lines (TOC above 100), no
   links between nodes (the router is the table of contents), every relative link resolves.

## Nodes, owners and scope rules

- A selector node has `scope.json`: `{"scope": <its path>, "match": {...}, "owner": "...", "facets": [...], "choices": []}`.
  `match` keys: `arch` (cfg arch / GGUF architecture names), `has` / `lacks` (cfg keys), `kv_heads`, `head_dim`,
  `sliding_window`, `layers` as `[lo, hi]`, `recurrent_kv`. Each arch value belongs to exactly one family node.
- A package has its own node when it needs facts or settings no family or facet can express: `model/<slug>/{README.md,scope.json}`, `match`
  `{"slug": [...]}` or `{"model_id": [...]}`; shape keys (`hidden`, `ffn`, `vocab`, `params_b`, ...) select a subset of a family instead
  (loom-fleet reference `targeted-fix.md`, section "One model, one device").
- `choices` is empty unless the node sets model-affecting choices (see `choices.py`). A new choice is opt-in by
  scope: its legacy value is today's behaviour and only the models the node matches change.
- A rule for one GPU family: `"when": {"devices": ["rdna3"]}` on a choice entry. Only a key declared `per_target=True` in
  `choices.py` may differ between the gfx11 kernel and its gfx12 twin in one build (a twin `K` / `K_rdna4`); the resolved value
  is then `{"rdna3": false, "rdna4": true}` and the generator of the twin reads `<ENV>__RDNA4` (`flash2_rdna4.twin_env`).
  Any other key needs a rule that covers every device it changes, else the resolver errors. Example: `arch/llm/moe/qwen3moe`.
  A per_target key of a kernel that is shared today (no twin yet) works the same once its generator emits the twin where the two
  values differ and nothing otherwise (`decode.attn.kv_once.d256` / `gen_llm.attention_twin`; `arch/llm/moe/qwen35moe` sets it for every drafted model; Qwen3.6-35B-A3B carried a per-GPU twin under SDK 0.2.0 and no longer does, `model/ggml-org-Qwen3.6-35B-A3B-loom/README.md`).
- Add a family: create `arch/llm/<topology>/<family>/{README.md,scope.json}`. Add a facet: `facet/<tag>/`.
  Add a topic: `topic/<name>.md`. Then run `python3 "${LOOM_SKILL_DIR}/kb.py" lint` (also in CI).
- `kb.py lint` checks: SKILL.md <= 500 lines and frontmatter limits for every skill, node size and TOC, links,
  no node-to-node links, no duplicated paragraph (>= 200 chars; intentional ones in `lint_allow.json`), and that every
  fleet arch maps to one family node.

## Where facts live (summary)

- Where facts live: one family only -> its family node; a property shared across families (MoE, drafter, vision,
  sliding window, hybrid, <= 2 KV heads) -> the facet; a task -> the topic. `kb.py route` prints exactly the files for
  a model, device and task.
