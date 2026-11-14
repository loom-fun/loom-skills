---
name: loom-kb
description: Router for the Loom knowledge base - which facts to read for a given model, GPU and task. Run `kb.py route` with a model config, GGUF or arch name plus device and topic, then read exactly the files it prints (device, quant, arch family, facets, topic). Use before porting, optimizing, benchmarking or rooflining any model, when asking "what is special about gemma4 / qwen35moe / flint", "which tricks apply to this model on the R9700", or when adding a model family, facet, device or scope rule to the knowledge base.
---

# Loom knowledge base (router)

**Bundled paths ? first shell action.** Before running commands, set `LOOM_SKILL_DIR` in the shell to the
absolute directory containing **this** `SKILL.md`, using its path from the skill listing or read tool.
In Claude Code, run `export LOOM_SKILL_DIR="${CLAUDE_SKILL_DIR}"` (the host substitutes the path).
In Codex, Pi or another host, export `LOOM_SKILL_DIR` to the loaded file's directory. Use a
shell-compatible absolute path (Git Bash on Windows). Rebind before using another skill's commands
and in every new shell; it is not the working directory or SDK root. On Windows, use native `C:/...`
paths for file tools; `/c/...` paths belong to Git Bash. Never run an unresolved placeholder. Run bundled commands by quoted absolute path; do not run `scripts/...` from the working directory
or change into the installed skill tree. Install the complete Loom suite: sibling paths are required.

Domain knowledge (what is special about a family, which optimization applies where, measured results) is stored
once, in the most specific node it applies to. The workflow skills (`loom-port-llm`, `loom-optimize`,
`loom-benchmark`, `loom-roofline`, `loom-fleet`, ...) hold process and checklists only and send you here.

## Route, then read

```bash
python3 "${LOOM_SKILL_DIR}/kb.py" route --config W/pkg/llm_config.json --device strix-halo --topic prefill,decode
python3 "${LOOM_SKILL_DIR}/kb.py" route --gguf model.gguf --device radeon-r9700 --topic porting
python3 "${LOOM_SKILL_DIR}/kb.py" route --arch gemma4 --facets drafted --device rdna4 --topic speculative
python3 "${LOOM_SKILL_DIR}/kb.py" list        # valid devices, families, facets, topics
```

It prints one file per line in reading order, then a `# resolved choices` footer. Read the printed files, in that
order, and nothing else from the knowledge base (an unknown arch, device or topic fails and lists the valid values).

| Printed first to last | Holds |
|---|---|
| `device/<family>/README.md`, `device/<family>/<device>/README.md` | GPU family and device facts: matrix ISA, memory, runners, calibration |
| `quant/<fmt>/README.md` | what a quant format needs (kernels, dequant, support matrix) |
| `arch/llm/README.md`, `arch/llm/<topology>/README.md`, `arch/llm/<topology>/<family>/README.md` | every LLM; dense / hybrid / moe; the one family (GGUF arch) |
| `facet/<tag>/README.md` | properties that cross families: drafted, vision, hybrid-gdn, moe, sliding-window, kv-le2 |
| `topic/<topic>.md` | task knowledge: prefill, decode, long-context, concurrency, speculative, spec-tuning, porting, kernels-ir, benchmark, roofline, org |

Topics are tasks, not model properties: name the ones the work needs (`--topic prefill,decode` to optimize,
`--topic porting,kernels-ir` to port). Selector nodes (device, quant, arch, facet) are matched from the model.

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
- `choices` is empty unless the node sets model-affecting choices (see `choices.py`). A new choice is opt-in by
  scope: its legacy value is today's behaviour and only the models the node matches change.
- Add a family: create `arch/llm/<topology>/<family>/{README.md,scope.json}`. Add a facet: `facet/<tag>/`.
  Add a topic: `topic/<name>.md`. Then run `python3 "${LOOM_SKILL_DIR}/kb.py" lint` (also in CI).
- `kb.py lint` checks: SKILL.md <= 500 lines and frontmatter limits for every skill, node size and TOC, links,
  no node-to-node links, no duplicated paragraph (>= 200 chars; intentional ones in `lint_allow.json`), and that every
  fleet arch maps to one family node.

## Mandate rules (every workflow that changes skills, kernels or cards)

These apply to `loom-port-llm`, `loom-optimize` and `loom-fleet`; they link here instead of copying.

1. **Declare the scope first**: the PR body carries a `loom-scope` block (kind, scopes, kernels; DECISIONS section 6)
   before any edit. Work only inside it.
2. **Opt in by scope**: a new optimization or setting applies to the measured scope (family, facet, device) and nothing
   wider; a model, GPU or quant it was not measured on keeps its old behaviour.
3. **Run `fleet/scope_lint.py` before opening the PR** and fix what it reports; a PR that reaches other models than its
   scope is split or declares `leak:` with the reason.
4. **Stop on out-of-mandate needs**: a change to a protected core file (generator core, `dual_target.py`, `build.py`,
   `kb.py`, `choices.py` legacy values, methodology, registries, card schema, workflows) is an `architecture` PR, not part
   of a targeted task. Report the need and stop; do not work around it.

The full process around these rules: `loom-fleet/reference/targeted-fix.md` (a narrow change, checklist) and
`loom-fleet/reference/campaign.md` (a team campaign); why the guards exist: `loom-fleet/reference/architecture.md`.
