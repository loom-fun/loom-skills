---
name: loom-kb
description: Router for the Loom knowledge base - which facts to read for a given model, GPU and task. Run `kb.py route` with a model slug or config, GGUF or arch name plus device and topic, then read exactly the files it prints (device, quant, arch family, facets, topic). Use before porting, optimizing, benchmarking or rooflining any model, when asking "what is special about gemma4 / qwen35moe / flint", "which tricks apply to this model on the R9700", or when adding a model family, facet, device or scope rule to the knowledge base.
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

Domain knowledge is stored once, in the most specific node it applies to; the workflow skills hold process only and send you here.

## Route, then read

```bash
python3 "${LOOM_SKILL_DIR}/kb.py" route --config W/pkg/llm_config.json --device strix-halo --topic prefill,decode
python3 "${LOOM_SKILL_DIR}/kb.py" route --arch gemma4 --facets drafted --device rdna4 --topic speculative
python3 "${LOOM_SKILL_DIR}/kb.py" route --arch z-image --device strix-halo --topic image-porting
python3 "${LOOM_SKILL_DIR}/kb.py" list        # valid devices, families, facets, topics
```

It prints one file per line in reading order, then a `# resolved choices` footer. Read the printed files, in that
order, and nothing else from the knowledge base (an unknown arch, device or topic fails and lists the valid values).

| Printed first to last | Holds |
|---|---|
| `device/...`, `quant/<fmt>/`, `arch/llm/...` (root, topology, family), `facet/<tag>/`, `model/<slug>/` | the selector nodes the model and device match: GPU family and device, quant formats, the one family, properties that cross families (drafted, vision, hybrid-gdn, moe, sliding-window, kv-le2, wide-head), and the one package (`model/<slug>`, only when it has its own facts or settings) |
| `<node>/README.md` | what is true of the node for every task |
| `<node>/<topic>.md` | that node's facts for one task; printed only when the topic is requested |
| `topic/<topic>.md` | the task's knowledge that applies to every model |

Use `image-porting` for image pipelines. Other topics include `porting`, `kernels-ir` (write kernels), `ir-debug`, `program-abi`, `cfg-review`, `starter` to port, `prefill`, `decode`,
`optimize-loop`, `long-context`, `concurrency`, `speculative`, `spec-tuning`, `roofline` to optimize; `sdk` (hrx-loom limits per SDK version, for kernel work); `benchmark`, `org`, `kb-authoring`.
Name only the ones the work needs: the router prints a node's task file for each topic you name and nothing else.

Adding or moving a fact, a family, a facet, a device or a topic: topic `kb-authoring` (`kb.py route --arch qwen3 --topic kb-authoring`).

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
