# Architecture of the skill repo: layout, registries, testing, guards

Explanation (why the repo is built this way). Process lives in [targeted-fix.md](targeted-fix.md) and [campaign.md](campaign.md); the
facts live in `skills/loom-kb`. Status when written: the layout, router, fleet tools, sampler, choice registry and the device/quant/methodology registries are on
the integration branch; the guards (`fleet/scope_lint.py`, `fleet/ownership.json`, `.github/workflows/impact.yml`) and the
device-generic A/B job generator (`.github/scripts/ab_jobs.py`) land from their own PRs, and this file names them where they matter.

## Contents
- Goals and the invariant
- Layout: where things live
- Axes: selectors and topics
- Choice registry and resolver
- Fleet manifest, programs-only build, program diff (T0)
- Sampled testing and promotion (T1-T3, the ratchet)
- Registries: devices, quants, methodology
- Guards
- Deliberate behaviour changes
- Deferred, and why

## Goals and the invariant

The structure serves six goals: separate concerns by GPU, quant, model type, topology (MoE vs dense) and family so a targeted
improvement does not touch every model; confidence from sampled tests instead of full-fleet runs; scale to every AMD GPU, model and
quant; efficient skill runs (an agent loads only the instructions that match its model); each campaign faster than the last; and
guards that keep agents doing targeted work from degrading the architecture.

**Invariant**: before any deliberate behaviour change, the fleet program diff between the working branch and its base is empty.
Every deliberate change is a scope rule, shown by the diff to touch exactly the models it declares, then A/B'd on DevLab.

## Layout: where things live

```
fleet/                         dev infrastructure (not installed with the skills)
  manifest.json                the published models: staging repo@revision, build args, GGUFs it reads
  <slug>/ llm_config.json, gguf/*.json.gz (header dumps), tokenizer_min.json     build inputs without any weights
  build_fleet.py  diff_fleet.py  sample_plan.py  verdict.py  promote_check.py  baselines.json   see fleet/README.md
  scope_lint.py  ownership.json                                                  the guards
skills/loom-kb/                the knowledge base: ONE skill, a router + selector/topic nodes (installed with the suite)
  SKILL.md  kb.py (route, lint, list)  resolve.py  registry.py  choices.py
  device/<family>/README.md + family.json;  device/<family>/<device>/README.md + device.json      rdna3, rdna4 (+ placeholders)
  quant/<fmt>/README.md + format.json;  arch/<modality>/..., arch/llm/<topology>/<family>/README.md + scope.json
  facet/<tag>/README.md + scope.json;  topic/<topic>.md;  method/methodology.json + methodology.md
skills/loom-*/                 workflow skills: process and checklists only; they say "run kb.py route and read what it prints"
```
Depth exists for ownership and selection; agents never navigate it, `kb.py route` prints the list. Keep the router skill flat (nodes one
hop away, SKILL.md under 500 lines, nodes under 300 lines with a table of contents above 100): `kb.py lint` enforces it.

## Axes: selectors and topics

Selectors say which model and device the work is about: `device`, `quant`, `arch` (lineage tree by GGUF architecture), `facet` (tags
that cross families: drafted, vision, hybrid-gdn, sliding-window, kv-le2, moe). Topics say which task: prefill, decode, long-context,
concurrency, speculative, kernels-ir, porting ... Phase is not a property of a model, so topics are flat files selected by the task
(`--topic`), while only selector nodes carry `scope.json` and choices. The matrix stays two-dimensional (model x device).

## Choice registry and resolver

- `choices.py` is the one registry of model-affecting knobs: `CHOICE(key, type, legacy, env, kernels, rows, devices, from_pr, doc)`.
  `legacy` is the behaviour before the registry existed, so introducing it changes no program. Tool-only variables (paths, test
  selectors) are listed with a reason in the not-choices list so the inventory is complete.
- A selector node's `scope.json`: `{"scope", "match": {...}, "owner", "choices": [{"when", "set", "evidence"}]}`. Match keys: arch,
  facets, kv_heads, head_dim, sliding_window, layers, quant formats, device/family.
- `resolve.resolve(cfg, devices)`: later wins, in the order registry legacy < device family < device < quant < arch root->leaf <
  facet < model (`llm_config.json["choices"]`). Two nodes of one level setting one key for one model is a lint error.
- Transport: `build.py` resolves, exports the registered env vars to the generators (they are unchanged), and writes
  `llm_config.json["resolved"]` (scopes, choices with their origin, devices) into the package. An environment variable set by hand wins
  but marks the build dirty and publishing refuses dirty builds.
- **New choices are opt-in by scope**: lint fails a PR whose registry diff changes a legacy value without kind `architecture`.
- A key declared for specific devices may only be set there. A key that changes a kernel shared by both targets of one dual-target
  package cannot differ per device inside it; it holds for the whole package or is a twin kernel (`K` / `K_rdna4`).

## Fleet manifest, programs-only build, program diff (T0)

The published models are build inputs without weights: configs from the packages, GGUF headers (committed, small) and the tokenizer
parts the programs read. `build_fleet.py` builds every package's programs + `model.json` at any git ref on CPU in minutes, and a
`--real` mode proves it byte-identical to a build from the real files. `diff_fleet.py` reports per model the changed files, kernels,
GPU target (gfx11 / gfx12 / both from the twin suffix and target tags), card rows (`rows_map.json`), and a stratum signature (arch
family, head_dim, query-heads-per-KV-head, KV bucket, draft kind, formats, dense/MoE, GPU family). T0 = this diff plus unit tests; it is
authoritative for what a change touches and runs on every PR (`impact.yml`).

## Sampled testing and promotion (T1-T3, the ratchet)

| Tier | When | What |
|---|---|---|
| T0 static | every PR | tests, fleet build at base and head, impact report, scope lint; CPU only |
| T1 sampled canary | PRs with a non-empty impact | `sample_plan.py`: strata of the touched models -> representatives per GPU family; A/B base vs head of the affected rows only (ABBA, 2+2 sessions, 6+6 near a threshold), text match, kernel tests |
| T2 post-merge confirm | after merge | the touched models (or two per stratum), the full protocol on the touched rows; the rule "outside its band in the same direction in two runs" |
| T3 rotation | a daily slice, the whole fleet weekly | head build vs published package, every model, every GPU, all rows; also catches changes outside the generator (SDK, serving, presets, drivers) |

`verdict.py` judges each row alone against `methodology.json` (band, T1 reject threshold, sessions by speed class, bimodal models
compared within a mode or flagged); the verdict of a stratum is its worst member; no geomean trading. The **promotion ratchet**
(`promote_check.py`): a (model, GPU) republishes only when every row >= published - band, gates pass and the text matches or is a
near-tie; a better head build stays unpublished (an improvement is recorded, not trusted) until a second run confirms it; a worse one
leaves the published package serving and gets a regression ticket. `fleet/baselines.json` stores the promoted value and run link per
(model, GPU, row), which also catches slow cumulative drift. The sampler was validated by replaying a past A/B: the representative's
verdict applied to its stratum is compared with every member's full verdict (agreement, false pass, false alarm, runner-hours).

## Registries: devices, quants, methodology

- Devices: `device/<family>/family.json` (generic target, twin suffix, wave size, matrix capability) and `<device>/device.json` (gfx id,
  names, memory, bandwidth, flops, runner labels, serve args, card fields, fit rows, roofline presets). Readers: the card library,
  the roofline, the dual-target pass, `build.py` presets, the A/B job generators. A new GPU of a known family is a registry row and a
  runner; behaviour-preserving, byte-identical programs and cards.
- Quants: `quant/<fmt>/format.json` is the support matrix; `formats.py` stays the code constants and a test checks they agree.
- Methodology: `method/methodology.json` holds rows, per-row bands, thresholds, sessions, bimodal models, gate sets and baselines per
  device; `methodology.md` is checked against it; readers: the compare and A/B tools, the draft tuner, the roofline, `make_info.py`.

## Guards

Every PR body carries a `loom-scope` block (kind, scopes, kernels, optional `leak:`). `fleet/scope_lint.py` (CI and locally with
`--local`) fails when:
1. changed files are outside what the kind allows (targeted: its nodes' directories, `scope.json`, tests; docs: markdown only;
   card: card tooling and cards; tooling: scripts, CI, tests, fleet tools, no starter or kb data; architecture: any);
2. the fleet diff touches a model the declared scopes do not match (a scope leak) without `leak:` on a horizontal or architecture PR;
3. a protected core file (`fleet/ownership.json`: generator core, `dual_target.py`, `build.py`, `kb.py`, `choices.py` legacy values,
   methodology, device and quant registries, card schema, workflows, the lint itself) changes without kind `architecture`;
4. a registry legacy value or key changes, or a new choice is not legacy-default;
5. `kb.py lint` fails (sizes, frontmatter, links, no ref chains, no duplicated paragraphs, every fleet arch has one family node).
Workflow skills carry the mandate: declare the scope first, work only inside it, opt in by scope, run the lint before the PR, and
anything else is a stop-and-report (`skills/loom-kb/SKILL.md`, "Mandate rules").

## Deliberate behaviour changes

Each is a scope rule shown by the fleet diff and measured by A/B. Example: the drafted qwen35moe decode_32k regression is fixed by a
rule at `arch/llm/moe/qwen35moe` + facet `drafted` that turns the decode K/V-once body for head_dim 256 off, not by a global
revert; the device scope follows the A/B because the key is not per-target. Per-row bands replace one band for the A/B compare.

## Deferred, and why

- Splitting `gen_llm.py` by topology and replacing the regex twin generators with capability emitters: high risk and no payoff for two
  GPU families while the registry, fleet diff and scope lint already bound the blast radius. Prepared: the dual-target pass reads its
  targets from the device registry. Revisit before CDNA or XDNA.
