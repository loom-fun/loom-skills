# Topic: reference (start from a published loom-fun package, category A only)

Every port and every optimization starts by looking for a published package that already solved most of it
(port.sh does this itself; by hand: `find_reference.py rank GGUF --repo ORG/REPO -o W/reference.json`).
- **Only category A counts**: a card made by the current skill, `info.json` with `"schema": "loom-card/2"` and
  `"report": "generated"`. Imported cards (`"report": "attached"`, old factory runs) are listed as skipped, never used.
- **Ranking:** same architecture and family (GGUF name, base-model chain), then same architecture, then shared
  building blocks (gated delta-net hybrid, MoE routing, vision tower, tool-call format, thinking switch, drafter
  type), then nearest size. The `generator` stamp of its `llm_config.json` says how stale it is.
- **What is taken:** the reference's example kit (when it was built from one: `reference/examples.json` maps the
  packages to `*-example` kits; with no category A package of the architecture, the example kit of that architecture alone); the cfg keys `inspect_gguf.py` left open (geometry, formats and eos ids always stay
  this GGUF's); its chat block when this template's derivation fails, accepted only if `chat_probe.py --verify`
  passes on THIS template; and its measured drafting table, from which `draft/tune.py` starts each method at the
  reference's best depth and climbs from there, and gives a method the reference ruled out (under 75 % of its
  winner) one probe (`TUNE_PRIOR=0` measures all). The llama.cpp baseline's own tuning never takes a prior.
- **Where it pays:** an architecture the starter doesn't cover (the reference's kit is applied automatically; its
  chat block replaces a failed derivation when it verifies). On the starter path (Qwen3.5 sizes) port.sh is already
  automatic: the reference trims the drafting search and pre-answers the cfg review (only geometry differs).
- **What is recorded:** `cfg["reference"]` (repo, revision, score, reasons, what was taken, what differs), which
  `build.py` copies into the package's `llm_config.json`.
- A reference saves time; it never lowers the bar: the derived port runs the same kernel tests and gates.
- `find_reference.py fetch REPO DIR` gets its package (no weights) to read or diff; `show` prints what an
  optimization should start from.

## Optimizing from the reference

Before measuring anything, read what the closest category A loom-fun package already
settled (`loom-port-llm/find_reference.py show WORK/reference.json`, written by port.sh; or `show loom-fun/REPO`):
its drafting choice and measured table, execution settings (prefill chunk), llama.cpp baseline arguments, margins
per metric and gates. Try its kept settings first and skip A/Bs it already decided for the same architecture; its
choices are a starting point to re-measure on this model, never a result to copy onto the card.
