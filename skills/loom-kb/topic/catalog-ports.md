# Topic: catalog-ports (port from a hand-written hrx-loom-library package: `CATALOG=`)

When ROCm/hrx-loom-library already has a package for an architecture the starter cannot generate (qwen4exp:
hyper-connections, per-layer-embedding hash table, sparse indexed attention), port it from that package instead of
writing a kit. `reference/qwen4exp-example` is one: `catalog/catalog.json` + the (adapted) package files.
```bash
CATALOG=qwen4exp "${LOOM_SKILL_DIR}/../loom-port-llm/port.sh" catalog ~/work/fn 8160 auto <MODEL_ID>   # or CATALOG=DIR (catalog.json + the package)
export LOOM_ROOT=$(cat ~/work/fn/loom_root)   # the pinned SDK's root, for qualify.sh / gates.sh / finalize.sh
```
Same stages and outputs as any port (`pkg/` with `model.json` cut to one variant and preset, `llm_config.json`, `cfg.json`,
`model_target.json` for the draft-off twin), so qualify, gates, perf, card, `publish_package.py prepare` and the
nightly list work unchanged. What differs (`starter/catalog.py`, `catalog.json`'s keys are documented there):
- **Weights** (repo, revision, split parts, the MTP head) and the tokenizer repo come from `catalog.json`; the package keeps
  the catalog's `@origin` names, which `prepare` preserves (it only pins the revisions).
- **SDK pin**: `catalog.json` `sdk` (hrx-loom wheel) goes to `llm_config.json`; `port.sh`, `port_e2e.sh`, the nightly and
  DevLab scripts install it with `loom-setup/scripts/ensure_sdk.sh` next to the default root (loom-setup "SDK pins").
- **Memory**: the budget is the package's *resident* set (`memory.resident_weights_bytes`, KV, recurrent rows, workspace),
  not the GGUF size; above the 60 % default it needs `memory.over_cap` (explicit, with a note) and `memcheck.sh` headroom.
  llama.cpp keeps `per_layer_token_embd` on the CPU by itself (`llamacpp_server.sh` forces nothing onto the GPU).
- **Kernel tests are the package's own** `loom_test` targets, run by `skills/loom-benchmark/scripts/kernel_tests.py` (no Bazel;
  `devlab_experiment` runs them on the GPU for catalog kits). Drafting is the MTP head or none (`tune.py` measures both); no
  reference lookup or DFlash/n-gram drafter. Vision comes from the package's vision variants (`VISION=0` keeps text only).
- **Rebuilding variants** (`devlab-perf` `build:`/`ref:` variants, `devlab-experiment`) cannot regenerate programs: use the `card` variant.
- `inspect_gguf.py` prints starter-coverage warnings for qwen4exp (arch unknown, `q5_1`, head_dim); a NOTE says they do not
  apply. Use it when the first thing needed is a package, not a kernel set; a new kit (below) only when none exists.
