# Qwen3.8-Flash-Next (arch `qwen4exp`): a catalog-package kit

`qwen4exp` (hyper-connections with 4 streams, a per-layer-embedding hash table, 36 gated delta-net layers, 12 sparse
indexed-attention layers, 512-expert MoE) is not something the starter generates. Its Loom programs are the
hand-written package in `ROCm/hrx-loom-library` (`model/Qwen/Qwen3.8-Flash-Next`). This kit carries that package, adapted so
the per-layer-embedding table (26.8 GiB, ~16 rows read per token) is read from the GGUF on demand
(`parameters.gather`) instead of being resident: ~77 GiB resident instead of 104.

| File | What it is |
|---|---|
| `catalog/catalog.json` | which weights (unsloth `UD-Q4_K_XL` at a pinned revision, the Q8_0 MTP head), which variant/preset to serve, the resident memory figures, and the SDK pin (hrx-loom 0.2.0.dev0, CI artifact; **expires**, see loom-setup/SKILL.md "SDK pins") |
| `catalog/model.json`, `catalog/program/**` | the package itself, as the catalog has it (copied from the fork; `starter/catalog.py` cuts it to one variant/preset) |
| `cfg.json` | reviewed config (nothing beyond what `inspect_gguf.py` reads) |

Use: `CATALOG=qwen4exp loom-port-llm/port.sh catalog WORKDIR 8160 auto MODEL_ID` (see loom-port-llm/SKILL.md, "Catalog packages").
Gates, perf, card, publish and nightly are the ordinary ones. Differences: no generated kernel tests (the catalog's own run
in its CI; the card's `kernel_tests` row has no evidence), drafting is the MTP head only, the Loom SDK is the pinned one,
the resident set (~77 GiB + KV) is above the default 60 %-of-RAM budget on purpose (`memory.over_cap`; serve.sh then wants
MemAvailable headroom), and the llama.cpp baseline needs a build that knows `qwen4exp` (else Loom-only gates, "unverified").
