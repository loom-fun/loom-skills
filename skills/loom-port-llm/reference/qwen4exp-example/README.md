# Qwen3.8-Flash-Next (arch `qwen4exp`): a catalog-package kit

`qwen4exp` (hyper-connections with 4 streams, a per-layer-embedding hash table, 36 gated delta-net layers, 12 sparse
indexed-attention layers, 512-expert MoE) is not something the starter generates. Its Loom programs are the
hand-written package in `ROCm/hrx-loom-library` (`model/Qwen/Qwen3.8-Flash-Next`). Since the library's v0.3.0 package
the per-layer-embedding table (26.8 GiB, ~16 rows read per token) stays in the GGUF and is read on demand into a bounded
row cache by the package itself (~77 GiB resident instead of 104), and the text controller is service version 2 (live
token and span counts, explicit HAL ownership).

| File | What it is |
|---|---|
| `catalog/catalog.json` | which weights (unsloth `UD-Q4_K_XL` at a pinned revision, the Q8_0 MTP head), which variant/preset to serve, the resident memory figures, and the SDK pin (hrx-loom 0.3.0 from PyPI with the wheel sha256; see loom-setup/SKILL.md "SDK pins") |
| `catalog/model.json`, `catalog/program/**` | the package itself: `hrx-loom-library` tag `v0.3.0` plus `library_delta.patch`, flattened by `flatten.py` (`starter/catalog.py` cuts it to one variant/preset) |
| `library_delta.patch` | our changes to the library tree at the tag: the thinking-on default of `chat.loom` (the Qwen3.8 template starts with thinking on; only `enable_thinking=false` turns it off), plus the measured prefill changes of #1417 |
| `flatten.py` | library tree -> self-contained `catalog/` (shared files below `program/shared/`, source programs only) |
| `cfg.json` | reviewed config (nothing beyond what `inspect_gguf.py` reads) |

Use: `CATALOG=qwen4exp loom-port-llm/port.sh catalog WORKDIR 8160 auto MODEL_ID` (see the loom-kb topic `catalog-ports`).
Gates, perf, card, publish and nightly are the ordinary ones. Differences: no generated kernel tests (the catalog's own run
in its CI; the card's `kernel_tests` row has no evidence), drafting is the MTP head only, the Loom SDK is the pinned one,
the resident set (~77 GiB + KV) is above the default 60 %-of-RAM budget on purpose (`memory.over_cap`; serve.sh then wants
MemAvailable headroom), and the llama.cpp baseline needs a build that knows `qwen4exp` (else Loom-only gates, "unverified").

## Refreshing from a library release

```bash
git clone https://github.com/ROCm/hrx-loom-library L && cd L && git checkout <tag>
git apply --index /path/to/library_delta.patch      # conflicts: re-do the hunks on the new tree, then regenerate the patch
python3 flatten.py . /path/to/catalog               # keeps catalog.json beside it
git diff --cached <tag> --binary > library_delta.patch
```
From v0.2.0 to v0.3.0 the library took over the on-demand PLE path (`ple_cache*.loom`, `ple_control.loom`), so the old gather hunks
(`text_epoch.loom`, `text_storage.loom`, the PLE stage) were dropped rather than re-done; only the `chat.loom` thinking default carried over.
Check: `fleet/build_fleet.py --tree . --slugs unsloth-Qwen3.8-Flash-Next-loom` and `loom-opt` of each changed file (parse and types). The
offline joint link (`fleet/link_check.py`, or `loom-link` of the inputs) is not a verdict for this package: the C++ input
(`publish.cxx`) needs the wheel's importer and the separately linked programs share symbol names (`@request_storage`,
`qwen38.packet_capacity`); a DevLab serve is the link check.
