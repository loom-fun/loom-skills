# Qwen3.8-Flash-Next (arch `qwen4exp`): a catalog-package kit

`qwen4exp` (hyper-connections with 4 streams, a per-layer-embedding hash table, 36 gated delta-net layers, 12 sparse
indexed-attention layers, 512-expert MoE) is not something the starter generates. Its Loom programs are the
hand-written package in `ROCm/hrx-loom-library` (`model/Qwen/Qwen3.8-Flash-Next`). This kit carries that package, adapted so
the per-layer-embedding table (26.8 GiB, ~16 rows read per token) is read from the GGUF on demand
(`parameters.gather`) instead of being resident: ~77 GiB resident instead of 104.

| File | What it is |
|---|---|
| `catalog/catalog.json` | which weights (unsloth `UD-Q4_K_XL` at a pinned revision, the Q8_0 MTP head), which variant/preset to serve, the resident memory figures, and the SDK pin (hrx-loom 0.2.0 from PyPI with the wheel sha256; see loom-setup/SKILL.md "SDK pins") |
| `catalog/model.json`, `catalog/program/**` | the package itself: `hrx-loom-library` tag `v0.2.0` plus `library_delta.patch` (on-demand PLE), flattened by `flatten.py` (`starter/catalog.py` cuts it to one variant/preset) |
| `library_delta.patch` | our changes to the library tree at the tag: the on-demand PLE stage and gather, `text_epoch.loom` / `text_storage.loom` (copies of the library's `program/text/qwen3_8/epoch.loom` and `storage.loom` with the gather), the thinking-default line in `chat.loom`, the PLE-residency variants/presets |
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
Re-doing a conflicting hunk: `@flash_next_ple_residual` gained `%staging` last (launch args `%capture, %residual`), the library's own
parameters stay first; `text_epoch.loom`/`text_storage.loom` are the library's `epoch.loom`/`storage.loom` with the gather hunks
(`diff -u` of the old pair gives them). v0.2.0rc1 (and v0.2.0, which changes no file the catalog uses) needed exactly those rewrites (the library added `%workload_count` to the
PLE residual command and `feedback.loom` to the storage chain); the gather (`parameters.gather`) code path is unchanged.
Check: `fleet/build_fleet.py --tree . --slugs unsloth-Qwen3.8-Flash-Next-loom`, then `loom-link` of **each input** with the
service's config (the joint link of `fleet/link_check.py` also fails on the unmodified library package: config key
`qwen38.packet_capacity` and the function of the same name collide when the three inputs are linked together).
