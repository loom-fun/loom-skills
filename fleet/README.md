# fleet/: the 51 published models as build inputs

Dev infrastructure (not installed with the skills). A change to the kit is judged by what it does to the programs of every
published model, without a GPU and without any weights.

| file | what |
|---|---|
| `manifest.json` | per model: staging repo@revision, model id, kit example, build args (context, pool, prefill, rows, max_tokens, draft_depth), every GGUF it reads (repo@revision, size, dump) |
| `<slug>/llm_config.json` | the published config (minus `tensors`, `generator`, `draft`: outputs of a build) overlaid on `inspect_gguf.py` of the GGUF |
| `<slug>/tokenizer_min.json` | the tokenizer's added tokens (content, id): all the programs read of the tokenizer (`gen_llm.py` stop-token restore) |
| `<slug>/gguf/<file>.json.gz` + `blobs/` | GGUF header dumps (metadata, tensor infos, size, rope_freqs data); vocab arrays and chat templates in the shared content-addressed `blobs/` |
| `gguf_stub.py` | `dump` a GGUF (file or URL, HTTP range reads) / `stub` a sparse header-only file `inspect_gguf.py` and `build.py` accept |
| `add_model.py` | add or refresh a model from its package directory (reads only small files) |
| `build_fleet.py` | `--ref REF` / `--tree PATH` `--out DIR`: programs-only build of the fleet, CPU only, both targets |
| `diff_fleet.py` | `BASE_DIR HEAD_DIR --json --md`: per model changed files, blocks, GPU target, card rows, strata, model.json |
| `scope_lint.py` | `--base REF --pr-body FILE [--impact IMPACT.json]`: does a PR stay inside its `loom-scope` block (files per kind, scope leak, protected files, registry legacy values, kb lint); `--local` builds the impact itself |
| `ownership.json` | protected core files, files each PR kind may change, kernel family -> files and rows_map rule ids (data; protected) |
| `rows_map.json` | kernel family -> card rows / gates (data; first matching rule wins) |

    python fleet/build_fleet.py --ref origin/r9700 --out $WORK/base --jobs 12      # ~2 min on 12 cores
    python fleet/build_fleet.py --tree .           --out $WORK/head --jobs 12
    python fleet/diff_fleet.py $WORK/base $WORK/head --json impact.json --md impact.md

Needs `python` with `gguf` (numpy) installed (inspect_gguf.py). The manifest's inputs are the CURRENT staging packages; a ref
is built with its own kit but these inputs. Proof that stubs lose nothing: `build_fleet.py --real DIR` builds from the HF
cache GGUFs and full tokenizers instead; the programs and model.json must be byte-identical to the stub build.
Adding a model: `python fleet/add_model.py SLUG PACKAGE_DIR --staging-revision REV` (PACKAGE_DIR = snapshot of its staging repo).
