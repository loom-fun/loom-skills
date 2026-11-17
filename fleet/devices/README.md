# fleet/devices/: per-device results of GPUs with no DevLab runner

One JSON file per GPU (`rx9070xt.json`, `gfx1150.json`), schema `loom-device-results/1` in `device_results.schema.json` (the same
subset of JSON Schema as `skills/loom-benchmark/scripts/card/info_schema.json`; `card_lib.validate_device_results` also checks what the schema
cannot say). The file is the only source of that GPU's data on a card: `make_info.py build` reads the entry of the card's model and
`render_card.py` shows the GPU only where the entry says the model fits. Rendering stays re-runnable from collected data (`regen.py ingest`
builds cards the same way): nothing here is measured by the renderer.

    python3 skills/loom-benchmark/scripts/card/make_info.py devices        # validate fleet/devices/*.json and summarize them (CI-safe, no GPU)

## File

| key | what |
|---|---|
| `schema` | `loom-device-results/1` |
| `device` | the GPU id (`radeon-rx9070xt`, `gfx1150`): a `skills/loom-kb/device/<family>/<id>/device.json` row (name, arch, memory, roofline, `preset_suffix`) |
| `gpu` | the board as the session found it, for readers of the file |
| `same_programs_as` | optional: a GPU whose programs and preset this one serves (RX 9070 XT: `radeon-r9700`); its preset is derived from that GPU's preset when the package has one |
| `share_section_with` | optional: `radeon-r9700` makes the card print ONE performance section headed `Radeon AI PRO R9700 / RX 9070 XT` (the R9700's table; this GPU adds no evidence, the validator refuses it). Absent/null: this GPU has its own section and its own `evidence` |
| `preset` | `{"always": true}`: every fitting model gets the GPU's preset (`chat-9070xt`: the base preset with `context_capacity` = `max_context`). Absent/false: only a model whose entry has `execution` overrides or admits less context than the base preset serves; every other model reuses the default preset (gfx1150 normally does) |
| `sdk`, `tool`, `package`, `serve` | how the results were produced (informational) |
| `provenance` | optional default source block (`kind: machine`, `device: {cpu, gpu, os}` naming the machine, `hrx_loom`) |
| `models.<slug>` | one entry per fleet model, below |

`models.<slug>` (`<slug>` = `fleet/<slug>`):

| key | what |
|---|---|
| `fits` (required) | the package loads and serves here. `false`: the card does not mention this GPU for the model; give a `reason` (one factual line, not printed) |
| `date` (required) | when `fits` and `max_context` were established |
| `max_context` | tokens per request the GPU admits (required when `fits`). Below the native context: the card says `serves up to N tokens per request` and `served_context` is recorded |
| `native_context` | informational |
| `serve_args` | the `loom-serve` flags the Quick start prints for this GPU and model, e.g. `["--memory_bytes=15000000000", "--state_slab_bytes=4mib"]`. `[]`: the package's own memory budget (like Strix Halo). Absent: the GPU's registry flags |
| `execution` | preset overrides over the base preset (`prefill_capacity`, `context_capacity`, `rows`, `draft_depth`); written into the GPU's preset together with `context_capacity` = `max_context` |
| `evidence` | when measured: `loom-card-evidence/2` documents of this GPU for this model, exactly what `make_info.py perf` / `gates` / `roofline` write, with a **`machine`** source (provenance: the named host, its device block and SDK). A performance session needs `--source-kind machine`; gates may use `machine` or `local` |
| `note` | one factual line for file readers (not printed) |

## What a card does with an entry

- `fits: true`: the card gets a `devices.<gpu>` entry (`serve_args`, `memory_bytes` from `--memory_bytes=`, `served_context`, `shares_section_with`),
  the GPU's Quick start block, its tag (`radeon-rx9070xt`, `gfx1150`) and, when the entry needs it, the preset `<first><suffix>` in `model.json`
  (`chat-9070xt`, `chat-gfx1150`). The default preset, the programs, weights and tokenizer are never touched.
- Evidence present and no `share_section_with`: a full performance section, its methodology and correctness column for that GPU, provenance
  = the named machine (the card says it is not a DevLab runner). Without evidence the card says the package serves there and the performance
  is not measured yet.
- `fits: false` or no entry for the model: the GPU does not appear on that card.

## Filling a file

    # on the named machine, after the measurements (the same perf.py / gates.sh logs a DevLab job produces):
    make_info.py perf --gpu gfx1150 --source-kind machine --device machine.json --hrx-loom hrx.json --package PKG \
        --loom-jsonl loom.jsonl --llama-jsonl llama.jsonl ... -o perf_ev.json
    make_info.py gates --gpu gfx1150 --source-kind machine ... -o gates_ev.json
    # paste the two documents into models.<slug>.evidence, set fits / date / max_context / serve_args, then:
    make_info.py devices            # validates; a card rebuild picks the entry up: regen.py ingest ..., or make_info.py build PKG --repo ...

`LOOM_DEVICE_RESULTS=DIR` (or `make_info.py build --device-results DIR`) reads another directory; `none` ignores the files (the card
tests do, so their goldens do not depend on the data).
