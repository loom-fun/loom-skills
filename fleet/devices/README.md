# fleet/devices/: per-device results of GPUs with no DevLab runner

One JSON file per GPU (`gfx1201-16gb.json`, `gfx1150.json`, `gfx1152.json`), schema `loom-device-results/1` in `device_results.schema.json` (the same
subset of JSON Schema as `skills/loom-benchmark/scripts/card/info_schema.json`; `card_lib.validate_device_results` also checks what the schema
cannot say). The file is the only source of that GPU's data on a card: `make_info.py build` reads the entry of the card's model and
`render_card.py` shows the GPU only where the entry says the model fits. Rendering stays re-runnable from collected data (`regen.py ingest`
builds cards the same way): nothing here is measured by the renderer.

    python3 skills/loom-benchmark/scripts/card/make_info.py devices        # validate fleet/devices/*.json and summarize them (CI-safe, no GPU)

For a device whose only lab backend is XSJ, the registry enables `runner.backend: "xsj"` with empty DevLab labels.
Use `kb.py route --arch <model-arch> --device <device> --topic xsj` for the lease and evidence workflow. Record
`--source-kind xsj` with the lab run ID, recorded device and SDK files, and the program digest captured on the host.

## gfx1201 is configured by VRAM class

One data file and one KB device node per memory class of the same die, not per product (owner decision, 2026-11-17):

| class | device id | VRAM (as the driver reports it) | products | preset | serve flags |
|---|---|---|---|---|---|
| 32 GB | `radeon-r9700` | >= 30 GiB | Radeon AI PRO R9700 | `chat-r9700` | `--pool_backing=fixed` + preload (0.2.0 pins only; the renderer drops it for 0.3.0) |
| 16 GB | `gfx1201-16gb` | 14 to 30 GiB | RX 9070 XT, RX 9070 | `chat-16gb` | `--memory_bytes=15000000000 --state_slab_bytes=4mib --weight_slab_bytes=64mib` + preload (0.2.0 pins only; the renderer drops it for 0.3.0) |
| below 14 GiB | not supported yet | | | | a 12 GB class can be added later from the `build.py` fit plus one measurement |

**Selection rule** (for Lemonade, Weaver and the site): arch `gfx1201` and VRAM >= 30 GiB -> the 32 GB class; 14 GiB <= VRAM < 30 GiB ->
the 16 GB class; below that -> not supported yet. The same rule is in `skills/loom-kb/device/rdna4/README.md`. The 16 GB class's file is
`gfx1201-16gb.json` (provenance names the Radeon RX 9070 XT machine that measured it). A file written before the rename
(`rx9070xt.json`, `"device": "radeon-rx9070xt"`) is still read: `card_lib.LEGACY_DEVICE_IDS` maps that id to `gfx1201-16gb`. Per-model `serve_args` in such a file are printed as they are, so drop them there: without them the class flags (registry + preload rule, preload on 0.2.0 pins only) apply.

## File

| key | what |
|---|---|
| `schema` | `loom-device-results/1` |
| `device` | the GPU id (`gfx1201-16gb`, `gfx1150`, `gfx1152`): a `skills/loom-kb/device/<family>/<id>/device.json` row (name, arch, memory, roofline, `preset_suffix`) |
| `gpu` | the board as the session found it, for readers of the file |
| `same_programs_as` | optional: a GPU whose programs and preset this one serves (the gfx1201 16 GB class: `radeon-r9700`); its preset is derived from that GPU's preset when the package has one |
| `share_section_with` | optional: `radeon-r9700` makes the card print ONE performance section headed `Radeon AI PRO R9700 / gfx1201 16 GB` (the R9700's table; this GPU adds no evidence, the validator refuses it). Absent/null: this GPU has its own section and its own `evidence` |
| `preset` | `{"always": true}`: every fitting model gets the GPU's preset (`chat-16gb`: the base preset with `context_capacity` = `max_context`). Absent/false: only a model whose entry has `execution` overrides or admits less context than the base preset serves; every other model reuses the default preset (gfx1150 normally does) |
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
| `serve_args` | the `loom-serve` flags the Quick start prints for this GPU and model, e.g. `["--memory_bytes=15000000000", "--state_slab_bytes=4mib"]`, printed as they are. `[]`: the package's own memory budget (like gfx1150). Absent: the GPU's registry flags (`device.json` card.serve_args) plus `--parameter_mode=preload` on 0.2.0 pins where the device's `card.preload_max_weight_bytes` rule says so (the gfx1201 classes: every package; hrx-loom 0.3.0 removed the flag and the renderer drops it) |
| `execution` | preset overrides over the base preset (`prefill_capacity`, `context_capacity`, `rows`, `draft_depth`); written into the GPU's preset together with `context_capacity` = `max_context` |
| `evidence` | when measured: `loom-card-evidence/2` documents from `make_info.py perf`, `gates` or `roofline`. For devices registered as XSJ-only, use a leased `xsj` source for performance sessions and gates. For other devices in this directory, performance uses a `machine` source and gates can use `machine` or `local`. |
| `note` | one factual line for file readers (not printed) |

## What a card does with an entry

- `fits: true`: the card gets a `devices.<gpu>` entry (`serve_args`, `memory_bytes` from `--memory_bytes=`, `served_context`, `shares_section_with`),
  the GPU's Quick start block, its tag (`gfx1201-16gb`, `gfx1150`) and, when the entry needs it, the preset `<first><suffix>` in `model.json`
  (`chat-16gb`, `chat-gfx1150`). The default preset, the programs, weights and tokenizer are never touched.
- Evidence present and no `share_section_with`: a full performance section, its methodology and correctness column for that GPU, provenance
  = the named machine (the card says it is not a DevLab runner). Without evidence the card says the package serves there and the performance
  is not measured yet.
- `fits: false` or no entry for the model: the GPU does not appear on that card.

## Host-dependent serve flags

`serve_args` is what the card prints after `loom-serve`, so a flag a stock host needs belongs there. gfx1150 on hrx-loom 0.2.0 needed
`--parameter_mode=preload` on a stock Ubuntu host (8 MiB locked memory): 0.2.0's default parameter mode fails at the first request
(historical; the cause and the other fixes are in `skills/loom-kb/device/rdna3/gfx1150/README.md`). hrx-loom 0.3.0 defaults to the
buffered `file` mode and the flag is dropped from Quick starts (`skills/loom-migrate/versions/v0.2.0-to-v0.3.0.md`, "Parameter loading"):
a device entry for a 0.3.0 card carries a host flag only when a stock-host run on 0.3.0 shows it is still needed.

## Filling a file

    # on the named machine, after the measurements (the same perf.py / gates.sh logs a DevLab job produces):
    make_info.py perf --gpu gfx1150 --source-kind machine --device machine.json --hrx-loom hrx.json --package PKG \
        --loom-jsonl loom.jsonl --llama-jsonl llama.jsonl ... -o perf_ev.json
    make_info.py gates --gpu gfx1150 --source-kind machine ... -o gates_ev.json
    # paste the two documents into models.<slug>.evidence, set fits / date / max_context / serve_args, then:
    make_info.py devices            # validates; a card rebuild picks the entry up: regen.py ingest ..., or make_info.py build PKG --repo ...

`LOOM_DEVICE_RESULTS=DIR` (or `make_info.py build --device-results DIR`) reads another directory; `none` ignores the files (the card
tests do, so their goldens do not depend on the data).
