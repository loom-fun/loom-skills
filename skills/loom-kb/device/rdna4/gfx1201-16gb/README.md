# Radeon gfx1201, 16 GB class (RX 9070 XT, RX 9070)

Navi 48 (the R9700's die with 16 GB): 64 CUs, GDDR6 on a 256-bit bus, 640 GB/s peak. The programs are the R9700's (gfx12-generic, `-r9700` build);
the package files are the same on every GPU. Measured on a Radeon RX 9070 XT.

- **Selection** (gfx1201 is configured by VRAM class): arch gfx1201 and VRAM >= 30 GiB -> `radeon-r9700` (32 GB class); 14 to 30 GiB -> this node;
  below 14 GiB -> not supported yet (a 12 GB class can be added later from the `build.py` fit and one measurement).
- Serve flags (`device.json` card): `--memory_bytes=15000000000` (16 GiB board - 2 GiB), `--state_slab_bytes=4mib`, `--weight_slab_bytes=64mib`,
  and `--parameter_mode=preload` (`preload_max_weight_bytes: null`: every package that fits). Elastic backing.
- No DevLab runner: `runner.labels` is empty. Results come from a named machine and live in `fleet/devices/gfx1201-16gb.json`
  (schema `fleet/devices/device_results.schema.json`, README there): per model, whether it fits, its largest admitted context, preset
  overrides and, when measured, performance rows with their provenance. A card shows this GPU only for a model whose entry there says it fits.
- Preset: `<first>-16gb` (`preset_suffix`): the R9700 preset with `context_capacity` = the measured max context, written by
  `make_info.py build` from the data file. Card heading: "Radeon gfx1201, 16 GB (RX 9070 XT, RX 9070)".
- Roofline: `roofline.py --device gfx1201-16gb` (spec preset, not measured).

Data: `device.json`.
