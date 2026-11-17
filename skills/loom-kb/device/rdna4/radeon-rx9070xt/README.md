# Radeon RX 9070 XT (gfx1201, 16 GB)

Navi 48 (the R9700's die with 16 GB): 64 CUs, GDDR6 on a 256-bit bus, 640 GB/s peak. The programs are the R9700's (gfx12-generic, `-r9700` build);
the package files are the same on every GPU.

- No DevLab runner: `runner.labels` is empty. Results come from a named machine and live in `fleet/devices/rx9070xt.json`
  (schema `fleet/devices/device_results.schema.json`, README there): per model, whether it fits, its largest admitted context, the
  serve flags, preset overrides and, when measured, performance rows with their provenance. A card shows this GPU only for a model whose
  entry there says it fits.
- Preset: `<first>-9070xt` (`preset_suffix`): the R9700 preset with `context_capacity` = the measured max context, written by
  `make_info.py build` from the data file.
- Roofline: `roofline.py --device radeon-rx9070xt` (spec preset, not measured).

Data: `device.json`.
