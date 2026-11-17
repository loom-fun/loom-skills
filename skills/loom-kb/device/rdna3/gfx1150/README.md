# gfx1150 (Radeon 890M, Strix Point)

RDNA 3.5 APU, 16 CUs, shared LPDDR5X (120 GB/s peak). Family `rdna3` (gfx11-generic): the programs are the base target's.

- No DevLab runner: `runner.labels` is empty. Results come from a named local machine and live in `fleet/devices/gfx1150.json`
  (schema `fleet/devices/device_results.schema.json`, README there). A card shows this GPU only for a model whose entry there has data.
- Preset: reuses the default preset; `<first>-gfx1150` (`preset_suffix`) is written only when the data needs other serving settings.
- Roofline: `roofline.py --device gfx1150` (spec preset, not measured).

Data: `device.json`.
