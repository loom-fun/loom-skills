# Radeon RX 7900 XTX (gfx1100, 24 GB)

Navi 31, 96 CUs, GDDR6 on a 384-bit bus: 960 GB/s peak, 123 TFLOP/s dense f16 WMMA. **All values are AMD spec, none measured**:
no DevLab runner is known for this device, so `runner.labels` is empty and no number of this device may appear on a card as measured.

- Family: `rdna3` (gfx11-generic): the programs are the base target's, byte-identical to the Strix Halo ones. This row adds no kernels.
- Pool: `memory_bytes` 24e9 is a conservative stand-in for what loom-serve would get; the fit table (`fit_table`) is not written yet:
  the fit tooling is still R9700-specific (see `loom-fleet/reference/campaign.md`, "Velocity").
- Preset: `<first>-rx7900xtx` (`preset_suffix`), only created when a package needs serving settings that differ from the default.
- Roofline: `roofline.py --device radeon-rx7900xtx` (spec preset; replace the bandwidth with a measured value once a runner exists).

Data: `device.json`.
