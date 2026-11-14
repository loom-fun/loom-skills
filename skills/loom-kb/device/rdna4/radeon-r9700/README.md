# Radeon AI PRO R9700 (gfx1201, 32 GB)

Navi 48, 64 CUs, GDDR6 on a 256-bit bus: 640 GB/s peak, 613 GB/s measured; 191 TFLOP/s dense f16 by spec. Two names: the card's GPU key
is `radeon-r9700`; every measurement runs on the DevLab Dispatch **R9700S** runner (same silicon, 2.35 GHz rocminfo clock, 214 TFLOP/s
measured) whose roofline preset is `radeon-r9700s` (the card uses it: `roofline.default`). Both presets are in `device.json`.

- Pool: the board reports 31.86 GiB; loom-serve gets 34.0 GB (`memory_bytes`). A package that does not fit at its native context is
  served at a reduced `served_context`; one whose weights do not fit is that GPU's `unsupported` state.
- Serving: `--pool_backing=fixed` (`card.serve_args`): the elastic backing's trim + compaction after each session stalls the next long
  prefill; fixed backing commits the preset's `pool_capacity` eagerly.
- Preset: `<first>-r9700` (`preset_suffix`; `build.py --prefill-r9700`, a larger prefill chunk, same programs); fit table
  `.github/devlab/r9700_fit.json`. Runners `[self-hosted, devlab-dispatch, radeon-r9700]`, about 3 concurrent.

Data: `device.json`.
