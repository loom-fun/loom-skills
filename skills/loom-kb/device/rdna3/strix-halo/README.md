# Strix Halo (gfx1151)

Ryzen AI Max+ 395, 40 CUs, LPDDR5X-8000 on a 256-bit bus: 256 GB/s peak, 216 GB/s measured streaming read; 59 TFLOP/s dense f16.
GPU memory is system RAM (the card's `--memory_bytes` is the package's own; `memory_bytes: null` here). DevLab runners:
`[strix-halo, devlab-dispatch, Linux, self-hosted]` - always include `Linux`, or a job can land on a Windows runner. The card's
default preset (`<first>`) is tuned here; no fit table (every package fits).

Data: `device.json` (card entry, roofline preset, runner, DevLab device block).
