# Strix Halo (gfx1151)

Ryzen AI Max+ 395, 40 CUs, LPDDR5X-8000 on a 256-bit bus: 256 GB/s peak, 216 GB/s measured streaming read; 59 TFLOP/s dense f16.
GPU memory is system RAM (`memory_bytes: null` here: there is no device pool to clamp to). DevLab runners:
`[strix-halo, devlab-dispatch, Linux, self-hosted]` - always include `Linux`, or a job can land on a Windows runner. The card's
default preset (`<first>`) is tuned here; no fit table (every package fits).
Serving: `--pool_backing=fixed` (`card.serve_args`, no `--memory_bytes`), as on the R9700: hrx-loom 0.2.0's elastic backing costs Strix 5-8 % steady prefill
and a 3-request ramp; fixed backing commits the preset's `context_capacity` KV and the weights eagerly. `--memory_bytes` stays the elastic option (several models share one budget).

Data: `device.json` (card entry, roofline preset, runner, DevLab device block).
