# Strix Halo (gfx1151)

Ryzen AI Max+ 395, 40 CUs, LPDDR5X-8000 on a 256-bit bus: 256 GB/s peak, 228 GB/s measured streaming read (Loom 1 GiB read kernel on a DevLab runner, with and without the active wait, run 44388427539); 59 TFLOP/s dense f16 spec, never sustained: random-data WMMA holds 43.8 TFLOP/s at the 85 W BIOS balanced limit (DevLab runners; the roofline default) and 48.4 at 123 W (BIOS performance), see prefill.md.
GPU memory is system RAM (`memory_bytes: null` here: there is no device pool to clamp to). DevLab runners:
`[strix-halo, devlab-dispatch, Linux, self-hosted]` - always include `Linux`, or a job can land on a Windows runner. The card's
default preset (`<first>`) is tuned here; no fit table (every package fits).
Serving: `--pool_backing=fixed` (`card.serve_args`, no `--memory_bytes`), as on the R9700: hrx-loom 0.2.0's elastic backing costs Strix 5-8 % steady prefill
and a 3-request ramp; fixed backing commits the preset's `context_capacity` KV and the weights eagerly. `--memory_bytes` stays the elastic option (several models share one budget).
Stock hosts: `--parameter_mode=preload` (`card.preload_max_weight_bytes` = 40 GB of weights, decimal) avoids ROCm/hrx-system#3332 (the default parameter mode fails under the stock 8 MiB memlock on Linux 7.0); the preloaded host copy shares the 128 GB with GPU memory (35B-A3B passed, the 122B OOMs), so a package above the limit keeps the default mode and its Quick start says to start loom-serve with `systemd-run --user ... -p SystemCallFilter=~io_uring_setup` (or raise `ulimit -l`). `card_lib.card_serve_args` / `package_serve_args` decide, the card and `devlab_perf.sh serve_alone` use the same function.

Data: `device.json` (card entry, roofline preset, runner, DevLab device block).
