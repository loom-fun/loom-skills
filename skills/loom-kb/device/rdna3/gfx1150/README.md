# gfx1150 (Radeon 890M, Strix Point)

RDNA 3.5 APU, 16 CUs, shared LPDDR5X (120 GB/s peak). Family `rdna3` (gfx11-generic): the programs are the base target's.

- No DevLab runner: `runner.labels` is empty. Results come from a named local machine and live in `fleet/devices/gfx1150.json`
  (schema `fleet/devices/device_results.schema.json`, README there). A card shows this GPU only for a model whose entry there has data.
- Preset: reuses the default preset; `<first>-gfx1150` (`preset_suffix`) is written only when the data needs other serving settings.
- Roofline: `roofline.py --device gfx1150`: 96 GB/s MEASURED (streaming copy, 80 % of the 120 GB/s LPDDR5X-7500 peak), 23.8 TFLOP/s spec.
- `kb.py route --device gfx1150` (aliases `strix-point`, `890m`). Measured on one laptop: HP OmniBook Ultra 14 (Ryzen AI 9 HX 375,
  32 GB LPDDR5X-7500), Ubuntu, Linux 7.0, 2026-11-16; the published programs compile and run unchanged (no new kernels).

## Stock Ubuntu host: start loom-serve without io_uring (model-independent, no root)

Ubuntu's default `RLIMIT_MEMLOCK` is 8 MiB (also the hard limit). hrx-loom 0.2.0 loads weights through GPU file staging buffers of
8 lanes x 2 MiB = 16 MiB (`hal/driver/amd/gpu/device.c:1303-1308`, `1343-1350`, hard-coded) and pins each through io_uring whenever the
kernel offers concurrent buffer registration (`gpu/queue/file_staging.c:1557-1562`; Linux 7.0 does). A failed pin fails the allocation
(`fabric/memory.c:552-559`; `async/platform/io_uring/proactor_registration.c:323-366` has no ENOMEM fallback: the RLIMIT_MEMLOCK fallback at
lines 1086/1138 covers `register_slab` only). Symptom: registration reports `ready`, the first request returns an empty reply, the log shows
`driver_failure ... proactor_registration.c:358: RESOURCE_EXHAUSTED; buffer pin preparation failed (12)` and the model stays `failed`.
No loom-serve flag, environment variable or package setting reaches the staging size or the proactor capabilities (`allowed_capabilities` is
never set outside tests).

User-level fix: start loom-serve with `io_uring_setup` filtered out, so IREE falls back to its POSIX proactor (`async/proactor_platform.c:39-46`;
ENOSYS/EPERM count as unavailable, `io_uring/uring.c:83-85`). An unprivileged systemd user unit can install the seccomp filter:

    systemd-run --user --pty --same-dir -p SystemCallFilter=~io_uring_setup -p SystemCallErrorNumber=ENOSYS \
      loom-serve --model=PKG '--device=amdgpu://?completion_spin_ns=4294967295' --memory_bytes=14000000000

The card prints it as the gfx1150 Quick start prefix (`device.json` `card.serve_prefix`). A host with `ulimit -l unlimited`
(`* - memlock unlimited` in `/etc/security/limits.d/`) can drop it. Measured with it: the full card methodology (func, dev gates, perf,
baselines) passes and the rows match the io_uring-off runs (`fleet/devices/gfx1150.json`).

**Memory budget = GTT, not 60 % of RAM.** GTT defaults to half of RAM (15.08 GiB on 32 GB). A budget above it (the Strix Halo quick start's
28e9) registers and then fails every allocation; `--pool_backing=fixed` fails at registration for a 262k-context preset. Serve elastic with
`--memory_bytes` <= GTT (14e9 on 32 GB); read it from `/sys/class/drm/card*/device/mem_info_gtt_total`.

## Serving settings (hrx-loom 0.2.0)

- Elastic backing, `--memory_bytes` = the GTT budget (14e9 on a 32 GB laptop), no extra serve flags, no new kernels.
- **Context cap per package.** 0.2.0 refuses registration when the budget cannot address one full context
  (`code 8: shared pool byte limit cannot address one model context`), so a 262k-context card needs a device preset with a smaller
  `context_capacity`: build with `LOOM_MEMORY_CAP_GIB=<GTT GiB>` (the kit fits the context to the budget; Qwen3.5-4B: 63488).
- Without `--memory_bytes` the server sizes its pool to MemTotal (32.4e9), not the GTT: Qwen3.5-4B then registers at 262k,
  completes a 128k-token request and fails at ~141k with `driver_failure RESOURCE_EXHAUSTED` (GTT full); the model stays `failed`
  until restart. No OOM kill, the desktop stays usable. Do not serve budget-less on this device.
- Fit of the fleet at 14e9 (`.github/scripts/fit.py --device gfx1150 --approx`): 11 native, 18 reduced context, 22 do not fit.

## Measured (Qwen3.5-4B Q4_K_M, hrx-loom 0.2.0, served context 63488, DFlash depth 3, stock host: memlock 8 MiB under the serve prefix, 2026-11-16)

| row | Loom | llama.cpp ROCm b1374 | llama.cpp Vulkan b12211 | Strix Halo card (Loom) |
|---|---|---|---|---|
| decode | 31.1 | 21.7 | 21.1 | 87.4 |
| decode @2k | 28.2 | 24.0 | 21.2 | 84.1 |
| decode @32k | 22.4 | 17.2 | 14.5 | 60.9 |
| prefill 1k | 996 | 703 | 608 | 2840 |
| prefill 2k | 1016 | 703 | 641 | 2906 |
| prefill 8k | 1021 | 625 | 632 | 2827 |
| prefill 16k | 1041 | 497 | 583 | 2649 |
| prefill 32k | 929 | 357 | 495 | 2369 |
| 4 streams (no drafting) | 82.3 | 74.4 | 69.4 | 223.4 |

func and the dev gates (consistency, spec, agent, golden, compare, vision, sampling) pass. Drafted acceptance 0.30 / 0.40 / 0.40.
Loom leads llama.cpp ROCm by +43 % decode, +30 % at 32k, +42-160 % prefill (the gap widens with prompt length), +11 % with 4 streams. The io_uring and POSIX I/O backends give the same rows within 1-2 %.
Absolute rates are ~0.36x Strix Halo, close to the compute ratio (16/40 CUs). hrx-loom 0.1.0 gave the same numbers within 1-3 %.
Provenance and raw values: `fleet/devices/gfx1150.json`. Loom reaches 0.83-0.88 of the 96 GB/s decode roofline and 0.30-0.35 of the prefill one (card section).

Data: `device.json`; decode levers measured here: `decode.md`.
