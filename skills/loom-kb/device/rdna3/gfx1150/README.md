# gfx1150 (Radeon 890M, Strix Point)

RDNA 3.5 APU, 16 CUs, shared LPDDR5X (120 GB/s peak). Family `rdna3` (gfx11-generic): the programs are the base target's.

- No DevLab runner: `runner.labels` is empty. Results come from a named local machine and live in `fleet/devices/gfx1150.json`
  (schema `fleet/devices/device_results.schema.json`, README there). A card shows this GPU only for a model whose entry there has data.
- Preset: reuses the default preset; `<first>-gfx1150` (`preset_suffix`) is written only when the data needs other serving settings.
- Roofline: `roofline.py --device gfx1150`: 96 GB/s MEASURED (streaming copy, 80 % of the 120 GB/s LPDDR5X-7500 peak), 23.8 TFLOP/s spec.
- `kb.py route --device gfx1150` (aliases `strix-point`, `890m`). Measured on one laptop: HP OmniBook Ultra 14 (Ryzen AI 9 HX 375,
  32 GB LPDDR5X-7500), Ubuntu, Linux 7.0, 2026-11-16; the published programs compile and run unchanged (no new kernels).

## Host setup it needs (model-independent, not in the card)

1. **Locked-memory limit.** loom-serve streams weights through io_uring registered buffers; Ubuntu's default `RLIMIT_MEMLOCK` (8 MiB, also the
   hard limit) makes every weight load fail AFTER registration reports `ready`: the server log fills with
   `driver_failure ... io_uring/proactor_registration.c:358: RESOURCE_EXHAUSTED; buffer pin preparation failed (12)` and every request
   returns `model is unavailable`. Fix (root, then log in again): `/etc/security/limits.d/90-loom.conf` with `* - memlock unlimited`
   (`card.host_setup` in `device.json` holds the wording a card prints; and `DefaultLimitMEMLOCK=infinity` in `/etc/systemd/{system,user}.conf` for services). Without root, a preload that makes
   `io_uring_setup` return ENOSYS sends IREE to its portable threaded backend and serving works (how the numbers below were taken);
   it is a diagnostic, not a deployment. Check: `ulimit -l` must print `unlimited` (or several GiB).
2. **Memory budget = GTT, not 60 % of RAM.** GTT defaults to half of RAM (15.08 GiB on 32 GB). A card's Strix Halo quick start
   (`--memory_bytes=28000000000`) registers and then fails every allocation; `--pool_backing=fixed` fails at registration (a 262k-context
   preset commits its whole pool: `libamdf ERRNO 0x0000000C`). Serve elastic with `--memory_bytes` <= GTT (14-15e9 on 32 GB). Too small
   (12e9 for Qwen3.5-4B) fails registration with code 8. Read the budget from `/sys/class/drm/card*/device/mem_info_gtt_total`.

## Serving settings (hrx-loom 0.2.0)

- Elastic backing, `--memory_bytes` = the GTT budget (14e9 on a 32 GB laptop), no extra serve flags, no new kernels.
- **Context cap per package.** 0.2.0 refuses registration when the budget cannot address one full context
  (`code 8: shared pool byte limit cannot address one model context`), so a 262k-context card needs a device preset with a smaller
  `context_capacity`: build with `LOOM_MEMORY_CAP_GIB=<GTT GiB>` (the kit fits the context to the budget; Qwen3.5-4B: 63488).
- Without `--memory_bytes` the server sizes its pool to MemTotal (32.4e9), not the GTT: Qwen3.5-4B then registers at 262k,
  completes a 128k-token request and fails at ~141k with `driver_failure RESOURCE_EXHAUSTED` (GTT full); the model stays `failed`
  until restart. No OOM kill, the desktop stays usable. Do not serve budget-less on this device.
- Fit of the fleet at 14e9 (`.github/scripts/fit.py --device gfx1150 --approx`): 11 native, 18 reduced context, 22 do not fit.

## Measured (Qwen3.5-4B Q4_K_M, hrx-loom 0.2.0, served context 63488, DFlash depth 3, 2026-11-16)

| row | Loom | llama.cpp ROCm b1374 | llama.cpp Vulkan b12211 | Strix Halo card (Loom) |
|---|---|---|---|---|
| decode | 31.1 | 20.4 | 20.8 | 87.4 |
| decode @2k | 28.4 | 24.1 | 23.0 | 84.1 |
| decode @32k | 21.9 | 16.1 | 14.4 | 60.9 |
| prefill 1k | 995 | 702 | 615 | 2840 |
| prefill 2k | 1018 | 706 | 644 | 2906 |
| prefill 8k | 1017 | 636 | 634 | 2827 |
| prefill 16k | 996 | 510 | 584 | 2649 |
| prefill 32k | 916 | 362 | 496 | 2369 |
| 4 streams (no drafting) | 81.8 | 71.4 | 68.9 | 223.4 |

func and the dev gates (consistency, spec, agent, golden, compare, vision, sampling) pass. Drafted acceptance 0.30 / 0.40 / 0.40.
Loom leads llama.cpp ROCm by +52 % decode, +36 % at 32k, +42-153 % prefill (the gap widens with prompt length), +15 % with 4 streams.
Absolute rates are ~0.36x Strix Halo, close to the compute ratio (16/40 CUs). hrx-loom 0.1.0 gave the same numbers within 1-3 %.
Provenance and raw values: `fleet/devices/gfx1150.json`. Loom reaches 0.83-0.88 of the 96 GB/s decode roofline and 0.30-0.35 of the prefill one (card section).

Data: `device.json`; decode levers measured here: `decode.md`.
