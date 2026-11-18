# gfx1150 (Radeon 890M, Strix Point)

RDNA 3.5 APU, 16 CUs, shared LPDDR5X (120 GB/s peak). Family `rdna3` (gfx11-generic): the programs are the base target's.

- No DevLab runner: `runner.labels` is empty. Results come from a named local machine and live in `fleet/devices/gfx1150.json`
  (schema `fleet/devices/device_results.schema.json`, README there). A card shows this GPU only for a model whose entry there has data.
- Preset: reuses the default preset; `<first>-gfx1150` (`preset_suffix`) is written only when the data needs other serving settings.
- Roofline: `roofline.py --device gfx1150`: 96 GB/s MEASURED (streaming copy, 80 % of the 120 GB/s LPDDR5X-7500 peak), 23.8 TFLOP/s spec.
- `kb.py route --device gfx1150` (aliases `strix-point`, `890m`). Measured on one laptop: HP OmniBook Ultra 14 (Ryzen AI 9 HX 375,
  32 GB LPDDR5X-7500), Ubuntu, Linux 7.0, 2026-11-16; the published programs compile and run unchanged (no new kernels).

## Stock Ubuntu host: `--parameter_mode=preload` (model-independent, no root)

Serve with `--parameter_mode=preload` on a host whose `ulimit -l` is the Ubuntu default (8 MiB). Without it, hrx-loom 0.2.0 registers
(`ready`), the first request returns an empty reply, the log shows
`driver_failure: status=RESOURCE_EXHAUSTED(8) ... io_uring/proactor_registration.c:358: RESOURCE_EXHAUSTED; buffer pin preparation failed (12)`
and the model stays `failed` (code 8) until restart. Measured with preload (Qwen3.5-4B, same machine): decode, prefill and 4 streams within
+-2 % of the default mode, decode @2k within the acceptance spread; cost: the parameter files are read into host memory (+3.4 GiB RSS).
`--parameter_mode=direct` fails like the default.

### Root cause (notes for the owner; hrx-system tag v0.2.0, `0cd370f`)

- `--parameter_mode=file` (default) and `direct` open parameter files for async I/O (`runtime/src/iree/tooling/parameter_util.c:50-56`); the
  AMD GPU device reads them through file staging buffers of a hard-coded 8 lanes x 2 MiB = 16 MiB per pool
  (`hal/driver/amd/gpu/device.c:1303-1308` queue family, `1343-1350` each transfer family).
- The staging memory is registered with io_uring whenever the proactor reports `CONCURRENT_BUFFER_REGISTRATION`
  (`hal/driver/amd/gpu/queue/file_staging.c:1557-1562`, comment: "Retained pins accelerate repeated I/O but are not required"). The capability is
  set when io_uring's buffer-clone probe succeeds (`async/platform/io_uring/proactor.c:238-251`); Linux 7.0 has it.
- Allocation calls `iree_async_proactor_prepare_buffer` for each proactor and fails on any error (`hal/driver/amd/fabric/memory.c:552-559`). The
  io_uring implementation pins the whole buffer (`async/platform/io_uring/proactor_registration.c:323-366`, error at 353-358) and has no ENOMEM
  fallback. The RLIMIT_MEMLOCK fallback (lines 1086-1095, 1138-1143; `async/region.h:208`) is on the other registration path,
  `fixed_buffers_register` / `register_slab`, not this one.
- No flag or environment variable changes the staging size or masks the capability: `allowed_capabilities` is only ever the default `ALL`
  (`async/proactor.h:450`). `preload` (`parameter_util.c:45-48`) gives host-memory file handles that never reach file staging.
- **Who it hits:** any AMD GPU served by the `amdgpu` driver on Linux (Strix Halo, R9700, RX 9070 XT and gfx1150 alike: `device.c` is not
  device-specific), in the default parameter mode, on a kernel with io_uring buffer clone, with `RLIMIT_MEMLOCK` below the staging pin (16 MiB
  per pool). Hosts with a raised memlock limit (as DevLab runners may have) or an older kernel do not see it. Windows (IOCP) is not affected.
- **Minimal SDK fix:** make the pin optional where the comment says it is: in `fabric/memory.c:552-559` (or for file staging only,
  `file_staging.c:1557`), treat `RESOURCE_EXHAUSTED` from `prepare_buffer` as "no registration" and keep the allocation
  (`file_staging_attach_view` already skips a missing `block->preparation`, `file_staging.c:707-710`), with the same WARNING the
  `register_slab` path prints. A smaller staging pool (e.g. 4 x 1 MiB) would also fit under 8 MiB, but it only moves the threshold.
- Other user-level fixes, each verified here: `ulimit -l unlimited` (root: `* - memlock unlimited` in `/etc/security/limits.d/`), or start
  loom-serve under `systemd-run --user --pty --same-dir -p SystemCallFilter=~io_uring_setup -p SystemCallErrorNumber=ENOSYS` (no root: IREE
  falls back to its POSIX proactor, `async/proactor_platform.c:39-46`). The full card methodology passed that way; it is where the
  evidence in `fleet/devices/gfx1150.json` comes from.

**Memory budget = GTT, not 60 % of RAM.** GTT defaults to half of RAM (15.08 GiB on 32 GB). A budget above it (the Strix Halo quick start's
28e9) registers and then fails every allocation; `--pool_backing=fixed` fails at registration for a 262k-context preset. Serve elastic with
`--memory_bytes` <= GTT (14e9 on 32 GB); read it from `/sys/class/drm/card*/device/mem_info_gtt_total`.

## Serving settings (hrx-loom 0.2.0)

- Elastic backing, `--memory_bytes` = the GTT budget (14e9 on a 32 GB laptop), no extra serve flags, no new kernels.
- **Context cap per package.** 0.2.0 refuses registration when the budget cannot address one full context
  (`code 8: shared pool byte limit cannot address one model context`), so a 262k-context card needs a device preset with a smaller
  `context_capacity`: build with `LOOM_MEMORY_CAP_GIB=<GTT GiB>` (the kit fits the context to the budget; Qwen3.5-4B: 63488).
  Measured with `test_9070xt.py --memory-bytes 14000000000 --state-slab-bytes none` (largest context a fresh server completes and then
  answers correctly after): Qwen3.5-2B 262144 (native, default preset), Qwen3.5-9B 40960 (above ~41k a full-length request drops the
  connection or garbles the next answer instead of a clean 503).
- Without `--memory_bytes` the server sizes its pool to MemTotal (32.4e9), not the GTT: Qwen3.5-4B then registers at 262k,
  completes a 128k-token request and fails at ~141k with `driver_failure RESOURCE_EXHAUSTED` (GTT full); the model stays `failed`
  until restart. No OOM kill, the desktop stays usable. Do not serve budget-less on this device.
- Fit of the fleet at 14e9 (`.github/scripts/fit.py --device gfx1150 --approx`): 11 native, 18 reduced context, 22 do not fit.

## Measured (Qwen3.5-4B Q4_K_M, hrx-loom 0.2.0, served context 63488, DFlash depth 3, stock host, memlock 8 MiB: default file mode under the seccomp prefix; preload spot-checked within +-2 %, 2026-11-16)

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

func and the dev gates (consistency, spec, agent, golden, compare, vision, sampling) pass. Drafted acceptance 0.327 / 0.418 / 0.392 (decode / decode @2k / decode @32k rows); the `spec` gate's acceptance 0.53.
Loom leads llama.cpp ROCm by +43 % decode, +30 % at 32k, +42-160 % prefill (the gap widens with prompt length), +11 % with 4 streams. The io_uring and POSIX I/O backends give the same rows within 1-2 %.
Absolute rates are ~0.36x Strix Halo, close to the compute ratio (16/40 CUs). hrx-loom 0.1.0 gave the same numbers within 1-3 %.
Provenance and raw values: `fleet/devices/gfx1150.json`. Achieved / roofline ceiling from the stored rows: decode 0.884 / 0.822 / 0.882 (short / 2k / 32k; single-pass ceilings at 96 GB/s, Loom drafts), prefill 0.301-0.361 (1k-32k), 4 streams 0.602. The roofline evidence's `efficiency` field is a different number: roofline.py's demonstrated fraction for the metric class (EFF, the far-from-roofline guardrail), not this ratio.

## Measured: Qwen3.5-2B and Qwen3.5-9B (hrx-loom 0.2.0, 2026-11-18, served as the cards print: elastic, `--memory_bytes=14e9`, preload)

Same machine, `devlab_perf.sh` run locally (no seccomp prefix: preload needs none), 3 sessions per row; llama.cpp ROCm b1374 / Vulkan b12211.

| row | 2B Loom | 2B ROCm | 2B Vulkan | 9B Loom (ctx 40960) | 9B ROCm | 9B Vulkan |
|---|---|---|---|---|---|---|
| decode | 63.3 | 52.1 | 55.5 | 26.1 | 16.4 | 17.6 |
| decode @32k | 48.6 | 42.5 | 43.6 | 19.0 | 13.5 | 11.6 |
| prefill 2k | 3057 | 1760 | 1703 | 699 | 449 | 400 |
| prefill 32k | 2406 | 1042 | 1320 | 613 | 301 | 339 |
| 4 streams | 224.6 | 148.0 | 152.2 | 61.1 | 48.2 | 49.7 |

func and the dev gates pass on both (9B spec acceptance 0.72; 2B has no drafter). 2B serves its native 262144 with the default preset;
9B gets `chat-gfx1150` (context 40960). Raw values and provenance: `fleet/devices/gfx1150.json`.

Data: `device.json`; decode levers measured here: `decode.md`.
