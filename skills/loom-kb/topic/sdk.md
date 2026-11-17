# Topic: sdk (hrx-loom SDK limits and bugs, per SDK version)

What the SDK could not do when a port or an optimization needed it: each entry is the limit, where it bit (evidence), the workaround in
use, and what an SDK change would unlock. Limits of one GPU family's code generation are in that device node's `sdk.md`. When the SDK
version changes, re-check the entries (delete the ones that are fixed, with the version that fixed them) and add a section for the new
version. Write-ups for the SDK team are built from this file.

## Contents
- hrx-loom 0.1.0: tools (compile, test, profile)
- hrx-loom 0.1.0: language and lowering
- hrx-loom 0.1.0: runtime (load-time preparation, scratch, first load)
- hrx-loom 0.1.0: grid-wide synchronization (persistent kernels)
- Verified to work (not limits)

## hrx-loom 0.1.0: tools (compile, test, profile)

Recorded by the mxfp4st campaign, 2026-11-15/16 (hrx-loom 0.1.0 from PyPI, DevLab R9700 runners).
- **No assembly listing.** `loom-compile --format=` offers only `amdgpu-hsaco` (`amdgpu-asm`, `-isa`, `llvm-ir`: "not available in this binary").
  Bit: the fp8 GEMM's gap to the WMMA peak was searched with timing ablations, 6 DevLab rounds. Workaround: disassemble the code object
  with ROCm's `llvm-objdump` (topic `optimize-loop`, "Local pre-flight"). Unlock: an asm dump or a per-kernel VGPR / occupancy / wait report.
- **No offline compile of a command program.** `loom-compile` roots are kernels; `loom-opt --pass=canonicalize` parses and type-checks
  launches, but `--pipeline=source-low` stops at select-templates (`LOWERING/045 missing_facts` for the dual-target `template.apply`) and
  `loom-opt` has no `--target`. Bit: command-program lowering errors are first seen on the runner. Workaround: kernel compiles for both
  targets + the canonicalize parse. Unlock: `loom-opt --pipeline=source-low --target=amdgpu:gfx1201` or a command-program compile product.
- **Kernel tests need a GPU.** `loom-test --device` accepts only a HAL driver with a linked target adapter; there is no CPU or interpreter
  backend for WMMA / fp8 ops. Bit: every kernel numeric check is a DevLab round trip (3-10 min with the queue). Workaround: kbench suites
  with a `loom-test` step; the generators' numpy references are unit-tested on the CPU. Unlock: CPU execution of `vector.mma` and fragment ops.
- **Large test modules fail to verify.** A module of about 65,000 lines fails with `ENCODING/004 view type references encoding value
  '%blocks' ... which is not defined at this point`, naming values of other kernels; every kernel compiles alone. Workaround: about 60
  check cases per test file (`gen_bench` TESTED). Unlock: fix the encoding-value tracking across a large module.
- **Test-API drift between builds.** 0.1.0 has no `check.expect.close` (use `check.expect.approximate actual() expected() : T, T`); a newer
  dev build (0.0.0.dev0) rejects `vector.fragment.load` origins it cannot prove in bounds (`origin + 16 <= rows`: add
  `index.assume lt(row, rows - 15)`). Kernels and tests must compile on both.
- **Profiling.** Hardware counters, counter ranges, device metrics and executable traces are unimplemented on the AMD GPU HAL; the
  per-dispatch timestamps work but drop records with the default ring. Details and the working recipe: topic `optimize-loop`, "Profiling".

## hrx-loom 0.1.0: language and lowering

- **Exact `expf` is rejected on amdgpu.** `scalar.expf` without `<afn>` fails the math legalization (`LOWERING/036` "legalize-math policy
  amdgpu-math rejected scalar.expf under math.exp.exact_f32"). Bit: the load-time `-exp(A_log)` of a safetensors hybrid (prepare op
  `negexp`). Workaround: `scalar.expf<afn>` (relative error ~1e-6 on 48 constants per layer; gates and chat coherent). Unlock: an exact
  f32 exp lowering (llama.cpp's converter computes it in f32 on the host).

## hrx-loom 0.1.0: runtime (load-time preparation, scratch, first load)

- **Preparation staging is four lanes of the largest input.** A weight policy's load-time preparation (`prepare.stage` + `weights.input`;
  `loom/serve/runtime/parameter/weights.h`, `inputs.h`) reserves a staging workspace of four read lanes, each as large as the largest
  captured input range of any prepared tensor, inside the device budget of weights + state. A tensor is one policy key, one recipe, one
  final buffer; nothing streams a tensor in chunks or prepares from another tensor's resident bytes. Bit: a 2.54 GB bf16 head staged whole
  exhausted the 32 GB R9700 at every context; head rows [0, N) staged as one range cost 4 x N x 10 KB (N = 81920: 3.4 GB) and served
  context. Workaround: split a large copy into separately named prepared tensors (staging = the largest piece x 4), each
  `input(source, key, off, len, align)` with an explicit key reading rows of the source (a prepared tensor needs no checkpoint entry of its
  own). Unlock: chunked, in-place or device-to-device preparation, or a lane-count knob.
- **A second preparation plan breaks the first load of a fresh VM (bug).** A separate policy on another parameter root of the same
  checkpoint (`head_weights.loom`) made the FIRST weight load on a fresh R9700 VM fail: HAL `RESOURCE_EXHAUSTED` at
  `runtime/src/iree/hal/drivers/amd/fabric/memory.c:394` (`memory_create`, libamdf ENOMEM) at ~28 GB with VRAM free before the serve,
  independent of the staging size. loom-serve drops the connection, the registration goes `retiring` and every later request is 503
  "model is unavailable" until a restart, which then loads fine. Not page cache, host RAM or a race (runs 43819656377 .. 43827038741).
  Workaround: every prepared tensor in the main weight policy (5/5 fresh-VM probes pass, run 43864913856); never hide it with a restart.
  Unlock: a first load that works with two plans, or a precise error naming what the second plan allocates.
- **Scratch is sized at every packing shape.** loom-serve prepares every command program (verify too) at every default packing shape
  (`schedule.c` `loom_serve_text_default_shapes`: tokens 1..prefill, spans 1..tokens, up to (1024, 1024)) and allocates scratch for the
  largest, while `build.py`'s `program_scratch` evaluates only (prefill, rows). Bit: a per-span scratch was fine at (1024, 4) and 29 GB at
  (1024, 1024): `RESOURCE_EXHAUSTED` at serve (run 43821931908). Workaround: cap per-span scratch (`min(spans, N)`) and check
  `program_scratch` at (prefill, prefill). Unlock: the program's real span bound as a fact of the packing shapes.

## hrx-loom 0.1.0: grid-wide synchronization (persistent kernels)

- **Expressible, not supported.** A device-wide barrier works on gfx1201: lane 0 spins on `view.atomic.rmw<addi>` / `view.atomic.load`
  (`scope = device`) between two `kernel.barrier<workgroup>`, with `buffer.fence scope(device)` release before and acquire after (without
  the fences a probe read stale neighbour data, kbench 43807976177). Missing: (a) a cooperative launch or residency guarantee
  (`kernel.launch.config` has workgroups, workgroup size and cluster size only; more workgroups than fit at once deadlock, and there is no
  `max_resident` / CU-count query in the launch-config region); (b) a native grid barrier (`kernel.barrier<global>` is a workgroup
  rendezvous with a global fence); (c) the device-scope acquire makes every mutable load of the kernel coherent, streamed weights
  included; (d) no instruction prefetch or persistent-scheduling control; the barrier counter must be zeroed by the program or derived
  (`target = (old / G + 1) * G`). Cost against launches: device node `decode`. Unlock: cooperative launch with a residency check + a
  native grid barrier.

## Verified to work (not limits)

- Parameter views above 2^31 bytes (`view<2542796800xi8>`, the 2.54 GB bf16 embedding and head) load and run on gfx1201 (ktest
  head_argmax / embed PASS, run 43770308768).
