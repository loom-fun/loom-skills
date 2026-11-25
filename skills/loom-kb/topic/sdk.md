# Topic: sdk (hrx-loom SDK limits and bugs, per SDK version)

What the SDK could not do when a port or an optimization needed it: each entry is the limit, where it bit (evidence), the workaround in
use, and what an SDK change would unlock. Limits of one GPU family's code generation are in that device node's `sdk.md`. When the SDK
version changes, re-check the entries (delete the ones that are fixed, with the version that fixed them) and add a section for the new
version. Write-ups for the SDK team are built from this file.

## Contents
- hrx-loom 0.2.0: rewritten chat history
- hrx-loom 0.1.0 / 0.2.0: SSA encoding ids are 16 bits ("source value %N has no target remap", ENCODING/004)
- hrx-loom 0.2.0: development-build ABI and comparison pins
- hrx-loom 0.1.0: tools (compile, test, profile)
- hrx-loom 0.1.0: language and lowering
- hrx-loom 0.1.0: runtime (load-time preparation, scratch, first load)
- hrx-loom 0.1.0: grid-wide synchronization (persistent kernels)
- Verified to work (not limits)

## hrx-loom 0.2.0: rewritten chat history

- **Generated prefixes are not automatically checkpointed.** Bolt thinking-on tool turns still re-prefill the whole history
  ([issue #40](https://github.com/loom-fun/loom-skills/issues/40),
  [current-SDK reproduction](https://github.com/loom-fun/loom-skills/actions/runs/44165222659/job/134329462508)).
  The model writes bare `<channel|>`; the unchanged template inserts a preceding newline in the next prompt. The first mismatches
  are tokens 197, 386 and 442 (token 101 becomes token 107, then 101), after 92.5%, 92.8% and 94.4% of the previous totals.
  All six thinking-on replies match the reference, but all three follow-ups report cached=0. Thinking off reuses over 99%.
  In shipped runtime `0cd370f2`, `service/text/service.c:loom_serve_text_capture_prefix` captures only prefill frontiers;
  `prefix_cache_capacity` cannot capture generated reasoning. Explicit checkpoints pin idle completed histories, and canonical-history
  validation still requires exact extension; increasing cache capacity does not add a rewind operation. The current safe behavior is
  to replay rewritten history. Unlock: generated-token prefix checkpoints or a correct common-prefix restore/replay path, preserving
  the model's output and the template. A passing agent gate excuses verified template rewrites and does **not** mean this performance
  target was achieved. Issue #40 remains open; do not replace generated tokens or change the template merely to obtain a cache hit.

## hrx-loom 0.1.0 / 0.2.0: SSA encoding ids are 16 bits ("source value %N has no target remap", ENCODING/004)

- **A type that references an SSA encoding stores the encoding's module-wide value id in 16 bits.** The text parser truncates
  silently (`format/text/parser/types.c`, `loom_parse_type_encoding`: `*out_encoding_id = (uint16_t)enc_value_id`; field
  `loom_type_t.encoding_id`, `ir/types.h`), so in a module with more than 65,536 values a `view<..., %layout>` whose `%layout` is
  defined after value 65,535 names value `id - 65536` of some earlier kernel. Symptoms: the parse verifier (`loom-compile` of the
  file) reports `ENCODING/004 ... references encoding value '%x' which is not defined at this point`; loom-serve does not verify a
  library, so the package compiles with 0 errors and registration fails when a compile request clones the kernel
  (`/v1/status`: code 5, `source value %87 has no target remap`, 87 = id - 65536; `rewrite/remap_payload.c`). Only kernels that a
  device compiles fail, so one GPU family can register and the other not.
- Bit: Qwen3.8-27B with both `quant/q4_0` choices ([R9700 run 44312519207](https://github.com/loom-fun/loom-skills/actions/runs/44312519207),
  [44332419573](https://github.com/loom-fun/loom-skills/actions/runs/44332419573), and gfx1151): the int8 matvecs and m256 GEMMs put
  the rdna3 flash attention's `%out_layout` at about value 65,600. Kernel counts were a coincidence (dropping any kernel before it
  moved it under the limit on gfx1151, not the rdna4 twin compiled on the R9700); Laguna-XS-2.1 (94k lines) and large kbench test
  modules hit the same limit.
- Workaround (kit): `dual_target.encodings_first` reorders a deep file (estimated depth above 60,000 values, or an encoding-typed line
  past 4 MB) so the leading funcs and every encoding-typed kernel come first, wherever they were; it raises if they alone are too deep.
  Before this fix it reordered only the kernels ahead of the first command program, and Qwen3.8-27B's MTP weight converter
  (`@llm_prep_q6_k_q4_0`) sits in the middle of its kernels.
- Split (kit, `split_library.py`, choice `target.library.split`, default auto): when the encoding kernels alone are too deep
  (unsloth Qwen3.8-27B UD-Q4_K_XL: six GEMM formats x tiles x gfx12 twins), build.py moves them (with their twins and the inline
  helpers they call) into `program/llm_kernels_N.loom` libraries listed BEFORE `program/llm.loom` in the card; llm.loom launches
  them through `kernel.decl` lines. The limit is per parsed file: loom-serve indexes each library as its own link source and links
  per-kernel closures. A link that materializes a command's whole closure (loom-link `--root=@llm_advance`) numbers values in
  library order, so the kernel files go first (llm.loom first fails: `cannot store target value %65545 in a 16-bit SSA encoding
  reference`). Kernel tests run one loom-test module per library file. Every published file is below 60,000 (deepest: Ornith-1.5-35B-A3B 59,399).
- Standalone repro (no package): a file with one filler kernel of N `index.constant`s, then
  `kernel.def export("k") @k() {...} launch(%out: buffer) {` ... `%layout = encoding.layout.strided [%c1, %c16] : encoding<layout>`,
  `%v = buffer.view %out[%zero] : buffer -> view<16x16xf32, %layout>`, a `view.store` through `%v`. `loom-compile r.loom --root=@k
  --target=amdgpu:gfx1151`: N = 65,000 compiles; N = 66,000 fails `ENCODING/004 ... encoding value '%v471'` (66,007 - 65,536).
- Unlock (SDK fix): store SSA encoding references as full 32-bit value ids in `loom_type_t` (or intern them in a side table like
  overflow dims), and until then make the parser fail with an explicit error when `enc_value_id > UINT16_MAX`, as
  `loom_ir_remap_leaf_type` already does for remapped targets; loom-serve should verify libraries before registering.

## hrx-loom 0.2.0: development-build ABI and comparison pins

- **A development version string does not identify a compatible export ABI.** Flash-Next programs built for source
  `314238a81c0f` (`0.2.0.dev0`, CI artifact 13628412621) fail registration on released source `0cd370f2baaa`:
  `text_publish_epoch_v1` declares 12 arguments but the runtime requires 13; the old image export also has 3 rather than 5 arguments.
  A failed arm is not a performance result ([diagnostic run](https://github.com/loom-fun/loom-skills/actions/runs/44164029861)).
  Use the exact source/wheel pin with historical programs, or regenerate against the current SDK and revalidate. Unused-argument
  adapters can isolate catalog arithmetic diagnostically; they do not prove package correctness or justify a production rollback.
  The original generator recreated the published program hash while sharing one checkpoint in the
  [valid old/current comparison](https://github.com/loom-fun/loom-skills/actions/runs/44188965599).
  Unlock: an explicit export-ABI revision in package metadata and an early compatibility check.
- **Keep the CI selector separate from its installation directory.** Selectors such as `ci:13628412621` must reach the installer
  unchanged, but Python venv rejects colons in Linux root paths. This was a DevLab tooling bug, fixed by normalizing only the
  directory name ([PR 1245](https://github.com/loom-fun/loom-skills/pull/1245)); implicit card pins already choose safe sibling roots.

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
  '%blocks' ... which is not defined at this point`, naming values of other kernels; every kernel compiles alone. Cause: 16-bit SSA
  encoding ids (section above). Workaround: about 60 check cases per test file (`gen_bench` TESTED).
- **Some kernels cannot be a loom-test case root as generated** (0.2.0). An indirect dispatch (`X_listed`: the workgroup count
  read from a buffer) fails `launch-config artifact still depends on non-scalar workload argument`; SwiGLU `g_silu_mul` proves
  its vector bound only for the program's constant width (`SUBRANGE/010` with a dynamic width). Workaround: make_kernel_tests
  tests a `kt_` copy (scalar count; the program's constant width), `ktest_variants.py`. Unlock: indirect dispatch in check
  cases, and specializing a case's literal arguments before the bounds proof.
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

- **GGUF tensor types (hrx-loom 0.3.0)**: the runtime GGUF parser (`runtime/src/iree/io/formats/gguf/gguf_parser.c`) has a trait
  table for GGML types 0..42 only; a parameter file holding another type fails registration with `GGML tensor type 143 not supported`
  (PrismML PTQ1_0, DevLab run 45530428087), whatever the program does with it. A fork-only type needs a file in a known packing (Ternary
  Bonsai: its Q2_0 file, type 42) and, for a denser layout, a load-time weight policy.

## Verified to work (not limits)

- Parameter views above 2^31 bytes (`view<2542796800xi8>`, the 2.54 GB bf16 embedding and head) load and run on gfx1201 (ktest
  head_argmax / embed PASS, run 43770308768).
