# Optional compact E4B cache experiment

This adapter is for one reviewed Strix Halo job, SDK 0.2.0 / gfx1151, exact Gemma 4 E4B
Q4_K_M text checkpoint, context 65536 / prefill 1024 / rows 1. It changes no default
program. No GPU result, physical-memory saving or production parity is claimed.

The source remains experimental and default OFF. Both comparison cards explicitly
use max_sessions 1, session/prefix/checkpoint cache capacity 0 and cache target 0.
There is no draft, vision, fork/history-copy, dynamic-row-growth or multi-session
support. Build and runtime guards reject unsupported configurations. The source
unit suite includes these guards and the real no-helper-kernel audit regression.

`generate_checks.py` extracts actual emitted legacy and compact kernels plus all
called helpers. `kernel_pins.json` rejects any changed numerical source before a
GPU launch. Symbols/exports alone are renamed; negative variants alter only the
explicit address recipes. F64 attention over F16 history uses independent owner
lists/page maps and all 8 Q / 2 KV heads. Q rounding is modeled separately for flash.
Synthetic producer checks include local/full and global/cut-frequency Neox RoPE,
RMS normalization, F16 stores, live-variant stores and shared store-disabled use.
All GPU arenas, including masked/future lanes, are initialized to finite values.

Fourteen positive cases cover local window and ring boundaries, a 1024-token
write-before-read overlap, position 65535, shuffled full-context global pages,
nonzero private origins, stale-slot reset and reversed listed-decode dispatch.
Each checks independent goldens and exact legacy/compact F32 output bits. Four
strict negatives use undersized 512/1024 rings, old V-plane stride and missing
fork-history copy. Scatter destinations are unique and within initialized arenas;
CPU controls prove comparator sensitivity. Every negative has an identically
seeded, separately passing legacy/correct-compact positive control. A negative
contains no baseline launch that could accidentally satisfy its expected failure.

`run_checks.py` requires exact positive samples/expectation counts and strict
EXPECT/006 raw approximate-comparator failures. Missing samples, skips, planning
errors, compile errors, XPASS and mismatched diagnostics stop before model arms.
No tolerance is widened: 3e-3 absolute/relative attention and F16 store goldens,
with exact candidate/legacy outputs. These are synthetic kernel checks; the
matched full-model phase uses the pinned real checkpoint and meaningful outputs.

`materialize_runner.py` derives, with one-match source anchors and a saved diff,
from ordinary runner `67a784a187d57f9ab78bdd2785e04b00322192b8`. This includes the reviewed empty-token download and checkpoint-pin corrections; its monitoring
contracts are unchanged from `7abe4277`. Monitoring, status,
readiness retry, stop and settle implementations are preserved. Changes are:
matched 64k/one-row configs and no-cache policy, the kernel gate before either
server, 65536-token retrieval plus over-limit rejection, and a fresh-session short
answer after the long request. The ordinary launcher's files are not edited.
The original `run_arm` body changes only 65535 to 65536; focused tests prove this.

The immutable checkpoint revision `bfc15c382204943c3a8fff0c750b94ae2364d7a3`
has 4977171584 bytes and SHA256
`85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87`.
The earlier ordinary-runner hash was incorrect. This correction changes neither
the pinned tensor metadata nor any numerical kernel, cache layout or reference.

Lead dispatch entrypoint, on the exact clean reviewed checkout containing both
this adapter and the source fix (ordinary pinned commit must exist in local git):

Use the existing `devlab-experiment.yml` workflow with experiment input
`.github/devlab/gemma-e4b-compact-kv.json` at the exact reviewed branch/head. The
workflow permits explicit dispatch only, validates the one frozen variant, and
invokes the entrypoint below. Its 30-minute job limit includes upload/cleanup;
the experiment step has a 1610-second TERM plus 10-second KILL limit, leaving up
to three minutes for checkout and artifact handling. Other experiments keep
their existing route and timeout.

```sh
GEMMA_E4B_DEVLAB_ACK=1 GITHUB_SHA=<reviewed-full-head> \
  .github/devlab/compact-kv/run.sh /absolute/job/artifacts
```

The entire entrypoint, including materialization, SDK setup, one official-cache
checkpoint fetch/hash, CPU fixture generation, GPU checks and both model arms, is
bounded by 1790 seconds TERM plus 10 seconds KILL. The kernel process has a 420-second
sub-budget. Failure or timeout means incomplete, never a RAM win or retry request.
Candidate runs first. The inherited stable-idle and inter-arm settle requirements
must pass before baseline starts. Exact short/long/reset outputs must agree.
Report committed storage, workspace subset, RSS/high-water, GTT/VRAM, process attribution,
system/page-cache/cgroup channels and speed separately; never sum overlapping
counters. Checkpoint fetch/hash warming receipts are retained. Any useful actual
memory reduction counts; no 3 GB or 1 GB pass/fail threshold applies.
PSS and per-file residency are not collected. The one-token long-context output
does not measure sustained decode; request throughput includes end-to-end work.
Runtime invalid-setting registration negatives are not executed by this pilot.

Expected address-storage change only: 6 GiB uniform KV to 1 GiB global + 80 MiB local
rings (1.078125 GiB), a 4.921875 GiB allocation reduction before workspace, weights and
runtime overhead. Workspace is not reduced or hidden by restricting SDK shapes.
Actual committed and physical allocations remain to be measured.

Local CPU verification uses SDK 0.2.0 Python with OPENBLAS_NUM_THREADS=2 and
OMP_NUM_THREADS=2, focused `test_compact_adapter.py`/`test_compact_kv.py`, native
`loom-opt` on both generated check modules, and offline gfx1151 compilation of
scatter/gather/bitcast helpers. Do not run `loom-test` on the local machine.
