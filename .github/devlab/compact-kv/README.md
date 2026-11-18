# Optional compact E4B cache experiment

This adapter is for one reviewed Strix Halo job, SDK020/gfx1151, exact Gemma4 E4B
Q4_K_M text checkpoint, context65536/prefill1024/rows1. It changes no default
program. No GPU result, physical-memory saving or production parity is claimed.

The source remains experimental and default OFF. Both comparison cards explicitly
use max_sessions1, session/prefix/checkpoint cache capacity0 and cache target0.
There is no draft, vision, fork/history-copy, dynamic-row-growth or multi-session
support. Build and runtime guards reject unsupported configurations. The source
unit suite includes these guards and the real no-helper-kernel audit regression.

`generate_checks.py` extracts actual emitted legacy and compact kernels plus all
called helpers. `kernel_pins.json` rejects any changed numerical source before a
GPU launch. Symbols/exports alone are renamed; negative variants alter only the
explicit address recipes. F64 attention over F16 history uses independent owner
lists/page maps and all8Q/2KV heads. Q rounding is modeled separately for flash.
Synthetic producer checks include local/full and global/cut-frequency Neox RoPE,
RMS normalization, F16 stores, live-variant stores and shared store-disabled use.
All GPU arenas, including masked/future lanes, are initialized to finite values.

Fourteen positive cases cover local window and ring boundaries, a1024-token
write-before-read overlap, position65535, shuffled full-context global pages,
nonzero private origins, stale-slot reset and reversed listed-decode dispatch.
Each checks independent goldens and exact legacy/compact F32 output bits. Four
strict negatives use undersized512/1024 rings, old V-plane stride and missing
fork-history copy. Scatter destinations are unique and within initialized arenas;
CPU controls prove comparator sensitivity. Every negative has an identically
seeded, separately passing legacy/correct-compact positive control. A negative
contains no baseline launch that could accidentally satisfy its expected failure.

`run_checks.py` requires exact positive samples/expectation counts and strict
EXPECT/006 raw approximate-comparator failures. Missing samples, skips, planning
errors, compile errors, XPASS and mismatched diagnostics stop before model arms.
No tolerance is widened:3e-3 absolute/relative attention and F16 store goldens,
with exact candidate/legacy outputs. These are synthetic kernel checks; the
matched full-model phase uses the pinned real checkpoint and meaningful outputs.

`materialize_runner.py` derives, with one-match source anchors and a saved diff,
from ordinary runner7abe4277c4ff30195592f80ff170df1b3ad19985. Monitoring, status,
readiness retry, stop and settle implementations are preserved. Changes are:
matched64k/one-row configs and no-cache policy, the kernel gate before either
server,65536-token retrieval plus over-limit rejection, and a fresh-session short
answer after the long request. The ordinary launcher's files are not edited.
The original `run_arm` body changes only65535 to65536; focused tests prove this.

Lead dispatch entrypoint, on the exact clean reviewed checkout containing both
this adapter and the source fix (ordinary pinned commit must exist in local git):

```sh
GEMMA_E4B_DEVLAB_ACK=1 GITHUB_SHA=<reviewed-full-head> \
  .github/devlab/compact-kv/run.sh /absolute/job/artifacts
```

The entire entrypoint, including materialization, SDK setup, one official-cache
checkpoint fetch/hash, CPU fixture generation, GPU checks and both model arms, is
bounded by1790 seconds TERM plus10 seconds KILL. The kernel process has a420-second
sub-budget. Failure or timeout means incomplete, never a RAM win or retry request.
Candidate runs first. The inherited stable-idle and inter-arm settle requirements
must pass before baseline starts. Exact short/long/reset outputs must agree.
Report committed storage, workspace subset, RSS/PSS, GTT/VRAM, process attribution,
system/page-cache/cgroup channels and speed separately; never sum overlapping
counters. Checkpoint fetch/hash warming receipts are retained. Any useful actual
memory reduction counts; no3GB or1GB pass/fail threshold applies.

Expected address-storage change only:6GiB uniform KV to1GiB global+80MiB local
rings (1.078125GiB), a4.921875GiB allocation reduction before workspace, weights and
runtime overhead. Workspace is not reduced or hidden by restricting SDK shapes.
Actual committed and physical allocations remain to be measured.

Local CPU verification uses SDK020 Python with OPENBLAS_NUM_THREADS=2 and
OMP_NUM_THREADS=2, focused `test_compact_adapter.py`/`test_compact_kv.py`, native
`loom-opt` on both generated check modules, and offline gfx1151 compilation of
scatter/gather/bitcast helpers. Do not run `loom-test` on the local machine.
