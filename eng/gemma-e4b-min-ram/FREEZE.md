# Frozen DevLab submission

This branch prepares one serial, target-only Gemma 4 E4B comparison for the official `devlab-experiment` workflow. It does not dispatch or start GPU work. The kit in this branch was generated afresh with the official `reference/apply_example.sh gemma4` command from clean `bacdd3ed241eb62f9578b909ef63eaa0751f3c49`; its kit hash is `e40fd8d3d77174bf809860ae91f6db2214a9e207dad341b663dc3af22f0127fd`, its builder hash is `faeb722cee49d011cd3422d79832dfb9533710845f7f9e8041b841ff9d363db4`, `dirty=false`, and `modified=false`. The absolute KB path emitted by the helper was removed from `KIT_SOURCE` for portability; the job explicitly sets `LOOM_KB_ROOT` to the same clean pinned checkout. The earlier dirty kit `2491e254...` remains only in the pre-review source artifact outside this branch and is not used by this experiment.

## Arms and claims

- Exact pinned `gemma-4-E4B-it-Q4_K_M.gguf` checkpoint; one session, no assistant/drafter or vision tower.
- Baseline: context 131,072, 4 rows. Candidate: context 65,536, 1 row. Both use the same target-only config, tokenizer, no draft, 1,024-token prefill setting, and current Loom SDK 0.2.0 pin.
- Each arm checks status and token accounting, runs a short Paris oracle, then a planted-answer retrieval prompt calibrated to exactly 65,535 server-counted input tokens and one generated token. Only HTTP 503 from that first short-answer request may retry, for at most 120 seconds; every attempt records status, body, latency and elapsed readiness time. No server restart or invisible warmup.
- Both arms must return identical short and retrieval text for identical prompt hashes, and satisfy the semantic Paris/blue checks. The existing #554 exact-checkpoint llama.cpp golden/compare run is provenance and a prior reference, not represented as a new runtime check in this job.
- Per-arm monitor validation requires a clean stop marker, nonempty sample set, exactly one stable matching `loom-serve` PID/start-ticks identity, selected `gfx1151` from `/v1/status`, one AMD sysfs card, and at least one non-null memory counter. Missing host channels are marked `unknown`; `total_ram_verified` is always false because RSS, UMA/GTT, page cache and cgroup observations can overlap.

## Fetch and time bounds

The runner uses the official pinned SDK installer. It reuses the standard Hugging Face cache and may fetch the exact 4,977,171,584-byte checkpoint once if absent; it records cache-hit status, lookup/fetch elapsed time, and file-size transfer estimate on a miss (the Hub helper does not expose actual network bytes for resumable transfers). Final package preparation hashes the actual file and binds the result to device/inode/size/mtime. The exact 15,142,283-byte tokenizer is fetched from its pinned maintained-card revision and SHA-checked. No cache eviction or global memory pressure is used. The already-approved job ceiling is 1,800 seconds; this includes SDK installation, fetch, both package builds, startup, both requests, one bounded inter-arm settle, and cleanup. A cold checkpoint fetch has previously been observed at about 73.8 MiB/s in #554, but this is not a speed guarantee. If setup or either arm misses the deadline, preserve the failure evidence; do not resubmit automatically.

Checkpoint hashing warms model file pages before the per-arm monitor begins. The job retains pre/post-hash system memory snapshots and reports the confound; same-file relative arm differences are useful, while absolute inference-attributable total RAM remains bounded/unknown rather than claimed from RSS or GTT alone.

Before arm1, the runner takes five samples of VRAM and GTT used on the single AMD PCI device within a 15-second process bound and requires their ranges to be within 16 MiB. Between arms, it records the same device's raw VRAM/GTT and host MemAvailable samples for at most 60 seconds. Arm2 starts only after five consecutive samples both return within 64 MiB of the pre-arm baseline and remain within a 16 MiB range. Host availability is observational only: it is not a settle gate and is never used to force cache eviction. On timeout, candidate evidence remains, baseline is marked `contaminated_not_run`, and no memory win is reported.

At startup, the job verifies `HEAD == GITHUB_SHA` (when running under Actions) and that the checkout has no tracked modifications. It uses the pinned clean-source `skills/loom-setup/scripts/{serve,stop,memcheck}.sh` helpers and verifies their hashes before arm1.

## Workflow route

After lead push/review, dispatch the existing workflow from this branch with:

```sh
gh workflow run devlab-experiment.yml \
  --ref codex/issue-campaign-20261117-gemma-e4b-min-ram \
  -f experiment=.github/devlab/gemma-e4b-min-ram.json
```

The JSON contains exactly one variant. The branch-only workflow override uses the explicit Strix Halo runner, caps the job at 30 minutes, uploads only the sanitized `out/` evidence, and removes the job workspace afterward. The normal workflow dispatch path is not changed on `main`.

## Local review completed

- The original 10 CPU behavioral tests pass, including readiness retry/non-503/timeout and monitor identity/device/counter rejection cases. The inter-arm settle follow-up has a separate mocked success/timeout test.
- Shell syntax checks pass for `run_job.sh` and `run_frontier.sh`; an `env -u HF_HOME` prefix check confirms the default HF cache path initializes under `set -u`; `git diff --check` is clean.
- No GPU execution, weight download, local cache reset, or checkpoint read was performed in this branch work.

## Terminal setup correction

The first dispatched run (`44205258468`) failed before model execution or weight fetch because Actions supplied `HF_TOKEN` as an empty string and Hugging Face Hub emitted the invalid `Bearer ` header. The checkpoint fetch paths now share a wrapper that maps unset or empty tokens to `None` while preserving a nonempty token. A mocked invocation test covers unset, empty, and present values; it passed without network access. This setup failure provides no memory or performance result. The run consumed 17 seconds after setup began; it did not reach either arm or the blue/Paris correctness gate.
