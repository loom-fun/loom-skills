# Measurement on leased XSJ hardware

XSJ is an official evidence source where it is the only lab measurement option for the device. Use this route for a device with `runner.backend: "xsj"` and no DevLab labels in its registry row.
Launch measurements with the installed XSJ lab workflow. Use the same package, work, sampling, correctness
thresholds and baseline configuration defined in the benchmark methodology (`skills/loom-kb/method/methodology.md`).

## Lease and run

1. Check lab access with `lab whoami` and inspect the lab's pool for the target architecture. Select free
   leasable hosts. The device registry identifies the architecture, and the lab inventory supplies current
   availability. Each leased host takes one active run and one engine or GPU workload at a time. When several
   hosts are in scope, split independent model jobs across them. Keep every engine in an initial comparison
   on the same host.
2. Run `lab init` before any changes. Inventory the GPU, driver, OS, RAM, Python, free disk and any paths the
   compiler or runner writes outside the run. Track those paths, including any pre-existing compiler cache.
   Keep dependencies, downloads, servers and output in the run directory. Ship an immutable kit revision
   and resolve public Hub and binary release revisions ahead of execution. Download on the leased host
   without copying local credentials there. If a large bundle transfer drops, a leased host can fetch
   the same immutable revision from a public repository. Verify the checked-out commit or the source
   archive's commit metadata inside that run.
3. Use `lab job` for durable work and `lab wait --save` for its observed exit and log. Exercise the exact
   package and preset that the card will advertise. Record the admitted context and deliberately run a
   completion near that limit. Preserve single-stream drafting classification and disable drafting in the
   four-stream suite on every engine. Save the reason and observed failure when a workload or backend is
   unavailable. Keep the correctness gate threshold and near-tie rules unchanged.
4. Collect artefacts with `lab fetch` before cleanup. Record the run ID, host alias, hardware and driver,
   shipped kit revision, checkpoint revisions, program digest, SDK identity and pinned baseline binaries
   and hashes. Keep raw timing samples, launch arguments, gates, any unsupported rows and the lab report.
   Save an evidence manifest and retain the reservation and job records as the independent trace of the lease.
5. Run `lab diff`, restore the tracked state with `lab cleanup`, verify cleanup and release the host.
   For an interrupted run, inspect `lab status` and resume or clean the specific run while its lease is
   active. Keep the backups if cleanup reports residual changes. If a lease was reassigned, use the lab
   workflow's discard and handoff rather than touching the reassigned host.

## Ingest and publish

Use the per-device data under `fleet/devices/` with the card generator's XSJ evidence source. Compare the
recorded program digest with the exact publishing package before ingest. Carry device and SDK identity
from the job, which allows the record to be traced back to the lease and raw evidence. Put the XSJ
preset only on models with a verified fit. Render and validate the updated card, review it, stage it,
and verify the live program digest, source and preset after upload. Where the GPU cannot serve a
benchmark context or an engine cannot meet the baseline configuration, report the corresponding cell
with its capability reason. Measurements on one host describe that host's configuration.

Follow the [Windows comparison runner recipe](../../loom-benchmark/reference/xsj-perf.md) and the
[per-device evidence ingestion reference](../../../fleet/devices/README.md). For a near-limit completion,
allow enough time for both the probe and its HTTP read deadline. Retain the actual result so an HTTP client
timeout and a server refusal can be distinguished.

On Windows, use Git Bash where an existing benchmark harness expects Bash, or a native Windows
orchestrator using the same workloads. Launch Windows processes under the lab job, check native exit
codes in PowerShell, and stop only the run's own processes. Point Python explicitly at a real interpreter
if the stock command opens the Microsoft Store.

When launching Git Bash from PowerShell, set `SHELL=/usr/bin/bash` inside Git Bash before
invoking the lab wrapper. OpenSSH starts the ProxyJump helper with the inherited shell. If it
inherits PowerShell, the helper can fail on the Bash `exec` command before negotiating a host key.
Check the original lease attempt has ended and verify the dashboard state before retrying.
