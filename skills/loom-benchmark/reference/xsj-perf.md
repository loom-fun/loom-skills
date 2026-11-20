# Windows gfx1152 performance runs

Use `xsj_perf.py` from a job on a leased Krackan host. Stage the checkout, selected package, extracted releases and original release archives before starting the job. Use the run-local Windows Python environment with `hrx-loom` and the benchmark dependencies installed. The package must have a `chat-gfx1152` preset with at least four rows. Its serving context, program hash, model manifest and exact GGUF are recorded.

```powershell
& $python 'skills/loom-benchmark/scripts/xsj_perf.py' `
  --host-alias amd-krk-1 --repo loom-fun/unsloth-Qwen3.5-4B-loom `
  --package $package --preset chat-gfx1152 `
  --loom "$venv/Scripts/loom-serve.exe" --gguf $gguf --draft $baselineDrafter `
  --rocm $rocmExe --rocm-archive $rocmZip `
  --vulkan $vulkanExe --vulkan-archive $vulkanZip `
  --out "$env:XSJ_EVIDENCE/perf"
exit $LASTEXITCODE
```

The job supplies `XSJ_RUN` and `XSJ_PORT`. The output directory must be new. Use the Windows gfx1152 ROCm release b1382 and the Windows x64 Vulkan release b12211. Keep the archive DLLs beside each executable. The runner verifies the release pins and the executable and sibling DLL contents against each archive. It checks that llama.cpp enumerates the leased GPU and selects that device explicitly.

For Qwen3.5-4B, stage the AtomicChat drafter named in the checked-in drafting classification. For Qwen3.5-9B, use the package's drafter. Omit `--draft` for Qwen3.5-2B. If the Vulkan release is unavailable, record the concrete reason with `--vulkan-unavailable` in place of its executable and archive options. When the release is available, use it and retain the evidence of any failure to run the classified drafter. A failed single-stream launch still leaves the four-stream mode eligible to run.

Each round runs the engines sequentially. A fresh one-slot llama.cpp process, or fresh Loom process serving the package's preset, measures both short and long real-text requests. The engines use the same configured context for these requests. Each four-stream measurement starts a fresh undrafted process. The ROCm four-stream process uses a shared KV pool of that capacity. The Vulkan four-stream process reserves that capacity per slot. Loom uses fixed backing and the SDK's active-wait URI. If the staged configuration uses parameter preloading, pass `--loom-arg=--parameter_mode=preload` to reproduce it.

The standard plan uses three fresh sessions per engine and mode. `--fast` uses the six-session, three-repeat plan from benchmark methodology rule 5. `--engine` and `--sessions` permit partial diagnostic runs. Each process's readiness, arguments, output and exit are recorded. A partial short/long pair stays in its session's raw log. Only a finished pair with confirmed teardown reaches the aggregate JSONL. A nonzero job status requires inspection of the per-mode result and failure logs.

The bundle contains `manifest.json`, `commands.jsonl`, per-command and server logs, `device.json`, `hrx-loom.json`, `programs_sha256.txt`, measured model manifests, `perf_*.jsonl` and `c4_*.txt`. The card importer can use the device, SDK and program stamp with the leased run ID and the workload logs. Retain the XSJ lease and job export alongside the bundle. Preserve the command output for HTTP refusals and short completions. Verify actual context-limit refusals before reporting a card row as unsupported.

The XSJ job owns the staged files, temporary directory and GPU. Use the lab run's normal cleanup to restore tracked host state after collecting evidence. Track the compiler's `LOCALAPPDATA/comgr` cache before serving because the driver can write there even with run-local cache settings.
