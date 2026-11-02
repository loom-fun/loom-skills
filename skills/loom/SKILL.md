---
name: loom
description: Entry point for AMD Loom work (hrx-loom SDK, loom-serve, loom-fetch, Loom IR kernels) on AMD GPUs/NPUs such as Strix Halo — installing, serving catalog models, the demo web UI, porting a new LLM to Loom, optimizing it while preserving correctness, and benchmarking against llama.cpp. Use whenever the user mentions Loom, loom-serve, loom-fetch, hrx-loom, .loom files, or evaluating/porting models for Loom.
---

# Loom (umbrella)

Loom = AMD's JIT compiler + runtime + model server (`hrx-loom` on PyPI, 0.1.0 pinned by these skills; source at
github.com/ROCm/hrx-system, models at github.com/ROCm/hrx-loom-library). Models
ship as **Loom IR source** (`.loom` kernels + command/control programs) plus a
package card (`model.json`) that points at unmodified checkpoints (GGUF or
safetensors). `loom-serve` compiles the program for the live device at startup
and exposes OpenAI-style HTTP endpoints.

## Pick the sub-skill

| Task | Skill |
|---|---|
| Install the SDK + library, fetch a catalog model, start/stop `loom-serve`, NPU serving | `loom-setup` |
| Browser playground (chat / ASR / TTS / image / embeddings) over running servers | `loom-demo-ui` |
| Make an arbitrary GGUF LLM servable by loom-serve (write the program) | `loom-port-llm` |
| What is possible for this model? Pick optimization targets (roofline vs llama.cpp) | `loom-roofline` (`${CLAUDE_SKILL_DIR}/../loom-roofline/SKILL.md`) |
| Make a ported model faster without breaking correctness | `loom-optimize` |
| Correctness gate vs llama.cpp, throughput methodology, reporting | `loom-benchmark` |
| Improve these skills themselves (faster end-to-end runs, fewer failures) without regressing KPIs | `loom-improve` (`${CLAUDE_SKILL_DIR}/../loom-improve/SKILL.md`) |

All paths in these skills assume the install root `~/loom` (SDK venv `~/loom/.venv`,
tools venv `~/loom/.tools`). Substitute your own root consistently. The scripts take overrides:
`LOOM_ROOT=` (install root for port.sh, gates.sh, pick_llamacpp.sh), `VENV=` (serve.sh,
run_kernel_tests.sh), `PY=` (pick_llamacpp.sh's python), `UI_VENV=` or the 2nd argument
(start_ui.sh), `LOOM_SERVE_STATE=` (serve.sh PID/log dir) and `LLAMACPP_STATE=`
(llamacpp_server.sh PID/log dir). On a shared machine set the state dirs inside your work
dir and give `pick_llamacpp.sh` a SEARCH_ROOT (3rd argument) so it only sees builds the
user approved.

## Windows

The whole pipeline (install, port, kernel tests, golden gate, perf, serve) works on
Windows 11 on Strix Halo. Run the scripts from **Git Bash**. Differences:
- Venv executables are in `Scripts/`, not `bin/`. Read every `~/loom/.venv/bin/X`
  and `~/loom/.tools/bin/python` in these skills as `~/loom/.venv/Scripts/X` and
  `~/loom/.tools/Scripts/python`. The scripts detect this themselves.
- The SDK is the same PyPI release (`hrx_loom-0.1.0-py3-none-win_amd64.whl`); it needs the
  Microsoft Visual C++ v14 x64 redistributable. `--device=amdgpu` reports `driver: amd,
  backend: libamdf` in `/v1/status`.
- Python: `install_loom.sh` uses `py -3` (override with `PYTHON=`). On Python < 3.15,
  text files default to cp1252, so open text files with `encoding="utf-8"`.
- `ss`, `pgrep` and `/sys` don't exist. Use these instead:
  - ports: `netstat -ano -p tcp | grep LISTENING`;
  - processes: `ps -W | grep -E 'loom-serve|llama-server'`;
  - GPU memory: `llama-server --list-devices` shows the HIP-visible VRAM (91.6 GiB
    on a 128 GB box with a 32 GiB BIOS carve-out).
- `build.py` symlinks the GGUF. Without Developer Mode it hard-links it instead,
  which needs the package and the GGUF on the same drive.
- **llama.cpp:** lemonade installs builds in `~/.cache/lemonade/bin/llamacpp/{rocm,rocm-stable,rocm-preview}`;
  one missing its HIP/BLAS DLLs silently runs on CPU (`llamacpp_server.sh` refuses it, see `loom-benchmark`).

Typical cold-start order: `loom-setup` → `loom-port-llm` (`port.sh`) → `loom-benchmark`
(`qualify.sh`: gates, baselines, roofline targets) → `loom-optimize` → `finalize.sh`.

**Time budget.** With no target and no time given, an end-to-end job ("get X working,
optimize it, report") gets **this model's roofline budget for the whole job**, first command to
published report (`loom-roofline/e2e_budget.py`: the pipeline's workload at this model's speed plus a
fixed allowance; `qualify.sh` gives optimization what is left, from the clock `port.sh` starts). A
stated time or target replaces it (`BUDGET_H=` / `E2E_BUDGET_MIN=`, `loom-roofline` section 0);
"optimize this model" on an existing port gets **30 min**. Announce the targets and the budget
without asking, iterate in `loom-optimize`, and act on the verdict `perf.py` prints (CONTINUE /
REVIEW / BUDGET / DONE, `loom-roofline` section 5). Report met or short for each metric, with the
approach curve.

## End-to-end requests: one turn, four commands

"Get X working, optimize it and report" is a single job. Run it without stopping:
```bash
K=${CLAUDE_SKILL_DIR}/..; W=~/work/<name>; export LOOM_ROOT=~/loom LOOM_SERVE_STATE=$W/state
bash $K/loom-setup/scripts/install_loom.sh $LOOM_ROOT          # only if $LOOM_ROOT/.venv/bin/loom-serve is missing
$K/loom-port-llm/port.sh <org>/<repo> $W 8160                   # 1. port: download, inspect, build, serve, kernel tests, smoke (~1 min)
G=$(ls $W/gguf/*.gguf)
$K/loom-benchmark/scripts/qualify.sh $W/pkg http://127.0.0.1:8160 $G $W/bench   # 2. llama.cpp pick, gates, baselines, roofline, budget
#                                                               # 3. optimize (loom-optimize) for the budget qualify printed
$K/loom-benchmark/scripts/finalize.sh $W/pkg http://127.0.0.1:8160 $G $W/bench  # 4. final perf, full gates, audit, report.html
```
1. **Port.** `port.sh` stops with an exit code naming the stage that needs hand-work
   (3: the chat template, or an architecture the starter doesn't cover). Then follow
   `loom-port-llm` for that stage only, and continue with step 2 once the package
   serves and its kernel tests pass.
2. **Qualify.** `qualify.sh` picks the fastest llama.cpp ROCm build on the machine, measures
   its baseline with Loom stopped, runs the gates (`LEVEL=quick`; llama.cpp references are filled
   with Loom stopped), measures the Loom baseline, runs the roofline (targets + budget) and prints
   `QUALIFY_RESULT`. A failing gate doesn't stop the measurements, but fix it before finalize.
   Exit 4 `NO_LLAMACPP`: no installed llama.cpp runs this architecture. Ask the user once whether to
   download a newer lemonade build (it needs their approval); otherwise continue Loom-only, and state
   in the report that correctness is not verified against a reference. The roofline table
   (`bench/roofline.json`) and the approach curve (`bench/targets.jsonl`) are
   required in the report; this step creates both.
3. **Optimize** (`loom-optimize`), under the rate controller, for the budget printed
   by qualify. After each kept change: rebuild, restart, `perf.py`, and
   `LEVEL=fast gates.sh` (≈1–3 min: the llama.cpp references are cached).
   Stop at BUDGET even if a lever is half done: revert to the last kept build.
4. **Finalize.** Serve the best build, run `finalize.sh`. It runs the full gates
   (with the full-context probe), the final perf, `audit.py` (must print
   `AUDIT PASS`) and writes `bench/report.html` + `report.md` from the logs. Add a
   short notes file (what was optimized, what was skipped and why) with
   `report.py --notes`, then publish `report.html`.

The budget is real: keep tool calls short, don't re-run a passing step, and don't
re-read files you already read. Long steps (`qualify.sh`, `finalize.sh`) print a
timestamp per stage (run them in the background with output to a log, then
`loom-benchmark/scripts/wait_log.sh LOG '^QUALIFY_RESULT|^FINALIZE '`; never `until ! pgrep -f ...`, which
matches itself); while one runs, prepare the next step instead of waiting idle: read,
write generator code, build packages. **Don't use the GPU while they run** (no A/B, perf or second
server until QUALIFY_RESULT): a Glimmer trial that ran A/Bs during qualify's gates slowed the
llama.cpp references (qualify 41 instead of ~30 min) and got A/B numbers too noisy to use.

Stop to ask only for what the hard rules require: a gated-license click, or approval of a
binary from a source other than PyPI and lemonade-sdk/llamacpp-rocm. Avoid both where possible:
- **Reuse what is installed.** Check `~/loom/.venv/bin/loom-serve` (Windows: `.venv/Scripts/loom-serve.exe`) and the HF cache.
- **Baseline = the fastest llama.cpp ROCm build** (latest lemonade release or an installed one):
  `qualify.sh` fetches the latest and picks the fastest (`loom-benchmark` section 0).

The `loom-roofline` skill is at `${CLAUDE_SKILL_DIR}/../loom-roofline/SKILL.md`.
Read it from there if it isn't in your skill list.

## Hard rules (learned the expensive way)

- **Never load a model whose weights exceed the GPU's GTT/unified-memory budget.**
  On the Strix Halo box (128 GB RAM) the GPU limit is ~61 GiB GTT. Qwen3.8-Flash-Next
  (104 GiB) hard-crashed the machine during the original evaluation. Never retry it on a 128 GB box. Check
  `du -shL <weights>` against `cat /sys/class/drm/card*/device/mem_info_gtt_total`
  before serving anything big. The scripts load **one model at a time** (Loom is stopped while
  llama.cpp runs), but `starter/pick_gguf.py` still keeps a GGUF that fits twice (a 40B repo's
  UD-Q5_K_XL instead of its 39 GiB Q8_0) until quant choice accounts for the KV pool at full context.
- **GPU memory is system RAM** (unified memory: GTT reports ~115 GiB on a 128 GB box, but the desktop
  and every process share it). Before loading any extra llama-server or loom-serve by hand, check
  `MemAvailable` in /proc/meminfo exceeds that model's weights + KV + 16 GiB: a second flint
  llama-server next to a 79 GB Loom server ran the machine out of RAM and the OOM killer took the
  desktop session.
- **Always pass `--memory_bytes`** to `loom-serve` (weights + KV pool + ~2 GiB
  workspace). Too small → HTTP 503 "retained state leaves insufficient completion
  capacity". Several models can share one budget with `--parameter_idle_ms=60000`.
- **Freshly downloaded binaries need explicit user approval before running**, except the two
  pre-approved sources: Loom from PyPI (pip) and lemonade-sdk/llamacpp-rocm releases
  (`fetch_llamacpp.sh`). Don't stop to ask about those.
- Gated Hugging Face models: give the user the exact model page URL so they can
  accept the license; don't try to work around the gate.
- Every chat/completions POST needs `Content-Type: application/json`. (Python
  `requests` with an empty `Expect:` header breaks the connection.)
- The first request after startup triggers lazy weight loading (seconds to a
  minute); don't time it.
- **GPU queues run out before memory does.** Each loom-serve, loom-test and
  llama-server process takes GPU user queues; with about six alive (other users'
  jobs count) a new one fails with `RESOURCE_EXHAUSTED ... user_queue_create(SDMA)`.
  Stop servers you no longer need before kernel tests or a new server.
- Stop servers by PID file (`loom-setup/scripts/stop.sh`). `pkill -f "...port..."`
  can match and kill your own shell.
- After a rebuild, make sure the old server is really gone. `serve.sh` refuses a
  busy port and checks that its own PID owns the port before printing READY.
  READY after about 1 s is normal, since compiled kernels are cached and weights
  load on the first request. A stale server shows up as a READY from a PID that
  isn't yours, or as old behavior.
- Before starting llama-server or loom-serve, check the port is free (`ss -ltnp`; Windows: `netstat -ano`).
- Golden references must be external (llama.cpp ROCm or the model's own reference
  implementation), never a numpy model you wrote yourself.

## Facts about this hardware (Strix Halo, Ryzen AI Max+ 395)

- GPU gfx1151 (RDNA 3.5), wave32, WMMA f16→f32 (~59 TFLOP/s peak), ~216 GB/s
  measured DRAM bandwidth, 32 MB MALL (last-level cache), ~61 GiB GTT.
- NPU: XDNA (`--device='xdna://0?columns=8' --pool_backing=fixed`). Only some
  catalog models (e.g. Qwen3-Embedding-0.6B) run there, and it needed an older
  SDK wheel (see `loom-setup`).
- `loom-benchmark` hardware counters are UNIMPLEMENTED on this GPU. Profile by
  ablation (remove or double a kernel, measure end to end) instead.

## Where things are documented upstream

- The docs site is rocm.github.io/hrx-system/loom. The same docs are in the repo:
  `git clone --depth 1 https://github.com/ROCm/hrx-system` and open
  `loom/docs/src/`.
  - `guide/` covers the IR: values, buffers/views, kernels, vectors, facts,
    and command programs.
  - `serving/model-programs.md` covers package authoring.
  - `serving/workflows/fetch-and-serve.md` and `live-model-development.md`
    cover the serving workflows.
  - `workflows/test-correctness.md` covers `check.case` tests.
- Each tool prints an agent-oriented manual: `loom-test --agents_md`,
  `loom-compile --agents_md`, `loom-serve --agents_md`, `loom-opt --agents_md`,
  `loom-run --agents_md`, `loom-format --agents_md`.
- Library model programs are the best worked examples. For example,
  `library/program/text/qwen3_8/chat.loom` is a ChatML control program with
  `<think>` parsing, and `library/model/*/` holds the cards.
