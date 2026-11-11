---
name: loom-setup
description: Install AMD Loom (hrx-loom from PyPI, or a ROCm/hrx-system CI wheel) and the hrx-loom-library catalog from a cold start, fetch catalog models with loom-fetch, and start/stop/inspect loom-serve on the GPU or NPU. Use for "install loom", "loom-fetch a model", "start loom-serve for X", "is the server ready", NPU serving.
---

# Loom setup: install → fetch → serve

**Bundled paths ? first shell action.** Before running commands, set `LOOM_SKILL_DIR` in the shell to the
absolute directory containing **this** `SKILL.md`, using its path from the skill listing or read tool.
In Claude Code, run `export LOOM_SKILL_DIR="${CLAUDE_SKILL_DIR}"` (the host substitutes the path).
In Codex, Pi or another host, export `LOOM_SKILL_DIR` to the loaded file's directory. Use a
shell-compatible absolute path (Git Bash on Windows). Rebind before using another skill's commands
and in every new shell; it is not the working directory or SDK root. On Windows, use native `C:/...`
paths for file tools; `/c/...` paths belong to Git Bash. Never run an unresolved placeholder. Run bundled commands by quoted absolute path; do not run `scripts/...` from the working directory
or change into the installed skill tree. Install the complete Loom suite: sibling paths are required.

Read the `loom` umbrella skill's hard rules first (memory limits, `--memory_bytes`,
binary approval).

## 1. Install (≈30 s with a warm pip cache, a few minutes cold)

```bash
bash "${LOOM_SKILL_DIR}/scripts/install_loom.sh" ~/loom            # hrx-loom 0.1.0 from PyPI (what the port templates target)
bash "${LOOM_SKILL_DIR}/scripts/install_loom.sh" ~/loom latest     # newest PyPI release
bash "${LOOM_SKILL_DIR}/scripts/install_loom.sh" ~/loom ci-latest  # newest main CI wheel (needs gh)
```
The script:
- creates `~/loom/.venv` with `pip install 'hrx-loom[hf]==0.1.0'`, which provides loom-serve,
  loom-fetch, loom-test, loom-run, loom-compile, loom-benchmark, loom-opt, loom-format, etc.
  (Linux x86-64 glibc ≥ 2.39 and Windows 11 x86-64 wheels; no ROCm SDK needed);
- clones `ROCm/hrx-loom-library` to `~/loom/library` and checks out the **catalog revision the
  SDK names** in `hrx_loom-*.dist-info/loom.json`, so catalog cards and SDK agree;
- creates `~/loom/.tools`, a venv with gguf, numpy, tokenizers, requests, jinja2, fastapi,
  uvicorn and httpx for the port, benchmark and UI scripts.

Facts:
- **Pin the version.** The `loom-port-llm` templates target the 0.1.0 text ABI (`text_*_v1`
  exports). A newer release can change it; then registration fails with a "root symbol … was
  not found" error and the templates need the diff of the upstream tiny-text fixture between
  the two tags (see `loom-port-llm`).
- `loom-fetch --list` shows the release's compatible catalog.
- **Windows:** run the script from Git Bash; venvs use `Scripts/`. See the `loom` skill's
  Windows section.
- `ci-latest` / `ci:ARTIFACT_ID` install development wheels from ROCm/hrx-system GitHub
  Actions artifacts (needs `gh auth status`; artifacts expire after about 90 days; `gh run
  download` fails on these zips, so the script uses `gh api .../artifacts/<id>/zip`).
- **NPU:** the Qwen3-Embedding NPU program compiled only with the wheel from hrx-system PR
  2259's CI (artifact 12623044407 at the time): `VENV_NAME=.venv-npu install_loom.sh ~/loom
  ci:12623044407`. If that artifact has expired, look for a newer build whose NPU path compiles.

### SDK pins (a package that needs another hrx-loom than the default)

A package whose programs need a newer SDK than PyPI 0.1.0 carries `"sdk": {hrx_loom, source, artifact_id, hrx_system_sha,
wheel_sha256, expires}` in its `llm_config.json` (a catalog package: from its `catalog.json`). Every script that installs
Loom for a package calls `scripts/ensure_sdk.sh ROOT llm_config.json`, which prints the root to use: ROOT when it already is
that wheel, else `ROOT-<artifact id>` (installed from the pinned CI artifact), so pinned and default SDKs coexist.
`LOOM_VERSION` set explicitly overrides the pin. **CI artifacts expire (~90 days; `sdk.expires`)**, and an expired pin makes
`ensure_sdk.sh` fail. Refresh it before then: `install_loom.sh ROOT-new ci-latest` (or `ci:ID`), rerun the gates on that SDK, then
`python3 scripts/sdk_pin.py stamp ROOT-new FILE` for the kit's `catalog.json` and the published card's `llm_config.json`
(republish the card; the nightly reads the pin from the card). A PyPI release, once one has the features, replaces the pin.

## 2. Fetch a catalog model

```bash
cd ~/loom && . .venv/bin/activate
L=library/model
loom-fetch $L/moonshine-ai/moonshine-streaming-tiny/model.json --list            # presets / variants (use --dry-run for sizes)
loom-fetch $L/moonshine-ai/moonshine-streaming-tiny/model.json --preset=streaming --output=models/moonshine --no-input
loom-fetch $L/Qwen/Qwen3.8-27B/model.json --preset=vision --output=models/qwen27b-vision --no-input --dry-run   # size check first
```
Notes:
- `--dry-run` prints what would be downloaded. Use it to check sizes against the
  memory budget.
- Weights come from Hugging Face; the HF cache (`~/.cache/huggingface/hub`) is reused.
  Gated repos need the user to accept the license on the HF page first.
- If fetch complains about a card/lock mismatch for a library model, add
  `--origin=library=$PWD/library` so program sources resolve against your checkout.
- Without a checkout you can fetch remote cards directly:
  `loom-fetch https://raw.githubusercontent.com/ROCm/hrx-loom-library/main/model/<org>/<name>/model.json --preset=… --output=…`
- Catalog models seen working on Strix Halo:

  | Model | Kind | Preset |
  |---|---|---|
  | Qwen3.8-27B | text + vision | `vision` |
  | moonshine-streaming-tiny | ASR (`recognition`) | `streaming` |
  | kyutai/pocket-tts | TTS (`speech`) | `streaming` |
  | krea/Krea-2-Turbo | image | `image` or `image-int8` |
  | Qwen3-Embedding-0.6B | embedding (NPU) | `embedding` |

## 3. Serve

```bash
S="${LOOM_SKILL_DIR}/scripts"
VENV=~/loom/.venv "$S/serve.sh" asr 8100 amdgpu 16000000000 ~/loom/models/moonshine/model.json
VENV=~/loom/.venv "$S/serve.sh" gpu 8100 amdgpu 53687091200 models/a/model.json models/b/model.json -- --parameter_idle_ms=60000
VENV=~/loom/.venv-npu "$S/serve.sh" npu 8101 'xdna://0?columns=8' 4000000000 models/qwen3-embedding/model.json -- --pool_backing=fixed
"$S/stop.sh" asr
```
`serve.sh` waits until `/v1/status` reports every model `registration: ready` and
checks that its PID owns the port. If it reports `failed`, it prints the compile
diagnostics and stops the server. Logs, PIDs and ports go in `~/.cache/loom-serve/NAME.{log,pid,port}`.
Use unique NAMEs. `stop.sh` refuses to kill a PID that is not a loom-serve.

Raw form: `loom-serve '--device=amdgpu://?completion_spin_ns=1000000000' --listen=127.0.0.1:PORT
--model=pkg/model.json --workers=8 --memory_bytes=N [--preset=NAME] [--parameter_idle_ms=60000]`.

**Active wait (the default on `amdgpu`).** `serve.sh` turns `amdgpu` into
`amdgpu://?completion_spin_ns=1000000000`: while GPU work is in flight, Loom's `amd-gpu-retire` thread spins on the
completion signal instead of parking on the KFD interrupt (one core while busy, 0% CPU when idle). Strix Halo firmware
that ties the data-fabric and memory clocks to CPU activity (SMU `0x0a640600`, BIOS 1.12, as on AMD DevLab) otherwise
holds FCLK/UCLK about 20% low while Loom's CPU sleeps, and Loom measures 10-45% slower (memory-bound kernels most).
llama.cpp is never affected: HIP keeps a core busy launching and waiting. With the active wait, energy per token
drops. `LOOM_ACTIVE_WAIT=0 serve.sh ...` keeps the SDK's default wait (A/B only).

## 4. Use / inspect

```bash
curl -s localhost:8100/v1/status | python3 -m json.tool      # per model: id, loom.kind, registration, compilation, parameters, status
curl -s localhost:8100/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<id from status>","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```
Endpoints by `loom.kind`:

| Kind | Endpoint | Request |
|---|---|---|
| `text` | `/v1/chat/completions` | Supports streaming and `stream_options.include_usage`. Reasoning arrives in `reasoning_content`. |
| `recognition` | `/v1/audio/transcriptions` | Multipart, `audio/pcm;rate=16000;channels=1`. |
| `speech` | `/v1/audio/speech` | Needs `input` and `voice`. `{"response_format":"pcm"}` gives 24 kHz s16le. |
| `image` | `/v1/images/generations` | `{"response_format":"b64_json","size":"512x512"}`. |
| `embedding` | `/v1/embeddings` | |

Transcription example. The server takes raw 16 kHz mono s16le PCM:
```bash
ffmpeg -loglevel error -i in.wav -ar 16000 -ac 1 -f s16le /tmp/x.pcm
curl -s localhost:8100/v1/audio/transcriptions -F model=moonshine-ai/moonshine-streaming-tiny \
  -F "file=@/tmp/x.pcm;type=audio/pcm;rate=16000;channels=1"            # -> {"text": "..."}
```
TTS: `curl -s localhost:8100/v1/audio/speech -H 'Content-Type: application/json' -d '{"model":"kyutai/pocket-tts","input":"Hello.","voice":"alba","response_format":"pcm"}' > out.pcm`
returns 24 kHz s16le mono. `voice` is required. Voices are the card's `voice/*.wav` names
(alba, marius, anna, …). `seed` is optional.

The `model` field is the card's `model.id`, not a file name. Unknown models return
404, and unsupported fields are rejected rather than ignored.

Server behavior:
- Startup compiles the program, typically 10–60 s on a first run (cached afterwards).
- The first request loads weights lazily, so exclude it from timing.
- Special tokens are stripped from decoded output text. A control program that must
  see structural tokens in the output needs them marked non-special in the
  package's `tokenizer.json`; see `loom-port-llm`.
- Upstream docs: `loom/docs/src/serving/workflows/{fetch-and-serve,serve-models,operate-model-server,observe-model-server}.md`
  in hrx-system.
