# vLLM as the gates' reference for safetensors packages

A package whose `llm_config.json` has `source.kind == "safetensors"` (for example a Quark MXFP4 checkpoint) has no GGUF, so
`gates.sh` uses vLLM ROCm serving the same checkpoint as the reference engine. Nothing else about the gates changes: the thresholds, the
near-tie rules and `compare.py`'s verdict are the same, and a failure that is only the reference's own numerics is shown as a failure on the
card with the evidence, never loosened.

- **Adapter**: `scripts/vllm_shim.py` is a llama-server look-alike (`/health`, `/props`, `/completion` with token ids in, greedy tokens + top-k
  log-probs out; `prompt_string` + `multimodal_data` for the vision gate) on top of vLLM's offline engine. Gates need only a URL.
  `scripts/vllm_server.sh` installs the pinned wheel (uv, python 3.12, `https://wheels.vllm.ai/rocm/VERSION/VARIANT`) and starts / stops it.
- **Settings**: bf16, greedy (`temperature 0`), seed 0, prefix caching off (every request is evaluated in full), one request at a time, the
  checkpoint's own quantization config (Quark MXFP4 is W4A4: vLLM emulates the fp4 activation quantization where the GPU has no native
  support), `max_model_len` = the package context + 1024 (at most 66560). `VLLM_VERSION` / `VLLM_ROCM_VARIANT` / `VLLM_EXTRA_ENV` select the build.
- **Cache**: references are cached like llama.cpp's (`llama_cache/`), keyed by `source.repo@revision`, the vLLM version (`LLAMA_CACHE_BUILD=vllm-VERSION`)
  and the request, so they do not depend on the machine running Loom: a gate job on another runner can restore the `llama_cache.tgz` a
  previous job uploaded.
  **Committed store**: `.github/devlab/reference_cache/<org>__<repo>__vllm-<VERSION>.tgz` (`scripts/reference_cache.py`: restore / merge / count; GGUF cards use `__llama-<BUILD>` the same way) holds the
  recorded references of a checkpoint; `devlab_perf.sh` restores it before every safetensors `gates` step, so a run whose requests are all cached never loads
  vLLM (its job line: `references: N cached before gates, M recorded in this job, vLLM loaded: yes|no`). A miss is recorded by that job; merge its
  `gates_V/llama_cache.tgz` into the store. Job field `reference_cache: "none"` skips the restore.
- **Card statement**: `bench/reference.json` (from `/props`: engine, version, checkpoint, revision, dtype, quantization) is what the card cites.
- **No micro-batch view**: llama.cpp's slot-cache decode-kernel view (`_micro_batch`) has no vLLM equivalent; `llama_views` leaves it out and the
  near-tie rule uses the single-batch view.
- **Fallback** (vLLM cannot run on any runner): transformers + amd-quark on CPU behind the same `/completion` shim (a second engine class in
  `vllm_shim.py`).

## Runner findings (vLLM 0.18.0 rocm700, Qwen3.8-27B Quark MXFP4, DevLab 2026-11-15)
- Install: uv + python 3.12 venv + the vllm.ai ROCm wheel, 35-80 s per runner (once; the venv is reused by later jobs). The runners lack three things the
  install adds: libopenmpi3t64/libnuma (apt, or downloaded .debs unpacked into the venv where there is no sudo), Python.h (python3.12-dev, else uv's
  standalone python), and hipcc + HIP headers (the TheRock `rocm[libraries,devel]` pip packages) for Quark's MXFP4 emulation, which JIT-compiles HIP kernels.
- R9700 (32 GB): the packed weights stay packed (18.1 GiB resident; vLLM dequantizes per forward), engine up in about 70-330 s (weights download included
  the first time: 332 s), peak VRAM 29.9 GiB at gpu_memory_utilization 0.92 and max_model_len 66560, host MemAvailable never below 11 GiB, RssAnon 1.7 GiB.
- Run-to-run determinism: the 22 dev-golden prompts recorded twice give identical tokens (22/22), so no `--alt` second recording is needed.
- Strix Halo: Loom's gates run there from references recorded on the R9700 (`reference_cache`, 234 s); vLLM itself did not come up on gfx1151 with the image
  tower enabled (the encoder-profile attention uses torch's math SDPA and asks for a 256 GiB matrix; a smaller `max_num_batched_tokens` does not change it).
  References are recorded on the R9700 and reused.
