---
name: loom-port-image
description: 'Port a diffusers text-to-image model (DiT transformer, text encoder, VAE decoder) to AMD Loom and loom-serve /v1/images/generations: a same-noise diffusers oracle, per-stage kernel tests, a model.json card with presets, digest-pinned image_check gates, PSNR/LPIPS/CLIPScore/PickScore quality with bootstrap intervals, block-alternating local timing against stable-diffusion.cpp, thenoise and diffusers to guide optimization, and a loom-fun Hugging Face card with accuracy tables. Use for "port FLUX/Z-Image/SD-style model X to Loom", "time my Loom image model against sd.cpp", and "publish an image card".'
---

# Port a text-to-image model to Loom

**Bundled paths — first shell action.** Before running commands, set `LOOM_SKILL_DIR` in the shell to the
absolute directory containing **this** `SKILL.md`, using its path from the skill listing or read tool.
In Claude Code, run `export LOOM_SKILL_DIR="${CLAUDE_SKILL_DIR}"` (the host substitutes the path).
In Codex, Pi or another host, export `LOOM_SKILL_DIR` to the loaded file's directory. Use a
shell-compatible absolute path (Git Bash on Windows). Rebind before using another skill's commands
and in every new shell; it is not the working directory or SDK root. On Windows, use native `C:/...`
paths for file tools; `/c/...` paths belong to Git Bash. Never run an unresolved placeholder. Run bundled commands by quoted absolute path; do not run `scripts/...` from the working directory
or change into the installed skill tree. Install the complete Loom suite: sibling paths are required.

The goal is a package directory that `loom-serve --model=PKG/model.json` serves as an
OpenAI-style `/v1/images/generations` endpoint:
- the unmodified publisher checkpoint (BF16 safetensors), pinned by sha256 in `model.json.lock`
  and fetched from the publisher's repository, never re-uploaded;
- the same seed gives the same image, and the image matches diffusers given the **same initial
  noise** to within the bf16 reference's own spread (PSNR, LPIPS);
- `image_check` passes: decoded images bit-exact against pinned digests, invalid requests
  rejected, clean shutdown, plus the `--lifecycle` and bounded-workspace presets;
- a quality comparison against diffusers, and local timings against stable-diffusion.cpp,
  thenoise and diffusers on the same machine to steer optimization (they stay with your logs).

Start from a pinned hand-written card in ROCm/hrx-loom-library. Each card's README describes
its program composition. Reuse catalog motifs from the same revision.

- [Krea 2 Turbo in the v0.3.0 catalog](https://github.com/ROCm/hrx-loom-library/tree/279afe368b82e0d7acfe38c4f47e8e7e5cfb7bc8/model/krea/Krea-2-Turbo)
  uses an eight-step DiT, Qwen3 text encoder and VAE, with an INT8 variant.
- [FLUX.2 Klein 4B](https://github.com/ROCm/hrx-loom-library/tree/ccd62d6a49576f25efdb7deef7b6f81a7b76fd6c/model/black-forest-labs/FLUX.2-klein-4B)
  and [Z-Image-Turbo](https://github.com/ROCm/hrx-loom-library/tree/ccd62d6a49576f25efdb7deef7b6f81a7b76fd6c/model/Tongyi-MAI/Z-Image-Turbo)
  come from the pinned revision of [upstream PR #126](https://github.com/ROCm/hrx-loom-library/pull/126).

To check out the same Z-Image catalog source used in DevLab, create a new catalog directory and
fetch the immutable revision. For Krea 2, use revision `279afe368b82e0d7acfe38c4f47e8e7e5cfb7bc8`
and path `model/krea/Krea-2-Turbo`.

```bash
CATALOG="$PWD/hrx-image-catalog"
git init -q "$CATALOG"
git -C "$CATALOG" fetch --depth=1 https://github.com/ROCm/hrx-loom-library.git ccd62d6a49576f25efdb7deef7b6f81a7b76fd6c
git -C "$CATALOG" checkout --detach FETCH_HEAD
CATALOG_CARD="$CATALOG/model/Tongyi-MAI/Z-Image-Turbo"
```

## Prerequisites

- Install the pinned `hrx-loom` 0.3.0 wheel with `loom-setup`. Keep the installer record
  `loom-install.json` with the test evidence. On Linux x86-64 (glibc ≥ 2.39) the release wheel
  SHA-256 is `6252bb0df262351175bdb2c4aaca22ffa63d80d791659aa4d2fa5c92a0af426f`.
- A reference Python with PyTorch ROCm, `diffusers`, `transformers`, `open_clip_torch` and `lpips`
  for the oracle and quality metrics. Keep it separate from the SDK venv. The Z-Image check uses
  PyTorch `2.14.0+rocm10.1.0rc2`, Diffusers `0.43.0`, Transformers `5.24.0`, NumPy `2.5.5`,
  scikit-image `0.26.0` and LPIPS `0.1.4`.
- Optional, for the comparison: [Lemonade](https://github.com/lemonade-sdk/lemonade) with its
  stable-diffusion.cpp (ROCm and Vulkan) and [thenoise](https://github.com/lemonade-sdk/thenoise) backends.

```bash
LOOM_IMAGE_SDK="$PWD/loom-image-sdk-0.3.0"
bash "${LOOM_SKILL_DIR}/../loom-setup/scripts/install_loom.sh" "$LOOM_IMAGE_SDK" 0.3.0
source "$LOOM_IMAGE_SDK/.venv/bin/activate"
loom-serve --version
```

## Workflow

1. **Oracle with Loom's noise.** Loom draws the initial latents with Philox4x32-10
   (`motif/tensor/random.loom`). `scripts/philox_noise.py` reproduces that draw in NumPy for a
   latent shape and seed; `scripts/oracle_diffusers.py` runs the diffusers pipeline from those
   latents and saves the image plus per-stage tensors (prompt embeddings, each transformer step,
   the final latent). Use `--offload-text-encoder` for the Z-Image Strix Halo reference.
   This keeps the denoiser on ROCm and decodes on CPU in FP32 using the original VAE input
   precision and scaling. Match the decode device and precision when comparing oracle runs.
   Without shared noise, compare distributional scores.
2. **Read the checkpoint, not the paper.** List every tensor (`safetensors` header) and map it to a
   `command.parameter` name before writing the card. Pin geometry rules from the pipeline code
   (patch size, latent channels, allowed sizes) rather than from documentation.
3. **Kernels by stage, smallest fixture first.** Text encoder block, one transformer block, one VAE
   tile. Each kernel gets a `loom-test` against a fixture cut from the oracle's stage tensors, with
   errors reported relative to the output RMS so near-zero rows do not dominate. Reuse library
   motifs (`linear_tiled`, `linear_wide`, `attention_expanded`, `rmsnorm`, `rotary`, `modulation`);
   one kernel per contraction shape keeps shape scalars out of the device ABI.
   Motif tests link their references and dependencies explicitly, for example
   `loom-test linear_wide_test.loom --library=bf16_reference_test.loom --library=linear_reference_test.loom --library=../linear_wide.loom --library=../linear_tiled.loom --device=rdna://0`.
   On Git Bash set `MSYS_NO_PATHCONV=1` so `--library=../x.loom` is not rewritten.
4. **Card.** `model.json` (variants, presets such as `image`, `image-bounded-workspace`, `lifecycle`)
   and `model.json.lock` via `loom-fetch --update-lock`. `scripts/loom_deps.py program/` prints the
   BUILD dependency list per program file. Follow the directory convention in
   [docs/model-authoring.md](https://github.com/ROCm/hrx-loom-library/blob/main/docs/model-authoring.md).
5. **Gates.** Copy `test/image_check.py` from the closest card, generate `test/reference.json`
   digests from a run you have checked against the oracle, then run `image_check`, `--lifecycle`
   and the bounded preset. Pinned digests are only meaningful after step 6 passes on the same build.
6. **Quality against the oracle.** `scripts/metrics.py` scores a directory of cases
   (`<prompt>_s<seed>_<WxH>/image.png`): PSNR and LPIPS against diffusers fp32 and bf16 on the same
   noise (pass `parity TEST REF --same-noise` after checking the initial latent), CLIPScore
   (open_clip ViT-L-14/openai) and PickScore (yuvalkirstain/PickScore_v1), and a
   bootstrap (10,000 resamples) of the mean paired per-case PickScore difference against diffusers
   bf16. A confidence interval that spans 0 is "no measurable difference".
7. **Time it locally.** `scripts/bench_images.py` times request in, PNG out over HTTP for Loom, Lemonade
   (stable-diffusion.cpp ROCm and Vulkan, thenoise) and a hand-started `sd-server`, on the fixed
   prompt set in `reference/prompts.json` (8 prompts × seeds 0, 42, 1234). Use `--alternate`
   (below). Keep every run in your log; the card publishes the final build's timings.
8. **Optimize** with `loom-optimize` and the profile from `loom-serve --device_profiling_mode=dispatch-events,executable-metadata`.
   Re-run steps 5 to 7 after every build you keep; keep each build's timing and quality in your own log.
9. **Card.** `scripts/stage_hf.py` turns the hrx-loom-library card into a self-contained
   package (motifs copied under `library/motif/`, Bazel archive program set dropped, checkpoint
   origins kept). Write `README.md` and `REPORT.md` from `reference/card-template.md`
   (performance and accuracy, best value per column in bold). With the public hrx-loom 0.3.0
   release, stage and qualify in new directories using the default fetch setting:

   ```bash
   STAGED="$PWD/staged-z-image"
   MATERIALIZED="$PWD/materialized-z-image"
   python3 "${LOOM_SKILL_DIR}/scripts/stage_hf.py" "$CATALOG_CARD" "$STAGED"
   loom-fetch "$STAGED/model.json" --preset=image --output="$MATERIALIZED" --no-input
   loom-serve --device=amdgpu --model="$MATERIALIZED/model.json" --listen=127.0.0.1:8211
   ```

   From a second terminal, wait for the model's registration to become ready at `/v1/status`
   before sending the seed-0 Z-Image request.

   ```bash
   curl -fsS http://127.0.0.1:8211/v1/status
   curl -fsS http://127.0.0.1:8211/v1/images/generations \
     -H 'Content-Type: application/json' \
     -d '{"model":"Tongyi-MAI/Z-Image-Turbo","prompt":"a red fox in the snow","seed":0,"size":"384x384","response_format":"b64_json"}' \
     -o z-image-response.json
   ```

   Run the source card's
   `image_check.py` on the materialized document, including the lifecycle and bounded-workspace
   presets when present. After uploading to staging, repeat fetch and qualification against
   the immutable staged revision.

## Timing fairly

These rules keep timings comparable from build to build and backend to backend, and they are what
the card's Methodology list reports.

- **One resident model.** A resident 4B image server holds about 21 GB; on a 32 GB GPU carve-out a
  second resident model spills into shared memory and both slow down. Time backends in
  **block-alternating sessions**: start one backend, run 8 prompts × 3 seeds, stop it, start the
  next, and rotate the block order over 3 repeats so thermal drift lands on every backend.
- **Same request everywhere.** Size, step count and guidance must match. Send top-level `steps`
  and `cfg_scale` to Lemonade. Its stable-diffusion.cpp wrapper translates them to
  `sd_cpp_extra_args.sample_params` for the actual backend. Direct sd-server requires the nested tag.
  For Loom, verify `denoising_steps` in the served card and supply
  `--loom-guidance LABEL=G` from the pinned program when guidance is absent from its interface.
  Supply `--loom-card LABEL=PATH` when timing an existing Loom process.
  For a TheNoise checkpoint whose guidance 1.0 disables CFG, use `--thenoise-cfg-free LABEL`
  to translate a comparison value of zero to 1.0 on that target. Record both values.
- Run an externally managed `sd-server` or `loom-url` target in its own block with other image
  backends stopped. The script rotates and unloads multiple targets only when it manages their
  residency.
- **Caches.** thenoise answers a repeat of its previous request from a stage cache, so consecutive
  requests to one backend must not share a prompt: order cases seed-major, prompt-minor.
- **Own-RNG backends.** stable-diffusion.cpp and thenoise draw their own noise, so their images are a
  different picture for the same seed. Compare them on CLIPScore and PickScore only, never PSNR or LPIPS.
- **Say what a row measures.** HTTP end to end, in-process, or device time; default Lemonade
  settings or a hand-tuned `sd-server` (`--diffusion-fa`, a GGUF quant). Report configurations
  Lemonade does not ship (thenoise `torch.compile`, INT8) as separate rows.
- Log the device, driver and every backend version with each run.

## Domain knowledge

`python3 "${LOOM_SKILL_DIR}/../loom-kb/kb.py" route --arch z-image --device strix-halo --topic image-porting,kernels-ir`
prints the diffusion architecture and device facts alongside the shared kernel guidance.
Use `--arch flux2` or `--arch krea2` for those pipelines.

## Limits

- The published Real-ESRGAN card uses the image-generation service. It is served through
  `/v1/images/generations` with the input as base64 raw RGB in `prompt` and the output size in `size`:
  check the prompt length and alphabet on the host in `@image_prepare_request_v1` and decode on the
  device, ship a tokenizer whose vocabulary is the base64 alphabet (the image service requires one),
  and expect an RGBA reply.
- Guidance-distilled checkpoints only have been ported; classifier-free guidance doubles the
  transformer batch and needs its own request option.

## Mandate

Declare the scope first, work inside it, opt in by scope, run `fleet/scope_lint.py` before a PR, stop on
out-of-mandate needs: the rules are in `loom-kb/SKILL.md` ("Mandate rules").
