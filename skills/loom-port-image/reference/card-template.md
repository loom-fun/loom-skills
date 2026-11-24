# Hugging Face card for a Loom image model

The layout of the loom-fun image cards
([FLUX.2-klein-4B](https://huggingface.co/loom-fun/black-forest-labs-FLUX.2-klein-4B-loom),
[Z-Image-Turbo](https://huggingface.co/loom-fun/Tongyi-MAI-Z-Image-Turbo-loom),
[Real-ESRGAN x4plus](https://huggingface.co/loom-fun/xinntao-Real-ESRGAN-x4plus-loom)). It follows the
loom-fun LLM cards (Quick start per GPU, Performance, Features, Correctness, Files) with the columns
an image model needs. Fill every value from a run you can name; leave out a row rather than estimate it.

Like the LLM cards, the card publishes performance: every backend you timed with
`scripts/bench_images.py`, the quality columns from `scripts/metrics.py`, and a methodology list
precise enough to reproduce them. `REPORT.md` carries the detail that does not fit the README.

## Package contents

| file | from |
|---|---|
| `model.json`, `model.json.lock` | `scripts/stage_hf.py` from the hrx-loom-library card |
| `program/…` and local assets | the card's referenced programs and inputs |
| `library/motif/tensor/*.loom` | the motifs the card links, copied so the package is self-contained |
| `README.md` | this template |
| `REPORT.md` | setup, the full prompt text, commands, per-stage times of in-process backends, and every Loom build measured |

Weights, tokenizer and license are **not** uploaded: `model.json.lock` pins them in the
publisher's repository at a revision, and `loom-fetch` downloads them from there.

## README.md sections, in order

1. Front matter: `base_model`, `license` (the checkpoint's), `pipeline_tag: text-to-image`,
   tags `loom`, `loom-serve`, the device tags you tested, and the model family.
2. `# <Model> for AMD Loom` and one paragraph: what the package is, which checkpoint, which
   endpoint, which device it was tested on.
3. `## Quick start`: "Open your GPU and copy its commands." and one `<details>` block per tested
   GPU with the tested SDK pin, `loom-fetch huggingface:loom-fun/<repo>:<preset>@<revision> --output=<dir>`
   and `loom-serve`. Use this default fetch setting for a qualified public hrx-loom 0.3.0
   PyPI or release pin. Keep `--materialize=copy` in the command for other pins or when the
   output directory needs independent copies of its assets. See the
   [Loom Setup fetch guide](../../loom-setup/SKILL.md#2-fetch-a-catalog-model) for how default
   materialization uses filesystem links and falls back to copies. One line for other GPUs
   the SDK supports: they can run it and have not been tested.
4. `## Send a request`: one request to `/v1/images/generations` with `seed` and `response_format: b64_json`.
5. `## Performance vs stable-diffusion.cpp`, `### <GPU> (<gfx>)`. One line with the size, steps
   and guidance, then three tables:

   | backend | s/image ↓ | Loom speedup | CLIPScore ↑ | PickScore ↑ | PickScore vs diffusers bf16 (95 % CI) | beats diffusers bf16 on |
   |---|---|---|---|---|---|---|

   One row per backend you timed: Loom, stable-diffusion.cpp ROCm and Vulkan builds via Lemonade,
   any hand-tuned `sd-server` (add a `weights` column when rows differ in weights), thenoise via
   Lemonade, diffusers bf16 and fp32. Loom speedup is backend s/image ÷ Loom s/image.

   | thenoise variant | s/image ↓ | Loom speedup |
   |---|---|---|

   thenoise in process, for configurations Lemonade does not ship (`torch.compile`, INT8).

   | backend | PSNR vs diffusers fp32 ↑ | LPIPS vs diffusers fp32 ↓ | LPIPS vs diffusers bf16 ↓ |
   |---|---|---|---|

   Loom and diffusers bf16 against diffusers started from the same noise.

   **Bold the best value in each column**, ties included, whichever row it is in, never the Loom row
   by default. Do not rank a column's reference row, rows covering only a subset of the cases, a
   column with one measured row, or the speedup column (it restates s/image). If another backend is
   faster or closer to the reference, say so in a sentence under the table.

   Then a **Methodology** list: device and driver, hrx-loom, Lemonade, stable-diffusion.cpp,
   thenoise and diffusers versions; the prompts and seeds (from `reference/prompts.json`); how
   s/image is taken and where each row's timing boundary is (HTTP end to end or in process);
   block-alternating sessions and repeats; the speedup definition; the precision and settings
   each backend ran; the metric models, the bootstrap and "beats"; the bold rule; which cases each
   metric covers; "Other GPUs: not tested."; and the dates, with a link to `REPORT.md`.

   A super-resolution card has no prompts or noise: time each input size in its own table, score
   against PyTorch fp32 (PSNR, max abs diff, pixels off by more than 1) and against the original
   images (PSNR, SSIM, LPIPS), and include a bicubic row.
6. `## Presets`, `## Features` (API, sizes, seed, steps, guidance, prompt length, image input,
   concurrency), `## Correctness` (gate, description, result per GPU, dated footnotes saying
   whether the gate ran on the uploaded package).
7. `## Files` (including `REPORT.md`), the versions line (card schema, hrx-loom, hrx-loom-library commit) and the source
   files with their pinned revisions and license.

## Before upload

- Fetch the staged repository with `loom-fetch` into a clean directory and run `image_check` on it;
  date that run in the Correctness footnote.
- See the [SDK 0.3.0 DevLab comparison](https://github.com/loom-fun/loom-skills/actions/runs/45425749984)
  for the full fetched manifests and image requests from the three published image presets.
- Check every link resolves, including the hrx-loom-library card link once its pull request merges.
- Check every bold value is the best in its column, and that each table row names what it
  measured (backend, build, weights, timing boundary).
- Check every number in the README appears in `REPORT.md` or a log you can name.
