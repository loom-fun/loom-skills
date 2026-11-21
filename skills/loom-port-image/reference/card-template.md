# Hugging Face card for a Loom image model

The layout of the loom-fun image cards
([FLUX.2-klein-4B](https://huggingface.co/loom-fun/black-forest-labs-FLUX.2-klein-4B-loom),
[Z-Image-Turbo](https://huggingface.co/loom-fun/Tongyi-MAI-Z-Image-Turbo-loom)). It follows the
loom-fun LLM cards (Quick start per GPU, Features, Correctness, Files) with the columns an image
model needs. Fill every value from a run you can name; leave out a row rather than estimate it.

The card carries accuracy, not timing. Keep the timings from `scripts/bench_images.py` with your
own logs: they guide optimization and are not part of the published card.

## Package contents

| file | from |
|---|---|
| `model.json`, `model.json.lock` | `scripts/stage_hf.py` from the hrx-loom-library card |
| `program/*.loom` | the card's programs |
| `library/motif/tensor/*.loom` | the motifs the card links, copied so the package is self-contained |
| `README.md` | this template |

Weights, tokenizer and license are **not** uploaded: `model.json.lock` pins them in the
publisher's repository at a revision, and `loom-fetch` downloads them from there.

## README.md sections, in order

1. Front matter: `base_model`, `license` (the checkpoint's), `pipeline_tag: text-to-image`,
   tags `loom`, `loom-serve`, the device tags you tested, and the model family.
2. `# <Model> for AMD Loom` and one paragraph: what the package is, which checkpoint, which
   endpoint, which device it was tested on.
3. `## Quick start`: "Open your GPU and copy its commands." and one `<details>` block per tested
   GPU with `pipx install`, `loom-fetch huggingface:loom-fun/<repo>:<preset> --materialize=copy --output=<dir>`
   and `loom-serve`. One line for other GPUs the SDK supports: they can run it and have not been tested.
4. `## Send a request`: one request to `/v1/images/generations` with `seed` and `response_format: b64_json`.
5. `## Accuracy`, `### <GPU> (<gfx>)`, against diffusers started from the same noise:

   | | PSNR vs diffusers fp32 ↑ | LPIPS vs diffusers fp32 ↓ | LPIPS vs diffusers bf16 ↓ | CLIPScore ↑ | PickScore ↑ |
   |---|---|---|---|---|---|

   One row for Loom and one for diffusers bf16, then the paired PickScore difference against
   diffusers bf16 with its 95 % bootstrap interval. Follow with a short list: device, hrx-loom and
   diffusers versions, the prompts and seeds (from `reference/prompts.json`), which cases each
   metric covers, the metric models and the bootstrap, "Other GPUs: not tested.", and the dates.
6. `## Presets`, `## Features` (API, sizes, seed, steps, guidance, prompt length, image input,
   concurrency), `## Correctness` (gate, description, result per GPU, dated footnotes saying
   whether the gate ran on the uploaded package).
7. `## Files`, the versions line (card schema, hrx-loom, hrx-loom-library commit) and the source
   files with their pinned revisions and license.

## Before upload

- Fetch the staged repository with `loom-fetch` into a clean directory and run `image_check` on it;
  date that run in the Correctness footnote.
- Check every link resolves, including the hrx-loom-library card link once its pull request merges.
- Check the README carries no timings: no seconds per image, no speedups, no backend comparisons.
