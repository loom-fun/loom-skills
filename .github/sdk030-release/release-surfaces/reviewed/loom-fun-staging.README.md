---
title: README
sdk: static
pinned: false
---

# Loom: Model optimization is no longer locked by the engine.

*Experimental staging preview. SDK `hrx-loom` 0.3.0.*

`hrx-loom` is a new end-to-end stack (compiler, runtime and serving tools) that lets anyone write, optimize, serve and share kernels and models without modifying the inference engine or the server.

Kernels are written in `.loom` files, a vector programming model reminiscent of MLIR that is easy for agents. To show what the stack can do early on, we ported 50+ models as an experiment, and you can fetch and serve them today on four AMD GPUs:

- AMD Ryzen AI Max+ 395 "Strix Halo" (gfx1151)
- AMD Radeon AI PRO R9700 (gfx1201, 32 GB)
- AMD Radeon gfx1201, 16 GB (RX 9070 XT, RX 9070)
- AMD Ryzen AI 300 "Strix Point" (gfx1150), one model so far

- [ROCm/hrx-system](https://github.com/ROCm/hrx-system): Loom compiler and server
- [loom-fun/loom-skills](https://github.com/loom-fun/loom-skills): agent skills to port and optimize
- [Loom staging model catalog](https://loom-fun.dev/?org=loom-fun-staging): models you can fetch and serve today

## Quick start

1. **Install**: Install `hrx-loom` 0.3.0 from PyPI.
   ```
   pipx install "hrx-loom[hf]==0.3.0"
   ```
2. **Fetch**: Pull a Loom model from Hugging Face. The form is `huggingface:ORG/REPO:PRESET`; always name the preset (each card's Quick start shows the one for your GPU).
   ```
   loom-fetch huggingface:loom-fun-staging/unsloth-Qwen3-0.6B-loom:chat --output=Qwen3-0.6B
   ```
3. **Serve**: Start a local OpenAI-compatible server.
   ```
   loom-serve --model=Qwen3-0.6B '--device=amdgpu://?completion_spin_ns=4294967295' --pool_backing=fixed
   ```

Each model card's Quick start has a block per GPU: open yours and copy its commands.

## Published cards

Use the [production Loom organization](https://huggingface.co/loom-fun) for released cards and instructions for older SDK versions.
