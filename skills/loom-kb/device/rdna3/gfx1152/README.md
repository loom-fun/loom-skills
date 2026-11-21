# gfx1152 (Krackan, Radeon 860M)

Krackan uses the existing gfx11-generic RDNA3 programs. Windows runs use the hrx-loom AMD GPU backend.
The roofline row is a specification ceiling for LPDDR5X-8000. Host memory bandwidth and the usable GPU
allocation vary by machine. [AMD Ryzen AI 7 350 specifications](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-7-350.html)
[AMD ROCm GPU specifications](https://rocmdocs.amd.com/en/develop/reference/gpu-specs.html)

Per-model serve flags, context and benchmark evidence live in the fleet device results. A card adds a
`chat-gfx1152` preset only for models with a verified fit entry. The retained context is the verified
per-request limit for the measured configuration. Use `kb.py route --arch qwen35 --device gfx1152 --topic xsj` for the measurement workflow.
