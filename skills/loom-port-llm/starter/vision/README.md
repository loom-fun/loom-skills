# starter/vision: image input for ported LLMs

Used by `port.sh` stage 2b and `build.py` when `cfg.json` has a `vision` block. Overview, the supported
projector families and **how to add one**: `../../reference/vision.md`.

| File | Role |
|---|---|
| `add_vision.py` | picks and downloads the repo's mmproj, writes `cfg["vision"]` (+ `cfg["mrope"]` for Qwen-VL) |
| `encoder.py` | `config(mmproj)` (family constants from GGUF metadata) and `build()`: the encoder as an ordered list of kernel launches per family |
| `gen_vision.py` | encoder kernels: norms, GEMM (the starter's WMMA GEMM + F16/BF16/F32 weights), patch embedding, positions, 2-D RoPE, attention (full/windowed), merges, Pillow-compatible resize |
| `gen_component.py` | `vision_encode` command program (one stage per processed size) and the `text_image_*_v1` component programs |
| `gen_chat_vision.py` | control-program code: size rule per family, image markers in chat content, marker-aware `prepare_input` |
| `build_vision.py` | patches the control/storage programs, writes the component files and the card entries; M-RoPE control code |
| `preprocess.py`, `ref_vit.py` | llama.cpp's preprocessing and NumPy reference encoders (development and tests; never a golden) |
| `trace_compare.py` | `ref_vit.py` vs a `llama-mtmd-debug` trace, tensor by tensor |
| `tests/run_tests.sh`, `test_encoder.py` | kernel tests vs NumPy; the whole encoder on real weights vs `ref_vit.py` |
