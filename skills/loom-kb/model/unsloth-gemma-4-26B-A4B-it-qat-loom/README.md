# google/gemma-4-26B-A4B-it-qat-q4_0-unquantized

This distinct QAT package uses `unsloth/gemma-4-26B-A4B-it-qat-GGUF` at `7b92b5b28818151e8669af2e45e88d6086f490dd`,
file `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (SHA256 `a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891`).
Its target weights are Q4_0/F32 despite the filename. The Gemma4 kit supplies the
architecture and Q4_0 expert support; no conversion to another quantization is implied.
The tokenizer is pinned to `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized` at `f1e06dc520982d9b9edd76859fdb7ab209449949`.

Baseline feature scope and qualification requirements are in the routed Gemma4 porting topic. This registration adds no measured device-fit claim.
