# google/gemma-4-31B-it-qat-q4_0-unquantized

This distinct QAT package uses `unsloth/gemma-4-31B-it-qat-GGUF` at `43cc1aeb31adf47ec06a854507ce552cd9862e6f`,
file `gemma-4-31B-it-qat-UD-Q4_K_XL.gguf` (SHA256 `00b5a7c497f0c8934033088c10a7fa9a4c015e46ee6d89e9c6890650ba5d0e71`).
Its target weights are Q4_0/F32 despite the filename. The Gemma4 kit supplies the
architecture and Q4_0 expert support; no conversion to another quantization is implied.
The tokenizer is pinned to `google/gemma-4-31B-it-qat-q4_0-unquantized` at `1e4d8beecacb8b7590c1d8bedd7335f687bf311f`.

Baseline feature scope and qualification requirements are in the routed Gemma4 porting topic.

Strix Halo context 81920 (the package's first preset, fleet manifest `context`): the largest that fits the 110 GB host
guard with the exact-QAT assistant drafting (build budget `--memory-gib 102`: KV pool 81920 positions = 75 GiB). DevLab run
44341902935: cold fixed storage 105.2 GB, exact 81920-token completion, structured over-limit rejection, MemAvailable
8-10 GB left while loaded. 86016 does not fit that budget (78.8 GiB of KV). R9700: 10240 (scope.json).
