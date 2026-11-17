# Facet safetensors: the golden reference is vLLM ROCm

There is no llama.cpp GGUF of a Quark checkpoint that is the same numbers, so safetensors cards compare Loom with vLLM ROCm serving the same checkpoint
with its own quantization config. The gates keep their shape: token ids in, greedy tokens and top log-probs out, the same `compare.py`, near-tie
rules and thresholds; a reference adapter answers the requests `gates.sh` makes of llama-server. Loom's activations (A8 or A16 against the producer's
dynamic A4, `facet/quark`) are more precise than the reference's, so a divergence that comes from that difference is a fail shown on the card with the
position and both log-prob gaps; thresholds are never loosened.

Fallback when vLLM cannot run on a runner: transformers with amd-quark on CPU (slow; short golden prompts only).

PLACEHOLDER (filled by the vllmref PR): the adapter module and flags, the DevLab install and the vLLM version and image that run on the R9700 runner,
which gates run (golden, spec, agent, vision), the reference's request log shape, timing and memory on the 32 GB card, and known divergences.
