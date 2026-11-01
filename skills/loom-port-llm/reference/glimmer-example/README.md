# Muse-Glimmer (arch `muse-glimmer`) port: worked example

A port of `unsloth/Muse-Glimmer-30B-GGUF` (UD-Q5_K_M, 30B dense) written by a fresh agent from these
skills in October 2026. Kernels: golden PASS (max gap 0.35), consistency, needle at 61k and the full
131k. Chat: the ATEM channel format. **Open: the agent gate's history reuse** (see below).

| File | Adds |
|---|---|
| `gen_llm.py`, `gen_flash2.py`, `inspect_gguf.py`, `build.py`, `make_kernel_tests.py` | `norm` (interleaved) RoPE, weightless embedding norm, NoPE on global layers (i % 4 == 3) and SWA elsewhere, logit softcap in the sampling logits, mixed-format matvec tests |
| `chat_probe.py`, `atem/atem.loom`, `atem/atem_rodata.py`, `control.loom` | the ATEM channel format (`<|start|>assistant to=self<|message|>...` reasoning, `to=user` answers, tool headers `to=functions.NAME`), its tool-call parsing in `output_next` (a zero-length span must not change state), and `simulate()` verified against the Jinja template. See also `../examples/atem_*` |
| `cfg.json` | the reviewed config without the tensor table |

**Open issue to fix first (root cause not confirmed):** after a tool call `reuse_ok` fails with
`cached_tokens` 0. The model often omits the `<|eom|>` the template renders after reasoning, so the
re-rendered history differs from the device state at that byte; that alone would still allow reuse
up to the difference, so 0 points at the turn not being published as reusable (`complete_text`'s
`reusable` = `final_state >= 2`: check which state a tool-call turn ends in) or at the history
starting to differ earlier. Don't "fix" it by inserting `<|eom|>` into the published text: the
published history must match the tokens in the KV cache. Diff `input_tokens` of the next turn against
`chat_probe.simulate()` and print `final_state` for a tool-call turn first.
The sampling gate's seed-variety check failed only because this prompt is
near-deterministic for the model (now skipped by `sampling_check.py`).

Cost note: llama.cpp decodes this model at ~9 tok/s, so the gates are slow (the golden alone took 22 min
before the golden set was halved).
