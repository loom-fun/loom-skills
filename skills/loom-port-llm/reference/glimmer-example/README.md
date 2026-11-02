# Muse-Glimmer (arch `muse-glimmer`) port: worked example

A port of `unsloth/Muse-Glimmer-30B-GGUF` (UD-Q5_K_M, 30B dense) written by fresh agents from these
skills (Oct-Nov 2026; latest run 71 min end to end, full gates and audit PASS, agent loop 7/7). Make a
kit with `../apply_example.sh glimmer W/kit`. The kernels need no changes: `norm` (interleaved) RoPE,
weightless embedding norm, NoPE on global layers (i % 4 == 3) and SWA elsewhere, and the logit softcap
in the sampling logits are starter features driven by `cfg.json`.

| File | Adds |
|---|---|
| `chat/atem.loom`, `chat/atem_rodata.py` | the ATEM channel format (`<|start|>assistant to=self<|message|>...` reasoning, `to=user` answers, tool headers `to=functions.NAME`) and its tool-call parsing in `output_next` (a zero-length span must not change state). `build.py` splices `chat/<call_format>.loom` over the generic chat functions |
| `starter.patch` | `chat_probe.py`: the ATEM model (`simulate()` verified against the Jinja template, `raw_from_chat()` / `raw_alternatives()` for the benchmark); `make_kernel_tests.py`: mixed-format matvec tests |
| `cfg.json` | the reviewed config without the tensor table |

The model sometimes writes `<|eom|>` before the `to=user` header: `chat_probe.raw_alternatives()` lists
both raw texts and `agent_golden.py` compares the closer one. The sampling gate's seed-variety check is
skipped when the prompt is near-deterministic for the model (it is for this one).

Cost note: llama.cpp decodes this model at ~9 tok/s, so the gates are slow (the golden alone took 22 min
before the golden set was halved).
