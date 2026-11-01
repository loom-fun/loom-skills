# Muse-Glimmer (arch `muse-glimmer`) port: worked example

A port of `unsloth/Muse-Glimmer-30B-GGUF` (UD-Q5_K_M, 30B dense) written by a fresh agent from these
skills in October 2026. Kernels: golden PASS (max gap 0.35), consistency, needle at 61k and the full
131k. Chat: the ATEM channel format. **Open: the agent gate's history reuse** (see below).

| File | Adds |
|---|---|
| `gen_llm.py`, `gen_flash2.py`, `inspect_gguf.py`, `build.py`, `make_kernel_tests.py` | `norm` (interleaved) RoPE, weightless embedding norm, NoPE on global layers (i % 4 == 3) and SWA elsewhere, logit softcap in the sampling logits, mixed-format matvec tests |
| `chat_probe.py`, `atem/atem.loom` (= the ATEM section of `control.loom`; `build.py` splices it in), `atem/atem_rodata.py`, `build.py` (the starter's, plus the ATEM hunk) | the ATEM channel format (`<|start|>assistant to=self<|message|>...` reasoning, `to=user` answers, tool headers `to=functions.NAME`), its tool-call parsing in `output_next` (a zero-length span must not change state), and `simulate()` verified against the Jinja template. See also `../examples/atem_*` |
| `cfg.json` | the reviewed config without the tensor table |

**Agent gate:** passes (rendering, reuse and calls 7/7) with the ATEM section of `control.loom` and this
`chat_probe.py` (a second fresh-agent run, Nov 2026, 76 min end to end, all full gates PASS). The model
sometimes writes `<|eom|>` before the `to=user` header: `chat_probe.raw_alternatives()` lists both raw
texts and `agent_golden.py` compares the closer one.
The sampling gate's seed-variety check failed only because this prompt is
near-deterministic for the model (now skipped by `sampling_check.py`).

Cost note: llama.cpp decodes this model at ~9 tok/s, so the gates are slow (the golden alone took 22 min
before the golden set was halved).
