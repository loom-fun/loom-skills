# Model: jdopensource-JoyAI-LLM-Flash-loom

Strix Halo prefill generator choices of the JoyAI-LLM-Flash card (deepseek2 example kit, IQ4_XS): the IQ4_XS inline dequant
GEMMs, the column-wave grouped MoE GEMMs and the MLA k_b / v_b prefill products; resolved by `resolve.py` at build time.
Measurements and stop reasons: the JoyAI pfgemm lane (PR #1361) in `scope.json` evidence and the `choices.py` docs.
