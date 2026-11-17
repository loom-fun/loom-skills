# Facet safetensors: porting a model from a checkpoint (recipe)

## Another MXFP4 safetensors model: what is configuration, what is code

Start: `python3 "${LOOM_SKILL_DIR}/../loom-port-llm/starter/inspect_safetensors.py" REPO_OR_DIR -o W/llm_config.json` (a hub repo is read with range
requests: header, config.json, tokenizer files; no weights), then route again with that config so the producer and quant nodes load.

| Step | Same architecture family as a ported model | New family |
|---|---|---|
| Name map | configuration: reuse `hf_names/<family>.json`; check every name resolves (the inspector lists unmapped tensors) | configuration: a new data file |
| Producer convention (Quark scale suffixes, exclude list) | configuration: read from the checkpoint's quantization config | code only for a producer the starter does not know |
| Quant format (`mxfp4s`) and its kernels | none: the format is registered, kernels are choice-selected | none |
| Geometry, chat template, tokenizer | `inspect_safetensors.py` output; run the cfg review (topic `cfg-review`) | same, plus `new-arch` if the architecture is new |
| Semantic differences of the framework layout (norm offset, A_log, conv1d, head order) | configuration: the family map's per-tensor entries | the entries are new; a layout no kernel reads natively needs a prepared copy |
| Package build | `build.py` from the config; `model.json` declares `format: safetensors` | same |
| Gates | reference = vLLM ROCm (topic `benchmark`) | same |

A checkpoint whose quantized tensors use another element order or scale type than the verified one is a new format key, not a flag on `mxfp4s`.

## Dry run without a GPU (the generality check)

Route the config (`kb.py route --config W/llm_config.json --topic porting`), build the package on CPU, and compile the programs for gfx1201 and gfx1151
with the SDK. A second Quark MXFP4 checkpoint should need a name map entry at most; anything more is a finding for this node.

Command lines (W = a work dir on disk; the HF cache holds the checkpoint once, linked and never copied):

    S=skills/loom-port-llm/starter
    python3 $S/inspect_safetensors.py REPO W/llm_config.json --checkpoint $(ls ~/.cache/huggingface/hub/models--ORG--NAME/snapshots/*/model.safetensors)
    curl -L https://huggingface.co/REPO/resolve/main/tokenizer.json -o W/tokenizer.json
    python3 $S/build.py W/llm_config.json W/pkg --tokenizer W/tokenizer.json --model-id ID --context N --pool N --memory-gib G   # drafting from the MTP block by default
    kb.py route --config W/llm_config.json --topic porting                                     # the nodes this port loads
    for k in $(grep -o 'export("[^"]*")' W/pkg/program/llm.loom | sed 's/export("\(.*\)")/\1/'); do loom-compile W/pkg/program/llm.loom --target=amdgpu:gfx1201 --root=$k --format=amdgpu-hsaco --output=/dev/null; done   # and gfx1151
    loom-opt W/pkg/program/llm.loom --pass=canonicalize >/dev/null                             # parses and type-checks the command programs (the SDK has no offline command-program compile)

What differs from the GGUF flow: `inspect_safetensors.py` instead of `inspect_gguf.py` (no GGUF reader; the chat block still comes from `chat_probe.py`), `--checkpoint`
instead of a GGUF path, the build writes `weights.loom` with the conversions of the family map and `model.json` declares `format: safetensors`; `publish_package.py prepare`
takes the HF repo as `--source` unchanged; a fleet entry keeps a header dump (`fleet/st_stub.py`, `fleet/add_model.py`) instead of a GGUF stub; kernel tests read the file with `st_tensors.py`.
Dry run of the first model (amd/Qwen3.8-27B-Quark-AWQ-MXFP4, CPU only): inspect 1 s, build 1.5 s, the 54 kernels compile for gfx1201 and gfx1151 (the two `g_silu_mul` artifacts of a standalone root aside), the
command programs type-check; DevLab runs are the first execution.

Second model (dry run 2026-11-15, amd/Llama-3.1-8B-Instruct-MXFP4-W4A4-MLCAL-C1000-GPTQ, a Quark smooth + GPTQ MXFP4 export of a different family, `LlamaForCausalLM`): the kb route, the
producer node, the `mxfp4s` kernels and the build were unchanged; the only per-model work was `hf_names/llama.json` (a data file) plus the inspector generalizations a plain
transformer needed (no `layer_types` / `gdn` / `mtp` in the map; llama3 `rope_parameters` -> `rope_freq_factors`; `bos_id` from the chat template when `add_bos_token` is absent).
Gotchas: run the starter with the system python (the SDK venv's `python3` has no jinja2, which `chat_probe.py` needs), keep the SDK `bin` on PATH only for `loom-compile` / `loom-opt`;
HF llama q/k rows are rotate_half (neox) ordered, so `rope_type` is `neox` and nothing is permuted (a GGUF llama says `norm`); `*.k_proj.output_scale` / `v_proj.output_scale`
(KV-cache scales of the export) are listed as unused: Loom's KV cache is its own. A build without weights links a placeholder checkpoint (`--checkpoint` of an empty file) and a header dump
(`fleet/st_stub.py dump REPO OUT.json.gz`, then `inspect_safetensors.py OUT.json.gz ...`); `build.py` then reports `weights 0.0 GiB`.
