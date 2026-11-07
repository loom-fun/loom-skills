"""Reads a GGUF and writes the LLM_CONFIG json that gen_llm.py, build.py and
the benchmark scripts consume. The "chat" block is derived from the embedded
chat template by chat_probe.py and verified against it (the printed OK/FAIL
lines); a FAIL or a probe error means fill or fix it by hand. Architecture quirks (sliding layers, NoPE layers, output gate, post norms,
embedding scale) must be checked against llama.cpp's src/models/<arch>.cpp.

usage: python inspect_gguf.py MODEL.gguf OUT.json   (needs `pip install gguf`)
"""
import json
import os
import sys

import gguf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chat_probe  # noqa: E402
import formats  # noqa: E402

path, out = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(path)


def field(key, default=None):
    f = r.fields.get(key)
    if f is None:
        return default
    v = f.contents()
    return v


arch = field("general.architecture")
A = lambda k, d=None: field(f"{arch}.{k}", d)
tensors = {}
for t in r.tensors:
    tensors[t.name] = {"type": t.tensor_type.name.lower(), "shape": [int(x) for x in t.shape]}
H = A("embedding_length")
heads = A("attention.head_count")
kv = A("attention.head_count_kv")
head_dim = A("attention.key_length") or H // heads
BLOCKS = A("block_count")
# Multi-token-prediction (NextN/MTP) blocks follow the trunk: blk.L.. carry nextn.* tensors.
# They are not trunk layers; cfg["mtp"] lists them for the speculative-decoding drafter.
_nextn = sorted({int(t.name.split(".")[1]) for t in r.tensors if ".nextn." in t.name})
L = BLOCKS - len(_nextn) if _nextn and _nextn[0] == BLOCKS - len(_nextn) else BLOCKS
vocab = tensors["token_embd.weight"]["shape"][1]
template = field("tokenizer.chat_template", "") or ""
eos = field("tokenizer.ggml.eos_token_id")
tokens = field("tokenizer.ggml.tokens") or []
im_end = tokens.index("<|im_end|>") if "<|im_end|>" in tokens else None
endoftext = tokens.index("<|endoftext|>") if "<|endoftext|>" in tokens else None
# Llama 3: llama.cpp also ends generation at <|eom_id|> (ipython turns) and <|end_of_text|> (end-of-generation tokens)
llama3_eog = [tokens.index(t) for t in ("<|eot_id|>", "<|eom_id|>", "<|end_of_text|>") if t in tokens]

cfg = {
    "name": field("general.name", arch),
    "arch": arch,
    "gguf": path,
    "hidden": H, "layers": L, "heads": heads, "kv_heads": kv if isinstance(kv, int) else max(kv),
    "head_dim": head_dim, "ffn": A("feed_forward_length"), "vocab": vocab,
    "context_length": A("context_length"),
    "rms_eps": A("attention.layer_norm_rms_epsilon", 1e-6),
    "rope_theta": A("rope.freq_base", 10000.0),
    # neox: rotary pairs (d, d + head_dim/2) -- qwen2/qwen3/gemma/...; norm: (2d, 2d+1) -- llama
    # muse-glimmer: llama4-style interleaved rope
    "rope_type": "norm" if arch in ("llama", "mistral", "granite", "muse-glimmer") else "neox",
    "qk_norm": any(k.endswith(".attn_q_norm.weight") for k in tensors),
    "attn_gate": "blk.0.attn_gate.weight" in tensors,
    "post_norms": "blk.0.post_attention_norm.weight" in tensors,
    "embed_norm": False,
    "embed_scale": None,
    "sliding_window": A("attention.sliding_window") or 0,
    "sliding_layers": [],
    "nope_layers": [],
    "tied_embeddings": "output.weight" not in tensors,
    "eos_id": eos,
    # head_select stops on any of these; keep <|endoftext|> for ChatML models.
    # (and the GGUF's end-of-turn token, e.g. <|eot|> of llama4-style templates)
    "extra_eos_ids": sorted({i for i in (im_end, endoftext, field("tokenizer.ggml.eot_token_id"), *llama3_eog) if i is not None and i != eos}),
    "bos_id": field("tokenizer.ggml.bos_token_id") if field("tokenizer.ggml.add_bos_token", False) else None,
    "chat": None,
    "tensors": tensors,
}
if arch not in ("llama", "mistral", "granite", "qwen2", "qwen3", "qwen35", "qwen3next", "gemma3", "gemma4", "phi3",
                "qwen3moe", "qwen35moe", "gpt-oss", "spark2_5"):
    print(f"WARNING: arch {arch!r} is unknown to the starter: rope_type '{cfg['rope_type']}' is a guess (neox pairs d/d+D/2, "
          "norm pairs 2d/2d+1; llama4-style models are norm). Check llama.cpp's src/models/<arch>.cpp (LLAMA_ROPE_TYPE_*) "
          "before trusting a golden FAIL or PASS")
_g = cfg["heads"] // max(cfg["kv_heads"], 1)
if _g & (_g - 1):
    print(f"NOTE: {_g} query heads per KV head (not a power of two): flash prefill stages K/V for the power of two "
          "below it and the extra waves skip staging (gen_flash2.py); check the long goldens")
# Sliding-window layers from the GGUF's per-layer pattern (gemma/flint/muse-glimmer style).
SWA_PERIOD = {"gpt-oss": 2}   # src/models/<arch>.cpp load_swa_pattern(ml, n): layers with i % n < n - 1 slide
pat = A("attention.sliding_window_pattern")
if cfg["sliding_window"] and isinstance(pat, list) and len(pat) in (L, BLOCKS):
    pat = pat[:L]
    cfg["sliding_layers"] = [i for i, sl in enumerate(pat) if sl]
elif cfg["sliding_window"] and isinstance(pat, int) and pat > 1:
    # a period instead of a list (llama.cpp set_swa_pattern(n)): every n-th layer is full attention
    cfg["sliding_layers"] = [i for i in range(L) if i % pat < pat - 1]
elif cfg["sliding_window"] and arch in SWA_PERIOD:
    # the period llama.cpp hard-codes for the architecture when the GGUF has none (load_swa_pattern(ml, n))
    cfg["sliding_layers"] = [i for i in range(L) if i % SWA_PERIOD[arch] < SWA_PERIOD[arch] - 1]
elif cfg["sliding_window"]:
    print(f"WARNING: sliding_window {cfg['sliding_window']} but no per-layer pattern in the GGUF: fill "
          "sliding_layers (and nope_layers) by hand from llama.cpp's src/models/<arch>.cpp or llama_trace.py")
# muse-glimmer (reference/glimmer-example): a weightless RMS norm on the token embeddings, and the full-attention
# layers (every 4th, from the sliding pattern) use no rope.
if arch == "muse-glimmer" and cfg["sliding_layers"]:
    cfg["embed_norm"] = True
    cfg["logit_scale"], cfg["logit_softcap"] = A("logit_scale"), A("final_logit_softcapping")
    cfg["nope_layers"] = [i for i in range(L) if i not in cfg["sliding_layers"]]
# Partial rotary embedding (e.g. qwen35: 64 of 256 dims).
rope_dims = A("rope.dimension_count")
if rope_dims and rope_dims != head_dim:
    cfg["rope_dims"] = rope_dims
# Llama-3-style RoPE frequency factors (rope_freqs.weight, one per rotary pair: llama.cpp divides that pair's
# frequency by it; Llama 3.1/3.2: 1 for high frequencies, up to 32 for low ones). Gemma 4's 1e30 cut is its kit's.
if arch != "gemma4":
    freqs = next((t.data for t in r.tensors if t.name == "rope_freqs.weight"), None)
    if freqs is not None and any(float(f) != 1.0 for f in freqs):
        cfg["rope_freq_factors"] = [float(f) for f in freqs]
# YaRN RoPE scaling (gen_llm.py yarn_constants; llama.cpp applies it whenever the GGUF says yarn: ext_factor 1, and
# the attention factor that leaves mscale = 1 + 0.1 ln(factor) unless rope.scaling.yarn_log_multiplier is set).
if A("rope.scaling.type") == "yarn" and (A("rope.scaling.factor") or 1.0) > 1.0:
    import math
    _f, _lm = float(A("rope.scaling.factor")), float(A("rope.scaling.yarn_log_multiplier", 0.0) or 0.0)
    _ms = lambda m: 1.0 if _f <= 1.0 else 0.1 * m * math.log(_f) + 1.0
    _af = (_ms(1.0) / _ms(_lm) if _lm else _ms(1.0)) / (1.0 + 0.1 * math.log(_f)) * float(A("rope.scaling.attn_factor", 1.0) or 1.0)
    cfg["rope_scaling"] = {"type": "yarn", "factor": _f, "orig_ctx": A("rope.scaling.original_context_length"),
                           "beta_fast": float(A("rope.scaling.yarn_beta_fast", 32.0) or 32.0),
                           "beta_slow": float(A("rope.scaling.yarn_beta_slow", 1.0) or 1.0), "attn_factor": _af}
    print("rope scaling:", cfg["rope_scaling"])
# The FFN's pre-norm stored as post_attention_norm (gpt-oss: attn_post_norm normalizes ffn_inp; no ffn_norm and no
# Gemma-style post_ffw_norm): it is the FFN norm, not a post-attention norm.
if ("blk.0.ffn_norm.weight" not in tensors and "blk.0.post_attention_norm.weight" in tensors
        and "blk.0.post_ffw_norm.weight" not in tensors and not A("ssm.inner_size")):
    cfg["post_norms"] = False
    cfg["ffn_norm_tensor"] = "post_attention_norm"
# Gemma 3n/4 geometry (reference/gemma4-example's gen_llm.py keys), from this GGUF: a reference package of another size
# (find_reference.py apply fills keys inspect leaves out) has other KV-sharing, per-layer-embedding and rope values.
if arch == "gemma4":
    import math
    shared = A("attention.shared_kv_layers", 0) or 0
    freqs = next((t.data for t in r.tensors if t.name == "rope_freqs.weight"), None)
    ple_dim = A("embedding_length_per_layer_input", 0) or 0
    cfg.update({
        "head_dim_swa": A("attention.key_length_swa"), "rope_theta_swa": A("rope.freq_base_swa"),
        "rope_dims": rope_dims or head_dim, "rope_dims_swa": A("rope.dimension_count_swa"),
        # global layers: only the rotary pairs whose rope_freqs factor is finite rotate (the rest are 1e30)
        "rope_freq_cut": int((freqs < 1e29).sum()) if freqs is not None else None,
        "kv_shared_from": L - shared if shared else None,
        "post_norm_eps": cfg["rms_eps"], "ffn_act": "gelu", "v_norm": True, "attn_scale": 1.0,
        "final_softcap": A("final_logit_softcapping"),
        "ple": {"dim": ple_dim, "embd_scale": math.sqrt(ple_dim), "proj_scale": 1 / math.sqrt(H),
                "combine_scale": 1 / math.sqrt(2)} if ple_dim and "per_layer_token_embd.weight" in tensors else None,
        "layer_output_scale": "blk.0.layer_output_scale.weight" in tensors,
        # token embeddings are scaled by sqrt(hidden) (llama.cpp gemma4: inpL * sqrt(n_embd))
        "embed_scale": math.sqrt(H),
        # full-attention layers (the rest slide); KV planes only for layers that own their KV
        "global_layers": [i for i in range(L) if i not in cfg["sliding_layers"]],
        "kv_layers": L - shared,
        # generation ends at <eos>, at the turn close, and where a tool result would follow a call
        "extra_eos_ids": [tokens.index(s) for s in ("<eos>", "<turn|>", "<|tool_response>") if s in tokens],
    })
    # What the example's kernels assume: one KV-head count for every layer, and a V projection on every KV-owning layer.
    no_v = [i for i in range(min(L, cfg["kv_shared_from"] or L)) if f"blk.{i}.attn_v.weight" not in tensors]
    if isinstance(kv, list) and len(set(kv)) > 1 or no_v:
        print(f"WARNING: unsupported layout: per-layer KV heads {sorted(set(kv)) if isinstance(kv, list) else kv}"
              f"{f', no attn_v (V = K) on layers {no_v[:4]}...' if no_v else ''}: gemma4-example's attention assumes "
              "one KV-head count and an attn_v on every KV-owning layer; extend gen_llm.py (agent port)")
# Sliding layers with their own RoPE at the same head_dim (spark2_5: freq_base_swa 1e4 + full rotary dims on the sliding
# layers, 5e6 + 64 of 256 dims on the global ones; llama.cpp defaults each to the global value when the key is absent).
if arch != "gemma4" and cfg["sliding_layers"] and (A("rope.freq_base_swa") or A("rope.dimension_count_swa")):
    cfg["rope_theta_swa"] = A("rope.freq_base_swa") or cfg["rope_theta"]
    cfg["rope_dims_swa"] = A("rope.dimension_count_swa") or rope_dims or head_dim
# FFN activations llama.cpp hard-codes per architecture (build_ffn's LLM_FFN_GELU: GEGLU, gelu-tanh); default SwiGLU.
FFN_ACT = {"spark2_5": "gelu"}
if arch in FFN_ACT:
    cfg["ffn_act"] = FFN_ACT[arch]
# Hybrid models: gated delta-net (linear attention) layers carry recurrent state
# instead of KV (llama.cpp qwen35/qwen3next: layer i is recurrent unless (i+1) % interval == 0).
if A("ssm.inner_size"):
    interval = A("full_attention_interval", 4)
    cfg["recurrent_layers"] = [i for i in range(L) if (i + 1) % interval != 0]
    cfg["gdn"] = {"conv_kernel": A("ssm.conv_kernel"), "k_heads": A("ssm.group_count"), "v_heads": A("ssm.time_step_rank"),
                  "head_k": A("ssm.state_size"), "inner": A("ssm.inner_size")}
    cfg["gdn"]["head_v"] = cfg["gdn"]["inner"] // cfg["gdn"]["v_heads"]
    cfg["kv_layers"] = L - len(cfg["recurrent_layers"])
    # qwen35-style attention: attn_q packs [query | sigmoid gate] per head; the FFN norm is post_attention_norm.
    attn0 = next(i for i in range(L) if i not in cfg["recurrent_layers"])
    if tensors[f"blk.{attn0}.attn_q.weight"]["shape"][1] == 2 * heads * head_dim:
        cfg["attn_gate"] = "fused_q"
    if f"blk.{attn0}.ffn_norm.weight" not in tensors and f"blk.{attn0}.post_attention_norm.weight" in tensors:
        cfg["post_norms"] = False
        cfg["ffn_norm_tensor"] = "post_attention_norm"
    print(f"hybrid: {len(cfg['recurrent_layers'])} gated delta-net layers + {cfg['kv_layers']} attention layers;", cfg["gdn"])
if _nextn and L != BLOCKS:
    cfg["mtp"] = {"layers": _nextn}
    print(f"MTP: {len(_nextn)} NextN block(s) {_nextn} after the {L}-layer trunk: speculative decoding (MTP) is available")
# Attention layers with one fused q | k | v projection (spark2_5: attn_qkv; hybrids' gated delta-net layers have their own
# attn_qkv, which this does not look at), and an attn_gate with one row per query head (sigmoid per head, spark2_5).
_attn0 = next((i for i in range(L) if i not in cfg.get("recurrent_layers", [])), None)
if _attn0 is not None and f"blk.{_attn0}.attn_qkv.weight" in tensors and f"blk.{_attn0}.attn_q.weight" not in tensors:
    if tensors[f"blk.{_attn0}.attn_qkv.weight"]["shape"][1] != (heads + 2 * cfg["kv_heads"]) * head_dim:
        print("WARNING: unsupported layout: attn_qkv is not [q | k | v] of heads + 2 kv_heads rows of head_dim (agent port)")
    cfg["attn_qkv"] = True
if _attn0 is not None and cfg["attn_gate"] is True and tensors[f"blk.{_attn0}.attn_gate.weight"]["shape"][1] == heads:
    cfg["attn_gate"] = "per_head"
# Attention projection biases (qwen2: q/k/v, added right after the projection, llama.cpp build_qkv): g_qk_rope_store
# adds those in cfg["attn_bias"] (gen_llm.py QKV_BIAS), with the same set on every attention layer. A fused-qkv or
# output bias, or a mixed set, has no kernel: a port would serve garbage while its kernel tests pass, so stop.
_attn = [i for i in range(L) if i not in cfg.get("recurrent_layers", [])] + (cfg.get("mtp") or {}).get("layers", [])
_qkv = {i: tuple(p for p in "qkv" if f"blk.{i}.attn_{p}.bias" in tensors) for i in _attn}
# An output-projection bias on every attention layer (gpt-oss) is added by g_bias_add (gen_llm.py OUT_BIAS).
_trunk_attn = [i for i in range(L) if i not in cfg.get("recurrent_layers", [])]
if _trunk_attn and all(f"blk.{i}.attn_output.bias" in tensors for i in _trunk_attn):
    cfg["attn_out_bias"] = True
    print("attention output bias: added by g_bias_add")
_other = sorted({t.split(".", 2)[2] for t in tensors if t.startswith("blk.") and t.split(".", 2)[2] in ("attn_qkv.bias", "attn_output.bias")}
                - ({"attn_output.bias"} if cfg.get("attn_out_bias") and not (cfg.get("mtp") or {}).get("layers") else set()))
if _other or len(set(_qkv.values())) > 1 or (any(_qkv.values()) and cfg["attn_gate"] == "fused_q"):
    print(f"WARNING: unsupported layout: attention biases {_other or sorted(set(_qkv.values()))}"
          f"{' with a fused [q | gate] projection' if cfg['attn_gate'] == 'fused_q' else ''}: gen_llm.py adds one q/k/v "
          "bias set shared by every attention layer (g_qk_rope_store); extend it (agent port)")
elif _attn and _qkv[_attn[0]]:
    cfg["attn_bias"] = list(_qkv[_attn[0]])
    print(f"attention biases: {cfg['attn_bias']} (added by g_qk_rope_store)")
# Attention sinks (one learned logit per query head, added to the softmax denominator).
cfg["attn_sinks"] = "blk.0.attn_sinks.weight" in tensors
# MoE FFN (gen_moe.py) after the leading dense layers: router (f32/bf16), top-k, optional selection bias, shared
# experts (Qwen: with a sigmoid gate, ffn_gate_inp_shexp). llama.cpp hard-codes some routing choices per architecture
# instead of writing them to the GGUF; MOE_DEFAULTS mirrors src/models/<arch>.cpp (build_moe_ffn's norm_w and gating).
MOE_DEFAULTS = {"qwen3moe": {"gating": 1, "norm_weights": True}, "qwen35moe": {"gating": 1, "norm_weights": True},
                "qwen3next": {"gating": 1, "norm_weights": True}, "qwen2moe": {"gating": 1, "norm_weights": False},
                # gpt-oss: SOFTMAX_WEIGHT (top-k of the biased logits, softmax over the k), LLM_FFN_SWIGLU_OAI_MOE
                "gpt-oss": {"gating": 3, "norm_weights": False, "act": "swiglu_oai", "swiglu_alpha": 1.702, "swiglu_limit": 7.0}}
if A("expert_count"):
    d = MOE_DEFAULTS.get(arch, {})
    cfg["moe"] = {"experts": A("expert_count"), "used": A("expert_used_count"), "ffn": A("expert_feed_forward_length"),
                  "leading_dense": A("leading_dense_block_count", 0), "gating": A("expert_gating_func", d.get("gating", 1)),
                  "norm_weights": bool(A("expert_weights_norm", d.get("norm_weights", False))),
                  "scale": A("expert_weights_scale", 1.0) or 1.0,
                  "probs_bias": any(k.endswith("exp_probs_b.bias") for k in tensors)}
    cfg["moe"].update({k: v for k, v in d.items() if k in ("act", "swiglu_alpha", "swiglu_limit")})
    _ld = cfg["moe"]["leading_dense"]
    if f"blk.{_ld}.ffn_gate_inp.bias" in tensors:   # the router's own bias, added to its logits (gen_moe gating 3)
        cfg["moe"]["router_bias"] = True
    if all(f"blk.{_ld}.ffn_{p}_exps.bias" in tensors for p in ("gate", "up", "down")):   # per-expert output biases
        cfg["moe"]["expert_bias"] = True
    if (cfg["moe"].get("router_bias") and cfg["moe"]["gating"] != 3) or (any(f"blk.{_ld}.ffn_{p}_exps.bias" in tensors for p in ("gate", "up", "down"))
                                                                          and not cfg["moe"].get("expert_bias")):
        print("WARNING: unsupported MoE biases: a router bias is wired for gating 3 only, expert biases need all of gate/up/down")
    sh = next((v for k, v in tensors.items() if k.endswith("ffn_gate_shexp.weight")), None)
    if sh:
        cfg["moe"]["shared_ffn"] = sh["shape"][1]
        if any(k.endswith("ffn_gate_inp_shexp.weight") for k in tensors):
            cfg["moe"]["shared_gate"] = True
    if any(k.endswith("ffn_gate_up_exps.weight") for k in tensors):
        print("WARNING: fused ffn_gate_up_exps tensors: the MoE kernels expect separate ffn_gate_exps / ffn_up_exps")
    if cfg["ffn"] is None:   # no dense FFN layer: the FFN buffers hold the shared experts (or one expert row)
        cfg["ffn"] = cfg["moe"].get("shared_ffn") or cfg["moe"]["ffn"]
    print("moe:", cfg["moe"])
    import gen_moe   # the expert kernels' formats (gen_moe.FMT)
    exps = lambda layers: {tensors[f"blk.{i}.ffn_{p}_exps.weight"]["type"] for i in layers for p in ("gate", "up", "down")
                           if f"blk.{i}.ffn_{p}_exps.weight" in tensors}
    bad = exps(range(cfg["moe"]["leading_dense"], L)) - set(gen_moe.FMT)
    if bad:
        print("WARNING: no kernels for", sorted(bad), "- the MoE expert kernels (gen_moe.py) decode", sorted(gen_moe.FMT))
    if cfg.get("mtp") and exps(cfg["mtp"]["layers"]) - set(gen_moe.FMT):
        print(f"NOTE: the MTP block's experts are {sorted(exps(cfg['mtp']['layers']))}, which the MoE expert kernels don't decode: "
              "MTP drafting is left out (DFlash and n-gram drafting are still candidates)")
        del cfg["mtp"]
def generation_block(tpl):
    """Text the template emits for add_generation_prompt (what decides thinking mode)."""
    i = tpl.rfind("add_generation_prompt")
    return tpl[i:i + 300] if i >= 0 else ""


if template:
    try:
        bos_text = tokens[field("tokenizer.ggml.bos_token_id")] if field("tokenizer.ggml.bos_token_id") is not None else ""
        chat, notes = chat_probe.derive(template, bos_text)
        for n in notes:
            print("chat note:", n)
        results = chat_probe.verify(template, chat, bos_text)
        for name, ok, detail in results:
            print("chat", "OK  " if ok else "SKIP" if ok is None else "FAIL", name, "-", detail)
        cfg["chat"] = chat
        if chat["call_format"] == "json" and "<|python_tag|>" in tokens:
            # Llama 3.1 opens a call with <|python_tag|> (the template never prints it). Kept in the decoded text
            # (nonspecial) so complete_text stores the turn as generated: the KV row then matches the canonical
            # history, and content ends before it (control_json.loom's output_next).
            chat["call_prefix_token"] = "<|python_tag|>"
            chat["nonspecial_tokens"] = sorted(set(chat.get("nonspecial_tokens", [])) | {"<|python_tag|>"})
        print(f"chat: thinking={chat['thinking']} (default {'on' if chat['thinking_default'] else 'off'}), "
              f"tool calls={chat['call_format']}; generation prompt {chat['generation_prompt']!r}")
        if chat.get("reasoning_instructions"):
            print(f"chat: reasoning efforts {'/'.join(chat['reasoning_instructions'])} (default {chat['reasoning_effort_default']}) "
                  "select the system block's reasoning instructions; none = thinking off")
        if any(ok is False for _, ok, _ in results):
            print("WARNING: the derived chat block does not reproduce the template (FAIL above); fix it by hand")
    except (chat_probe.ProbeError, ValueError, KeyError) as e:
        print("WARNING: chat template not derivable automatically:", e)
fmts = sorted({v["type"] for k, v in tensors.items() if k.endswith(".weight") and len(v["shape"]) == 2})
print(f"{cfg['name']}: arch={arch} L={L} H={H} heads={heads}/{cfg['kv_heads']} d={head_dim} ffn={cfg['ffn']} vocab={vocab} formats={fmts}")
unsupported = [f for f in fmts if f not in formats.DECODES]   # what gen_llm.py decodes (formats.py)
if unsupported:
    print("WARNING: no kernels for", unsupported, "- add dequant helpers (see reference/porting-notes.md)")
if head_dim not in (64, 128, 256):
    print("WARNING: head_dim", head_dim, "is not 64, 128 or 256: the attention kernels support only those")
if cfg["chat"] is None:
    print("WARNING: no chat block; fill it by hand from tokenizer.chat_template (fields: see chat_probe.py)")
json.dump(cfg, open(out, "w"), indent=1)
print("wrote", out)
