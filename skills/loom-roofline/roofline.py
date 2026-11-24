#!/usr/bin/env python3
"""Roofline + target picker for a GGUF LLM on a Loom device.

usage: roofline.py MODEL.gguf [--cfg cfg.json] [--baseline perf.json] [--out roofline.json]
                   [--bw 228e9] [--power 85w|123w|spec | --flops 59e12] [--ctx 1000,2000,16000,32000]

Reads tensor sizes straight from the GGUF, plus (optionally) the port's cfg.json for
geometry the GGUF does not spell out (sliding_layers, head_dim_swa, kv_layers for KV
sharing). Prints, per metric, the hardware ceiling, the *practical* ceiling (ceiling x
the efficiency Loom has demonstrated on this hardware, see EFF), and - if --baseline
gives llama.cpp numbers - the target, stretch and margin over llama.cpp.

--baseline is perf_log.jsonl (its latest llama.cpp rows are used), or a JSON object of
llama.cpp ROCm numbers, keys as in perf.py output:
  {"decode_tps": 52.7, "decode_tps_2k": 48.7, "prefill_1000": 1593, "prefill_2000": 1678,
   "c4_steady": 157}
"""
import argparse, json, math, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loom-kb"))
import registry  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loom-benchmark", "scripts"))
from bench_common import perf_metrics  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loom-port-llm", "starter"))
from formats import split_parts  # noqa: E402

# Demonstrated efficiencies on Strix Halo (fraction of the roofline actually reached by
# a tuned Loom port, end to end). Update these when a port beats them; cite the source.
EFF = {
    "decode": 0.90,      # dense. Bolt-E4B 61.1 / 68.1 (the loom-kb `roofline` topic (`kb.py route --topic roofline`))
    "decode_moe": 0.78,  # expert gather. Flint 101 / 130
    "prefill": 0.33,     # of the SPEC-peak ceiling (efficiency() rescales to the power mode's). End to end. Bolt-E4B with d256/d512 flash prefill: 2339 / 7094 (Flint 0.22-0.28)
    "c4": 0.70,          # dense. Bolt 0.65-0.66
    "c4_moe": 0.85,      # Flint 0.74, LFM2.5-24B-A2B 0.91
}
# The same for speculative decoding, against drafted_ceiling() (tokens per verify pass = 1 + acceptance x depth,
# drafter cost not counted, so these sit below EFF). Measured on the published loom-fun cards, DFlash depth 3.
EFF_DRAFTED = {
    "decode": 0.69,      # dense. Qwen3.8-27B 29.2 / 42.4 (Muse-Glimmer-30B 0.68, Bolt-E4B 0.48)
    "decode_moe": 0.52,  # Flint 92.4 / 177.8 (the only drafted MoE port so far)
    "c4": 0.45,          # dense, 4 streams. Muse-Glimmer-30B 68.8 / 151.4 (Qwen3.8-27B 0.37, Bolt-E4B 0.29)
    "c4_moe": 0.44,      # Flint 134.2 / 301.8
}
# Stretch: plausible with known-but-unbuilt levers (not yet demonstrated end to end).
STRETCH = {
    "decode": 0.95,      # small-op fusion + attention at the bandwidth limit
    "decode_moe": 0.88,  # batched expert gather
    "prefill": 0.45,     # GEMM efficiency already seen in isolation, plus flash attention
    "c4": 0.85,
    "c4_moe": 0.95,
}
# Fixed per-layer cost per decode step (launches, norms, small kernels, sync) that does
# not shrink with model size. Best seen: Bolt-E4B 1.69 ms over 42 layers = 40 us.
# It dominates small models (Qwen2.5-0.5B: 1.8 ms of weights, 24 layers).
LAYER_US = 35
LAYER_US = 35
# K/V streaming at long context needs no small kernels: the bytes are one long contiguous read per head. The best
# demonstrated fraction of the bandwidth ceiling on decode @32k: llama.cpp ROCm 0.965 / Vulkan 0.955 (VL-8B and
# Qwen3-8B, whose K/V is half of the pass bytes there), Loom 0.92. The EFF decode cap (0.90, set by the per-layer small
# kernels) applies to the weight bytes only; the K/V bytes of a decode row at ctx tokens stream at KV_EFF.
KV_EFF = 0.97
# The 4-stream row (c4_steady.py): each stream sends a ~30-token prompt and decodes 400 tokens (its --max-tokens default), timed in the
# window where all four decode: the mean context a stream's pass reads is ~30 + 400 / 2 = 230 tokens of K/V (it was modeled at 600, which
# over-counted the K/V a 4-stream pass reads).
C4_CTX = 230
GATHERED = re.compile(r"(^|\.)(token_embd|per_layer_token_embd)\.weight$")
EXPERT = re.compile(r"_exps\.weight$")


def load(path):
    import gguf
    r = gguf.GGUFReader(path)
    meta = {}
    for f in r.fields.values():
        try:
            v = f.contents()
        except Exception:
            continue
        meta[f.name] = v
    tensors = {t.name: (int(t.n_bytes), [int(x) for x in t.shape]) for t in r.tensors}
    types = {t.name: t.tensor_type.name for t in r.tensors}
    for part in split_parts(path)[1:]:   # a split GGUF: the tensors of every part
        pr = gguf.GGUFReader(part)
        tensors.update({t.name: (int(t.n_bytes), [int(x) for x in t.shape]) for t in pr.tensors})
        types.update({t.name: t.tensor_type.name for t in pr.tensors})
    meta["_tensor_types"] = types   # ggml type name per tensor (the format-aware ceiling's decode cost)
    return meta, tensors


def geometry(meta, cfg):
    arch = meta.get("general.architecture", "")
    g = lambda k, d=None: meta.get(f"{arch}.{k}", d)
    first = lambda v: v[0] if isinstance(v, list) else v
    L = int(cfg.get("layers", g("block_count")))
    hidden = int(cfg.get("hidden", g("embedding_length")))
    heads = int(cfg.get("heads", first(g("attention.head_count"))))
    kvh = int(cfg.get("kv_heads", first(g("attention.head_count_kv", heads))))
    hd = int(cfg.get("head_dim", g("attention.key_length", hidden // heads)))
    hd_swa = int(cfg.get("head_dim_swa", g("attention.key_length_swa", hd)))
    window = int(cfg.get("sliding_window", g("attention.sliding_window", 0)) or 0)
    pat = g("attention.sliding_window_pattern")
    if "sliding_layers" in cfg:
        sliding = set(cfg["sliding_layers"])
    elif isinstance(pat, list) and len(pat) == L:
        sliding = {i for i, s in enumerate(pat) if s}
    else:
        sliding = set()
    kv_layers = int(cfg.get("kv_layers", L))
    n_exp = int(g("expert_count", 0) or 0)
    n_used = int(g("expert_used_count", 0) or 0)
    # Gemma-4 12B/26B/31B: the global layers have fewer KV heads (cfg kv_heads_global; kv_heads is the sliding count)
    kvh_l = [kvh if i in sliding else int(cfg.get("kv_heads_global") or kvh) for i in range(L)]
    # qwen4exp (reference/qwen4exp-example): sparse indexed attention (an indexer scores the compressed keys, the top_k
    # positions are attended) and a per-layer-embedding table of which a token reads one row per hash head
    ratios = g("attention.compress_ratios") or []
    # MLA (deepseek2, absorbed): one cache row per token (kv_lora latent + rope dims) serves as K, and its first kv_lora
    # values as V, so the cache is read once (not K + V) and attention's PV product is kv_lora wide. Prefill is cheaper
    # expanded (as vLLM/SGLang/llama.cpp run it): kv_b turns each key's latent into per-head K nope + V, then attention is
    # (nope + rope) wide for QK and v wide for PV; the expansion is 2 * kv_lora * heads * (nope + v) flops per key.
    mcfg = cfg.get("mla") or {}
    mla = mcfg.get("kv_lora") or int(g("attention.kv_lora_rank", 0) or 0)
    mla_qk = int(mcfg.get("kq_dims") or g("attention.key_length_mla", 0) or 0)
    mla_vd = int(mcfg.get("v_dims") or g("attention.value_length_mla", 0) or 0)
    return dict(arch=arch, mla_v=int(mla), mla_qk=mla_qk, mla_vd=mla_vd, L=L, hidden=hidden, heads=heads, kvh=kvh, kvh_l=kvh_l, hd=hd, hd_swa=hd_swa,
                window=window, sliding=sliding, kv_layers=kv_layers, n_exp=n_exp, n_used=n_used,
                top_k=int(g("attention.indexer.top_k", 0) or 0), idx_heads=int(g("attention.indexer.head_count", 0) or 0),
                idx_dim=int(g("attention.indexer.key_length", 0) or 0), ratios=list(ratios) if len(ratios) == L else [],
                gather_rows=len(g("ple.head_vocab_sizes", []) or []),
                n_rot=int(cfg.get("rope_dims") or g("rope.dimension_count", 0) or 0))


def weight_bytes(tensors, geo):
    """Bytes of weights streamed per decode step (1 token), and matmul params for flops. Lookup tables are gathered:
    one row per token (token_embd), or one row per hash head (qwen4exp's per_layer_token_embd: 16 rows of 90 B out of
    26.8 GiB, so the table is in the footprint but not in the bandwidth). Hyper-connection matrices (hc_*) stream like
    any weight; the 4 residual streams' activations are negligible next to them."""
    tied = "output.weight" not in tensors
    read, params, head_bytes, detail = 0.0, 0.0, 0, {}
    for name, (nb, shape) in tensors.items():
        numel = math.prod(shape)
        if GATHERED.search(name):
            rows = shape[-1] if len(shape) > 1 else 1
            b = nb / rows * (geo["gather_rows"] or 1) if "per_layer" in name else nb / rows   # rows read per token
            if tied and name == "token_embd.weight":
                b, head_bytes = nb, nb          # tied head reads it all
                params += numel
            kind = "embed/head"
        elif EXPERT.search(name) and geo["n_exp"]:
            b = nb * geo["n_used"] / geo["n_exp"]
            params += numel * geo["n_used"] / geo["n_exp"]
            kind = "experts (active)"
        else:
            b = nb
            if len(shape) > 1:
                params += numel
            if name == "output.weight":
                head_bytes = nb
                kind = "embed/head"
            else:
                kind = "attn" if ".attn_" in name else "ffn" if ".ffn_" in name else "other"
        read += b
        detail[kind] = detail.get(kind, 0) + b
    return read, params, head_bytes, detail


def attn_layers(tensors, L):
    """Layers that attend: those with a q projection. Hybrids (LFM2 short-conv, Mamba,
    delta-net) have attention in only some blocks; KV-shared layers still have q."""
    # Gated delta-net layers (qwen35/qwen3next) have a fused attn_qkv but no attention: they carry ssm_* tensors.
    recurrent = {i for i in range(L) if f"blk.{i}.ssm_a" in tensors or f"blk.{i}.ssm_conv1d.weight" in tensors}
    found = {i for i in range(L) if i not in recurrent and any(f"blk.{i}.attn_q{s}.weight" in tensors for s in ("", "kv"))}
    return found or set(range(L))


def sparse_ratio(geo, i):
    """The indexer's key compression ratio of attention layer i (qwen4exp), 0 for a dense layer."""
    return geo["ratios"][i] if geo["top_k"] and geo["ratios"] else 0


def kv_read_bytes(geo, ctx):
    """F16 K+V bytes read per decode token with `ctx` tokens cached (every attending layer). A sparse-attention layer
    reads the top_k selected positions' K+V plus its indexer's keys (ctx / ratio positions, BF16): an estimate, the
    selection itself is not modelled."""
    tot = 0
    for i in sorted(geo["attn"]):
        sl = i in geo["sliding"]
        hd = geo["hd_swa"] if sl else geo["hd"]
        n = min(ctx, geo["window"]) if (sl and geo["window"]) else ctx
        if sparse_ratio(geo, i):
            tot += (ctx // geo["ratios"][i]) * geo["idx_dim"] * 2
            n = min(n, geo["top_k"])
        tot += (1 if geo.get("mla_v") else 2) * geo["kvh_l"][i] * hd * 2 * n
    return tot


def attn_flops_per_token(geo, P):
    """Prefill attention flops per prompt token, causal, prompt length P."""
    tot = 0
    for i in sorted(geo["attn"]):
        sl = i in geo["sliding"]
        hd = geo["hd_swa"] if sl else geo["hd"]
        avg = min(P / 2, geo["window"]) if (sl and geo["window"]) else P / 2
        if sparse_ratio(geo, i):   # the indexer scores every compressed key, attention covers the top_k only
            tot += 2 * geo["idx_heads"] * geo["idx_dim"] * (P / 2) / geo["ratios"][i]
            avg = min(avg, geo["top_k"])
        absorbed = 2 * geo["heads"] * (hd + (geo.get("mla_v") or hd)) * avg
        if geo.get("mla_v") and geo.get("mla_qk") and geo.get("mla_vd"):
            nope = geo["mla_qk"] - (hd - geo["mla_v"])
            expanded = (2 * geo["heads"] * (geo["mla_qk"] + geo["mla_vd"]) * avg
                        + 2 * geo["mla_v"] * geo["heads"] * (nope + geo["mla_vd"]))
            absorbed = min(absorbed, expanded)
        tot += absorbed
    return tot


# Devices (--device): the memory bandwidth a streaming kernel reaches and the peak dense f16 WMMA rate
# (CUs x 512 flops/clock x boost clock on RDNA3/3.5; RDNA4 doubles the dense f16 WMMA rate per CU: x 1024). The CLI's --bw / --flops override them.
# EFF / STRETCH / LAYER_US above were calibrated on Strix Halo; on another device they are an out-of-sample estimate
# until a row in the loom-kb `roofline` topic (`kb.py route --topic roofline`) says otherwise.
# The device presets are rows of the registry (loom-kb/device/<family>/<device>/device.json "roofline": spec presets and the DevLab
# runner's measured one, e.g. radeon-r9700 vs radeon-r9700s); the notes on bandwidth and peak flops live there.
# Compute ceilings are stated at a power mode: "flops" is the spec sheet's peak (the "spec" mode); a preset's "power_modes"
# hold measured sustained random-data WMMA rates per platform power limit (Strix Halo: 85 W BIOS balanced, the DevLab
# runners' mode, is the default; 123 W BIOS performance). Every compute ceiling and prefill efficiency is "@ <mode label>".
DEVICES = {k: dict(name=p["name"], bw=p["bw"], flops=p["flops"],
                   power_modes={"spec": {"label": "spec", "flops": p["flops"]}, **p.get("power_modes", {})},
                   power_default=p.get("power_default", "spec"))
           for k, p in registry.roofline_presets().items()}
AT_CEILING = registry.methodology()["roofline"]["at_ceiling"]
DEVICE_KEY = "strix-halo"


def compute_ceiling(device=DEVICE_KEY, power=None, flops=None):
    """The compute ceiling of a device preset: {"flops", "label", "mode", "spec_flops"}. power: a key of the preset's
    power_modes ("spec" always exists; None = the preset's power_default). flops: an explicit override (label "override")."""
    dev = DEVICES[device]
    mode = power or dev["power_default"]
    if mode not in dev["power_modes"]:
        raise ValueError(f"{device}: no power mode {mode!r} (have {', '.join(sorted(dev['power_modes']))})")
    pm = dev["power_modes"][mode]
    if flops and flops != pm["flops"]:
        return dict(flops=flops, label="override", mode="override", spec_flops=dev["flops"])
    return dict(flops=pm["flops"], label=pm["label"], mode=mode, spec_flops=dev["flops"])


DEVICE, BW = DEVICES[DEVICE_KEY]["name"], DEVICES[DEVICE_KEY]["bw"]
COMPUTE = compute_ceiling()
FLOPS = COMPUTE["flops"]   # the default power mode's sustained rate (Strix Halo: 85 W); DEVICES[...]["flops"] is the spec peak


def shape_of(meta, tensors, cfg=None):
    """What the ceilings need from a GGUF's metadata and tensors (plus the port's cfg.json): geometry, the weight
    bytes one decode token reads (by kind), the matmul params and the head's params."""
    geo = geometry(meta, cfg or {})
    geo["attn"] = attn_layers(tensors, geo["L"])
    wb, params, head, detail = weight_bytes(tensors, geo)
    hb = tensors.get("output.weight", tensors.get("token_embd.weight"))[1]
    types = meta.get("_tensor_types") or {}
    tlist = [(n, nb, shape, types.get(n, "F16")) for n, (nb, shape) in tensors.items()]
    return dict(geo=geo, wb=wb, params=params, head=head, detail=detail, head_params=hb[0] * hb[1], tensors=tlist)


def model_shape(gguf_path, cfg=None):
    meta, tensors = load(gguf_path)
    return shape_of(meta, tensors, cfg)


TENSOR_BYTES = {"bf16": 2.0, "f16": 2.0, "f32": 4.0, "mxfp4s": 17 / 32}   # bytes per element of a llm_config.json tensor type (mxfp4s: 4-bit nibbles + one E8M0 scale per 32)


def model_shape_from_config(cfg):
    """The same model shape for a package with no GGUF (a safetensors checkpoint): the tensor table and geometry of its llm_config.json
    (internal GGUF-style names, shapes [in, out], type -> bytes per element). The next-token-prediction block (blk.L, L = layers) is a
    drafter: like the drafter's passes, its weights are not counted (a verify pass and an undrafted decode read the trunk only)."""
    L = int(cfg["layers"])
    tensors = {}
    for name, t in cfg["tensors"].items():
        m = re.match(r"blk\.(\d+)\.", name)
        if m and int(m.group(1)) >= L:
            continue
        bpe = TENSOR_BYTES.get(t["type"])
        if bpe is None:
            raise ValueError(f"{name}: tensor type {t['type']!r} has no byte size in roofline.TENSOR_BYTES")
        tensors[name] = (int(math.prod(t["shape"]) * bpe), [int(x) for x in t["shape"]])
    types = {n: {"mxfp4s": "MXFP4"}.get(t["type"], t["type"].upper()) for n, t in cfg["tensors"].items()}
    return shape_of({"general.architecture": cfg["arch"], "_tensor_types": types}, tensors, cfg)


def pass_bytes(m, tokens=1, ctx=0, streams=1):
    """Bytes one forward pass reads when it decodes `tokens` tokens at once (one per stream, or a verify pass):
    the weights once (MoE: the expected distinct experts those tokens route to) plus each stream's KV at ctx."""
    w, exp = m["wb"], m["detail"].get("experts (active)", 0)
    if exp and tokens > 1:   # tokens route independently: ~min(n_exp, tokens*k) distinct experts per layer
        n, k = m["geo"]["n_exp"], m["geo"]["n_used"]
        w = w - exp + exp * n * (1 - (1 - k / n) ** tokens) / k
    return w + streams * kv_read_bytes(m["geo"], ctx)


def ceilings(m, bw=BW, flops=FLOPS, ctxs=(1000, 2000, 16000, 32000), gain=1.0):
    """Hardware ceilings in tok/s by roofline.json metric name: decode_tps (no context), decode_tps_<c//1000>k and
    prefill_<P> for each size in ctxs, c4_steady (4 streams, KV at C4_CTX tokens). gain multiplies decode and c4 (a
    measured drafting gain, --draft-tune)."""
    out = {"decode_tps": bw / pass_bytes(m) * gain}
    for c in ctxs:
        out[f"decode_tps_{c // 1000}k"] = bw / pass_bytes(m, ctx=c) * gain
    for P in ctxs:   # 2 flops per matmul param per token, the head on the last token only, causal attention
        out[f"prefill_{P}"] = flops / (2 * (m["params"] - m["head_params"]) + attn_flops_per_token(m["geo"], P))
    out["c4_steady"] = 4 * bw / pass_bytes(m, tokens=4, ctx=C4_CTX, streams=4) * gain
    if m["geo"]["n_exp"]:   # MoE: streams of similar text share experts; the hard bound (all 4 tokens on the same experts) is what a measurement may not exceed
        out["c4_steady_shared"] = 4 * bw / (pass_bytes(m, ctx=C4_CTX) + 3 * kv_read_bytes(m["geo"], C4_CTX)) * gain
    return out


def drafted_ceiling(m, depth, acceptance, bw=BW, ctx=0, streams=1):
    """Decode ceiling with speculative decoding at `depth` drafted tokens and `acceptance` (accepted / proposed draft
    tokens, the spec gate's count): each verify pass checks depth+1 positions per stream and commits on average
    1 + acceptance*depth tokens per stream, for one read of the weights (MoE: the distinct experts of all those
    positions) and of each stream's KV. The drafter's own passes are not counted: an upper bound."""
    return streams * (1 + acceptance * depth) * bw / pass_bytes(m, tokens=streams * (depth + 1), ctx=ctx, streams=streams)


# The drafted rows by roofline.json metric name: (KV context read per decoded token, streams). card_lib.DRAFTED_ROWS
# names the same rows by card key.
DRAFTED_ROWS = {"decode_tps": (0, 1), "decode_tps_2k": (2000, 1), "decode_tps_32k": (32000, 1), "c4_steady": (C4_CTX, 4)}


def drafted_row_ceiling(m, metric, depth, acceptance, bw=BW):
    """A drafted row's ceiling (roofline.json metric name) at `acceptance`, ideally the one its own timed requests
    measured (perf.py, c4_steady.py "acceptance"): drafted_ceiling() at the row's context and streams. None for a
    row speculative decoding does not apply to. The card (make_info.py roofline) and audit.py both use this."""
    if metric not in DRAFTED_ROWS or acceptance is None or not depth:
        return None
    ctx, streams = DRAFTED_ROWS[metric]
    return drafted_ceiling(m, depth, acceptance, bw, ctx=ctx, streams=streams)


# ---------------------------------------------------------------------------------------------------------------------
# Format-aware ceiling: every cost ANY engine must pay on this device for this model and weight format, and no software
# inefficiency (launch gaps, stalls, a GEMM below peak, a tile that re-decodes its weights). Per op (each weight matrix,
# each layer's attention and its elementwise work), time = max(matrix, vector ALU, memory) - perfect overlap inside an
# op - and the step is the sum over ops. Rates are hardware peaks measured by microbenchmarks (device preset: bw, the
# power mode's WMMA flops, its "valu" and "trans" rates, devlab-kbench wmma_peak / valu_peak), never our kernels' counts.
#  matrix: flops at the WMMA rate, rows in 16-row tiles (a GEMM over T tokens runs ceil(T/16)*16 rows; an MoE expert
#          receiving r of the T tokens costs E[ceil(r/16)]*16 rows, r ~ Binomial(T, k/E)).
#  valu:   lane-instructions (one 32-bit VALU op on one lane; a packed-f16 op handles 2 values) at the "valu" rate (f32
#          softmax / elementwise ops at "valu_dual": VOPD dual issue), and transcendentals (exp, rcp) at the "trans"
#          rate, overlapped: max of the two. Weight decode is paid once per
#          weight touched per pass (FMT: the format's minimum, below), softmax 2 ops + 1 exp per (query, key, head)
#          score, norms / RoPE / SwiGLU / residual adds / f32 -> f16 conversions at their elementwise minimum.
#  memory: the weights touched (MoE: the expected distinct experts E*(1-(1-k/E)^T) per layer), the K/V (or MLA latent)
#          read once and the new rows written once, each layer's input and output activations once (f16).
# A weight matrix takes the cheapest of three implementations: f16 WMMA on weights decoded once (FMT "f16"), int8 WMMA on
# integer codes with the block scales applied to the int32 partial sums (FMT "i8" per weight + "out" per token row,
# output and scale block, + activation quantization; iu8 WMMA = f16 rate on gfx11), or VALU dot products (v_dot2 f16 /
# v_dot4 i8, no 16-row padding: what wins for a few tokens). Weights may be repacked once at load time (any bit layout
# of the same bytes), so the decode counts assume the friendliest layout.
#
# FMT: VALU lane-instructions per weight value (the minimum, derived from the format definitions in ggml-common.h):
#  f16:  to an f16 WMMA operand with the scale applied. 4-bit linear codes: magic-exponent insertion, (w & 0x000F000F |
#        0x64006400) gives two f16 values 1024+q per op; masks at bits 0-3 and 4-7 of each half need no shift, the
#        other 4 nibbles of a word one shift: 5 ops / 8 values = 0.625, then v_pk_add (offset) + v_pk_mul (scale) per 2
#        values (1.0; a v_pk_fma would be one op but runs at 0.3x the issue rate, valu_peak): 1.625. 8-bit codes: one
#        v_perm per 2 values builds 1024+q (0.5) + pk_add + pk_mul (1.0): 1.5. Split planes (Q5_*, Q6_K, Q2_K/Q3_K) and
#        codebooks (IQ4_*): the int8 codes (below), then v_perm + pk_add + pk_mul: i8 + 1.5.
#        MXFP4 (e2m1): sign and 3 bits placed into an f16's exponent/mantissa give 2^-14 x the value (normal and
#        subnormal alike): 7 ops / 8 values for the bit placement + v_pk_mul by 2^14 x the E8M0 scale (0.5): 1.375.
#        f32 -> f16: one v_cvt_pk per 2 values (0.5). f16 / bf16: 0 (bf16 WMMA runs at the f16 rate).
#  i8:   to int8 codes (4 per register): nibbles (w & 0x0F0F0F0F), (w >> 4) & 0x0F0F0F0F: 3 ops / 8 values (0.375);
#        a 5th / 6th bit plane merged with one shift + and_or per 4 values (+0.5); 2-bit 0.25, 3-bit 0.75; a 16-entry
#        int8 codebook (IQ4_NL / IQ4_XS, llama.cpp's get_int_from_table_16) 2 v_perm + 1 select per 4 values (+0.75);
#        e2m1 as int8 (2 x the value: an 8-entry table, 1 v_perm per 4 values + the sign, +0.5). None: no integer path.
#  out:  ops per (token row, output, scale block) of the int8 path: cvt i32 -> f32 + fma with the block scale (2);
#        +1 fma for a min / offset term (affine formats: the block's activation sum x the min).
#  blk:  values per scale block; sb: ops per block to build its scale (6-bit K-quant scales unpacked, d x sc, ...).
_LIN4 = dict(blk=32, f16=1.625, i8=0.375, out=3)
FMT = {
    "F16": dict(blk=0, f16=0.0, i8=None), "BF16": dict(blk=0, f16=0.0, i8=None), "F32": dict(blk=0, f16=0.5, i8=None),
    "Q8_0": dict(blk=32, f16=1.5, i8=0.0, out=2, sb=1),
    "Q4_0": dict(_LIN4, sb=1), "Q4_1": dict(_LIN4, sb=2), "Q4_K": dict(_LIN4, sb=4),
    "Q5_0": dict(blk=32, f16=2.375, i8=0.875, out=3, sb=1), "Q5_1": dict(blk=32, f16=2.375, i8=0.875, out=3, sb=2),
    "Q5_K": dict(blk=32, f16=2.375, i8=0.875, out=3, sb=4), "Q6_K": dict(blk=16, f16=2.375, i8=0.875, out=3, sb=2),
    "Q2_K": dict(blk=16, f16=1.75, i8=0.25, out=3, sb=4), "Q3_K": dict(blk=16, f16=2.25, i8=0.75, out=3, sb=4),
    "IQ4_NL": dict(blk=32, f16=2.625, i8=1.125, out=2, sb=1), "IQ4_XS": dict(blk=32, f16=2.625, i8=1.125, out=2, sb=4),
    "MXFP4": dict(blk=32, f16=1.375, i8=0.875, out=2, sb=1),
}
# Other codebook formats (IQ1/IQ2/IQ3, TQ*): the 16-entry codebook cost, an estimate (no model in the fleet uses them).
FMT_DEFAULT = dict(blk=32, f16=2.625, i8=1.125, out=2, sb=4)
CHUNK = 4096           # the card preset's prefill chunk (build.py MoE default); a shorter prompt is one pass of its length
ACT_BYTES = 2          # f16 activations


def fmt_cost(ty):
    return FMT.get(ty, FMT_DEFAULT)


def _ceil16_expect(T, p):
    """E[ceil(r/16)] * 16 for r ~ Binomial(T, p): the expected WMMA rows of an MoE expert."""
    if T <= 0 or p <= 0:
        return 0.0
    if p >= 1:
        return 16 * math.ceil(T / 16)
    lp, lq = math.log(p), math.log1p(-p)
    acc, lc = 0.0, 0.0           # lc = log C(T, r)
    for r in range(0, T + 1):
        if r:
            lc += math.log(T - r + 1) - math.log(r)
        lw = lc + r * lp + (T - r) * lq
        if lw > -60:
            acc += math.exp(lw) * math.ceil(r / 16)
    return 16 * acc


def _gemm_time(numel, K, N, ty, rows_mat, rows_real, rates, groups=1):
    """(seconds, bound) of the compute of `groups` weight matrices of numel values each (K inputs, N outputs), each weight
    decoded once, over rows_real token rows in all (rows_mat after 16-row tiling, summed over the groups): the cheapest of
    f16 WMMA, int8 WMMA (block scales on the int32 partial sums) and VALU dot products (no tile padding)."""
    F, V = rates["flops"], rates["valu"]
    c = fmt_cost(ty)
    blocks = numel / c["blk"] if c["blk"] else 0
    setup = groups * blocks * c.get("sb", 0)
    conv = 0.5 * rows_real * N                       # f32 -> f16 of the outputs (packed)
    mat = 2 * rows_mat * numel / F
    cand = [(mat, (groups * numel * c["f16"] + setup + conv) / V)]
    if c["i8"] is not None:
        cand.append((mat, (groups * numel * c["i8"] + setup + rows_real * blocks * c["out"] + rows_real * K + conv) / V))
    cand.append((0.0, (groups * numel * c["f16"] + setup + rows_real * numel / 2 + conv) / V))   # v_dot2_f32_f16
    t, m_t, v_t = min((max(m_t, v_t), m_t, v_t) for m_t, v_t in cand)
    return t, "matrix" if m_t >= v_t else "valu"


def _attn_keys(geo, i, prefix, T):
    """Total (query, key) pairs of attention layer i for T queries after `prefix` cached tokens (causal), and the
    indexer's flops (qwen4exp sparse layers)."""
    sl = i in geo["sliding"]
    win = geo["window"] if (sl and geo["window"]) else 0
    topk = geo["top_k"] if sparse_ratio(geo, i) else 0
    if T == 1:
        n = prefix + 1
        pairs = min(n, win) if win else n
        pairs = min(pairs, topk) if topk else pairs
        idx = 2 * geo["idx_heads"] * geo["idx_dim"] * n / geo["ratios"][i] if topk else 0
        return pairs, idx
    avg = prefix + (T + 1) / 2
    if win:
        avg = min(avg, win)
    idx = 2 * geo["idx_heads"] * geo["idx_dim"] * avg / geo["ratios"][i] * T if topk else 0
    if topk:
        avg = min(avg, topk)
    return avg * T, idx


def _key_bytes(geo, i):
    """Cache bytes per token of attention layer i (MLA: the latent + rope row once; else K + V, f16)."""
    hd = geo["hd_swa"] if i in geo["sliding"] else geo["hd"]
    return (1 if geo.get("mla_v") else 2) * geo["kvh_l"][i] * hd * 2


def format_pass(m, T, prefix, rates, streams=1, last=True, pad_rows=None):
    """Seconds of one forward pass over T tokens at the format-aware ceiling, and the breakdown.
    Prefill: T tokens of one sequence after `prefix` cached tokens (streams=1); the head runs on the last token of the
    last pass only. Decode: T = streams tokens, one per stream, each after `prefix` cached tokens; the head runs on
    every token. pad_rows: the rows dense GEMMs, attention queries and elementwise kernels run (loom-serve's
    power-of-two epoch); experts keep their real tokens."""
    geo, F, BW = m["geo"], rates["flops"], rates["bw"]
    V, X = rates["valu"], rates["trans"]
    DU = V / rates.get("valu_dual", V)      # f32 softmax / elementwise ops may dual-issue (VOPD): fewer issue slots
    decode = streams > 1 or T == 1
    R = pad_rows or T                       # rows the per-token work runs
    rows16 = 16 * math.ceil(R / 16)
    E, k = geo["n_exp"], geo["n_used"]
    distinct = E * (1 - (1 - k / E) ** T) if E else 0
    exp_rows = _ceil16_expect(T, k / E) if E else 0
    head_rows = T if decode else (1 if last else 0)
    ops = {}                                # category -> [seconds, {bound: seconds}]

    def add(cat, t, bound):
        o = ops.setdefault(cat, [0.0, {}])
        o[0] += t
        o[1][bound] = o[1].get(bound, 0.0) + t

    def op(cat, *cands):
        """cands: (matrix seconds, valu ops, transcendentals, bytes) per implementation; the cheapest one counts."""
        best = None
        for mat, valu, trans, mem in cands:
            v = max(valu / V, trans / X)
            t = max(mat, v, mem / BW)
            if best is None or t < best[0]:
                best = (t, "matrix" if t == mat else "valu" if t == v else "memory")
        add(cat, *best)

    tied = not any(n == "output.weight" for n, *_ in m["tensors"])
    mla_kv = {}
    for name, nb, shape, ty in m["tensors"]:
        numel = math.prod(shape)
        lm = re.match(r"blk\.(\d+)\.", name)
        if GATHERED.search(name):
            rows = shape[-1] if len(shape) > 1 else 1
            per = (geo["gather_rows"] or 1) if "per_layer" in name else 1
            byt = min(nb, nb / rows * per * R)
            op("embed", (0, (numel / rows) * per * R * fmt_cost(ty)["f16"], 0, byt))
            if tied and name == "token_embd.weight" and head_rows:
                t, b = _gemm_time(numel, shape[0], shape[1], ty, 16 * math.ceil(head_rows / 16), head_rows, rates)
                t = max(t, nb / BW)
                add("head", t, "memory" if t == nb / BW else b)
            continue
        if name == "output.weight":
            if head_rows:
                t, b = _gemm_time(numel, shape[0], shape[1], ty, 16 * math.ceil(head_rows / 16), head_rows, rates)
                t2 = max(t, nb / BW)
                add("head", t2, "memory" if t2 == nb / BW else b)
            continue
        if len(shape) < 2:                   # norm weights, biases: read once; their work is in "elementwise"
            op("elementwise", (0, 0, 0, nb))
            continue
        if lm and re.search(r"\.attn_(k_b|v_b)\.weight$", name) and geo.get("mla_v"):
            mla_kv.setdefault(int(lm.group(1)), []).append((name, nb, shape, ty))
            continue
        if EXPERT.search(name) and E:
            per_numel, per_nb = numel / E, nb / E
            # matrix rows: the sum over experts of E[16*ceil(r/16)]; decode each distinct expert's weights once
            t, b = _gemm_time(per_numel, shape[0], shape[1], ty, exp_rows * E, T * k, rates, groups=distinct)
            t2 = max(t, per_nb * distinct / BW)
            add("experts", t2, "memory" if t2 == per_nb * distinct / BW else b)
            continue
        K, N = shape[0], numel // shape[0]
        t, b = _gemm_time(numel, K, N, ty, rows16, R, rates)
        t2 = max(t, nb / BW)
        cat = "attn_proj" if ".attn_" in name else "ffn" if ".ffn_" in name else "other"
        add(cat, t2, "memory" if t2 == nb / BW else b)

    # attention core per attending layer
    for i in sorted(geo["attn"]):
        hd = geo["hd_swa"] if i in geo["sliding"] else geo["hd"]
        heads, kvh = geo["heads"], geo["kvh_l"][i]
        if decode:
            pairs, idx = _attn_keys(geo, i, prefix, 1)
            pairs, idx = pairs * streams, idx * streams
            mem = streams * _key_bytes(geo, i) * (min(prefix + 1, geo["window"]) if (i in geo["sliding"] and geo["window"]) else prefix + 1)
            if sparse_ratio(geo, i):
                mem = streams * ((prefix // geo["ratios"][i]) * geo["idx_dim"] * 2 + _key_bytes(geo, i) * min(prefix + 1, geo["top_k"]))
            per_pair_rows = 16 * math.ceil(heads / kvh / 16) * kvh   # a stream's query heads per KV head, 16-row tiles
        else:
            pairs, idx = _attn_keys(geo, i, prefix, T)
            pairs_pad = _attn_keys(geo, i, prefix, R)[0] if R != T else pairs
            win = geo["window"] if (i in geo["sliding"] and geo["window"]) else 0
            mem = _key_bytes(geo, i) * ((min(prefix, win) if win else prefix) + T)
            if sparse_ratio(geo, i):
                mem += (prefix // geo["ratios"][i]) * geo["idx_dim"] * 2
        if geo.get("mla_v"):
            kb = mla_kv.get(i, [])
            kb_numel = sum(math.prod(sh) for _, _, sh, _ in kb)
            kb_nb = sum(nb for _, nb, _, _ in kb)
            kb_dec = sum(math.prod(sh) * fmt_cost(ty)["f16"] for _, _, sh, ty in kb)
            qk_abs, v_abs = hd, geo["mla_v"]
            if decode:
                # absorbed (the decode form): heads are the rows of one shared latent row per key
                mat = 2 * (per_pair_rows * pairs) * (qk_abs + v_abs) / F + 2 * 16 * math.ceil(T / 16) * kb_numel / F
                op("attention", (mat, 2 * DU * heads * pairs + kb_dec, heads * pairs, mem + kb_nb))
            else:
                # three forms, the cheapest counts: absorbed (latent rows as K and V); expanded, re-expanding every
                # visible key's latent this pass (kv_b on prefix + T keys); expanded with the expanded K/V of earlier
                # passes kept in memory (read once, the new keys' written once, next to their latent rows)
                pp = pairs_pad
                vals, tr = 2 * DU * heads * pp + kb_dec, heads * pp
                att_x = 2 * heads * pp * (geo["mla_qk"] + geo["mla_vd"])
                exp_b = heads * (geo["mla_qk"] - (hd - geo["mla_v"]) + geo["mla_vd"]) * 2    # expanded K nope + V per key, f16
                op("attention",
                   ((2 * heads * pp * (qk_abs + v_abs) + 2 * rows16 * kb_numel) / F, vals, tr, mem + kb_nb),
                   ((att_x + 2 * (prefix + T) * kb_numel) / F, vals, tr, mem + kb_nb),
                   ((att_x + 2 * T * kb_numel) / F, vals, tr, exp_b * (prefix + T) + _key_bytes(geo, i) * T + kb_nb))
        else:
            if decode:   # each stream's query heads per KV head are the rows (16-row tiles)
                mat = 2 * per_pair_rows * pairs * 2 * hd / F
                pp = pairs
            else:
                pp = pairs_pad
                mat = 2 * heads * pp * 2 * hd / F
            op("attention", (mat + idx / F, 2 * DU * heads * pp, heads * pp, mem))

    # elementwise per layer: norms (2 ops per element), RoPE (1.5 per rotated element), SwiGLU (3 ops + exp + rcp per
    # intermediate element), router softmax/sigmoid (2 ops + 1 exp per logit), 2 residual adds; each layer's input and
    # output activations once
    hidden, L = geo["hidden"], geo["L"]
    norm_el = sum(math.prod(sh) for n, _, sh, _ in m["tensors"] if len(sh) == 1 and n.endswith("norm.weight") and n.startswith("blk."))
    ffn_inter = 0.0
    for name, nb, shape, ty in m["tensors"]:
        if re.search(r"\.ffn_up(_shexp)?\.weight$", name):
            ffn_inter += shape[1]
        elif re.search(r"\.ffn_up_exps\.weight$", name) and E:
            ffn_inter += shape[1] * k
    rope_el = (geo["n_rot"] or 0) * (geo["heads"] + (1 if geo.get("mla_v") else geo["kvh"])) * len(geo["attn"])
    router = E * sum(1 for n, *_ in m["tensors"] if n.endswith("ffn_gate_inp.weight"))
    valu = DU * R * (2 * norm_el + 1.5 * rope_el + 3 * ffn_inter + 2 * router + 2 * 0.5 * hidden * L)
    trans = R * (2 * ffn_inter + router)
    op("elementwise", (0, valu, trans, L * R * hidden * ACT_BYTES * 2))
    total = sum(v[0] for v in ops.values())
    by_bound = {}
    for v in ops.values():
        for b, t in v[1].items():
            by_bound[b] = by_bound.get(b, 0.0) + t
    return total, ops, by_bound


def format_rates(device=DEVICE_KEY, power=None, flops=None, bw=None):
    """The rates of the format-aware ceiling: {"flops", "bw", "valu", "trans"} at the power mode (None if the preset has
    no measured VALU rates)."""
    dev = DEVICES[device]
    comp = compute_ceiling(device, power, flops)
    pm = dev["power_modes"].get(comp["mode"], {}) if comp["mode"] != "override" else dev["power_modes"].get(dev["power_default"], {})
    if not pm.get("valu"):
        return None
    return dict(flops=comp["flops"], bw=bw or dev["bw"], valu=pm["valu"], valu_dual=pm.get("valu_dual", pm["valu"]), trans=pm["trans"])


def _summ(ops, by_bound, total):
    return dict(bound=max(by_bound, key=by_bound.get),
                ms_by_bound={b: round(t * 1e3, 3) for b, t in sorted(by_bound.items())},
                ops={c: dict(ms=round(v[0] * 1e3, 3), bound=max(v[1], key=v[1].get)) for c, v in sorted(ops.items())},
                ms=round(total * 1e3, 3))


def format_ceiling(m, metric, rates, chunk=CHUNK, padded=False):
    """The format-aware ceiling of one roofline.json metric (tok/s) and its breakdown, or None (no VALU rates).
    prefill_<P>: passes of `chunk` tokens (a shorter prompt is one pass); padded: each pass runs loom-serve's
    power-of-two epoch shape (min(chunk, next power of two of the pass)). decode_tps[_<c>k]: 1 token after c cached;
    c4_steady: 4 streams, one token each, after C4_CTX cached (independent routing: the distinct-expert union)."""
    if rates is None:
        return None
    if metric.startswith("prefill_"):
        P = int(metric.split("_")[1])
        total, agg_ops, agg_b, done = 0.0, {}, {}, 0
        while done < P:
            T = min(chunk, P - done)
            pad = min(chunk, 1 << (T - 1).bit_length()) if padded else None
            t, ops, bb = format_pass(m, T, done, rates, last=done + T >= P, pad_rows=pad)
            total += t
            for c, v in ops.items():
                a = agg_ops.setdefault(c, [0.0, {}])
                a[0] += v[0]
                for b, x in v[1].items():
                    a[1][b] = a[1].get(b, 0.0) + x
            for b, x in bb.items():
                agg_b[b] = agg_b.get(b, 0.0) + x
            done += T
        return dict(ceiling=P / total, **_summ(agg_ops, agg_b, total))
    if metric == "c4_steady":
        t, ops, bb = format_pass(m, 4, C4_CTX, rates, streams=4)
        return dict(ceiling=4 / t, **_summ(ops, bb, t))
    ctx = decode_ctx(metric) if metric != "decode_tps" else 0
    t, ops, bb = format_pass(m, 1, ctx, rates)
    return dict(ceiling=1 / t, **_summ(ops, bb, t))


def efficiency(m, metric, ceiling, table=None, tokens_per_pass=None, compute=None):
    """The fraction of `ceiling` Loom ports have demonstrated for this kind of metric (table EFF, or STRETCH):
    prefill, c4 (dense / MoE) or decode (dense / MoE), decode capped by the fixed per-layer cost (LAYER_US).
    tokens_per_pass: a drafted_ceiling() (speculative decoding): EFF_DRAFTED, the per-layer cost paid once per pass.
    compute: the compute_ceiling() the prefill ceiling used (None: the spec peak, any device): the table's prefill fraction is of the
    spec-peak ceiling, so it scales by spec / flops (the target in tok/s does not depend on the power mode's label)."""
    table = table or (EFF_DRAFTED if tokens_per_pass else EFF)
    moe = bool(m["geo"]["n_exp"])
    if metric.startswith("prefill"):
        if compute is None:   # no power mode given: the ceiling is the spec peak the table was calibrated against
            return min(1.0, table["prefill"])
        return min(1.0, table["prefill"] * compute["spec_flops"] / compute["flops"])
    if metric.startswith("c4"):
        return table["c4_moe" if moe else "c4"]
    floor_ms = m["geo"]["L"] * LAYER_US / 1000
    t = (tokens_per_pass or 1) * 1000
    eff = min(table["decode_moe" if moe else "decode"], (t / (t / ceiling + floor_ms)) / ceiling)
    ctx = decode_ctx(metric)
    if ctx:   # the K/V share of the pass streams at KV_EFF, the weight share at eff
        kv = kv_read_bytes(m["geo"], ctx)
        share = kv / pass_bytes(m, ctx=ctx)
        eff = 1 / ((1 - share) / eff + share / max(KV_EFF, eff))
    return eff


def decode_ctx(metric):
    """The context of a decode row by roofline.json metric name (decode_tps_32k, or the card key decode_32k): 0 if none."""
    mt = re.fullmatch(r"decode(?:_tps)?_(\d+)k", metric)
    return int(mt.group(1)) * 1000 if mt else 0


def pick_target(name, base, ceiling, eff, stretch):
    """target = ceiling x demonstrated efficiency; stretch = ceiling x STRETCH.

    The margin over llama.cpp is reported as is, with no cap and no label: it is
    whatever the hardware allows given llama.cpp's own efficiency (2x-6x is normal
    when llama.cpp is far from the roofline). The only class is 'at-ceiling'
    (< 3 % headroom: target parity and spend the effort elsewhere).
    """
    practical, st = ceiling * eff, ceiling * stretch
    r = dict(metric=name, ceiling=round(ceiling, 1), target=round(practical, 1),
             stretch=round(st, 1))
    if base is None:
        return r
    h = practical / base - 1
    r.update(baseline=base, baseline_eff=round(base / ceiling, 3),
             margin=round(h, 3), stretch_margin=round(st / base - 1, 3),
             at_ceiling=h < AT_CEILING)
    if h < AT_CEILING:
        r["target"] = base
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--cfg")
    ap.add_argument("--baseline")
    ap.add_argument("--out")
    ap.add_argument("--device", default=DEVICE_KEY, choices=sorted(DEVICES),
                    help="hardware preset for --bw / --flops (default strix-halo)")
    ap.add_argument("--bw", type=float, default=None, help="streaming bandwidth, bytes/s (default: the --device preset)")
    ap.add_argument("--flops", type=float, default=None, help="dense f16 flop/s (default: the --device preset at --power)")
    ap.add_argument("--power", default=None,
                    help="power mode of the compute ceiling: a key of the preset's power_modes, or 'spec' for the spec-sheet peak "
                         "(default: the preset's power_default; strix-halo: 85w, the DevLab runners' BIOS mode; also 123w)")
    ap.add_argument("--ctx", default="1000,2000,16000,32000")
    ap.add_argument("--budget-h", type=float, default=0.5,
                    help="the user's time budget for optimization (default 0.5 h; qualify.sh computes it for end-to-end jobs); saved for progress.py")
    ap.add_argument("--reserve-min", type=float, default=10,
                    help="minutes kept back at the end of the budget for the final gates + report "
                         "(qualify.sh passes 0: its budget already excludes them)")
    ap.add_argument("--draft-tune", default=None,
                    help="bench/draft_tune.json: with a drafter chosen, the decode and c4 ceilings are multiplied by its "
                         "measured gain (tok/s with drafting / without, same requests): one weight pass then commits "
                         "several tokens, so the 1-token bandwidth ceiling no longer bounds decode")
    ap.add_argument("--chunk", type=int, default=CHUNK,
                    help="prefill chunk of the format-aware ceiling (default 4096, the card preset's; a shorter prompt is one pass)")
    ap.add_argument("--goal", action="append", default=[],
                    help="user target, METRIC=TOK_S (e.g. decode_tps=70); replaces target/stretch for it")
    a = ap.parse_args()
    dev = DEVICES[a.device]
    a.bw = a.bw or dev["bw"]
    try:
        comp = compute_ceiling(a.device, a.power, a.flops)
    except ValueError as e:
        ap.error(str(e))
    a.flops = comp["flops"]
    cfg = json.load(open(a.cfg)) if a.cfg else {}
    m = model_shape(a.gguf, cfg)
    geo, wb, params, head, detail = m["geo"], m["wb"], m["params"], m["head"], m["detail"]
    base = {}
    if a.baseline and a.baseline.endswith(".jsonl"):   # perf_log.jsonl: latest llama.cpp rows
        for l in open(a.baseline):
            r = json.loads(l)
            if r.get("kind") != "llama":
                continue
            base.update(perf_metrics(r))
    elif a.baseline:
        base = json.load(open(a.baseline))
    ctxs = [int(x) for x in a.ctx.split(",")]

    gain = 1.0
    if a.draft_tune and os.path.exists(a.draft_tune):
        # the better of the two measured drafting gains: Loom's (draft_tune.json) and llama.cpp's own
        # (llamacpp_spec.json, same directory): either shows what drafting gets on this model and text
        for f in (a.draft_tune, os.path.join(os.path.dirname(a.draft_tune) or ".", "llamacpp_spec.json")):
            if os.path.exists(f):
                try:
                    t = json.load(open(f))
                except ValueError:   # empty or partial: that tuning step failed; no drafting gain from it
                    continue
                if t.get("kind", "none") != "none" and t.get("baseline_tps"):
                    gain = max(gain, t["decode_tps"] / t["baseline_tps"])
    rows = [pick_target(k, base.get(k), c, efficiency(m, k, c, compute=comp), efficiency(m, k, c, STRETCH, compute=comp))
            for k, c in ceilings(m, a.bw, a.flops, ctxs, gain).items() if k != "c4_steady_shared"]
    if m["geo"]["n_exp"]:   # the c4 ceiling assumes independent routing; streams of similar text share experts, so the hard
        # bound (all 4 tokens on the same experts) is what a measurement may not exceed (Qwen3-30B-A3B: 1.17 x ceiling)
        for r in rows:
            if r["metric"] == "c4_steady":
                r["bound"] = round(4 * a.bw / (pass_bytes(m, ctx=C4_CTX) + 3 * kv_read_bytes(m["geo"], C4_CTX)) * gain, 1)

    # the format-aware ceiling (next to the first-order one, unchanged): see format_pass()
    rates = format_rates(a.device, a.power, a.flops, a.bw)
    for r in rows:
        f = format_ceiling(m, r["metric"], rates, a.chunk) if gain == 1.0 else None
        if f is None:
            continue
        r["ceiling_format"] = round(f.pop("ceiling"), 1)
        r["format"] = f
        if r["metric"].startswith("prefill_"):
            fp = format_ceiling(m, r["metric"], rates, a.chunk, padded=True)
            r["ceiling_format_padded"] = round(fp.pop("ceiling"), 1)
            r["format_padded"] = fp

    out = dict(gguf=a.gguf, cfg=a.cfg and os.path.abspath(a.cfg), geometry={k: (sorted(v) if isinstance(v, set) else v)
                                      for k, v in geo.items()},
               weight_bytes_per_token=round(wb), head_bytes=head,
               bytes_by_kind={k: round(v) for k, v in detail.items()},
               matmul_params=round(params), efficiency_assumed=EFF, stretch_assumed=STRETCH,
               device=dev["name"] if a.bw == dev["bw"] and comp["mode"] != "override" else dev["name"] + ", overridden", bw=a.bw, flops=a.flops,
               power_mode=comp["label"], flops_spec=comp["spec_flops"], draft_gain=round(gain, 3), metrics=rows,
               format_rates=rates and dict(rates, chunk=a.chunk),
               campaign=dict(budget_h=a.budget_h, reserve_min=a.reserve_min,
                             goals={g.split("=")[0]: float(g.split("=")[1]) for g in a.goal}))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print(f"weights read / decode token: {wb/1e9:.3f} GB  (head {head/1e9:.3f} GB)  "
          + "  ".join(f"{k} {v/1e9:.2f}" for k, v in detail.items()))
    at = f" @ {comp['label']}" if comp["label"] != "spec" else " (spec)"
    print(f"matmul params: {params/1e9:.2f} B    BW {a.bw/1e9:.0f} GB/s   compute {a.flops/1e12:.1f} TFLOP/s{at}"
          + (f" (spec {comp['spec_flops']/1e12:.0f})" if comp["spec_flops"] != a.flops else "")
          + (f"   decode/c4 ceilings x{gain:.2f} (measured drafting gain, --draft-tune)" if gain > 1 else ""))
    if rates:
        print(f"format-aware: VALU {rates['valu']/1e12:.2f} T lane-op/s, exp {rates['trans']/1e12:.2f} T/s, prefill chunk {a.chunk}; "
              "+padding = loom-serve's power-of-two epochs (SDK, not hardware)")
    hdr = (f"{'metric':24} {'ceiling':>9} {'format':>9} {'+padding':>9} {'bound':>7} {'llama.cpp':>9} {'eff':>5} "
           f"{'TARGET':>9} {'margin':>8} {'stretch':>9} {'margin':>8}")
    print(hdr)
    for r in rows:
        b = r.get("baseline")
        fc, fpd = r.get("ceiling_format"), r.get("ceiling_format_padded")
        line = (f"{r['metric'] + (at if r['metric'].startswith('prefill') else ''):24} {r['ceiling']:9.1f} "
                f"{fc if fc is not None else '-':>9} {fpd if fpd is not None else '-':>9} "
                f"{r['format']['bound'] if 'format' in r else '-':>7} ")
        if b:
            line += (f"{b:9.1f} {r['baseline_eff']:5.2f} {r['target']:9.1f} {r['margin']:+8.0%} "
                     f"{r['stretch']:9.1f} {r['stretch_margin']:+8.0%}"
                     + ("  at-ceiling: target parity" if r["at_ceiling"] else ""))
        else:
            line += f"{'-':>9} {'-':>5} {r['target']:9.1f} {'':>8} {r['stretch']:9.1f}"
        print(line)

if __name__ == "__main__":
    main()
