#!/usr/bin/env python3
"""Roofline + target picker for a GGUF LLM on a Loom device.

usage: roofline.py MODEL.gguf [--cfg cfg.json] [--baseline perf.json] [--out roofline.json]
                   [--bw 216e9] [--flops 59e12] [--ctx 1000,2000,16000,32000]

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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loom-benchmark", "scripts"))
from bench_common import perf_metrics  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loom-port-llm", "starter"))
from formats import split_parts  # noqa: E402

# Demonstrated efficiencies on Strix Halo (fraction of the roofline actually reached by
# a tuned Loom port, end to end). Update these when a port beats them; cite the source.
EFF = {
    "decode": 0.90,      # dense. Bolt-E4B 61.1 / 68.1 (reference/calibration.md)
    "decode_moe": 0.78,  # expert gather. Flint 101 / 130
    "prefill": 0.33,     # end to end. Bolt-E4B with d256/d512 flash prefill: 2339 / 7094 (Flint 0.22-0.28)
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
    for part in split_parts(path)[1:]:   # a split GGUF: the tensors of every part
        tensors.update({t.name: (int(t.n_bytes), [int(x) for x in t.shape]) for t in gguf.GGUFReader(part).tensors})
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
    return dict(arch=arch, L=L, hidden=hidden, heads=heads, kvh=kvh, kvh_l=kvh_l, hd=hd, hd_swa=hd_swa,
                window=window, sliding=sliding, kv_layers=kv_layers, n_exp=n_exp, n_used=n_used,
                top_k=int(g("attention.indexer.top_k", 0) or 0), idx_heads=int(g("attention.indexer.head_count", 0) or 0),
                idx_dim=int(g("attention.indexer.key_length", 0) or 0), ratios=list(ratios) if len(ratios) == L else [],
                gather_rows=len(g("ple.head_vocab_sizes", []) or []))


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
        tot += 2 * geo["kvh_l"][i] * hd * 2 * n
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
        tot += 2 * 2 * geo["heads"] * hd * avg
    return tot


# Devices (--device): the memory bandwidth a streaming kernel reaches and the peak dense f16 WMMA rate
# (CUs x 512 flops/clock x boost clock on RDNA3/3.5/4). The CLI's --bw / --flops override them.
# EFF / STRETCH / LAYER_US above were calibrated on Strix Halo; on another device they are an out-of-sample estimate
# until a row in reference/calibration.md says otherwise.
DEVICES = {
    # LPDDR5X-8000, 256-bit (256 GB/s peak); 40 CUs x 512 x 2.9 GHz
    "strix-halo": dict(name='AMD Ryzen AI Max+ 395 "Strix Halo" (gfx1151)', bw=216e9, flops=59e12),
    # GDDR6 20 Gbps, 256-bit (640 GB/s peak), 32 GB; 64 CUs x 512 x 2.92 GHz = 95.7 TFLOP/s dense (191 with sparsity).
    # bw: 600 GB/s (94 % of peak). llama.cpp's own decode on the DevLab R9700S already streams 569 GB/s
    # (Qwen2.5-32B Q4_K_M, Vulkan b12211, 29.3 tok/s x 19.4 GB per token = 569 GB/s), so a streaming kernel reaches at least that.
    "radeon-r9700": dict(name="AMD Radeon AI PRO R9700 (gfx1201, 32 GB)", bw=600e9, flops=95.7e12),
    # The AMD DevLab Dispatch `radeon-r9700` runners: R9700S, the same Navi 48 / 640 GB/s memory at a 2.35 GHz
    # max engine clock (rocminfo): 64 x 512 x 2.35 GHz = 77.0 TFLOP/s dense f16
    "radeon-r9700s": dict(name="AMD Radeon AI PRO R9700S (gfx1201, 32 GB, 2.35 GHz)", bw=600e9, flops=77.0e12),
}
DEVICE_KEY = "strix-halo"
DEVICE, BW, FLOPS = DEVICES[DEVICE_KEY]["name"], DEVICES[DEVICE_KEY]["bw"], DEVICES[DEVICE_KEY]["flops"]


def shape_of(meta, tensors, cfg=None):
    """What the ceilings need from a GGUF's metadata and tensors (plus the port's cfg.json): geometry, the weight
    bytes one decode token reads (by kind), the matmul params and the head's params."""
    geo = geometry(meta, cfg or {})
    geo["attn"] = attn_layers(tensors, geo["L"])
    wb, params, head, detail = weight_bytes(tensors, geo)
    hb = tensors.get("output.weight", tensors.get("token_embd.weight"))[1]
    return dict(geo=geo, wb=wb, params=params, head=head, detail=detail, head_params=hb[0] * hb[1])


def model_shape(gguf_path, cfg=None):
    meta, tensors = load(gguf_path)
    return shape_of(meta, tensors, cfg)


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
    prefill_<P> for each size in ctxs, c4_steady (4 streams, KV at ~600 tokens). gain multiplies decode and c4 (a
    measured drafting gain, --draft-tune)."""
    out = {"decode_tps": bw / pass_bytes(m) * gain}
    for c in ctxs:
        out[f"decode_tps_{c // 1000}k"] = bw / pass_bytes(m, ctx=c) * gain
    for P in ctxs:   # 2 flops per matmul param per token, the head on the last token only, causal attention
        out[f"prefill_{P}"] = flops / (2 * (m["params"] - m["head_params"]) + attn_flops_per_token(m["geo"], P))
    out["c4_steady"] = 4 * bw / pass_bytes(m, tokens=4, ctx=600, streams=4) * gain
    if m["geo"]["n_exp"]:   # MoE: streams of similar text share experts; the hard bound (all 4 tokens on the same experts) is what a measurement may not exceed
        out["c4_steady_shared"] = 4 * bw / (pass_bytes(m, ctx=600) + 3 * kv_read_bytes(m["geo"], 600)) * gain
    return out


def drafted_ceiling(m, depth, acceptance, bw=BW, ctx=0, streams=1):
    """Decode ceiling with speculative decoding at `depth` drafted tokens and `acceptance` (accepted / proposed draft
    tokens, the spec gate's count): each verify pass checks depth+1 positions per stream and commits on average
    1 + acceptance*depth tokens per stream, for one read of the weights (MoE: the distinct experts of all those
    positions) and of each stream's KV. The drafter's own passes are not counted: an upper bound."""
    return streams * (1 + acceptance * depth) * bw / pass_bytes(m, tokens=streams * (depth + 1), ctx=ctx, streams=streams)


# The drafted rows by roofline.json metric name: (KV context read per decoded token, streams). card_lib.DRAFTED_ROWS
# names the same rows by card key.
DRAFTED_ROWS = {"decode_tps": (0, 1), "decode_tps_2k": (2000, 1), "decode_tps_32k": (32000, 1), "c4_steady": (600, 4)}


def drafted_row_ceiling(m, metric, depth, acceptance, bw=BW):
    """A drafted row's ceiling (roofline.json metric name) at `acceptance`, ideally the one its own timed requests
    measured (perf.py, c4_steady.py "acceptance"): drafted_ceiling() at the row's context and streams. None for a
    row speculative decoding does not apply to. The card (make_info.py roofline) and audit.py both use this."""
    if metric not in DRAFTED_ROWS or acceptance is None or not depth:
        return None
    ctx, streams = DRAFTED_ROWS[metric]
    return drafted_ceiling(m, depth, acceptance, bw, ctx=ctx, streams=streams)


def efficiency(m, metric, ceiling, table=None, tokens_per_pass=None):
    """The fraction of `ceiling` Loom ports have demonstrated for this kind of metric (table EFF, or STRETCH):
    prefill, c4 (dense / MoE) or decode (dense / MoE), decode capped by the fixed per-layer cost (LAYER_US).
    tokens_per_pass: a drafted_ceiling() (speculative decoding): EFF_DRAFTED, the per-layer cost paid once per pass."""
    table = table or (EFF_DRAFTED if tokens_per_pass else EFF)
    moe = bool(m["geo"]["n_exp"])
    if metric.startswith("prefill"):
        return table["prefill"]
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
             at_ceiling=h < 0.03)
    if h < 0.03:
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
    ap.add_argument("--flops", type=float, default=None, help="peak dense f16 flop/s (default: the --device preset)")
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
    ap.add_argument("--goal", action="append", default=[],
                    help="user target, METRIC=TOK_S (e.g. decode_tps=70); replaces target/stretch for it")
    a = ap.parse_args()
    dev = DEVICES[a.device]
    a.bw = a.bw or dev["bw"]
    a.flops = a.flops or dev["flops"]
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
    rows = [pick_target(k, base.get(k), c, efficiency(m, k, c), efficiency(m, k, c, STRETCH))
            for k, c in ceilings(m, a.bw, a.flops, ctxs, gain).items() if k != "c4_steady_shared"]
    if m["geo"]["n_exp"]:   # the c4 ceiling assumes independent routing; streams of similar text share experts, so the hard
        # bound (all 4 tokens on the same experts) is what a measurement may not exceed (Qwen3-30B-A3B: 1.17 x ceiling)
        for r in rows:
            if r["metric"] == "c4_steady":
                r["bound"] = round(4 * a.bw / (pass_bytes(m, ctx=600) + 3 * kv_read_bytes(m["geo"], 600)) * gain, 1)

    out = dict(gguf=a.gguf, cfg=a.cfg and os.path.abspath(a.cfg), geometry={k: (sorted(v) if isinstance(v, set) else v)
                                      for k, v in geo.items()},
               weight_bytes_per_token=round(wb), head_bytes=head,
               bytes_by_kind={k: round(v) for k, v in detail.items()},
               matmul_params=round(params), efficiency_assumed=EFF, stretch_assumed=STRETCH,
               device=dev["name"], bw=a.bw, flops=a.flops, draft_gain=round(gain, 3), metrics=rows,
               campaign=dict(budget_h=a.budget_h, reserve_min=a.reserve_min,
                             goals={g.split("=")[0]: float(g.split("=")[1]) for g in a.goal}))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print(f"weights read / decode token: {wb/1e9:.3f} GB  (head {head/1e9:.3f} GB)  "
          + "  ".join(f"{k} {v/1e9:.2f}" for k, v in detail.items()))
    print(f"matmul params: {params/1e9:.2f} B    BW {a.bw/1e9:.0f} GB/s   peak {a.flops/1e12:.0f} TFLOP/s"
          + (f"   decode/c4 ceilings x{gain:.2f} (measured drafting gain, --draft-tune)" if gain > 1 else ""))
    hdr = (f"{'metric':16} {'ceiling':>9} {'llama.cpp':>9} {'eff':>5} "
           f"{'TARGET':>9} {'margin':>8} {'stretch':>9} {'margin':>8}")
    print(hdr)
    for r in rows:
        b = r.get("baseline")
        line = f"{r['metric']:16} {r['ceiling']:9.1f} "
        if b:
            line += (f"{b:9.1f} {r['baseline_eff']:5.2f} {r['target']:9.1f} {r['margin']:+8.0%} "
                     f"{r['stretch']:9.1f} {r['stretch_margin']:+8.0%}"
                     + ("  at-ceiling: target parity" if r["at_ceiling"] else ""))
        else:
            line += f"{'-':>9} {'-':>5} {r['target']:9.1f} {'':>8} {r['stretch']:9.1f}"
        print(line)

if __name__ == "__main__":
    main()
