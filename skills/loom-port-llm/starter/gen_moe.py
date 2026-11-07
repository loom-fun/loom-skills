"""MoE FFN kernels, wired by gen_llm.py from cfg["moe"] (inspect_gguf.py): a router (gating 1: softmax, Qwen3 /
Qwen3.5 MoE; gating 2: sigmoid with an optional selection bias, flint / LFM2-MoE), top-k weights (normalized or not),
per-assignment expert matvecs (assignment a = token * KU + slot), weighted combine, and the prefill path (tokens
bucketed by expert into 64-row tiles, grouped WMMA GEMMs). Shared experts are the starter's dense FFN (gen_llm.py);
a Qwen-style shared-expert gate (sigmoid(h . w) per token) scales them inside the combine.

g_moe_route        h[t] -> ids[t, KU] (i32), wts[t, KU] (f32)   (g_moe_route_tr: the same for prefill epochs)
g_moe_gu_<fmt>     act[a, FE] = silu(Wg[e] h[t]) * (Wu[e] h[t])     (e = ids[a], t = a / KU)
g_moe_dn_<fmt>     ys[a, H]  = Wd[e] act[a]
g_moe_shared_gate  sg[t]     = sigmoid(h[t] . w)                    (Qwen shared-expert gate; optional)
g_moe_combine      y[t, H]   = sg[t] * y[t, :] + sum_k wts[t, k] ys[t*KU + k, :]   (y: the shared experts' output)
"""
import os
import re
FMT = {"q4_k": 144, "q5_k": 176, "q6_k": 210, "q8_0": 34}
ROUTE_TR = int(__import__("os").environ.get("ROUTE_TR", "0"))   # router tokens per workgroup (0: largest of 1/2/4/8 whose LDS fits 48 KiB)
SB = {"q4_k": 256, "q5_k": 256, "q6_k": 256, "q8_0": 32}
GATE_NAME = {1: "softmax(logits)", 2: "sigmoid(logit)"}   # llama.cpp's expert_gating_func values


def row_bytes(fmt, k):
    return k // SB[fmt] * FMT[fmt]


def scaled(v, ind, scale):
    """llama.cpp's expert_weights_scale after the normalization (laguna: 2.5): %{v}s = %{v} * scale (empty at 1)."""
    return "" if scale == 1.0 else f"\n{ind}%{v}_k = scalar.constant {float(scale)!r} : f32\n{ind}%{v}s = scalar.mulf %{v}, %{v}_k : f32"


def route_decode(H, E, KU, gating, norm, bias, wty="f32", scale=1.0):
    """Decode-shaped epochs (t < 64): one workgroup per token. Router logits (f32 or bf16 weights) into LDS, then a
    parallel top-k: one thread per expert, KU rounds of workgroup max/min-index reductions (flint_router_topk pattern).
    Each lane reads 32-value chunks: few, wide loads in flight, which suits one latency-bound workgroup."""
    HG = H // 32
    assert E % 8 == 0 and E <= 256 and KU <= 8
    EPW = E // 8
    load = (f"""      %wraw = vector.load %w_view[%e, %bg, %c0] : view<{E}x{HG}x32xbf16> -> vector<32xbf16>
      %wv = vector.extf %wraw : vector<32xbf16> to vector<32xf32>""" if wty == "bf16" else
            f"""      %wv = vector.load %w_view[%e, %bg, %c0] : view<{E}x{HG}x32xf32> -> vector<32xf32>""")
    assert gating in (1, 2) and not (gating == 1 and bias), f"expert gating {gating} (bias {bias}): 1 softmax, 2 sigmoid"
    prob = "scalar.logisticf<afn> %lgt : f32" if gating == 2 else "scalar.addf %lgt, %zero_f : f32"
    score = "scalar.addf %prob, %bias_v : f32" if bias else "scalar.addf %prob, %zero_f : f32"
    bias_load = f"""    %b_view = buffer.view %b[%zero] : buffer -> view<{E}xf32>
    %bias_v = view.load %b_view[%be] : view<{E}xf32> -> f32""" if bias else ""
    norm_text = f"""  %min_sum = scalar.constant 6.103515625e-05 : f32
  %wsum = scalar.maxnumf %total, %min_sum : f32""" if norm else "  %wsum = scalar.constant 1.0 : f32"
    # softmax: thread e holds logit e (out of range: -3e38); probabilities over all E experts (workgroup max and
    # sum), and the top-k picks on them, as llama.cpp does
    pn, sn = ("%prob", "%score0") if gating == 2 else ("%prob_r", "%raw")
    softmax = "" if gating == 2 else """  %lmax = kernel.workgroup.reduce<maxnumf> %raw : f32
  %lsh = scalar.subf %raw, %lmax : f32
  %lex = scalar.expf<afn> %lsh : f32
  %lex_in = scf.select %in_range, %lex, %zero_f : f32
  %lden = kernel.workgroup.reduce<addf> %lex_in : f32
  %prob = scalar.divf %lex_in, %lden : f32
  %score0 = scf.select %in_range, %prob, %removed : f32
"""
    return f"""
// Router: wave w scores experts w*{EPW}..; then thread e holds expert e and {KU} rounds of
// workgroup reductions pick the top {KU} (ties: lowest index) of {GATE_NAME[gating]}{' + bias' if bias else ''}.
kernel.def export("g_moe_route") @g_moe_route(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %x: buffer, %w: buffer, %b: buffer, %ids: buffer, %wts: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, 7)] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  %cHG = index.constant {HG} : index
  %cE = index.constant {E} : index
  %zero_f = scalar.constant 0.0 : f32
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
  %w_view = buffer.view %w[%zero] : buffer -> view<{E}x{HG}x32x{wty}>
  %lg_bytes = index.constant {E * 4} : offset
  %lg = buffer.alloca<workgroup> align(16) %lg_bytes : buffer
  %lg_view = buffer.view %lg[%zero] : buffer -> view<{E}xf32>
  %first = index.cmp eq, %lane, %c0 : index
  %cEPW = index.constant {EPW} : index
  %e0 = index.mul %wave, %cEPW : index
  scf.for %j = [%c0 to %cEPW step %c1] {{
    %e_raw = index.add %e0, %j : index
    %e = index.assume %e_raw [range(%e_raw, 0, {E - 1})] : index
    %part = scf.for %g = [%lane to %cHG step %c32](%acc = %zero_f : f32) -> (f32) {{
      %bg = index.assume %g [range(%g, 0, {HG - 1})] : index
      %xv = vector.load %x_view[%token, %bg, %c0] : view<[%tokens]x{HG}x32xf32> -> vector<32xf32>
{load}
      %d = vector.dotf<reassoc|contract> %wv, %xv, %zero_f : vector<32xf32>, vector<32xf32>, f32
      %nx = scalar.addf %acc, %d : f32
      scf.yield %nx : f32
    }}
    %logit = kernel.subgroup.reduce<addf> %part : f32
    scf.if %first {{
      view.store %logit, %lg_view[%e] : f32, view<{E}xf32>
    }}
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %t0 = kernel.workitem.id<x> : index
  %tb = index.assume %t0 [range(%t0, 0, 255)] : index
  %in_range = index.cmp ult, %tb, %cE : index
  %removed = scalar.constant -3.0e38 : f32
  {pn}, {sn} = scf.if %in_range -> (f32, f32) {{
    %be = index.assume %tb [range(%tb, 0, {E - 1})] : index
    %lgt = view.load %lg_view[%be] : view<{E}xf32> -> f32
    %prob_v = {prob.replace('%lgt', '%lgt')}
{bias_load.replace('%prob', '%prob_v')}
    %score_v = {score.replace('%prob', '%prob_v')}
    scf.yield %prob_v, %score_v : f32, f32
  }} else {{
    scf.yield %zero_f, %removed : f32, f32
  }}
{softmax}  %ids_view = buffer.view %ids[%zero] : buffer -> view<[%tokens]x{KU}xi32>
  %wts_view = buffer.view %wts[%zero] : buffer -> view<[%tokens]x{KU}xf32>
  %t32 = index.cast %tb : index to i32
  %big_id = scalar.constant 1000000 : i32
  %cKU = index.constant {KU} : index
  %minus1 = scalar.constant -1 : i32
  %rest, %total, %my_w, %my_k = scf.for %round = [%c0 to %cKU step %c1](%sc = %score0 : f32, %sum = %zero_f : f32, %mw = %zero_f : f32, %mk = %minus1 : i32) -> (f32, f32, f32, i32) {{
    %best = kernel.workgroup.reduce<maxnumf> %sc : f32
    %is_best = scalar.cmpf oeq, %sc, %best : f32
    %cand = scf.select %is_best, %t32, %big_id : i32
    %winner = kernel.workgroup.reduce<minsi> %cand : i32
    %won = scalar.cmpi eq, %winner, %t32 : i32
    %br = index.assume %round [range(%round, 0, {KU - 1})] : index
    scf.if %won {{
      view.store %t32, %ids_view[%token, %br] : i32, view<[%tokens]x{KU}xi32>
    }}
    %mine = scf.select %won, %prob, %zero_f : f32
    %wp = kernel.workgroup.reduce<addf> %mine : f32
    %round32 = index.cast %br : index to i32
    %next_mw = scf.select %won, %prob, %mw : f32
    %next_mk = scf.select %won, %round32, %mk : i32
    %next_sc = scf.select %won, %removed, %sc : f32
    %next_sum = scalar.addf %sum, %wp : f32
    scf.yield %next_sc, %next_sum, %next_mw, %next_mk : f32, f32, f32, i32
  }}
{norm_text}
  // The winner thread of each round keeps its prob in registers and writes the normalized weight
  // itself: reading back other threads' global stores raced through the per-WGP cache when
  // neighboring tokens' workgroups share a cache line (prefill outputs varied run to run).
  %c0_i32r = scalar.constant 0 : i32
  %writer = scalar.cmpi sge, %my_k, %c0_i32r : i32
  scf.if %writer {{
    %kb = scalar.assume %my_k [range(%my_k, 0, {KU - 1})] : i32
    %bk = index.cast %kb : i32 to index
    %wn = scalar.divf %my_w, %wsum : f32{scaled("wn", "    ", scale)}
    view.store %wn{"s" if scale != 1.0 else ""}, %wts_view[%token, %bk] : f32, view<[%tokens]x{KU}xf32>
  }}
  kernel.return
}}
"""


def route_prefill(H, E, KU, gating, norm, bias, wty="f32", scale=1.0):
    """g_moe_route_tr, for prefill epochs (t >= 64; decode keeps g_moe_route: one token per workgroup, wide loads in
    flight, was 34 % faster per decode step). Router logits (f32 or bf16 weights) into LDS, then the top-k per token.
    Logits: TR tokens per workgroup, their h rows staged in LDS; lane l reads the 4-value chunks l, l+32, ... of each
    128-value block, so a wave's weight load is 512 contiguous bytes and its LDS reads are conflict-free (the earlier
    32-value chunk per lane cost 4.4 ms per 2048-token layer on Flint, now 0.96 ms).
    Top-k: wave r takes token r, lane l holds experts l, l+32, ...: KU rounds of subgroup max / min-index reductions
    (ties: lowest index), no workgroup barriers. The winners' weights stay in registers (see README: a global
    write/read-back of them raced)."""
    HG = H // 32
    assert E % 8 == 0 and E <= 256 and KU <= 8 and H % 128 == 0
    assert gating in (1, 2) and not (gating == 1 and bias), f"expert gating {gating} (bias {bias}): 1 softmax, 2 sigmoid"
    EPW = E // 8
    TR = ROUTE_TR or max(t for t in (1, 2, 4, 8) if t * (H + E) * 4 <= 48 * 1024)
    prob = "scalar.logisticf<afn> %lgt : f32" if gating == 2 else "scalar.addf %lgt, %zero_f : f32"
    score = "scalar.addf %prob_v, %bias_v : f32" if bias else "scalar.addf %prob_v, %zero_f : f32"
    bias_load = f"""    %b_view = buffer.view %b[%zero] : buffer -> view<{E}xf32>
    %bias_v = view.load %b_view[%be] : view<{E}xf32> -> f32
""" if bias else ""
    norm_text = f"""    %min_sum = scalar.constant 6.103515625e-05 : f32
    %wsum = scalar.maxnumf %total, %min_sum : f32""" if norm else "    %wsum = scalar.constant 1.0 : f32"
    acc_in = ", ".join(f"%a{r} = %zero_f : f32" for r in range(TR))
    acc_ty = ", ".join(["f32"] * TR)
    acc_out = ", ".join(f"%p{r}" for r in range(TR))
    HB = H // 128
    wl = (f"""      %wraw = vector.load %w4_view[%e, %bb, %lane, %c0] : view<{E}x{HB}x32x4xbf16> -> vector<4xbf16>
      %wv = vector.extf %wraw : vector<4xbf16> to vector<4xf32>""" if wty == "bf16" else
          f"""      %wv = vector.load %w4_view[%e, %bb, %lane, %c0] : view<{E}x{HB}x32x4xf32> -> vector<4xf32>""")
    dots = "".join(f"""      %xv{r} = vector.load %xs4_view[{r}, %bb, %lane, %c0] : view<{TR}x{HB}x32x4xf32> -> vector<4xf32>
      %d{r} = vector.dotf<reassoc|contract> %wv, %xv{r}, %zero_f : vector<4xf32>, vector<4xf32>, f32
      %n{r} = scalar.addf %a{r}, %d{r} : f32
""" for r in range(TR))
    nxt = ", ".join(f"%n{r}" for r in range(TR))
    stores = "".join(f"""    %logit{r} = kernel.subgroup.reduce<addf> %p{r} : f32
    scf.if %first {{
      view.store %logit{r}, %lg_view[{r}, %e] : f32, view<{TR}x{E}xf32>
    }}
""" for r in range(TR))

    # Phase 2: wave r (< TR) takes token r; lane l holds experts l, l+32, ...: KU rounds of subgroup
    # reductions (max score, then the lowest expert id at that score) instead of workgroup reductions.
    EPL = (E + 31) // 32
    T2 = [f"""  %c0_i32r = scalar.constant 0 : i32
  %removed = scalar.constant -3.0e38 : f32
  %big_id = scalar.constant 1000000 : i32
  %ids_view = buffer.view %ids[%zero] : buffer -> view<[%tokens]x{KU}xi32>
  %wts_view = buffer.view %wts[%zero] : buffer -> view<[%tokens]x{KU}xf32>
  %wave_tok = index.cmp ult, %wave, %cTR : index
  scf.if %wave_tok {{
    %r = index.assume %wave [range(%wave, 0, {TR - 1})] : index
    %tok_value = index.add %token0, %r : index
    %tok_live = index.cmp ult, %tok_value, %tokens : index
    %tok_c = scf.select %tok_live, %tok_value, %c0 : index
    %token = index.assume %tok_c [lt(%tok_c, %tokens)] : index
    %store_lane = scalar.andi %first, %tok_live : i1
"""]
    load_lines = lambda i: f"""    %ce{i} = index.constant {32 * i} : index
    %e{i}_v = index.add %lane, %ce{i} : index
    %e{i}_in = index.cmp ult, %e{i}_v, %cE : index
    %e{i}_c = scf.select %e{i}_in, %e{i}_v, %c0 : index
    %be{i} = index.assume %e{i}_c [range(%e{i}_c, 0, {E - 1})] : index
    %id{i} = index.cast %e{i}_v : index to i32
    %lgt{i} = view.load %lg_view[%r, %be{i}] : view<{TR}x{E}xf32> -> f32
"""
    if gating == 1:   # softmax over the token's E logits (subgroup max and sum; lane l holds experts l, l+32, ...)
        for i in range(EPL):
            T2.append(load_lines(i) + f"    %lm{i} = scf.select %e{i}_in, %lgt{i}, %removed : f32\n"
                      + (f"    %lmx{i} = scalar.maxnumf %lmx{i - 1}, %lm{i} : f32\n" if i else "    %lmx0 = scalar.addf %lm0, %zero_f : f32\n"))
        T2.append(f"    %smax = kernel.subgroup.reduce<maxnumf> %lmx{EPL - 1} : f32\n")
        for i in range(EPL):
            T2.append(f"""    %lsh{i} = scalar.subf %lm{i}, %smax : f32
    %lex{i} = scalar.expf<afn> %lsh{i} : f32
    %lexm{i} = scf.select %e{i}_in, %lex{i}, %zero_f : f32
""" + (f"    %lsum{i} = scalar.addf %lsum{i - 1}, %lexm{i} : f32\n" if i else "    %lsum0 = scalar.addf %lexm0, %zero_f : f32\n"))
        T2.append(f"    %sden = kernel.subgroup.reduce<addf> %lsum{EPL - 1} : f32\n")
        for i in range(EPL):
            T2.append(f"""    %pr{i} = scalar.divf %lexm{i}, %sden : f32
    %s{i}_0 = scf.select %e{i}_in, %pr{i}, %removed : f32
""")
    for i in range(EPL if gating == 2 else 0):
        T2.append(load_lines(i) + f"""    %pr{i} = {prob.replace('%lgt', f'%lgt{i}')}
""" + (f"""    %b_view{i} = buffer.view %b[%zero] : buffer -> view<{E}xf32>
    %bias{i} = view.load %b_view{i}[%be{i}] : view<{E}xf32> -> f32
    %sv{i} = scalar.addf %pr{i}, %bias{i} : f32
""" if bias else f"""    %sv{i} = scalar.addf %pr{i}, %zero_f : f32
""") + f"""    %s{i}_0 = scf.select %e{i}_in, %sv{i}, %removed : f32
""")
    T2.append("    %sum_0 = scalar.addf %zero_f, %zero_f : f32\n")
    for k in range(KU):
        T2.append(f"    %lb{k}_0 = scalar.addf %s0_{k}, %zero_f : f32\n    %li{k}_0 = scalar.addi %id0, %c0_i32r : i32\n")
        for i in range(1, EPL):
            T2.append(f"""    %gt{k}_{i} = scalar.cmpf ogt, %s{i}_{k}, %lb{k}_{i - 1} : f32
    %lb{k}_{i} = scf.select %gt{k}_{i}, %s{i}_{k}, %lb{k}_{i - 1} : f32
    %li{k}_{i} = scf.select %gt{k}_{i}, %id{i}, %li{k}_{i - 1} : i32
""")
        L = EPL - 1
        T2.append(f"""    %best{k} = kernel.subgroup.reduce<maxnumf> %lb{k}_{L} : f32
    %isb{k} = scalar.cmpf oeq, %lb{k}_{L}, %best{k} : f32
    %cand{k} = scf.select %isb{k}, %li{k}_{L}, %big_id : i32
    %win{k} = kernel.subgroup.reduce<minsi> %cand{k} : i32
    %m{k}_0 = scalar.addf %zero_f, %zero_f : f32
""")
        for i in range(EPL):
            T2.append(f"""    %w{k}_{i} = scalar.cmpi eq, %id{i}, %win{k} : i32
    %m{k}_{i + 1} = scf.select %w{k}_{i}, %pr{i}, %m{k}_{i} : f32
    %s{i}_{k + 1} = scf.select %w{k}_{i}, %removed, %s{i}_{k} : f32
""")
        T2.append(f"""    %wp{k} = kernel.subgroup.reduce<addf> %m{k}_{EPL} : f32
    %sum_{k + 1} = scalar.addf %sum_{k}, %wp{k} : f32
    scf.if %store_lane {{
      view.store %win{k}, %ids_view[%token, {k}] : i32, view<[%tokens]x{KU}xi32>
    }}
""")
    T2.append(f"    %total = scalar.addf %sum_{KU}, %zero_f : f32\n" + norm_text.replace("    %", "    %") + "\n")
    T2.append(f"""    scf.if %store_lane {{
""" + "".join(f"""      %wn{k} = scalar.divf %wp{k}, %wsum : f32{scaled(f"wn{k}", "      ", scale)}
      view.store %wn{k}{"s" if scale != 1.0 else ""}, %wts_view[%token, {k}] : f32, view<[%tokens]x{KU}xf32>
""" for k in range(KU)) + """    }
  }
""")
    topk = "".join(T2)
    return f"""
// Router: TR={TR} tokens per workgroup, their h rows staged in LDS; wave w scores experts w*{EPW}.. for
// all {TR} tokens (one weight load serves {TR} tokens); then per token, thread e holds expert e and {KU}
// rounds of workgroup reductions pick the top {KU} (ties: lowest index) of {GATE_NAME[gating]}{' + bias' if bias else ''}.
kernel.def export("g_moe_route_tr") @g_moe_route_tr(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cTRm1 = index.constant {TR - 1} : index
  %cTR = index.constant {TR} : index
  %padded = index.add %tokens, %cTRm1 : index
  %groups = index.div %padded, %cTR : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %x: buffer, %w: buffer, %b: buffer, %ids: buffer, %wts: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %group_id = kernel.workgroup.id<x> : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, 7)] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  %c256 = index.constant 256 : index
  %cHG = index.constant {HG} : index
  %cE = index.constant {E} : index
  %cTR = index.constant {TR} : index
  %zero_f = scalar.constant 0.0 : f32
  %token0 = index.mul %group_id, %cTR : index
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
  %xs_bytes = index.constant {TR * H * 4} : offset
  %xs = buffer.alloca<workgroup> align(16) %xs_bytes : buffer
  %xs_view = buffer.view %xs[%zero] : buffer -> view<{TR}x{HG}x32xf32>
  %xs4_view = buffer.view %xs[%zero] : buffer -> view<{TR}x{HB}x32x4xf32>
  %w4_view = buffer.view %w[%zero] : buffer -> view<{E}x{HB}x32x4x{wty}>
  %cHB = index.constant {HB} : index
  %lg_bytes = index.constant {TR * E * 4} : offset
  %lg = buffer.alloca<workgroup> align(16) %lg_bytes : buffer
  %lg_view = buffer.view %lg[%zero] : buffer -> view<{TR}x{E}xf32>
  %tid0 = kernel.workitem.id<x> : index
  %tid = index.assume %tid0 [range(%tid0, 0, 255)] : index
  %cvec = index.constant {TR * HG} : index
  %zero32 = vector.constant 0.0 : vector<32xf32>
  scf.for %v = [%tid to %cvec step %c256] {{
    %r_value = index.div %v, %cHG : index
    %r = index.assume %r_value [range(%r_value, 0, {TR - 1})] : index
    %g_value = index.rem %v, %cHG : index
    %g = index.assume %g_value [range(%g_value, 0, {HG - 1})] : index
    %tok_value = index.add %token0, %r : index
    %tok_in = index.cmp ult, %tok_value, %tokens : index
    %tok_c = scf.select %tok_in, %tok_value, %c0 : index
    %tok = index.assume %tok_c [lt(%tok_c, %tokens)] : index
    %xg = vector.load %x_view[%tok, %g, %c0] : view<[%tokens]x{HG}x32xf32> -> vector<32xf32>
    %xsel = scf.select %tok_in, %xg, %zero32 : vector<32xf32>
    vector.store %xsel, %xs_view[%r, %g, %c0] : vector<32xf32>, view<{TR}x{HG}x32xf32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %first = index.cmp eq, %lane, %c0 : index
  %cEPW = index.constant {EPW} : index
  %e0 = index.mul %wave, %cEPW : index
  scf.for %j = [%c0 to %cEPW step %c1] {{
    %e_raw = index.add %e0, %j : index
    %e = index.assume %e_raw [range(%e_raw, 0, {E - 1})] : index
    {acc_out} = scf.for %bq = [%c0 to %cHB step %c1]({acc_in}) -> ({acc_ty}) {{
      %bb = index.assume %bq [range(%bq, 0, {HB - 1})] : index
{wl}
{dots}      scf.yield {nxt} : {acc_ty}
    }}
{stores}  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
{topk}  kernel.return
}}
"""


def route_prefill_live(H, E, KU, gating, norm, bias, wty="f32", scale=1.0):
    """g_moe_route_tr_live: g_moe_route_tr given the epoch's %info; a workgroup whose tokens are all padding (info
    word 2 = 0; a 1026-token prompt at chunk 4096 runs as a 2048-row epoch) does nothing. g_moe_bucket takes live
    tokens only, so their ids / weights are never read."""
    text = route_prefill(H, E, KU, gating, norm, bias, wty, scale)
    text = text[text.index('kernel.def export("g_moe_route_tr")'):]
    TR = int(re.search(r"%cTR = index.constant (\d+) : index", text).group(1))
    def rep(a, b):
        nonlocal text
        assert text.count(a) == 1, a
        text = text.replace(a, b)
    rep('export("g_moe_route_tr") @g_moe_route_tr(', 'export("g_moe_route_tr_live") @g_moe_route_tr_live(')
    rep("%ids: buffer, %wts: buffer) {", "%ids: buffer, %wts: buffer, %info: buffer) {")
    live = ["  %gl_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>\n  %gl_dead = scalar.constant 0 : i32\n"]
    for r in range(TR):
        live.append(f"""  %gl_c{r} = index.constant {r} : index
  %gl_t{r}v = index.add %token0, %gl_c{r} : index
  %gl_in{r} = index.cmp ult, %gl_t{r}v, %tokens : index
  %gl_t{r}c = scf.select %gl_in{r}, %gl_t{r}v, %c0 : index
  %gl_t{r} = index.assume %gl_t{r}c [lt(%gl_t{r}c, %tokens)] : index
  %gl_w{r} = view.load %gl_view[%gl_t{r}, 2] : view<[%tokens]x4xi32> -> i32
  %gl_l{r} = scalar.cmpi ne, %gl_w{r}, %gl_dead : i32
  %gl_a{r} = scalar.andi %gl_in{r}, %gl_l{r} : i1
""" + (f"  %gl_any{r} = scalar.ori %gl_any{r - 1}, %gl_a{r} : i1\n" if r else "  %gl_any0 = scalar.andi %gl_a0, %gl_a0 : i1\n"))
    live.append(f"  scf.if %gl_any{TR - 1} {{\n")
    rep("  %token0 = index.mul %group_id, %cTR : index\n", "  %token0 = index.mul %group_id, %cTR : index\n" + "".join(live))
    rep("  kernel.return\n}\n", "  }\n  kernel.return\n}\n")
    return "\n" + text


def row_scale_tiles(row_scale):
    """g_row_scale_f16_tiles: the starter's g_row_scale_f16 over the bucketed expert rows (max_tiles x 64), skipping
    the rows of unused tiles (tile_info expert < 0; g_moe_gdn skips those tiles). max_tiles is sized for the worst
    case (t*KU/64 + E tiles), so most of them are unused."""
    text = row_scale[row_scale.index('kernel.def export("g_row_scale_f16")'):]
    def rep(a, b):
        nonlocal text
        assert text.count(a) == 1, a
        text = text.replace(a, b)
    rep('export("g_row_scale_f16") @g_row_scale_f16(', 'export("g_row_scale_f16_tiles") @g_row_scale_f16_tiles(')
    rep("%scales: buffer, %x16: buffer) {", "%scales: buffer, %x16: buffer, %tile_info: buffer) {")
    rep("  %row = index.assume %row_id [lt(%row_id, %rows)] : index\n", f"""  %row = index.assume %row_id [lt(%row_id, %rows)] : index
  %rt_zero = index.constant 0 : offset
  %rt_c64 = index.constant {TILE} : index
  %rt_tiles = index.div %rows, %rt_c64 : index
  %rt_tile_v = index.div %row, %rt_c64 : index
  %rt_tile = index.assume %rt_tile_v [lt(%rt_tile_v, %rt_tiles)] : index
  %rt_view = buffer.view %tile_info[%rt_zero] : buffer -> view<[%rt_tiles]x2xi32>
  %rt_expert = view.load %rt_view[%rt_tile, 0] : view<[%rt_tiles]x2xi32> -> i32
  %rt_none = scalar.constant 0 : i32
  %rt_used = scalar.cmpi sge, %rt_expert, %rt_none : i32
  scf.if %rt_used {{
""")
    rep("  kernel.return\n}\n", "  }\n  kernel.return\n}\n")
    return "\n" + text


def route(H, E, KU, gating, norm, bias, wty="f32", scale=1.0):
    """Both routers: g_moe_route (decode epochs) and g_moe_route_tr (prefill epochs); same outputs."""
    return route_decode(H, E, KU, gating, norm, bias, wty, scale) + route_prefill(H, E, KU, gating, norm, bias, wty, scale)


def expert_mv(fmt, K, N, E, KU, R, mode):
    """mode 'gu': act[a] = silu(Wg[e] x[a/KU]) * Wu[e] x[a/KU]; mode 'dn': ys[a] = Wd[e] x[a]."""
    groups = K // 32
    lanes = (groups + 31) // 32 * 32
    waves = lanes // 32
    rb = row_bytes(fmt, K)
    eb = rb * N
    nw = 2 if mode == "gu" else 1
    name = f"g_moe_{mode}_{fmt}_k{K}_r{R}"
    wargs = "%w: buffer, %w2: buffer" if nw == 2 else "%w: buffer"
    xrows = "%xrows" if mode == "gu" else "%tokens"
    L = [f"""
kernel.def export("{name}") @{name}(%n: index, %tokens: index) {{
  %c1 = index.constant 1 : index
  %lanes = index.constant {lanes} : index
  %rows_per = index.constant {R} : index
  %row_groups = index.div %n, %rows_per : index
  kernel.launch.config workgroups(%tokens, %row_groups, %c1) workgroup_size(%lanes, %c1, %c1) : index
}} launch(%n_arg: index, %tokens_arg: index, %ids: buffer, %x: buffer, {wargs}, %y: buffer) {{
  %n = index.assume %n_arg [range(%n_arg, {R}, 262144), multiple_of(%n_arg, {R})] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 65536)] : index
  %group_id = kernel.workgroup.id<y> : index
  %a_id = kernel.workgroup.id<x> : index
  %a = index.assume %a_id [lt(%a_id, %tokens)] : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, {lanes - 1})] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, {waves - 1})] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %cKU = index.constant {KU} : index
  %rows_per = index.constant {R} : index
  %cW = index.constant {waves} : index
  %row_bytes = index.constant {rb} : offset
  %expert_bytes = index.constant {eb} : offset
  %block_bytes = index.constant {FMT[fmt]} : offset
  %zero_f = scalar.constant 0.0 : f32
  %zero32 = vector.constant 0.0 : vector<32xf32>
  %c0_i32 = scalar.constant 0 : i32
  %cE1_i32 = scalar.constant {E - 1} : i32
  %ids_view = buffer.view %ids[%zero] : buffer -> view<[%tokens]xi32>
  %eid = view.load %ids_view[%a] : view<[%tokens]xi32> -> i32
  %elo = scalar.maxsi %eid, %c0_i32 : i32
  %ehi = scalar.minsi %elo, %cE1_i32 : i32
  %eb32 = scalar.assume %ehi [range(%ehi, 0, {E - 1})] : i32
  %e = index.cast %eb32 : i32 to index
  %e_off = index.scale %e, %expert_bytes : index, offset -> offset"""]
    if mode == "gu":
        L.append(f"""  %xrows = index.div %tokens, %cKU : index
  %xr_v = index.div %a, %cKU : index
  %xr = index.assume %xr_v [lt(%xr_v, %xrows)] : index""")
    else:
        L.append("  %xr = index.assume %a [lt(%a, %tokens)] : index")
    L.append(f"""  %x_view = buffer.view %x[%zero] : buffer -> view<[{xrows}]x{groups}x32xf32>
  %row_base = index.mul %group_id, %rows_per : index
  %ok = index.cmp ult, %workitem, %groups_c : index""".replace("%groups_c", f"%cgroups"))
    L.insert(-1, f"  %cgroups = index.constant {groups} : index")
    L.append(f"""  %g_last = index.constant {groups - 1} : index
  %g_c = index.min %workitem, %g_last : index
  %g = index.assume %g_c [range(%g_c, 0, {groups - 1})] : index
  %cgpb = index.constant {SB[fmt] // 32} : index
  %blk = index.div %g, %cgpb : index
  %within = index.rem %g, %cgpb : index
  %ba = index.scale %blk, %block_bytes : index, offset -> offset
  %xg = scf.if %ok -> (vector<32xf32>) {{
    %v = vector.load %x_view[%xr, %g, %c0] : view<[{xrows}]x{groups}x32xf32> -> vector<32xf32>
    scf.yield %v : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}
  %xs = vector.reduce<addf> %xg, %zero_f : vector<32xf32>, f32
  %part_bytes = index.constant {waves * R * nw * 4} : offset
  %parts = buffer.alloca<workgroup> align(16) %part_bytes : buffer
  %pv = buffer.view %parts[%zero] : buffer -> view<{nw}x{waves}x{R}xf32>
  %first = index.cmp eq, %lane, %c0 : index
  scf.for %r = [%c0 to %rows_per step %c1] {{
    %row_value = index.add %row_base, %r : index
    %row = index.assume %row_value [lt(%row_value, %n)] : index
    %ro = index.scale %row, %row_bytes : index, offset -> offset
    %rb = index.assume %r [range(%r, 0, {R - 1})] : index
    %o0 = index.add %ro, %ba : offset
    %o = index.add %o0, %e_off : offset""")
    import os as _os
    fold = _os.environ.get("MOE_FOLD", "1") != "0"
    for wi, w in enumerate(["w", "w2"][:nw]):
        if fold and fmt == "q4_k":
            L.append(f"""    %wv{wi}, %sc{wi}, %mn{wi} = scf.if %ok -> (vector<32xf32>, f32, f32) {{
      %v, %sa, %sb = func.call @g_q4_k_codes(%{w}, %o, %within) : (buffer, offset, index) -> (vector<32xf32>, f32, f32)
      scf.yield %v, %sa, %sb : vector<32xf32>, f32, f32
    }} else {{
      scf.yield %zero32, %zero_f, %zero_f : vector<32xf32>, f32, f32
    }}
    %qd{wi} = vector.dotf<reassoc|contract> %wv{wi}, %xg, %zero_f : vector<32xf32>, vector<32xf32>, f32
    %qm{wi} = scalar.mulf %mn{wi}, %xs : f32
    %qs{wi} = scalar.mulf %qd{wi}, %sc{wi} : f32
    %d{wi} = scalar.subf %qs{wi}, %qm{wi} : f32""")
        elif fold:
            L.append(f"""    %wv{wi} = scf.if %ok -> (vector<32xf32>) {{
      %v = func.call @g_{fmt}_values(%{w}, %o, %within) : (buffer, offset, index) -> (vector<32xf32>)
      scf.yield %v : vector<32xf32>
    }} else {{
      scf.yield %zero32 : vector<32xf32>
    }}
    %d{wi} = vector.dotf<reassoc|contract> %wv{wi}, %xg, %zero_f : vector<32xf32>, vector<32xf32>, f32""")
        else:
          L.append(f"""    %d{wi} = scf.if %ok -> (f32) {{
      %dv = func.call @g_{fmt}_dot(%{w}, %o, %within, %xg) : (buffer, offset, index, vector<32xf32>) -> (f32)
      scf.yield %dv : f32
    }} else {{
      scf.yield %zero_f : f32
    }}""")
        L.append(f"""    %s{wi} = kernel.subgroup.reduce<addf> %d{wi} : f32
    scf.if %first {{
      view.store %s{wi}, %pv[{wi}, %wave, %rb] : f32, view<{nw}x{waves}x{R}xf32>
    }}""")
    L.append(f"""  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %outputs = index.constant {R} : index
  %lanes_step = index.constant {lanes} : index
  scf.for %wr0 = [%workitem to %outputs step %lanes_step] {{
    %rr = index.assume %wr0 [range(%wr0, 0, {R - 1})] : index""")
    for wi in range(nw):
        L.append(f"""    %t{wi} = scf.for %ww = [%c0 to %cW step %c1](%acc = %zero_f : f32) -> (f32) {{
      %bw = index.assume %ww [range(%ww, 0, {waves - 1})] : index
      %p = view.load %pv[{wi}, %bw, %rr] : view<{nw}x{waves}x{R}xf32> -> f32
      %nx = scalar.addf %acc, %p : f32
      scf.yield %nx : f32
    }}""")
    if nw == 2:
        L.append("""    %silu = scalar.siluf<afn> %t0 : f32
    %value = scalar.mulf %silu, %t1 : f32""")
    else:
        L.append("    %value = scalar.addf %t0, %zero_f : f32")
    L.append("""    %out_row = index.add %row_base, %rr : index
    func.call @g_store_row(%y, %tokens, %n, %a, %out_row, %value, %c0_i32) : (buffer, index, index, index, index, f32, i32)
  }
  kernel.return
}
""")
    return "\n".join(L)


def combine(H, KU, acc=False):
    return f"""
kernel.def export("g_moe_combine") @g_moe_combine(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cb = index.constant {(H + 255) // 256} : index
  kernel.launch.config workgroups(%tokens, %cb, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %ys: buffer, %wts: buffer, %y: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %thread_id = kernel.workitem.id<x> : index
  %blk_id = kernel.workgroup.id<y> : index
  %c256 = index.constant 256 : index
  %boff = index.mul %blk_id, %c256 : index
  %c_raw = index.add %thread_id, %boff : index
  %cH = index.constant {H} : index
  %in = index.cmp ult, %c_raw, %cH : index
  scf.if %in {{
    %ch = index.assume %c_raw [range(%c_raw, 0, {H - 1})] : index
    %zero = index.constant 0 : offset
    %zero_f = scalar.constant 0.0 : f32
    %ys_view = buffer.view %ys[%zero] : buffer -> view<[%tokens]x{KU}x{H}xf32>
    %w_view = buffer.view %wts[%zero] : buffer -> view<[%tokens]x{KU}xf32>
    %y_view = buffer.view %y[%zero] : buffer -> view<[%tokens]x{H}xf32>
""" + (f"    %y_old = view.load %y_view[%token, %ch] : view<[%tokens]x{H}xf32> -> f32\n" if acc else "") + "\n".join(f"""    %w{k} = view.load %w_view[%token, {k}] : view<[%tokens]x{KU}xf32> -> f32
    %v{k} = view.load %ys_view[%token, {k}, %ch] : view<[%tokens]x{KU}x{H}xf32> -> f32
    %a{k} = scalar.fmaf %w{k}, %v{k}, {("%y_old" if acc else "%zero_f") if k == 0 else f"%a{k - 1}"} : f32""" for k in range(KU)) + f"""
    view.store %a{KU - 1}, %y_view[%token, %ch] : f32, view<[%tokens]x{H}xf32>
  }}
  kernel.return
}}
"""


def shared_gated(text):
    """A combine kernel (combine / combine_sorted with acc=True) that scales the shared experts' output already in y
    by the token's gate sg[t] (g_moe_shared_gate) before adding the routed experts: y = sg * y + sum_k w_k ys_k."""
    def rep(a, b):
        nonlocal text
        assert text.count(a) == 1, a
        text = text.replace(a, b)
    rep("%wts: buffer, %y: buffer) {", "%wts: buffer, %sg: buffer, %y: buffer) {")
    rep("    %y_old = view.load %y_view[%token, %ch]", """    %sg_view = buffer.view %sg[%zero] : buffer -> view<[%tokens]xf32>
    %sg_t = view.load %sg_view[%token] : view<[%tokens]xf32> -> f32
    %y_sh = view.load %y_view[%token, %ch]""")
    rep(" -> f32\n    %w0 = ", " -> f32\n    %y_old = scalar.mulf %y_sh, %sg_t : f32\n    %w0 = ")
    return text


def shared_gate(H, wty="f32"):
    """g_moe_shared_gate: sg[t] = sigmoid(h[t] . w) (Qwen2/3/3.5-MoE ffn_gate_inp_shexp: one output per token, f32 or
    bf16), one 256-thread workgroup per token, f32 accumulation."""
    HG = H // 32
    load = (f"""    %wraw = vector.load %w_view[%bg, %c0] : view<{HG}x32xbf16> -> vector<32xbf16>
    %wv = vector.extf %wraw : vector<32xbf16> to vector<32xf32>""" if wty == "bf16" else
            f"""    %wv = vector.load %w_view[%bg, %c0] : view<{HG}x32xf32> -> vector<32xf32>""")
    return f"""
kernel.def export("g_moe_shared_gate") @g_moe_shared_gate(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %x: buffer, %w: buffer, %sg: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %t0 = kernel.workitem.id<x> : index
  %tid = index.assume %t0 [range(%t0, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c256 = index.constant 256 : index
  %cHG = index.constant {HG} : index
  %zero_f = scalar.constant 0.0 : f32
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
  %w_view = buffer.view %w[%zero] : buffer -> view<{HG}x32x{wty}>
  %part = scf.for %g = [%tid to %cHG step %c256](%acc = %zero_f : f32) -> (f32) {{
    %bg = index.assume %g [range(%g, 0, {HG - 1})] : index
    %xv = vector.load %x_view[%token, %bg, %c0] : view<[%tokens]x{HG}x32xf32> -> vector<32xf32>
{load}
    %d = vector.dotf<reassoc|contract> %wv, %xv, %zero_f : vector<32xf32>, vector<32xf32>, f32
    %nx = scalar.addf %acc, %d : f32
    scf.yield %nx : f32
  }}
  %dot = kernel.workgroup.reduce<addf> %part : f32
  %first = index.cmp eq, %tid, %c0 : index
  scf.if %first {{
    %gate = scalar.logisticf<afn> %dot : f32
    %sg_view = buffer.view %sg[%zero] : buffer -> view<[%tokens]xf32>
    view.store %gate, %sg_view[%token] : f32, view<[%tokens]xf32>
  }}
  kernel.return
}}
"""


# ---------------------------------------------------------------- prefill: grouped expert GEMMs
TILE = 64


def bucket(E, KU):
    """Assignments a = t*KU + k bucketed by expert into 64-row tiles (flint_moe_bucket pattern)."""
    assert E <= 256
    return f"""
kernel.def export("g_moe_bucket") @g_moe_bucket(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %max_tiles_arg: index, %info: buffer, %ids: buffer, %tile_info: buffer, %rowmap: buffer, %pos: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %max_tiles = index.assume %max_tiles_arg [range(%max_tiles_arg, 1, 65536)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %cKU = index.constant {KU} : index
  %cT = index.constant {TILE} : index
  %c256 = index.constant 256 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %cshift = scalar.constant 6 : i32
  %cTm1 = scalar.constant {TILE - 1} : i32
  %minus_one = scalar.constant -1 : i32
  %cEm1 = scalar.constant {E - 1} : i32
  %cE = index.constant {E} : index
  %lane32 = index.cast %lane : index to i32
  %assignments = index.mul %tokens, %cKU : index
  %ids_view = buffer.view %ids[%zero] : buffer -> view<[%tokens]x{KU}xi32>
  %token_info = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %counter_bytes = index.constant 2048 : offset
  %counters = buffer.alloca<workgroup> align(16) %counter_bytes : buffer
  %count_view = buffer.view %counters[%zero] : buffer -> view<256xi32>
  %cursor_origin = index.constant 1024 : offset
  %cursor_view = buffer.view %counters[%cursor_origin] : buffer -> view<256xi32>
  view.store %c0_i32, %count_view[%lane] : i32, view<256xi32>
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  scf.for %a = [%lane to %assignments step %c256] {{
    %t_value = index.div %a, %cKU : index
    %t = index.assume %t_value [lt(%t_value, %tokens)] : index
    %s_value = index.rem %a, %cKU : index
    %s = index.assume %s_value [range(%s_value, 0, {KU - 1})] : index
    %id = view.load %ids_view[%t, %s] : view<[%tokens]x{KU}xi32> -> i32
    %low = scalar.maxsi %id, %c0_i32 : i32
    %clamped = scalar.minsi %low, %cEm1 : i32
    %bounded = scalar.assume %clamped [range(%clamped, 0, {E - 1})] : i32
    %e = index.cast %bounded : i32 to index
    %live32 = view.load %token_info[%t, 2] : view<[%tokens]x4xi32> -> i32
    %live = scalar.cmpi ne, %live32, %c0_i32 : i32
    scf.if %live {{
      %old = view.atomic.rmw<addi> %c1_i32, %count_view[%e] {{ordering = relaxed, scope = workgroup}} : i32, view<256xi32> -> i32
    }}
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %count32 = view.load %count_view[%lane] : view<256xi32> -> i32
  %padded = scalar.addi %count32, %cTm1 : i32
  %tiles32 = scalar.shrui %padded, %cshift : i32
  %base32 = kernel.workgroup.scan<addi> %tiles32 {{direction = forward, mode = exclusive}} : i32
  %total32 = kernel.workgroup.reduce<addi> %tiles32 : i32
  %info_view = buffer.view %tile_info[%zero] : buffer -> view<[%max_tiles]x2xi32>
  %rowmap_rows = index.mul %max_tiles, %cT : index
  %rowmap_view = buffer.view %rowmap[%zero] : buffer -> view<[%rowmap_rows]xi32>
  %pos_view = buffer.view %pos[%zero] : buffer -> view<[%assignments]xi32>
  %base_bound = scalar.assume %base32 [range(%base32, 0, 65535)] : i32
  %base = index.cast %base_bound : i32 to index
  %tiles_bound = scalar.assume %tiles32 [range(%tiles32, 0, 65535)] : i32
  %tiles = index.cast %tiles_bound : i32 to index
  %count_bound = scalar.assume %count32 [range(%count32, 0, 2097152)] : i32
  %count = index.cast %count_bound : i32 to index
  scf.for %j = [%c0 to %tiles step %c1] {{
    %tile_value = index.add %base, %j : index
    %tile = index.assume %tile_value [lt(%tile_value, %max_tiles)] : index
    %first = index.mul %j, %cT : index
    %remaining = index.sub %count, %first : index
    %full = index.cmp uge, %remaining, %cT : index
    %rows = scf.select %full, %cT, %remaining : index
    %rows32 = index.cast %rows : index to i32
    view.store %lane32, %info_view[%tile, 0] : i32, view<[%max_tiles]x2xi32>
    view.store %rows32, %info_view[%tile, 1] : i32, view<[%max_tiles]x2xi32>
  }}
  %total_bound = scalar.assume %total32 [range(%total32, 0, 65535)] : i32
  %total = index.cast %total_bound : i32 to index
  %unused_start = index.add %total, %lane : index
  scf.for %u = [%unused_start to %max_tiles step %c256] {{
    %bu = index.assume %u [lt(%u, %max_tiles)] : index
    view.store %minus_one, %info_view[%bu, 0] : i32, view<[%max_tiles]x2xi32>
    view.store %c0_i32, %info_view[%bu, 1] : i32, view<[%max_tiles]x2xi32>
  }}
  %first_row = index.mul %base, %cT : index
  %first_row32 = index.cast %first_row : index to i32
  view.store %first_row32, %cursor_view[%lane] : i32, view<256xi32>
  %used_end = index.add %first_row, %count : index
  %tile_rows = index.mul %tiles, %cT : index
  %end_row = index.add %first_row, %tile_rows : index
  scf.for %r = [%used_end to %end_row step %c1] {{
    %row = index.assume %r [range(%r, 0, 2097151), lt(%r, %rowmap_rows)] : index
    view.store %minus_one, %rowmap_view[%row] : i32, view<[%rowmap_rows]xi32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  scf.for %a = [%lane to %assignments step %c256] {{
    %t_value = index.div %a, %cKU : index
    %t = index.assume %t_value [lt(%t_value, %tokens)] : index
    %s_value = index.rem %a, %cKU : index
    %s = index.assume %s_value [range(%s_value, 0, {KU - 1})] : index
    %id = view.load %ids_view[%t, %s] : view<[%tokens]x{KU}xi32> -> i32
    %low = scalar.maxsi %id, %c0_i32 : i32
    %clamped = scalar.minsi %low, %cEm1 : i32
    %bounded = scalar.assume %clamped [range(%clamped, 0, {E - 1})] : i32
    %e = index.cast %bounded : i32 to index
    %live32p = view.load %token_info[%t, 2] : view<[%tokens]x4xi32> -> i32
    %livep = scalar.cmpi ne, %live32p, %c0_i32 : i32
    %ba = index.assume %a [lt(%a, %assignments)] : index
    scf.if %livep {{
      %row32 = view.atomic.rmw<addi> %c1_i32, %cursor_view[%e] {{ordering = relaxed, scope = workgroup}} : i32, view<256xi32> -> i32
      %row_bound = scalar.assume %row32 [range(%row32, 0, 2097151)] : i32
      %row_value = index.cast %row_bound : i32 to index
      %row = index.assume %row_value [lt(%row_value, %rowmap_rows)] : index
      %a32 = index.cast %a : index to i32
      view.store %a32, %rowmap_view[%row] : i32, view<[%rowmap_rows]xi32>
      view.store %row32, %pos_view[%ba] : i32, view<[%assignments]xi32>
    }} else {{
      view.store %minus_one, %pos_view[%ba] : i32, view<[%assignments]xi32>
    }}
  }}
  kernel.return
}}
"""


def grouped_gemm(fmt, mode, N, K, E, KU, gemm_fn):
    """64x64-tile WMMA GEMM over the bucketed rows of one expert per row tile (rows = max_tiles*64).
    mode 'swiglu' (gate/up): A rows gathered from x16[token = rowmap/KU]; 'store' (down): A = sorted rows."""
    import gen_gemm
    rb = N and (K // SB[fmt] * FMT[fmt])
    eb = rb * N
    text = gemm_fn(fmt, mode)
    base = f"g_gemm_{fmt}_{mode}"
    name = f"g_moe_g{'gu' if mode == 'swiglu' else 'dn'}_{fmt}_k{K}"
    def rep(a, b, cnt=1):
        nonlocal text
        assert text.count(a) == cnt, (name, a, text.count(a))
        text = text.replace(a, b)
    rep(f'export("{base}") @{base}(', f'export("{name}") @{name}(')
    rep("launch(%rows_arg: index, %k_arg: index, %n_arg: index, %x: buffer, %scales: buffer,",
        "launch(%rows_arg: index, %k_arg: index, %n_arg: index, %xrows_arg: index, %tile_info: buffer, %rowmap: buffer, %x: buffer, %scales: buffer,")
    # rows = max_tiles * 64 <= 8192 tokens * KU + 64 * E; n is this kernel's N. Tight bounds keep the output tile's
    # dynamic-stride extent (rows * n * 4 bytes) inside what the AMDGPU fragment store accepts (rows 2^20 x n 2^18 failed).
    max_rows = 8192 * KU + TILE * E
    rep("[range(%rows_arg, 64, 8192), multiple_of(%rows_arg, 64)]", f"[range(%rows_arg, 64, {max_rows}), multiple_of(%rows_arg, 64)]")
    rep("[range(%n_arg, 64, 262144), multiple_of(%n_arg, 64)]", f"[range(%n_arg, 64, {N}), multiple_of(%n_arg, 64)]")
    rep("  %origin = index.constant 0 : offset\n", "  %origin = index.constant 0 : offset\n  %xrows = index.assume %xrows_arg [range(%xrows_arg, 1, 1048576)] : index\n")
    # skip tiles of no expert; expert weight offset
    rep("  %workitem_id = kernel.workitem.id<x> : index\n", f"""  %ti_view = buffer.view %tile_info[%origin] : buffer -> view<[%row_tile_count]x2xi32>
  %tile_expert = view.load %ti_view[%row_group, 0] : view<[%row_tile_count]x2xi32> -> i32
  %c0_i32g = scalar.constant 0 : i32
  %tile_used = scalar.cmpi sge, %tile_expert, %c0_i32g : i32
  scf.if %tile_used {{
  %cEm1g = scalar.constant {E - 1} : i32
  %e_lo = scalar.minsi %tile_expert, %cEm1g : i32
  %e_b = scalar.assume %e_lo [range(%e_lo, 0, {E - 1})] : i32
  %expert = index.cast %e_b : i32 to index
  %expert_bytes = index.constant {eb} : offset
  %expert_off = index.scale %expert, %expert_bytes : index, offset -> offset
  %workitem_id = kernel.workitem.id<x> : index
""")
    rep("  %w_row_origin = index.scale %bounded_w_out_row, %row_bytes : index, offset -> offset\n",
        "  %w_row_origin0 = index.scale %bounded_w_out_row, %row_bytes : index, offset -> offset\n  %w_row_origin = index.add %w_row_origin0, %expert_off : offset\n")
    rep("  %x_view = buffer.view %x[%origin] : buffer -> view<[%rows]x[%k]xf16>", "  %x_view = buffer.view %x[%origin] : buffer -> view<[%xrows]x[%k]xf16>")
    rep("view<[%rows]x[%k]xf16>", "view<[%xrows]x[%k]xf16>", text.count("view<[%rows]x[%k]xf16>"))
    rep("view<[%rows]x2xf32>", "view<[%xrows]x2xf32>", text.count("view<[%rows]x2xf32>"))
    if mode == "swiglu":
        gather = lambda pre, row_expr, out: f"""  %{pre}_prow_v = {row_expr}
  %{pre}_rm_rows = index.mul %row_tile_count, %c64 : index
  %{pre}_prow = index.assume %{pre}_prow_v [lt(%{pre}_prow_v, %{pre}_rm_rows)] : index
  %{pre}_rm_view = buffer.view %rowmap[%origin] : buffer -> view<[%{pre}_rm_rows]xi32>
  %{pre}_a32 = view.load %{pre}_rm_view[%{pre}_prow] : view<[%{pre}_rm_rows]xi32> -> i32
  %{pre}_valid = scalar.cmpi sge, %{pre}_a32, %c0_i32g : i32
  %{pre}_ac = scalar.maxsi %{pre}_a32, %c0_i32g : i32
  %{pre}_ab = scalar.assume %{pre}_ac [range(%{pre}_ac, 0, 1048575)] : i32
  %{pre}_a = index.cast %{pre}_ab : i32 to index
  %{pre}_cKU = index.constant {KU} : index
  %{pre}_t = index.div %{pre}_a, %{pre}_cKU : index
  %{out} = index.assume %{pre}_t [lt(%{pre}_t, %xrows)] : index
"""
        rep("  %x_row = index.add %row_origin, %stage_row : index\n  %bounded_x_row = index.assume %x_row [lt(%x_row, %rows)] : index\n",
            gather("gx", "index.add %row_origin, %stage_row : index", "bounded_x_row"))
        rep("    %x_half = vector.load %x_view[%bounded_x_row, %x_col] : view<[%xrows]x[%k]xf16> -> vector<16xf16>\n",
            """    %x_half_raw = vector.load %x_view[%bounded_x_row, %x_col] : view<[%xrows]x[%k]xf16> -> vector<16xf16>
    %zero16h = vector.constant 0.0 : vector<16xf16>
    %x_half = scf.select %gx_valid, %x_half_raw, %zero16h : vector<16xf16>
""")
        rep("    %inv_row = index.assume %inv_row_value [lt(%inv_row_value, %rows)] : index\n",
            gather("gi", "index.add %inv_row_value, %c0 : index", "inv_row"))
    else:
        rep("  %bounded_x_row = index.assume %x_row [lt(%x_row, %rows)] : index", "  %bounded_x_row = index.assume %x_row [lt(%x_row, %xrows)] : index")
        rep("    %inv_row = index.assume %inv_row_value [lt(%inv_row_value, %rows)] : index", "    %inv_row = index.assume %inv_row_value [lt(%inv_row_value, %xrows)] : index")
    rep("  kernel.return\n}\n", "  }\n  kernel.return\n}\n")
    if mode in os.environ.get("MOE_N_FASTEST", "swiglu,store").split(","):
        # output-column tiles fastest, on by default (MOE_N_FASTEST=0 off): consecutive workgroups share one expert row
        # tile, so A and that expert's weights stay in L2. Prefill +74 % at 1k on Qwen3.6-35B-A3B (lfm2moe: +24-30 %);
        # the same tiles in another order, so the same results
        rep("workgroups(%row_tiles, %out_tiles, %c1)", "workgroups(%out_tiles, %row_tiles, %c1)")
        rep("  %row_group_id = kernel.workgroup.id<x> : index\n  %out_group_id = kernel.workgroup.id<y> : index",
            "  %row_group_id = kernel.workgroup.id<y> : index\n  %out_group_id = kernel.workgroup.id<x> : index")
    return text


def combine_sorted(H, KU, acc=False):
    return f"""
kernel.def export("g_moe_combine_sorted") @g_moe_combine_sorted(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cb = index.constant {(H + 255) // 256} : index
  kernel.launch.config workgroups(%tokens, %cb, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %rows_arg: index, %ys: buffer, %pos: buffer, %wts: buffer, %y: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 1048576)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %thread_id = kernel.workitem.id<x> : index
  %blk_id = kernel.workgroup.id<y> : index
  %c256 = index.constant 256 : index
  %boff = index.mul %blk_id, %c256 : index
  %c_raw = index.add %thread_id, %boff : index
  %cH = index.constant {H} : index
  %in = index.cmp ult, %c_raw, %cH : index
  scf.if %in {{
    %ch = index.assume %c_raw [range(%c_raw, 0, {H - 1})] : index
    %zero = index.constant 0 : offset
    %zero_f = scalar.constant 0.0 : f32
    %c0_i32 = scalar.constant 0 : i32
    %ys_view = buffer.view %ys[%zero] : buffer -> view<[%rows]x{H}xf32>
    %pos_view = buffer.view %pos[%zero] : buffer -> view<[%tokens]x{KU}xi32>
    %w_view = buffer.view %wts[%zero] : buffer -> view<[%tokens]x{KU}xf32>
    %y_view = buffer.view %y[%zero] : buffer -> view<[%tokens]x{H}xf32>
""" + (f"    %y_old = view.load %y_view[%token, %ch] : view<[%tokens]x{H}xf32> -> f32\n" if acc else "") + "\n".join(f"""    %w{k} = view.load %w_view[%token, {k}] : view<[%tokens]x{KU}xf32> -> f32
    %p{k} = view.load %pos_view[%token, {k}] : view<[%tokens]x{KU}xi32> -> i32
    %ok{k} = scalar.cmpi sge, %p{k}, %c0_i32 : i32
    %v{k} = scf.if %ok{k} -> (f32) {{
      %pb{k} = scalar.assume %p{k} [range(%p{k}, 0, 1048575)] : i32
      %pi{k} = index.cast %pb{k} : i32 to index
      %pr{k} = index.assume %pi{k} [lt(%pi{k}, %rows)] : index
      %vv{k} = view.load %ys_view[%pr{k}, %ch] : view<[%rows]x{H}xf32> -> f32
      scf.yield %vv{k} : f32
    }} else {{
      scf.yield %zero_f : f32
    }}
    %a{k} = scalar.fmaf %w{k}, %v{k}, {("%y_old" if acc else "%zero_f") if k == 0 else f"%a{k - 1}"} : f32""" for k in range(KU)) + f"""
    view.store %a{KU - 1}, %y_view[%token, %ch] : f32, view<[%tokens]x{H}xf32>
  }}
  kernel.return
}}
"""
