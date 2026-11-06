"""Gemma-4 "assistant" drafters (GGUF arch gemma4-assistant): Gemma 4's own multi-token-prediction head, shipped
as a separate small GGUF (e.g. AtomicChat/gemma-4-E4B-it-assistant-GGUF), for gemma4-family targets.

llama.cpp src/models/gemma4-assistant.cpp, per draft round (draft_depth autoregressive rounds, like MTP):
  x = target_embedding[token] * sqrt(H)          (the target's token embedding, Gemma scale)
  h = previous round's h_next, or for round 0 the target's post-final-norm hidden of the last consumed input
  cur = pre_projection([x | h])                  (2H -> E, E = the assistant width, 256 for E4B)
  4 layers: attn_norm -> q (QA heads) -> q_norm -> RoPE -> attention over the TARGET's K/V (sliding layers read
            target layer L-2's cache, full layers L-1's: the assistant has no K/V of its own) -> attn_output ->
            post_attention_norm -> +residual -> ffn_norm -> GELU(gate) * up -> down -> post_ffw_norm -> +residual
            -> * layer_output_scale
  logits = token_embd(assistant, tied) . output_norm(cur);   h_next = post_projection(output_norm(cur)) (E -> H)
The proposal is the greedy token. Verification (draft/gen_draft.py) is unchanged: output stays exactly the target's.

Loom mapping: the MTP machinery of gen_draft.py (per-row carry of the target's final normalized hidden, written by
@llm_as_warm after every epoch; g_mtp_prepare / g_mtp_gather / g_mtp_advance per round). Attention reuses the trunk's
own decode kernels for those two target layers: the assistant's QA heads are placed into the target's QH-head
layout so that assistant head a reads KV head a // (QA / KVH), and the token's visible range is clamped to the
anchor's context (positions < P: the anchor and the drafts have no target K/V yet). The propose stage therefore
binds the TARGET's origins table and arena (draft/build_draft.py), not the draft cache.
"""
import math


def config(path):
    import gguf
    r = gguf.GGUFReader(path)
    f = lambda k, d=None: (r.fields[k].contents() if k in r.fields else d)
    assert f("general.architecture") == "gemma4-assistant", f"{path}: not a gemma4-assistant GGUF"
    A = lambda k, d=None: f(f"gemma4-assistant.{k}", d)
    tensors = {t.name: {"type": t.tensor_type.name.lower(), "shape": [int(x) for x in t.shape]} for t in r.tensors}
    import numpy as np
    cut = None
    if "rope_freqs.weight" in tensors:
        v = np.asarray([t for t in r.tensors if t.name == "rope_freqs.weight"][0].data)
        cut = int((v < 1e10).sum())          # pairs that rotate (llama.cpp freq factors 1e30 = no rotation)
    swa = list(A("attention.sliding_window_pattern"))
    return {"gguf": path, "layers": A("block_count"), "E": A("embedding_length"), "H": A("embedding_length_out"),
            "ffn": A("feed_forward_length"), "heads": A("attention.head_count"), "kv_heads": A("attention.head_count_kv"),
            "D": A("attention.key_length"), "D_swa": A("attention.key_length_swa"), "swa": swa,
            "theta": A("rope.freq_base"), "theta_swa": A("rope.freq_base_swa"), "rd": A("rope.dimension_count"),
            "rd_swa": A("rope.dimension_count_swa"), "eps": A("attention.layer_norm_rms_epsilon", 1e-6), "cut": cut,
            "vocab": tensors["token_embd.weight"]["shape"][1], "tensors": tensors}


def target_attention(program, L, swa_layer, full_layer):
    """The trunk's decode attention launches (split + combine) and their layer/sliding constants for the two target
    layers whose K/V the assistant reads, from the generated program text."""
    out = {}
    for kind, i in (("swa", swa_layer), ("full", full_layer)):
        a = program.index(f"// ---- layer {i}")
        b = program.find(f"// ---- layer {i + 1}", a + 1)
        b = b if b > 0 else program.index("\n  %head = command.parameter", a)
        sec = program[a:b].split("\n")
        consts = [l.strip() for l in sec if l.strip().startswith((f"%layer_{i} =", f"%sliding_{i} ="))]
        split = [l.strip() for l in sec if "_attention_split[%t]" in l and "listed" not in l]
        comb = [l.strip() for l in sec if "_attention_combine[%t]" in l and "listed" not in l]
        assert len(consts) == 2 and len(split) == 1 and len(comb) == 1, (kind, i, consts, split, comb)
        out[kind] = {"layer": i, "consts": consts, "split": split[0], "combine": comb[0]}
    return out


def _slot(a, GA, GT):
    return (a // GA) * GT + (a % GA)


def kernels(g):
    A = g["assist"]
    E, H, QA, KVH = A["E"], g["H"], A["heads"], A["kv_heads"]
    QT, GT = g["QH"], g["QH"] // g["KVH"]
    GA = QA // KVH
    assert KVH == g["KVH"] and QA % KVH == 0 and GA <= GT, "assistant KV heads must match the target's"
    assert E % 32 == 0 and E <= 8192
    EPS = float(A["eps"])
    T = []
    T.append(f"""
// x[t] = target_embedding[id] * sqrt({H}) (Gemma scale; the assistant embeds with the target's table).
kernel.def export("g_as_embed") @g_as_embed(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %ids: buffer, %table: buffer, %x: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [range(%token_id, 0, 2147483646), lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c256i = index.constant 256 : index
  %cgpb = index.constant {g["SB"][g["EFMT"]] // 32} : index
  %groups = index.constant {g["HG"]} : index
  %ids_view = buffer.view %ids[%zero] : buffer -> view<[%tokens]xi32>
  %raw = view.load %ids_view[%token] : view<[%tokens]xi32> -> i32
  %c0_i32 = scalar.constant 0 : i32
  %vocab_last = scalar.constant {g["V"] - 1} : i32
  %low = scalar.maxsi %raw, %c0_i32 : i32
  %clamped = scalar.minsi %low, %vocab_last : i32
  %bounded = scalar.assume %clamped [range(%clamped, 0, {g["V"] - 1})] : i32
  %id = index.cast %bounded : i32 to index
  %row_bytes = index.constant {g["row_bytes"](g["EFMT"], H)} : offset
  %block_bytes = index.constant {g["FMT"][g["EFMT"]]} : offset
  %row_origin = index.scale %id, %row_bytes : index, offset -> offset
  %scale_s = scalar.constant {math.sqrt(H)!r} : f32
  %scales = vector.splat %scale_s : vector<32xf32>
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{g["HG"]}x32xf32>
  scf.for %gi = [%lane to %groups step %c256i] {{
    %gb = index.assume %gi [range(%gi, 0, {g["HG"] - 1})] : index
    %block = index.div %gb, %cgpb : index
    %within = index.rem %gb, %cgpb : index
    %block_add = index.scale %block, %block_bytes : index, offset -> offset
    %origin = index.add %row_origin, %block_add : offset
    %v = func.call @g_{g["EFMT"]}_values(%table, %origin, %within) : (buffer, offset, index) -> (vector<32xf32>)
    %s = vector.mulf %v, %scales : vector<32xf32>
    vector.store %s, %x_view[%token, %gb, %c0] : vector<32xf32>, view<[%tokens]x{g["HG"]}x32xf32>
  }}
  kernel.return
}}

// out[t] = rmsnorm(a[t]) * w (+ b[t]) (* scale): the assistant's {E}-wide norms and residuals. add/scale are flags.
kernel.def export("g_as_norm") @g_as_norm(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %add: i32, %scaled: i32, %a: buffer, %w: buffer, %b: buffer, %scale: buffer, %out: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [range(%token_id, 0, 2147483646), lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c256i = index.constant 256 : index
  %width = index.constant {E} : index
  %zero_f = scalar.constant 0.0 : f32
  %c0_i32 = scalar.constant 0 : i32
  %a_view = buffer.view %a[%zero] : buffer -> view<[%tokens]x{E}xf32>
  %b_view = buffer.view %b[%zero] : buffer -> view<[%tokens]x{E}xf32>
  %o_view = buffer.view %out[%zero] : buffer -> view<[%tokens]x{E}xf32>
  %w_view = buffer.view %w[%zero] : buffer -> view<{E}xf32>
  %local = scf.for %c0v = [%lane to %width step %c256i](%acc = %zero_f : f32) -> (f32) {{
    %c = index.assume %c0v [range(%c0v, 0, {E - 1})] : index
    %v = view.load %a_view[%token, %c] : view<[%tokens]x{E}xf32> -> f32
    %n = scalar.fmaf %v, %v, %acc : f32
    scf.yield %n : f32
  }}
  %sum = kernel.workgroup.reduce<addf> %local : f32
  %count = scalar.constant {float(E)!r} : f32
  %mean = scalar.divf %sum, %count : f32
  %eps = scalar.constant {EPS!r} : f32
  %stable = scalar.addf %mean, %eps : f32
  %inv = scalar.rsqrtf %stable : f32
  %adds = scalar.cmpi ne, %add, %c0_i32 : i32
  %scales = scalar.cmpi ne, %scaled, %c0_i32 : i32
  %s_view = buffer.view %scale[%zero] : buffer -> view<1xf32>
  %one = scalar.constant 1.0 : f32
  %sv = scf.if %scales -> (f32) {{
    %l = view.load %s_view[0] : view<1xf32> -> f32
    scf.yield %l : f32
  }} else {{
    scf.yield %one : f32
  }}
  scf.for %c0v = [%lane to %width step %c256i] {{
    %c = index.assume %c0v [range(%c0v, 0, {E - 1})] : index
    %v = view.load %a_view[%token, %c] : view<[%tokens]x{E}xf32> -> f32
    %wv = view.load %w_view[%c] : view<{E}xf32> -> f32
    %n = scalar.mulf %v, %inv : f32
    %y = scalar.mulf %n, %wv : f32
    %r = scf.if %adds -> (f32) {{
      %bv = view.load %b_view[%token, %c] : view<[%tokens]x{E}xf32> -> f32
      %s = scalar.addf %y, %bv : f32
      scf.yield %s : f32
    }} else {{
      scf.yield %y : f32
    }}
    %o = scalar.mulf %r, %sv : f32
    view.store %o, %o_view[%token, %c] : f32, view<[%tokens]x{E}xf32>
  }}
  kernel.return
}}

// Visible range for the assistant's queries: the anchor's context only (positions < P). info rows hold the draft
// position P + round; queries see keys up to position P - 1; a span at P = 0 has no context (not live).
kernel.def export("g_as_visible") @g_as_visible(%tokens: index) {{
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
}} launch(%tokens_arg: index, %round_arg: index, %info: buffer, %out: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %round = index.assume %round_arg [range(%round_arg, 0, 7)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [range(%token_id, 0, 2147483646), lt(%token_id, %tokens)] : index
  %zero = index.constant 0 : offset
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %iv = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %ov = buffer.view %out[%zero] : buffer -> view<[%tokens]x4xi32>
  %row = view.load %iv[%token, 0] : view<[%tokens]x4xi32> -> i32
  %pos = view.load %iv[%token, 1] : view<[%tokens]x4xi32> -> i32
  %live = view.load %iv[%token, 2] : view<[%tokens]x4xi32> -> i32
  %span = view.load %iv[%token, 3] : view<[%tokens]x4xi32> -> i32
  %r32 = index.cast %round : index to i32
  %back = scalar.addi %r32, %c1_i32 : i32
  %vis = scalar.subi %pos, %back : i32
  %has = scalar.cmpi sge, %vis, %c0_i32 : i32
  %vis0 = scalar.maxsi %vis, %c0_i32 : i32
  %live_b = scalar.cmpi ne, %live, %c0_i32 : i32
  %ok = scalar.andi %live_b, %has : i1
  %live_o = scalar.extui %ok : i1 to i32
  view.store %row, %ov[%token, 0] : i32, view<[%tokens]x4xi32>
  view.store %vis0, %ov[%token, 1] : i32, view<[%tokens]x4xi32>
  view.store %live_o, %ov[%token, 2] : i32, view<[%tokens]x4xi32>
  view.store %span, %ov[%token, 3] : i32, view<[%tokens]x4xi32>
  kernel.return
}}
""")
    for D, theta, rd, cut in ((A["D_swa"], A["theta_swa"], A["rd_swa"], None), (A["D"], A["theta"], A["rd"], A["cut"])):
        HALF = D // 2
        pairs = min(HALF, cut) if cut else HALF
        LN = 2.0 * math.log(theta) / rd
        T.append(f"""
// Assistant queries for head_dim {D}: per-head rmsnorm * q_norm, NeoX RoPE at the draft position (pairs < {pairs}
// rotate), written into the target's {QT}-head layout (assistant head a -> slot (a / {GA}) * {GT} + a % {GA}).
kernel.def export("g_as_qrope_d{D}") @g_as_qrope_d{D}(%tokens: index) {{
  %cq = index.constant {QA} : index
  %lanes = index.constant {HALF} : index
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%tokens, %cq, %c1) workgroup_size(%lanes, %c1, %c1) : index
}} launch(%tokens_arg: index, %q: buffer, %w: buffer, %info: buffer, %qt: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [range(%token_id, 0, 2147483646), lt(%token_id, %tokens)] : index
  %head_id = kernel.workgroup.id<y> : index
  %ah = index.assume %head_id [range(%head_id, 0, {QA - 1})] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, {HALF - 1})] : index
  %half = index.constant {HALF} : index
  %lane2 = index.add %lane, %half : index
  %zero = index.constant 0 : offset
  %c0_i32 = scalar.constant 0 : i32
  %iv = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %pos32 = view.load %iv[%token, 1] : view<[%tokens]x4xi32> -> i32
  %q_view = buffer.view %q[%zero] : buffer -> view<[%tokens]x{QA}x{D}xf32>
  %w_view = buffer.view %w[%zero] : buffer -> view<{D}xf32>
  %a = view.load %q_view[%token, %ah, %lane] : view<[%tokens]x{QA}x{D}xf32> -> f32
  %b = view.load %q_view[%token, %ah, %lane2] : view<[%tokens]x{QA}x{D}xf32> -> f32
  %aa = scalar.mulf %a, %a : f32
  %bb = scalar.mulf %b, %b : f32
  %sq = scalar.addf %aa, %bb : f32
  %sum = kernel.workgroup.reduce<addf> %sq : f32
  %inv_d = scalar.constant {1.0 / D!r} : f32
  %mean = scalar.mulf %sum, %inv_d : f32
  %eps = scalar.constant {EPS!r} : f32
  %var = scalar.addf %mean, %eps : f32
  %inv = scalar.rsqrtf %var : f32
  %wa = view.load %w_view[%lane] : view<{D}xf32> -> f32
  %wb = view.load %w_view[%lane2] : view<{D}xf32> -> f32
  %na = scalar.mulf %a, %inv : f32
  %nb = scalar.mulf %b, %inv : f32
  %sa = scalar.mulf %na, %wa : f32
  %sb = scalar.mulf %nb, %wb : f32
  %pair_i = index.cast %lane : index to i32
  %pair_f = scalar.uitofp %pair_i : i32 to f32
  %pos_f = scalar.sitofp %pos32 : i32 to f32
  %step = scalar.constant {-LN!r} : f32
  %lf = scalar.mulf %pair_f, %step : f32
  %freq = scalar.expf<afn> %lf : f32
  %angle_raw = scalar.mulf %pos_f, %freq : f32
  %npairs = index.constant {pairs} : index
  %rot = index.cmp ult, %lane, %npairs : index
  %zero_f = scalar.constant 0.0 : f32
  %angle = scf.select %rot, %angle_raw, %zero_f : f32
  %sine = scalar.sinf<afn> %angle : f32
  %cosine = scalar.cosf<afn> %angle : f32
  %ac = scalar.mulf %sa, %cosine : f32
  %bs = scalar.mulf %sb, %sine : f32
  %bc = scalar.mulf %sb, %cosine : f32
  %as = scalar.mulf %sa, %sine : f32
  %ra = scalar.subf %ac, %bs : f32
  %rb = scalar.addf %bc, %as : f32
  %ga = index.constant {GA} : index
  %gt = index.constant {GT} : index
  %kv = index.div %ah, %ga : index
  %within = index.rem %ah, %ga : index
  %s0 = index.mul %kv, %gt : index
  %s1 = index.add %s0, %within : index
  %slot = index.assume %s1 [range(%s1, 0, {QT - 1})] : index
  %qt_view = buffer.view %qt[%zero] : buffer -> view<[%tokens]x{QT}x{D}xf32>
  view.store %ra, %qt_view[%token, %slot, %lane] : f32, view<[%tokens]x{QT}x{D}xf32>
  view.store %rb, %qt_view[%token, %slot, %lane2] : f32, view<[%tokens]x{QT}x{D}xf32>
  kernel.return
}}

// o[t, a] = target-layout attention output ot[t, slot(a)] (head_dim {D}).
kernel.def export("g_as_gather_d{D}") @g_as_gather_d{D}(%tokens: index) {{
  %cq = index.constant {QA} : index
  %lanes = index.constant {HALF} : index
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%tokens, %cq, %c1) workgroup_size(%lanes, %c1, %c1) : index
}} launch(%tokens_arg: index, %ot: buffer, %o: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [range(%token_id, 0, 2147483646), lt(%token_id, %tokens)] : index
  %head_id = kernel.workgroup.id<y> : index
  %ah = index.assume %head_id [range(%head_id, 0, {QA - 1})] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, {HALF - 1})] : index
  %half = index.constant {HALF} : index
  %lane2 = index.add %lane, %half : index
  %zero = index.constant 0 : offset
  %ga = index.constant {GA} : index
  %gt = index.constant {GT} : index
  %kv = index.div %ah, %ga : index
  %within = index.rem %ah, %ga : index
  %s0 = index.mul %kv, %gt : index
  %s1 = index.add %s0, %within : index
  %slot = index.assume %s1 [range(%s1, 0, {QT - 1})] : index
  %ot_view = buffer.view %ot[%zero] : buffer -> view<[%tokens]x{QT}x{D}xf32>
  %o_view = buffer.view %o[%zero] : buffer -> view<[%tokens]x{QA}x{D}xf32>
  %a = view.load %ot_view[%token, %slot, %lane] : view<[%tokens]x{QT}x{D}xf32> -> f32
  %b = view.load %ot_view[%token, %slot, %lane2] : view<[%tokens]x{QT}x{D}xf32> -> f32
  view.store %a, %o_view[%token, %ah, %lane] : f32, view<[%tokens]x{QA}x{D}xf32>
  view.store %b, %o_view[%token, %ah, %lane2] : f32, view<[%tokens]x{QA}x{D}xf32>
  kernel.return
}}
""")
    V = A["vocab"]
    T.append(f"""
// proposals[t] = argmax over the {V} logits of row t (lowest index among equal maxima).
kernel.def export("g_as_argmax") @g_as_argmax(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %logits: buffer, %out: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [range(%token_id, 0, 2147483646), lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c256i = index.constant 256 : index
  %cv = index.constant {V} : index
  %lv = buffer.view %logits[%zero] : buffer -> view<[%tokens]x{V}xf32>
  %ninf = scalar.constant -3.0e38 : f32
  %best = scf.for %i0 = [%lane to %cv step %c256i](%m = %ninf : f32) -> (f32) {{
    %i = index.assume %i0 [range(%i0, 0, {V - 1})] : index
    %v = view.load %lv[%token, %i] : view<[%tokens]x{V}xf32> -> f32
    %n = scalar.maxnumf %m, %v : f32
    scf.yield %n : f32
  }}
  %top = kernel.workgroup.reduce<maxnumf> %best : f32
  %big = scalar.constant {V} : i32
  %first = scf.for %i0 = [%lane to %cv step %c256i](%k = %big : i32) -> (i32) {{
    %i = index.assume %i0 [range(%i0, 0, {V - 1})] : index
    %v = view.load %lv[%token, %i] : view<[%tokens]x{V}xf32> -> f32
    %eq = scalar.cmpf oeq, %v, %top : f32
    %i32 = index.cast %i : index to i32
    %cand = scf.select %eq, %i32, %big : i32
    %n = scalar.minui %k, %cand : i32
    scf.yield %n : i32
  }}
  %pick = kernel.workgroup.reduce<minui> %first : i32
  %last = scalar.constant {V - 1} : i32
  %safe = scalar.minui %pick, %last : i32
  %ov = buffer.view %out[%zero] : buffer -> view<[%tokens]xi32>
  %c0 = index.constant 0 : index
  %lead = index.cmp eq, %lane, %c0 : index
  scf.if %lead {{
    view.store %safe, %ov[%token] : i32, view<[%tokens]xi32>
  }}
  kernel.return
}}
""")
    return "\n".join(T)


def programs(g):
    A = g["assist"]
    E, H, QA, F = A["E"], g["H"], A["heads"], A["ffn"]
    QT = g["QH"]
    lin, dsize, dfmt = g["lin"], g["asize"], g["afmt"]
    att = g["assist_attention"]
    V = A["vocab"]
    common = f"""  %t = config.get @llm.tokens : index
  %spans = config.get @llm.spans : index
  %context = config.get @llm.context : index
  %pool = config.get @llm.pool : index
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c64 = index.constant 64 : index
  %c128 = index.constant 128 : index
  %zero_i32 = scalar.constant 0 : i32
  %one_i32 = scalar.constant 1 : i32
  %gemm = index.cmp uge, %t, %c64 : index
  %single = index.cmp eq, %t, %c1 : index
  %big = index.cmp uge, %t, %c128 : index
  %c256_rows = index.constant 256 : index
  %t_rem256 = index.rem %t, %c256_rows : index
  %t_full256 = index.cmp eq, %t_rem256, %c0 : index
  %t_ge256 = index.cmp uge, %t, %c256_rows : index
  %huge = scalar.andi %t_full256, %t_ge256 : i1
  %scale_entry = index.constant 8 : offset
  %scale_size = index.scale %t, %scale_entry : index, offset -> offset
  %info_bytes = index.constant 16 : offset
  %info_size = index.scale %t, %info_bytes : index, offset -> offset
  %info = buffer.alloca<global> align(256) %info_size : buffer
"""

    def buf(name, width, f16=False):
        s = (f"  %{name}_bytes = index.constant {width * 4} : offset\n  %{name}_size = index.scale %t, %{name}_bytes : index, offset -> offset\n"
             f"  %{name} = buffer.alloca<global> align(256) %{name}_size : buffer\n")
        if f16:
            s += (f"  %{name}16_bytes = index.constant {width * 2} : offset\n  %{name}16_size = index.scale %t, %{name}16_bytes : index, offset -> offset\n"
                  f"  %{name}16 = buffer.alloca<global> align(256) %{name}16_size : buffer\n  %s_{name} = buffer.alloca<global> align(256) %scale_size : buffer\n")
        return s

    def scaled(name, width_ssa):
        return f"  scf.if %gemm {{\n    kernel.launch @g_row_scale_f16[%t](%t, {width_ssa}, %{name}, %s_{name}, %{name}16) : [index](index, index, buffer, buffer, buffer)\n  }}\n"

    def L(tname, x, w, y, k_ssa, n_ssa, kval, mode="store", w2=None):
        return "  " + lin(dfmt(tname), f"%{x}", f"%s_{x}", (w, w2) if mode == "swiglu" else w, dsize(tname), f"%{y}", k_ssa, n_ssa, kval, mode) + "\n"

    def W(name, root="%w_as"):
        t = A["tensors"][name]
        if t["type"] == "f32":
            n = 1
            for s in t["shape"]:
                n *= s
            return f'command.parameter {root}, "{name}" : view<{n}xf32>'
        return f'command.parameter {root}, "{name}" : view<{dsize(name)}xi8>'

    warm = f"""
// Assistant catch-up: only the per-row carry (the target's final normalized hidden of the last consumed input);
// the assistant reads the target's own K/V, so it keeps no cache of its own.
command.program.def public @llm_as_warm() launch(%target: buffer, %metadata: buffer, %origins: buffer, %arena: buffer, %carry: buffer, %tokens: buffer) {{
  %t = config.get @llm.tokens : index
  %spans = config.get @llm.spans : index
  %owners = index.constant 1 : index
  kernel.launch @g_mtp_save[%spans](%t, %spans, %owners, %metadata, %target, %carry) : [index](index, index, index, buffer, buffer, buffer)
  command.return
}}
"""
    Dmax = max(A["D"], A["D_swa"])
    P = [f"""
// Assistant proposals: draft_depth autoregressive rounds for the PROPOSE spans (one token per span per round),
// attending the target's K/V; each greedy token goes into the verifier's input slot and feeds the next round.
command.program.def public @llm_as_propose() launch(%w_embed: buffer, %w_as: buffer, %epoch: buffer, %target_tokens: buffer, %origins: buffer, %arena: buffer, %carry: buffer) {{
{common}  %owners = index.constant 1 : index
  %epoch_spans = config.get @llm.epoch_spans : index
  %epoch_tokens = config.get @llm.epoch_tokens : index
  %width = config.get @llm.width : index
  %depth = index.sub %width, %c1 : index
  %nH = index.constant {H} : index
  %nH2 = index.constant {2 * H} : index
  %nE = index.constant {E} : index
  %nF = index.constant {F} : index
  %nV = index.constant {V} : index
  %meta_bytes = index.constant {16 + 40 * 8192} : offset
  %metadata = buffer.alloca<global> align(256) %meta_bytes : buffer
  %c4_o = index.constant 4 : offset
  %word_size = index.scale %t, %c4_o : index, offset -> offset
  %tokens = buffer.alloca<global> align(256) %word_size : buffer
  %slots = buffer.alloca<global> align(256) %word_size : buffer
  %proposals = buffer.alloca<global> align(256) %word_size : buffer
  %vis = buffer.alloca<global> align(256) %info_size : buffer
{buf("hidden", H)}{buf("x", H)}{buf("xh", 2 * H, True)}{buf("cur", E)}{buf("nrm", E, True)}{buf("y", E)}{buf("qa", QA * Dmax)}{buf("oa", QA * Dmax, True)}{buf("f", F, True)}{buf("logits", V)}{buf("qt", QT * Dmax)}{buf("ot", QT * Dmax)}  %pacc_bytes = index.constant {QT * 64 * Dmax * 4} : offset
  %pml_bytes = index.constant {QT * 64 * 2 * 4} : offset
  %pacc_size = index.scale %t, %pacc_bytes : index, offset -> offset
  %pml_size = index.scale %t, %pml_bytes : index, offset -> offset
  %pacc = buffer.alloca<global> align(256) %pacc_size : buffer
  %pml = buffer.alloca<global> align(256) %pml_size : buffer
  %no_scale = buffer.alloca<global> align(256) %c4_o : buffer
  %embedding = command.parameter %w_embed, "token_embd.weight" : view<{g["esize"]}xi8>
  %w_pre = {W("nextn.pre_projection.weight")}
  %w_post = {W("nextn.post_projection.weight")}
  %w_head = {W("token_embd.weight")}
  %w_onorm = {W("output_norm.weight")}
""" + "".join(f"  {s}\n" for kind in ("swa", "full") for s in att[kind]["consts"]) + """  kernel.launch @g_mtp_prepare(%epoch_spans, %t, %epoch_tokens, %epoch, %target_tokens, %metadata, %tokens, %slots) : (index, index, index, buffer, buffer, buffer, buffer, buffer)
  kernel.launch @g_mtp_gather[%t](%t, %owners, %metadata, %carry, %hidden) : [index](index, index, buffer, buffer, buffer)
  scf.for %round = [%c0 to %depth step %c1] unroll {
  kernel.launch @g_token_info[%t](%t, %spans, %metadata, %info) : [index](index, index, buffer, buffer)
  kernel.launch @g_as_visible[%t](%t, %round, %info, %vis) : [index](index, index, buffer, buffer)
"""]
    P.append(f"""  kernel.launch @g_as_embed[%t](%t, %tokens, %embedding, %x) : [index](index, buffer, view<{g["esize"]}xi8>, buffer)
  kernel.launch @g_mtp_concat[%t](%t, %x, %hidden, %xh) : [index](index, buffer, buffer, buffer)
{scaled("xh", "%nH2")}{L("nextn.pre_projection.weight", "xh", "%w_pre", "cur", "%nH2", "%nE", 2 * H)}""")
    for l in range(A["layers"]):
        swa = A["swa"][l]
        D = A["D_swa"] if swa else A["D"]
        kind = "swa" if swa else "full"
        b = f"blk.{l}."
        lt = att[kind]["layer"]
        split = att[kind]["split"].replace("(%t, ", "(%t, ", 1)
        split = split.replace(", %q, %info, ", ", %qt, %vis, ")
        comb = att[kind]["combine"].replace(", %info, ", ", %vis, ").replace(", %q, %o)", ", %qt, %ot)")
        assert "%qt" in split and "%vis" in split and "%ot" in comb, (split, comb)
        P.append(f"""  // ---- assistant layer {l} ({kind}: target layer {lt}'s K/V, head_dim {D})
  %an_{l} = {W(b + "attn_norm.weight")}
  %aq_{l} = {W(b + "attn_q.weight")}
  %aqn_{l} = {W(b + "attn_q_norm.weight")}
  %ao_{l} = {W(b + "attn_output.weight")}
  %apn_{l} = {W(b + "post_attention_norm.weight")}
  %fn_{l} = {W(b + "ffn_norm.weight")}
  %fg_{l} = {W(b + "ffn_gate.weight")}
  %fu_{l} = {W(b + "ffn_up.weight")}
  %fd_{l} = {W(b + "ffn_down.weight")}
  %fpn_{l} = {W(b + "post_ffw_norm.weight")}
  %los_{l} = {W(b + "layer_output_scale.weight")}
  %nQ_{l} = index.constant {QA * D} : index
  kernel.launch @g_as_norm[%t](%t, %zero_i32, %zero_i32, %cur, %an_{l}, %cur, %no_scale, %nrm) : [index](index, i32, i32, buffer, view<{E}xf32>, buffer, buffer, buffer)
{scaled("nrm", "%nE")}{L(b + "attn_q.weight", "nrm", f"%aq_{l}", "qa", "%nE", f"%nQ_{l}", E)}  kernel.launch @g_as_qrope_d{D}[%t](%t, %qa, %aqn_{l}, %info, %qt) : [index](index, buffer, view<{D}xf32>, buffer, buffer)
  {split}
  {comb}
  kernel.launch @g_as_gather_d{D}[%t](%t, %ot, %oa) : [index](index, buffer, buffer)
{scaled("oa", f"%nQ_{l}")}{L(b + "attn_output.weight", "oa", f"%ao_{l}", "y", f"%nQ_{l}", "%nE", QA * D)}  kernel.launch @g_as_norm[%t](%t, %one_i32, %zero_i32, %y, %apn_{l}, %cur, %no_scale, %cur) : [index](index, i32, i32, buffer, view<{E}xf32>, buffer, buffer, buffer)
  kernel.launch @g_as_norm[%t](%t, %zero_i32, %zero_i32, %cur, %fn_{l}, %cur, %no_scale, %nrm) : [index](index, i32, i32, buffer, view<{E}xf32>, buffer, buffer, buffer)
{scaled("nrm", "%nE")}{L(b + "ffn_gate.weight", "nrm", f"%fg_{l}", "f", "%nE", "%nF", E, "swiglu", f"%fu_{l}")}{scaled("f", "%nF")}{L(b + "ffn_down.weight", "f", f"%fd_{l}", "y", "%nF", "%nE", F)}  kernel.launch @g_as_norm[%t](%t, %one_i32, %one_i32, %y, %fpn_{l}, %cur, %los_{l}, %cur) : [index](index, i32, i32, buffer, view<{E}xf32>, buffer, view<1xf32>, buffer)
""")
    P.append(f"""  kernel.launch @g_as_norm[%t](%t, %zero_i32, %zero_i32, %cur, %w_onorm, %cur, %no_scale, %nrm) : [index](index, i32, i32, buffer, view<{E}xf32>, buffer, buffer, buffer)
{scaled("nrm", "%nE")}{L("token_embd.weight", "nrm", "%w_head", "logits", "%nE", "%nV", E)}  kernel.launch @g_as_argmax[%t](%t, %logits, %proposals) : [index](index, buffer, buffer)
{L("nextn.post_projection.weight", "nrm", "%w_post", "hidden", "%nE", "%nH", E)}  kernel.launch @g_mtp_advance(%round, %t, %epoch_tokens, %epoch, %metadata, %slots, %proposals, %tokens, %target_tokens) : (index, index, index, buffer, buffer, buffer, buffer, buffer, buffer)
  }}
  command.return
}}
""")
    decls = ("config.decl @llm.epoch_spans : %value: index where [range(%value, 1, 8192)]\n\n"
             "config.decl @llm.epoch_tokens : %value: index where [range(%value, 1, 8192)]\n")
    return decls + warm + "".join(P)
