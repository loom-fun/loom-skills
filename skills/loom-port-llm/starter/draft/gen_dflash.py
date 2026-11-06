"""DFlash / DFlash2 block-diffusion drafters (z-lab, GGUF arch "dflash") for starter ports.

A DFlash drafter is a small non-causal transformer that proposes a whole block at once: the block is
[anchor, <mask> x (block_size - 1)] at positions P..P+B-1; every draft layer attends to the block (no causal
mask) and to "context" K/V built from the target's hidden states at earlier positions. The target's token
embedding and output head are shared.
- Context: the target's residual stream after the layers in dflash.target_layers (HF hidden_states indices,
  i.e. after layer t-1) is captured for every consumed token (g_dfl_tap in @llm_advance / @llm_verify),
  projected by fc (NT*H -> H) and enc.output_norm, then each draft layer's attn_k / attn_v (+ k_norm, RoPE at
  the token's position) are stored in the draft K/V cache (2 planes per draft layer, same page ids as the
  target pool). @llm_dflash_warm does this after every epoch for the committed inputs only.
- DFlash2 adds: a sliding window over the context (dflash.attention.sliding_window), conv modulation
  (attn/ffn_conv_proj give per-position, per-16-channel-group coefficients; with the static attn/ffn_conv_base
  they form a 2-tap causal convolution inside the block, branch 0 before the projections, branch 1 on the
  sublayer output added to the residual), and a candidate selector (top-16 target-head candidates per position,
  rescored with a rank-256 transition term pred[prev] . (selector_hidden(h) * succ[cand]), chosen greedily
  along the chain from the anchor).
- @llm_dflash_propose runs the drafter for every PROPOSE span and writes draft_depth proposals into the
  verifier's input slots. Acceptance (draft/gen_draft.py) is unchanged: output stays exactly the target's.
Adapted from hrx-loom-library model/Qwen/Qwen3.8-27B/program/{dflash,dflash_context,dflash_proposal}.loom and
kernel/dflash_*.loom, motif/tensor/grouped_convolution.loom (Apache-2.0).
"""
import math


def config(path):
    """Drafter geometry from its GGUF header (cfg["dflash"])."""
    import gguf
    r = gguf.GGUFReader(path)
    f = lambda k, d=None: (r.fields[k].contents() if k in r.fields else d)
    assert f("general.architecture") == "dflash", f"{path}: not a DFlash drafter GGUF"
    A = lambda k, d=None: f(f"dflash.{k}", d)
    tensors = {t.name: {"type": t.tensor_type.name.lower(), "shape": [int(x) for x in t.shape]} for t in r.tensors}
    H, heads, kv = A("embedding_length"), A("attention.head_count"), A("attention.head_count_kv")
    d = A("attention.key_length") or H // heads
    conv = None
    if "blk.0.attn_conv_proj.weight" in tensors:
        conv = {"taps": A("conv_kernel_size", 2), "group": A("conv_group_size", 16)}
    sel = None
    if "selector_hidden.weight" in tensors:
        sel = {"rank": A("selector_rank", tensors["selector_hidden.weight"]["shape"][1]), "top_k": A("selector_top_k", 16)}
    return {
        "gguf": path, "name": f("general.name", "dflash"), "layers": A("block_count"), "hidden": H, "heads": heads,
        "kv_heads": kv, "head_dim": d, "ffn": A("feed_forward_length"), "rms_eps": A("attention.layer_norm_rms_epsilon", 1e-6),
        "rope_theta": A("rope.freq_base", 10000.0), "block_size": A("block_size", 16), "target_layers": list(A("target_layers")),
        "window": A("attention.sliding_window") or 0, "conv": conv, "selector": sel,
        "logit_scale": A("logit_scale"), "softcap": A("final_logit_softcapping"), "embed_scale": A("embedding_scale"),
        "mask_id": f("tokenizer.ggml.mask_token_id"), "tensors": tensors,
    }


def tap_layers(cfg):
    """Trunk layer index after which each feature is captured (HF hidden_states[t] = output of layer t - 1)."""
    return [t - 1 for t in cfg["dflash"]["target_layers"]]


def own_embed(g):
    """The drafter embeds its block itself (llama.cpp dflash.cpp): the target's raw token embedding row times the
    drafter's embedding_scale (Gemma-family drafters carry sqrt(hidden)), then its first attn_norm with its own eps.
    The trunk's g_embed does exactly that only when the target applies no embedding norm/scale of its own and the
    drafter neither scales nor uses another eps (Qwen): then propose keeps g_embed and the package is unchanged."""
    F = g["dflash"]
    return not (g.get("embed_plain", True) and not F.get("embed_scale") and float(F["rms_eps"]) == float(g["EPS"]))


def embed_kernel(g):
    F = g["dflash"]
    H, HG, V, EF = g["H"], g["HG"], g["V"], g["EFMT"]
    scale = float(F.get("embed_scale") or 1.0)
    return f"""
// Draft block embedding (own_embed): x[t] = embedding[id] * {scale!r}, h[t] = rmsnorm(x[t]) * w (drafter eps).
kernel.def export("g_dfl_embed") @g_dfl_embed(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %ids: buffer, %table: buffer, %w: buffer, %x: buffer, %h: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %cgpb = index.constant {g["SB"][EF] // 32} : index
  %groups = index.constant {HG} : index
  %zero_f = scalar.constant 0.0 : f32
  %zero32 = vector.constant 0.0 : vector<32xf32>
  %ids_view = buffer.view %ids[%zero] : buffer -> view<[%tokens]xi32>
  %raw = view.load %ids_view[%token] : view<[%tokens]xi32> -> i32
  %c0_i32 = scalar.constant 0 : i32
  %vocab_last = scalar.constant {V - 1} : i32
  %low = scalar.maxsi %raw, %c0_i32 : i32
  %clamped = scalar.minsi %low, %vocab_last : i32
  %bounded = scalar.assume %clamped [range(%clamped, 0, {V - 1})] : i32
  %id = index.cast %bounded : i32 to index
  %row_bytes = index.constant {g["row_bytes"](EF, H)} : offset
  %block_bytes = index.constant {g["FMT"][EF]} : offset
  %row_origin = index.scale %id, %row_bytes : index, offset -> offset
  %active = index.cmp ult, %lane, %groups : index
  %values = scf.if %active -> (vector<32xf32>) {{
    %block = index.div %lane, %cgpb : index
    %within = index.rem %lane, %cgpb : index
    %block_add = index.scale %block, %block_bytes : index, offset -> offset
    %origin = index.add %row_origin, %block_add : offset
    %v = func.call @g_{EF}_values(%table, %origin, %within) : (buffer, offset, index) -> (vector<32xf32>)
    scf.yield %v : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}
  %scale_s = scalar.constant {scale!r} : f32
  %scales = vector.splat %scale_s : vector<32xf32>
  %scaled = vector.mulf %values, %scales : vector<32xf32>
  %squares = vector.mulf %scaled, %scaled : vector<32xf32>
  %local = vector.reduce<addf> %squares, %zero_f : vector<32xf32>, f32
  %sum = kernel.workgroup.reduce<addf> %local : f32
  %count = scalar.constant {float(H)!r} : f32
  %mean = scalar.divf %sum, %count : f32
  %eps = scalar.constant {float(F["rms_eps"])!r} : f32
  %stable = scalar.addf %mean, %eps : f32
  %inverse = scalar.rsqrtf %stable : f32
  %inverses = vector.splat %inverse : vector<32xf32>
  scf.if %active {{
    %bl = index.assume %lane [range(%lane, 0, {HG - 1})] : index
    %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
    %h_view = buffer.view %h[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
    %w_view = buffer.view %w[%zero] : buffer -> view<{HG}x32xf32>
    vector.store %scaled, %x_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x{HG}x32xf32>
    %weights = vector.load %w_view[%bl, %c0] : view<{HG}x32xf32> -> vector<32xf32>
    %n = vector.mulf %scaled, %inverses : vector<32xf32>
    %hv = vector.mulf %n, %weights : vector<32xf32>
    vector.store %hv, %h_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x{HG}x32xf32>
  }}
  kernel.return
}}
"""


def rope_info_kernel():
    return """
// M-RoPE targets: rinfo = info with the temporal rotary position in column 1 (what llama.cpp gives a DFlash drafter:
// degenerate sections). Media epochs carry (t, h, w) in input-record words 4..6; otherwise KV position + the span's
// offset from the table after the records (records = 0: after the header count; else after `records` capacity).
kernel.def export("g_dfl_rope_info") @g_dfl_rope_info(%tokens: index) {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%tokens_arg: index, %spans_arg: index, %records_arg: index, %meta: buffer, %ids: buffer, %info: buffer, %rinfo: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %records_cap = index.assume %records_arg [range(%records_arg, 0, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [range(%token_id, 0, 2147483646), lt(%token_id, %tokens)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c4 = index.constant 4 : index
  %c0_i32 = scalar.constant 0 : i32
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %out = buffer.view %rinfo[%zero] : buffer -> view<[%tokens]x4xi32>
  %row32 = view.load %info_view[%token, 0] : view<[%tokens]x4xi32> -> i32
  %pos32 = view.load %info_view[%token, 1] : view<[%tokens]x4xi32> -> i32
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %span32 = view.load %info_view[%token, 3] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %header = buffer.view %meta[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %flag = view.load %header[2] : view<4xi32> -> i32
  %mixed = scalar.cmpi ne, %flag, %c0_i32 : i32
  %rope = scf.if %live -> (i32) {
    %t = scf.if %mixed -> (i32) {
      %c8 = index.constant 8 : index
      %n = index.mul %tokens, %c8 : index
      %rec = buffer.view %ids[%zero] : buffer -> view<[%n]xi32>
      %base = index.mul %token, %c8 : index
      %e4 = index.add %base, %c4 : index
      %b4 = index.assume %e4 [lt(%e4, %n)] : index
      %pt = view.load %rec[%b4] : view<[%n]xi32> -> i32
      scf.yield %pt : i32
    } else {
      %count_b = scalar.assume %count32 [range(%count32, 1, 8192)] : i32
      %count = index.cast %count_b : i32 to index
      %use_header = index.cmp eq, %records_cap, %c0 : index
      %records = scf.select %use_header, %count, %records_cap : index
      %sl = scalar.maxsi %span32, %c0_i32 : i32
      %sb = scalar.assume %sl [range(%sl, 0, 8191)] : i32
      %span0 = index.cast %sb : i32 to index
      %last = index.sub %records, %c1 : index
      %span1 = index.min %span0, %last : index
      %span = index.assume %span1 [lt(%span1, %records)] : index
      %c36 = index.constant 36 : offset
      %c16 = index.constant 16 : offset
      %plan = index.scale %records, %c36 : index, offset -> offset
      %table = index.add %c16, %plan : offset
      %tv = buffer.view %meta[%table] : buffer -> view<[%records]xi32>
      %delta = view.load %tv[%span] : view<[%records]xi32> -> i32
      %linear = scalar.addi %pos32, %delta : i32
      scf.yield %linear : i32
    }
    scf.yield %t : i32
  } else {
    scf.yield %pos32 : i32
  }
  view.store %row32, %out[%token, 0] : i32, view<[%tokens]x4xi32>
  view.store %rope, %out[%token, 1] : i32, view<[%tokens]x4xi32>
  view.store %live32, %out[%token, 2] : i32, view<[%tokens]x4xi32>
  view.store %span32, %out[%token, 3] : i32, view<[%tokens]x4xi32>
  kernel.return
}
"""


def kernels(g):
    F = g["dflash"]
    H, HG = g["H"], g["HG"]
    NT, QH, KVH, D, BS = len(F["target_layers"]), F["heads"], F["kv_heads"], F["head_dim"], F["block_size"]
    assert D % 64 == 0 and D <= 256 and QH % KVH == 0, "DFlash kernels: head_dim 64/128/256, grouped heads"
    HALF = D // 2
    GQ = QH // KVH
    EPS = F["rms_eps"]
    LN = 2.0 * math.log(F["rope_theta"]) / D
    POSB = KVH * D * 2
    W = F["window"] or 0
    MR = g["MROPE"]
    T = [embed_kernel(g)] if own_embed(g) else []
    if MR:
        T.append(rope_info_kernel())
    # ---- feature taps
    T.append(f"""
// feat[t][slot] = x[t] (the residual stream after a target layer listed in dflash.target_layers).
kernel.def export("g_dfl_tap") @g_dfl_tap(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %slot_arg: index, %x: buffer, %features: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %slot = index.assume %slot_arg [range(%slot_arg, 0, {NT - 1})] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [range(%token_id, 0, 2147483646), lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %groups = index.constant {HG} : index
  %active = index.cmp ult, %lane, %groups : index
  scf.if %active {{
    %bl = index.assume %lane [range(%lane, 0, {HG - 1})] : index
    %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
    %f_view = buffer.view %features[%zero] : buffer -> view<[%tokens]x{NT}x{HG}x32xf32>
    %v = vector.load %x_view[%token, %bl, %c0] : view<[%tokens]x{HG}x32xf32> -> vector<32xf32>
    vector.store %v, %f_view[%token, %slot, %bl, %c0] : vector<32xf32>, view<[%tokens]x{NT}x{HG}x32xf32>
  }}
  kernel.return
}}
""")
    # shared per-head rmsnorm + NeoX RoPE: lane l owns dims l and l + D/2
    def norm_rope(src_view, head, pos):
        return f"""    %a = view.load {src_view}[%token, {head}, %lane] : view<[%tokens]x%NHx{D}xf32> -> f32
    %b = view.load {src_view}[%token, {head}, %lane2] : view<[%tokens]x%NHx{D}xf32> -> f32
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
    %pair = index.cast %lane : index to i32
    %pair_f = scalar.uitofp %pair : i32 to f32
    %pos_f = scalar.sitofp {pos} : i32 to f32
    %step = scalar.constant {-LN!r} : f32
    %lf = scalar.mulf %pair_f, %step : f32
    %freq = scalar.expf<afn> %lf : f32
    %angle = scalar.mulf %pos_f, %freq : f32
    %sine = scalar.sinf<afn> %angle : f32
    %cosine = scalar.cosf<afn> %angle : f32
    %ac = scalar.mulf %sa, %cosine : f32
    %bs = scalar.mulf %sb, %sine : f32
    %bc = scalar.mulf %sb, %cosine : f32
    %as = scalar.mulf %sa, %sine : f32
    %ra = scalar.subf %ac, %bs : f32
    %rb = scalar.addf %bc, %as : f32
"""
    head_prelude = f"""  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [range(%token_id, 0, 2147483646), lt(%token_id, %tokens)] : index
  %head_id = kernel.workgroup.id<y> : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, {HALF - 1})] : index
  %half = index.constant {HALF} : index
  %lane2 = index.add %lane, %half : index
  %zero = index.constant 0 : offset
  %c0_i32 = scalar.constant 0 : i32
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %row32 = view.load %info_view[%token, 0] : view<[%tokens]x4xi32> -> i32
  %position32 = view.load %info_view[%token, 1] : view<[%tokens]x4xi32> -> i32
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %w_view = buffer.view %w[%zero] : buffer -> view<{D}xf32>
"""
    kv_offset = f"""
// Byte offset of (plane, position) in the draft K/V cache for `row` (16-position pages, same page maps as the
// target; positions are {POSB} bytes: {KVH} KV heads x {D} dims of F16).
func.def inline @g_dfl_kv_offset(%origins: buffer, %row: index, %position: index, %blocks: index, %plane: index, %pool: index) -> (offset) {{
  %c1 = index.constant 1 : index
  %c16 = index.constant 16 : index
  %record_stride = index.constant 40 : offset
  %position_bytes = index.constant {POSB} : offset
  %bounded_row = index.assume %row [range(%row, 0, 2147483646)] : index
  %bounded_position = index.assume %position [range(%position, 0, 262143)] : index
  %record_offset = index.scale %bounded_row, %record_stride : index, offset -> offset
  %records = buffer.view %origins[%record_offset] : buffer -> view<5xi64>
  %map64 = view.load %records[4] : view<5xi64> -> i64
  %map_bound = scalar.assume %map64 [range(%map64, 0, 9223372036854775807)] : i64
  %map_origin = index.cast %map_bound : i64 to offset
  %page_value = index.div %bounded_position, %c16 : index
  %last = index.sub %blocks, %c1 : index
  %page = index.assume %page_value [range(%page_value, 0, %last), lt(%page_value, %blocks)] : index
  %map = buffer.view %origins[%map_origin] : buffer -> view<[%blocks]xi32>
  %physical32 = view.load %map[%page] : view<[%blocks]xi32> -> i32
  %bounded_physical = scalar.assume %physical32 [range(%physical32, 0, 1048575)] : i32
  %physical = index.cast %bounded_physical : i32 to index
  %page_base = index.mul %physical, %c16 : index
  %within = index.rem %bounded_position, %c16 : index
  %slot = index.add %page_base, %within : index
  %plane_slot = index.madd %plane, %pool, %slot : index
  %bytes = index.scale %plane_slot, %position_bytes : index, offset -> offset
  func.return %bytes : offset
}}
"""
    T.append(kv_offset)
    T.append(f"""
// Context K/V of draft layer `layer` at each consumed token's position: rmsnorm(k) * k_norm, RoPE, F16.
kernel.def export("g_dfl_kv_store") @g_dfl_kv_store(%tokens: index) {{
  %ckv = index.constant {KVH} : index
  %lanes = index.constant {HALF} : index
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%tokens, %ckv, %c1) workgroup_size(%lanes, %c1, %c1) : index
}} launch(%tokens_arg: index, %layer_arg: index, %context_arg: index, %pool_arg: index, %k: buffer, %v: buffer, %w: buffer, %info: buffer, %origins: buffer, %cache: buffer{", %rinfo: buffer" if MR else ""}) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {F["layers"] - 1})] : index
  %context = index.assume %context_arg [range(%context_arg, 16, 262144)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 1048576)] : index
{head_prelude}  %kh = index.assume %head_id [range(%head_id, 0, {KVH - 1})] : index
  %k_view = buffer.view %k[%zero] : buffer -> view<[%tokens]x{KVH}x{D}xf32>
  %v_view = buffer.view %v[%zero] : buffer -> view<[%tokens]x{KVH}x{D}xf32>
{"  %rinfo_view = buffer.view %rinfo[%zero] : buffer -> view<[%tokens]x4xi32>" + chr(10) + "  %rope32 = view.load %rinfo_view[%token, 1] : view<[%tokens]x4xi32> -> i32" + chr(10) if MR else ""}""" + norm_rope("%k_view", "%kh", "%rope32" if MR else "%position32").replace("%NH", str(KVH)) + f"""  scf.if %live {{
    %c15 = index.constant 15 : index
    %c16 = index.constant 16 : index
    %c2 = index.constant 2 : index
    %row_b = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
    %row = index.cast %row_b : i32 to index
    %pos_b = scalar.assume %position32 [range(%position32, 0, 262143)] : i32
    %position = index.cast %pos_b : i32 to index
    %padded = index.add %context, %c15 : index
    %blocks = index.div %padded, %c16 : index
    %k_plane = index.mul %layer, %c2 : index
    %k_off = func.call @g_dfl_kv_offset(%origins, %row, %position, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
    %pos_bytes = index.constant {POSB} : offset
    %plane_stride = index.scale %pool, %pos_bytes : index, offset -> offset
    %v_off = index.add %k_off, %plane_stride : offset
    %k_cache = buffer.view %cache[%k_off] : buffer -> view<{KVH}x{D}xf16>
    %v_cache = buffer.view %cache[%v_off] : buffer -> view<{KVH}x{D}xf16>
    %ha = scalar.fptrunc %ra : f32 to f16
    %hb = scalar.fptrunc %rb : f32 to f16
    view.store %ha, %k_cache[%kh, %lane] : f16, view<{KVH}x{D}xf16>
    view.store %hb, %k_cache[%kh, %lane2] : f16, view<{KVH}x{D}xf16>
    %va = view.load %v_view[%token, %kh, %lane] : view<[%tokens]x{KVH}x{D}xf32> -> f32
    %vb = view.load %v_view[%token, %kh, %lane2] : view<[%tokens]x{KVH}x{D}xf32> -> f32
    %vha = scalar.fptrunc %va : f32 to f16
    %vhb = scalar.fptrunc %vb : f32 to f16
    view.store %vha, %v_cache[%kh, %lane] : f16, view<{KVH}x{D}xf16>
    view.store %vhb, %v_cache[%kh, %lane2] : f16, view<{KVH}x{D}xf16>
  }}
  kernel.return
}}

// Block queries (in place) and block keys/values (F32 scratch): rmsnorm + RoPE at the token's position.
kernel.def export("g_dfl_q_prep") @g_dfl_q_prep(%tokens: index) {{
  %cq = index.constant {QH} : index
  %lanes = index.constant {HALF} : index
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%tokens, %cq, %c1) workgroup_size(%lanes, %c1, %c1) : index
}} launch(%tokens_arg: index, %q: buffer, %w: buffer, %info: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
{head_prelude}  %qh = index.assume %head_id [range(%head_id, 0, {QH - 1})] : index
  %q_view = buffer.view %q[%zero] : buffer -> view<[%tokens]x{QH}x{D}xf32>
""" + norm_rope("%q_view", "%qh", "%position32").replace("%NH", str(QH)) + f"""  view.store %ra, %q_view[%token, %qh, %lane] : f32, view<[%tokens]x{QH}x{D}xf32>
  view.store %rb, %q_view[%token, %qh, %lane2] : f32, view<[%tokens]x{QH}x{D}xf32>
  kernel.return
}}

kernel.def export("g_dfl_block_kv") @g_dfl_block_kv(%tokens: index) {{
  %ckv = index.constant {KVH} : index
  %lanes = index.constant {HALF} : index
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%tokens, %ckv, %c1) workgroup_size(%lanes, %c1, %c1) : index
}} launch(%tokens_arg: index, %k: buffer, %w: buffer, %info: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
{head_prelude}  %kh = index.assume %head_id [range(%head_id, 0, {KVH - 1})] : index
  %k_view = buffer.view %k[%zero] : buffer -> view<[%tokens]x{KVH}x{D}xf32>
""" + norm_rope("%k_view", "%kh", "%position32").replace("%NH", str(KVH)) + f"""  view.store %ra, %k_view[%token, %kh, %lane] : f32, view<[%tokens]x{KVH}x{D}xf32>
  view.store %rb, %k_view[%token, %kh, %lane2] : f32, view<[%tokens]x{KVH}x{D}xf32>
  kernel.return
}}
""")
    # ---- attention: one workgroup per (block token, query head); 8 waves split the keys, each lane owns D/32 dims
    DPL = D // 32
    vt = f"vector<{DPL}xf32>"
    win = (f"""  %window_tail = index.constant {W - 1} : index
  %earliest = index.max %absolute, %window_tail : index
  %ctx_start = index.sub %earliest, %window_tail : index""" if W else "  %ctx_start = index.constant 0 : index")
    T.append((f"""
// Draft attention for block token t (position P + j) and query head h: keys are the context K/V at positions
// [start, P) (start = 0, or the sliding window) plus every token of the block (non-causal). Online softmax
// per wave over its share of the keys, then a cross-wave merge in workgroup memory.
kernel.def export("g_dfl_attention") @g_dfl_attention(%tokens: index) {{
  %cq = index.constant {QH} : index
  %c256 = index.constant 256 : index
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%tokens, %cq, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %layer_arg: index, %context_arg: index, %pool_arg: index, %metadata: buffer, %info: buffer, %q: buffer, %bk: buffer, %bv: buffer, %origins: buffer, %cache: buffer, %out: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {F["layers"] - 1})] : index
  %context = index.assume %context_arg [range(%context_arg, 16, 262144)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 1048576)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [range(%token_id, 0, 2147483646), lt(%token_id, %tokens)] : index
  %head_id = kernel.workgroup.id<y> : index
  %qh = index.assume %head_id [range(%head_id, 0, {QH - 1})] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, 7)] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c16o = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1i = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c8 = index.constant 8 : index
  %c15 = index.constant 15 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c0_i32 = scalar.constant 0 : i32
  %zero_f = scalar.constant 0.0 : f32
  %neg = scalar.constant -3.0e38 : f32
  %scale = scalar.constant {D ** -0.5!r} : f32
  %zv = vector.constant 0.0 : {vt}
  %gq = index.constant {GQ} : index
  %kh = index.div %qh, %gq : index
  %khb = index.assume %kh [range(%kh, 0, {KVH - 1})] : index
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %row32 = view.load %info_view[%token, 0] : view<[%tokens]x4xi32> -> i32
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %span32 = view.load %info_view[%token, 3] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %acc_bytes = index.constant {8 * (D + 2) * 4} : offset
  %merge = buffer.alloca<workgroup> align(16) %acc_bytes : buffer
  %mv = buffer.view %merge[%zero] : buffer -> view<8x{D + 2}xf32>
  %dbase = index.mul %lane, %c1i : index
  scf.if %live {{
    %records = buffer.view %metadata[%c16o] : buffer -> view<[%spans]x9xi32>
    %span_b = scalar.assume %span32 [range(%span32, 0, 8191)] : i32
    %span_v = index.cast %span_b : i32 to index
    %span = index.assume %span_v [range(%span_v, 0, 8191), lt(%span_v, %spans)] : index
    %length32 = view.load %records[%span, 0] : view<[%spans]x9xi32> -> i32
    %pos32 = view.load %records[%span, 1] : view<[%spans]x9xi32> -> i32
    %cursor32 = view.load %records[%span, 4] : view<[%spans]x9xi32> -> i32
    %len_b = scalar.assume %length32 [range(%length32, 1, 64)] : i32
    %pos_b = scalar.assume %pos32 [range(%pos32, 0, 262143)] : i32
    %cur_b = scalar.assume %cursor32 [range(%cursor32, 0, 8191)] : i32
    %block_len = index.cast %len_b : i32 to index
    %block_pos = index.cast %pos_b : i32 to index
    %cursor = index.cast %cur_b : i32 to index
    %rel0 = index.sub %token, %cursor : index
    %absolute = index.add %block_pos, %rel0 : index
{win}
    %ctx_count = index.sub %block_pos, %ctx_start : index
    %keys = index.add %ctx_count, %block_len : index
    %row_b = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
    %row = index.cast %row_b : i32 to index
    %padded = index.add %context, %c15 : index
    %blocks = index.div %padded, %c16 : index
    %k_plane = index.mul %layer, %c2 : index
    %pos_bytes = index.constant {POSB} : offset
    %plane_stride = index.scale %pool, %pos_bytes : index, offset -> offset
    %q_view = buffer.view %q[%zero] : buffer -> view<[%tokens]x{QH}x{D}xf32>
    %bk_view = buffer.view %bk[%zero] : buffer -> view<[%tokens]x{KVH}x{D}xf32>
    %bv_view = buffer.view %bv[%zero] : buffer -> view<[%tokens]x{KVH}x{D}xf32>
    %d0 = index.mul %lane, %c1i : index
""" + "".join(f"""    %dd{i}c = index.constant {32 * i} : index
    %dd{i} = index.add %lane, %dd{i}c : index
    %qv{i} = view.load %q_view[%token, %qh, %dd{i}] : view<[%tokens]x{QH}x{D}xf32> -> f32
""" for i in range(DPL)) + f"""    %m_f, %l_f, {", ".join(f"%a_f{i}" for i in range(DPL))} = scf.for %key = [%wave to %keys step %c8](%m = %neg : f32, %l = %zero_f : f32, {", ".join(f"%a{i} = %zero_f : f32" for i in range(DPL))}) -> (f32, f32, {", ".join(["f32"] * DPL)}) {{
      %in_ctx = index.cmp ult, %key, %ctx_count : index
{"".join(f"      %kx{i}, %vx{i} = scf.if %in_ctx -> (f32, f32) {{" + chr(10) + ("        %kpos = index.add %ctx_start, %key : index" + chr(10) + "        %koff = func.call @g_dfl_kv_offset(%origins, %row, %kpos, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)" + chr(10) + "        %voff = index.add %koff, %plane_stride : offset" + chr(10) + f"        %kc = buffer.view %cache[%koff] : buffer -> view<{KVH}x{D}xf16>" + chr(10) + f"        %vc = buffer.view %cache[%voff] : buffer -> view<{KVH}x{D}xf16>" + chr(10)) + f"        %kh16 = view.load %kc[%khb, %dd{i}] : view<{KVH}x{D}xf16> -> f16" + chr(10) + f"        %vh16 = view.load %vc[%khb, %dd{i}] : view<{KVH}x{D}xf16> -> f16" + chr(10) + "        %kf = scalar.extf %kh16 : f16 to f32" + chr(10) + "        %vf = scalar.extf %vh16 : f16 to f32" + chr(10) + "        scf.yield %kf, %vf : f32, f32" + chr(10) + "      } else {" + chr(10) + "        %bj = index.sub %key, %ctx_count : index" + chr(10) + "        %btok0 = index.add %cursor, %bj : index" + chr(10) + "        %btok = index.assume %btok0 [range(%btok0, 0, 8191), lt(%btok0, %tokens)] : index" + chr(10) + f"        %kf = view.load %bk_view[%btok, %khb, %dd{i}] : view<[%tokens]x{KVH}x{D}xf32> -> f32" + chr(10) + f"        %vf = view.load %bv_view[%btok, %khb, %dd{i}] : view<[%tokens]x{KVH}x{D}xf32> -> f32" + chr(10) + "        scf.yield %kf, %vf : f32, f32" + chr(10) + "      }" + chr(10) for i in range(DPL))}""" + "".join(f"      %p{i} = scalar.mulf %qv{i}, %kx{i} : f32\n" for i in range(DPL)) + "      %dot0 = scalar.addf %p0, %zero_f : f32\n" + "".join(f"      %dot{i} = scalar.addf %dot{i - 1}, %p{i} : f32\n" for i in range(1, DPL)) + f"""      %dsum = kernel.subgroup.reduce<addf> %dot{DPL - 1} : f32
      %s = scalar.mulf %dsum, %scale : f32
      %mn = scalar.maxnumf %m, %s : f32
      %dm = scalar.subf %m, %mn : f32
      %corr = scalar.expf<afn> %dm : f32
      %ds = scalar.subf %s, %mn : f32
      %pw = scalar.expf<afn> %ds : f32
      %lc = scalar.mulf %l, %corr : f32
      %ln = scalar.addf %lc, %pw : f32
""" + "".join(f"""      %ac{i} = scalar.mulf %a{i}, %corr : f32
      %an{i} = scalar.fmaf %pw, %vx{i}, %ac{i} : f32
""" for i in range(DPL)) + f"""      scf.yield %mn, %ln, {", ".join(f"%an{i}" for i in range(DPL))} : f32, f32, {", ".join(["f32"] * DPL)}
    }}
    %first_lane = index.cmp eq, %lane, %c0 : index
""" + "".join(f"""    view.store %a_f{i}, %mv[%wave, %dd{i}] : f32, view<8x{D + 2}xf32>
""" for i in range(DPL)) + f"""    %dm_slot = index.constant {D} : index
    %dl_slot = index.constant {D + 1} : index
    scf.if %first_lane {{
      view.store %m_f, %mv[%wave, %dm_slot] : f32, view<8x{D + 2}xf32>
      view.store %l_f, %mv[%wave, %dl_slot] : f32, view<8x{D + 2}xf32>
    }}
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %leader_wave = index.cmp eq, %wave, %c0 : index
  %write = scalar.andi %live, %leader_wave : i1
  scf.if %write {{
    %dm_slot = index.constant {D} : index
    %dl_slot = index.constant {D + 1} : index
    %gm = scf.for %w = [%c0 to %c8 step %c1i](%mx = %neg : f32) -> (f32) {{
      %wb = index.assume %w [range(%w, 0, 7)] : index
      %mw = view.load %mv[%wb, %dm_slot] : view<8x{D + 2}xf32> -> f32
      %n = scalar.maxnumf %mx, %mw : f32
      scf.yield %n : f32
    }}
    %o_view = buffer.view %out[%zero] : buffer -> view<[%tokens]x{QH}x{D}xf32>
""" + "".join(f"""    %dd{i}c = index.constant {32 * i} : index
    %dd{i} = index.add %lane, %dd{i}c : index
""" for i in range(DPL)) + f"""    %tl, {", ".join(f"%ta{i}" for i in range(DPL))} = scf.for %w = [%c0 to %c8 step %c1i](%l = %zero_f : f32, {", ".join(f"%a{i} = %zero_f : f32" for i in range(DPL))}) -> (f32, {", ".join(["f32"] * DPL)}) {{
      %wb = index.assume %w [range(%w, 0, 7)] : index
      %mw = view.load %mv[%wb, %dm_slot] : view<8x{D + 2}xf32> -> f32
      %lw = view.load %mv[%wb, %dl_slot] : view<8x{D + 2}xf32> -> f32
      %dmw = scalar.subf %mw, %gm : f32
      %fw = scalar.expf<afn> %dmw : f32
      %ln = scalar.fmaf %lw, %fw, %l : f32
""" + "".join(f"""      %aw{i} = view.load %mv[%wb, %dd{i}] : view<8x{D + 2}xf32> -> f32
      %an{i} = scalar.fmaf %aw{i}, %fw, %a{i} : f32
""" for i in range(DPL)) + f"""      scf.yield %ln, {", ".join(f"%an{i}" for i in range(DPL))} : f32, {", ".join(["f32"] * DPL)}
    }}
    %inv_l = scalar.divf %one_f, %tl : f32
""" + "".join(f"""    %res{i} = scalar.mulf %ta{i}, %inv_l : f32
    view.store %res{i}, %o_view[%token, %qh, %dd{i}] : f32, view<[%tokens]x{QH}x{D}xf32>
""" for i in range(DPL)) + """  }
  kernel.return
}
""").replace("  %zero_f = scalar.constant 0.0 : f32\n  %neg", "  %zero_f = scalar.constant 0.0 : f32\n  %one_f = scalar.constant 1.0 : f32\n  %neg", 1))
    # ---- residual add + rmsnorm (draft layers; no post norms)
    T.append(f"""
// x[t] += y[t]; h[t] = rmsnorm(x[t]) * w.
kernel.def export("g_dfl_add_norm") @g_dfl_add_norm(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %add: i32, %y: buffer, %x: buffer, %w: buffer, %h: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [range(%token_id, 0, 2147483646), lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c0_i32 = scalar.constant 0 : i32
  %groups = index.constant {HG} : index
  %zero_f = scalar.constant 0.0 : f32
  %zero32 = vector.constant 0.0 : vector<32xf32>
  %count = scalar.constant {float(H)} : f32
  %adds = scalar.cmpi ne, %add, %c0_i32 : i32
  %active = index.cmp ult, %lane, %groups : index
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
  %y_view = buffer.view %y[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
  %xv = scf.if %active -> (vector<32xf32>) {{
    %bl = index.assume %lane [range(%lane, 0, {HG - 1})] : index
    %a = vector.load %x_view[%token, %bl, %c0] : view<[%tokens]x{HG}x32xf32> -> vector<32xf32>
    %s = scf.if %adds -> (vector<32xf32>) {{
      %b = vector.load %y_view[%token, %bl, %c0] : view<[%tokens]x{HG}x32xf32> -> vector<32xf32>
      %sum = vector.addf %a, %b : vector<32xf32>
      vector.store %sum, %x_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x{HG}x32xf32>
      scf.yield %sum : vector<32xf32>
    }} else {{
      scf.yield %a : vector<32xf32>
    }}
    scf.yield %s : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}
  %sq = vector.mulf %xv, %xv : vector<32xf32>
  %local = vector.reduce<addf> %sq, %zero_f : vector<32xf32>, f32
  %sum2 = kernel.workgroup.reduce<addf> %local : f32
  %mean = scalar.divf %sum2, %count : f32
  %eps = scalar.constant {EPS!r} : f32
  %stable = scalar.addf %mean, %eps : f32
  %inv = scalar.rsqrtf %stable : f32
  %invs = vector.splat %inv : vector<32xf32>
  scf.if %active {{
    %bl = index.assume %lane [range(%lane, 0, {HG - 1})] : index
    %w_view = buffer.view %w[%zero] : buffer -> view<{HG}x32xf32>
    %h_view = buffer.view %h[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
    %wv = vector.load %w_view[%bl, %c0] : view<{HG}x32xf32> -> vector<32xf32>
    %n = vector.mulf %xv, %invs : vector<32xf32>
    %o = vector.mulf %n, %wv : vector<32xf32>
    vector.store %o, %h_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x{HG}x32xf32>
  }}
  kernel.return
}}
""")
    if F["conv"]:
        K, GW = F["conv"]["taps"], F["conv"]["group"]
        NGR = H // GW
        C = 2 * K * NGR
        T.append(f"""
// DFlash2 grouped causal convolution inside each block: out[t][c] = sum_tap (base[branch][tap][c] +
// dyn[t][branch][tap][c / {GW}]) * in[t - tap][c] for taps reaching back to the block's first token.
// add != 0: residual[t] += out[t] (branch 1); else out is written to `output`.
kernel.def export("g_dfl_conv") @g_dfl_conv(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %branch_arg: index, %add: i32, %metadata: buffer, %info: buffer, %input: buffer, %dyn: buffer, %base: buffer, %output: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %branch = index.assume %branch_arg [range(%branch_arg, 0, 1)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [range(%token_id, 0, 2147483646), lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c16o = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1i = index.constant 1 : index
  %c4 = index.constant 4 : index
  %c0_i32 = scalar.constant 0 : i32
  %adds = scalar.cmpi ne, %add, %c0_i32 : i32
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  // Tokens past the proposing blocks (a cohort stage holds a power of two of blocks; fewer may propose) are not
  // live: their span is -1. They read span 0 and convolve nothing, so every access stays in bounds (an
  // unguarded span -1 read the metadata far out of range: DATA_LOSS on elastic backing under concurrency).
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %span_raw = view.load %info_view[%token, 3] : view<[%tokens]x4xi32> -> i32
  %span32 = scf.select %live, %span_raw, %c0_i32 : i32
  %span_b = scalar.assume %span32 [range(%span32, 0, 8191)] : i32
  %span_v = index.cast %span_b : i32 to index
  %span = index.assume %span_v [range(%span_v, 0, 8191), lt(%span_v, %spans)] : index
  %records = buffer.view %metadata[%c16o] : buffer -> view<[%spans]x9xi32>
  %cursor32 = view.load %records[%span, 4] : view<[%spans]x9xi32> -> i32
  %cur_b = scalar.assume %cursor32 [range(%cursor32, 0, 8191)] : i32
  %cursor = index.cast %cur_b : i32 to index
  %rel = index.sub %token, %cursor : index
  %reach = index.add %rel, %c1i : index
  %taps_c = index.constant {K} : index
  %taps_all = index.min %reach, %taps_c : index
  %taps = scf.select %live, %taps_all, %c0 : index
  %x_view = buffer.view %input[%zero] : buffer -> view<[%tokens]x{H}xf32>
  %d_view = buffer.view %dyn[%zero] : buffer -> view<[%tokens]x2x{K}x{NGR}xf32>
  %b_view = buffer.view %base[%zero] : buffer -> view<2x{K}x{H}xf32>
  %o_view = buffer.view %output[%zero] : buffer -> view<[%tokens]x{H}xf32>
  %cH = index.constant {H} : index
  %c1024 = index.constant 1024 : index
  %gw = index.constant {GW} : index
  %first = index.mul %lane, %c4 : index
  %zv = vector.constant 0.0 : vector<4xf32>
  scf.for %ch0 = [%first to %cH step %c1024] {{
    %ch = index.assume %ch0 [range(%ch0, 0, {H - 4}), multiple_of(%ch0, 4)] : index
    %group0 = index.div %ch, %gw : index
    %group = index.assume %group0 [range(%group0, 0, {NGR - 1})] : index
    %acc = scf.for %tap = [%c0 to %taps step %c1i](%s = %zv : vector<4xf32>) -> (vector<4xf32>) {{
      %src0 = index.sub %token, %tap : index
      %src = index.assume %src0 [range(%src0, 0, 8191), lt(%src0, %tokens)] : index
      %tb = index.assume %tap [range(%tap, 0, {K - 1})] : index
      %xv = vector.load %x_view[%src, %ch] : view<[%tokens]x{H}xf32> -> vector<4xf32>
      %wv = vector.load %b_view[%branch, %tb, %ch] : view<2x{K}x{H}xf32> -> vector<4xf32>
      %cf = view.load %d_view[%token, %branch, %tb, %group] : view<[%tokens]x2x{K}x{NGR}xf32> -> f32
      %cv = vector.splat %cf : vector<4xf32>
      %coef = vector.addf %wv, %cv : vector<4xf32>
      %n = vector.fmaf %coef, %xv, %s : vector<4xf32>
      scf.yield %n : vector<4xf32>
    }}
    %res = scf.if %adds -> (vector<4xf32>) {{
      %old = vector.load %o_view[%token, %ch] : view<[%tokens]x{H}xf32> -> vector<4xf32>
      %sum = vector.addf %old, %acc : vector<4xf32>
      scf.yield %sum : vector<4xf32>
    }} else {{
      scf.yield %acc : vector<4xf32>
    }}
    vector.store %res, %o_view[%token, %ch] : vector<4xf32>, view<[%tokens]x{H}xf32>
  }}
  kernel.return
}}
""")
    # ---- proposal batch / publication
    depth_rows = "%depth"
    T.append(f"""
// Draft blocks for the PROPOSE spans: block b = [anchor, <mask> x {BS - 1}] at positions P..P+{BS - 1}
// (packed tokens b*{BS}..), its verifier slot, and head records for rows 1..depth (record length j + 2 reads
// block token j + 1).
kernel.def export("g_dfl_prepare") @g_dfl_prepare() {{
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
}} launch(%spans_arg: index, %blocks_arg: index, %tokens_arg: index, %depth_arg: index, %metadata: buffer, %ids: buffer, %draft: buffer, %draft_ids: buffer, %slots: buffer, %rows_meta: buffer) {{
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %blocks = index.assume %blocks_arg [range(%blocks_arg, 1, 1024)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %depth = index.assume %depth_arg [range(%depth_arg, 1, 7)] : index
  %zero = index.constant 0 : offset
  %c16 = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1i = index.constant 1 : index
  %c8 = index.constant 8 : index
  %c9 = index.constant 9 : index
  // block length: anchor + depth masks, at most the trained block size (llama.cpp's n_max + 1)
  %bs_full = index.constant {BS} : index
  %width_b = index.add %depth, %c1i : index
  %bs = index.min %bs_full, %width_b : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %c2_i32 = scalar.constant 2 : i32
  %bs_i32 = index.cast %bs : index to i32
  %mask = scalar.constant {F["mask_id"]} : i32
  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %eos = view.load %header[1] : view<4xi32> -> i32
  %mixed_word = view.load %header[2] : view<4xi32> -> i32
  %mixed = scalar.cmpi ne, %mixed_word, %c0_i32 : i32
  %id_stride = scf.select %mixed, %c8, %c1i : index
  %n_ids = index.mul %tokens, %id_stride : index
  %ids_flat = buffer.view %ids[%zero] : buffer -> view<[%n_ids]xi32>
  %count_b = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count_v = index.cast %count_b : i32 to index
  %count = index.min %count_v, %spans : index
  %records = buffer.view %metadata[%c16] : buffer -> view<[%spans]x9xi32>
  %out = buffer.view %draft[%c16] : buffer -> view<[%blocks]x9xi32>
  %dt = index.mul %blocks, %bs : index
  %dids = buffer.view %draft_ids[%zero] : buffer -> view<[%dt]xi32>
  %slot_v = buffer.view %slots[%zero] : buffer -> view<[%blocks]xi32>
  %row_cap = index.mul %blocks, %depth : index
  %rows = buffer.view %rows_meta[%c16] : buffer -> view<[%row_cap]x9xi32>
  scf.for %i = [%c0 to %dt step %c1i] {{
    %ib = index.assume %i [range(%i, 0, 8191), lt(%i, %dt)] : index
    view.store %mask, %dids[%ib] : i32, view<[%dt]xi32>
  }}
  %total = scf.for %s = [%c0 to %count step %c1i](%k = %c0 : index) -> (index) {{
    %sb = index.assume %s [range(%s, 0, 8191), lt(%s, %spans)] : index
    %flags = view.load %records[%sb, 6] : view<[%spans]x9xi32> -> i32
    %prop_bits = scalar.andi %flags, %c2_i32 : i32
    %proposes = scalar.cmpi ne, %prop_bits, %c0_i32 : i32
    %fits = index.cmp ult, %k, %blocks : index
    %take = scalar.andi %proposes, %fits : i1
    scf.if %take {{
      %kb = index.assume %k [range(%k, 0, 1023), lt(%k, %blocks)] : index
      scf.for %w = [%c0 to %c9 step %c1i] {{
        %wb = index.assume %w [range(%w, 0, 8)] : index
        %v = view.load %records[%sb, %wb] : view<[%spans]x9xi32> -> i32
        view.store %v, %out[%kb, %wb] : i32, view<[%blocks]x9xi32>
      }}
      %base = index.mul %kb, %bs : index
      %base32 = index.cast %base : index to i32
      view.store %bs_i32, %out[%kb, 0] : i32, view<[%blocks]x9xi32>
      view.store %c0_i32, %out[%kb, 2] : i32, view<[%blocks]x9xi32>
      view.store %base32, %out[%kb, 4] : i32, view<[%blocks]x9xi32>
      view.store %c1_i32, %out[%kb, 6] : i32, view<[%blocks]x9xi32>
      %cursor32 = view.load %records[%sb, 4] : view<[%spans]x9xi32> -> i32
      %cursor_b = scalar.assume %cursor32 [range(%cursor32, 0, 8191)] : i32
      %cursor = index.cast %cursor_b : i32 to index
      %ae0 = index.mul %cursor, %id_stride : index
      %ae = index.assume %ae0 [range(%ae0, 0, 65535), lt(%ae0, %n_ids)] : index
      %anchor = view.load %ids_flat[%ae] : view<[%n_ids]xi32> -> i32
      %bb = index.assume %base [range(%base, 0, 8191), lt(%base, %dt)] : index
      view.store %anchor, %dids[%bb] : i32, view<[%dt]xi32>
      view.store %cursor32, %slot_v[%kb] : i32, view<[%blocks]xi32>{"" if not MR else chr(10) + """      %sc = index.assume %s [range(%s, 0, 8191), lt(%s, %count)] : index
      %c36m = index.constant 36 : offset
      %m_plan = index.scale %count, %c36m : index, offset -> offset
      %m_table = index.add %c16, %m_plan : offset
      %m_tv = buffer.view %metadata[%m_table] : buffer -> view<[%count]xi32>
      %delta = view.load %m_tv[%sc] : view<[%count]xi32> -> i32
      %d_plan = index.scale %blocks, %c36m : index, offset -> offset
      %d_table = index.add %c16, %d_plan : offset
      %d_tv = buffer.view %draft[%d_table] : buffer -> view<[%blocks]xi32>
      view.store %delta, %d_tv[%kb] : i32, view<[%blocks]xi32>"""}
      scf.for %j = [%c0 to %depth step %c1i] {{
        %r0 = index.mul %kb, %depth : index
        %r1 = index.add %r0, %j : index
        %rb = index.assume %r1 [range(%r1, 0, 8191), lt(%r1, %row_cap)] : index
        scf.for %w = [%c0 to %c9 step %c1i] {{
          %wb = index.assume %w [range(%w, 0, 8)] : index
          %v = view.load %out[%kb, %wb] : view<[%blocks]x9xi32> -> i32
          view.store %v, %rows[%rb, %wb] : i32, view<[%row_cap]x9xi32>
        }}
        %j32 = index.cast %j : index to i32
        %two = scalar.constant 2 : i32
        %len = scalar.addi %j32, %two : i32
        view.store %len, %rows[%rb, 0] : i32, view<[%row_cap]x9xi32>
      }}
    }}
    %inc = scf.select %take, %c1i, %c0 : index
    %kn = index.add %k, %inc : index
    scf.yield %kn : index
  }}
  %total32 = index.cast %total : index to i32
  %rows_total = index.mul %total, %depth : index
  %rows32 = index.cast %rows_total : index to i32
  %dh = buffer.view %draft[%zero] : buffer -> view<4xi32>
  view.store %total32, %dh[0] : i32, view<4xi32>
  view.store %eos, %dh[1] : i32, view<4xi32>
  view.store %c0_i32, %dh[2] : i32, view<4xi32>
  view.store %c0_i32, %dh[3] : i32, view<4xi32>
  %rh = buffer.view %rows_meta[%zero] : buffer -> view<4xi32>
  view.store %rows32, %rh[0] : i32, view<4xi32>
  view.store %eos, %rh[1] : i32, view<4xi32>
  view.store %c0_i32, %rh[2] : i32, view<4xi32>
  view.store %c0_i32, %rh[3] : i32, view<4xi32>
  kernel.return
}}

// proposals[b * depth + j] -> verifier slot (cursor + 1 + j) of block b.
kernel.def export("g_dfl_publish") @g_dfl_publish() {{
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
}} launch(%blocks_arg: index, %tokens_arg: index, %depth_arg: index, %metadata: buffer, %draft: buffer, %slots: buffer, %proposals: buffer, %ids: buffer) {{
  %blocks = index.assume %blocks_arg [range(%blocks_arg, 1, 1024)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %depth = index.assume %depth_arg [range(%depth_arg, 1, 7)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1i = index.constant 1 : index
  %c8 = index.constant 8 : index
  %c0_i32 = scalar.constant 0 : i32
  %mheader = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %mixed_word = view.load %mheader[2] : view<4xi32> -> i32
  %mixed = scalar.cmpi ne, %mixed_word, %c0_i32 : i32
  %id_stride = scf.select %mixed, %c8, %c1i : index
  %n_ids = index.mul %tokens, %id_stride : index
  %ids_flat = buffer.view %ids[%zero] : buffer -> view<[%n_ids]xi32>
  %dh = buffer.view %draft[%zero] : buffer -> view<4xi32>
  %count32 = view.load %dh[0] : view<4xi32> -> i32
  %count_b = scalar.assume %count32 [range(%count32, 0, 1024)] : i32
  %count_v = index.cast %count_b : i32 to index
  %count = index.min %count_v, %blocks : index
  %slot_v = buffer.view %slots[%zero] : buffer -> view<[%blocks]xi32>
  %row_cap = index.mul %blocks, %depth : index
  %prop = buffer.view %proposals[%zero] : buffer -> view<[%row_cap]xi32>
  scf.for %b = [%c0 to %count step %c1i] {{
    %bb = index.assume %b [range(%b, 0, 1023), lt(%b, %blocks)] : index
    %slot32 = view.load %slot_v[%bb] : view<[%blocks]xi32> -> i32
    %slot_b = scalar.assume %slot32 [range(%slot32, 0, 8191)] : i32
    %slot = index.cast %slot_b : i32 to index
    scf.for %j = [%c0 to %depth step %c1i] {{
      %r0 = index.mul %bb, %depth : index
      %r1 = index.add %r0, %j : index
      %rb = index.assume %r1 [range(%r1, 0, 8191), lt(%r1, %row_cap)] : index
      %token = view.load %prop[%rb] : view<[%row_cap]xi32> -> i32
      %t0 = index.add %slot, %j : index
      %t1 = index.add %t0, %c1i : index
      %e0 = index.mul %t1, %id_stride : index
      %e = index.assume %e0 [range(%e0, 0, 65535), lt(%e0, %n_ids)] : index
      view.store %token, %ids_flat[%e] : i32, view<[%n_ids]xi32>
    }}
  }}
  kernel.return
}}
""")
    if F["selector"]:
        T.append(selector_kernels(g))
    return "\n".join(T)


def selector_kernels(g):
    """DFlash2 top-k candidates (from the head's chunk maxima + logits) and the low-rank transition chain."""
    F = g["dflash"]
    K = F["selector"]["top_k"]
    R = F["selector"]["rank"]
    CH, V = g["CHUNKS"], g["V"]
    assert K == 16 and R % 32 == 0 and R <= 256
    RG = R // 32                       # 32-value groups per selector row
    scale = F["logit_scale"] or 1.0
    cap = F["softcap"]
    sfmt = F["tensors"]["selector_predecessor.weight"]["type"]
    unary = f"""      %zs = scalar.mulf %lg, %lscale : f32""" + (f"""
      %capv = scalar.constant {float(cap)!r} : f32
      %zd = scalar.divf %zs, %capv : f32
      %zt = scalar.tanhf<afn> %zd : f32
      %zu = scalar.mulf %zt, %capv : f32""" if cap else """
      %zu = scalar.addf %zs, %zero_f : f32""")
    return f"""
// Top-{K} candidates per row: the top-{K} vocabulary entries lie in the {K} chunks with the largest maxima, so
// pick those chunks, then the top-{K} of their 128-logit slices. Scores get the drafter's logit transform.
kernel.def export("g_dfl_topk") @g_dfl_topk(%rows: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%rows, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%rows_arg: index, %rows_meta: buffer, %partials: buffer, %logits: buffer, %ids: buffer, %scores: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 8192)] : index
  %row_id = kernel.workgroup.id<x> : index
  %row = index.assume %row_id [range(%row_id, 0, 2147483646), lt(%row_id, %rows)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1i = index.constant 1 : index
  %c128 = index.constant 128 : index
  %c256 = index.constant 256 : index
  %ck = index.constant {K} : index
  %chunks = index.constant {CH} : index
  %c0_i32 = scalar.constant 0 : i32
  %big = scalar.constant 2147483647 : i32
  %neg = scalar.constant -3.0e38 : f32
  %zero_f = scalar.constant 0.0 : f32
  %lscale = scalar.constant {float(scale)!r} : f32
  %header = buffer.view %rows_meta[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %count_b = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count = index.cast %count_b : i32 to index
  %live = index.cmp ult, %row, %count : index
  %sel_bytes = index.constant {K * 4 * 2 + K * 128 * 4} : offset
  %sel = buffer.alloca<workgroup> align(16) %sel_bytes : buffer
  %chosen = buffer.view %sel[%zero] : buffer -> view<{K}xi32>
  %c64o = index.constant {K * 4} : offset
  %pool_ids_off = index.constant {K * 8} : offset
  %slice = buffer.view %sel[%pool_ids_off] : buffer -> view<{K * 128}xf32>
  scf.if %live {{
    %pv = buffer.view %partials[%zero] : buffer -> view<[%rows]x{CH}x2xi32>
    %lv = buffer.view %logits[%zero] : buffer -> view<[%rows]x{CH * 128}xf32>
    %iv = buffer.view %ids[%zero] : buffer -> view<[%rows]x{K}xi32>
    %sv = buffer.view %scores[%zero] : buffer -> view<[%rows]x{K}xf32>
    %first = index.cmp eq, %lane, %c0 : index
    // 1. the K chunks with the largest maxima (ties: lowest chunk)
    scf.for %r = [%c0 to %ck step %c1i] {{
      %rb = index.assume %r [range(%r, 0, {K - 1})] : index
      %bv, %bi = scf.for %c = [%lane to %chunks step %c256](%v = %neg : f32, %i = %big : i32) -> (f32, i32) {{
        %cb = index.assume %c [range(%c, 0, {CH - 1})] : index
        %bits = view.load %pv[%row, %cb, %c0] : view<[%rows]x{CH}x2xi32> -> i32
        %val = scalar.bitcast %bits : i32 to f32
        %c32 = index.cast %cb : index to i32
        %taken = scf.for %p = [%c0 to %rb step %c1i](%t = %c0_i32 : i32) -> (i32) {{
          %pb = index.assume %p [range(%p, 0, {K - 1})] : index
          %pc = view.load %chosen[%pb] : view<{K}xi32> -> i32
          %same = scalar.cmpi eq, %pc, %c32 : i32
          %one = scalar.constant 1 : i32
          %nt = scf.select %same, %one, %t : i32
          scf.yield %nt : i32
        }}
        %free = scalar.cmpi eq, %taken, %c0_i32 : i32
        %better = scalar.cmpf ogt, %val, %v : f32
        %use = scalar.andi %free, %better : i1
        %nv = scf.select %use, %val, %v : f32
        %ni = scf.select %use, %c32, %i : i32
        scf.yield %nv, %ni : f32, i32
      }}
      %best = kernel.workgroup.reduce<maxnumf> %bv : f32
      %is_best = scalar.cmpf oeq, %bv, %best : f32
      %cand = scf.select %is_best, %bi, %big : i32
      %win = kernel.workgroup.reduce<minsi> %cand : i32
      scf.if %first {{
        view.store %win, %chosen[%rb] : i32, view<{K}xi32>
      }}
      kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    }}
    // 2. stage their logits
    %slice_n = index.constant {K * 128} : index
    scf.for %e = [%lane to %slice_n step %c256] {{
      %eb = index.assume %e [range(%e, 0, {K * 128 - 1})] : index
      %which = index.div %eb, %c128 : index
      %within = index.rem %eb, %c128 : index
      %wb = index.assume %which [range(%which, 0, {K - 1})] : index
      %ch32 = view.load %chosen[%wb] : view<{K}xi32> -> i32
      %ch_b = scalar.assume %ch32 [range(%ch32, 0, {CH - 1})] : i32
      %chi = index.cast %ch_b : i32 to index
      %col0 = index.mul %chi, %c128 : index
      %col1 = index.add %col0, %within : index
      %col = index.assume %col1 [range(%col1, 0, {CH * 128 - 1})] : index
      %vocab_c = index.constant {V} : index
      %inv = index.cmp ult, %col, %vocab_c : index
      %lg0 = view.load %lv[%row, %col] : view<[%rows]x{CH * 128}xf32> -> f32
      %lg = scf.select %inv, %lg0, %neg : f32
      view.store %lg, %slice[%eb] : f32, view<{K * 128}xf32>
    }}
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    // 3. top-K of the staged logits (ties: lowest vocabulary id)
    scf.for %r = [%c0 to %ck step %c1i] {{
      %rb = index.assume %r [range(%r, 0, {K - 1})] : index
      %bv, %bi = scf.for %e = [%lane to %slice_n step %c256](%v = %neg : f32, %i = %big : i32) -> (f32, i32) {{
        %eb = index.assume %e [range(%e, 0, {K * 128 - 1})] : index
        %val = view.load %slice[%eb] : view<{K * 128}xf32> -> f32
        %which = index.div %eb, %c128 : index
        %within = index.rem %eb, %c128 : index
        %wb = index.assume %which [range(%which, 0, {K - 1})] : index
        %ch32 = view.load %chosen[%wb] : view<{K}xi32> -> i32
        %w32 = index.cast %within : index to i32
        %c128_i32 = scalar.constant 128 : i32
        %base = scalar.muli %ch32, %c128_i32 : i32
        %id = scalar.addi %base, %w32 : i32
        %better = scalar.cmpf ogt, %val, %v : f32
        %eq = scalar.cmpf oeq, %val, %v : f32
        %lower = scalar.cmpi slt, %id, %i : i32
        %tie = scalar.andi %eq, %lower : i1
        %use = scalar.ori %better, %tie : i1
        %nv = scf.select %use, %val, %v : f32
        %ni = scf.select %use, %id, %i : i32
        scf.yield %nv, %ni : f32, i32
      }}
      %best = kernel.workgroup.reduce<maxnumf> %bv : f32
      %is_best = scalar.cmpf oeq, %bv, %best : f32
      %cand = scf.select %is_best, %bi, %big : i32
      %win = kernel.workgroup.reduce<minsi> %cand : i32
      // remove the winner from the pool
      %c128_i32 = scalar.constant 128 : i32
      scf.for %e = [%lane to %slice_n step %c256] {{
        %eb = index.assume %e [range(%e, 0, {K * 128 - 1})] : index
        %which = index.div %eb, %c128 : index
        %within = index.rem %eb, %c128 : index
        %wb = index.assume %which [range(%which, 0, {K - 1})] : index
        %ch32 = view.load %chosen[%wb] : view<{K}xi32> -> i32
        %w32 = index.cast %within : index to i32
        %base = scalar.muli %ch32, %c128_i32 : i32
        %id = scalar.addi %base, %w32 : i32
        %hit = scalar.cmpi eq, %id, %win : i32
        scf.if %hit {{
          view.store %neg, %slice[%eb] : f32, view<{K * 128}xf32>
        }}
      }}
      scf.if %first {{
        %lg = scalar.addf %best, %zero_f : f32
{unary}
        view.store %win, %iv[%row, %rb] : i32, view<[%rows]x{K}xi32>
        view.store %zu, %sv[%row, %rb] : f32, view<[%rows]x{K}xf32>
      }}
      kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    }}
  }}
  kernel.return
}}

// Greedy chain per block: score(c) = unary(c) + sum_r pred[prev][r] * hidden[r] * succ[c][r]; prev starts
// at the anchor and becomes each position's winner. 16 candidates x 8 lanes (each a 32-value group).
kernel.def export("g_dfl_chain") @g_dfl_chain(%blocks: index) {{
  %c1 = index.constant 1 : index
  %c128 = index.constant 128 : index
  kernel.launch.config workgroups(%blocks, %c1, %c1) workgroup_size(%c128, %c1, %c1) : index
}} launch(%blocks_arg: index, %depth_arg: index, %tokens_arg: index, %block_tokens: index, %draft: buffer, %draft_ids: buffer, %ids: buffer, %scores: buffer, %hidden: buffer, %pred: buffer, %succ: buffer, %out: buffer) {{
  %blocks = index.assume %blocks_arg [range(%blocks_arg, 1, 1024)] : index
  %depth = index.assume %depth_arg [range(%depth_arg, 1, 7)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %block_id = kernel.workgroup.id<x> : index
  %block = index.assume %block_id [range(%block_id, 0, 2147483646), lt(%block_id, %blocks)] : index
  %thread_id = kernel.workitem.id<x> : index
  %thread = index.assume %thread_id [range(%thread_id, 0, 127)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1i = index.constant 1 : index
  %c8 = index.constant 8 : index
  %c0_i32 = scalar.constant 0 : i32
  %big = scalar.constant 2147483647 : i32
  %zero_f = scalar.constant 0.0 : f32
  %zv = vector.constant 0.0 : vector<32xf32>
  %dh = buffer.view %draft[%zero] : buffer -> view<4xi32>
  %count32 = view.load %dh[0] : view<4xi32> -> i32
  %count_b = scalar.assume %count32 [range(%count32, 0, 1024)] : i32
  %count = index.cast %count_b : i32 to index
  %live = index.cmp ult, %block, %count : index
  scf.if %live {{
    %cand = index.div %thread, %c8 : index
    %cb = index.assume %cand [range(%cand, 0, 15)] : index
    %part = index.rem %thread, %c8 : index
    %group_c = index.constant {RG} : index
    %has_group = index.cmp ult, %part, %group_c : index
    %pg = index.assume %part [range(%part, 0, 7)] : index
    %bsz = index.mul %block, %block_tokens : index
    %dt = index.mul %blocks, %block_tokens : index
    %dids = buffer.view %draft_ids[%zero] : buffer -> view<[%dt]xi32>
    %bb = index.assume %bsz [range(%bsz, 0, 8191), lt(%bsz, %dt)] : index
    %anchor = view.load %dids[%bb] : view<[%dt]xi32> -> i32
    %row_cap = index.mul %blocks, %depth : index
    %iv = buffer.view %ids[%zero] : buffer -> view<[%row_cap]x16xi32>
    %sv = buffer.view %scores[%zero] : buffer -> view<[%row_cap]x16xf32>
    %hv = buffer.view %hidden[%zero] : buffer -> view<[%tokens]x{RG}x32xf32>
    %ov = buffer.view %out[%zero] : buffer -> view<[%row_cap]xi32>
    %row_bytes = index.constant {g["row_bytes"](sfmt, R)} : offset
    %block_bytes = index.constant {g["FMT"][sfmt]} : offset
    %gpb = index.constant {g["SB"][sfmt] // 32} : index
    %blk = index.div %pg, %gpb : index
    %within = index.rem %pg, %gpb : index
    %blk_off = index.scale %blk, %block_bytes : index, offset -> offset
    %last = scf.for %j = [%c0 to %depth step %c1i](%prev = %anchor : i32) -> (i32) {{
      %r0 = index.mul %block, %depth : index
      %r1 = index.add %r0, %j : index
      %rb = index.assume %r1 [range(%r1, 0, 8191), lt(%r1, %row_cap)] : index
      %id = view.load %iv[%rb, %cb] : view<[%row_cap]x16xi32> -> i32
      %unary = view.load %sv[%rb, %cb] : view<[%row_cap]x16xf32> -> f32
      %tok0 = index.add %bsz, %j : index
      %tok1 = index.add %tok0, %c1i : index
      %tok = index.assume %tok1 [range(%tok1, 0, 8191), lt(%tok1, %tokens)] : index
      %prod = scf.if %has_group -> (f32) {{
        %prev_b = scalar.assume %prev [range(%prev, 0, {V - 1})] : i32
        %id_b = scalar.assume %id [range(%id, 0, {V - 1})] : i32
        %pi = index.cast %prev_b : i32 to index
        %ci = index.cast %id_b : i32 to index
        %po = index.scale %pi, %row_bytes : index, offset -> offset
        %co = index.scale %ci, %row_bytes : index, offset -> offset
        %pa = index.add %po, %blk_off : offset
        %ca = index.add %co, %blk_off : offset
        %pvv = func.call @g_{sfmt}_values(%pred, %pa, %within) : (buffer, offset, index) -> (vector<32xf32>)
        %cvv = func.call @g_{sfmt}_values(%succ, %ca, %within) : (buffer, offset, index) -> (vector<32xf32>)
        %hg = vector.load %hv[%tok, %pg, %c0] : view<[%tokens]x{RG}x32xf32> -> vector<32xf32>
        %ph = vector.mulf %pvv, %hg : vector<32xf32>
        %d = vector.dotf<reassoc|contract> %ph, %cvv, %zero_f : vector<32xf32>, vector<32xf32>, f32
        scf.yield %d : f32
      }} else {{
        scf.yield %zero_f : f32
      }}
      %transition = kernel.subgroup.reduce<addf> %prod {{cluster_size = 8}} : f32
      %score = scalar.addf %unary, %transition : f32
      %best = kernel.workgroup.reduce<maxnumf> %score : f32
      %wins = scalar.cmpf oeq, %score, %best : f32
      %elig = scf.select %wins, %id, %big : i32
      %chosen = kernel.workgroup.reduce<minsi> %elig : i32
      %leader = index.cmp eq, %thread, %c0 : index
      scf.if %leader {{
        view.store %chosen, %ov[%rb] : i32, view<[%row_cap]xi32>
      }}
      scf.yield %chosen : i32
    }}
  }}
  kernel.return
}}
"""


# ---------------------------------------------------------------------------------------------------
def _w(root, name, g):
    T = g["dflash"]["tensors"]
    t = T[name]
    if t["type"] == "f32":
        n = 1
        for s in t["shape"]:
            n *= s
        return f'command.parameter {root}, "{name}" : view<{n}xf32>'
    return f'command.parameter {root}, "{name}" : view<{g["dsize"](name)}xi8>'


def context_params(g):
    """The drafter's context projections, declared in one order in warm and propose (shared placement)."""
    out = []
    for l in range(g["dflash"]["layers"]):
        for n, s in (("attn_k.weight", "dk"), ("attn_v.weight", "dv"), ("attn_k_norm.weight", "dkn")):
            out.append(f"  %{s}_{l} = " + _w("%w_ctx", f"blk.{l}.{n}", g) + "\n")
    return "".join(out)


def programs(g):
    F = g["dflash"]
    H, NB, BS = g["H"], F["layers"], F["block_size"]
    NT = len(F["target_layers"])
    QD, KD, FFd = F["heads"] * F["head_dim"], F["kv_heads"] * F["head_dim"], F["ffn"]
    lin, dfmt, dsize = g["lin"], g["dfmt"], g["dsize"]
    tensors = F["tensors"]
    MR = g["MROPE"]
    rpos = "%rinfo" if MR else "%info"

    def rinfo(records):
        """M-RoPE targets: the drafter rotates at the target's temporal position (llama.cpp's degenerate sections)."""
        if not MR:
            return ""
        return ("  %rinfo = buffer.alloca<global> align(256) %info_size : buffer\n"
                f"  kernel.launch @g_dfl_rope_info[%t](%t, %spans, {records}, %metadata, %tokens, %info, %rinfo) : [index](index, index, index, buffer, buffer, buffer, buffer)\n")
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
  %nH = index.constant {H} : index
  %nDQ = index.constant {QD} : index
  %nDKV = index.constant {KD} : index
  %nDF = index.constant {FFd} : index
  %scale_entry = index.constant 8 : offset
  %scale_size = index.scale %t, %scale_entry : index, offset -> offset
"""

    def buf(name, width, f16=False):
        s = f"  %{name}_bytes = index.constant {width * 4} : offset\n  %{name}_size = index.scale %t, %{name}_bytes : index, offset -> offset\n  %{name} = buffer.alloca<global> align(256) %{name}_size : buffer\n"
        if f16:
            s += (f"  %{name}16_bytes = index.constant {width * 2} : offset\n  %{name}16_size = index.scale %t, %{name}16_bytes : index, offset -> offset\n"
                  f"  %{name}16 = buffer.alloca<global> align(256) %{name}16_size : buffer\n  %s_{name} = buffer.alloca<global> align(256) %scale_size : buffer\n")
        return s

    def scaled(name, width_ssa):
        return f"  scf.if %gemm {{\n    kernel.launch @g_row_scale_f16[%t](%t, {width_ssa}, %{name}, %s_{name}, %{name}16) : [index](index, index, buffer, buffer, buffer)\n  }}\n"

    def L(fmt_name, x, w, y, k_ssa, n_ssa, kval, mode="store", w2=None):
        return "  " + lin(dfmt(fmt_name), f"%{x}", f"%s_{x}", (w, w2) if mode == "swiglu" else w, dsize(fmt_name), f"%{y}", k_ssa, n_ssa, kval, mode) + "\n"

    # ---- warm
    warm = [f"""
// DFlash catch-up: context K/V of every draft layer at each committed position, from the captured target
// features (fc -> enc.output_norm -> attn_k / attn_v, k_norm, RoPE).
command.program.def public @llm_dflash_warm() launch(%w_enc: buffer, %w_ctx: buffer, %features: buffer, %metadata: buffer, %origins: buffer, %arena: buffer, %tokens: buffer) {{
{common}{context_params(g)}  %nFEAT = index.constant {NT * H} : index
  %feat_scale = buffer.alloca<global> align(256) %scale_size : buffer
  %feat16_bytes = index.constant {NT * H * 2} : offset
  %feat16_size = index.scale %t, %feat16_bytes : index, offset -> offset
  %features16 = buffer.alloca<global> align(256) %feat16_size : buffer
  %s_features = buffer.alloca<global> align(256) %scale_size : buffer
{buf("x", H)}{buf("h", H, True)}{buf("k", KD)}{buf("v", KD)}  %info_bytes = index.constant 16 : offset
  %info_size = index.scale %t, %info_bytes : index, offset -> offset
  %info = buffer.alloca<global> align(256) %info_size : buffer
  kernel.launch @g_token_info[%t](%t, %spans, %metadata, %info) : [index](index, index, buffer, buffer)
{rinfo("%c0")}  %w_fc = {_w("%w_enc", "fc.weight", g)}
  %w_encn = {_w("%w_enc", "enc.output_norm.weight", g)}
{scaled("features", "%nFEAT")}{L("fc.weight", "features", "%w_fc", "x", "%nFEAT", "%nH", NT * H)}  kernel.launch @g_spec_rmsnorm[%t](%t, %x, %w_encn, %h) : [index](index, buffer, view<{H}xf32>, buffer)
{scaled("h", "%nH")}"""]
    for l in range(NB):
        warm.append(L(f"blk.{l}.attn_k.weight", "h", f"%dk_{l}", "k", "%nH", "%nDKV", H)
                    + L(f"blk.{l}.attn_v.weight", "h", f"%dv_{l}", "v", "%nH", "%nDKV", H)
                    + f"  %dl_{l} = index.constant {l} : index\n"
                    f"  kernel.launch @g_dfl_kv_store[%t](%t, %dl_{l}, %context, %pool, %k, %v, %dkn_{l}, %info, %origins, %arena{', %rinfo' if MR else ''}) : [index](index, index, index, index, buffer, buffer, view<{F['head_dim']}xf32>, buffer, buffer, buffer{', buffer' if MR else ''})\n")
    warm.append("  command.return\n}\n")
    warm = "".join(warm)

    # ---- propose
    head_name, hsize, esize = g["head_name"], g["head_size"], g["esize"]
    head_root = "" if head_name == "token_embd.weight" else "%w_head: buffer, "
    head_ref = "%w_embed" if head_name == "token_embd.weight" else "%w_head"
    vis = ", %metadata, %info" if g["VISION"] else ""
    vty = ", buffer, buffer" if g["VISION"] else ""
    conv = F["conv"]
    embed_launch = (f"  kernel.launch @g_dfl_embed[%t](%t, %tokens, %embedding, %first_norm, %x, %h) : [index](index, buffer, view<{esize}xi8>, view<{H}xf32>, buffer, buffer)\n"
                    if own_embed(g) else
                    f"  kernel.launch @g_embed[%t](%t, %tokens{vis}, %embedding, %first_norm, %x, %h) : [index](index, buffer{vty}, view<{esize}xi8>, view<{H}xf32>, buffer, buffer)\n")
    P = [f"""
// DFlash proposals: the drafter runs [anchor, <mask> x {BS - 1}] blocks for the PROPOSE spans and publishes
// draft_depth tokens per block into the verifier's input slots.
command.program.def public @llm_dflash_propose() launch(%w_embed: buffer, {head_root}%w_draft: buffer, %w_ctx: buffer, %epoch: buffer, %target_tokens: buffer, %origins: buffer, %arena: buffer) {{
{common}{context_params(g)}  %epoch_spans = config.get @llm.epoch_spans : index
  %epoch_tokens = config.get @llm.epoch_tokens : index
  %width = config.get @llm.width : index
  %depth = index.sub %width, %c1 : index
  %bsz_full = index.constant {BS} : index
  %bsz = index.min %bsz_full, %width : index
  %meta_bytes = index.constant {16 + 40 * 1024} : offset
  %metadata = buffer.alloca<global> align(256) %meta_bytes : buffer
  %rows_meta_bytes = index.constant {16 + 36 * 8192} : offset
  %rows_meta = buffer.alloca<global> align(256) %rows_meta_bytes : buffer
  %c4_o = index.constant 4 : offset
  %word_size = index.scale %t, %c4_o : index, offset -> offset
  %tokens = buffer.alloca<global> align(256) %word_size : buffer
  %slots = buffer.alloca<global> align(256) %word_size : buffer
  %row_cap = index.mul %spans, %depth : index
  %row_words = index.scale %row_cap, %c4_o : index, offset -> offset
  %proposals = buffer.alloca<global> align(256) %row_words : buffer
{buf("x", H)}{buf("h", H, True)}{buf("y", H)}{buf("q", QD)}{buf("k", KD)}{buf("v", KD)}{buf("o", QD, True)}{buf("f", FFd, True)}  %info_bytes = index.constant 16 : offset
  %info_size = index.scale %t, %info_bytes : index, offset -> offset
  %info = buffer.alloca<global> align(256) %info_size : buffer
  %chunks_c = index.constant {g["CHUNKS"]} : index
  %partial_bytes = index.constant {g["CHUNKS"] * 8} : offset
  %partial_size = index.scale %row_cap, %partial_bytes : index, offset -> offset
  %partials = buffer.alloca<global> align(256) %partial_size : buffer
  %logit_bytes = index.constant {g["CHUNKS"] * 128 * 4} : offset
  %logit_size = index.scale %row_cap, %logit_bytes : index, offset -> offset
  %logits = buffer.alloca<global> align(256) %logit_size : buffer
  %embedding = command.parameter %w_embed, "token_embd.weight" : view<{esize}xi8>
  %head = command.parameter {head_ref}, "{head_name}" : view<{hsize}xi8>
  kernel.launch @g_dfl_prepare(%epoch_spans, %spans, %epoch_tokens, %depth, %epoch, %target_tokens, %metadata, %tokens, %slots, %rows_meta) : (index, index, index, index, buffer, buffer, buffer, buffer, buffer, buffer)
  kernel.launch @g_token_info[%t](%t, %spans, %metadata, %info) : [index](index, index, buffer, buffer)
{rinfo("%spans")}  %first_norm = {_w("%w_draft", "blk.0.attn_norm.weight", g)}
{embed_launch}"""]
    if conv:
        NGR = H // conv["group"]
        CW = 2 * conv["taps"] * NGR
        P.append(buf("dyn", CW) + buf("hc", H, True) + f"  %nDC = index.constant {CW} : index\n  %branch0 = index.constant 0 : index\n  %branch1 = index.constant 1 : index\n")
    for l in range(NB):
        b = f"blk.{l}."
        w = lambda n, root="%w_draft": _w(root, b + n, g)
        T = [f"  // ---- draft layer {l}\n",
             f"  %dqn_{l} = {w('attn_q_norm.weight')}\n  %dq_{l} = {w('attn_q.weight')}\n  %do_{l} = {w('attn_output.weight')}\n"
             f"  %dfn_{l} = {w('ffn_norm.weight')}\n  %dg_{l} = {w('ffn_gate.weight')}\n  %du_{l} = {w('ffn_up.weight')}\n  %dd_{l} = {w('ffn_down.weight')}\n"]
        src = "h"
        if conv:
            T.append(f"  %dacp_{l} = {w('attn_conv_proj.weight')}\n  %dacb_{l} = {w('attn_conv_base')}\n"
                     f"  %dfcp_{l} = {w('ffn_conv_proj.weight')}\n  %dfcb_{l} = {w('ffn_conv_base')}\n")
            T.append(scaled("h", "%nH") + L(b + "attn_conv_proj.weight", "h", f"%dacp_{l}", "dyn", "%nH", "%nDC", H)
                     + f"  kernel.launch @g_dfl_conv[%t](%t, %spans, %branch0, %zero_i32, %metadata, %info, %h, %dyn, %dacb_{l}, %hc) : [index](index, index, index, i32, buffer, buffer, buffer, buffer, view<{2 * conv['taps'] * H}xf32>, buffer)\n")
            src = "hc"
        T.append(scaled(src, "%nH")
                 + L(b + "attn_q.weight", src, f"%dq_{l}", "q", "%nH", "%nDQ", H)
                 + L(b + "attn_k.weight", src, f"%dk_{l}", "k", "%nH", "%nDKV", H)
                 + L(b + "attn_v.weight", src, f"%dv_{l}", "v", "%nH", "%nDKV", H)
                 + f"""  %dl_{l} = index.constant {l} : index
  kernel.launch @g_dfl_q_prep[%t](%t, %q, %dqn_{l}, {rpos}) : [index](index, buffer, view<{F['head_dim']}xf32>, buffer)
  kernel.launch @g_dfl_block_kv[%t](%t, %k, %dkn_{l}, {rpos}) : [index](index, buffer, view<{F['head_dim']}xf32>, buffer)
  kernel.launch @g_dfl_attention[%t](%t, %spans, %dl_{l}, %context, %pool, %metadata, %info, %q, %k, %v, %origins, %arena, %o) : [index](index, index, index, index, index, buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer)
""" + scaled("o", "%nDQ") + L(b + "attn_output.weight", "o", f"%do_{l}", "y", "%nDQ", "%nH", QD))
        if conv:
            T.append(f"  kernel.launch @g_dfl_conv[%t](%t, %spans, %branch1, %one_i32, %metadata, %info, %y, %dyn, %dacb_{l}, %x) : [index](index, index, index, i32, buffer, buffer, buffer, buffer, view<{2 * conv['taps'] * H}xf32>, buffer)\n"
                     f"  kernel.launch @g_dfl_add_norm[%t](%t, %zero_i32, %y, %x, %dfn_{l}, %h) : [index](index, i32, buffer, buffer, view<{H}xf32>, buffer)\n"
                     + scaled("h", "%nH") + L(b + "ffn_conv_proj.weight", "h", f"%dfcp_{l}", "dyn", "%nH", "%nDC", H)
                     + f"  kernel.launch @g_dfl_conv[%t](%t, %spans, %branch0, %zero_i32, %metadata, %info, %h, %dyn, %dfcb_{l}, %hc) : [index](index, index, index, i32, buffer, buffer, buffer, buffer, view<{2 * conv['taps'] * H}xf32>, buffer)\n")
        else:
            T.append(f"  kernel.launch @g_dfl_add_norm[%t](%t, %one_i32, %y, %x, %dfn_{l}, %h) : [index](index, i32, buffer, buffer, view<{H}xf32>, buffer)\n")
        T.append(scaled(src, "%nH") + L(b + "ffn_gate.weight", src, f"%dg_{l}", "f", "%nH", "%nDF", H, "swiglu", f"%du_{l}")
                 + scaled("f", "%nDF") + L(b + "ffn_down.weight", "f", f"%dd_{l}", "y", "%nDF", "%nH", FFd))
        nxt = f"%dan_{l + 1}" if l + 1 < NB else "%dout_norm"
        if l + 1 == NB:
            T.append(f"  %dout_norm = {_w('%w_draft', 'output_norm.weight', g)}\n")
        else:
            T.append(f"  %dan_{l + 1} = {_w('%w_draft', f'blk.{l + 1}.attn_norm.weight', g)}\n")
        if conv:
            T.append(f"  kernel.launch @g_dfl_conv[%t](%t, %spans, %branch1, %one_i32, %metadata, %info, %y, %dyn, %dfcb_{l}, %x) : [index](index, index, index, i32, buffer, buffer, buffer, buffer, view<{2 * conv['taps'] * H}xf32>, buffer)\n"
                     f"  kernel.launch @g_dfl_add_norm[%t](%t, %zero_i32, %y, %x, {nxt}, %h) : [index](index, i32, buffer, buffer, view<{H}xf32>, buffer)\n")
        else:
            T.append(f"  kernel.launch @g_dfl_add_norm[%t](%t, %one_i32, %y, %x, {nxt}, %h) : [index](index, i32, buffer, buffer, view<{H}xf32>, buffer)\n")
        P.append("".join(T))
    # head over rows 1..depth of each block
    P.append(f"""  %one_row = index.cmp eq, %row_cap, %c1 : index
  scf.if %one_row {{
    kernel.launch @g_head_mv[%row_cap](%row_cap, %t, %rows_meta, %h, %head, %partials, %logits) : [index](index, index, buffer, buffer, view<{hsize}xi8>, buffer, buffer)
  }} else {{
    kernel.launch @g_head_mv_t4[%row_cap](%row_cap, %t, %rows_meta, %h, %head, %partials, %logits) : [index](index, index, buffer, buffer, view<{hsize}xi8>, buffer, buffer)
  }}
""")
    sel = F["selector"]
    if sel:
        R = sel["rank"]
        P.append(f"""  %cand_bytes = index.constant 64 : offset
  %cand_size = index.scale %row_cap, %cand_bytes : index, offset -> offset
  %cand_ids = buffer.alloca<global> align(256) %cand_size : buffer
  %cand_scores = buffer.alloca<global> align(256) %cand_size : buffer
  %nSR = index.constant {R} : index
{buf("sh", R)}  %w_sel = {_w("%w_draft", "selector_hidden.weight", g)}
  %w_pred = {_w("%w_draft", "selector_predecessor.weight", g)}
  %w_succ = {_w("%w_draft", "selector_successor.weight", g)}
  kernel.launch @g_dfl_topk[%row_cap](%row_cap, %rows_meta, %partials, %logits, %cand_ids, %cand_scores) : [index](index, buffer, buffer, buffer, buffer, buffer)
{scaled("h", "%nH")}{L("selector_hidden.weight", "h", "%w_sel", "sh", "%nH", "%nSR", H)}  kernel.launch @g_dfl_chain[%spans](%spans, %depth, %t, %bsz, %metadata, %tokens, %cand_ids, %cand_scores, %sh, %w_pred, %w_succ, %proposals) : [index](index, index, index, index, buffer, buffer, buffer, buffer, buffer, view<{dsize("selector_predecessor.weight")}xi8>, view<{dsize("selector_successor.weight")}xi8>, buffer)
""")
    else:
        P.append("  kernel.launch @g_head_select[%row_cap](%row_cap, %rows_meta, %partials, %logits, %proposals) : [index](index, buffer, buffer, buffer, buffer)\n")
    P.append("  kernel.launch @g_dfl_publish(%spans, %epoch_tokens, %depth, %epoch, %metadata, %slots, %proposals, %target_tokens) : (index, index, index, buffer, buffer, buffer, buffer, buffer)\n  command.return\n}\n")
    decls = ("config.decl @llm.epoch_spans : %value: index where [range(%value, 1, 8192)]\n\n"
             "config.decl @llm.epoch_tokens : %value: index where [range(%value, 1, 8192)]\n")
    return decls + warm + "".join(P)
