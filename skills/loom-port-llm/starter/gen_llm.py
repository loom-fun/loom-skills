# Generates llm.loom: kernel library + forward-pass command program for a dense
# decoder-only LLM stored as GGUF, driven by a config JSON from inspect_gguf.py.
#
# Generalized from the Muse Glimmer 30B port; end-to-end validated (llama.cpp
# golden, 1k-32k prompts) on Qwen3-0.6B. Supported:
#   * head_dim 64, 128 or 256, any query/KV head counts with heads % kv_heads == 0
#   * q/k/v projection biases (qwen2: cfg attn_bias, added by g_qk_rope_store)
#   * GGUF weight formats in formats.py (q4_k, q5_k, q6_k, q8_0, q5_0, ...; mixed per tensor is fine)
#   * RMSNorm, SwiGLU FFN, NeoX or interleaved ("norm") RoPE, optional per-head
#     Q/K RMSNorm, optional per-layer sliding window, optional sigmoid attention
#     output gate and post-attention/post-FFN norms (Glimmer), optional
#     embedding RMSNorm, tied or separate output head.
# The head keeps the top-64 candidates for sampling; an optional final logit softcap/scale
# (cfg logit_softcap / logit_scale) is applied to the sampling logits (argmax unaffected).
import json, math, os, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_gdn import gdn_geometry, gdn_kernels, split_qgate  # noqa: E402

CFG = json.load(open(os.environ.get("LLM_CONFIG") or sys.argv[1]))
H = CFG["hidden"]; HG = H // 32          # hidden size, 32-value groups per hidden row
QH, KVH, D = CFG["heads"], CFG["kv_heads"], CFG["head_dim"]
V, L = CFG["vocab"], CFG["layers"]
# Per-layer FFN width (Gemma-4 E2B: 6144 then 12288); buffers are sized for the widest.
FFL = CFG["ffn"] if isinstance(CFG["ffn"], list) else [CFG["ffn"]] * L
FF = max(FFL)
FF_UNIFORM = len(set(FFL)) == 1
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "draft"))
import gen_draft  # noqa: E402  speculative decoding (LLM_DRAFT, set by build.py): verify + drafter programs
MTP_LAYER = None
if gen_draft.enabled() and gen_draft.DRAFT["kind"] == "mtp":
    MTP_LAYER = CFG["mtp"]["layers"][0]
    _w = CFG["tensors"].get(f"blk.{MTP_LAYER}.ffn_gate.weight", {"shape": [0, FF]})["shape"][1]   # MoE block: no dense FFN
    assert _w in FFL, f"MTP block FFN width {_w} differs from the trunk's {sorted(set(FFL))}"
    FFL = FFL + [_w] * (MTP_LAYER + 1 - len(FFL))   # layer_text(MTP_LAYER) reads FFL[MTP_LAYER]
def nff(i):
    """index SSA of layer i's FFN width."""
    return "%nFF" if FF_UNIFORM else f"%nFF{FFL[i]}"
# Layers whose ffn_gate and ffn_up use different formats (unsloth UD quants) run two projections + g_silu_mul.
MIXED_FFN = any(n.endswith("ffn_gate.weight") and CFG["tensors"].get(n.replace("ffn_gate", "ffn_up"), {}).get("type") not in (None, t["type"])
                for n, t in CFG["tensors"].items())
G = QH // KVH                             # query heads per KV head (waves per attention workgroup)
# Prefill flash attention (gen_flash2.py) tiles head_dim over 32*G lanes: it needs a power-of-two group.
# Otherwise (Qwen3.8: 6) every prefill token takes the listed split-K path (correct, slower prefill).
NO_FLASH = os.environ.get("NO_FLASH2") == "1"
assert D in (64, 128, 256), "attention kernels support head_dim 64, 128 or 256 (lane l owns dims l, l+32, ...)"
DL = D // 32                              # head dims per lane in the attention kernels
RD = CFG.get("rope_dims") or D            # rotary dims (partial RoPE rotates the first RD dims)
assert RD % 64 == 0 and RD <= D, "rope_dims must be a multiple of 64"
assert QH % KVH == 0 and G <= 16, "need heads % kv_heads == 0 and at most 16 query heads per KV head"
assert HG <= 256 and H % 32 == 0, "hidden must be a multiple of 32 and at most 8192"
import attn_rules  # noqa: E402  decode attention variants (ATTN_* env), shared with make_kernel_tests.py
from attn_rules import UNROLL, ATTN_MT, ATTN_MT_FROM  # noqa: E402
POS_BYTES = KVH * D * 2                   # F16 K (or V) bytes per position
WINDOW = CFG.get("sliding_window") or 0   # keys kept by sliding layers (0: none)
GATE = CFG.get("attn_gate", False)       # True: separate attn_gate tensor; "fused_q": attn_q packs [q | gate] per head
FUSED_Q = GATE == "fused_q"
# "per_head": attn_gate has one row per query head (spark2_5): sigmoid(x Wg^T)[head] scales that head's whole output.
# g_gate_heads broadcasts the logit over the head's D dims, so every attention path applies it as an elementwise gate.
HEAD_GATE = GATE == "per_head"
# Attention layers project q | k | v with one fused attn_qkv tensor (spark2_5); g_split_qkv splits the rows.
QKV_FUSED = bool(CFG.get("attn_qkv", False))
GELU = CFG.get("ffn_act") == "gelu"      # GEGLU FFN (gelu-tanh, llama.cpp LLM_FFN_GELU) instead of SwiGLU
RECUR = set(CFG.get("recurrent_layers", []))  # gated delta-net layers (hybrid models)
GDN = gdn_geometry(CFG) if RECUR else None
FFN_NORM = CFG.get("ffn_norm_tensor", "ffn_norm")
MOE = CFG.get("moe")                      # MoE FFN (gen_moe.py: router + top-k experts, optional shared experts)
MOE_LAYERS = set(range(MOE["leading_dense"], CFG["layers"])) | ({MTP_LAYER} if MTP_LAYER is not None else set()) if MOE else set()
FS = (MOE or {}).get("shared_ffn", 0)     # shared-expert FFN width (0: none)
SHGATE = (MOE or {}).get("shared_gate", False)   # Qwen-style sigmoid gate on the shared experts (ffn_gate_inp_shexp)
SINKS = CFG.get("attn_sinks", False)      # learned per-head softmax sink logits (attn_sinks)
assert not (SINKS and GATE), "attention sinks ride in the output gate's argument of the attention kernels: not both"
POST_NORM = CFG.get("post_norms", False)
POST_EPS = CFG.get("post_norm_eps", 1e-8)
EMBED_NORM = CFG.get("embed_norm", False)
_LS = CFG.get("logit_scale") or 1.0; _SC = CFG.get("logit_softcap") or 0.0
if _SC:  # sampling sees llama.cpp's logits: softcap * tanh(scale * z / softcap) (argmax unaffected)
    SOFTCAP_STORE = (f"            %sc_y = scalar.mulf %best, %sc_k : f32\n            %sc_y2 = scalar.addf %sc_y, %sc_y : f32\n"
                     f"            %sc_e = scalar.expf<afn> %sc_y2 : f32\n            %sc_one = scalar.constant 1.0 : f32\n            %sc_two = scalar.constant 2.0 : f32\n"
                     f"            %sc_d = scalar.addf %sc_e, %sc_one : f32\n            %sc_q = scalar.divf %sc_two, %sc_d : f32\n            %sc_t = scalar.subf %sc_one, %sc_q : f32\n"
                     f"            %sc_cap = scalar.constant {float(_SC)} : f32\n            %sc_v = scalar.mulf %sc_t, %sc_cap : f32\n"
                     f"            view.store %sc_v, %cand_v[%rr] : f32, view<64xf32>")
    SOFTCAP_STORE = f"            %sc_k = scalar.constant {float(_LS) / float(_SC)} : f32\n" + SOFTCAP_STORE
else:
    SOFTCAP_STORE = "            view.store %best, %cand_v[%rr] : f32, view<64xf32>"
EPS = CFG["rms_eps"]
QK_NORM = CFG.get("qk_norm", False)
# Attention projection biases (qwen2: q, k, v; inspect_gguf.py lists them): g_qk_rope_store adds each as it reads the
# projection, before the QK norm and RoPE, so every producer (decode matvecs, prefill GEMMs, verify) gets them.
QKV_BIAS = [p for p in ("q", "k", "v") if p in (CFG.get("attn_bias") or [])]
assert not (QKV_BIAS and FUSED_Q), "attn_q bias with a fused [q | gate] projection: split the bias too"
ROPE = CFG.get("rope_type", "neox")       # "neox": pairs (d, d+64); "norm": pairs (2d, 2d+1)
THETA = CFG["rope_theta"]
# Sliding layers with their own RoPE base / rotary dims at the same head_dim (spark2_5: rope.freq_base_swa,
# rope.dimension_count_swa; llama.cpp hparams.n_rot(il), rope_freq_base_train_swa): g_qk_rope_store_swa.
# (A different head_dim on sliding layers, Gemma-4, is reference/gemma4-example's.)
RD_SWA, THETA_SWA = CFG.get("rope_dims_swa") or RD, CFG.get("rope_theta_swa") or THETA
# cfg rope_scaling_swa "none": the global layers' YaRN does not apply to the sliding ones (mellum), which then take
# g_qk_rope_store_swa even at the same base and dims.
SWA_ROPE = bool(CFG.get("sliding_layers")) and (CFG.get("head_dim_swa") or D) == D and (
    (RD_SWA, THETA_SWA) != (RD, THETA) or bool(CFG.get("rope_scaling")) and CFG.get("rope_scaling_swa") == "none")
assert not SWA_ROPE or (RD_SWA % 64 == 0 and RD_SWA <= D), "rope_dims_swa must be a multiple of 64"
FREQ_FACTORS = CFG.get("rope_freq_factors")  # per rotary pair: frequency / factor (llama3 rope_freqs; global layers only)
TENS = CFG["tensors"]                     # tensor name -> {"type": fmt, "shape": [...]}
# Gate activation (cfg attn_gate_act): sigmoid (default), or softplus (laguna) on a per-head gate, which g_gate_heads
# applies to the head's logit so the attention kernels multiply by the stored value.
GATE_ACT = CFG.get("attn_gate_act", "sigmoid")
assert GATE_ACT == "sigmoid" or (HEAD_GATE and GATE_ACT == "softplus"), f"attn_gate_act {GATE_ACT}: per-head softplus or sigmoid"
# Query heads per layer (cfg layer_heads; laguna: 48 on full-attention layers, 64 on sliding ones). QH is the widest
# (buffers); the attention kernels of a narrower layer are generated again for its head count (suffix _h<heads>).
LH = CFG.get("layer_heads") or [QH] * L
NARROW = sorted(set(LH) - {QH})
assert all(h % KVH == 0 and h // KVH <= 16 for h in NARROW), "layer_heads: multiples of kv_heads, at most 16 per KV head"
OUT_BIAS = CFG.get("attn_out_bias", False)   # attn_output.bias on every attention layer (gpt-oss): g_bias_add
# YaRN RoPE scaling (cfg rope_scaling, inspect_gguf.py; llama.cpp ggml rope_yarn with ext_factor 1): pair i's
# frequency theta^(-2i/RD) times 1 - (1 - 1/factor) * ramp(i), ramp = clamp((i - low) / (high - low), 0, 1), and the
# rotated values times mscale.
YARN = CFG.get("rope_scaling") if (CFG.get("rope_scaling") or {}).get("type") == "yarn" else None


def yarn_constants(y, rd, theta):
    """(low, 1 / max(0.001, high - low), 1 - 1/factor, mscale) of llama.cpp's YaRN (ggml_rope_yarn_corr_dims) for a
    RoPE of rd rotated dims and base theta; y: a cfg rope_scaling block (factor, orig_ctx, beta_fast/slow, attn_factor)."""
    f, n0 = float(y["factor"]), float(y["orig_ctx"])
    dim = lambda rot: rd * math.log(n0 / (rot * 2 * math.pi)) / (2 * math.log(theta))
    low, high = max(0.0, float(math.floor(dim(y.get("beta_fast", 32.0))))), min(rd - 1.0, float(math.ceil(dim(y.get("beta_slow", 1.0)))))
    mscale = y.get("attn_factor", 1.0) * (1.0 + 0.1 * math.log(f))
    return low, 1.0 / max(0.001, high - low), 1.0 - 1.0 / f, mscale

from formats import FMT, SB, LOWBIT, ICODES as IQ_FMTS  # noqa: E402  bytes / values per block of each decoded format


def row_bytes(fmt, k):
    return k // SB[fmt] * FMT[fmt]


def fmt_of(name):
    return TENS[name]["type"]


def tensor_bytes(name):
    t = TENS[name]
    n = 1
    for d in t["shape"]:
        n *= d
    if t["type"] == "f32":
        return n * 4
    if t["type"] in ("f16", "bf16"):
        return n * 2
    return n // SB[t["type"]] * FMT[t["type"]]


COMMON = r"""
// Paged KV: 2 planes per layer (layer-major, K then V); one position holds the
// KV heads x 128 F16 values in 16-position pages mapped per session row.
func.def inline @g_kv_offset(%origins: buffer, %row: index, %position: index, %blocks: index, %plane: index, %pool: index) -> (offset) {
  %c1 = index.constant 1 : index
  %c16 = index.constant 16 : index
  %record_stride = index.constant 40 : offset
  %position_bytes = index.constant 512 : offset
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
}

// Q4_K / Q5_K 6-bit scale and minimum of group `group` (0-7) of a superblock,
// from hrx-loom-library motif/format/ggml/q4_k.loom.
func.def inline @g_k_scale_minimum(%packed_scales: vector<3xi32>, %group: index) -> (i32, i32) {
  %bounded_group = index.assume %group [range(%group, 0, 7)] : index
  %scale0 = vector.extract %packed_scales[0] : vector<3xi32> -> i32
  %scale1 = vector.extract %packed_scales[1] : vector<3xi32> -> i32
  %scale2 = vector.extract %packed_scales[2] : vector<3xi32> -> i32
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c2_i32 = scalar.constant 2 : i32
  %c4_i32 = scalar.constant 4 : i32
  %c15_i32 = scalar.constant 15 : i32
  %c48_i32 = scalar.constant 48 : i32
  %is_low_group = index.cmp ult, %bounded_group, %c4 : index
  %lane = index.rem %bounded_group, %c4 : index
  %shift_index = index.mul %lane, %c8 : index
  %shift = index.cast %shift_index : index to i32
  %high_shift = scalar.addi %shift, %c2_i32 : i32
  %minimum_shift = scalar.addi %shift, %c4_i32 : i32
  %scale_low_source = scf.select %is_low_group, %scale0, %scale2 : i32
  %minimum_low_source = scf.select %is_low_group, %scale1, %scale2 : i32
  %scale_high_shift = scf.select %is_low_group, %shift, %high_shift : i32
  %minimum_low_shift = scf.select %is_low_group, %shift, %minimum_shift : i32
  %scale_low_shifted = scalar.shrui %scale_low_source, %shift : i32
  %scale_low = scalar.andi %scale_low_shifted, %c15_i32 : i32
  %scale_high_shifted = scalar.shrui %scale0, %scale_high_shift : i32
  %scale_high = scalar.andi %scale_high_shifted, %c48_i32 : i32
  %scale = scalar.ori %scale_low, %scale_high : i32
  %minimum_low_shifted = scalar.shrui %minimum_low_source, %minimum_low_shift : i32
  %minimum_low = scalar.andi %minimum_low_shifted, %c15_i32 : i32
  %minimum_high_shifted = scalar.shrui %scale1, %scale_high_shift : i32
  %minimum_high = scalar.andi %minimum_high_shifted, %c48_i32 : i32
  %minimum = scalar.ori %minimum_low, %minimum_high : i32
  func.return %scale, %minimum : i32, i32
}

// (d * scale, dmin * minimum) of group `group` from a 16-byte K-quant header.
func.def inline @g_k_metadata(%header: vector<4xi32>, %group: index) -> (f32, f32) {
  %halves = vector.bitcast %header : vector<4xi32> to vector<8xf16>
  %d_half = vector.extract %halves[0] : vector<8xf16> -> f16
  %minimum_half = vector.extract %halves[1] : vector<8xf16> -> f16
  %d = scalar.extf %d_half : f16 to f32
  %dmin = scalar.extf %minimum_half : f16 to f32
  %first = vector.extract %header[1] : vector<4xi32> -> i32
  %second = vector.extract %header[2] : vector<4xi32> -> i32
  %third = vector.extract %header[3] : vector<4xi32> -> i32
  %packed = vector.from_elements %first, %second, %third : vector<3xi32>
  %scale, %minimum = func.call @g_k_scale_minimum(%packed, %group) : (vector<3xi32>, index) -> (i32, i32)
  %scale_float = scalar.uitofp %scale : i32 to f32
  %minimum_float = scalar.uitofp %minimum : i32 to f32
  %scaled = scalar.mulf %d, %scale_float : f32
  %minimum_scaled = scalar.mulf %dmin, %minimum_float : f32
  func.return %scaled, %minimum_scaled : f32, f32
}

// The 32 values of group `group` (0-7) of a Q4_K superblock at `origin`.
func.def inline @g_q4_k_values(%w: buffer, %origin: offset, %group: index) -> (vector<32xf32>) {
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %header_bytes = index.constant 16 : offset
  %page_bytes = index.constant 32 : offset
  %nibble_mask = vector.constant 252645135 : vector<8xi32>
  %header_view = buffer.view %w[%origin] : buffer -> view<4xi32>
  %header = vector.load %header_view[0] : view<4xi32> -> vector<4xi32>
  %scale, %minimum = func.call @g_k_metadata(%header, %group) : (vector<4xi32>, index) -> (f32, f32)
  %page = index.div %group, %c2 : index
  %nibble = index.rem %group, %c2 : index
  %page_add = index.scale %page, %page_bytes : index, offset -> offset
  %payload = index.add %origin, %header_bytes : offset
  %codes_origin = index.add %payload, %page_add : offset
  %code_view = buffer.view %w[%codes_origin] : buffer -> view<8xi32>
  %packed = vector.load %code_view[0] : view<8xi32> -> vector<8xi32>
  %shift_bits = index.mul %nibble, %c4 : index
  %shift_i32 = index.cast %shift_bits : index to i32
  %shift = vector.splat %shift_i32 : vector<8xi32>
  %shifted = vector.shrui %packed, %shift : vector<8xi32>
  %masked = vector.andi %shifted, %nibble_mask : vector<8xi32>
  %bytes = vector.bitcast %masked : vector<8xi32> to vector<32xi8>
  %codes = vector.uitofp %bytes : vector<32xi8> to vector<32xf32>
  %scales = vector.splat %scale : vector<32xf32>
  %minima = vector.splat %minimum : vector<32xf32>
  %scaled = vector.mulf %codes, %scales : vector<32xf32>
  %values = vector.subf %scaled, %minima : vector<32xf32>
  func.return %values : vector<32xf32>
}




// Q6_K (210-byte superblock): ql[128] low nibbles, qh[64] high 2 bits,
// 16 int8 scales, F16 d. Group g = 4*half + j covers values half*128 + j*32
// + l: q = ((ql[half*64 + (j%2)*32 + l] >> 4*(j/2)) & 15)
//        | (((qh[half*32 + l] >> 2*j) & 3) << 4), value = d * sc[half*8 + 2j
// + l/16] * (q - 32). Superblocks are only 2-byte aligned, so codes load as bytes.
// Group `group` of a Q6_K superblock as 32 F32 values.
func.def inline @g_q6_k_values(%w: buffer, %origin: offset, %group: index) -> (vector<32xf32>) {
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %qh_base = index.constant 128 : index
  %sc_base = index.constant 192 : index
  %d_bytes = index.constant 208 : offset
  %thirty_two = vector.constant 32.0 : vector<32xf32>
  %low_mask = vector.constant 15 : vector<32xi8>
  %two_bits = vector.constant 3 : vector<32xi8>
  %four8 = vector.constant 4 : vector<32xi8>
  %bounded = index.assume %group [range(%group, 0, 7)] : index
  %half = index.div %bounded, %c4 : index
  %j = index.rem %bounded, %c4 : index
  %j_low = index.rem %j, %c2 : index
  %j_high = index.div %j, %c2 : index
  %ql_half = index.mul %half, %c64 : index
  %ql_part = index.mul %j_low, %c32 : index
  %ql_index = index.add %ql_half, %ql_part : index
  %qh_half = index.mul %half, %c32 : index
  %qh_index = index.add %qh_base, %qh_half : index
  %sc_half = index.mul %half, %c8 : index
  %sc_j = index.mul %j, %c2 : index
  %sc_part = index.add %sc_half, %sc_j : index
  %sc_index = index.add %sc_base, %sc_part : index
  %ql_add = index.cast %ql_index : index to offset
  %qh_add = index.cast %qh_index : index to offset
  %sc_add = index.cast %sc_index : index to offset
  %ql_origin = index.add %origin, %ql_add : offset
  %qh_origin = index.add %origin, %qh_add : offset
  %sc_origin = index.add %origin, %sc_add : offset
  %d_origin = index.add %origin, %d_bytes : offset
  %ql_view = buffer.view %w[%ql_origin] : buffer -> view<32xi8>
  %qh_view = buffer.view %w[%qh_origin] : buffer -> view<32xi8>
  %sc_view = buffer.view %w[%sc_origin] : buffer -> view<2xi8>
  %d_view = buffer.view %w[%d_origin] : buffer -> view<1xf16>
  %ql = vector.load %ql_view[0] : view<32xi8> -> vector<32xi8>
  %qh = vector.load %qh_view[0] : view<32xi8> -> vector<32xi8>
  %sc = vector.load %sc_view[0] : view<2xi8> -> vector<2xi8>
  %d_half = view.load %d_view[0] : view<1xf16> -> f16
  %d = scalar.extf %d_half : f16 to f32
  %shift_low = index.mul %j_high, %c4 : index
  %shift_high = index.mul %j, %c2 : index
  %shift_low32 = index.cast %shift_low : index to i32
  %shift_high32 = index.cast %shift_high : index to i32
  %shift_low8 = scalar.trunci %shift_low32 : i32 to i8
  %shift_high8 = scalar.trunci %shift_high32 : i32 to i8
  %shift_low_v = vector.splat %shift_low8 : vector<32xi8>
  %shift_high_v = vector.splat %shift_high8 : vector<32xi8>
  %ql_shifted = vector.shrui %ql, %shift_low_v : vector<32xi8>
  %low = vector.andi %ql_shifted, %low_mask : vector<32xi8>
  %qh_shifted = vector.shrui %qh, %shift_high_v : vector<32xi8>
  %qh_bits = vector.andi %qh_shifted, %two_bits : vector<32xi8>
  %high = vector.shli %qh_bits, %four8 : vector<32xi8>
  %combined = vector.ori %low, %high : vector<32xi8>
  %codes = vector.uitofp %combined : vector<32xi8> to vector<32xf32>
  %centered = vector.subf %codes, %thirty_two : vector<32xf32>
  %s0_i8 = vector.extract %sc[0] : vector<2xi8> -> i8
  %s1_i8 = vector.extract %sc[1] : vector<2xi8> -> i8
  %s0f = scalar.sitofp %s0_i8 : i8 to f32
  %s1f = scalar.sitofp %s1_i8 : i8 to f32
  %s0 = scalar.mulf %s0f, %d : f32
  %s1 = scalar.mulf %s1f, %d : f32
  %s0_v = vector.splat %s0 : vector<16xf32>
  %s1_v = vector.splat %s1 : vector<16xf32>
  %scale_v = vector.concat<0> %s0_v, %s1_v : vector<16xf32>, vector<16xf32> -> vector<32xf32>
  %values = vector.mulf %centered, %scale_v : vector<32xf32>
  func.return %values : vector<32xf32>
}

// Q5_K: header (d, dmin, 12 scale bytes), 32 high-bit bytes, 128 nibble bytes.
// Value of group g: d*scale_g * (nibble | (qh_bit_g << 4)) - dmin*min_g.
// Group `group` of a Q5_K superblock as 32 F32 values.
func.def inline @g_q5_k_values(%w: buffer, %origin: offset, %group: index) -> (vector<32xf32>) {
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %high_bytes = index.constant 16 : offset
  %low_bytes = index.constant 48 : offset
  %page_bytes = index.constant 32 : offset
  %nibble_mask = vector.constant 252645135 : vector<8xi32>
  %bit_mask = vector.constant 16843009 : vector<8xi32>
  %four = vector.constant 4 : vector<8xi32>
  %header_view = buffer.view %w[%origin] : buffer -> view<4xi32>
  %header = vector.load %header_view[0] : view<4xi32> -> vector<4xi32>
  %scale, %minimum = func.call @g_k_metadata(%header, %group) : (vector<4xi32>, index) -> (f32, f32)
  %page = index.div %group, %c2 : index
  %nibble = index.rem %group, %c2 : index
  %page_add = index.scale %page, %page_bytes : index, offset -> offset
  %payload = index.add %origin, %low_bytes : offset
  %codes_origin = index.add %payload, %page_add : offset
  %code_view = buffer.view %w[%codes_origin] : buffer -> view<8xi32>
  %packed = vector.load %code_view[0] : view<8xi32> -> vector<8xi32>
  %high_origin = index.add %origin, %high_bytes : offset
  %high_view = buffer.view %w[%high_origin] : buffer -> view<8xi32>
  %high_packed = vector.load %high_view[0] : view<8xi32> -> vector<8xi32>
  %shift_bits = index.mul %nibble, %c4 : index
  %shift_i32 = index.cast %shift_bits : index to i32
  %shift = vector.splat %shift_i32 : vector<8xi32>
  %shifted = vector.shrui %packed, %shift : vector<8xi32>
  %low = vector.andi %shifted, %nibble_mask : vector<8xi32>
  %group_i32 = index.cast %group : index to i32
  %group_shift = vector.splat %group_i32 : vector<8xi32>
  %high_shifted = vector.shrui %high_packed, %group_shift : vector<8xi32>
  %high_bit = vector.andi %high_shifted, %bit_mask : vector<8xi32>
  %high = vector.shli %high_bit, %four : vector<8xi32>
  %combined = vector.ori %low, %high : vector<8xi32>
  %bytes = vector.bitcast %combined : vector<8xi32> to vector<32xi8>
  %codes = vector.uitofp %bytes : vector<32xi8> to vector<32xf32>
  %scales = vector.splat %scale : vector<32xf32>
  %minima = vector.splat %minimum : vector<32xf32>
  %scaled = vector.mulf %codes, %scales : vector<32xf32>
  %values = vector.subf %scaled, %minima : vector<32xf32>
  func.return %values : vector<32xf32>
}

// Group `group` of a Q4_K superblock as raw codes plus (d*scale, dmin*min):
// value = codes * scale - minimum.
func.def inline @g_q4_k_codes(%w: buffer, %origin: offset, %group: index) -> (vector<32xf32>, f32, f32) {
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %header_bytes = index.constant 16 : offset
  %page_bytes = index.constant 32 : offset
  %nibble_mask = vector.constant 252645135 : vector<8xi32>
  %header_view = buffer.view %w[%origin] : buffer -> view<4xi32>
  %header = vector.load %header_view[0] : view<4xi32> -> vector<4xi32>
  %scale, %minimum = func.call @g_k_metadata(%header, %group) : (vector<4xi32>, index) -> (f32, f32)
  %page = index.div %group, %c2 : index
  %nibble = index.rem %group, %c2 : index
  %page_add = index.scale %page, %page_bytes : index, offset -> offset
  %payload = index.add %origin, %header_bytes : offset
  %codes_origin = index.add %payload, %page_add : offset
  %code_view = buffer.view %w[%codes_origin] : buffer -> view<8xi32>
  %packed = vector.load %code_view[0] : view<8xi32> -> vector<8xi32>
  %shift_bits = index.mul %nibble, %c4 : index
  %shift_i32 = index.cast %shift_bits : index to i32
  %shift = vector.splat %shift_i32 : vector<8xi32>
  %shifted = vector.shrui %packed, %shift : vector<8xi32>
  %masked = vector.andi %shifted, %nibble_mask : vector<8xi32>
  %bytes = vector.bitcast %masked : vector<8xi32> to vector<32xi8>
  %codes = vector.uitofp %bytes : vector<32xi8> to vector<32xf32>
  func.return %codes, %scale, %minimum : vector<32xf32>, f32, f32
}

func.def inline @g_store_row(%y: buffer, %tokens: index, %n: index, %token: index, %row: index, %value: f32, %accumulate: i32) {
  %zero = index.constant 0 : offset
  %c0_i32 = scalar.constant 0 : i32
  %bounded_token = index.assume %token [lt(%token, %tokens)] : index
  %bounded_row = index.assume %row [lt(%row, %n)] : index
  %y_view = buffer.view %y[%zero] : buffer -> view<[%tokens]x[%n]xf32>
  %add = scalar.cmpi ne, %accumulate, %c0_i32 : i32
  %result = scf.if %add -> (f32) {
    %old = view.load %y_view[%bounded_token, %bounded_row] : view<[%tokens]x[%n]xf32> -> f32
    %sum = scalar.addf %old, %value : f32
    scf.yield %sum : f32
  } else {
    scf.yield %value : f32
  }
  view.store %result, %y_view[%bounded_token, %bounded_row] : f32, view<[%tokens]x[%n]xf32>
  func.return
}
"""


COMMON = (COMMON.replace("%position_bytes = index.constant 512 : offset", f"%position_bytes = index.constant {POS_BYTES} : offset")
          .replace("// Paged KV: 104 planes (layer-major, K then V); one position holds 2 KV heads\n// x 128 F16 values (512 bytes)",
                   f"// Paged KV: {2 * L} planes (layer-major, K then V); one position holds {KVH} KV heads\n// x 128 F16 values ({POS_BYTES} bytes)"))

COMMON += r"""
// Q8_0 block (34 bytes): F16 scale d, 32 int8 codes; value = d * q.
func.def inline @g_q8_0_values(%w: buffer, %origin: offset, %group: index) -> (vector<32xf32>) {
  %two = index.constant 2 : offset
  %scale_view = buffer.view %w[%origin] : buffer -> view<1xf16>
  %code_origin = index.add %origin, %two : offset
  %code_view = buffer.view %w[%code_origin] : buffer -> view<32xi8>
  %scale_half = view.load %scale_view[0] : view<1xf16> -> f16
  %scale = scalar.extf %scale_half : f16 to f32
  %codes = vector.load %code_view[0] : view<32xi8> -> vector<32xi8>
  %values = vector.sitofp %codes : vector<32xi8> to vector<32xf32>
  %scales = vector.splat %scale : vector<32xf32>
  %scaled = vector.mulf %values, %scales : vector<32xf32>
  func.return %scaled : vector<32xf32>
}

// Q5_K group as values (for heads/embeddings) is g_q5_k_values; Q4_K is g_q4_k_values.
"""

# ACT_Q8=1 (default): decode-shaped projections (t < 64) and the head read int8 activations (g_quant_q8, once per
# activation: llama.cpp's Q8_1 rule, d = amax/127 rounded to F16, q = round(x/d)) and contract the weight codes with
# integer dot4 products, as llama.cpp's mul_mat_vec_q does. The 1/4/8/16-token variants share the per-token
# arithmetic and the wave-sum tree (bfly_store), so decode and verify round identically. ACT_Q8=0: f32 activations.
ACT_Q8 = os.environ.get("ACT_Q8", "1") == "1"
# IQ_FMTS (formats.ICODES): the formats with an @g_<fmt>_icodes helper (Q5_0: Q5_0 below, emitted when used)
ICODES = r"""
// Integer weight codes of one 32-value group for the Q8 activation matvec (ACT_Q8).
func.def inline @g_q8_0_icodes(%w: buffer, %origin: offset, %group: index) -> (vector<32xi8>, f32) {
  %two = index.constant 2 : offset
  %scale_view = buffer.view %w[%origin] : buffer -> view<1xf16>
  %code_origin = index.add %origin, %two : offset
  %code_view = buffer.view %w[%code_origin] : buffer -> view<32xi8>
  %scale_half = view.load %scale_view[0] : view<1xf16> -> f16
  %scale = scalar.extf %scale_half : f16 to f32
  %codes = vector.load %code_view[0] : view<32xi8> -> vector<32xi8>
  func.return %codes, %scale : vector<32xi8>, f32
}

// Q4_0 (the load-time proposal head copy, draft/prep_weights.py): codes 0..15 with minimum 8 d, so the int8 dot
// subtracts 8 x (activation code sum) like a K-quant minimum.
func.def inline @g_q4_0_icodes(%w: buffer, %origin: offset, %group: index) -> (vector<32xi8>, f32, f32) {
  %two = index.constant 2 : offset
  %scale_view = buffer.view %w[%origin] : buffer -> view<1xf16>
  %code_origin = index.add %origin, %two : offset
  %code_view = buffer.view %w[%code_origin] : buffer -> view<16xi8>
  %scale_half = view.load %scale_view[0] : view<1xf16> -> f16
  %scale = scalar.extf %scale_half : f16 to f32
  %packed = vector.load %code_view[0] : view<16xi8> -> vector<16xi8>
  %nibble_mask = vector.constant 15 : vector<16xi8>
  %four = vector.constant 4 : vector<16xi8>
  %low = vector.andi %packed, %nibble_mask : vector<16xi8>
  %high = vector.shrui %packed, %four : vector<16xi8>
  %codes = vector.concat<0> %low, %high : vector<16xi8>, vector<16xi8> -> vector<32xi8>
  %eight = scalar.constant 8.0 : f32
  %minimum = scalar.mulf %scale, %eight : f32
  func.return %codes, %scale, %minimum : vector<32xi8>, f32, f32
}

func.def inline @g_q4_k_icodes(%w: buffer, %origin: offset, %group: index) -> (vector<32xi8>, f32, f32) {
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %header_bytes = index.constant 16 : offset
  %page_bytes = index.constant 32 : offset
  %nibble_mask = vector.constant 252645135 : vector<8xi32>
  %header_view = buffer.view %w[%origin] : buffer -> view<4xi32>
  %header = vector.load %header_view[0] : view<4xi32> -> vector<4xi32>
  %scale, %minimum = func.call @g_k_metadata(%header, %group) : (vector<4xi32>, index) -> (f32, f32)
  %page = index.div %group, %c2 : index
  %nibble = index.rem %group, %c2 : index
  %page_add = index.scale %page, %page_bytes : index, offset -> offset
  %payload = index.add %origin, %header_bytes : offset
  %codes_origin = index.add %payload, %page_add : offset
  %code_view = buffer.view %w[%codes_origin] : buffer -> view<8xi32>
  %packed = vector.load %code_view[0] : view<8xi32> -> vector<8xi32>
  %shift_bits = index.mul %nibble, %c4 : index
  %shift_i32 = index.cast %shift_bits : index to i32
  %shift = vector.splat %shift_i32 : vector<8xi32>
  %shifted = vector.shrui %packed, %shift : vector<8xi32>
  %masked = vector.andi %shifted, %nibble_mask : vector<8xi32>
  %bytes = vector.bitcast %masked : vector<8xi32> to vector<32xi8>
  func.return %bytes, %scale, %minimum : vector<32xi8>, f32, f32
}

func.def inline @g_q5_k_icodes(%w: buffer, %origin: offset, %group: index) -> (vector<32xi8>, f32, f32) {
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %high_bytes = index.constant 16 : offset
  %low_bytes = index.constant 48 : offset
  %page_bytes = index.constant 32 : offset
  %nibble_mask = vector.constant 252645135 : vector<8xi32>
  %bit_mask = vector.constant 16843009 : vector<8xi32>
  %four = vector.constant 4 : vector<8xi32>
  %header_view = buffer.view %w[%origin] : buffer -> view<4xi32>
  %header = vector.load %header_view[0] : view<4xi32> -> vector<4xi32>
  %scale, %minimum = func.call @g_k_metadata(%header, %group) : (vector<4xi32>, index) -> (f32, f32)
  %page = index.div %group, %c2 : index
  %nibble = index.rem %group, %c2 : index
  %page_add = index.scale %page, %page_bytes : index, offset -> offset
  %payload = index.add %origin, %low_bytes : offset
  %codes_origin = index.add %payload, %page_add : offset
  %code_view = buffer.view %w[%codes_origin] : buffer -> view<8xi32>
  %packed = vector.load %code_view[0] : view<8xi32> -> vector<8xi32>
  %high_origin = index.add %origin, %high_bytes : offset
  %high_view = buffer.view %w[%high_origin] : buffer -> view<8xi32>
  %high_packed = vector.load %high_view[0] : view<8xi32> -> vector<8xi32>
  %shift_bits = index.mul %nibble, %c4 : index
  %shift_i32 = index.cast %shift_bits : index to i32
  %shift = vector.splat %shift_i32 : vector<8xi32>
  %shifted = vector.shrui %packed, %shift : vector<8xi32>
  %low = vector.andi %shifted, %nibble_mask : vector<8xi32>
  %group_i32 = index.cast %group : index to i32
  %group_shift = vector.splat %group_i32 : vector<8xi32>
  %high_shifted = vector.shrui %high_packed, %group_shift : vector<8xi32>
  %high_bit = vector.andi %high_shifted, %bit_mask : vector<8xi32>
  %high = vector.shli %high_bit, %four : vector<8xi32>
  %combined = vector.ori %low, %high : vector<8xi32>
  %bytes = vector.bitcast %combined : vector<8xi32> to vector<32xi8>
  func.return %bytes, %scale, %minimum : vector<32xi8>, f32, f32
}

// Q6_K: signed codes (q - 32) and d * scale of the two 16-value halves.
func.def inline @g_q6_k_icodes(%w: buffer, %origin: offset, %group: index) -> (vector<32xi8>, f32, f32) {
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %qh_base = index.constant 128 : index
  %sc_base = index.constant 192 : index
  %d_bytes = index.constant 208 : offset
  %thirty_two = vector.constant 32 : vector<32xi8>
  %low_mask = vector.constant 15 : vector<32xi8>
  %two_bits = vector.constant 3 : vector<32xi8>
  %four8 = vector.constant 4 : vector<32xi8>
  %bounded = index.assume %group [range(%group, 0, 7)] : index
  %half = index.div %bounded, %c4 : index
  %j = index.rem %bounded, %c4 : index
  %j_low = index.rem %j, %c2 : index
  %j_high = index.div %j, %c2 : index
  %ql_half = index.mul %half, %c64 : index
  %ql_part = index.mul %j_low, %c32 : index
  %ql_index = index.add %ql_half, %ql_part : index
  %qh_half = index.mul %half, %c32 : index
  %qh_index = index.add %qh_base, %qh_half : index
  %sc_half = index.mul %half, %c8 : index
  %sc_j = index.mul %j, %c2 : index
  %sc_part = index.add %sc_half, %sc_j : index
  %sc_index = index.add %sc_base, %sc_part : index
  %ql_add = index.cast %ql_index : index to offset
  %qh_add = index.cast %qh_index : index to offset
  %sc_add = index.cast %sc_index : index to offset
  %ql_origin = index.add %origin, %ql_add : offset
  %qh_origin = index.add %origin, %qh_add : offset
  %sc_origin = index.add %origin, %sc_add : offset
  %d_origin = index.add %origin, %d_bytes : offset
  %ql_view = buffer.view %w[%ql_origin] : buffer -> view<32xi8>
  %qh_view = buffer.view %w[%qh_origin] : buffer -> view<32xi8>
  %sc_view = buffer.view %w[%sc_origin] : buffer -> view<2xi8>
  %d_view = buffer.view %w[%d_origin] : buffer -> view<1xf16>
  %ql = vector.load %ql_view[0] : view<32xi8> -> vector<32xi8>
  %qh = vector.load %qh_view[0] : view<32xi8> -> vector<32xi8>
  %sc = vector.load %sc_view[0] : view<2xi8> -> vector<2xi8>
  %d_half = view.load %d_view[0] : view<1xf16> -> f16
  %d = scalar.extf %d_half : f16 to f32
  %shift_low = index.mul %j_high, %c4 : index
  %shift_high = index.mul %j, %c2 : index
  %shift_low32 = index.cast %shift_low : index to i32
  %shift_high32 = index.cast %shift_high : index to i32
  %shift_low8 = scalar.trunci %shift_low32 : i32 to i8
  %shift_high8 = scalar.trunci %shift_high32 : i32 to i8
  %shift_low_v = vector.splat %shift_low8 : vector<32xi8>
  %shift_high_v = vector.splat %shift_high8 : vector<32xi8>
  %ql_shifted = vector.shrui %ql, %shift_low_v : vector<32xi8>
  %low = vector.andi %ql_shifted, %low_mask : vector<32xi8>
  %qh_shifted = vector.shrui %qh, %shift_high_v : vector<32xi8>
  %qh_bits = vector.andi %qh_shifted, %two_bits : vector<32xi8>
  %high = vector.shli %qh_bits, %four8 : vector<32xi8>
  %combined = vector.ori %low, %high : vector<32xi8>
  %signed = vector.subi %combined, %thirty_two : vector<32xi8>
  %s0_i8 = vector.extract %sc[0] : vector<2xi8> -> i8
  %s1_i8 = vector.extract %sc[1] : vector<2xi8> -> i8
  %s0f = scalar.sitofp %s0_i8 : i8 to f32
  %s1f = scalar.sitofp %s1_i8 : i8 to f32
  %s0 = scalar.mulf %s0f, %d : f32
  %s1 = scalar.mulf %s1f, %d : f32
  func.return %signed, %s0, %s1 : vector<32xi8>, f32, f32
}

// [rows, width] F32 -> int8 activations for the _q8 matvecs (llama.cpp quantize_q8_1): per 32-value group
// d = amax / 127 (stored F16-rounded), q = round(x / d). Layout: rows x groups x 32 int8 codes, then at byte
// rows * width: rows x groups x (d, sum of q) F32. One lane per group, launched beside g_row_scale_f16.
kernel.def export("g_quant_q8") @g_quant_q8(%rows: index, %width: index) {
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  %c63 = index.constant 63 : index
  %c64 = index.constant 64 : index
  %groups = index.div %width, %c32 : index
  %gp = index.add %groups, %c63 : index
  %wgs = index.div %gp, %c64 : index
  kernel.launch.config workgroups(%wgs, %rows, %c1) workgroup_size(%c64, %c1, %c1) : index
} launch(%rows_arg: index, %width_arg: index, %x: buffer, %xq: buffer) {
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 8192)] : index
  %width = index.assume %width_arg [range(%width_arg, 32, 262144), multiple_of(%width_arg, 32)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %zero_f = scalar.constant 0.0 : f32
  %c127 = scalar.constant 127.0 : f32
  %zero_i = scalar.constant 0 : i32
  %zero8i = vector.constant 0 : vector<8xi32>
  %ones8 = vector.constant 1 : vector<32xi8>
  %zero32 = vector.constant 0.0 : vector<32xf32>
  %groups = index.div %width, %c32 : index
  %wg = kernel.workgroup.id<x> : index
  %row_id = kernel.workgroup.id<y> : index
  %row = index.assume %row_id [lt(%row_id, %rows)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 63)] : index
  %gbase = index.mul %wg, %c64 : index
  %gi = index.add %gbase, %lane : index
  %ok = index.cmp ult, %gi, %groups : index
  scf.if %ok {
    %g = index.assume %gi [lt(%gi, %groups)] : index
    %x_view = buffer.view %x[%zero] : buffer -> view<[%rows]x[%groups]x32xf32>
    %q_view = buffer.view %xq[%zero] : buffer -> view<[%rows]x[%groups]x8xi32>
    %m_bytes = index.cast %width : index to offset
    %m_off = index.scale %rows, %m_bytes : index, offset -> offset
    %m_view = buffer.view %xq[%m_off] : buffer -> view<[%rows]x[%groups]x2xf32>
    %v = vector.load %x_view[%row, %g, %c0] : view<[%rows]x[%groups]x32xf32> -> vector<32xf32>
    %a = vector.absf %v : vector<32xf32>
    %amax = vector.reduce<maxnumf> %a, %zero_f : vector<32xf32>, f32
    %d = scalar.divf %amax, %c127 : f32
    %nz = scalar.cmpf one, %amax, %zero_f : f32
    %dv = vector.splat %d : vector<32xf32>
    %sc = vector.divf %v, %dv : vector<32xf32>
    %rd = vector.roundf %sc : vector<32xf32>
    %nzv = vector.splat %nz : vector<32xi1>
    %rs = vector.select %nzv, %rd, %zero32 : vector<32xf32>
    %q = vector.fptosi %rs : vector<32xf32> to vector<32xi8>
    %qw = vector.bitcast %q : vector<32xi8> to vector<8xi32>
    vector.store %qw, %q_view[%row, %g, %c0] : vector<8xi32>, view<[%rows]x[%groups]x8xi32>
    %qs = vector.dot4i<s8s8> %q, %ones8, %zero8i : vector<32xi8>, vector<32xi8>, vector<8xi32>
    %qsi = vector.reduce<addi> %qs, %zero_i : vector<8xi32>, i32
    %qsf = scalar.sitofp %qsi : i32 to f32
    %dh = scalar.fptrunc %d : f32 to f16
    %da = scalar.extf %dh : f16 to f32
    %meta = vector.from_elements %da, %qsf : vector<2xf32>
    vector.store %meta, %m_view[%row, %g, %c0] : vector<2xf32>, view<[%rows]x[%groups]x2xf32>
  }
  kernel.return
}
"""


# emitted only when an F16 matrix is used (see main)
F16_VALUES = r"""
// F16: 32 values (64 bytes) of a row.
func.def inline @g_f16_values(%w: buffer, %origin: offset, %group: index) -> (vector<32xf32>) {
  %view = buffer.view %w[%origin] : buffer -> view<32xf16>
  %raw = vector.load %view[0] : view<32xf16> -> vector<32xf16>
  %values = vector.extf %raw : vector<32xf16> to vector<32xf32>
  func.return %values : vector<32xf32>
}
"""

# emitted only when a BF16 matrix is used (see main): the f32-activation matvec and (through gen_gemm.py) the GEMM
BF16_VALUES = r"""
// BF16: 32 values (64 bytes) of a row.
func.def inline @g_bf16_values(%w: buffer, %origin: offset, %group: index) -> (vector<32xf32>) {
  %view = buffer.view %w[%origin] : buffer -> view<32xbf16>
  %raw = vector.load %view[0] : view<32xbf16> -> vector<32xbf16>
  %values = vector.extf %raw : vector<32xbf16> to vector<32xf32>
  func.return %values : vector<32xf32>
}
"""

# emitted only when an F32 matrix is used (see main)
F32_VALUES = r"""
// F32: 32 values (128 bytes) of a row (small projections some GGUFs keep unquantized).
func.def inline @g_f32_values(%w: buffer, %origin: offset, %group: index) -> (vector<32xf32>) {
  %view = buffer.view %w[%origin] : buffer -> view<32xf32>
  %values = vector.load %view[0] : view<32xf32> -> vector<32xf32>
  func.return %values : vector<32xf32>
}
"""


# emitted only when an MXFP4 tensor is used (see main; gpt-oss keeps its experts in MXFP4 inside its "Q4_K_M" files).
# Values for the MoE expert matvecs (f32 activations) and the grouped GEMMs' dequant16 (gen_gemm.HELPER).
MXFP4 = r"""
// MXFP4 block (17 bytes, ggml block_mxfp4): E8M0 exponent e, 16 bytes of E2M1 nibbles (value j < 16: qs[j] & 15,
// j >= 16: qs[j - 16] >> 4); value j = kvalues_mxfp4[nibble] * 2^(e - 128) (ggml GGML_E8M0_TO_FP32_HALF; the
// table holds twice the E2M1 values: 0 1 2 3 4 6 8 12, then the same negated).
func.def inline @g_mxfp4_values(%w: buffer, %origin: offset, %group: index) -> (vector<32xf32>) {
  %one = index.constant 1 : offset
  %e_view = buffer.view %w[%origin] : buffer -> view<1xi8>
  %code_origin = index.add %origin, %one : offset
  %code_view = buffer.view %w[%code_origin] : buffer -> view<16xi8>
  %e8 = view.load %e_view[0] : view<1xi8> -> i8
  %e = scalar.extui %e8 : i8 to i32
  %c1_i = scalar.constant 1 : i32
  %c2_i = scalar.constant 2 : i32
  %c23_i = scalar.constant 23 : i32
  %tiny_base = scalar.constant 2097152 : i32
  %e_low = scalar.cmpi ult, %e, %c2_i : i32
  %tiny = scalar.shli %tiny_base, %e : i32
  %em1 = scalar.subi %e, %c1_i : i32
  %normal = scalar.shli %em1, %c23_i : i32
  %bits = scf.select %e_low, %tiny, %normal : i32
  %d = scalar.bitcast %bits : i32 to f32
  %payload = vector.load %code_view[0] : view<16xi8> -> vector<16xi8>
  %mask = vector.constant 15 : vector<16xi8>
  %four = vector.constant 4 : vector<16xi8>
  %lower = vector.andi %payload, %mask : vector<16xi8>
  %shifted = vector.shrui %payload, %four : vector<16xi8>
  %upper = vector.andi %shifted, %mask : vector<16xi8>
  %k0 = scalar.constant 0 : i8
  %k1 = scalar.constant 1 : i8
  %k2 = scalar.constant 2 : i8
  %k3 = scalar.constant 3 : i8
  %k4 = scalar.constant 4 : i8
  %k5 = scalar.constant 6 : i8
  %k6 = scalar.constant 8 : i8
  %k7 = scalar.constant 12 : i8
  %k9 = scalar.constant -1 : i8
  %k10 = scalar.constant -2 : i8
  %k11 = scalar.constant -3 : i8
  %k12 = scalar.constant -4 : i8
  %k13 = scalar.constant -6 : i8
  %k14 = scalar.constant -8 : i8
  %k15 = scalar.constant -12 : i8
  %codebook = vector.from_elements %k0, %k1, %k2, %k3, %k4, %k5, %k6, %k7, %k0, %k9, %k10, %k11, %k12, %k13, %k14, %k15 : vector<16xi8>
  %lo_codes = vector.table.lookup %codebook[%lower] : vector<16xi8>, vector<16xi8> -> vector<16xi8>
  %hi_codes = vector.table.lookup %codebook[%upper] : vector<16xi8>, vector<16xi8> -> vector<16xi8>
  %lo = vector.sitofp %lo_codes : vector<16xi8> to vector<16xf32>
  %hi = vector.sitofp %hi_codes : vector<16xi8> to vector<16xf32>
  %codes = vector.concat<0> %lo, %hi : vector<16xf32>, vector<16xf32> -> vector<32xf32>
  %ds = vector.splat %d : vector<32xf32>
  %values = vector.mulf %codes, %ds : vector<32xf32>
  func.return %values : vector<32xf32>
}
"""


# emitted only when a Q5_0 tensor is used (see main): llama.cpp's Q4_K_M keeps projections whose rows are not a multiple
# of 256 in Q5_0 (Qwen2.5-0.5B: hidden 896). Integer codes for the int8 matvecs and head (ACT_Q8), values for the rest
# (embedding, f32 matvecs, the GEMM's dequant16 through gen_gemm.HELPER).
Q5_0 = r"""
// Q5_0 block (22 bytes): F16 d, 32 high bits qh (bit j: value j), 16 bytes of nibbles (value j < 16: qs[j] & 15,
// j >= 16: qs[j - 16] >> 4); value j = d * (code_j - 16), code_j = nibble | bit_j << 4. Codes 0..31 with minimum
// 16 d, so the int8 dot subtracts 16 x (activation code sum) like a K-quant minimum.
func.def inline @g_q5_0_icodes(%w: buffer, %origin: offset, %group: index) -> (vector<32xi8>, f32, f32) {
  %two = index.constant 2 : offset
  %six = index.constant 6 : offset
  %scale_view = buffer.view %w[%origin] : buffer -> view<1xf16>
  %high_origin = index.add %origin, %two : offset
  %high_view = buffer.view %w[%high_origin] : buffer -> view<4xi8>
  %code_origin = index.add %origin, %six : offset
  %code_view = buffer.view %w[%code_origin] : buffer -> view<16xi8>
  %scale_half = view.load %scale_view[0] : view<1xf16> -> f16
  %scale = scalar.extf %scale_half : f16 to f32
  %packed = vector.load %code_view[0] : view<16xi8> -> vector<16xi8>
  %nibble_mask = vector.constant 15 : vector<16xi8>
  %four = vector.constant 4 : vector<16xi8>
  %low = vector.andi %packed, %nibble_mask : vector<16xi8>
  %high = vector.shrui %packed, %four : vector<16xi8>
  %nibbles = vector.concat<0> %low, %high : vector<16xi8>, vector<16xi8> -> vector<32xi8>
  %nibble_words = vector.bitcast %nibbles : vector<32xi8> to vector<8xi32>
  %high_bytes = vector.load %high_view[0] : view<4xi8> -> vector<4xi8>
  %high_word_v = vector.bitcast %high_bytes : vector<4xi8> to vector<1xi32>
  %high_word = vector.extract %high_word_v[0] : vector<1xi32> -> i32
  %high_splat = vector.splat %high_word : vector<8xi32>
  // word k holds values 4k..4k+3 (one per byte): bits 4k..4k+3 of qh, one into bit 4 of each byte
  %zero_i = scalar.constant 0 : i32
  %step_i = scalar.constant 4 : i32
  %word_shift = vector.iota %zero_i step %step_i : vector<8xi32>
  %bits = vector.shrui %high_splat, %word_shift : vector<8xi32>
  %m1 = vector.constant 1 : vector<8xi32>
  %m2 = vector.constant 2 : vector<8xi32>
  %m4 = vector.constant 4 : vector<8xi32>
  %m8 = vector.constant 8 : vector<8xi32>
  %s4 = vector.constant 4 : vector<8xi32>
  %s11 = vector.constant 11 : vector<8xi32>
  %s18 = vector.constant 18 : vector<8xi32>
  %s25 = vector.constant 25 : vector<8xi32>
  %b0m = vector.andi %bits, %m1 : vector<8xi32>
  %b1m = vector.andi %bits, %m2 : vector<8xi32>
  %b2m = vector.andi %bits, %m4 : vector<8xi32>
  %b3m = vector.andi %bits, %m8 : vector<8xi32>
  %b0 = vector.shli %b0m, %s4 : vector<8xi32>
  %b1 = vector.shli %b1m, %s11 : vector<8xi32>
  %b2 = vector.shli %b2m, %s18 : vector<8xi32>
  %b3 = vector.shli %b3m, %s25 : vector<8xi32>
  %h01 = vector.ori %b0, %b1 : vector<8xi32>
  %h23 = vector.ori %b2, %b3 : vector<8xi32>
  %high_bits = vector.ori %h01, %h23 : vector<8xi32>
  %code_words = vector.ori %nibble_words, %high_bits : vector<8xi32>
  %codes = vector.bitcast %code_words : vector<8xi32> to vector<32xi8>
  %sixteen = scalar.constant 16.0 : f32
  %minimum = scalar.mulf %scale, %sixteen : f32
  func.return %codes, %scale, %minimum : vector<32xi8>, f32, f32
}

func.def inline @g_q5_0_values(%w: buffer, %origin: offset, %group: index) -> (vector<32xf32>) {
  %codes, %scale, %minimum = func.call @g_q5_0_icodes(%w, %origin, %group) : (buffer, offset, index) -> (vector<32xi8>, f32, f32)
  %codes_f = vector.uitofp %codes : vector<32xi8> to vector<32xf32>
  %sixteen = vector.constant 16.0 : vector<32xf32>
  %centered = vector.subf %codes_f, %sixteen : vector<32xf32>
  %ds = vector.splat %scale : vector<32xf32>
  %values = vector.mulf %centered, %ds : vector<32xf32>
  func.return %values : vector<32xf32>
}
"""


def iq_decode(fmt, wi, w, ok, o, within):
    """ACT_Q8 weight side: integer codes %wq{wi} (+ scale %sc{wi}, and %mn{wi}: Q4_K/Q5_K minimum, Q6_K second
    half scale) of the lane's 32-value group, once per row."""
    two = fmt != "q8_0"
    rt = "vector<32xi8>, f32, f32" if two else "vector<32xi8>, f32"
    return (f"""    %wq{wi}, %sc{wi}""" + (f", %mn{wi}" if two else "") + f""" = scf.if {ok} -> ({rt}) {{
      %v, %a""" + (", %b" if two else "") + f""" = func.call @g_{fmt}_icodes(%{w}, {o}, {within}) : (buffer, offset, index) -> ({rt})
      scf.yield %v, %a""" + (", %b" if two else "") + f""" : {rt}
    }} else {{
      scf.yield %zero32i8, %zero_f""" + (", %zero_f" if two else "") + f""" : {rt}
    }}""" + "".join(f"\n    %wk{wi}_{k} = vector.slice %wq{wi}[{4 * k}] : vector<32xi8> -> vector<4xi8>" for k in range(8)))


def iq_dot(fmt, wi, j, out):
    """ACT_Q8: this lane's contribution of weight group %wq{wi} to token j (int8 codes %q{j}, scale %da{j},
    code sum %xs{j}), as llama.cpp's vec_dot_*_q8_1: an exact integer dot (eight accumulating dot4 steps), then
    F32 scaling."""
    def chain(k0, k1, name):
        t, acc = [], "%zero1i"
        for k in range(k0, k1):
            t.append(f"""    %qk{j}_{k}_{wi}{name} = vector.slice %q{j}[{4 * k}] : vector<32xi8> -> vector<4xi8>
    %ac{j}_{k}_{wi}{name} = vector.dot4i<s8s8> %wk{wi}_{k}, %qk{j}_{k}_{wi}{name}, {acc} : vector<4xi8>, vector<4xi8>, vector<1xi32>""")
            acc = f"%ac{j}_{k}_{wi}{name}"
        t.append(f"    %ie{wi}_{j}{name} = vector.extract {acc}[0] : vector<1xi32> -> i32")
        return t, f"%ie{wi}_{j}{name}"
    if fmt == "q6_k":
        t0, ilo = chain(0, 4, "l"); t1, ihi = chain(4, 8, "h")
        t = t0 + t1
        t.append(f"""    %flo{wi}_{j} = scalar.sitofp {ilo} : i32 to f32
    %fhi{wi}_{j} = scalar.sitofp {ihi} : i32 to f32
    %plo{wi}_{j} = scalar.mulf %flo{wi}_{j}, %sc{wi} : f32
    %pq{wi}_{j} = scalar.fmaf %fhi{wi}_{j}, %mn{wi}, %plo{wi}_{j} : f32
    {out} = scalar.mulf %pq{wi}_{j}, %da{j} : f32""")
        return "\n".join(t)
    t, di = chain(0, 8, "")
    t.append(f"    %df{wi}_{j} = scalar.sitofp {di} : i32 to f32")
    if fmt == "q8_0":
        t.append(f"""    %wd{wi}_{j} = scalar.mulf %sc{wi}, %da{j} : f32
    {out} = scalar.mulf %wd{wi}_{j}, %df{wi}_{j} : f32""")
    else:
        t.append(f"""    %pd{wi}_{j} = scalar.mulf %df{wi}_{j}, %sc{wi} : f32
    %pm{wi}_{j} = scalar.mulf %xs{j}, %mn{wi} : f32
    %pq{wi}_{j} = scalar.subf %pd{wi}_{j}, %pm{wi}_{j} : f32
    {out} = scalar.mulf %pq{wi}_{j}, %da{j} : f32""")
    return "\n".join(t)


def iq_load(j, ok, tok, groups, tokens="%tokens"):
    """ACT_Q8 activation side: token `tok`'s int8 codes, F16-rounded scale and code sum for this lane's group."""
    return f"""  %q{j}, %da{j}, %xs{j} = scf.if {ok} -> (vector<32xi8>, f32, f32) {{
    %tk{j} = index.assume {tok} [lt({tok}, {tokens})] : index
    %cw = vector.load %xq_view[%tk{j}, %g, %c0] : view<[{tokens}]x{groups}x8xi32> -> vector<8xi32>
    %cb = vector.bitcast %cw : vector<8xi32> to vector<32xi8>
    %mt = vector.load %xm_view[%tk{j}, %g, %c0] : view<[{tokens}]x{groups}x2xf32> -> vector<2xf32>
    %m0 = vector.extract %mt[0] : vector<2xf32> -> f32
    %m1 = vector.extract %mt[1] : vector<2xf32> -> f32
    scf.yield %cb, %m0, %m1 : vector<32xi8>, f32, f32
  }} else {{
    scf.yield %zero32i8, %zero_f, %zero_f : vector<32xi8>, f32, f32
  }}"""


BFLY = (16, 8, 4, 2, 1)


def bfly_setup(TG):
    """Lane constants of bfly_store: bit d of the lane, the token whose wave sum this lane ends up holding (%bt_idx)
    and whether it stores it (%bt_st)."""
    t = ["", "  %bw32 = scalar.constant 32 : i32", "  %bc2 = index.constant 2 : index"]
    T, idx, st = TG, [], []
    for d in BFLY:
        t.append(f"""  %bd{d} = scalar.constant {d} : i32
  %bdi{d} = index.constant {d} : index
  %bq{d} = index.div %lane, %bdi{d} : index
  %br{d} = index.rem %bq{d}, %bc2 : index
  %bb{d} = index.cmp ne, %br{d}, %c0 : index""")
        if T > 1:
            T //= 2
            t.append(f"  %bh{d} = index.constant {T} : index\n  %bo{d} = scf.select %bb{d}, %bh{d}, %c0 : index")
            idx.append(f"%bo{d}")
        else:
            st.append(f"%bb{d}")
    acc = "%c0"
    for k, v in enumerate(idx):
        t.append(f"  %bi{k} = index.add {acc}, {v} : index"); acc = f"%bi{k}"
    t.append(f"  %bt_idx_v = index.add {acc}, %c0 : index\n  %bt_idx = index.assume %bt_idx_v [range(%bt_idx_v, 0, {TG - 1})] : index")
    t.append("  %bt_true = scalar.constant true : i1")
    acc = "%bt_true"
    for k, b in enumerate(st):
        t.append(f"  %bn{k} = scalar.xori {b}, %bt_true : i1\n  %bs{k} = scalar.andi {acc}, %bn{k} : i1"); acc = f"%bs{k}"
    t.append(f"  %bt_st = scalar.andi {acc}, %bt_true : i1")
    return "\n".join(t)


# BFLY_PERM: an int8 gate/up (swiglu) matvec of 8+ tokens has each lane hold token j ^ %bt_idx in slot j (bfly_slot),
# so every butterfly step keeps slots [0, h) and sends slots [h, 2h) without the two selects per pair (two weights
# per row: 60 of ~880 instructions per row at 16 tokens). The per-token pairing tree is unchanged, so results are
# bit-identical. Muse-Glimmer gate/up 6656x19968 t16: -3.5 % at 16 rows, -5.5 % with 32 (iq_r). Single-weight
# projections keep plain slots: the permuted activation loads are uncoalesced, and that cost more than the selects
# (down 19968x6656 t16 +9 %). MV_BFLY_PERM=0 turns it off.
BFLY_PERM = os.environ.get("MV_BFLY_PERM", "1") == "1"


def bfly_slot(j, TG, base, name, perm):
    """IR for %{name}: token index base + j, or base + (j ^ %bt_idx) with perm (BFLY_PERM)."""
    if not perm or TG == 1:
        return f"  %tj{j}_off = index.constant {j} : index\n  %{name} = index.add {base}, %tj{j}_off : index"
    return (("  %bt_i32 = index.cast %bt_idx : index to i32\n" if j == 0 else "")
            + f"  %{name}_c = scalar.constant {j} : i32\n  %{name}_x = scalar.xori %{name}_c, %bt_i32 : i32\n"
            f"  %{name}_o = index.cast %{name}_x : i32 to index\n  %{name} = index.add {base}, %{name}_o : index")


def bfly_store(vals, p, store, presorted=False):
    """Wave sums of the lanes' per-token partials `vals` (TG of them), stored by store(value): a recursive-halving
    butterfly (each step a lane keeps half of its tokens and adds its xor-partner's half) while several tokens remain,
    then plain xor steps. Every token's sum is the same xor-16/8/4/2/1 pairing tree for any TG (and for TG = 1), so
    multi-token variants round exactly like the 1-token kernel, with ~TG + 5 shuffles instead of 5 TG."""
    t, T = [], len(vals)
    for d in BFLY:
        if T > 1 and presorted:
            h = T // 2
            nv = []
            for i in range(h):
                q = f"%{p}_{d}_{i}"
                t.append(f"""    {q}r, {q}u = kernel.subgroup.shuffle<xor> {vals[i + h]}, %bd{d}, %bw32 : f32, i32, i32
    {q} = scalar.addf {vals[i]}, {q}r : f32""")
                nv.append(q)
            vals, T = nv, h
        elif T > 1:
            h = T // 2
            nv = []
            for i in range(h):
                lo, hi, q = vals[i], vals[i + h], f"%{p}_{d}_{i}"
                t.append(f"""    {q}s = scf.select %bb{d}, {lo}, {hi} : f32
    {q}k = scf.select %bb{d}, {hi}, {lo} : f32
    {q}r, {q}u = kernel.subgroup.shuffle<xor> {q}s, %bd{d}, %bw32 : f32, i32, i32
    {q} = scalar.addf {q}k, {q}r : f32""")
                nv.append(q)
            vals, T = nv, h
        else:
            q = f"%{p}_{d}"
            t.append(f"""    {q}r, {q}u = kernel.subgroup.shuffle<xor> {vals[0]}, %bd{d}, %bw32 : f32, i32, i32
    {q} = scalar.addf {vals[0]}, {q}r : f32""")
            vals = [q]
    t.append(f"    scf.if %bt_st {{\n      {store(vals[0])}\n    }}")
    return "\n".join(t)


def mv(fmt, K, R=16, TG=1, swiglu=False, suffix="", iq=False, pair=False):
    """Decode matrix-vector: a workgroup covers one K row with one lane per
    32-value group (lanes = groups rounded up to a wave), R output rows and TG
    tokens. Each lane keeps its x group for the TG tokens in registers and
    decodes each weight group once for all of them, so weights stream once per
    token group and activations once per workgroup. Waves reduce with shuffles;
    one barrier folds the wave partials.
    pair (int8 only): two same-shape weights of one input (attention q + gate, k + v) in one launch, y = x W^T and
    y2 = x W2^T, each row rounded exactly as the single-weight kernel rounds it (see PAIR_PROJ)."""
    groups = K // 32
    assert not pair or (iq and not swiglu and groups <= 1024)
    if groups > 1024:
        return mv_wide(fmt, K, R, TG, swiglu, suffix)
    lanes = (groups + 31) // 32 * 32
    waves = lanes // 32
    rb = row_bytes(fmt, K)
    nw = 2 if swiglu or pair else 1
    name = f"g_mv_{fmt}_k{K}" + ("_swiglu" if swiglu else "_pair" if pair else "") + (f"_t{TG}" if TG > 1 else "") + ("_q8" if iq else "") + suffix
    # int8 variants above MV_CHUNK tokens run the token chunks one after another in the workgroup: registers stay at
    # the chunk's level and later chunks re-read this workgroup's weight rows from cache, not DRAM.
    CHK = min(TG, int(os.environ.get("MV_CHUNK", "16"))) if iq else TG   # chunks of 8 measured no faster
    NCH = TG // CHK
    assert NCH * CHK == TG
    PERM = iq and nw == 2 and BFLY_PERM and CHK >= 8
    wargs = "%w: buffer, %w2: buffer" if nw == 2 else "%w: buffer"
    yargs = "%y: buffer, %y2: buffer" if pair else "%y: buffer"
    L = [f"""
kernel.def export("{name}") @{name}(%n: index, %tokens: index) {{
  %c1 = index.constant 1 : index
  %lanes = index.constant {lanes} : index
  %rows_per = index.constant {R} : index
  %tg = index.constant {TG} : index
  %tg_less = index.constant {TG - 1} : index
  %row_groups = index.div %n, %rows_per : index
  %tg_pad = index.add %tokens, %tg_less : index
  %token_groups = index.div %tg_pad, %tg : index
  kernel.launch.config workgroups(%token_groups, %row_groups, %c1) workgroup_size(%lanes, %c1, %c1) : index
}} launch(%n_arg: index, %tokens_arg: index, %accumulate: i32, %x: buffer, {wargs}, {yargs}) {{
  %n = index.assume %n_arg [range(%n_arg, {R}, 262144), multiple_of(%n_arg, {R})] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %group_id = kernel.workgroup.id<y> : index
  %tgroup_id = kernel.workgroup.id<x> : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, {lanes - 1})] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, {waves - 1})] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c8 = index.constant 8 : index
  %rows_per = index.constant {R} : index
  %tg = index.constant {TG} : index
  %cW = index.constant {waves} : index
  %groups = index.constant {groups} : index
  %row_bytes = index.constant {rb} : offset
  %block_bytes = index.constant {FMT[fmt]} : offset
  %zero_f = scalar.constant 0.0 : f32
  %zero32 = vector.constant 0.0 : vector<32xf32>
""" + (f"""
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{groups}x32xf32>""" if not iq else f"""
  %xq_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{groups}x8xi32>
  %xm_k = index.constant {K} : offset
  %xm_off = index.scale %tokens, %xm_k : index, offset -> offset
  %xm_view = buffer.view %x[%xm_off] : buffer -> view<[%tokens]x{groups}x2xf32>""") + f"""
  %row_base = index.mul %group_id, %rows_per : index
  %token0 = index.mul %tgroup_id, %tg : index
  %ok = index.cmp ult, %workitem, %groups : index
  %g_last = index.constant {groups - 1} : index
  %g_c = index.min %workitem, %g_last : index
  %g = index.assume %g_c [range(%g_c, 0, {groups - 1})] : index
  %cgpb = index.constant {SB[fmt] // 32} : index
  %blk = index.div %g, %cgpb : index
  %within = index.rem %g, %cgpb : index
  %ba = index.scale %blk, %block_bytes : index, offset -> offset""" + ("""
  %zero_i = scalar.constant 0 : i32
  %zero8i = vector.constant 0 : vector<8xi32>
  %zero32i8 = vector.constant 0 : vector<32xi8>
  %zero1i = vector.constant 0 : vector<1xi32>""" + bfly_setup(CHK) if iq else "")]
    PV = f"""  %part_bytes = index.constant {waves * R * TG * nw * 4} : offset
  %parts = buffer.alloca<workgroup> align(16) %part_bytes : buffer
  %pv = buffer.view %parts[%zero] : buffer -> view<{nw}x{waves}x{R}x{TG}xf32>
  %first = index.cmp eq, %lane, %c0 : index"""
    if NCH > 1:
        L.append(PV + f"""
  %nch = index.constant {NCH} : index
  %chk = index.constant {CHK} : index
  scf.for %ch = [%c0 to %nch step %c1] {{
  %chb = index.mul %ch, %chk : index
  %chbase = index.add %token0, %chb : index""")
    else:
        L.append("  %chb = index.constant 0 : index\n  %chbase = index.add %token0, %chb : index")
    for j in range(CHK):
        L.append((f"""  %tj{j}_off = index.constant {j} : index
  %tok{j}_raw = index.add %chbase, %tj{j}_off : index
  %live{j} = index.cmp ult, %tok{j}_raw, %tokens : index
  %use{j} = scalar.andi %live{j}, %ok : i1
  %x{j} = scf.if %use{j} -> (vector<32xf32>) {{
    %tk{j} = index.assume %tok{j}_raw [lt(%tok{j}_raw, %tokens)] : index
    %v = vector.load %x_view[%tk{j}, %g, %c0] : view<[%tokens]x{groups}x32xf32> -> vector<32xf32>
    scf.yield %v : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}
  %xs{j} = vector.reduce<addf> %x{j}, %zero_f : vector<32xf32>, f32""" if not iq else bfly_slot(j, CHK, "%token0", f"tok{j}_raw", PERM) + f"""
  %live{j} = index.cmp ult, %tok{j}_raw, %tokens : index
  %use{j} = scalar.andi %live{j}, %ok : i1
  %q{j}, %da{j}, %xs{j} = scf.if %use{j} -> (vector<32xi8>, f32, f32) {{
    %tk{j} = index.assume %tok{j}_raw [lt(%tok{j}_raw, %tokens)] : index
    %cw = vector.load %xq_view[%tk{j}, %g, %c0] : view<[%tokens]x{groups}x8xi32> -> vector<8xi32>
    %cb = vector.bitcast %cw : vector<8xi32> to vector<32xi8>
    %mt = vector.load %xm_view[%tk{j}, %g, %c0] : view<[%tokens]x{groups}x2xf32> -> vector<2xf32>
    %m0 = vector.extract %mt[0] : vector<2xf32> -> f32
    %m1 = vector.extract %mt[1] : vector<2xf32> -> f32
    scf.yield %cb, %m0, %m1 : vector<32xi8>, f32, f32
  }} else {{
    scf.yield %zero32i8, %zero_f, %zero_f : vector<32xi8>, f32, f32
  }}"""))
    L.append(("" if NCH > 1 else PV + "\n") + (f"""  %bt_sv = index.add %bt_idx, %chb : index
  %bt_store = index.assume %bt_sv [range(%bt_sv, 0, {TG - 1})] : index
""" if iq else "") + f"""  scf.for %r = [%c0 to %rows_per step %c1] {{
    %row_value = index.add %row_base, %r : index
    %row = index.assume %row_value [lt(%row_value, %n)] : index
    %ro = index.scale %row, %row_bytes : index, offset -> offset
    %rb = index.assume %r [range(%r, 0, {R - 1})] : index
    %o = index.add %ro, %ba : offset""")
    for wi, w in enumerate(["w", "w2"] if nw == 2 else ["w"]):
        if iq:
            # integer codes once per row; per token one dot4 contraction against the int8 activations
            L.append(iq_decode(fmt, wi, w, "%ok", "%o", "%within"))
            for j in range(CHK):
                L.append(iq_dot(fmt, wi, j, f"%pa{wi}_{j}"))
            L.append(bfly_store([f"%pa{wi}_{j}" for j in range(CHK)], f"b{wi}",
                                lambda v: f"view.store {v}, %pv[{wi}, %wave, %rb, %bt_store] : f32, view<{nw}x{waves}x{R}x{TG}xf32>", presorted=PERM))
            continue
        if fmt == "q4_k":
            # codes once per row; per token scale * dot(codes, x) - minimum * sum(x)
            L.append(f"""    %wv{wi}, %sc{wi}, %mn{wi} = scf.if %ok -> (vector<32xf32>, f32, f32) {{
      %v, %a, %b = func.call @g_q4_k_codes(%{w}, %o, %within) : (buffer, offset, index) -> (vector<32xf32>, f32, f32)
      scf.yield %v, %a, %b : vector<32xf32>, f32, f32
    }} else {{
      scf.yield %zero32, %zero_f, %zero_f : vector<32xf32>, f32, f32
    }}""")
        else:
            L.append(f"""    %wv{wi} = scf.if %ok -> (vector<32xf32>) {{
      %v = func.call @g_{fmt}_values(%{w}, %o, %within) : (buffer, offset, index) -> (vector<32xf32>)
      scf.yield %v : vector<32xf32>
    }} else {{
      scf.yield %zero32 : vector<32xf32>
    }}""")
        for j in range(TG):
            if fmt == "q4_k":
                L.append(f"""    %qd{wi}_{j} = vector.dotf<reassoc|contract> %wv{wi}, %x{j}, %zero_f : vector<32xf32>, vector<32xf32>, f32
    %qm{wi}_{j} = scalar.mulf %mn{wi}, %xs{j} : f32
    %pa{wi}_{j} = scalar.fmaf %qd{wi}_{j}, %sc{wi}, %qm{wi}_{j} : f32""".replace("scalar.fmaf %qd", "scalar.fmaf %qd") .replace(f"%pa{wi}_{j} = scalar.fmaf %qd{wi}_{j}, %sc{wi}, %qm{wi}_{j} : f32", f"%qs{wi}_{j} = scalar.mulf %qd{wi}_{j}, %sc{wi} : f32\n    %pa{wi}_{j} = scalar.subf %qs{wi}_{j}, %qm{wi}_{j} : f32") + f"""
    %s{wi}_{j} = kernel.subgroup.reduce<addf> %pa{wi}_{j} : f32
    scf.if %first {{
      view.store %s{wi}_{j}, %pv[{wi}, %wave, %rb, {j}] : f32, view<{nw}x{waves}x{R}x{TG}xf32>
    }}""")
                continue
            L.append(f"""    %pa{wi}_{j} = vector.dotf<reassoc|contract> %wv{wi}, %x{j}, %zero_f : vector<32xf32>, vector<32xf32>, f32
    %s{wi}_{j} = kernel.subgroup.reduce<addf> %pa{wi}_{j} : f32
    scf.if %first {{
      view.store %s{wi}_{j}, %pv[{wi}, %wave, %rb, {j}] : f32, view<{nw}x{waves}x{R}x{TG}xf32>
    }}""")
    L.append("  }" + ("\n  }" if NCH > 1 else ""))
    L.append(f"""  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %outputs = index.constant {R * TG} : index
  %lanes_step = index.constant {lanes} : index
  // R*TG outputs may exceed the lane count (one-wave rows): lanes stride over them.
  scf.for %wr0 = [%workitem to %outputs step %lanes_step] {{
    %wr = index.assume %wr0 [range(%wr0, 0, {R * TG - 1})] : index
    %rr_v = index.div %wr, %tg : index
    %rr = index.assume %rr_v [range(%rr_v, 0, {R - 1})] : index
    %jj_v = index.rem %wr, %tg : index
    %jj = index.assume %jj_v [range(%jj_v, 0, {TG - 1})] : index
    %tok_raw = index.add %token0, %jj : index
    %tok_live = index.cmp ult, %tok_raw, %tokens : index
    scf.if %tok_live {{""")
    for wi in range(nw):
        L.append(f"""      %t{wi} = scf.for %ww = [%c0 to %cW step %c1](%acc = %zero_f : f32) -> (f32) {{
        %bw = index.assume %ww [range(%ww, 0, {waves - 1})] : index
        %p = view.load %pv[{wi}, %bw, %rr, %jj] : view<{nw}x{waves}x{R}x{TG}xf32> -> f32
        %nx = scalar.addf %acc, %p : f32
        scf.yield %nx : f32
      }}""")
    if swiglu:
        if GELU:
            L.append(gelu_scalar("silu", "t0") + "\n      %value = scalar.mulf %silu, %t1 : f32")
        else:
            L.append("""      %silu = scalar.siluf<afn> %t0 : f32
      %value = scalar.mulf %silu, %t1 : f32""")
    else:
        L.append("      %value = scalar.addf %t0, %zero_f : f32")
    L.append("""      %out_row = index.add %row_base, %rr : index
      func.call @g_store_row(%y, %tokens, %n, %tok_raw, %out_row, %value, %accumulate) : (buffer, index, index, index, index, f32, i32)""")
    if pair:
        L.append("""      %value2 = scalar.addf %t1, %zero_f : f32
      func.call @g_store_row(%y2, %tokens, %n, %tok_raw, %out_row, %value2, %accumulate) : (buffer, index, index, index, index, f32, i32)""")
    L.append("""    }
  }
  kernel.return
}
""")
    return "\n".join(L)


def mv_wide(fmt, K, R=16, TG=1, swiglu=False, suffix=""):
    """mv() for K > 32768 (more 32-value groups than a 1024-lane workgroup has lanes; e.g. a DFlash drafter's
    feature projection over 5 taps of a 6656-wide model). Same name, launch shape and arguments as mv(); each
    lane accumulates GPL groups strided by 1024 per row and token, the waves reduce as in mv()."""
    groups = K // 32
    lanes, waves = 1024, 32
    GPL = (groups + lanes - 1) // lanes
    rb = row_bytes(fmt, K)
    nw = 2 if swiglu else 1
    name = f"g_mv_{fmt}_k{K}" + ("_swiglu" if swiglu else "") + (f"_t{TG}" if TG > 1 else "") + suffix
    wargs = "%w: buffer, %w2: buffer" if swiglu else "%w: buffer"
    L = [f"""
kernel.def export("{name}") @{name}(%n: index, %tokens: index) {{
  %c1 = index.constant 1 : index
  %lanes = index.constant {lanes} : index
  %rows_per = index.constant {R} : index
  %tg = index.constant {TG} : index
  %tg_less = index.constant {TG - 1} : index
  %row_groups = index.div %n, %rows_per : index
  %tg_pad = index.add %tokens, %tg_less : index
  %token_groups = index.div %tg_pad, %tg : index
  kernel.launch.config workgroups(%token_groups, %row_groups, %c1) workgroup_size(%lanes, %c1, %c1) : index
}} launch(%n_arg: index, %tokens_arg: index, %accumulate: i32, %x: buffer, {wargs}, %y: buffer) {{
  %n = index.assume %n_arg [range(%n_arg, {R}, 262144), multiple_of(%n_arg, {R})] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %group_id = kernel.workgroup.id<y> : index
  %tgroup_id = kernel.workgroup.id<x> : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, {lanes - 1})] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, {waves - 1})] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %rows_per = index.constant {R} : index
  %tg = index.constant {TG} : index
  %cW = index.constant {waves} : index
  %groups = index.constant {groups} : index
  %row_bytes = index.constant {rb} : offset
  %block_bytes = index.constant {FMT[fmt]} : offset
  %zero_f = scalar.constant 0.0 : f32
  %zero32 = vector.constant 0.0 : vector<32xf32>
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{groups}x32xf32>
  %row_base = index.mul %group_id, %rows_per : index
  %token0 = index.mul %tgroup_id, %tg : index
  %g_last = index.constant {groups - 1} : index
  %cgpb = index.constant {SB[fmt] // 32} : index
  %part_bytes = index.constant {waves * R * TG * nw * 4} : offset
  %parts = buffer.alloca<workgroup> align(16) %part_bytes : buffer
  %pv = buffer.view %parts[%zero] : buffer -> view<{nw}x{waves}x{R}x{TG}xf32>
  %first = index.cmp eq, %lane, %c0 : index"""]
    for p in range(GPL):
        L.append(f"""  %gs{p} = index.constant {p * lanes} : index
  %gi{p} = index.add %workitem, %gs{p} : index
  %ok{p} = index.cmp ult, %gi{p}, %groups : index
  %gc{p} = index.min %gi{p}, %g_last : index
  %g{p} = index.assume %gc{p} [range(%gc{p}, 0, {groups - 1})] : index
  %blk{p} = index.div %g{p}, %cgpb : index
  %within{p} = index.rem %g{p}, %cgpb : index
  %ba{p} = index.scale %blk{p}, %block_bytes : index, offset -> offset""")
    L.append(f"""  scf.for %r = [%c0 to %rows_per step %c1] {{
    %row_value = index.add %row_base, %r : index
    %row = index.assume %row_value [lt(%row_value, %n)] : index
    %ro = index.scale %row, %row_bytes : index, offset -> offset
    %rb = index.assume %r [range(%r, 0, {R - 1})] : index""")
    for wi, w in enumerate(["w", "w2"] if swiglu else ["w"]):
        for p in range(GPL):
            L.append(f"""    %o{wi}_{p} = index.add %ro, %ba{p} : offset
    %wv{wi}_{p} = scf.if %ok{p} -> (vector<32xf32>) {{
      %v = func.call @g_{fmt}_values(%{w}, %o{wi}_{p}, %within{p}) : (buffer, offset, index) -> (vector<32xf32>)
      scf.yield %v : vector<32xf32>
    }} else {{
      scf.yield %zero32 : vector<32xf32>
    }}""")
        for j in range(TG):
            prev = "%zero_f"
            for p in range(GPL):
                # activations are re-read per row (L1/L2): caching GPL x TG groups in registers does not fit 1024 lanes
                L.append(f"""    %tj{wi}_{j}_{p} = index.constant {j} : index
    %tok{wi}_{j}_{p} = index.add %token0, %tj{wi}_{j}_{p} : index
    %live{wi}_{j}_{p} = index.cmp ult, %tok{wi}_{j}_{p}, %tokens : index
    %use{wi}_{j}_{p} = scalar.andi %live{wi}_{j}_{p}, %ok{p} : i1
    %pa{wi}_{j}_{p} = scf.if %use{wi}_{j}_{p} -> (f32) {{
      %tk = index.assume %tok{wi}_{j}_{p} [lt(%tok{wi}_{j}_{p}, %tokens)] : index
      %xv = vector.load %x_view[%tk, %g{p}, %c0] : view<[%tokens]x{groups}x32xf32> -> vector<32xf32>
      %d = vector.dotf<reassoc|contract> %wv{wi}_{p}, %xv, {prev} : vector<32xf32>, vector<32xf32>, f32
      scf.yield %d : f32
    }} else {{
      scf.yield {prev} : f32
    }}""")
                prev = f"%pa{wi}_{j}_{p}"
            L.append(f"""    %s{wi}_{j} = kernel.subgroup.reduce<addf> {prev} : f32
    scf.if %first {{
      view.store %s{wi}_{j}, %pv[{wi}, %wave, %rb, {j}] : f32, view<{nw}x{waves}x{R}x{TG}xf32>
    }}""")
    L.append("  }")
    L.append(f"""  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %outputs = index.constant {R * TG} : index
  %lanes_step = index.constant {lanes} : index
  scf.for %wr0 = [%workitem to %outputs step %lanes_step] {{
    %wr = index.assume %wr0 [range(%wr0, 0, {R * TG - 1})] : index
    %rr_v = index.div %wr, %tg : index
    %rr = index.assume %rr_v [range(%rr_v, 0, {R - 1})] : index
    %jj_v = index.rem %wr, %tg : index
    %jj = index.assume %jj_v [range(%jj_v, 0, {TG - 1})] : index
    %tok_raw = index.add %token0, %jj : index
    %tok_live = index.cmp ult, %tok_raw, %tokens : index
    scf.if %tok_live {{""")
    for wi in range(nw):
        L.append(f"""      %t{wi} = scf.for %ww = [%c0 to %cW step %c1](%acc = %zero_f : f32) -> (f32) {{
        %bw = index.assume %ww [range(%ww, 0, {waves - 1})] : index
        %p = view.load %pv[{wi}, %bw, %rr, %jj] : view<{nw}x{waves}x{R}x{TG}xf32> -> f32
        %nx = scalar.addf %acc, %p : f32
        scf.yield %nx : f32
      }}""")
    if swiglu:
        if GELU:
            L.append(gelu_scalar("silu", "t0") + "\n      %value = scalar.mulf %silu, %t1 : f32")
        else:
            L.append("""      %silu = scalar.siluf<afn> %t0 : f32
      %value = scalar.mulf %silu, %t1 : f32""")
    else:
        L.append("      %value = scalar.addf %t0, %zero_f : f32")
    L.append("""      %out_row = index.add %row_base, %rr : index
      func.call @g_store_row(%y, %tokens, %n, %tok_raw, %out_row, %value, %accumulate) : (buffer, index, index, index, index, f32, i32)
    }
  }
  kernel.return
}
""")
    return "\n".join(L)


EFMT = fmt_of("token_embd.weight")
_EN = """  %squares = vector.mulf %values, %values : vector<32xf32>
  %local = vector.reduce<addf> %squares, %zero_f : vector<32xf32>, f32
  %sum = kernel.workgroup.reduce<addf> %local : f32
  %eps0 = scalar.constant 1.0e-5 : f32
  %count0 = scalar.constant {H}.0 : f32
  %mean = scalar.divf %sum, %count0 : f32
  %stable = scalar.addf %mean, %eps0 : f32
  %inverse = scalar.rsqrtf %stable : f32
  %inverses = vector.splat %inverse : vector<32xf32>
  %normalized = vector.mulf %values, %inverses : vector<32xf32>
""".replace("{H}", str(H))
_ES = CFG.get("embed_scale")  # e.g. sqrt(hidden) for Gemma
EMBED_PRE = (_EN if EMBED_NORM else
             (f"  %escale = vector.splat {'%'}escale_s : vector<32xf32>\n  %normalized = vector.mulf %values, %escale : vector<32xf32>\n" if _ES else
              "  %normalized = vector.addf %values, %zero32 : vector<32xf32>\n"))
if _ES:
    EMBED_PRE = f"  %escale_s = scalar.constant {float(_ES)} : f32\n" + EMBED_PRE


def stop_restore():
    """Ids for restoring a retained row's stop token. The head reports any end-of-turn id as the
    service terminal (eos_id), and on the next turn loom-serve feeds that *prediction* in place of the
    first delimiter of the new suffix, i.e. where the canonical history (complete_text) has
    chat.stop_after_calls / stop_after_answer. Without a fix the reused row holds <eos> where a fresh
    prompt holds the model's real stop token (GLM style: <|observation|> after calls, <|user|> after an
    answer), so reused turns diverge from the same prompt sent fresh. A fed terminal is therefore mapped
    back: to the calls stop when the next input token opens a tool result (the token after
    stop_after_calls in tool_item_prefix), else to the answer stop -- the same choice complete_text makes.
    Returns (eos, calls_id, follower_id, answer_id) or None."""
    chat = CFG.get("chat") or {}
    sc, sa = chat.get("stop_after_calls", ""), chat.get("stop_after_answer", "")
    path = os.environ.get("LLM_TOKENIZER")
    if not (sc and sa and path and CFG.get("extra_eos_ids")):
        return None
    added = {t["content"]: t["id"] for t in json.load(open(path, encoding="utf-8")).get("added_tokens", [])}
    rest = chat.get("tool_item_prefix", "")
    rest = rest[len(sc):] if rest.startswith(sc) else ""
    follow = max((t for t in added if rest.startswith(t)), key=len, default=None)
    if sc not in added or sa not in added or follow is None:
        print(f"gen_llm: stop-token restore disabled (stop markers / tool-result opener are not single added tokens)", file=sys.stderr)
        return None
    return CFG["eos_id"], added[sc], added[follow], added[sa]


STOP_RESTORE = stop_restore()
VISION = CFG.get("vision")    # image input: epochs with media carry 8-word input records (token, feature byte offset, ...)
def ids_prelude(ind="  "):
    """Defines %ids_flat / %id_stride: token i is %ids_flat[i * stride]; with media (metadata word 2 set by
    control.loom's encode_epoch) the input is the host's packet of 8-word records followed by feature rows."""
    if not VISION:
        return ""
    return (f"{ind}%meta_view = buffer.view %meta[%zero] : buffer -> view<4xi32>\n"
            f"{ind}%mixed_word = view.load %meta_view[2] : view<4xi32> -> i32\n"
            f"{ind}%mixed_zero = scalar.constant 0 : i32\n"
            f"{ind}%mixed = scalar.cmpi ne, %mixed_word, %mixed_zero : i32\n"
            f"{ind}%stride_one = index.constant 1 : index\n"
            f"{ind}%stride_eight = index.constant 8 : index\n"
            f"{ind}%id_stride = scf.select %mixed, %stride_eight, %stride_one : index\n"
            f"{ind}%n_ids = index.mul %tokens, %id_stride : index\n"
            f"{ind}%ids_flat = buffer.view %ids[%zero] : buffer -> view<[%n_ids]xi32>\n")
def ids_load(dst, tok, ind="  "):
    """%dst = token id of input %tok."""
    if not VISION:
        return f"{ind}{dst} = view.load %ids_view[{tok}] : view<[%tokens]xi32> -> i32\n"
    return (f"{ind}{dst}_e0 = index.mul {tok}, %id_stride : index\n"
            f"{ind}{dst}_e = index.assume {dst}_e0 [lt({dst}_e0, %n_ids)] : index\n"
            f"{ind}{dst} = view.load %ids_flat[{dst}_e] : view<[%n_ids]xi32> -> i32\n")

EMBED_RAW = ids_load("%raw", "%token")
if STOP_RESTORE:
    _e, _sc, _fo, _sa = STOP_RESTORE
    EMBED_RAW = ids_load("%raw_in", "%token") + f"""
  // A fed service terminal is the previous turn's stop token: restore the real one (see stop_restore).
  %term_id = scalar.constant {_e} : i32
  %stop_calls_id = scalar.constant {_sc} : i32
  %stop_follow_id = scalar.constant {_fo} : i32
  %stop_answer_id = scalar.constant {_sa} : i32
  %one_idx = index.constant 1 : index
  %next_value = index.add %token, %one_idx : index
  %has_next = index.cmp ult, %next_value, %tokens : index
  %next_sel = scf.select %has_next, %next_value, %token : index
  %next_token = index.assume %next_sel [lt(%next_sel, %tokens)] : index
{ids_load("%next_raw", "%next_token")}
  %opens_result = scalar.cmpi eq, %next_raw, %stop_follow_id : i32
  %after_calls = scalar.andi %opens_result, %has_next : i1
  %stop_id = scf.select %after_calls, %stop_calls_id, %stop_answer_id : i32
  %is_term = scalar.cmpi eq, %raw_in, %term_id : i32
  %raw = scf.select %is_term, %stop_id, %raw_in : i32
"""
# Image features: a record whose word 1 (feature byte offset in the packet) is nonzero carries a projected
# row (F32, H wide) that replaces the token embedding as is: no embedding scale or norm (llama.cpp feeds
# raw embeddings unscaled).
FEATURE_FLAG = ("""  %feat_word = scf.if %mixed -> (i32) {{
    %fw_e0 = index.mul %token, %id_stride : index
    %fw_e1 = index.add %fw_e0, %stride_one : index
    %fw_e = index.assume %fw_e1 [lt(%fw_e1, %n_ids)] : index
    %fw = view.load %ids_flat[%fw_e] : view<[%n_ids]xi32> -> i32
    scf.yield %fw : i32
  }} else {{
    scf.yield %mixed_zero : i32
  }}
  // padding tokens of the epoch shape read stale records: only live inputs carry features
  %tinfo_view = buffer.view %tinfo[%zero] : buffer -> view<[%tokens]x4xi32>
  %live_word = view.load %tinfo_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %is_live = scalar.cmpi ne, %live_word, %mixed_zero : i32
  %has_feat = scalar.cmpi ne, %feat_word, %mixed_zero : i32
  %is_feat = scalar.andi %has_feat, %is_live : i1
  %feat_one_byte = index.constant 1 : offset
""".replace("{{", "{").replace("}}", "}") if VISION else "")
FEATURE_PICK = ("""    %vv = scf.if %is_feat -> (vector<32xf32>) {
      %fb_lane = index.assume %lane [range(%lane, 0, %HGM1%)] : index
      %fo_u = scalar.maxsi %feat_word, %mixed_zero : i32
      %fo_i = index.cast %fo_u : i32 to index
      %fo = index.scale %fo_i, %one_byte : index, offset -> offset
      %fview = buffer.view %ids[%fo] : buffer -> view<%HG%x32xf32>
      %fv = vector.load %fview[%fb_lane, %c0] : view<%HG%x32xf32> -> vector<32xf32>
      scf.yield %fv : vector<32xf32>
    } else {
      scf.yield %v : vector<32xf32>
    }
""".replace("%HGM1%", str(HG - 1)).replace("%HG%", str(HG)).replace("%one_byte", "%feat_one_byte") if VISION else "")
if VISION:
    # An embedding norm (Muse-Glimmer) applies to every input row, projected features included (llama.cpp
    # norms the selected embeddings); only the token embedding scale (Gemma) skips features (scale_rows).
    EMBED_PRE = EMBED_PRE.replace("vector.splat %escale_s :", "vector.splat %escale_eff :")
    if _ES:
        EMBED_PRE = EMBED_PRE.replace("  %escale = vector.splat", "  %escale_one = scalar.constant 1.0 : f32\n  %escale_eff = scf.select %is_feat, %escale_one, %escale_s : f32\n  %escale = vector.splat", 1)
EMBED_PRE += "  %count = scalar.constant " + str(float(H)) + " : f32\n  %eps = scalar.constant " + str(EPS) + " : f32\n"

NORMS = f"""
kernel.def export("g_token_info") @g_token_info(%tokens: index) {{
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %metadata: buffer, %info: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %zero = index.constant 0 : offset
  %c16 = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %count_bound = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count_value = index.cast %count_bound : i32 to index
  %fewer = index.cmp ult, %count_value, %spans : index
  %count = scf.select %fewer, %count_value, %spans : index
  %packets = buffer.view %metadata[%c16] : buffer -> view<[%spans]x9xi32>
  %token32 = index.cast %token : index to i32
  %minus_one = scalar.constant -1 : i32
  %row, %position, %live, %span = scf.for %s = [%c0 to %count step %c1](%r = %minus_one : i32, %p = %c0_i32 : i32, %l = %c0_i32 : i32, %found = %minus_one : i32) -> (i32, i32, i32, i32) {{
    %bounded_s = index.assume %s [lt(%s, %spans)] : index
    %length32 = view.load %packets[%bounded_s, 0] : view<[%spans]x9xi32> -> i32
    %position32 = view.load %packets[%bounded_s, 1] : view<[%spans]x9xi32> -> i32
    %row32 = view.load %packets[%bounded_s, 3] : view<[%spans]x9xi32> -> i32
    %input32 = view.load %packets[%bounded_s, 4] : view<[%spans]x9xi32> -> i32
    %end32 = scalar.addi %input32, %length32 : i32
    %after_begin = scalar.cmpi sge, %token32, %input32 : i32
    %before_end = scalar.cmpi slt, %token32, %end32 : i32
    %inside = scalar.andi %after_begin, %before_end : i1
    %relative = scalar.subi %token32, %input32 : i32
    %absolute = scalar.addi %position32, %relative : i32
    %s32 = index.cast %bounded_s : index to i32
    %next_r = scf.select %inside, %row32, %r : i32
    %next_p = scf.select %inside, %absolute, %p : i32
    %next_l = scf.select %inside, %c1_i32, %l : i32
    %next_s = scf.select %inside, %s32, %found : i32
    scf.yield %next_r, %next_p, %next_l, %next_s : i32, i32, i32, i32
  }}
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  view.store %row, %info_view[%token, 0] : i32, view<[%tokens]x4xi32>
  view.store %position, %info_view[%token, 1] : i32, view<[%tokens]x4xi32>
  view.store %live, %info_view[%token, 2] : i32, view<[%tokens]x4xi32>
  view.store %span, %info_view[%token, 3] : i32, view<[%tokens]x4xi32>
  kernel.return
}}

// x[t] = embedding[id] (optionally rmsnorm'd / scaled), h[t] = rmsnorm(x[t]) * w.
// 256 lanes; lane g < {HG} dequantizes {EFMT} group g of the row.
kernel.def export("g_embed") @g_embed(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %ids: buffer{", %meta: buffer, %tinfo: buffer" if VISION else ""}, %table: buffer, %w: buffer, %x: buffer, %h: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c8 = index.constant {SB[EFMT] // 32} : index
  %groups = index.constant {HG} : index
  %zero_f = scalar.constant 0.0 : f32
  %ids_view = buffer.view %ids[%zero] : buffer -> view<[%tokens]xi32>
{ids_prelude()}{EMBED_RAW}{FEATURE_FLAG}  %c0_i32 = scalar.constant 0 : i32
  %vocab_last = scalar.constant {V - 1} : i32
  %low = scalar.maxsi %raw, %c0_i32 : i32
  %clamped = scalar.minsi %low, %vocab_last : i32
  %bounded = scalar.assume %clamped [range(%clamped, 0, {V - 1})] : i32
  %id = index.cast %bounded : i32 to index
  %row_bytes = index.constant {row_bytes(EFMT, H)} : offset
  %block_bytes = index.constant {FMT[EFMT]} : offset
  %row_origin = index.scale %id, %row_bytes : index, offset -> offset
  %active = index.cmp ult, %lane, %groups : index
  %zero32 = vector.constant 0.0 : vector<32xf32>
  %values = scf.if %active -> (vector<32xf32>) {{
    %block = index.div %lane, %c8 : index
    %within = index.rem %lane, %c8 : index
    %block_add = index.scale %block, %block_bytes : index, offset -> offset
    %origin = index.add %row_origin, %block_add : offset
    %v = func.call @g_{EFMT}_values(%table, %origin, %within) : (buffer, offset, index) -> (vector<32xf32>)
{FEATURE_PICK}    scf.yield {"%vv" if VISION else "%v"} : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}
{EMBED_PRE}  %squares2 = vector.mulf %normalized, %normalized : vector<32xf32>
  %local2 = vector.reduce<addf> %squares2, %zero_f : vector<32xf32>, f32
  %sum2 = kernel.workgroup.reduce<addf> %local2 : f32
  %mean2 = scalar.divf %sum2, %count : f32
  %stable2 = scalar.addf %mean2, %eps : f32
  %inverse2 = scalar.rsqrtf %stable2 : f32
  %inverses2 = vector.splat %inverse2 : vector<32xf32>
  scf.if %active {{
    %bl = index.assume %lane [range(%lane, 0, {HG - 1})] : index
    %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
    %h_view = buffer.view %h[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
    %w_view = buffer.view %w[%zero] : buffer -> view<{HG}x32xf32>
    vector.store %normalized, %x_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x{HG}x32xf32>
    %weights = vector.load %w_view[%bl, %c0] : view<{HG}x32xf32> -> vector<32xf32>
    %n2 = vector.mulf %normalized, %inverses2 : vector<32xf32>
    %hv = vector.mulf %n2, %weights : vector<32xf32>
    vector.store %hv, %h_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x{HG}x32xf32>
  }}
  kernel.return
}}

// x[t] += {'rmsnorm(y[t]) * w_post' if POST_NORM else 'y[t]'}; h[t] = rmsnorm(x[t]) * w_next.
kernel.def export("g_post_add_norm") @g_post_add_norm(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %y: buffer, %w_post: buffer, %x: buffer, %w_next: buffer, %h: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %groups = index.constant {HG} : index
  %zero_f = scalar.constant 0.0 : f32
  %zero32 = vector.constant 0.0 : vector<32xf32>
  %count = scalar.constant {float(H)} : f32
  %active = index.cmp ult, %lane, %groups : index
  %bl_raw = index.min %lane, %groups : index
  %y_view = buffer.view %y[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
  %yv, %xv = scf.if %active -> (vector<32xf32>, vector<32xf32>) {{
    %bl = index.assume %lane [range(%lane, 0, {HG - 1})] : index
    %a = vector.load %y_view[%token, %bl, %c0] : view<[%tokens]x{HG}x32xf32> -> vector<32xf32>
    %b = vector.load %x_view[%token, %bl, %c0] : view<[%tokens]x{HG}x32xf32> -> vector<32xf32>
    scf.yield %a, %b : vector<32xf32>, vector<32xf32>
  }} else {{
    scf.yield %zero32, %zero32 : vector<32xf32>, vector<32xf32>
  }}
  %ysq = vector.mulf %yv, %yv : vector<32xf32>
  %ylocal = vector.reduce<addf> %ysq, %zero_f : vector<32xf32>, f32
  %ysum = kernel.workgroup.reduce<addf> %ylocal : f32
  %ymean = scalar.divf %ysum, %count : f32
  %post_eps = scalar.constant {POST_EPS} : f32
  %ystable = scalar.addf %ymean, %post_eps : f32
  %yinv = scalar.rsqrtf %ystable : f32
  %yinvs = vector.splat %yinv : vector<32xf32>
  %w_post_view = buffer.view %w_post[%zero] : buffer -> view<{HG}x32xf32>
  %w_next_view = buffer.view %w_next[%zero] : buffer -> view<{HG}x32xf32>
  %xn = scf.if %active -> (vector<32xf32>) {{
    %bl = index.assume %lane [range(%lane, 0, {HG - 1})] : index
    %wp = vector.load %w_post_view[%bl, %c0] : view<{HG}x32xf32> -> vector<32xf32>
{"    %yn = vector.mulf %yv, %yinvs : vector<32xf32>" if POST_NORM else "    %yn = vector.addf %yv, %zero32 : vector<32xf32>"}
{"    %r = vector.mulf %yn, %wp : vector<32xf32>" if POST_NORM else "    %r = vector.addf %yn, %zero32 : vector<32xf32>"}
    %sum = vector.addf %xv, %r : vector<32xf32>
    vector.store %sum, %x_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x{HG}x32xf32>
    scf.yield %sum : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}
  %xsq = vector.mulf %xn, %xn : vector<32xf32>
  %xlocal = vector.reduce<addf> %xsq, %zero_f : vector<32xf32>, f32
  %xsum = kernel.workgroup.reduce<addf> %xlocal : f32
  %xmean = scalar.divf %xsum, %count : f32
  %eps = scalar.constant {EPS} : f32
  %xstable = scalar.addf %xmean, %eps : f32
  %xinv = scalar.rsqrtf %xstable : f32
  %xinvs = vector.splat %xinv : vector<32xf32>
  scf.if %active {{
    %bl = index.assume %lane [range(%lane, 0, {HG - 1})] : index
    %wn = vector.load %w_next_view[%bl, %c0] : view<{HG}x32xf32> -> vector<32xf32>
    %xnn = vector.mulf %xn, %xinvs : vector<32xf32>
    %hv = vector.mulf %xnn, %wn : vector<32xf32>
    %h_view = buffer.view %h[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
    vector.store %hv, %h_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x{HG}x32xf32>
  }}
  kernel.return
}}
"""


ROW_SCALE = r"""

// Per-row power-of-two scale (as g_row_scale) plus the scaled row as F16, the
// GEMM A operand: x16[r] = f16(x[r] * scale[r]).
kernel.def export("g_row_scale_f16") @g_row_scale_f16(%rows: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%rows, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%rows_arg: index, %k_arg: index, %x: buffer, %scales: buffer, %x16: buffer) {
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 2097152)] : index
  %k = index.assume %k_arg [range(%k_arg, 256, 65536), multiple_of(%k_arg, 256)] : index
  %row_id = kernel.workgroup.id<x> : index
  %row = index.assume %row_id [lt(%row_id, %rows)] : index
  %lane = kernel.workitem.id<x> : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %zero_f = scalar.constant 0.0 : f32
  // vector<4> steps (k is a multiple of 256): each step of a scalar loop waits for its load (and in pass 2
  // for the previous step's store), so 4x fewer steps is ~4x less exposed latency.
  %c4 = index.constant 4 : index
  %k4 = index.div %k, %c4 : index
  %x4_view = buffer.view %x[%zero] : buffer -> view<[%rows]x[%k4]x4xf32>
  %h4_view = buffer.view %x16[%zero] : buffer -> view<[%rows]x[%k4]x4xf16>
  %local = scf.for %c = [%lane to %k4 step %c256](%m = %zero_f : f32) -> (f32) {
    %bc = index.assume %c [lt(%c, %k4)] : index
    %v = vector.load %x4_view[%row, %bc, %c0] : view<[%rows]x[%k4]x4xf32> -> vector<4xf32>
    %a = vector.absf %v : vector<4xf32>
    %n = vector.reduce<maxnumf> %a, %m : vector<4xf32>, f32
    scf.yield %n : f32
  }
  %maximum = kernel.workgroup.reduce<maxnumf> %local : f32
  %bits = scalar.bitcast %maximum : f32 to i32
  %c23 = scalar.constant 23 : i32
  %c255 = scalar.constant 255 : i32
  %shifted = scalar.shrui %bits, %c23 : i32
  %biased = scalar.andi %shifted, %c255 : i32
  %c127 = scalar.constant 127 : i32
  %c13 = scalar.constant 13 : i32
  %c0_i32 = scalar.constant 0 : i32
  %c100 = scalar.constant 100 : i32
  %unbiased = scalar.subi %biased, %c127 : i32
  %raw_e = scalar.subi %unbiased, %c13 : i32
  %positive = scalar.maxsi %raw_e, %c0_i32 : i32
  %e = scalar.minsi %positive, %c100 : i32
  %scale_exp = scalar.subi %c127, %e : i32
  %inverse_exp = scalar.addi %c127, %e : i32
  %scale_bits = scalar.shli %scale_exp, %c23 : i32
  %inverse_bits = scalar.shli %inverse_exp, %c23 : i32
  %scale = scalar.bitcast %scale_bits : i32 to f32
  %inverse = scalar.bitcast %inverse_bits : i32 to f32
  %scale4 = vector.splat %scale : vector<4xf32>
  scf.for %c = [%lane to %k4 step %c256] {
    %bc = index.assume %c [lt(%c, %k4)] : index
    %v = vector.load %x4_view[%row, %bc, %c0] : view<[%rows]x[%k4]x4xf32> -> vector<4xf32>
    %sv = vector.mulf %v, %scale4 : vector<4xf32>
    %hv = vector.fptrunc %sv : vector<4xf32> to vector<4xf16>
    vector.store %hv, %h4_view[%row, %bc, %c0] : vector<4xf16>, view<[%rows]x[%k4]x4xf16>
  }
  %first = index.cmp eq, %lane, %c0 : index
  scf.if %first {
    %s_view = buffer.view %scales[%zero] : buffer -> view<[%rows]x2xf32>
    view.store %scale, %s_view[%row, %c0] : f32, view<[%rows]x2xf32>
    view.store %inverse, %s_view[%row, %c1] : f32, view<[%rows]x2xf32>
  }
  kernel.return
}
"""



# g_row_scale_f16_live: g_row_scale_f16 for the epoch's token rows, given %info. A padding row (info live = 0)
# gets scale 0 and zero f16 values without reading its x row, and the prefill GEMMs skip a row tile whose
# scales are all 0 (gen_gemm GEMM_SKIP_PADDING). An epoch is padded up to its power-of-two shape (a 1026-token
# prompt at prefill chunk 4096 runs as a 2048-token epoch), so this sizes the GEMM work by the live tokens.
ROW_SCALE_LIVE = (ROW_SCALE
    .replace('export("g_row_scale_f16") @g_row_scale_f16(', 'export("g_row_scale_f16_live") @g_row_scale_f16_live(')
    .replace("launch(%rows_arg: index, %k_arg: index, %x: buffer,", "launch(%rows_arg: index, %info: buffer, %k_arg: index, %x: buffer,")
    .replace("  %x4_view = buffer.view %x[%zero] : buffer -> view<[%rows]x[%k4]x4xf32>\n", """  %x4_view = buffer.view %x[%zero] : buffer -> view<[%rows]x[%k4]x4xf32>
  %live_view = buffer.view %info[%zero] : buffer -> view<[%rows]x4xi32>
  %live_word = view.load %live_view[%row, 2] : view<[%rows]x4xi32> -> i32
  %dead_zero = scalar.constant 0 : i32
  %row_live = scalar.cmpi ne, %live_word, %dead_zero : i32
  %zero4 = vector.splat %zero_f : vector<4xf32>
""")
    .replace("    %v = vector.load %x4_view[%row, %bc, %c0] : view<[%rows]x[%k4]x4xf32> -> vector<4xf32>\n    %a = vector.absf %v : vector<4xf32>",
             "    %v_raw = vector.load %x4_view[%row, %bc, %c0] : view<[%rows]x[%k4]x4xf32> -> vector<4xf32>\n    %v = scf.select %row_live, %v_raw, %zero4 : vector<4xf32>\n    %a = vector.absf %v : vector<4xf32>")
    .replace("  %scale = scalar.bitcast %scale_bits : i32 to f32\n  %inverse = scalar.bitcast %inverse_bits : i32 to f32\n",
             "  %scale_l = scalar.bitcast %scale_bits : i32 to f32\n  %inverse_l = scalar.bitcast %inverse_bits : i32 to f32\n"
             "  %scale = scf.select %row_live, %scale_l, %zero_f : f32\n  %inverse = scf.select %row_live, %inverse_l, %zero_f : f32\n")
    .replace("    %v = vector.load %x4_view[%row, %bc, %c0] : view<[%rows]x[%k4]x4xf32> -> vector<4xf32>\n    %sv = vector.mulf %v, %scale4 : vector<4xf32>",
             "    %v_raw2 = vector.load %x4_view[%row, %bc, %c0] : view<[%rows]x[%k4]x4xf32> -> vector<4xf32>\n    %v = scf.select %row_live, %v_raw2, %zero4 : vector<4xf32>\n    %sv = vector.mulf %v, %scale4 : vector<4xf32>"))
assert ROW_SCALE_LIVE.count("%row_live") == 5, ROW_SCALE_LIVE.count("%row_live")


def live_row_scale(program):
    """In @llm_advance (prefill; the drafting programs keep g_row_scale_f16: their draft-block rows are not live but
    are computed), the epoch's token-row scales (g_row_scale_f16[%t]) take %info, so padding rows get scale 0."""
    import re as _re
    parts = _re.split(r"(?m)^(?=command\.program\.def )", program)
    for n, p in enumerate(parts):
        if p.startswith("command.program.def public @llm_advance("):
            parts[n] = _re.sub(r"kernel\.launch @g_row_scale_f16\[%t\]\(%t, (.*?)\) : \[index\]\(index, index, buffer, buffer, buffer\)",
                               r"kernel.launch @g_row_scale_f16_live[%t](%t, %info, \1) : [index](index, buffer, index, buffer, buffer, buffer)", p)
    return "".join(parts)

# Live-row skip (prefill padding). An epoch runs at its power-of-two shape, so a 1026-token prompt at prefill chunk
# 4096 is a 2048-row epoch, half padding. live_row_scale already gives padding rows scale 0 (the prefill GEMMs skip
# their tiles); live_row_skip makes the per-row kernels free on them too. Every kernel launched over the epoch's rows
# ([%t], one workgroup per token row: `%r = kernel.workgroup.id<x>` then `index.assume %r [lt(%r, %tokens)]`) in a
# prefill branch of @llm_advance gets a `_lv` copy wrapped in `scf.if` row live (%info word 2 != 0); a padding row
# does nothing, except that a kernel with a `%scales` row-scale output still writes scale 0 for it. Rows are
# independent in these kernels, and nothing reads a padding row's results (attention keys are stored per live
# row, the MoE buckets and the head take live rows only). Launches shared with decode-shaped epochs become
# `scf.if %gemm { _lv } else { original }`. Not changed: attention (its own padding handling), g_token_info,
# g_embed, decode-shaped branches, and launches writing the epoch's interface buffers (%features, %outputs, ...).
import re  # noqa: E402
_TOKEN_ROW = re.compile(r"  %(\w+) = kernel\.workgroup\.id<x> : index\n  %(\w+) = index\.assume %\1 \[(?:range\(%\1, [^)]*\), )?lt\(%\1, %(\w+)\)\] : index\n")


def live_skip_kernel(text, name):
    m = _TOKEN_ROW.search(text)
    head, rest = text.split(" launch(", 1)
    params, body = rest.split(") {\n", 1)
    has_info = re.search(r"(^|, )%info: buffer", params) is not None
    info = "%info" if has_info else "%lv_info"
    if not has_info:
        params += ", %lv_info: buffer"
    assert body.count("kernel.return") == 1, name
    tok, rows = m.group(2), m.group(3)
    cut = body.index(m.group(0)) + len(m.group(0))
    guard = f"""  %lv_zero = index.constant 0 : offset
  %lv_view = buffer.view {info}[%lv_zero] : buffer -> view<[%{rows}]x4xi32>
  %lv_word = view.load %lv_view[%{tok}, 2] : view<[%{rows}]x4xi32> -> i32
  %lv_dead = scalar.constant 0 : i32
  %lv_live = scalar.cmpi ne, %lv_word, %lv_dead : i32
  scf.if %lv_live {{
"""
    els = ""
    if re.search(r"(^|, )%scales: buffer", params):
        els = f"""  }} else {{
    %lv_lane = kernel.workitem.id<x> : index
    %lv_c0 = index.constant 0 : index
    %lv_c1 = index.constant 1 : index
    %lv_first = index.cmp eq, %lv_lane, %lv_c0 : index
    scf.if %lv_first {{
      %lv_zf = scalar.constant 0.0 : f32
      %lv_s = buffer.view %scales[%lv_zero] : buffer -> view<[%{rows}]x2xf32>
      view.store %lv_zf, %lv_s[%{tok}, %lv_c0] : f32, view<[%{rows}]x2xf32>
      view.store %lv_zf, %lv_s[%{tok}, %lv_c1] : f32, view<[%{rows}]x2xf32>
    }}
"""
    end = body.rindex("  kernel.return\n}")
    body = body[:cut] + guard + body[cut:end] + els + "  }\n" + body[end:]
    head = head.replace(f'export("{name}") @{name}(', f'export("{name}_lv") @{name}_lv(')
    assert f"@{name}_lv(" in head, name
    return head + " launch(" + params + ") {\n" + body, has_info


def live_row_skip(text, prefill_conds=("%mm", "%gemm", "%big", "%huge"), skip_names=r"attention|token_info|embed"):
    """`text`: the whole model.loom. Returns it with the per-row prefill launches of @llm_advance on `_lv` kernels."""
    defs = {}
    for d in re.split(r"(?m)^(?=kernel\.def )", text):
        m = re.match(r'kernel\.def export\("(\w+)"\)', d)
        if m and "  kernel.return\n}\n" in d:
            defs[m.group(1)] = d[:d.index("  kernel.return\n}\n") + len("  kernel.return\n}\n")]
    a = text.index("command.program.def public @llm_advance(")
    b = text.find("\ncommand.program.def ", a + 10)
    b = len(text) if b < 0 else b + 1
    new, out, stack = {}, [], []
    for line in text[a:b].split("\n"):
        s = line.strip()
        mi = re.match(r"scf\.if (%\w+) \{$", s)
        if mi:
            stack.append([mi.group(1), "then"])
        elif s == "} else {" and stack:
            stack[-1][1] = "else"
        elif s == "}" and stack:
            stack.pop()
        ml = re.match(r"(\s*)kernel\.launch @(\w+)\[%t\]\(%t(, .*)?\) : \[index\]\(index(, .*)?\)$", line)
        if (ml and ml.group(2) in defs and not re.search(skip_names, ml.group(2)) and _TOKEN_ROW.search(defs[ml.group(2)])
                and not any(c in prefill_conds and br == "else" for c, br in stack)
                and not re.search(r"%(features|outputs|tokens|metadata)\b", ml.group(3) or "")):
            ind, k, args, tys = ml.group(1), ml.group(2), ml.group(3) or "", ml.group(4) or ""
            if k not in new:
                new[k] = live_skip_kernel(defs[k], k)
            has_info = new[k][1]
            lv = f"{ind}kernel.launch @{k}_lv[%t](%t{args}{'' if has_info else ', %info'}) : [index](index{tys}{'' if has_info else ', buffer'})"
            if any(c in prefill_conds and br == "then" for c, br in stack):
                out.append(lv)
            else:
                out += [f"{ind}scf.if %gemm {{", "  " + lv, f"{ind}}} else {{", "  " + line, f"{ind}}}"]
            continue
        out.append(line)
    return text[:a] + "\n".join(out) + text[b:] + "".join("\n" + t for t, _ in new.values())

# g_post_add_norm_rs[_live]: g_post_add_norm followed by g_row_scale_f16[_live] of its output row in one kernel
# (prefill: the post-attention / post-FFN residual add, the next norm and the next GEMM's f16 operand). Each lane
# already holds its 32 normalized values, so the row's max |h| is one more workgroup reduce: no second launch and
# no re-read of h (twice) from memory. Same values as the two kernels (h f32 is still written: the 4-token matvecs
# and the decode-shaped paths read it). Derived from NORMS' g_post_add_norm and ROW_SCALE's exponent code, so a
# kit's POST_NORM / eps settings carry over.
def post_add_norm_rs(live):
    a = NORMS.index('kernel.def export("g_post_add_norm")')
    t = NORMS[a:]
    t = t[:t.index("\n}\n") + 3]
    name = "g_post_add_norm_rs_live" if live else "g_post_add_norm_rs"
    t = t.replace('export("g_post_add_norm") @g_post_add_norm(', f'export("{name}") @{name}(')
    t = t.replace("launch(%tokens_arg: index, %y: buffer,", "launch(%tokens_arg: index, " + ("%info: buffer, " if live else "") + "%y: buffer,")
    t = t.replace("%w_next: buffer, %h: buffer)", "%w_next: buffer, %h: buffer, %scales: buffer, %h16: buffer)")
    old_tail = t[t.rindex("  scf.if %active {"):]
    assert "vector.store %hv, %h_view" in old_tail and old_tail.rstrip().endswith("kernel.return\n}"), old_tail
    body = old_tail[:old_tail.index("  }\n  kernel.return")]
    body = body.replace("  scf.if %active {", "  %hfin = scf.if %active -> (vector<32xf32>) {", 1) + "    scf.yield %hv : vector<32xf32>\n  } else {\n    scf.yield %zero32 : vector<32xf32>\n  }\n"
    rs = ROW_SCALE
    expo = rs[rs.index("  %bits = scalar.bitcast %maximum"):rs.index("  %scale4 = vector.splat")]
    live_code = ""
    if live:
        live_code = """  %live_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %live_word = view.load %live_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %dead_zero = scalar.constant 0 : i32
  %row_live = scalar.cmpi ne, %live_word, %dead_zero : i32
"""
        expo = expo.replace("  %scale = scalar.bitcast %scale_bits : i32 to f32\n  %inverse = scalar.bitcast %inverse_bits : i32 to f32\n",
                            "  %scale_l = scalar.bitcast %scale_bits : i32 to f32\n  %inverse_l = scalar.bitcast %inverse_bits : i32 to f32\n"
                            "  %scale = scf.select %row_live, %scale_l, %zero_f : f32\n  %inverse = scf.select %row_live, %inverse_l, %zero_f : f32\n")
        assert "%scale_l" in expo
    hval = "%hfin"
    if live:
        live_code += "  %hsel = scf.select %row_live, %hfin, %zero32 : vector<32xf32>\n"
        hval = "%hsel"
    new_tail = body + live_code + f"""  %habs = vector.absf {hval} : vector<32xf32>
  %hlocal = vector.reduce<maxnumf> %habs, %zero_f : vector<32xf32>, f32
  %maximum = kernel.workgroup.reduce<maxnumf> %hlocal : f32
""" + expo + f"""  %scale32 = vector.splat %scale : vector<32xf32>
  scf.if %active {{
    %bl16 = index.assume %lane [range(%lane, 0, {HG - 1})] : index
    %hsv = vector.mulf {hval}, %scale32 : vector<32xf32>
    %h16v = vector.fptrunc %hsv : vector<32xf32> to vector<32xf16>
    %h16_view = buffer.view %h16[%zero] : buffer -> view<[%tokens]x{HG}x32xf16>
    vector.store %h16v, %h16_view[%token, %bl16, %c0] : vector<32xf16>, view<[%tokens]x{HG}x32xf16>
  }}
  %first = index.cmp eq, %lane, %c0 : index
  scf.if %first {{
    %c1s = index.constant 1 : index
    %s_view = buffer.view %scales[%zero] : buffer -> view<[%tokens]x2xf32>
    view.store %scale, %s_view[%token, %c0] : f32, view<[%tokens]x2xf32>
    view.store %inverse, %s_view[%token, %c1s] : f32, view<[%tokens]x2xf32>
  }}
  kernel.return
}}
"""
    return "\n" + t[:t.rindex("  scf.if %active {")] + new_tail


def fuse_post_add_row_scale(program):
    """g_post_add_norm immediately followed (an optional `// ---- layer` comment between) by a row-scale launch of
    its output (`scf.if C { g_row_scale_f16[_live](same rows, H wide, %h ...) } [else { ... }]`) becomes
    `scf.if C { g_post_add_norm_rs[_live] } else { g_post_add_norm; ... }`. Returns (program, used kernel texts)."""
    import re as _re
    pat = _re.compile(
        r"(?P<ind>[ ]*)kernel\.launch @g_post_add_norm\[(?P<n>%\w+)\]\((?P=n), (?P<args>%\w+, %\w+, %\w+, %\w+), (?P<h>%\w+)\) : \[index\]\(index, (?P<ty>buffer, view<\d+xf32>, buffer, view<\d+xf32>), buffer\)\n"
        r"(?P<cmt>(?:[ ]*//[^\n]*\n)?)"
        r"(?P<ind2>[ ]*)scf\.if (?P<c>%\w+) \{\n"
        r"[ ]*kernel\.launch @g_row_scale_f16(?P<live>_live)?\[(?P=n)\]\((?P=n), (?P<info>%info, )?%nH, (?P=h), (?P<s>%\w+), (?P<h16>%\w+)\) : \[index\]\([^\n]*\)\n"
        r"(?P=ind2)\}(?P<els> else \{\n(?P<eb>(?:[^\n]*\n)*?)(?P=ind2)\})?\n")
    used = set()
    def rep(m):
        live = bool(m.group("live"))
        used.add(live)
        i2 = m.group("ind2")
        name = "g_post_add_norm_rs_live" if live else "g_post_add_norm_rs"
        orig = f"{i2}  kernel.launch @g_post_add_norm[{m.group('n')}]({m.group('n')}, {m.group('args')}, {m.group('h')}) : [index](index, {m.group('ty')}, buffer)\n"
        fused = (f"{i2}  kernel.launch @{name}[{m.group('n')}]({m.group('n')}, {'%info, ' if live else ''}{m.group('args')}, {m.group('h')}, {m.group('s')}, {m.group('h16')})"
                 f" : [index](index, {'buffer, ' if live else ''}{m.group('ty')}, buffer, buffer, buffer)\n")
        return (m.group("cmt") + f"{i2}scf.if {m.group('c')} {{\n" + fused + f"{i2}}} else {{\n" + orig
                + (m.group("eb") or "") + f"{i2}}}\n")
    program = pat.sub(rep, program)
    return program, "".join(post_add_norm_rs(lv) for lv in sorted(used))


# DeepStack (Qwen3-VL, cfg["vision"]["deepstack"] = n): an image row is n + 1 slices of H floats; g_embed takes slice 0,
# and after trunk layer k < n, g_deepstack adds slice k + 1 to the residual of image inputs and renormalizes their h
# (llama.cpp qwen3vl.cpp: deepstack_out = l_out + inp_embd[(k + 1) * n_embd ...]). Text inputs are untouched.
DEEPSTACK = int((VISION or {}).get("deepstack") or 0)
if DEEPSTACK:
    NORMS += f"""
// x[t] += feature slice %level of image input t; h[t] = rmsnorm(x[t]) * w_next (image inputs only).
kernel.def export("g_deepstack") @g_deepstack(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %level_arg: index, %ids: buffer, %meta: buffer, %tinfo: buffer, %w: buffer, %x: buffer, %h: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %level = index.assume %level_arg [range(%level_arg, 1, {DEEPSTACK})] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %groups = index.constant {HG} : index
  %zero_f = scalar.constant 0.0 : f32
  %zero32 = vector.constant 0.0 : vector<32xf32>
{ids_prelude()}{FEATURE_FLAG}  %active = index.cmp ult, %lane, %groups : index
  %work = scalar.andi %active, %is_feat : i1
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
  %sum_x = scf.if %work -> (vector<32xf32>) {{
    %bl = index.assume %lane [range(%lane, 0, {HG - 1})] : index
    %xv = vector.load %x_view[%token, %bl, %c0] : view<[%tokens]x{HG}x32xf32> -> vector<32xf32>
    %fo_u = scalar.maxsi %feat_word, %mixed_zero : i32
    %fo_i = index.cast %fo_u : i32 to index
    %fo = index.scale %fo_i, %feat_one_byte : index, offset -> offset
    %slice_bytes = index.constant {H * 4} : offset
    %slice_at = index.scale %level, %slice_bytes : index, offset -> offset
    %so = index.add %fo, %slice_at : offset
    %fview = buffer.view %ids[%so] : buffer -> view<{HG}x32xf32>
    %fv = vector.load %fview[%bl, %c0] : view<{HG}x32xf32> -> vector<32xf32>
    %s = vector.addf %xv, %fv : vector<32xf32>
    scf.yield %s : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}
  %squares = vector.mulf %sum_x, %sum_x : vector<32xf32>
  %local = vector.reduce<addf> %squares, %zero_f : vector<32xf32>, f32
  %sum = kernel.workgroup.reduce<addf> %local : f32
  %count = scalar.constant {float(H)} : f32
  %eps = scalar.constant {EPS} : f32
  %mean = scalar.divf %sum, %count : f32
  %stable = scalar.addf %mean, %eps : f32
  %inverse = scalar.rsqrtf %stable : f32
  %inverses = vector.splat %inverse : vector<32xf32>
  scf.if %work {{
    %bl = index.assume %lane [range(%lane, 0, {HG - 1})] : index
    %h_view = buffer.view %h[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
    %w_view = buffer.view %w[%zero] : buffer -> view<{HG}x32xf32>
    vector.store %sum_x, %x_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x{HG}x32xf32>
    %weights = vector.load %w_view[%bl, %c0] : view<{HG}x32xf32> -> vector<32xf32>
    %n2 = vector.mulf %sum_x, %inverses : vector<32xf32>
    %hv = vector.mulf %n2, %weights : vector<32xf32>
    vector.store %hv, %h_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x{HG}x32xf32>
  }}
  kernel.return
}}
"""


def deepstack_wiring(program):
    """After the closing g_post_add_norm of trunk layers 0 .. DEEPSTACK-1 (it wrote x and h), launch g_deepstack."""
    for i in range(DEEPSTACK):
        a = re.search(rf"  // ---- layer {i}[ \n]", program).start(); b = re.search(rf"  // ---- layer {i + 1}[ \n]", program).start()
        m = list(re.finditer(r"  kernel\.launch @g_post_add_norm\[%t\]\(%t, %y, %\w+, %x, (%\w+), %h\) : [^\n]*\n", program[a:b]))[-1]
        at = a + m.end()
        program = (program[:at] + f"  %ds_level_{i} = index.constant {i + 1} : index\n"
                   f"  kernel.launch @g_deepstack[%t](%t, %ds_level_{i}, %tokens, %metadata, %info, {m.group(1)}, %x, %h) : [index](index, index, buffer, buffer, buffer, view<{H}xf32>, buffer, buffer)\n"
                   + program[at:])
    return program


MROPE = CFG.get("mrope")   # {"sections": [t, h, w, e], "interleaved": bool}: Qwen-VL multi-axis RoPE (image input)


def mrope_positions():
    """Per-token (t, h, w) rotary positions. Media epochs (metadata word 2) carry them in input-record
    words 4..6 (control.loom's vision encode); ordinary epochs use the KV position plus the span's
    source-position offset (a table after the plan in the metadata, written by encode_epoch)."""
    return """  %mr_meta = buffer.view %meta[%zero] : buffer -> view<4xi32>
  %mr_count32 = view.load %mr_meta[0] : view<4xi32> -> i32
  %mr_flag = view.load %mr_meta[2] : view<4xi32> -> i32
  %mr_mixed = scalar.cmpi ne, %mr_flag, %c0_i32 : i32
  %mr_span32 = view.load %info_view[%token, 3] : view<[%tokens]x4xi32> -> i32
  %mr_cb = scalar.assume %mr_count32 [range(%mr_count32, 1, 8192)] : i32
  %mr_count = index.cast %mr_cb : i32 to index
  %mr_sl = scalar.maxsi %mr_span32, %c0_i32 : i32
  %mr_one = scalar.constant 1 : i32
  %mr_last = scalar.subi %mr_cb, %mr_one : i32
  %mr_sc = scalar.minsi %mr_sl, %mr_last : i32
  %mr_sb = scalar.assume %mr_sc [range(%mr_sc, 0, 8191)] : i32
  %mr_span0 = index.cast %mr_sb : i32 to index
  %mr_span = index.assume %mr_span0 [lt(%mr_span0, %mr_count)] : index
  %mr_c36 = index.constant 36 : offset
  %mr_c16 = index.constant 16 : offset
  %mr_plan = index.scale %mr_count, %mr_c36 : index, offset -> offset
  %mr_table = index.add %mr_c16, %mr_plan : offset
  %mr_tv = buffer.view %meta[%mr_table] : buffer -> view<[%mr_count]xi32>
  %mr_delta = view.load %mr_tv[%mr_span] : view<[%mr_count]xi32> -> i32
  %mr_linear = scalar.addi %position_low, %mr_delta : i32
  %mr_t32, %mr_h32, %mr_w32 = scf.if %mr_mixed -> (i32, i32, i32) {
    %mr_c8 = index.constant 8 : index
    %mr_n = index.mul %tokens, %mr_c8 : index
    %mr_rec = buffer.view %ids[%zero] : buffer -> view<[%mr_n]xi32>
    %mr_base = index.mul %token, %mr_c8 : index
    %mr_e4 = index.add %mr_base, %c4 : index
    %mr_c5 = index.constant 5 : index
    %mr_c6 = index.constant 6 : index
    %mr_e5 = index.add %mr_base, %mr_c5 : index
    %mr_e6 = index.add %mr_base, %mr_c6 : index
    %mr_b4 = index.assume %mr_e4 [lt(%mr_e4, %mr_n)] : index
    %mr_b5 = index.assume %mr_e5 [lt(%mr_e5, %mr_n)] : index
    %mr_b6 = index.assume %mr_e6 [lt(%mr_e6, %mr_n)] : index
    %mr_pt = view.load %mr_rec[%mr_b4] : view<[%mr_n]xi32> -> i32
    %mr_ph = view.load %mr_rec[%mr_b5] : view<[%mr_n]xi32> -> i32
    %mr_pw = view.load %mr_rec[%mr_b6] : view<[%mr_n]xi32> -> i32
    scf.yield %mr_pt, %mr_ph, %mr_pw : i32, i32, i32
  } else {
    scf.yield %mr_linear, %mr_linear, %mr_linear : i32, i32, i32
  }
  %mr_tf = scalar.sitofp %mr_t32 : i32 to f32
  %mr_hf = scalar.sitofp %mr_h32 : i32 to f32
  %mr_wf = scalar.sitofp %mr_w32 : i32 to f32
"""


def mrope_pick(j, idx):
    """SSA of the rotary position of frequency pair idx (an index SSA): llama.cpp's (i)mrope sections."""
    s0, s1, s2 = MROPE["sections"][:3]
    sect = sum(MROPE["sections"])
    MRP.append(f"""  %mr_ns{j} = index.constant {sect} : index
  %mr_sec{j} = index.rem {idx}, %mr_ns{j} : index""")
    if MROPE.get("interleaved", True):   # qwen3vl: h if i%3==1, w if i%3==2, t if i%3==0 (within 3x the section)
        MRP.append(f"""  %mr_m3c{j} = index.constant 3 : index
  %mr_m3{j} = index.rem %mr_sec{j}, %mr_m3c{j} : index
  %mr_one{j} = index.constant 1 : index
  %mr_two{j} = index.constant 2 : index
  %mr_lh{j} = index.constant {3 * s1} : index
  %mr_lw{j} = index.constant {3 * s2} : index
  %mr_ish0{j} = index.cmp eq, %mr_m3{j}, %mr_one{j} : index
  %mr_ish1{j} = index.cmp ult, %mr_sec{j}, %mr_lh{j} : index
  %mr_ish{j} = scalar.andi %mr_ish0{j}, %mr_ish1{j} : i1
  %mr_isw0{j} = index.cmp eq, %mr_m3{j}, %mr_two{j} : index
  %mr_isw1{j} = index.cmp ult, %mr_sec{j}, %mr_lw{j} : index
  %mr_isw{j} = scalar.andi %mr_isw0{j}, %mr_isw1{j} : i1""")
    else:                                  # qwen2vl: contiguous sections t | h | w
        MRP.append(f"""  %mr_lt{j} = index.constant {s0} : index
  %mr_lh{j} = index.constant {s0 + s1} : index
  %mr_lw{j} = index.constant {s0 + s1 + s2} : index
  %mr_ish0{j} = index.cmp uge, %mr_sec{j}, %mr_lt{j} : index
  %mr_ish1{j} = index.cmp ult, %mr_sec{j}, %mr_lh{j} : index
  %mr_ish{j} = scalar.andi %mr_ish0{j}, %mr_ish1{j} : i1
  %mr_isw0{j} = index.cmp uge, %mr_sec{j}, %mr_lh{j} : index
  %mr_isw1{j} = index.cmp ult, %mr_sec{j}, %mr_lw{j} : index
  %mr_isw{j} = scalar.andi %mr_isw0{j}, %mr_isw1{j} : i1""")
    MRP.append(f"""  %mr_pw{j} = scf.select %mr_isw{j}, %mr_wf, %mr_tf : f32
  %mr_p{j} = scf.select %mr_ish{j}, %mr_hf, %mr_pw{j} : f32""")
    return f"%mr_p{j}"
MRP = []   # selection lines emitted before each pair's angle


def freq_factor(j, factors):
    """llama3 rope_freqs (FREQ_FACTORS): divide pair j's frequency by its factor, picked from the runs of equal
    factors by the pair index %li{j} (Llama 3.x: eight runs). Empty without factors."""
    if not factors:
        return ""
    runs = [(i, f) for i, f in enumerate(factors) if i == 0 or f != factors[i - 1]]
    t = [f"\n  %ff{j}_0 = scalar.constant {float(runs[0][1])!r} : f32"]
    for r, (start, f) in enumerate(runs[1:], 1):
        t.append(f"""  %ffs{j}_{r} = scalar.constant {start} : i32
  %ffc{j}_{r} = scalar.cmpi sge, %li{j}, %ffs{j}_{r} : i32
  %ffv{j}_{r} = scalar.constant {float(f)!r} : f32
  %ff{j}_{r} = scf.select %ffc{j}_{r}, %ffv{j}_{r}, %ff{j}_{r - 1} : f32""")
    t.append(f"  %if{j}f = scalar.divf %if{j}, %ff{j}_{len(runs) - 1} : f32")
    return "\n".join(t)


def yarn_freq(j, y, rd, theta, factors=None):
    """YaRN y (a rope_scaling block, None = off) of a RoPE with rd rotated dims and base theta: lines turning pair j's
    frequency %if{j} (%if{j}f after freq_factor with factors; pair index %lf{j}) into that name + "y"."""
    if not y:
        return ""
    src = f"%if{j}f" if factors else f"%if{j}"
    low, inv, mix, _ = yarn_constants(y, rd, theta)
    return f"""
  %yl{j} = scalar.constant {low!r} : f32
  %yi{j} = scalar.constant {inv!r} : f32
  %ym{j} = scalar.constant {mix!r} : f32
  %yd{j} = scalar.subf %lf{j}, %yl{j} : f32
  %yr{j} = scalar.mulf %yd{j}, %yi{j} : f32
  %yc{j}a = scalar.maxnumf %yr{j}, %zero_f : f32
  %yc{j} = scalar.minnumf %yc{j}a, %one_f : f32
  %ys{j} = scalar.mulf %yc{j}, %ym{j} : f32
  %yf{j} = scalar.subf %one_f, %ys{j} : f32
  {src}y = scalar.mulf {src}, %yf{j} : f32"""


def yarn_scale(j, y, rd, theta):
    """YaRN y (as yarn_freq): %cr{j}m / %sr{j}m = cos / sin times mscale."""
    if not y:
        return ""
    return f"""
  %ymsc{j} = scalar.constant {yarn_constants(y, rd, theta)[3]!r} : f32
  %cr{j}m = scalar.mulf %cr{j}, %ymsc{j} : f32
  %sr{j}m = scalar.mulf %sr{j}, %ymsc{j} : f32"""


def qk_rope(rd=None, theta=None, name="g_qk_rope_store", factors=FREQ_FACTORS, yarn=YARN):
    """Per-head (optional) RMSNorm + RoPE for QH query and KVH key heads, with
    keys/values stored to the paged F16 KV planes: one 256-lane workgroup per
    token. Lane l of a wave owns D/32 of a head's dims: NeoX: l, l+32, ...
    with rotary pairs (d, d + RD/2) over the first RD (rope_dims) dims and the
    rest passed through; interleaved ("norm", D = 128 only): 4l..4l+3 with
    pairs (0,1),(2,3). Each pair's angle is computed once for every head. rd / theta / name / factors / yarn:
    another rotary width, base, kernel name, frequency factors and YaRN block (cfg rope_scaling; None: none) for
    layers with their own RoPE (sliding layers, qk_rope_swa)."""
    rd, theta = rd or RD, theta or THETA
    n = DL
    if ROPE == "neox":
        dims = ["%lane"] + [f"%d{j}" for j in range(1, n)]
        dim_defs = "".join(f"  %d{j}c = index.constant {32 * j} : index\n  %d{j} = index.add %lane, %d{j}c : index\n" for j in range(1, n))
        half = rd // 64  # pair offset in lanes-of-32 units
        freq_idx = [dims[j] for j in range(half)]
        pairs = [(j, j + half, j) for j in range(half)]
        ln_scale = 2.0 * math.log(theta) / rd   # angle = position * exp(-d * 2 ln(theta) / RD)
    else:
        assert D == 128 and rd == 128, "interleaved RoPE is implemented for head_dim 128"
        dims = ["%e0", "%e1", "%e2", "%e3"]
        dim_defs = "  %c1 = index.constant 1 : index\n  %e0 = index.mul %lane, %c4 : index\n  %e1 = index.add %e0, %c1 : index\n  %e2 = index.add %e0, %c2 : index\n  %e3 = index.add %e0, %c3 : index\n  %f0 = index.mul %lane, %c2 : index\n  %f1 = index.add %f0, %c1 : index\n"
        freq_idx = ["%f0", "%f1"]; pairs = [(0, 1, 0), (2, 3, 1)]
        ln_scale = math.log(theta) / 64.0
    yt = ", ".join(["f32"] * n)
    xs = ", ".join(f"%x{i}" for i in range(n))
    t = [f"""
kernel.def export("{name}") @{name}(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %layer_arg: index, %rope: i32, %context_arg: index, %pool_arg: index, %q: buffer, %k: buffer, %v: buffer, %q_norm: buffer, %k_norm: buffer, %info: buffer, %origins: buffer, %arena: buffer{", %meta: buffer, %ids: buffer" if MROPE else ""}{"".join(f", %{p}_bias: buffer" for p in QKV_BIAS)}) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {L - 1})] : index
  %context = index.assume %context_arg [range(%context_arg, 16, 262144)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 1048576)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, 7)] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c15 = index.constant 15 : index
  %c16 = index.constant 16 : index
  %cqh = index.constant {QH} : index
  %call = index.constant {QH + KVH} : index
  %c0_i32 = scalar.constant 0 : i32
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %row32 = view.load %info_view[%token, 0] : view<[%tokens]x4xi32> -> i32
  %position32 = view.load %info_view[%token, 1] : view<[%tokens]x4xi32> -> i32
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %position_low = scalar.maxsi %position32, %c0_i32 : i32
{dim_defs}  %apply_rope = scalar.cmpi ne, %rope, %c0_i32 : i32
  %position_f = scalar.sitofp %position_low : i32 to f32
{mrope_positions() if MROPE else ""}  %ln_scale = scalar.constant {ln_scale!r} : f32
  %neg = scalar.constant -1.0 : f32
  %inv_two_pi = scalar.constant 0.15915494309189535 : f32
  %two_pi = scalar.constant 6.283185307179586 : f32
  %half = scalar.constant 0.5 : f32
  %one_f = scalar.constant 1.0 : f32
  %zero_f = scalar.constant 0.0 : f32
"""]
    for j, idx in enumerate(freq_idx):
        if MROPE:
            mrope_pick(j, idx); t.append("\n".join(MRP)); MRP.clear()
        t.append(f"""  // angle = position * theta^(-2i/RD) = position * exp(-i * 2 ln(theta) / RD)
  %li{j} = index.cast {idx} : index to i32
  %lf{j} = scalar.sitofp %li{j} : i32 to f32
  %ex{j}a = scalar.mulf %lf{j}, %ln_scale : f32
  %ex{j}b = scalar.mulf %ex{j}a, %neg : f32
  %if{j} = scalar.expf<afn> %ex{j}b : f32{freq_factor(j, factors)}{yarn_freq(j, yarn, rd, theta, factors)}
  %ra{j} = scalar.mulf {"%mr_p" + str(j) if MROPE else "%position_f"}, %if{j}{"f" if factors else ""}{"y" if yarn else ""} : f32
  %tr{j} = scalar.mulf %ra{j}, %inv_two_pi : f32
  %ts{j} = scalar.addf %tr{j}, %half : f32
  %ti{j} = scalar.fptosi %ts{j} : f32 to i32
  %tf{j} = scalar.sitofp %ti{j} : i32 to f32
  %wr{j} = scalar.mulf %tf{j}, %two_pi : f32
  %an{j} = scalar.subf %ra{j}, %wr{j} : f32
  %cr{j} = scalar.cosf<afn> %an{j} : f32
  %sr{j} = scalar.sinf<afn> %an{j} : f32{yarn_scale(j, yarn, rd, theta)}
  %cos{j} = scf.select %apply_rope, %cr{j}{"m" if yarn else ""}, %one_f : f32
  %sin{j} = scf.select %apply_rope, %sr{j}{"m" if yarn else ""}, %zero_f : f32""")
    t.append(f"""  %q_view = buffer.view %q[%zero] : buffer -> view<[%tokens]x{QH}x{D}xf32>
  %k_view = buffer.view %k[%zero] : buffer -> view<[%tokens]x{KVH}x{D}xf32>
  %v_view = buffer.view %v[%zero] : buffer -> view<[%tokens]x{KVH}x{D}xf32>
  %qn_view = buffer.view %q_norm[%zero] : buffer -> view<{D}xf32>
  %kn_view = buffer.view %k_norm[%zero] : buffer -> view<{D}xf32>""")
    heads = {"q": QH, "k": KVH, "v": KVH}
    for p in QKV_BIAS:
        t.append(f"  %{p}b_view = buffer.view %{p}_bias[%zero] : buffer -> view<{heads[p]}x{D}xf32>")

    def load(dst, p, head, dim, ind):
        """%dst = this token's projection p at (head, dim), plus its bias when the model has one."""
        ld = f"view.load %{p}_view[%token, {head}, {dim}] : view<[%tokens]x{heads[p]}x{D}xf32> -> f32"
        if p not in QKV_BIAS:
            return f"{ind}%{dst} = {ld}"
        return (f"{ind}%{dst}p = {ld}\n{ind}%{dst}b = view.load %{p}b_view[{head}, {dim}] : view<{heads[p]}x{D}xf32> -> f32\n"
                f"{ind}%{dst} = scalar.addf %{dst}p, %{dst}b : f32")
    if QK_NORM:
        for i in range(n):
            t.append(f"  %w{i}q = view.load %qn_view[{dims[i]}] : view<{D}xf32> -> f32\n  %w{i}k = view.load %kn_view[{dims[i]}] : view<{D}xf32> -> f32")
    t.append(f"""  %dims = scalar.constant {float(D)!r} : f32
  %eps = scalar.constant {EPS!r} : f32
  scf.for %head = [%wave to %call step %c8] {{
    %bh = index.assume %head [range(%head, 0, {QH + KVH - 1})] : index
    %is_query = index.cmp ult, %bh, %cqh : index
    %kvh_value = index.sub %bh, %cqh : index
    {", ".join(f"%a{i}" for i in range(n))} = scf.if %is_query -> ({yt}) {{
      %qh = index.assume %bh [range(%bh, 0, {QH - 1})] : index""")
    for i in range(n):
        t.append(load(f"x{i}", "q", "%qh", dims[i], "      "))
    t.append(f"""      scf.yield {xs} : {yt}
    }} else {{
      %kh = index.assume %kvh_value [range(%kvh_value, 0, {KVH - 1})] : index""")
    for i in range(n):
        t.append(load(f"x{i}", "k", "%kh", dims[i], "      "))
    t.append(f"""      scf.yield {xs} : {yt}
    }}""")
    if QK_NORM:
        t.append("    %s0 = scalar.mulf %a0, %a0 : f32")
        for i in range(1, n):
            t.append(f"    %s{i} = scalar.fmaf %a{i}, %a{i}, %s{i - 1} : f32")
        t.append(f"""    %sum = kernel.subgroup.reduce<addf> %s{n - 1} : f32
    %mean = scalar.divf %sum, %dims : f32
    %stable = scalar.addf %mean, %eps : f32
    %inverse = scalar.rsqrtf %stable : f32""")
        for i in range(n):
            t.append(f"""    %w{i} = scf.select %is_query, %w{i}q, %w{i}k : f32
    %n{i}a = scalar.mulf %a{i}, %inverse : f32
    %n{i} = scalar.mulf %n{i}a, %w{i} : f32""")
    else:
        for i in range(n):
            t.append(f"    %n{i} = scalar.addf %a{i}, %zero_f : f32")
    rotated = set()
    for (x, y, j) in pairs:
        rotated |= {x, y}
        t.append(f"""    %rc{x} = scalar.mulf %n{x}, %cos{j} : f32
    %rs{y} = scalar.mulf %n{y}, %sin{j} : f32
    %r{x} = scalar.subf %rc{x}, %rs{y} : f32
    %rc{y} = scalar.mulf %n{y}, %cos{j} : f32
    %rs{x} = scalar.mulf %n{x}, %sin{j} : f32
    %r{y} = scalar.addf %rc{y}, %rs{x} : f32""")
    for i in range(n):
        if i not in rotated:
            t.append(f"    %r{i} = scalar.addf %n{i}, %zero_f : f32")
    t.append(f"""    scf.if %is_query {{
      %qh = index.assume %bh [range(%bh, 0, {QH - 1})] : index""")
    for i in range(n):
        t.append(f"      view.store %r{i}, %q_view[%token, %qh, {dims[i]}] : f32, view<[%tokens]x{QH}x{D}xf32>")
    t.append(f"""    }} else {{
      scf.if %live {{
        %kh = index.assume %kvh_value [range(%kvh_value, 0, {KVH - 1})] : index
        %row_bound = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
        %row = index.cast %row_bound : i32 to index
        %position_bound = scalar.assume %position_low [range(%position_low, 0, 262143)] : i32
        %position = index.cast %position_bound : i32 to index
        %padded = index.add %context, %c15 : index
        %blocks = index.div %padded, %c16 : index
        %k_plane = index.mul %layer, %c2 : index
        %k_offset = func.call @g_kv_offset(%origins, %row, %position, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
        %position_bytes_v = index.constant {POS_BYTES} : offset
        %plane_stride = index.scale %pool, %position_bytes_v : index, offset -> offset
        %v_offset = index.add %k_offset, %plane_stride : offset
        %k_cache = buffer.view %arena[%k_offset] : buffer -> view<{KVH}x{D}xf16>
        %v_cache = buffer.view %arena[%v_offset] : buffer -> view<{KVH}x{D}xf16>""")
    for i in range(n):
        t.append(f"""        %h{i} = scalar.fptrunc %r{i} : f32 to f16
        view.store %h{i}, %k_cache[%kh, {dims[i]}] : f16, view<{KVH}x{D}xf16>
{load(f"vv{i}", "v", "%kh", dims[i], "        ")}
        %vh{i} = scalar.fptrunc %vv{i} : f32 to f16
        view.store %vh{i}, %v_cache[%kh, {dims[i]}] : f16, view<{KVH}x{D}xf16>""")
    t.append("""      }
    }
  }
  kernel.return
}
""")
    return "\n".join(t)


def listed(text, name, grid_rest):
    """Variant of a per-token kernel dispatched over the tokens in `list`
    (count read from a device buffer for indirect dispatch)."""
    a = text.index(f'kernel.def export("{name}")')
    b = text.index("launch(", a)
    header = f"""kernel.def export("{name}_listed") @{name}_listed(%count_buffer: buffer) {{
  %origin_c = index.constant 0 : offset
  %counts = buffer.view %count_buffer[%origin_c] : buffer -> view<1xi32>
  %word = view.load %counts[0] : view<1xi32> -> i32
  %bounded = scalar.assume %word [range(%word, 0, 8192)] : i32
  %count = index.cast %bounded : i32 to index
{grid_rest}
}} """
    body = text[b:]
    body = body.replace("launch(%tokens_arg: index, ", "launch(%tokens_arg: index, %list: buffer, ", 1)
    old_token = """  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index"""
    assert old_token in body
    body = body.replace(old_token, """  %slot = kernel.workgroup.id<x> : index
  %zero_l = index.constant 0 : offset
  %list_view = buffer.view %list[%zero_l] : buffer -> view<[%tokens]xi32>
  %bounded_slot = index.assume %slot [lt(%slot, %tokens)] : index
  %token32 = view.load %list_view[%bounded_slot] : view<[%tokens]xi32> -> i32
  %token_bound = scalar.assume %token32 [range(%token32, 0, 8191)] : i32
  %token_value = index.cast %token_bound : i32 to index
  %token = index.assume %token_value [lt(%token_value, %tokens)] : index""", 1)
    end = body.index("kernel.return\n}\n") + len("kernel.return\n}\n")
    return header + body[:end]


ALLRED_SETUP = "".join(f"""  %ad{d} = scalar.constant {d} : i32
  %adi{d} = index.constant {d} : index
  %aq{d} = index.div %lane, %adi{d} : index
  %ar{d}_2 = index.constant 2 : index
  %arm{d} = index.rem %aq{d}, %ar{d}_2 : index
  %ab{d} = index.cmp ne, %arm{d}, %c0 : index
""" for d in (16, 8, 4, 2, 1)) + "  %aw32 = scalar.constant 32 : i32\n"


def allreduce(vals, p, outs):
    """Wave sums of the U lane-partials `vals`, all of them in every lane (named `outs`): recursive halving (each
    step a lane keeps half and adds its xor-partner's half), plain xor steps once one value is left, then the
    all-gather that reverses the halving."""
    t, T, steps = [], len(vals), []
    for d in (16, 8, 4, 2, 1):
        if T > 1:
            h = T // 2
            nv = []
            for i in range(h):
                q = f"%{p}{d}_{i}"
                t.append(f"""      {q}s = scf.select %ab{d}, {vals[i]}, {vals[i + h]} : f32
      {q}k = scf.select %ab{d}, {vals[i + h]}, {vals[i]} : f32
      {q}r, {q}u = kernel.subgroup.shuffle<xor> {q}s, %ad{d}, %aw32 : f32, i32, i32
      {q} = scalar.addf {q}k, {q}r : f32""")
                nv.append(q)
            vals, T = nv, h
            steps.append(d)
        else:
            q = f"%{p}{d}p"
            t.append(f"""      {q}r, {q}u = kernel.subgroup.shuffle<xor> {vals[0]}, %ad{d}, %aw32 : f32, i32, i32
      {q} = scalar.addf {vals[0]}, {q}r : f32""")
            vals = [q]
    for d in reversed(steps):   # gather: the partner holds the other half of the next-larger block
        lo, hi = [], []
        for i, v in enumerate(vals):
            q = f"%{p}g{d}_{i}"
            t.append(f"""      {q}r, {q}u = kernel.subgroup.shuffle<xor> {v}, %ad{d}, %aw32 : f32, i32, i32
      {q}l = scf.select %ab{d}, {q}r, {v} : f32
      {q}h = scf.select %ab{d}, {v}, {q}r : f32""")
            lo.append(f"{q}l"); hi.append(f"{q}h")
        vals = lo + hi
    for o, v in zip(outs, vals):
        t.append(f"      {o} = scalar.addf {v}, %zero_f : f32")
    return "\n".join(t)


# Decode/verify attention: up to ATTN_MT tokens of one span per wave (1 = one token per workgroup). A verify epoch's
# tokens are consecutive positions of one row; with one token per workgroup each streamed the whole KV again and
# the per-wave loop is latency-bound, so a depth-7 verify at 32k cost 8 decode attentions (Qwen3-8B: drafting at
# 7.7 tok/s against 19.6 undrafted). Exact: every token keeps its own split range, key masks, block partition and
# operation order (blocks with none of its keys leave its state untouched), so its result is bit-identical.
# From ATTN_MT_FROM on (shorter contexts keep one token per workgroup: few keys per split, and the group's serial
# work in one wave measured ~3 % slower drafted decode at short context on Qwen3-8B). attn_rules.multi_token decides.


def multi_token_split(text, P, U, T):
    """`P`_attention_split with a multi-token path: when the T epoch tokens of this token's aligned group are live,
    of one row and at consecutive positions (a verify span), the group's first workgroup computes all T (one K/V
    load per key for the T queries of its head) and the others exit; any other token takes the per-token path."""
    a = text.index(f'kernel.def export("{P}_attention_split")')
    b = text.index("  kernel.return\n}\n", a)
    k = text[a:b]
    S = int(re.search(r"%cS = index.constant (\d+) : index", k).group(1))
    anchor = "  %live = scalar.andi %live_token, %needed : i1\n  scf.if %live {\n"
    assert k.count(anchor) == 1, "multi_token_split: anchor"
    vt, ht = "vector<4xf32>", "vector<4xf16>"
    scale = re.search(r"%scale = scalar.constant (\S+) : f32", k).group(1)   # the per-token path's (a kit may change it)
    assert "%position32 = view.load %info_view[%token, 1]" in k, "multi_token_split: position"

    def group(p, ind):
        """Row, position and liveness of the group's T tokens (clamped to the last token)."""
        return "\n".join(f"""{ind}%{p}_tr{j} = index.add %sg_base, %c{j}u : index
{ind}%{p}_in{j} = index.cmp ult, %{p}_tr{j}, %tokens : index
{ind}%{p}_tc{j} = index.min %{p}_tr{j}, %sg_last : index
{ind}%{p}_t{j} = index.assume %{p}_tc{j} [range(%{p}_tc{j}, 0, 8191), lt(%{p}_tc{j}, %tokens)] : index
{ind}%{p}_row{j} = view.load %info_view[%{p}_t{j}, 0] : view<[%tokens]x4xi32> -> i32
{ind}%{p}_pos{j} = view.load %info_view[%{p}_t{j}, 1] : view<[%tokens]x4xi32> -> i32
{ind}%{p}_live{j} = view.load %info_view[%{p}_t{j}, 2] : view<[%tokens]x4xi32> -> i32""" for j in range(T))
    # Below ATTN_MT_FROM (this token's own position) the group is not read: short contexts pay one compare.
    det = [f"""  %live = scalar.andi %live_token, %needed : i1
  // multi-token path: this token's aligned group of {T}
  %sg_true = scalar.constant true : i1
  %sg_false = scalar.constant false : i1
  %sg_cT = index.constant {T} : index
  %sg_rem = index.rem %token, %sg_cT : index
  %sg_base = index.sub %token, %sg_rem : index
  %sg_leader = index.cmp eq, %sg_rem, %c0 : index
  %sg_last = index.sub %tokens, %c1 : index
  %sg_from = scalar.constant {ATTN_MT_FROM} : i32
  %sg_own_far = scalar.cmpi sge, %position32, %sg_from : i32
  %sg_uniform = scf.if %sg_own_far -> (i1) {{
{group("sg", "    ")}"""]
    for j in range(T):
        det.append(f"""    %sg_lv{j} = scalar.cmpi ne, %sg_live{j}, %c0_i32 : i32
    %sg_rq{j} = scalar.cmpi eq, %sg_row{j}, %sg_row0 : i32
    %sg_jc{j} = scalar.constant {j} : i32
    %sg_pe{j} = scalar.addi %sg_pos0, %sg_jc{j} : i32
    %sg_pq{j} = scalar.cmpi eq, %sg_pos{j}, %sg_pe{j} : i32
    %sg_a{j} = scalar.andi %sg_in{j}, %sg_lv{j} : i1
    %sg_b{j} = scalar.andi %sg_a{j}, %sg_rq{j} : i1
    %sg_ok{j} = scalar.andi %sg_b{j}, %sg_pq{j} : i1
    %sg_u{j} = scalar.andi {'%sg_true' if j == 0 else f'%sg_u{j - 1}'}, %sg_ok{j} : i1""")
    det.append(f"""    %sg_far = scalar.cmpi sge, %sg_pos0, %sg_from : i32
    %sg_all = scalar.andi %sg_u{T - 1}, %sg_far : i1
    scf.yield %sg_all : i1
  }} else {{
    scf.yield %sg_false : i1
  }}
  %sg_split = scalar.xori %sg_uniform, %sg_true : i1
  %single_live = scalar.andi %live, %sg_split : i1
  %mt_go = scalar.andi %sg_uniform, %sg_leader : i1
  scf.if %single_live {{
""")
    k = k.replace(anchor, "\n".join(det), 1)
    m = [f"""  scf.if %mt_go {{
{group("mg", "    ")}
    %mt_row_b = scalar.assume %mg_row0 [range(%mg_row0, 0, 2147483646)] : i32
    %mt_row = index.cast %mt_row_b : i32 to index
    %mt_padded = index.add %context, %c15 : index
    %mt_blocks = index.div %mt_padded, %c16 : index
    %mt_k_plane = index.mul %layer, %c2 : index
    %mt_pbytes = index.constant {POS_BYTES} : offset
    %mt_plane_stride = index.scale %pool, %mt_pbytes : index, offset -> offset
    %mt_is_sl = scalar.cmpi ne, %sliding, %c0_i32 : i32
    %mt_huge = index.constant 1073741824 : index
    %mt_q_view = buffer.view %q[%zero] : buffer -> view<[%tokens]x{QH}x32x4xf32>
    %mt_scale = scalar.constant {scale} : f32
    %mt_scale_v = vector.splat %mt_scale : {vt}
    %mt_zero4 = vector.splat %zero_f : {vt}"""]
    for j in range(T):
        m.append(f"""    %mt_pb{j} = scalar.assume %mg_pos{j} [range(%mg_pos{j}, 0, 262143)] : i32
    %mt_p{j} = index.cast %mt_pb{j} : i32 to index
    %mt_far{j} = index.cmp uge, %mt_p{j}, %window : index
    %mt_uw{j} = scalar.andi %mt_is_sl, %mt_far{j} : i1
    %mt_ws{j} = index.sub %mt_p{j}, %window_less : index
    %mt_wls{j} = scf.select %mt_uw{j}, %mt_ws{j}, %c0 : index
    %mt_wl{j} = index.assume %mt_wls{j} [range(%mt_wls{j}, 0, 262143)] : index
    %mt_end{j} = index.add %mt_p{j}, %c1 : index
    %mt_visv{j} = index.sub %mt_end{j}, %mt_wl{j} : index
    %mt_vis{j} = index.assume %mt_visv{j} [range(%mt_visv{j}, 1, 262144)] : index
    %mt_se{j} = func.call @{P}_split_count(%mt_vis{j}) : (index) -> (index)
    %mt_act{j} = index.cmp ult, %split, %mt_se{j} : index
    %mt_sl{j} = index.sub %mt_se{j}, %c1 : index
    %mt_pp{j} = index.add %mt_vis{j}, %mt_sl{j} : index
    %mt_per{j} = index.div %mt_pp{j}, %mt_se{j} : index
    %mt_sr{j} = index.mul %split, %mt_per{j} : index
    %mt_st{j} = index.min %mt_sr{j}, %mt_vis{j} : index
    %mt_spr{j} = index.add %mt_sr{j}, %mt_per{j} : index
    %mt_sp{j} = index.min %mt_spr{j}, %mt_vis{j} : index
    %mt_lo0{j} = index.add %mt_wl{j}, %mt_st{j} : index
    %mt_hi0{j} = index.add %mt_wl{j}, %mt_sp{j} : index
    %mt_lo{j} = scf.select %mt_act{j}, %mt_lo0{j}, %mt_huge : index
    %mt_hi{j} = scf.select %mt_act{j}, %mt_hi0{j}, %c0 : index
    %mt_qr{j} = vector.load %mt_q_view[%mg_t{j}, %bounded_head, %lane, %c0] : view<[%tokens]x{QH}x32x4xf32> -> {vt}
    %mt_q{j} = vector.mulf %mt_qr{j}, %mt_scale_v : {vt}""")
    lo, hi = "%mt_lo0", "%mt_hi0"
    m.append("    %mt_lou0 = index.add %mt_lo0, %c0 : index\n    %mt_hiu0 = index.add %mt_hi0, %c0 : index")
    for j in range(1, T):
        m.append(f"    %mt_lou{j} = index.min %mt_lou{j - 1}, %mt_lo{j} : index\n    %mt_hiu{j} = index.max %mt_hiu{j - 1}, %mt_hi{j} : index")
    iters = ", ".join(f"%mm{j} = %neg_big : f32, %ml{j} = %zero_f : f32, %ma{j} = %mt_zero4 : {vt}" for j in range(T))
    rtypes = ", ".join(f"f32, f32, {vt}" for _ in range(T))
    outs = ", ".join(f"%mfm{j}, %mfl{j}, %mfa{j}" for j in range(T))
    m.append(f"""    %mt_lo_q = index.div %mt_lou{T - 1}, %cU : index
    %mt_lo_al = index.mul %mt_lo_q, %cU : index
    {outs} = scf.for %mj = [%mt_lo_al to %mt_hiu{T - 1} step %cU]({iters}) -> ({rtypes}) {{
      %mkb = func.call @g_kv_offset(%origins, %mt_row, %mj, %mt_blocks, %mt_k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
      %mjend = index.add %mj, %cU : index""")
    for i in range(U):
        m.append(f"""      %mkey{i} = index.add %mj, %c{i}u : index
      %mkd{i} = index.constant {i * POS_BYTES} : offset
      %mko{i} = index.add %mkb, %mkd{i} : offset
      %mkc{i} = buffer.view %arena[%mko{i}] : buffer -> view<{KVH}x32x4xf16>
      %mkh{i} = vector.load %mkc{i}[%kv_head, %lane, %c0] : view<{KVH}x32x4xf16> -> {ht}
      %mk{i} = vector.extf %mkh{i} : {ht} to {vt}
      %mvo{i} = index.add %mko{i}, %mt_plane_stride : offset
      %mvc{i} = buffer.view %arena[%mvo{i}] : buffer -> view<{KVH}x32x4xf16>
      %mvh{i} = vector.load %mvc{i}[%kv_head, %lane, %c0] : view<{KVH}x32x4xf16> -> {ht}
      %mv{i} = vector.extf %mvh{i} : {ht} to {vt}""")
    yields = []
    for j in range(T):
        for i in range(U):
            m.append(f"""      %mokl{j}_{i} = index.cmp uge, %mkey{i}, %mt_lo{j} : index
      %mokh{j}_{i} = index.cmp ult, %mkey{i}, %mt_hi{j} : index
      %mok{j}_{i} = scalar.andi %mokl{j}_{i}, %mokh{j}_{i} : i1
      %mprod{j}_{i} = vector.mulf %mt_q{j}, %mk{i} : {vt}
      %mpart{j}_{i} = vector.reduce<addf> %mprod{j}_{i}, %zero_f : {vt}, f32""")
        m.append(allreduce([f"%mpart{j}_{i}" for i in range(U)], f"mar{j}x", [f"%msr{j}_{i}" for i in range(U)]))
        prev = f"%mm{j}"
        for i in range(U):
            m.append(f"""      %ms{j}_{i} = scf.select %mok{j}_{i}, %msr{j}_{i}, %neg_big : f32
      %mmx{j}_{i} = scalar.maxnumf {prev}, %ms{j}_{i} : f32""")
            prev = f"%mmx{j}_{i}"
        m.append(f"""      %mshift{j} = scalar.subf %mm{j}, {prev} : f32
      %malpha{j} = scalar.expf<afn> %mshift{j} : f32
      %mls{j} = scalar.mulf %ml{j}, %malpha{j} : f32
      %mav{j} = vector.splat %malpha{j} : {vt}
      %macs{j} = vector.mulf %ma{j}, %mav{j} : {vt}""")
        pl, pa = f"%mls{j}", f"%macs{j}"
        for i in range(U):
            m.append(f"""      %md{j}_{i} = scalar.subf %ms{j}_{i}, {prev} : f32
      %mp{j}_{i} = scalar.expf<afn> %md{j}_{i} : f32
      %mla{j}_{i} = scalar.addf {pl}, %mp{j}_{i} : f32
      %mpv{j}_{i} = vector.splat %mp{j}_{i} : {vt}
      %maa{j}_{i} = vector.fmaf %mpv{j}_{i}, %mv{i}, {pa} : {vt}""")
            pl, pa = f"%mla{j}_{i}", f"%maa{j}_{i}"
        m.append(f"""      %mbr1{j} = index.cmp ult, %mj, %mt_hi{j} : index
      %mbr2{j} = index.cmp ugt, %mjend, %mt_lo{j} : index
      %mbr{j} = scalar.andi %mbr1{j}, %mbr2{j} : i1
      %mnm{j} = scf.select %mbr{j}, {prev}, %mm{j} : f32
      %mnl{j} = scf.select %mbr{j}, {pl}, %ml{j} : f32
      %mna{j} = scf.select %mbr{j}, {pa}, %ma{j} : {vt}""")
        yields += [f"%mnm{j}", f"%mnl{j}", f"%mna{j}"]
    m.append(f"      scf.yield {', '.join(yields)} : {rtypes}\n    }}")
    m.append(f"""    %mt_acc_view = buffer.view %part_acc[%zero] : buffer -> view<[%tokens]x{QH}x{S}x32x4xf32>
    %mt_ml_view = buffer.view %part_ml[%zero] : buffer -> view<[%tokens]x{QH}x{S}x2xf32>
    %mt_first = index.cmp eq, %lane, %c0 : index""")
    for j in range(T):
        m.append(f"""    scf.if %mt_act{j} {{
      vector.store %mfa{j}, %mt_acc_view[%mg_t{j}, %bounded_head, %split, %lane, %c0] : {vt}, view<[%tokens]x{QH}x{S}x32x4xf32>
      scf.if %mt_first {{
        view.store %mfm{j}, %mt_ml_view[%mg_t{j}, %bounded_head, %split, 0] : f32, view<[%tokens]x{QH}x{S}x2xf32>
        view.store %mfl{j}, %mt_ml_view[%mg_t{j}, %bounded_head, %split, 1] : f32, view<[%tokens]x{QH}x{S}x2xf32>
      }}
    }}""")
    m.append("  }\n")
    return text[:a] + k + "\n".join(m) + text[b:]


def attention_text(S, P):
    """Split-K causal GQA attention: workgroup (token, KV head, split) of 16
    waves, wave w = query head 16*kv + w, lane l = dims 4l..4l+3. Sliding
    layers see the last 2048 positions (including the current one). The
    combine kernel folds the splits and applies the sigmoid output gate."""
    U = UNROLL
    body = []
    pa = attn_rules.paged(U)
    if pa:   # absolute U-aligned key blocks lie in one 16-position KV page: one page lookup per block
        body.append(f"""      %kb_off = func.call @g_kv_offset(%origins, %row, %j, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)""")
    for i in range(U):
        body.append((f"""      %key{i}_raw = index.add %j, %c{i}u : index
      %okl{i} = index.cmp uge, %key{i}_raw, %lo_abs : index
      %okh{i} = index.cmp ult, %key{i}_raw, %hi_abs : index
      %ok{i} = scalar.andi %okl{i}, %okh{i} : i1
      %kd{i} = index.constant {i * POS_BYTES} : offset
      %ko{i} = index.add %kb_off, %kd{i} : offset""" if pa else f"""      %key{i}_raw = index.add %j, %c{i}u : index
      %ok{i} = index.cmp ult, %key{i}_raw, %stop : index
      %key{i}_c = scf.select %ok{i}, %key{i}_raw, %j : index
      %kp{i} = index.add %window_low, %key{i}_c : index
      %ko{i} = func.call @g_kv_offset(%origins, %row, %kp{i}, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)""") + f"""
      %kc{i} = buffer.view %arena[%ko{i}] : buffer -> view<{KVH}x32x4xf16>
      %kh{i} = vector.load %kc{i}[%kv_head, %lane, %c0] : view<{KVH}x32x4xf16> -> vector<4xf16>
      %k{i} = vector.extf %kh{i} : vector<4xf16> to vector<4xf32>
      %vo{i} = index.add %ko{i}, %plane_stride : offset
      %vc{i} = buffer.view %arena[%vo{i}] : buffer -> view<{KVH}x32x4xf16>
      %vh{i} = vector.load %vc{i}[%kv_head, %lane, %c0] : view<{KVH}x32x4xf16> -> vector<4xf16>
      %v{i} = vector.extf %vh{i} : vector<4xf16> to vector<4xf32>""")
    ab = attn_rules.butterfly(U)
    for i in range(U):
        body.append(f"""      %prod{i} = vector.mulf %q4, %k{i} : vector<4xf32>
      %part{i} = vector.reduce<addf> %prod{i}, %zero_f : vector<4xf32>, f32""" + ("" if ab else f"""
      %sr{i} = kernel.subgroup.reduce<addf> %part{i} : f32"""))
    if ab:   # all U key scores of the block in one butterfly all-reduce (~2U shuffles instead of 5U)
        body.append(allreduce([f"%part{i}" for i in range(U)], "ar", [f"%sr{i}" for i in range(U)]))
    for i in range(U):
        body.append(f"""      %s{i} = scf.select %ok{i}, %sr{i}, %neg_big : f32""")
    # ATTN_ILP independent online-softmax chains per block (keys interleaved by halves), merged after the loop:
    # shorter dependent chains, more overlap in a latency-bound loop.
    ILP = attn_rules.ilp(U)
    yields = []
    for c in range(ILP):
        sx = "" if c == 0 else str(c + 1)
        keys = list(range(c * U // ILP, (c + 1) * U // ILP))
        prev = f"%m{sx}"
        for i in keys:
            body.append(f"      %mx{i} = scalar.maxnumf {prev}, %s{i} : f32"); prev = f"%mx{i}"
        body.append(f"""      %m_shift{sx} = scalar.subf %m{sx}, {prev} : f32
      %alpha{sx} = scalar.expf<afn> %m_shift{sx} : f32
      %l_scaled{sx} = scalar.mulf %l{sx}, %alpha{sx} : f32
      %alpha_v{sx} = vector.splat %alpha{sx} : vector<4xf32>
      %acc_s{sx} = vector.mulf %acc{sx}, %alpha_v{sx} : vector<4xf32>""")
        pl, pa = f"%l_scaled{sx}", f"%acc_s{sx}"
        for i in keys:
            body.append(f"""      %d{i} = scalar.subf %s{i}, {prev} : f32
      %p{i} = scalar.expf<afn> %d{i} : f32
      %la{i} = scalar.addf {pl}, %p{i} : f32
      %pv{i} = vector.splat %p{i} : vector<4xf32>
      %aa{i} = vector.fmaf %pv{i}, %v{i}, {pa} : vector<4xf32>"""); pl, pa = f"%la{i}", f"%aa{i}"
        yields += [prev, pl, pa]
    body.append(f"      scf.yield {', '.join(yields)} : " + ", ".join(["f32, f32, vector<4xf32>"] * ILP))
    ILP_ARGS = "".join(f", %m{c + 1} = %neg_big : f32, %l{c + 1} = %zero_f : f32, %acc{c + 1} = %zero4 : vector<4xf32>" for c in range(1, ILP))
    ILP_TYPES = ", f32, f32, vector<4xf32>" * (ILP - 1)
    ILP_OUTS = "".join(f", %mf{c + 1}, %lf{c + 1}, %af{c + 1}" for c in range(1, ILP))
    ILP_MERGE = ""
    FIN = ("%m_final0, %l_final0, %acc_final0" + ILP_OUTS) if ILP > 1 else "%m_final, %l_final, %acc_final"
    if ILP > 1:
        mm = "%m_r"
        ILP_MERGE = "    %m_r = scalar.addf %m_final0, %zero_f : f32\n" + "".join(f"    %mm{c} = scalar.maxnumf {'%m_r' if c == 1 else f'%mm{c - 1}'}, %mf{c + 1} : f32\n" for c in range(1, ILP))
        top = f"%mm{ILP - 1}"
        ILP_MERGE += f"""    %w0s = scalar.subf %m_final0, {top} : f32
    %w0 = scalar.expf<afn> %w0s : f32
    %wl0 = scalar.mulf %l_final0, %w0 : f32
    %w0v = vector.splat %w0 : vector<4xf32>
    %wa0 = vector.mulf %acc_final0, %w0v : vector<4xf32>
"""
        pl, pa = "%wl0", "%wa0"
        for c in range(1, ILP):
            ILP_MERGE += f"""    %w{c}s = scalar.subf %mf{c + 1}, {top} : f32
    %w{c} = scalar.expf<afn> %w{c}s : f32
    %wl{c} = scalar.fmaf %lf{c + 1}, %w{c}, {pl} : f32
    %w{c}v = vector.splat %w{c} : vector<4xf32>
    %wa{c} = vector.fmaf %af{c + 1}, %w{c}v, {pa} : vector<4xf32>
"""
            pl, pa = f"%wl{c}", f"%wa{c}"
        ILP_MERGE += f"""    %m_final = scalar.addf {top}, %zero_f : f32
    %l_final = scalar.addf {pl}, %zero_f : f32
    %acc_final = vector.addf {pa}, %zero4 : vector<4xf32>
"""
    consts = "".join(f"  %c{i}u = index.constant {i} : index\n" for i in range(U))
    GATE_TEXT = """    %gate_view = buffer.view %gate[%zero] : buffer -> view<[%tokens]x{QH}x32x4xf32>
    %g = vector.load %gate_view[%token, %head, %lane, %c0] : view<[%tokens]x{QH}x32x4xf32> -> vector<4xf32>
    %sig = vector.logisticf<afn> %g : vector<4xf32>
    %result = vector.mulf %attended, %sig : vector<4xf32>""".replace("{QH}", str(QH)).replace(
        "vector.logisticf<afn> %g", "vector.addf %g, %zero4" if GATE_ACT != "sigmoid" else "vector.logisticf<afn> %g") if GATE else "    %result = vector.addf %attended, %zero4 : vector<4xf32>"
    if SINKS:
        SINK_LOAD = f"""    %sinks_view = buffer.view %gate[%zero] : buffer -> view<{QH}xf32>
    %sink = view.load %sinks_view[%head] : view<{QH}xf32> -> f32
"""
        SINK_W = """    %sink_shift = scalar.subf %sink, %maximum : f32
    %sink_w = scalar.expf<afn> %sink_shift : f32
"""
        M_INIT, D_INIT = "%sink", "%sink_w"
    else:
        SINK_LOAD = SINK_W = ""
        M_INIT, D_INIT = "%neg_big", "%zero_f"
    text = f"""
// Splits for `visible` keys: about 32 keys each, at most {S}.
func.def inline @{P}_split_count(%visible: index) -> (index) {{
  %c1 = index.constant 1 : index
  %c127 = index.constant 31 : index
  %c128 = index.constant 32 : index
  %cmax = index.constant {S} : index
  %padded = index.add %visible, %c127 : index
  %target = index.div %padded, %c128 : index
  %capped = index.min %target, %cmax : index
  %count_value = index.max %capped, %c1 : index
  %count = index.assume %count_value [range(%count_value, 1, {S})] : index
  func.return %count : index
}}

kernel.def export("{P}_attention_split") @{P}_attention_split(%tokens: index) {{
  %c1 = index.constant 1 : index
  %ckvh = index.constant {KVH} : index
  %cS = index.constant {S} : index
  %clanes = index.constant {G * 32} : index
  kernel.launch.config workgroups(%tokens, %ckvh, %cS) workgroup_size(%clanes, %c1, %c1) : index
}} launch(%tokens_arg: index, %layer_arg: index, %sliding: i32, %context_arg: index, %pool_arg: index, %q: buffer, %info: buffer, %origins: buffer, %arena: buffer, %part_acc: buffer, %part_ml: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, 51)] : index
  %context = index.assume %context_arg [range(%context_arg, 16, 262144)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 1048576)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %kv_id = kernel.workgroup.id<y> : index
  %kv_head = index.assume %kv_id [range(%kv_id, 0, {KVH - 1})] : index
  %split_id = kernel.workgroup.id<z> : index
  %split = index.assume %split_id [range(%split_id, 0, {S - 1})] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, {G - 1})] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c15 = index.constant 15 : index
  %c16 = index.constant 16 : index
  %window = index.constant {WINDOW or 262144} : index
  %window_less = index.constant {(WINDOW or 262144) - 1} : index
  %c0_i32 = scalar.constant 0 : i32
  %cU = index.constant {U} : index
{consts}{ALLRED_SETUP}  %zero_f = scalar.constant 0.0 : f32
  %neg_big = scalar.constant -3.0e38 : f32
  %cg = index.constant {G} : index
  %head = index.madd %kv_head, %cg, %wave : index
  %bounded_head = index.assume %head [range(%head, 0, {QH - 1})] : index
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %row32 = view.load %info_view[%token, 0] : view<[%tokens]x4xi32> -> i32
  %position32 = view.load %info_view[%token, 1] : view<[%tokens]x4xi32> -> i32
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %needed = scalar.constant true : i1
  %live_token = scalar.cmpi ne, %live32, %c0_i32 : i32
  %live = scalar.andi %live_token, %needed : i1
  scf.if %live {{
    %row_bound = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
    %row = index.cast %row_bound : i32 to index
    %position_bound = scalar.assume %position32 [range(%position32, 0, 262143)] : i32
    %position = index.cast %position_bound : i32 to index
    %padded = index.add %context, %c15 : index
    %blocks = index.div %padded, %c16 : index
    %k_plane = index.mul %layer, %c2 : index
    %v_plane = index.add %k_plane, %c1 : index
    // V of a position sits one plane (pool positions x 512 bytes) after its K.
    %position_bytes_v = index.constant {POS_BYTES} : offset
    %plane_stride = index.scale %pool, %position_bytes_v : index, offset -> offset
    %is_sliding = scalar.cmpi ne, %sliding, %c0_i32 : i32
    %far = index.cmp uge, %position, %window : index
    %use_window = scalar.andi %is_sliding, %far : i1
    %window_start = index.sub %position, %window_less : index
    %window_low = scf.select %use_window, %window_start, %c0 : index
    %end = index.add %position, %c1 : index
    %visible_value = index.sub %end, %window_low : index
    %visible = index.assume %visible_value [range(%visible_value, 1, 262144)] : index
    %s_eff = func.call @{P}_split_count(%visible) : (index) -> (index)
    %active_split = index.cmp ult, %split, %s_eff : index
    scf.if %active_split {{
    %s_less = index.sub %s_eff, %c1 : index
    %per_padded = index.add %visible, %s_less : index
    %per = index.div %per_padded, %s_eff : index
    %start_raw = index.mul %split, %per : index
    %start = index.min %start_raw, %visible : index
    %stop_raw = index.add %start_raw, %per : index
    %stop = index.min %stop_raw, %visible : index
    %q_view = buffer.view %q[%zero] : buffer -> view<[%tokens]x{QH}x32x4xf32>
    %q_raw = vector.load %q_view[%token, %bounded_head, %lane, %c0] : view<[%tokens]x{QH}x32x4xf32> -> vector<4xf32>
    %scale = scalar.constant 0.08838834764831845 : f32
    %scale_v = vector.splat %scale : vector<4xf32>
    %q4 = vector.mulf %q_raw, %scale_v : vector<4xf32>
    %zero4 = vector.splat %zero_f : vector<4xf32>
""" + ("""    %lo_abs = index.add %window_low, %start : index
    %hi_abs = index.add %window_low, %stop : index
    %lo_q = index.div %lo_abs, %cU : index
    %lo_al = index.mul %lo_q, %cU : index
    """ + FIN + """ = scf.for %j = [%lo_al to %hi_abs step %cU](%m = %neg_big : f32, %l = %zero_f : f32, %acc = %zero4 : vector<4xf32>""" + ILP_ARGS + """) -> (f32, f32, vector<4xf32>""" + ILP_TYPES + """) {""" if pa else """    """ + FIN + """ = scf.for %j = [%start to %stop step %cU](%m = %neg_big : f32, %l = %zero_f : f32, %acc = %zero4 : vector<4xf32>""" + ILP_ARGS + """) -> (f32, f32, vector<4xf32>""" + ILP_TYPES + """) {""") + f"""
{chr(10).join(body)}
    }}
{ILP_MERGE}
    %acc_view = buffer.view %part_acc[%zero] : buffer -> view<[%tokens]x{QH}x{S}x32x4xf32>
    vector.store %acc_final, %acc_view[%token, %bounded_head, %split, %lane, %c0] : vector<4xf32>, view<[%tokens]x{QH}x{S}x32x4xf32>
    %first = index.cmp eq, %lane, %c0 : index
    scf.if %first {{
      %ml_view = buffer.view %part_ml[%zero] : buffer -> view<[%tokens]x{QH}x{S}x2xf32>
      view.store %m_final, %ml_view[%token, %bounded_head, %split, 0] : f32, view<[%tokens]x{QH}x{S}x2xf32>
      view.store %l_final, %ml_view[%token, %bounded_head, %split, 1] : f32, view<[%tokens]x{QH}x{S}x2xf32>
    }}
    }}
  }}
  kernel.return
}}

// o[t, head] = (sum_s e^(m_s - M) acc_s / sum_s e^(m_s - M) l_s) * sigmoid(gate[t, head]).
kernel.def export("{P}_attention_combine") @{P}_attention_combine(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  %cqh = index.constant {QH} : index
  kernel.launch.config workgroups(%tokens, %cqh, %c1) workgroup_size(%c32, %c1, %c1) : index
}} launch(%tokens_arg: index, %sliding: i32, %info: buffer, %part_acc: buffer, %part_ml: buffer, %gate: buffer, %out: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %head_id = kernel.workgroup.id<y> : index
  %head = index.assume %head_id [range(%head_id, 0, {QH - 1})] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %cSc = index.constant {S} : index
  %c0_i32 = scalar.constant 0 : i32
  %zero_f = scalar.constant 0.0 : f32
  %neg_big = scalar.constant -3.0e38 : f32
  %zero4 = vector.splat %zero_f : vector<4xf32>
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %needed = scalar.constant true : i1
  %out_view = buffer.view %out[%zero] : buffer -> view<[%tokens]x{QH}x32x4xf32>
  scf.if %needed {{
  scf.if %live {{
    %position32 = view.load %info_view[%token, 1] : view<[%tokens]x4xi32> -> i32
    %position_bound = scalar.assume %position32 [range(%position32, 0, 262143)] : i32
    %position = index.cast %position_bound : i32 to index
    %window = index.constant {WINDOW or 262144} : index
    %window_less = index.constant {(WINDOW or 262144) - 1} : index
    %is_sliding = scalar.cmpi ne, %sliding, %c0_i32 : i32
    %far = index.cmp uge, %position, %window : index
    %use_window = scalar.andi %is_sliding, %far : i1
    %window_start = index.sub %position, %window_less : index
    %window_low = scf.select %use_window, %window_start, %c0 : index
    %end = index.add %position, %c1 : index
    %visible_value = index.sub %end, %window_low : index
    %visible = index.assume %visible_value [range(%visible_value, 1, 262144)] : index
    %s_eff = func.call @{P}_split_count(%visible) : (index) -> (index)
    %ml_view = buffer.view %part_ml[%zero] : buffer -> view<[%tokens]x{QH}x{S}x2xf32>
    %acc_view = buffer.view %part_acc[%zero] : buffer -> view<[%tokens]x{QH}x{S}x32x4xf32>
{SINK_LOAD}    %maximum = scf.for %s = [%c0 to %s_eff step %c1](%mx = {M_INIT} : f32) -> (f32) {{
      %bs = index.assume %s [range(%s, 0, {S - 1})] : index
      %ms = view.load %ml_view[%token, %head, %bs, 0] : view<[%tokens]x{QH}x{S}x2xf32> -> f32
      %next = scalar.maxnumf %mx, %ms : f32
      scf.yield %next : f32
    }}
{SINK_W}    %den, %total = scf.for %s = [%c0 to %s_eff step %c1](%d = {D_INIT} : f32, %acc = %zero4 : vector<4xf32>) -> (f32, vector<4xf32>) {{
      %bs = index.assume %s [range(%s, 0, {S - 1})] : index
      %ms = view.load %ml_view[%token, %head, %bs, 0] : view<[%tokens]x{QH}x{S}x2xf32> -> f32
      %ls = view.load %ml_view[%token, %head, %bs, 1] : view<[%tokens]x{QH}x{S}x2xf32> -> f32
      %shift = scalar.subf %ms, %maximum : f32
      %w = scalar.expf<afn> %shift : f32
      %wl = scalar.mulf %w, %ls : f32
      %d_next = scalar.addf %d, %wl : f32
      %part = vector.load %acc_view[%token, %head, %bs, %lane, %c0] : view<[%tokens]x{QH}x{S}x32x4xf32> -> vector<4xf32>
      %w_v = vector.splat %w : vector<4xf32>
      %acc_next = vector.fmaf %w_v, %part, %acc : vector<4xf32>
      scf.yield %d_next, %acc_next : f32, vector<4xf32>
    }}
    %one = scalar.constant 1.0 : f32
    %inv = scalar.divf %one, %den : f32
    %inv_v = vector.splat %inv : vector<4xf32>
    %attended = vector.mulf %total, %inv_v : vector<4xf32>
{GATE_TEXT}
    vector.store %result, %out_view[%token, %head, %lane, %c0] : vector<4xf32>, view<[%tokens]x{QH}x32x4xf32>
  }} else {{
    vector.store %zero4, %out_view[%token, %head, %lane, %c0] : vector<4xf32>, view<[%tokens]x{QH}x32x4xf32>
  }}
  }}
  kernel.return
}}
"""
    if P == "g" and attn_rules.multi_token(U, KVH):
        text = multi_token_split(text, P, U, ATTN_MT)
    if D != 128:
        text = (text.replace("x32x4xf16", f"x32x{DL}xf16").replace("vector<4xf16>", f"vector<{DL}xf16>")
                .replace("x32x4xf32", f"x32x{DL}xf32").replace("vector<4xf32>", f"vector<{DL}xf32>")
                .replace("0.08838834764831845", repr(D ** -0.5)))
    split_l = listed(text, f"{P}_attention_split", f"""  %c1 = index.constant 1 : index
  %ckvh = index.constant {KVH} : index
  %cS = index.constant {S} : index
  %clanes = index.constant {G * 32} : index
  kernel.launch.config workgroups(%count, %ckvh, %cS) workgroup_size(%clanes, %c1, %c1) : index""")
    combine_l = listed(text, f"{P}_attention_combine", f"""  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  %cqh = index.constant {QH} : index
  kernel.launch.config workgroups(%count, %cqh, %c1) workgroup_size(%c32, %c1, %c1) : index""")
    return text, split_l, combine_l


# Long-context decode attention over verification spans (drafting packages). The split kernel streams the whole KV
# once per token, so a verify pass of n tokens cost n single-token passes at long context (Qwen2.5-0.5B at 32k, depth-2
# drafting: 249 tok/s, 2.65 tokens and 10.6 ms per pass, 7 ms of it attention). On full-attention layers, tokens at
# position >= SEG_MIN take the span body inside g_attention_split instead: a segment (a run of up to depth + 1
# consecutive same-row tokens) is run by its leader's workgroup per (KV head, split), which reads each 32-key block's
# K and V once for all its tokens; the other tokens of the segment exit. Lane l takes key j + l for the scores (each token's q row from workgroup memory, a
# wave max and sum per token and block); the values accumulate lanes-over-dims, each key's p read back from workgroup
# memory. The partials keep the split kernel's layout, so g_attention_combine is unchanged. A token's blocks are
# absolute 32-aligned and a block outside its own range is skipped, so it computes the same bits alone or in any
# segment: drafted output equals undrafted output. Below SEG_MIN the split kernel stays as fast or faster.
# On for head_dim 64 and 128 with at most 2 KV heads (few workgroups per token for the split kernel); the span length
# is capped so the workgroup tiles fit 64 KB. Measured losses outside that: 8 KV heads (Qwen3-4B DFlash, -17 %
# locally), head_dim 256 with one KV head (Gemma-3-1B, 8k/16k -40 %). SEG_ATTN=0 keeps the split kernel everywhere.
SEG_MIN = int(os.environ.get("SEG_MIN", "8192"))


def seg_tokens():
    """Tokens per segment: the verification span (draft depth + 1), capped so the q and p tiles fit 64 KB of
    workgroup memory (longer runs go one token at a time); 0 when the decode path keeps the split kernel."""
    if not (os.environ.get("SEG_ATTN", "1") == "1" and gen_draft.enabled() and D in (64, 128) and KVH <= 2):
        return 0
    ts = min(gen_draft.DRAFT["depth"] + 1, 65536 // (G * 4 * (D + 32)))
    return ts if ts >= 2 else 0


def attention_seg(split_text):
    """The span body, from the decode split kernel's text (attention_text(64, "g")): same partials."""
    TS, S, NC, PB = seg_tokens(), 64, D // 32, POS_BYTES
    V = f"vector<{DL}xf32>"
    # values: VW dims per lane, so each key's V row spans LPK lanes and one load step covers KS keys
    VW = 4 if D == 64 else DL   # head_dim 64: two keys per load step (2 dims per lane lost on DevLab: -21 %)
    LPK, VA = D // VW, f"vector<{VW}xf32>"
    KS = 32 // LPK
    a = split_text.index('kernel.def export("g_attention_split")')
    head = split_text[a:split_text.index("  %info_view = buffer.view", a)].replace("g_attention_split", "g_attention_seg")
    L = [head, f"""  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %row32 = view.load %info_view[%token, 0] : view<[%tokens]x4xi32> -> i32
  %seg_min = scalar.constant {SEG_MIN} : i32
  %true = scalar.constant true : i1
  %tokens_less = index.sub %tokens, %c1 : index
  %c32k = index.constant 32 : index
  %pb = index.constant {PB} : offset
  %far_lo = index.constant {1 << 30} : index
  %scale = scalar.constant {D ** -0.5!r} : f32
  %scale_v = vector.splat %scale : {V}
  %zero_acc = vector.splat %zero_f : {VA}
  %one_f = scalar.constant 1.0 : f32
  %q_bytes = index.constant {TS * G * D * 4} : offset
  %p_bytes = index.constant {G * TS * 32 * 4} : offset
  %q_tile = buffer.alloca<workgroup> align(16) %q_bytes : buffer
  %p_tile = buffer.alloca<workgroup> align(16) %p_bytes : buffer
  %q_lds = buffer.view %q_tile[%zero] : buffer -> view<{TS}x{G}x32x{DL}xf32>
  %q_row = buffer.view %q_tile[%zero] : buffer -> view<{TS}x{G}x{D}xf32>
  %p_lds = buffer.view %p_tile[%zero] : buffer -> view<{G}x{TS}x32xf32>
  %q_view = buffer.view %q[%zero] : buffer -> view<[%tokens]x{QH}x32x{DL}xf32>
  %cont0 = func.call @g_seg_next(%info, %tokens, %token, %seg_min) : (buffer, index, index, i32) -> (i1)
  %first = scalar.xori %cont0, %true : i1"""]
    # the token is a segment leader (g_seg_leader); members: the next tokens of its run when it starts it
    for k in range(1, TS):
        prev_m = "%first" if k == 1 else f"%member{k - 1}"
        L.append(f"""  %cb{k} = index.constant {k} : index
  %ta{k} = index.add %token, %cb{k} : index
  %next{k} = func.call @g_seg_next(%info, %tokens, %ta{k}, %seg_min) : (buffer, index, index, i32) -> (i1)
  %member{k} = scalar.andi {prev_m}, %next{k} : i1""")
    L.append("""  %row_bound = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
  %row = index.cast %row_bound : i32 to index
  %padded = index.add %context, %c15 : index
  %blocks = index.div %padded, %c16 : index
  %k_plane = index.mul %layer, %c2 : index
  %plane_stride = index.scale %pool, %pb : index, offset -> offset""")
    lo, hi = "%far_lo", "%c0"
    for k in range(TS):
        L.append(f"""    %ck{k} = index.constant {k} : index
    %tr{k} = index.add %token, %ck{k} : index
    %tm{k} = index.min %tr{k}, %tokens_less : index
    %tok{k} = index.assume %tm{k} [lt(%tm{k}, %tokens)] : index
    %position32_{k} = view.load %info_view[%tok{k}, 1] : view<[%tokens]x4xi32> -> i32
    %position_bound_{k} = scalar.assume %position32_{k} [range(%position32_{k}, 0, 262143)] : i32
    %position_{k} = index.cast %position_bound_{k} : i32 to index
    %visible_value_{k} = index.add %position_{k}, %c1 : index
    %visible_{k} = index.assume %visible_value_{k} [range(%visible_value_{k}, 1, 262144)] : index
    %s_eff_{k} = func.call @g_split_count(%visible_{k}) : (index) -> (index)
    %in_split_{k} = index.cmp ult, %split, %s_eff_{k} : index
    %active_{k} = scalar.andi %member{k}, %in_split_{k} : i1
    %s_less_{k} = index.sub %s_eff_{k}, %c1 : index
    %per_padded_{k} = index.add %visible_{k}, %s_less_{k} : index
    %per_{k} = index.div %per_padded_{k}, %s_eff_{k} : index
    %start_raw_{k} = index.mul %split, %per_{k} : index
    %start_{k} = index.min %start_raw_{k}, %visible_{k} : index
    %stop_raw_{k} = index.add %start_raw_{k}, %per_{k} : index
    %stop_{k} = index.min %stop_raw_{k}, %visible_{k} : index
    %lo_sel_{k} = scf.select %active_{k}, %lo_{k}, %far_lo : index
    %hi_sel_{k} = scf.select %active_{k}, %hi_{k}, %c0 : index
    %lo_min_{k} = index.min {lo}, %lo_sel_{k} : index
    %hi_max_{k} = index.max {hi}, %hi_sel_{k} : index
    scf.if %active_{k} {{
      %qr{k} = vector.load %q_view[%tok{k}, %bounded_head, %lane, %c0] : view<[%tokens]x{QH}x32x{DL}xf32> -> {V}
      %qs{k} = vector.mulf %qr{k}, %scale_v : {V}
      vector.store %qs{k}, %q_lds[%ck{k}, %wave, %lane, %c0] : {V}, view<{TS}x{G}x32x{DL}xf32>
    }}""")
        lo, hi = f"%lo_min_{k}", f"%hi_max_{k}"
    L.append(f"""    %hi_u_less = index.sub %hi_u, %c1 : index
    %lo_b = index.div %lo_u, %c32k : index
    %lo_al = index.mul %lo_b, %c32k : index
    %upper = index.cmp uge, %lane, %c16 : index
    %clpk = index.constant {LPK} : index
    %vsub = index.div %lane, %clpk : index
    %vlane = index.rem %lane, %clpk : index
    %vsub_odd = index.cmp ne, %vsub, %c0 : index
""" + "".join(f"    %pg{g} = index.constant {4 * g} : index\n" for g in range(8)) + """    kernel.barrier<workgroup> scope(subgroup) ordering(acq_rel)""")
    st = ", ".join(f"%sm{k} = %neg_big : f32, %sl{k} = %zero_f : f32, %sacc{k} = %zero_acc : {VA}" for k in range(TS))
    ty = ", ".join([f"f32, f32, {VA}"] * TS)
    fin = ", ".join(f"%fm{k}, %fl{k}, %fa{k}" for k in range(TS))
    L.append(f"""    {fin} = scf.for %j = [%lo_al to %hi_u step %c32k]({st}) -> ({ty}) {{
      %j16 = index.add %j, %c16 : index
      %j32 = index.add %j, %c32k : index
      %alo_a = index.max %j, %lo_u : index
      %anc_lo = index.min %alo_a, %hi_u_less : index
      %ahi_a = index.max %j16, %lo_u : index
      %anc_hi = index.min %ahi_a, %hi_u_less : index
      %base_lo = func.call @g_kv_offset(%origins, %row, %anc_lo, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
      %base_hi = func.call @g_kv_offset(%origins, %row, %anc_hi, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
      %key = index.add %j, %lane : index
      %oku_l = index.cmp uge, %key, %lo_u : index
      %oku_h = index.cmp ult, %key, %hi_u : index
      %ok_u = scalar.andi %oku_l, %oku_h : i1
      %anc = scf.select %upper, %anc_hi, %anc_lo : index
      %kbase = scf.select %upper, %base_hi, %base_lo : offset
      %key_c = scf.select %ok_u, %key, %anc : index
      %kdelta = index.sub %key_c, %anc : index
      %kdb = index.scale %kdelta, %pb : index, offset -> offset
      %kofs = index.add %kbase, %kdb : offset
      %krow = buffer.view %arena[%kofs] : buffer -> view<{KVH}x{D}xf16>""")
    for c in range(NC):
        L.append(f"""      %kd{c} = index.constant {32 * c} : index
      %kh{c} = vector.load %krow[%kv_head, %kd{c}] : view<{KVH}x{D}xf16> -> vector<32xf16>
      %kf{c} = vector.extf %kh{c} : vector<32xf16> to vector<32xf32>""")
    for k in range(TS):
        dots, prev = [], "%zero_f"
        for c in range(NC):
            dots.append(f"""        %qc{k}_{c} = vector.load %q_row[%ck{k}, %wave, %kd{c}] : view<{TS}x{G}x{D}xf32> -> vector<32xf32>
        %sd{k}_{c} = vector.dotf<reassoc|contract> %kf{c}, %qc{k}_{c}, {prev} : vector<32xf32>, vector<32xf32>, f32""")
            prev = f"%sd{k}_{c}"
        L.append(f"""      %bl{k} = index.cmp ult, %j, %hi_{k} : index
      %bh{k} = index.cmp ugt, %j32, %lo_{k} : index
      %bb{k} = scalar.andi %bl{k}, %bh{k} : i1
      %use{k} = scalar.andi %bb{k}, %active_{k} : i1
      %nm{k}, %nl{k}, %al{k} = scf.if %use{k} -> (f32, f32, f32) {{
        %okl{k} = index.cmp uge, %key, %lo_{k} : index
        %okh{k} = index.cmp ult, %key, %hi_{k} : index
        %ok{k} = scalar.andi %okl{k}, %okh{k} : i1
{chr(10).join(dots)}
        %score{k} = scf.select %ok{k}, {prev}, %neg_big : f32
        %bmax{k} = kernel.subgroup.reduce<maxnumf> %score{k} : f32
        %m_new{k} = scalar.maxnumf %sm{k}, %bmax{k} : f32
        %m_shift{k} = scalar.subf %sm{k}, %m_new{k} : f32
        %alpha{k} = scalar.expf<afn> %m_shift{k} : f32
        %dp{k} = scalar.subf %score{k}, %m_new{k} : f32
        %p{k} = scalar.expf<afn> %dp{k} : f32
        %psum{k} = kernel.subgroup.reduce<addf> %p{k} : f32
        %l_new{k} = scalar.fmaf %sl{k}, %alpha{k}, %psum{k} : f32
        view.store %p{k}, %p_lds[%wave, %ck{k}, %lane] : f32, view<{G}x{TS}x32xf32>
        scf.yield %m_new{k}, %l_new{k}, %alpha{k} : f32, f32, f32
      }} else {{
        scf.yield %sm{k}, %sl{k}, %one_f : f32, f32, f32
      }}
      %alv{k} = vector.splat %al{k} : {VA}
      %as{k} = vector.mulf %sacc{k}, %alv{k} : {VA}""")
    # values: lanes over dims, each of the block's 32 keys read once for every token; a key outside the union (or
    # another token's) is read at a valid clamped position and weighted by p = +0, an exact no-op (acc is never -0)
    L.append(f"""      kernel.barrier<workgroup> scope(subgroup) ordering(acq_rel)
      %vlo = index.add %base_lo, %plane_stride : offset
      %vhi = index.add %base_hi, %plane_stride : offset
      %vrow_lo = buffer.view %arena[%vlo] : buffer -> view<16x{KVH}x{LPK}x{VW}xf16>
      %vrow_hi = buffer.view %arena[%vhi] : buffer -> view<16x{KVH}x{LPK}x{VW}xf16>
      %pe_lo_a = index.add %j, %c15 : index
      %pe_lo = index.min %pe_lo_a, %hi_u_less : index
      %c31k = index.constant 31 : index
      %pe_hi_a = index.add %j, %c31k : index
      %pe_hi = index.min %pe_hi_a, %hi_u_less : index""")
    for k in range(TS):   # each token's 32 p values of the block, 4 per load
        for g in range(8):
            L.append(f"""      %pq{k}_{g} = vector.load %p_lds[%wave, %ck{k}, %pg{g}] : view<{G}x{TS}x32xf32> -> vector<4xf32>""")
    prev = {k: f"%as{k}" for k in range(TS)}
    for jj in range(0, 32, KS):   # keys jj .. jj + KS - 1, lane sub-group i takes key jj + i (KS <= 2)
        h = "hi" if jj >= 16 else "lo"
        L.append(f"""      %vc{jj} = index.constant {jj} : index
      %vk0_{jj} = index.add %j, %vc{jj} : index
      %vk{jj} = index.add %vk0_{jj}, %vsub : index
      %vka{jj} = index.min %vk{jj}, %pe_{h} : index
      %vkc{jj} = index.max %vka{jj}, %anc_{h} : index
      %vi{jj} = index.sub %vkc{jj}, %anc_{h} : index
      %vix{jj} = index.assume %vi{jj} [range(%vi{jj}, 0, 15)] : index
      %vh{jj} = vector.load %vrow_{h}[%vix{jj}, %kv_head, %vlane, %c0] : view<16x{KVH}x{LPK}x{VW}xf16> -> vector<{VW}xf16>
      %vf{jj} = vector.extf %vh{jj} : vector<{VW}xf16> to {VA}""")
        for k in range(TS):
            pick = (f"""      %px{jj}_{k} = vector.extract %pq{k}_{jj // 4}[{jj % 4}] : vector<4xf32> -> f32""" if KS == 1 else
                    f"""      %pxa{jj}_{k} = vector.extract %pq{k}_{jj // 4}[{jj % 4}] : vector<4xf32> -> f32
      %pxb{jj}_{k} = vector.extract %pq{k}_{jj // 4}[{jj % 4 + 1}] : vector<4xf32> -> f32
      %px{jj}_{k} = scf.select %vsub_odd, %pxb{jj}_{k}, %pxa{jj}_{k} : f32""")
            L.append(pick + f"""
      %pe{jj}_{k} = scf.select %use{k}, %px{jj}_{k}, %zero_f : f32
      %pv{jj}_{k} = vector.splat %pe{jj}_{k} : {VA}
      %va{jj}_{k} = vector.fmaf %pv{jj}_{k}, %vf{jj}, {prev[k]} : {VA}""")
            prev[k] = f"%va{jj}_{k}"
    L.append("      kernel.barrier<workgroup> scope(subgroup) ordering(acq_rel)")
    L.append("      scf.yield " + ", ".join(f"%nm{k}, %nl{k}, {prev[k]}" for k in range(TS)) + f" : {ty}\n    }}")
    if KS > 1:   # each lane group summed half the keys: add the partner group's sums, then DL dims per lane
        L.append(f"""    %fold_w = scalar.constant 32 : i32
    %fold_d = scalar.constant {LPK} : i32
    %acc_rows = buffer.view %q_tile[%zero] : buffer -> view<{G}x{D}xf32>
    %acc_out = buffer.view %q_tile[%zero] : buffer -> view<{G}x32x{DL}xf32>
    %lead_group = index.cmp eq, %vsub, %c0 : index
    %vw_c = index.constant {VW} : index
    %vdim = index.mul %vlane, %vw_c : index""")
        for k in range(TS):
            el = []
            for e in range(VW):
                L.append(f"""    %fe{k}_{e} = vector.extract %fa{k}[{e}] : {VA} -> f32
    %fr{k}_{e}, %fu{k}_{e} = kernel.subgroup.shuffle<xor> %fe{k}_{e}, %fold_d, %fold_w : f32, i32, i32
    %fs{k}_{e} = scalar.addf %fe{k}_{e}, %fr{k}_{e} : f32""")
                el.append(f"%fs{k}_{e}")
            L.append(f"""    %fv{k} = vector.from_elements {", ".join(el)} : {VA}
    kernel.barrier<workgroup> scope(subgroup) ordering(acq_rel)
    scf.if %lead_group {{
      vector.store %fv{k}, %acc_rows[%wave, %vdim] : {VA}, view<{G}x{D}xf32>
    }}
    kernel.barrier<workgroup> scope(subgroup) ordering(acq_rel)
    %fd{k} = vector.load %acc_out[%wave, %lane, %c0] : view<{G}x32x{DL}xf32> -> {V}""")
    fa = {k: (f"%fd{k}" if KS > 1 else f"%fa{k}") for k in range(TS)}
    L.append(f"""    %acc_view = buffer.view %part_acc[%zero] : buffer -> view<[%tokens]x{QH}x{S}x32x{DL}xf32>
    %ml_view = buffer.view %part_ml[%zero] : buffer -> view<[%tokens]x{QH}x{S}x2xf32>
    %first_lane = index.cmp eq, %lane, %c0 : index""")
    for k in range(TS):
        L.append(f"""    scf.if %active_{k} {{
      vector.store {fa[k]}, %acc_view[%tok{k}, %bounded_head, %split, %lane, %c0] : {V}, view<[%tokens]x{QH}x{S}x32x{DL}xf32>
      scf.if %first_lane {{
        view.store %fm{k}, %ml_view[%tok{k}, %bounded_head, %split, 0] : f32, view<[%tokens]x{QH}x{S}x2xf32>
        view.store %fl{k}, %ml_view[%tok{k}, %bounded_head, %split, 1] : f32, view<[%tokens]x{QH}x{S}x2xf32>
      }}
    }}""")
    text = "\n".join(L[1:])   # the split kernel's own preamble (ids, constants) precedes it
    # names without an op of their own: token 0 is always a member; a token's range is [start, stop)
    text = re.sub(r"    %active_0 = scalar\.andi %member0, %in_split_0 : i1\n", "", text).replace("%active_0", "%in_split_0")
    text = re.sub(r"%lo_(\d+)\b", r"%start_\1", text)
    text = re.sub(r"%hi_(\d+)\b", r"%stop_\1", text)
    return text.replace("%lo_u", lo).replace("%hi_u", hi)


SEG_LEAD = """
// Does token %t (live, at position >= %min) lead a segment? It starts a run of consecutive same-row tokens, or sits
// {TS} or more tokens into one (such tokens go alone).
func.def inline @g_seg_leader(%info: buffer, %tokens: index, %t: index, %min: i32) -> (i1) {
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c_ts = index.constant {TS} : index
  %run = scf.for %k = [%c0 to %c_ts step %c1](%r = %c0 : index) -> (index) {
    %back = index.sub %t, %k : index
    %ok = index.cmp ugt, %t, %k : index
    %r_eq = index.cmp eq, %r, %k : index
    %go = scalar.andi %ok, %r_eq : i1
    %cont = scf.if %go -> (i1) {
      %c = func.call @g_seg_next(%info, %tokens, %back, %min) : (buffer, index, index, i32) -> (i1)
      scf.yield %c : i1
    } else {
      %no = scalar.constant false : i1
      scf.yield %no : i1
    }
    %r1 = index.add %r, %c1 : index
    %r_next = scf.select %cont, %r1, %r : index
    scf.yield %r_next : index
  }
  %starts = index.cmp eq, %run, %c0 : index
  %deep = index.cmp uge, %run, %c_ts : index
  %lead = scalar.ori %starts, %deep : i1
  func.return %lead : i1
}
"""
SEG_NEXT = """
// Token u continues token u - 1: same row, the next position, both live, u - 1 at position >= %min (u in [1, tokens)).
func.def inline @g_seg_next(%info: buffer, %tokens: index, %u: index, %min: i32) -> (i1) {
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %inside = index.cmp ult, %u, %tokens : index
  %after = index.cmp ugt, %u, %c0 : index
  %in_range = scalar.andi %inside, %after : i1
  %result = scf.if %in_range -> (i1) {
    %iv = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
    %ub = index.assume %u [lt(%u, %tokens)] : index
    %pu = index.sub %ub, %c1 : index
    %pb = index.assume %pu [lt(%pu, %tokens)] : index
    %row_u = view.load %iv[%ub, 0] : view<[%tokens]x4xi32> -> i32
    %row_p = view.load %iv[%pb, 0] : view<[%tokens]x4xi32> -> i32
    %pos_u = view.load %iv[%ub, 1] : view<[%tokens]x4xi32> -> i32
    %pos_p = view.load %iv[%pb, 1] : view<[%tokens]x4xi32> -> i32
    %live_u = view.load %iv[%ub, 2] : view<[%tokens]x4xi32> -> i32
    %live_p = view.load %iv[%pb, 2] : view<[%tokens]x4xi32> -> i32
    %pos_next = scalar.addi %pos_p, %c1_i32 : i32
    %same_row = scalar.cmpi eq, %row_u, %row_p : i32
    %next_pos = scalar.cmpi eq, %pos_u, %pos_next : i32
    %lu = scalar.cmpi ne, %live_u, %c0_i32 : i32
    %lp = scalar.cmpi ne, %live_p, %c0_i32 : i32
    %c_a = scalar.andi %same_row, %next_pos : i1
    %c_b = scalar.andi %lu, %lp : i1
    %far_p = scalar.cmpi uge, %pos_p, %min : i32
    %c_c = scalar.andi %c_b, %far_p : i1
    %cont = scalar.andi %c_a, %c_c : i1
    scf.yield %cont : i1
  } else {
    %no = scalar.constant false : i1
    scf.yield %no : i1
  }
  func.return %result : i1
}
"""
def seg_attention(text):
    """The generated module with the long-context decode share moved into g_attention_split: on a full-attention
    layer a token at position >= SEG_MIN that leads a segment (g_seg_leader) runs the span body for all the
    segment's tokens (writing their partials), the other tokens of a segment exit, and every other token takes the
    per-token path. One launch as before: no leader list, no extra launch per layer."""
    a = text.index('kernel.def export("g_attention_split")')
    old = "    scf.if %active_split {\n"
    b = text.index(old, a)
    body = attention_seg(text)
    names = set()   # the span body's own SSA names, renamed so they never shadow the split kernel's
    for line in body.split("\n"):
        m = re.match(r"\s*((?:%\w+, )*%\w+) = ", line)
        if m:
            names.update(m.group(1).split(", "))
        if "scf.for" in line:
            names.update(re.findall(r"(?:scf\.for |[(,] ?)(%\w+) = ", line))
    for n in sorted(names, key=len, reverse=True):
        body = re.sub(rf"(?<![\w%]){re.escape(n)}\b", "%sq_" + n[1:], body)
    wg = [l for l in body.split("\n") if re.match(r"\s*%sq_(q_bytes|p_bytes|q_tile|p_tile) = ", l)]   # workgroup tiles: kernel scope
    body = "\n".join(l for l in body.split("\n") if l not in wg)
    top = text.index("  scf.if %live {\n", a)
    text = text[:top] + "\n".join(wg) + "\n" + text[top:b] + f"""    %seg_min = index.constant {SEG_MIN} : index
    %seg_far = index.cmp uge, %position, %seg_min : index
    %seg_full = scalar.cmpi eq, %sliding, %c0_i32 : i32
    %seg_dom = scalar.andi %seg_far, %seg_full : i1
    %seg_true = scalar.constant true : i1
    %seg_not = scalar.xori %seg_dom, %seg_true : i1
    %split_here = scalar.andi %active_split, %seg_not : i1
    %seg_min32 = scalar.constant {SEG_MIN} : i32
    %seg_lead = scf.if %seg_dom -> (i1) {{
      %sl = func.call @g_seg_leader(%info, %tokens, %token, %seg_min32) : (buffer, index, index, i32) -> (i1)
      scf.yield %sl : i1
    }} else {{
      %sf = scalar.constant false : i1
      scf.yield %sf : i1
    }}
    scf.if %seg_lead {{
{body}
    }}
    scf.if %split_here {{
""" + text[b + len(old):]
    return text + SEG_NEXT + SEG_LEAD.replace("{TS}", str(seg_tokens()))


def attention():
    """Decode attention with up to 64 splits; the prefill fallback (listed,
    indexed by token over up to 1024 rows) keeps 16 splits."""
    dec, _, _ = attention_text(64, "g")
    pre, split_l, combine_l = attention_text(16, "g16")
    count16 = pre[pre.index("// Splits for"):pre.index('kernel.def export("g16_attention_split")')]
    return dec + "\n" + count16 + split_l + "\n" + combine_l + FALLBACK


FALLBACK = r"""
// Prefill attention work lists, built once per epoch from the token info. A token's path depends only on its own
// span (a run of one row's consecutive positions), never on where the epoch packs it: tiles[1 + i] is the first
// token of the i-th 16-token tile that lies inside one span and starts at a position multiple of 16, in token order
// (neighbouring workgroups then read neighbouring keys; token 0 past the count): g_attention_flash2 computes those,
// and tiles[0] is their count. list holds every other token (g16_attention_split_listed computes those; count[0] is
// the list length for indirect dispatch).
kernel.def export("g_attention_fallback_list") @g_attention_fallback_list(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %info: buffer, %tiles: buffer, %list: buffer, %count: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 16, 8192), multiple_of(%tokens_arg, 16)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c255 = index.constant 255 : index
  %c256 = index.constant 256 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %c15_i32 = scalar.constant 15 : i32
  %true = scalar.constant true : i1
  %scratch_bytes = index.constant 1040 : offset
  %scratch = buffer.alloca<workgroup> align(16) %scratch_bytes : buffer
  %scratch_view = buffer.view %scratch[%zero] : buffer -> view<260xi32>
  %first = index.cmp eq, %lane, %c0 : index
  %c256_slot = index.constant 256 : index
  %c257_slot = index.constant 257 : index
  scf.if %first {
    view.store %c0_i32, %scratch_view[%c256_slot] : i32, view<260xi32>
  }
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %tiles_view = buffer.view %tiles[%zero] : buffer -> view<[%tokens]xi32>
  %list_view = buffer.view %list[%zero] : buffer -> view<[%tokens]xi32>
  // Tile starts, in token order: lane L takes the L-th run of `chunk` tokens, counts its tile starts, the first lane
  // turns the counts into offsets, and every lane writes its starts after those of the lanes before it.
  %chunk_round = index.add %tokens, %c255 : index
  %chunk = index.div %chunk_round, %c256 : index
  %run = index.mul %lane, %chunk : index
  %mine = scf.for %i = [%c0 to %chunk step %c1](%n = %c0_i32 : i32) -> (i32) {
    %token_value = index.add %run, %i : index
    %inside = index.cmp ult, %token_value, %tokens : index
    %token = scf.select %inside, %token_value, %c0 : index
    %starts = func.call @g_tile_starts(%info, %tokens, %token) : (buffer, index, index) -> (i1)
    %counted = scalar.andi %starts, %inside : i1
    %step = scf.select %counted, %c1_i32, %c0_i32 : i32
    %next = scalar.addi %n, %step : i32
    scf.yield %next : i32
  }
  view.store %mine, %scratch_view[%lane] : i32, view<260xi32>
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  scf.if %first {
    %total = scf.for %l = [%c0 to %c256 step %c1](%sum = %c0_i32 : i32) -> (i32) {
      %bounded_l = index.assume %l [range(%l, 0, 255)] : index
      %count_l = view.load %scratch_view[%bounded_l] : view<260xi32> -> i32
      view.store %sum, %scratch_view[%bounded_l] : i32, view<260xi32>
      %sum_next = scalar.addi %sum, %count_l : i32
      scf.yield %sum_next : i32
    }
    view.store %total, %tiles_view[%c0] : i32, view<[%tokens]xi32>
    view.store %total, %scratch_view[%c257_slot] : i32, view<260xi32>
  }
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %offset = view.load %scratch_view[%lane] : view<260xi32> -> i32
  %written = scf.for %i = [%c0 to %chunk step %c1](%n = %offset : i32) -> (i32) {
    %token_value = index.add %run, %i : index
    %inside = index.cmp ult, %token_value, %tokens : index
    %token = scf.select %inside, %token_value, %c0 : index
    %starts = func.call @g_tile_starts(%info, %tokens, %token) : (buffer, index, index) -> (i1)
    %counted = scalar.andi %starts, %inside : i1
    %next = scf.if %counted -> (i32) {
      %n_bound = scalar.assume %n [range(%n, 0, 511)] : i32
      %slot_value = index.cast %n_bound : i32 to index
      %entry_value = index.add %slot_value, %c1 : index
      %entry = index.assume %entry_value [lt(%entry_value, %tokens)] : index
      %token32 = index.cast %token : index to i32
      view.store %token32, %tiles_view[%entry] : i32, view<[%tokens]xi32>
      %n_next = scalar.addi %n, %c1_i32 : i32
      scf.yield %n_next : i32
    } else {
      scf.yield %n : i32
    }
    scf.yield %next : i32
  }
  // The entries past the count are token 0, so a workgroup without a tile can read its entry.
  %listed_total32 = view.load %scratch_view[%c257_slot] : view<260xi32> -> i32
  %listed_total = index.cast %listed_total32 : i32 to index
  %c16 = index.constant 16 : index
  %slots = index.div %tokens, %c16 : index
  %fill_from = index.add %listed_total, %lane : index
  scf.for %e = [%fill_from to %slots step %c256] {
    %e_bound = index.assume %e [range(%e, 0, 8191)] : index
    %fill_value = index.add %e_bound, %c1 : index
    %fill_slot = index.assume %fill_value [range(%fill_value, 1, 8192), lt(%fill_value, %tokens)] : index
    view.store %c0_i32, %tiles_view[%fill_slot] : i32, view<[%tokens]xi32>
  }
  // Every other token: the listed fallback's (order does not matter, each is computed on its own).
  scf.for %t = [%lane to %tokens step %c256] {
    %token = index.assume %t [lt(%t, %tokens)] : index
    %pos = view.load %info_view[%token, %c1] : view<[%tokens]x4xi32> -> i32
    %skew32 = scalar.andi %pos, %c15_i32 : i32
    %skew = index.cast %skew32 : i32 to index
    %has_start = index.cmp ule, %skew, %token : index
    %start_value = index.sub %token, %skew : index
    %start_or_token = scf.select %has_start, %start_value, %token : index
    %start = index.assume %start_or_token [lt(%start_or_token, %tokens)] : index
    %uniform, %r_unused, %p_unused = func.call @g_tile_uniform(%info, %tokens, %start) : (buffer, index, index) -> (i1, i32, i32)
    %covered = scalar.andi %uniform, %has_start : i1
    %not_covered = scalar.xori %covered, %true : i1
    scf.if %not_covered {
      %slot32 = view.atomic.rmw<addi> %c1_i32, %scratch_view[%c256_slot] {ordering = relaxed, scope = workgroup} : i32, view<260xi32> -> i32
      %slot_bound = scalar.assume %slot32 [range(%slot32, 0, 8191)] : i32
      %slot_value = index.cast %slot_bound : i32 to index
      %slot = index.assume %slot_value [lt(%slot_value, %tokens)] : index
      %token32 = index.cast %token : index to i32
      view.store %token32, %list_view[%slot] : i32, view<[%tokens]xi32>
    }
  }
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  scf.if %first {
    %fallbacks = view.load %scratch_view[%c256_slot] : view<260xi32> -> i32
    %count_view = buffer.view %count[%zero] : buffer -> view<1xi32>
    view.store %fallbacks, %count_view[0] : i32, view<1xi32>
  }
  kernel.return
}
"""


CHUNKS = (V + 127) // 128


def eos_text():
    ids = CFG.get("extra_eos_ids", [])
    t = ["        %terminal_0 = scalar.constant false : i1"]
    for i, e in enumerate(ids):
        t.append(f"        %eid{i} = scalar.constant {e} : i32\n        %eis{i} = scalar.cmpi eq, %winner, %eid{i} : i32\n        %terminal_{i + 1} = scalar.ori %terminal_{i}, %eis{i} : i1")
    t.append(f"        %token = scf.select %terminal_{len(ids)}, %eos, %winner : i32")
    return "\n".join(t)


def head():
    """Token selection per selected span: greedy argmax over the per-chunk
    (max logit, argmax) partials when the span's temperature is 0; otherwise
    top-k (k <= 64) candidates from the logits, temperature softmax, top-p,
    and a counter-based draw keyed by (seed, position). Extra end-of-turn ids
    map to the service terminal. Sampling parameters ride in span fields
    2 (temperature f32), 5 (top_p f32), 7 (top_k), 8 (seed) of the metadata."""
    return f"""
kernel.def export("g_head_select") @g_head_select(%spans: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%spans, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%spans_arg: index, %metadata: buffer, %partials: buffer, %logits: buffer, %outputs: buffer) {{
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %span_id = kernel.workgroup.id<x> : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c16 = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %chunks = index.constant {CHUNKS} : index
  %vocab = index.constant {V} : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %zero_f = scalar.constant 0.0 : f32
  %neg_inf = scalar.constant -3.0e38 : f32
  %pos_inf = scalar.constant 3.0e38 : f32
  %big = scalar.constant 2147483647 : i32
  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %count_bound = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count = index.cast %count_bound : i32 to index
  %live_span = index.cmp ult, %span, %count : index
  %cand_bytes = index.constant 512 : offset
  %cand = buffer.alloca<workgroup> align(16) %cand_bytes : buffer
  %cand_v = buffer.view %cand[%zero] : buffer -> view<64xf32>
  %c256o = index.constant 256 : offset
  %cand_i = buffer.view %cand[%c256o] : buffer -> view<64xi32>
  scf.if %live_span {{
    %packets = buffer.view %metadata[%c16] : buffer -> view<[%spans]x9xi32>
    %flags = view.load %packets[%span, 6] : view<[%spans]x9xi32> -> i32
    %bits = scalar.andi %flags, %c1_i32 : i32
    %selected = scalar.cmpi ne, %bits, %c0_i32 : i32
    scf.if %selected {{
      %temperature_bits = view.load %packets[%span, 2] : view<[%spans]x9xi32> -> i32
      %temperature = scalar.bitcast %temperature_bits : i32 to f32
      %greedy = scalar.cmpf ole, %temperature, %zero_f : f32
      %first = index.cmp eq, %lane, %c0 : index
      %chosen_token = scf.if %greedy -> (i32) {{
        %partial_view = buffer.view %partials[%zero] : buffer -> view<[%spans]x{CHUNKS}x2xi32>
        %bv, %bi = scf.for %c = [%lane to %chunks step %c256](%v = %neg_inf : f32, %i = %big : i32) -> (f32, i32) {{
          %bounded_c = index.assume %c [range(%c, 0, {CHUNKS - 1})] : index
          %bits_value = view.load %partial_view[%span, %bounded_c, %c0] : view<[%spans]x{CHUNKS}x2xi32> -> i32
          %id = view.load %partial_view[%span, %bounded_c, %c1] : view<[%spans]x{CHUNKS}x2xi32> -> i32
          %value = scalar.bitcast %bits_value : i32 to f32
          %better = scalar.cmpf ogt, %value, %v : f32
          %nv = scf.select %better, %value, %v : f32
          %ni = scf.select %better, %id, %i : i32
          scf.yield %nv, %ni : f32, i32
        }}
        %best = kernel.workgroup.reduce<maxnumf> %bv : f32
        %is_best = scalar.cmpf oeq, %bv, %best : f32
        %candidate = scf.select %is_best, %bi, %big : i32
        %winner = kernel.workgroup.reduce<minsi> %candidate : i32
        scf.yield %winner : i32
      }} else {{
        %top_p_bits = view.load %packets[%span, 5] : view<[%spans]x9xi32> -> i32
        %top_p_raw = scalar.bitcast %top_p_bits : i32 to f32
        %top_k_raw = view.load %packets[%span, 7] : view<[%spans]x9xi32> -> i32
        %seed = view.load %packets[%span, 8] : view<[%spans]x9xi32> -> i32
        %c64_i32 = scalar.constant 64 : i32
        %k_set = scalar.cmpi sgt, %top_k_raw, %c0_i32 : i32
        %k_small = scalar.minsi %top_k_raw, %c64_i32 : i32
        %k32 = scf.select %k_set, %k_small, %c64_i32 : i32
        %k_bound = scalar.assume %k32 [range(%k32, 1, 64)] : i32
        %k = index.cast %k_bound : i32 to index
        %logit_view = buffer.view %logits[%zero] : buffer -> view<[%spans]x{CHUNKS * 128}xf32>
        // k rounds: the largest logit strictly after the previous pick in (value desc, index asc) order.
        %last_v, %last_i = scf.for %r = [%c0 to %k step %c1](%pv = %pos_inf : f32, %pi = %c0_i32 : i32) -> (f32, i32) {{
          %minus_one = scalar.constant -1 : i32
          %prev_i = scalar.addi %pi, %c0_i32 : i32
          %first_round = index.cmp eq, %r, %c0 : index
          %floor_i = scf.select %first_round, %minus_one, %prev_i : i32
          %lv, %li = scf.for %v = [%lane to %vocab step %c256](%bv = %neg_inf : f32, %bi = %big : i32) -> (f32, i32) {{
            %bounded_v = index.assume %v [range(%v, 0, {V - 1})] : index
            %x = view.load %logit_view[%span, %bounded_v] : view<[%spans]x{CHUNKS * 128}xf32> -> f32
            %v32 = index.cast %bounded_v : index to i32
            %below = scalar.cmpf olt, %x, %pv : f32
            %tie = scalar.cmpf oeq, %x, %pv : f32
            %after = scalar.cmpi sgt, %v32, %floor_i : i32
            %tie_after = scalar.andi %tie, %after : i1
            %eligible = scalar.ori %below, %tie_after : i1
            %greater = scalar.cmpf ogt, %x, %bv : f32
            %take = scalar.andi %eligible, %greater : i1
            %nv = scf.select %take, %x, %bv : f32
            %ni = scf.select %take, %v32, %bi : i32
            scf.yield %nv, %ni : f32, i32
          }}
          %best = kernel.workgroup.reduce<maxnumf> %lv : f32
          %is_best = scalar.cmpf oeq, %lv, %best : f32
          %cand_id = scf.select %is_best, %li, %big : i32
          %winner = kernel.workgroup.reduce<minsi> %cand_id : i32
          scf.if %first {{
            %rr = index.assume %r [range(%r, 0, 63)] : index
{SOFTCAP_STORE}
            view.store %winner, %cand_i[%rr] : i32, view<64xi32>
          }}
          scf.yield %best, %winner : f32, i32
        }}
        kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
        // One lane: softmax(l / T) over the k candidates, keep the top-p prefix, draw.
        %one_f = scalar.constant 1.0 : f32
        %p_ok = scalar.cmpf ogt, %top_p_raw, %zero_f : f32
        %p_low = scalar.minnumf %top_p_raw, %one_f : f32
        %top_p = scf.select %p_ok, %p_low, %one_f : f32
        %inv_t = scalar.divf %one_f, %temperature : f32
        %m = view.load %cand_v[0] : view<64xf32> -> f32
        %total = scf.for %r = [%c0 to %k step %c1](%acc = %zero_f : f32) -> (f32) {{
          %rr = index.assume %r [range(%r, 0, 63)] : index
          %l = view.load %cand_v[%rr] : view<64xf32> -> f32
          %d = scalar.subf %l, %m : f32
          %ds = scalar.mulf %d, %inv_t : f32
          %w = scalar.expf<afn> %ds : f32
          %next = scalar.addf %acc, %w : f32
          scf.yield %next : f32
        }}
        %mass = scalar.mulf %top_p, %total : f32
        // Candidates kept: the shortest prefix whose weight reaches top_p of the total.
        %kept_total, %kept_done = scf.for %r = [%c0 to %k step %c1](%acc = %zero_f : f32, %done = %greedy : i1) -> (f32, i1) {{
          %rr = index.assume %r [range(%r, 0, 63)] : index
          %l = view.load %cand_v[%rr] : view<64xf32> -> f32
          %d = scalar.subf %l, %m : f32
          %ds = scalar.mulf %d, %inv_t : f32
          %w = scalar.expf<afn> %ds : f32
          %added = scalar.addf %acc, %w : f32
          %next = scf.select %done, %acc, %added : f32
          %reached = scalar.cmpf oge, %next, %mass : f32
          %now_done = scalar.ori %done, %reached : i1
          scf.yield %next, %now_done : f32, i1
        }}
        // Counter-based uniform in [0, 1) from (seed, position of the drawn token).
        %start32 = view.load %packets[%span, 1] : view<[%spans]x9xi32> -> i32
        %length32 = view.load %packets[%span, 0] : view<[%spans]x9xi32> -> i32
        %position = scalar.addi %start32, %length32 : i32
        %golden = scalar.constant -1640531527 : i32
        %mix_a = scalar.constant -2048144789 : i32
        %mix_b = scalar.constant 2146121005 : i32
        %mix_c = scalar.constant -2073254261 : i32
        %c15_i32 = scalar.constant 15 : i32
        %c16_i32 = scalar.constant 16 : i32
        %c8_i32 = scalar.constant 8 : i32
        %h0 = scalar.muli %seed, %golden : i32
        %p0 = scalar.muli %position, %mix_a : i32
        %h1 = scalar.xori %h0, %p0 : i32
        %s1 = scalar.shrui %h1, %c16_i32 : i32
        %h2 = scalar.xori %h1, %s1 : i32
        %h3 = scalar.muli %h2, %mix_b : i32
        %s2 = scalar.shrui %h3, %c15_i32 : i32
        %h4 = scalar.xori %h3, %s2 : i32
        %h5 = scalar.muli %h4, %mix_c : i32
        %s3 = scalar.shrui %h5, %c16_i32 : i32
        %h6 = scalar.xori %h5, %s3 : i32
        %u24 = scalar.shrui %h6, %c8_i32 : i32
        %uf = scalar.uitofp %u24 : i32 to f32
        %scale24 = scalar.constant 5.9604644775390625e-08 : f32
        %u = scalar.mulf %uf, %scale24 : f32
        %target = scalar.mulf %u, %kept_total : f32
        %chosen, %picked = scf.for %r = [%c0 to %k step %c1](%acc = %zero_f : f32, %pick = %big : i32) -> (f32, i32) {{
          %rr = index.assume %r [range(%r, 0, 63)] : index
          %l = view.load %cand_v[%rr] : view<64xf32> -> f32
          %id = view.load %cand_i[%rr] : view<64xi32> -> i32
          %d = scalar.subf %l, %m : f32
          %ds = scalar.mulf %d, %inv_t : f32
          %w = scalar.expf<afn> %ds : f32
          %next = scalar.addf %acc, %w : f32
          %unset = scalar.cmpi eq, %pick, %big : i32
          %hit = scalar.cmpf ogt, %next, %target : f32
          %take = scalar.andi %unset, %hit : i1
          %np = scf.select %take, %id, %pick : i32
          scf.yield %next, %np : f32, i32
        }}
        %fallback = view.load %cand_i[0] : view<64xi32> -> i32
        %unset_final = scalar.cmpi eq, %picked, %big : i32
        %sampled = scf.select %unset_final, %fallback, %picked : i32
        scf.yield %sampled : i32
      }}
      scf.if %first {{
        // Other end-of-turn ids also end the turn: report the service terminal.
        %winner = scalar.addi %chosen_token, %c0_i32 : i32
        %eos = view.load %header[1] : view<4xi32> -> i32
{eos_text()}
        %result = buffer.view %outputs[%zero] : buffer -> view<[%spans]xi32>
        view.store %token, %result[%span] : i32, view<[%spans]xi32>
      }}
    }}
  }}
  kernel.return
}}
"""


from gen_gemm import HELPER as GEMM_HELPER   # noqa: E402  formats without a 128/256-row kernel of their own
GEMM_M256 = os.environ.get("GEMM_M256", "1") not in ("", "0", "false", "no")


def linear(fmt, x, s, w, wsize, y, k, n, kval, mode="store", gemm_only=False, width=None):
    """Projection y = x W^T: WMMA GEMM for 64+ tokens (128-row tiles from 128),
    else the decode matrix-vector kernel (1-token or 4-token variant). n: the output width, a literal or an NVAL
    symbol; a symbol another generator defines (the drafters' %nDQ, %nE, ...) comes with its value as width."""
    base = f"g_mv_{fmt}_k{kval}" + ("_swiglu" if mode == "swiglu" else "")
    if mode == "swiglu":
        wa, wb = w
        args = f"%t, {k}, {n}, {x}, {s}, {wa}, {wb}, {y}"
        types = f"index, index, index, buffer, buffer, view<{wsize}xi8>, view<{wsize}xi8>, buffer"
        mv_args = f"({n}, %t, %zero_i32, {x}, {wa}, {wb}, {y}) : [index, index](index, index, i32, buffer, view<{wsize}xi8>, view<{wsize}xi8>, buffer)"
    else:
        args = f"%t, {k}, {n}, {x}, {s}, {w}, {y}"
        types = f"index, index, index, buffer, buffer, view<{wsize}xi8>, buffer"
        mv_args = f"({n}, %t, %zero_i32, {x}, {w}, {y}) : [index, index](index, index, i32, buffer, view<{wsize}xi8>, buffer)"
    x16 = {"%h": "%h16", "%o": "%o16", "%f": "%f16"}.get(x, x + "16")   # f16 copy of the activation (GEMM input)
    args = args.replace(f", {x}, {s},", f", {x16}, {s},")
    iq = ACT_Q8 and fmt in IQ_FMTS and kval // 32 <= 1024   # wider K (mv_wide: DFlash feature projections) stays f32
    if iq:   # int8 copy (g_quant_q8, launched beside the g_row_scale_f16 that makes x16): see quant_wiring()
        mv_args = mv_args.replace(f", %zero_i32, {x},", f", %zero_i32, {x16}q,")
    m128 = f"kernel.launch @g_gemm_{fmt}_{mode}_m128[%t, {n}]({args}) : [index, index]({types})"
    if GEMM_M256 and fmt not in GEMM_HELPER:   # 256-row tiles: half the weight dequant per MMA (gen_gemm.gemm_big256)
        m128 = f"""scf.if %huge {{
          kernel.launch @g_gemm_{fmt}_{mode}_m256[%t, {n}]({args}) : [index, index]({types})
        }} else {{
          {m128}
        }}"""
    gemm = f"""scf.if %big {{
        {m128}
      }} else {{
        kernel.launch @g_gemm_{fmt}_{mode}[%t, {n}]({args}) : [index, index]({types})
      }}"""
    # The 128/256-row kernels tile the outputs by 128 (swiglu: 64 of each weight) and compute only whole tiles: an
    # output width they do not divide (gpt-oss: 2880) keeps the 64-row kernel, or the last columns are never written.
    nv = width if width is not None else int(n) if str(n).isdigit() else NVAL.get(n)
    assert nv, f"linear(): output width {n} is neither a literal nor an NVAL symbol: pass width="
    if nv % (64 if mode == "swiglu" else 128):
        gemm = f"kernel.launch @g_gemm_{fmt}_{mode}[%t, {n}]({args}) : [index, index]({types})"
    if gemm_only:
        return gemm
    return f"""scf.if {"%c_false" if "lin" in DIAG else "%gemm"} {{
      {gemm}
    }} else {{
      scf.if %single {{
        kernel.launch @{mv_name(fmt, kval, mode == "swiglu", n, 1, iq)}[{n}, %t]{mv_args}
      }} else {{
        {mv_wide_tokens(fmt, kval, mode == "swiglu", n, mv_args) if iq else
         f"kernel.launch @{mv_name(fmt, kval, mode == 'swiglu', n, 4)}[{n}, %t]{mv_args}"}
      }}
    }}"""


_LIN = [0]


def mv_wide_tokens(fmt, kval, sw, n, mv_args):
    """int8-activation matvec for 2+ tokens: the 4-token kernel up to 4 tokens, the 8-token one up to 8, else the
    16-token one (one weight pass per 16 tokens: 4 drafting streams at depth 3). Every variant rounds each token
    exactly as the 1-token kernel does."""
    _LIN[0] += 1
    k = _LIN[0]
    return (f"""%c4_l{k} = index.constant 4 : index
        %le4_l{k} = index.cmp ule, %t, {"%c0" if os.environ.get("FORCE16") else f"%c4_l{k}"} : index
        %c8_l{k} = index.constant 8 : index
        %le8_l{k} = index.cmp ule, %t, {"%c0" if os.environ.get("FORCE16") else f"%c8_l{k}"} : index
        scf.if %le4_l{k} {{
          kernel.launch @{mv_name(fmt, kval, sw, n, 4, True)}[{n}, %t]{mv_args}
        }} else {{
          scf.if %le8_l{k} {{
            kernel.launch @{mv_name(fmt, kval, sw, n, 8, True)}[{n}, %t]{mv_args}
          }} else {{
            kernel.launch @{mv_name(fmt, kval, sw, n, 16, True)}[{n}, %t]{mv_args}
          }}
        }}""")

def head_mv(TG, fmt, iq=False, suffix=""):
    """Output head argmax: workgroup = 128 vocabulary rows x TG spans,
    one lane per 32-value group of the 6656-wide row with each span's
    activations in registers; weights stream once per group of TG spans.
    suffix names a variant over another format (the Q4_0 proposal head, PROPOSE_HEAD)."""
    R = 128
    groups = HG
    lanes = (groups + 31) // 32 * 32
    waves = lanes // 32
    rb = row_bytes(fmt, H)
    name = "g_head_mv" + (f"_t{TG}" if TG > 1 else "") + ("_q8" if iq else "") + suffix
    L = [f"""
kernel.def export("{name}") @{name}(%spans: index) {{
  %c1 = index.constant 1 : index
  %lanes = index.constant {lanes} : index
  %chunks = index.constant {CHUNKS} : index
  %tg = index.constant {TG} : index
  %tg_less = index.constant {TG - 1} : index
  %padded = index.add %spans, %tg_less : index
  %span_groups = index.div %padded, %tg : index
  kernel.launch.config workgroups(%span_groups, %chunks, %c1) workgroup_size(%lanes, %c1, %c1) : index
}} launch(%spans_arg: index, %tokens_arg: index, %metadata: buffer, %h: buffer, %w: buffer, %partials: buffer, %logits: buffer) {{
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %chunk_id = kernel.workgroup.id<y> : index
  %chunk = index.assume %chunk_id [range(%chunk_id, 0, {CHUNKS - 1})] : index
  %sgroup = kernel.workgroup.id<x> : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, {lanes - 1})] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, {waves - 1})] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c16o = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c8 = index.constant 8 : index
  %tgc = index.constant {TG} : index
  %cW = index.constant {waves} : index
  %rows = index.constant {R} : index
  %groups = index.constant {groups} : index
  %vocab = index.constant {V} : index
  %row_bytes = index.constant {rb} : offset
  %block_bytes = index.constant {FMT[fmt]} : offset
  %zero_f = scalar.constant 0.0 : f32
  %zero32 = vector.constant 0.0 : vector<32xf32>
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %false = scalar.constant false : i1
  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %count_bound = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count = index.cast %count_bound : i32 to index
  %packets = buffer.view %metadata[%c16o] : buffer -> view<[%spans]x9xi32>
""" + (f"""
  %x_view = buffer.view %h[%zero] : buffer -> view<[%tokens]x{groups}x32xf32>""" if not iq else f"""
  %xq_view = buffer.view %h[%zero] : buffer -> view<[%tokens]x{groups}x8xi32>
  %xm_k = index.constant {H} : offset
  %xm_off = index.scale %tokens, %xm_k : index, offset -> offset
  %xm_view = buffer.view %h[%xm_off] : buffer -> view<[%tokens]x{groups}x2xf32>
  %zero_i = scalar.constant 0 : i32
  %zero8i = vector.constant 0 : vector<8xi32>
  %zero32i8 = vector.constant 0 : vector<32xi8>
  %zero1i = vector.constant 0 : vector<1xi32>""") + f"""
  %span0 = index.mul %sgroup, %tgc : index
  %ok = index.cmp ult, %workitem, %groups : index
  %g_last = index.constant {groups - 1} : index
  %g_c = index.min %workitem, %g_last : index
  %g = index.assume %g_c [range(%g_c, 0, {groups - 1})] : index
  %cgpb = index.constant {SB[fmt] // 32} : index
  %blk = index.div %g, %cgpb : index
  %within = index.rem %g, %cgpb : index
  %ba = index.scale %blk, %block_bytes : index, offset -> offset
  %first_row = index.mul %chunk, %rows : index"""]
    # BFLY_PERM for the 16-span int8 head: slot j holds span j ^ %bt_idx (its token and liveness looked up per lane,
    # once per workgroup), so the butterfly needs no selects (see bfly_slot); the tree is unchanged. Muse-Glimmer
    # 202048 x 6656 Q6_K, 12 and 16 spans: 9.0 -> 8.5 ms; 8 spans flat. HEAD_BFLY_PERM=0 turns it off.
    HP = iq and BFLY_PERM and TG >= 16 and os.environ.get("HEAD_BFLY_PERM", "1") == "1"
    if HP:
        L.append(bfly_setup(TG))

    def head_perm_span(j):
        return "\n" + bfly_slot(j, TG, "%span0", f"spanp{j}", True) + f"""
  %in_sp{j} = index.cmp ult, %spanp{j}, %spans : index
  %in_cp{j} = index.cmp ult, %spanp{j}, %count : index
  %inp{j} = scalar.andi %in_sp{j}, %in_cp{j} : i1
  %livep{j}, %tokenp{j} = scf.if %inp{j} -> (i1, index) {{
    %sp = index.assume %spanp{j} [lt(%spanp{j}, %spans)] : index
    %flags = view.load %packets[%sp, 6] : view<[%spans]x9xi32> -> i32
    %bits = scalar.andi %flags, %c1_i32 : i32
    %selected = scalar.cmpi ne, %bits, %c0_i32 : i32
    %length32 = view.load %packets[%sp, 0] : view<[%spans]x9xi32> -> i32
    %input32 = view.load %packets[%sp, 4] : view<[%spans]x9xi32> -> i32
    %last32 = scalar.addi %input32, %length32 : i32
    %tok32 = scalar.subi %last32, %c1_i32 : i32
    %tok_low = scalar.maxsi %tok32, %c0_i32 : i32
    %tok_bound = scalar.assume %tok_low [range(%tok_low, 0, 8191)] : i32
    %tok_value = index.cast %tok_bound : i32 to index
    %one_l = index.constant 1 : index
    %tokens_last = index.sub %tokens, %one_l : index
    %tok_safe = index.min %tok_value, %tokens_last : index
    scf.yield %selected, %tok_safe : i1, index
  }} else {{
    scf.yield %false, %c0 : i1, index
  }}
  %usep{j} = scalar.andi %livep{j}, %ok : i1"""
    for j in range(TG):
        L.append(f"""  %sj{j} = index.constant {j} : index
  %span{j}_raw = index.add %span0, %sj{j} : index
  %in_s{j} = index.cmp ult, %span{j}_raw, %spans : index
  %in_c{j} = index.cmp ult, %span{j}_raw, %count : index
  %in{j} = scalar.andi %in_s{j}, %in_c{j} : i1
  %live{j}, %token{j} = scf.if %in{j} -> (i1, index) {{
    %sp = index.assume %span{j}_raw [lt(%span{j}_raw, %spans)] : index
    %flags = view.load %packets[%sp, 6] : view<[%spans]x9xi32> -> i32
    %bits = scalar.andi %flags, %c1_i32 : i32
    %selected = scalar.cmpi ne, %bits, %c0_i32 : i32
    %length32 = view.load %packets[%sp, 0] : view<[%spans]x9xi32> -> i32
    %input32 = view.load %packets[%sp, 4] : view<[%spans]x9xi32> -> i32
    %last32 = scalar.addi %input32, %length32 : i32
    %tok32 = scalar.subi %last32, %c1_i32 : i32
    %tok_low = scalar.maxsi %tok32, %c0_i32 : i32
    %tok_bound = scalar.assume %tok_low [range(%tok_low, 0, 8191)] : i32
    %tok_value = index.cast %tok_bound : i32 to index
    %one_l = index.constant 1 : index
    %tokens_last = index.sub %tokens, %one_l : index
    %tok_safe = index.min %tok_value, %tokens_last : index
    scf.yield %selected, %tok_safe : i1, index
  }} else {{
    scf.yield %false, %c0 : i1, index
  }}
  %use{j} = scalar.andi %live{j}, %ok : i1""" + (head_perm_span(j) if HP else "") + (f"""
  %x{j} = scf.if %use{j} -> (vector<32xf32>) {{
    %tk = index.assume %token{j} [lt(%token{j}, %tokens)] : index
    %v = vector.load %x_view[%tk, %g, %c0] : view<[%tokens]x{groups}x32xf32> -> vector<32xf32>
    scf.yield %v : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}""" if not iq else "\n" + (iq_load(j, f"%usep{j}", f"%tokenp{j}", groups) if HP else iq_load(j, f"%use{j}", f"%token{j}", groups))))
    if iq:
        L.append(head_iq_body(TG, fmt, R, waves, rb, HP))
        return "\n".join(L)
    L.append(f"""  %part_bytes = index.constant {waves * R * TG * 4} : offset
  %parts = buffer.alloca<workgroup> align(16) %part_bytes : buffer
  %pv = buffer.view %parts[%zero] : buffer -> view<{waves}x{R}x{TG}xf32>
  %lane_first = index.cmp eq, %lane, %c0 : index
  scf.for %r = [%c0 to %rows step %c1] {{
    %row_value = index.add %first_row, %r : index
    %in_vocab = index.cmp ult, %row_value, %vocab : index
    %load = scalar.andi %in_vocab, %ok : i1
    %last_row = index.constant {V - 1} : index
    %row_c = index.min %row_value, %last_row : index
    %ro = index.scale %row_c, %row_bytes : index, offset -> offset
    %o = index.add %ro, %ba : offset
    %rb = index.assume %r [range(%r, 0, {R - 1})] : index""" + ("\n" + iq_decode(fmt, 0, "w", "%load", "%o", "%within") if iq else f"""
    %wv = scf.if %load -> (vector<32xf32>) {{
      %v = func.call @g_{fmt}_values(%w, %o, %within) : (buffer, offset, index) -> (vector<32xf32>)
      scf.yield %v : vector<32xf32>
    }} else {{
      scf.yield %zero32 : vector<32xf32>
    }}"""))
    for j in range(TG):
        L.append(f"""    %pa{j} = vector.dotf<reassoc|contract> %wv, %x{j}, %zero_f : vector<32xf32>, vector<32xf32>, f32""" + f"""
    %ps{j} = kernel.subgroup.reduce<addf> %pa{j} : f32
    scf.if %lane_first {{
      view.store %ps{j}, %pv[%wave, %rb, {j}] : f32, view<{waves}x{R}x{TG}xf32>
    }}""")
    L.append("  }")
    L.append("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    L.append(f"  %partial_view = buffer.view %partials[%zero] : buffer -> view<[%spans]x{CHUNKS}x2xi32>")
    L.append(f"  %logit_view = buffer.view %logits[%zero] : buffer -> view<[%spans]x{CHUNKS * 128}xf32>")
    for j in range(TG):
        L.append(f"""  %writer{j} = index.cmp eq, %workitem, %sj{j} : index
  %store{j} = scalar.andi %writer{j}, %live{j} : i1
  scf.if %store{j} {{
    %neg_inf = scalar.constant -3.0e38 : f32
    %best, %best_id = scf.for %r = [%c0 to %rows step %c1](%bv = %neg_inf : f32, %bi = %c0_i32 : i32) -> (f32, i32) {{
      %rb = index.assume %r [range(%r, 0, {R - 1})] : index
      %total = scf.for %ww = [%c0 to %cW step %c1](%acc = %zero_f : f32) -> (f32) {{
        %bw = index.assume %ww [range(%ww, 0, {waves - 1})] : index
        %p = view.load %pv[%bw, %rb, {j}] : view<{waves}x{R}x{TG}xf32> -> f32
        %nx = scalar.addf %acc, %p : f32
        scf.yield %nx : f32
      }}
      %row_value = index.add %first_row, %r : index
      %in_vocab = index.cmp ult, %row_value, %vocab : index
      %ls = index.assume %span{j}_raw [lt(%span{j}_raw, %spans)] : index
      %lrow = index.assume %row_value [range(%row_value, 0, {CHUNKS * 128 - 1})] : index
      %stored = scf.select %in_vocab, %total, %neg_inf : f32
      view.store %stored, %logit_view[%ls, %lrow] : f32, view<[%spans]x{CHUNKS * 128}xf32>
      %greater = scalar.cmpf ogt, %total, %bv : f32
      %better = scalar.andi %greater, %in_vocab : i1
      %row32 = index.cast %row_value : index to i32
      %v = scf.select %better, %total, %bv : f32
      %i = scf.select %better, %row32, %bi : i32
      scf.yield %v, %i : f32, i32
    }}
    %ss = index.assume %span{j}_raw [lt(%span{j}_raw, %spans)] : index
    %bits_value = scalar.bitcast %best : f32 to i32
    view.store %bits_value, %partial_view[%ss, %chunk, %c0] : i32, view<[%spans]x{CHUNKS}x2xi32>
    view.store %best_id, %partial_view[%ss, %chunk, %c1] : i32, view<[%spans]x{CHUNKS}x2xi32>
  }}""")
    L.append("  kernel.return\n}\n")
    return "\n".join(L)



def head_iq_body(TG, fmt, R, waves, rb, perm=False):
    """Row loop and argmax of the int8 head (ACT_Q8): the 128-row chunk runs in sub-blocks of RB rows so the
    workgroup's partials (waves x RB x TG F32) stay ~14 KB at any TG (a 128 x 8-span buffer, 28 KB, held one or two
    workgroups per CU: Muse-Glimmer 202048 x 6656 Q6_K, 8 spans 10.9 ms vs 5.1 for 1 span). Lane j < TG carries span
    j's running (max, argmax) across sub-blocks. Per-span lane partials sum with the int8 matvecs' shared-tree butterfly
    (the same xor-16/8/4/2/1 tree for every TG), then over waves in order, rows ascending: every variant rounds and
    breaks ties exactly as the 1-span kernel does, so verification matches decode."""
    RB = min(R, max(16, 512 // TG))
    NSB = R // RB
    assert NSB * RB == R
    sel_live, sel_span = "%false", "%c0"
    t = ["" if perm else bfly_setup(TG), f"""  %part_bytes = index.constant {waves * RB * TG * 4} : offset
  %parts = buffer.alloca<workgroup> align(16) %part_bytes : buffer
  %pv = buffer.view %parts[%zero] : buffer -> view<{waves}x{RB}x{TG}xf32>
  %rbk = index.constant {RB} : index
  %nsb = index.constant {NSB} : index
  %neg_inf = scalar.constant -3.0e38 : f32"""]
    for j in range(TG):
        t.append(f"""  %wsel{j} = index.cmp eq, %workitem, %sj{j} : index
  %wl{j} = scf.select %wsel{j}, %live{j}, {sel_live} : i1
  %ws{j} = scf.select %wsel{j}, %span{j}_raw, {sel_span} : index""")
        sel_live, sel_span = f"%wl{j}", f"%ws{j}"
    t.append(f"""  %my_live = scalar.andi {sel_live}, %bt_true : i1
  %my_span = index.add {sel_span}, %c0 : index
  %tg_last = index.constant {TG - 1} : index
  %my_tok_c = index.min %workitem, %tg_last : index
  %my_tok = index.assume %my_tok_c [range(%my_tok_c, 0, {TG - 1})] : index
  %partial_view = buffer.view %partials[%zero] : buffer -> view<[%spans]x{CHUNKS}x2xi32>
  %logit_view = buffer.view %logits[%zero] : buffer -> view<[%spans]x{CHUNKS * 128}xf32>
  %best, %best_id = scf.for %sb = [%c0 to %nsb step %c1](%cbv = %neg_inf : f32, %cbi = %c0_i32 : i32) -> (f32, i32) {{
  %sb_row = index.mul %sb, %rbk : index
  %blk_row = index.add %first_row, %sb_row : index
  scf.for %r = [%c0 to %rbk step %c1] {{
    %row_value = index.add %blk_row, %r : index
    %in_vocab = index.cmp ult, %row_value, %vocab : index
    %load = scalar.andi %in_vocab, %ok : i1
    %last_row = index.constant {V - 1} : index
    %row_c = index.min %row_value, %last_row : index
    %ro = index.scale %row_c, %row_bytes : index, offset -> offset
    %o = index.add %ro, %ba : offset
    %rb = index.assume %r [range(%r, 0, {RB - 1})] : index
""" + iq_decode(fmt, 0, "w", "%load", "%o", "%within"))
    t.append("\n".join(iq_dot(fmt, 0, j, f"%pa{j}") for j in range(TG)))
    t.append(bfly_store([f"%pa{j}" for j in range(TG)], "hb",
                        lambda v: f"view.store {v}, %pv[%wave, %rb, %bt_idx] : f32, view<{waves}x{RB}x{TG}xf32>", presorted=perm))
    t.append(f"""  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %nbv, %nbi = scf.if %my_live -> (f32, i32) {{
    %ls = index.assume %my_span [lt(%my_span, %spans)] : index
    %bv2, %bi2 = scf.for %r = [%c0 to %rbk step %c1](%bv = %cbv : f32, %bi = %cbi : i32) -> (f32, i32) {{
      %rb = index.assume %r [range(%r, 0, {RB - 1})] : index
      %total = scf.for %ww = [%c0 to %cW step %c1](%acc = %zero_f : f32) -> (f32) {{
        %bw = index.assume %ww [range(%ww, 0, {waves - 1})] : index
        %p = view.load %pv[%bw, %rb, %my_tok] : view<{waves}x{RB}x{TG}xf32> -> f32
        %nx = scalar.addf %acc, %p : f32
        scf.yield %nx : f32
      }}
      %row_value = index.add %blk_row, %r : index
      %in_vocab = index.cmp ult, %row_value, %vocab : index
      %lrow = index.assume %row_value [range(%row_value, 0, {CHUNKS * 128 - 1})] : index
      %stored = scf.select %in_vocab, %total, %neg_inf : f32
      view.store %stored, %logit_view[%ls, %lrow] : f32, view<[%spans]x{CHUNKS * 128}xf32>
      %greater = scalar.cmpf ogt, %total, %bv : f32
      %better = scalar.andi %greater, %in_vocab : i1
      %row32 = index.cast %row_value : index to i32
      %v = scf.select %better, %total, %bv : f32
      %i = scf.select %better, %row32, %bi : i32
      scf.yield %v, %i : f32, i32
    }}
    scf.yield %bv2, %bi2 : f32, i32
  }} else {{
    scf.yield %cbv, %cbi : f32, i32
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  scf.yield %nbv, %nbi : f32, i32
  }}
  scf.if %my_live {{
    %ss = index.assume %my_span [lt(%my_span, %spans)] : index
    %bits_value = scalar.bitcast %best : f32 to i32
    view.store %bits_value, %partial_view[%ss, %chunk, %c0] : i32, view<[%spans]x{CHUNKS}x2xi32>
    view.store %best_id, %partial_view[%ss, %chunk, %c1] : i32, view<[%spans]x{CHUNKS}x2xi32>
  }}
  kernel.return
}}
""")
    return "\n".join(t)


USED_MV = set()      # (fmt, K, swiglu, TG) default-row (R=16) decode matrix-vector kernels the model launches
USED_MV_R = set()    # (fmt, K, swiglu, R, TG) adaptive-row variants
NVAL = {"%nQH": QH, "%nQKV": (QH + 2 * KVH) * D, "%nH": H, "%nFF": FF, **({} if FF_UNIFORM else {f"%nFF{w}": w for w in set(FFL)}), "%nKV": KVH * D, "%nQ": QH * D, "%nQ2": 2 * QH * D, "%nFS": FS}
NVAL.update({f"%nQ{h}": h * D for h in NARROW} | {f"%nQH{h}": h for h in NARROW})   # narrower layer_heads layers
if GDN:
    NVAL.update({"%nC": GDN["C"], "%nVD": GDN["VD"], "%nHV": GDN["HV"]})
from mv_rows import pick_r, iq_r  # noqa: E402  (shared with make_kernel_tests.py)


def mv_name(fmt, kval, sw, n, TG, iq=False):
    nv = NVAL.get(n, None) if isinstance(n, str) else n
    R = pick_r(nv, kval, TG)
    base = f"g_mv_{fmt}_k{kval}" + {1: "_swiglu", 2: "_pair"}.get(int(sw), "") + (f"_t{TG}" if TG > 1 else "")
    if iq:
        # int8 variants: a row's arithmetic does not depend on R (explicit ops, explicit butterfly), so the 8/16-token
        # kernels take more rows per workgroup (fewer activation reloads; mv_rows.iq_r)
        R = iq_r(nv, kval, TG, sw)
        USED_MVQ.add((fmt, kval, sw, R, TG))
        return base + "_q8" + (f"_r{R}" if R != 16 else "")
    if R == 16:
        USED_MV.add((fmt, kval, sw, TG))
        return base
    USED_MV_R.add((fmt, kval, sw, R, TG))
    return base + f"_r{R}"
USED_GEMM = {}       # fmt -> set of GEMM modes
USED_MVQ = set()     # (fmt, K, swiglu, R, TG) of the int8-activation matvecs (ACT_Q8); swiglu 2 = pair (PAIR_PROJ)


def head_wiring(program):
    """ACT_Q8 with an integer-format head: every `%c = cmp eq spans, 1; scf.if %c { g_head_mv } else { g_head_mv_t4 }`
    dispatch quantizes its hidden rows once (g_quant_q8) and launches the int8 head for 1, 2-4, 5-8 or 9+ spans (one
    head pass per 16 spans: 4 drafting streams verify 16 rows and propose 12 at depth 3; every variant rounds each span
    as the 1-span kernel does, so verify matches decode)."""
    import re as _re
    pat = _re.compile(r"(?P<i>[ ]*)(?P<c>%\w+) = index\.cmp eq, (?P<sp>%\w+), %c1 : index\n"
                      r"\s*scf\.if (?P=c) \{\n\s*kernel\.launch @g_head_mv\[(?P=sp)\]\((?P=sp), (?P<t>%\w+), (?P<m>%\w+), (?P<h>%\w+), (?P<r>[^)]*)\) : (?P<ty>[^\n]*)\n"
                      r"\s*\} else \{\n\s*kernel\.launch @g_head_mv_t4\[(?P=sp)\][^\n]*\n\s*\}")
    n = [0]
    def sub(m):
        n[0] += 1; k = n[0]; i = m.group("i"); sp, t, h = m.group("sp"), m.group("t"), m.group("h")
        def launch(name, ind):
            return f"{ind}kernel.launch @{name}[{sp}]({sp}, {t}, {m.group('m')}, %hq{k}, {m.group('r')}) : {m.group('ty')}"
        return (f"{i}%hq{k}_b = index.constant {2 * H} : offset\n{i}%hq{k}_s = index.scale {t}, %hq{k}_b : index, offset -> offset\n"
                f"{i}%hq{k} = buffer.alloca<global> align(256) %hq{k}_s : buffer\n{i}%hq{k}_w = index.constant {H} : index\n"
                f"{i}kernel.launch @g_quant_q8[{t}, %hq{k}_w]({t}, %hq{k}_w, {h}, %hq{k}) : [index, index](index, index, buffer, buffer)\n"
                f"{i}{m.group('c')} = index.cmp eq, {sp}, %c1 : index\n{i}%hq{k}_c4 = index.constant 4 : index\n"
                f"{i}%hq{k}_le4 = index.cmp ule, {sp}, %hq{k}_c4 : index\n"
                f"{i}scf.if {m.group('c')} {{\n{launch('g_head_mv_q8', i + '  ')}\n{i}}} else {{\n"
                f"{i}  scf.if %hq{k}_le4 {{\n{launch('g_head_mv_t4_q8', i + '    ')}\n{i}  }} else {{\n"
                f"{i}    %hq{k}_c8 = index.constant 8 : index\n{i}    %hq{k}_le8 = index.cmp ule, {sp}, %hq{k}_c8 : index\n"
                f"{i}    scf.if %hq{k}_le8 {{\n{launch('g_head_mv_t8_q8', i + '      ')}\n{i}    }} else {{\n"
                f"{launch('g_head_mv_t16_q8', i + '      ')}\n{i}    }}\n{i}  }}\n{i}}}")
    out = pat.sub(sub, program)
    assert "@g_head_mv[" not in out and "@g_head_mv_t4[" not in out, "ACT_Q8: a head dispatch was not rewritten (set ACT_Q8=0)"
    return out


def quant_wiring(program):
    """ACT_Q8: beside every `scf.if %gemm|%mm { g_row_scale_f16(rows, width, X, S, X16) }` launch g_quant_q8(X -> X16q)
    in the else branch (the decode path), and allocate each X16q next to X16 (same size: 2 bytes per value >= the
    1.25 an int8 copy needs). The _q8 matvecs read X16q exactly where the GEMM reads X16, so the GEMM's own
    freshness invariant covers them."""
    import re as _re
    pat = _re.compile(r"(scf\.if (%\w+) \{\s*kernel\.launch @g_row_scale_f16\[[^\]]*\]\((%\w+), (%\w+), (%\w+), (%\w+), (%\w+)\) : \[index\]\(index, index, buffer, buffer, buffer\)\s*\})")
    read = set(_re.findall(r"kernel\.launch @g_mv_\w*_q8\w*\[[^\]]*\]\([^,]+, [^,]+, [^,]+, (%\w+),", program))
    def sub(m):
        rows, width, x, x16 = m.group(3), m.group(4), m.group(5), m.group(7)
        if x16 + "q" not in read:   # only f32 consumers (e.g. a DFlash feature projection wider than 32768)
            return m.group(1)
        return (m.group(1) + f" else {{\n    kernel.launch @g_quant_q8[{rows}, {width}]({rows}, {width}, {x}, {x16}q) : [index, index](index, index, buffer, buffer)\n  }}")
    program = pat.sub(sub, program)
    program = _re.sub(r"(?m)^(\s*)(%\w+16) = buffer\.alloca<global> align\(256\) (%\w+) : buffer$",
                      lambda m: f"{m.group(0)}\n{m.group(1)}{m.group(2)}q = buffer.alloca<global> align(256) {m.group(3)} : buffer"
                      if m.group(2) + "q" in read else m.group(0), program)
    read = set(_re.findall(r"kernel\.launch @g_mv_\w*_q8\w*\[[^\]]*\]\([^,]+, [^,]+, [^,]+, (%\w+),", program))
    made = set(_re.findall(r"kernel\.launch @g_quant_q8\[[^\]]*\]\([^,]+, [^,]+, [^,]+, (%\w+)\)", program))
    assert read <= made, f"ACT_Q8: int8 activations read but never quantized: {sorted(read - made)} (set ACT_Q8=0)"
    return program


def lin(fmt, x, s, w, wsize, y, k, n, kval, mode="store", width=None):
    USED_GEMM.setdefault(fmt, set()).add(mode)
    return linear(fmt, x, s, w, wsize, y, k, n, kval, mode, width=width)


# PAIR_PROJ: two projections of the same input with the same format and shape (attention q + gate, k + v) take one
# dual-weight int8 launch at 8 and 16 tokens (verify steps of 2-4 drafting streams), as gate/up already do: one
# activation load and butterfly setup for both weights, and the select-free butterfly (BFLY_PERM). Each weight's rows
# round exactly as in the single-weight kernel (kernel tests mvpair_*). Prefill (GEMM), 1-token decode and the 4-token
# variant keep separate launches. Muse-Glimmer 4 streams (c4ep, same server): q + gate -1 to -2 ms per 153 ms epoch.
# k + v (6656 x 256) measured -16 us per layer in isolation but no faster end to end (+1.5 ms per epoch with both pairs;
# the small separate k and v launches presumably overlap neighbouring launches), so it is opt-in (PAIR_PROJ=1 or kv,
# which also pairs the DFlash drafter's k + v).
DIAG = set(os.environ.get("DIAG", "").split(","))   # temporary diagnostics
PAIR_PROJ = os.environ.get("PAIR_PROJ", "qg")   # qg (default): q + gate only; kv: k + v only; 1: both; 0: off


def lin2(fmt, x, s, wa, wb, wsize, ya, yb, k, n, kval, width=None):
    """Projections ya = x Wa^T and yb = x Wb^T (same format and size): two lin() calls, except that with PAIR_PROJ the
    int8 8/16-token matvecs take one dual-weight (pair) launch."""
    iq = ACT_Q8 and fmt in IQ_FMTS and kval // 32 <= 1024
    if not (PAIR_PROJ != "0" and iq):
        return lin(fmt, x, s, wa, wsize, ya, k, n, kval, width=width) + "\n  " + lin(fmt, x, s, wb, wsize, yb, k, n, kval, width=width)
    x16 = {"%h": "%h16", "%o": "%o16", "%f": "%f16"}.get(x, x + "16")
    a1 = lambda w, y: f"({n}, %t, %zero_i32, {x16}q, {w}, {y}) : [index, index](index, index, i32, buffer, view<{wsize}xi8>, buffer)"
    a2 = f"({n}, %t, %zero_i32, {x16}q, {wa}, {wb}, {ya}, {yb}) : [index, index](index, index, i32, buffer, view<{wsize}xi8>, view<{wsize}xi8>, buffer, buffer)"
    gemm = lambda w, y: linear(fmt, x, s, w, wsize, y, k, n, kval, gemm_only=True, width=width)
    USED_GEMM.setdefault(fmt, set()).add("store")
    _LIN[0] += 1
    c = _LIN[0]
    return f"""scf.if %gemm {{
      {gemm(wa, ya)}
      {gemm(wb, yb)}
    }} else {{
      %c4_p{c} = index.constant 4 : index
      %le4_p{c} = index.cmp ule, %t, {"%c1" if os.environ.get("FORCE16") else f"%c4_p{c}"} : index
      scf.if %le4_p{c} {{
        scf.if %single {{
          kernel.launch @{mv_name(fmt, kval, False, n, 1, True)}[{n}, %t]{a1(wa, ya)}
          kernel.launch @{mv_name(fmt, kval, False, n, 1, True)}[{n}, %t]{a1(wb, yb)}
        }} else {{
          kernel.launch @{mv_name(fmt, kval, False, n, 4, True)}[{n}, %t]{a1(wa, ya)}
          kernel.launch @{mv_name(fmt, kval, False, n, 4, True)}[{n}, %t]{a1(wb, yb)}
        }}
      }} else {{
        %c8_p{c} = index.constant 8 : index
        %le8_p{c} = index.cmp ule, %t, {"%c0" if os.environ.get("FORCE16") else f"%c8_p{c}"} : index
        scf.if %le8_p{c} {{
          kernel.launch @{mv_name(fmt, kval, 2, n, 8, True)}[{n}, %t]{a2}
        }} else {{
          kernel.launch @{mv_name(fmt, kval, 2, n, 16, True)}[{n}, %t]{a2}
        }}
      }}
    }}"""


def lin_mv(fmt, x, w, wsize, y, n, kval):
    """Always the matrix-vector kernels (any token count): for tiny projections (n = 16)."""
    a = f"({n}, %t, %zero_i32, {x}, {w}, {y}) : [index, index](index, index, i32, buffer, view<{wsize}xi8>, buffer)"
    few = f"kernel.launch @{mv_name(fmt, kval, False, n, 4)}[{n}, %t]{a}"
    nv = NVAL.get(n, None) if isinstance(n, str) else n
    if nv and nv % 16 == 0 and mv_name(fmt, kval, False, n, 4).endswith(("_r1", "_r2", "_r4", "_r8")):
        # Prefill: 16 rows per workgroup. The decode-tuned R (as few as 1 row) re-reads each 4-token activation
        # block once per row: Qwen3.8-27B's 48-row ssm_alpha/beta were ~5 % of 2k prefill at R=1.
        USED_MV.add((fmt, kval, False, 4))
        few = f"""scf.if %gemm {{
      kernel.launch @g_mv_{fmt}_k{kval}_t4[{n}, %t]{a}
    }} else {{
      {few}
    }}"""
    return f"""scf.if %single {{
    kernel.launch @{mv_name(fmt, kval, False, n, 1)}[{n}, %t]{a}
  }} else {{
    {few}
  }}"""


USED_MOE = set()     # (fmt, K, N, mode) expert matvec / grouped GEMM kernels
MOE_R = int(os.environ.get("MOE_R", "8"))


def moe_layer(i, b, nm, sz, par):
    """Shared experts (dense swiglu FFN of width FS, stored into y; with SHGATE scaled per token by
    sigmoid(h . ffn_gate_inp_shexp) inside the combine), then the routed experts (moe_routed) added into y."""
    T = []
    if FS:
        T += [par("w_gsh", "ffn_gate_shexp"), par("w_ush", "ffn_up_shexp"), par("w_dsh", "ffn_down_shexp")]
        fsg = fmt_of(nm("ffn_gate_shexp"))
        T.append("  " + lin(fsg, "%h", "%s_h", (f"%w_gsh_{i}", f"%w_ush_{i}"), sz("ffn_gate_shexp"), "%f", "%nH", "%nFS", H, mode="swiglu") + "\n")
        T.append(f"  scf.if %gemm {{\n    kernel.launch @g_row_scale_f16[%t](%t, %nFS, %f, %s_f, %f16) : [index](index, index, buffer, buffer, buffer)\n  }}\n")
        T.append("  " + lin(fmt_of(nm("ffn_down_shexp")), "%f", "%s_f", f"%w_dsh_{i}", sz("ffn_down_shexp"), "%y", "%nFS", "%nH", FS) + "\n")
    if SHGATE:
        sty = TENS[b + "ffn_gate_inp_shexp.weight"]["type"]
        T.append(f"""  %wsg_{i} = command.parameter %weights, "{b}ffn_gate_inp_shexp.weight" : view<{H}x{sty}>
  kernel.launch @g_moe_shared_gate_{sty}[%t](%t, %h, %wsg_{i}, %sg) : [index](index, buffer, view<{H}x{sty}>, buffer)
""")
    return "".join(T) + moe_routed(i, b)


def moe_routed(i, b, x="%h", rx="%h", out="%y"):
    """Layer i's routed experts on input buffer X (prefill: its f16 copy %h16/%s_h, made here when X is not %h), the
    router reading RX: router -> per-assignment gate/up (swiglu) and down -> weighted combine into OUT (added to it
    when the layer has shared experts, scaled by the shared gate with SHGATE). cfg.moe gate_up_fused: one
    ffn_gate_up_exps tensor (each expert's gate rows, then its up rows); down_scale: ffn_down_exps.scale, a
    per-expert output scale folded into the combine weights (g_moe_wscale)."""
    E, KU, FE = MOE["experts"], MOE["used"], MOE["ffn"]
    fused = MOE.get("gate_up_fused")
    gn, un = (b + "ffn_gate_up_exps.weight",) * 2 if fused else (b + "ffn_gate_exps.weight", b + "ffn_up_exps.weight")
    fg, fu, fd = fmt_of(gn), fmt_of(un), fmt_of(b + "ffn_down_exps.weight")
    sg, su, sd = tensor_bytes(gn), tensor_bytes(un), tensor_bytes(b + "ffn_down_exps.weight")
    # Gate and up in different formats (MiniCPM-V-4.7's Q4_K_M: Q4_K gate, Q6_K up): kernels "GATE+UP" (gen_moe.py)
    gu = fg if fg == fu else f"{fg}+{fu}"
    gun = gu.replace("+", "_") + ("_fused" if fused else "")
    USED_MOE.add((gu, H, FE, "gu")); USED_MOE.add((fd, FE, H, "dn"))
    R = MOE_R
    rty = TENS[b + "ffn_gate_inp.weight"]["type"]
    sgarg, sgty = ("%sg, ", "buffer, ") if SHGATE else ("", "")
    bias = f"%ebias_{i}" if MOE["probs_bias"] else "%wts"
    bias_ty = f"view<{E}xf32>" if MOE["probs_bias"] else "buffer"
    rbias = ""
    if MOE.get("router_bias"):   # gating 3: the router's own bias, added to the logits (gen_moe.route's bias)
        assert MOE["gating"] == 3 and not MOE["probs_bias"], "a router bias is wired for gating 3 (softmax of the picked) only"
        bias, bias_ty = f"%rbias_{i}", f"view<{E}xf32>"
        rbias = f'  %rbias_{i} = command.parameter %weights, "{b}ffn_gate_inp.bias" : view<{E}xf32>\n'
    # per-expert output biases (gpt-oss): after the gate/up and down weights of each expert kernel
    eb = {p: f'  %{p}be_{i} = command.parameter %weights, "{b}ffn_{n}_exps.bias" : view<{E * (FE if p != "d" else H)}xf32>\n'
          for p, n in (("g", "gate"), ("u", "up"), ("d", "down"))} if MOE.get("expert_bias") else {}
    gu_b, gu_bt = (f", %gbe_{i}, %ube_{i}", f", view<{E * FE}xf32>, view<{E * FE}xf32>") if eb else ("", "")
    dn_b, dn_bt = (f", %dbe_{i}", f", view<{E * H}xf32>") if eb else ("", "")
    wu = f"%wge_{i}" if fused else f"%wue_{i}"
    wsc = (f"    kernel.launch @g_moe_wscale[%t](%t, %ids, %wts, %dsc_{i}) : [index](index, buffer, buffer, view<{E}xf32>)\n"
           if MOE.get("down_scale") else "")
    x16 = "" if x == "%h" else f"    kernel.launch @g_row_scale_f16[%t](%t, %nH, {x}, %s_h, %h16) : [index](index, index, buffer, buffer, buffer)\n"
    return f"""  %router_{i} = command.parameter %weights, "{b}ffn_gate_inp.weight" : view<{E * H}x{rty}>
""" + (f"""  %ebias_{i} = command.parameter %weights, "{b}exp_probs_b.bias" : view<{E}xf32>
""" if MOE["probs_bias"] else "") + rbias + "".join(eb.values()) + (f"""  %dsc_{i} = command.parameter %weights, "{b}ffn_down_exps.scale" : view<{E}xf32>
""" if MOE.get("down_scale") else "") + f"""  %wge_{i} = command.parameter %weights, "{gn}" : view<{sg}xi8>
""" + ("" if fused else f"""  %wue_{i} = command.parameter %weights, "{un}" : view<{su}xi8>
""") + f"""  %wde_{i} = command.parameter %weights, "{b}ffn_down_exps.weight" : view<{sd}xi8>
  scf.if {"%c_false" if "moe" in DIAG else "%gemm"} {{
    kernel.launch @g_moe_route_tr_live[%t](%t, {rx}, %router_{i}, {bias}, %ids, %wts, %info) : [index](index, buffer, view<{E * H}x{rty}>, {bias_ty}, buffer, buffer, buffer)
{wsc}    kernel.launch @g_moe_bucket[%t](%t, %max_tiles, %info, %ids, %tile_info, %rowmap, %pos) : [index](index, index, buffer, buffer, buffer, buffer, buffer)
{x16}    kernel.launch @g_moe_ggu_{gun}_k{H}[%max_rows, %nFE](%max_rows, %nH, %nFE, %t, %tile_info, %rowmap, %h16, %s_h, %wge_{i}, {wu}{gu_b}, %act_sorted) : [index, index](index, index, index, index, buffer, buffer, buffer, buffer, view<{sg}xi8>, view<{su}xi8>{gu_bt}, buffer)
    kernel.launch @g_row_scale_f16_tiles[%max_rows](%max_rows, %nFE, %act_sorted, %s_rows, %act16, %tile_info) : [index](index, index, buffer, buffer, buffer, buffer)
    kernel.launch @g_moe_gdn_{fd}_k{FE}[%max_rows, %nH](%max_rows, %nFE, %nH, %max_rows, %tile_info, %rowmap, %act16, %s_rows, %wde_{i}{dn_b}, %ys_sorted) : [index, index](index, index, index, index, buffer, buffer, buffer, buffer, view<{sd}xi8>{dn_bt}, buffer)
    kernel.launch @g_moe_combine_sorted[%t](%t, %max_rows, %ys_sorted, %pos, %wts, {sgarg}{out}) : [index](index, index, buffer, buffer, buffer, {sgty}buffer)
  }} else {{
    kernel.launch @g_moe_route[%t](%t, {rx}, %router_{i}, {bias}, %ids, %wts) : [index](index, buffer, view<{E * H}x{rty}>, {bias_ty}, buffer, buffer)
{wsc}    kernel.launch @g_moe_gu_{gun}_k{H}_r{R}[%nFE, %ta](%nFE, %ta, %ids, {x}, %wge_{i}, {wu}{gu_b}, %act) : [index, index](index, index, buffer, buffer, view<{sg}xi8>, view<{su}xi8>{gu_bt}, buffer)
    kernel.launch @g_moe_dn_{fd}_k{FE}_r{R}[%nH, %ta](%nH, %ta, %ids, %act, %wde_{i}{dn_b}, %ys) : [index, index](index, index, buffer, buffer, view<{sd}xi8>{dn_bt}, buffer)
    kernel.launch @g_moe_combine[%t](%t, %ys, %wts, {sgarg}{out}) : [index](index, buffer, buffer, {sgty}buffer)
  }}
"""


def moe_alloc():
    """Scratch buffers of the MoE layers (decode: per assignment; prefill: bucketed 64-row tiles, worst case)."""
    KU, FE = MOE["used"], MOE["ffn"]
    return f"""  %nFS = index.constant {max(FS, 32)} : index
  %cKU = index.constant {KU} : index
  %ta = index.mul %t, %cKU : index
  %nFE = index.constant {FE} : index
  %ids_bytes = index.constant {KU * 4} : offset
  %ids_size = index.scale %t, %ids_bytes : index, offset -> offset
  %ids = buffer.alloca<global> align(256) %ids_size : buffer
  %wts = buffer.alloca<global> align(256) %ids_size : buffer
  %act_bytes = index.constant {KU * FE * 4} : offset
  %act_size = index.scale %t, %act_bytes : index, offset -> offset
  %act = buffer.alloca<global> align(256) %act_size : buffer
  %ys_bytes = index.constant {KU * H * 4} : offset
  %ys_size = index.scale %t, %ys_bytes : index, offset -> offset
  %ys = buffer.alloca<global> align(256) %ys_size : buffer
  %c64m = index.constant 64 : index
  %cEm = index.constant {MOE["experts"]} : index
  %ta_tiles = index.div %ta, %c64m : index
  %max_tiles = index.add %ta_tiles, %cEm : index
  %max_rows = index.mul %max_tiles, %c64m : index
  %b8 = index.constant 8 : offset
  %b4 = index.constant 4 : offset
  %ti_size = index.scale %max_tiles, %b8 : index, offset -> offset
  %tile_info = buffer.alloca<global> align(256) %ti_size : buffer
  %rm_size = index.scale %max_rows, %b4 : index, offset -> offset
  %rowmap = buffer.alloca<global> align(256) %rm_size : buffer
  %pos_size = index.scale %ta, %b4 : index, offset -> offset
  %pos = buffer.alloca<global> align(256) %pos_size : buffer
  %fe4 = index.constant {FE * 4} : offset
  %fe2 = index.constant {FE * 2} : offset
  %h4 = index.constant {H * 4} : offset
  %as_size = index.scale %max_rows, %fe4 : index, offset -> offset
  %act_sorted = buffer.alloca<global> align(256) %as_size : buffer
  %a16_size = index.scale %max_rows, %fe2 : index, offset -> offset
  %act16 = buffer.alloca<global> align(256) %a16_size : buffer
  %sr_size = index.scale %max_rows, %b8 : index, offset -> offset
  %s_rows = buffer.alloca<global> align(256) %sr_size : buffer
  %yss_size = index.scale %max_rows, %h4 : index, offset -> offset
  %ys_sorted = buffer.alloca<global> align(256) %yss_size : buffer
""" + ("""  %sg_size = index.scale %t, %b4 : index, offset -> offset
  %sg = buffer.alloca<global> align(256) %sg_size : buffer
""" if SHGATE else "")


def moe_kernels(row_scale, gemm_fn):
    """The MoE layers' kernels (gen_moe.py) for the formats and shapes model() used (USED_MOE)."""
    import gen_moe as GM
    rty = TENS[f"blk.{MOE['leading_dense']}.ffn_gate_inp.weight"]["type"]
    a = (H, MOE["experts"], MOE["used"], MOE["gating"], MOE["norm_weights"], MOE["probs_bias"] or MOE.get("router_bias", False), rty)
    parts = [GM.route(*a, scale=MOE.get("scale", 1.0)), GM.route_prefill_live(*a, scale=MOE.get("scale", 1.0)), GM.row_scale_tiles(row_scale)]
    # gate/up activation: silu (default) or gpt-oss's clamped swiglu (cfg moe.act, alpha and limit from llama.cpp)
    act = ("swiglu_oai", MOE.get("swiglu_alpha", 1.702), MOE.get("swiglu_limit", 7.0)) if MOE.get("act") == "swiglu_oai" else None
    xb = dict(bias=True, act=act) if MOE.get("expert_bias") else dict(act=act) if act else {}
    for fmt, K, N, mode in sorted(USED_MOE):
        mv = GM.expert_mv(fmt, K, N, MOE["experts"], MOE["used"], MOE_R, mode, fused=bool(MOE.get("gate_up_fused")) and mode == "gu", **xb)
        # GEGLU experts: the expert matvec's scalar silu, the grouped GEMM's vector silu, as gelu-tanh
        parts.append(mv.replace("    %silu = scalar.siluf<afn> %t0 : f32\n", gelu_scalar("silu", "t0", "    ") + "\n") if GELU else mv)
    comb, comb_s = GM.combine(H, MOE["used"], acc=bool(FS)), GM.combine_sorted(H, MOE["used"], acc=bool(FS))
    if SHGATE:
        comb, comb_s = GM.shared_gated(comb), GM.shared_gated(comb_s)
        for sty in sorted({t["type"] for n, t in TENS.items() if n.endswith("ffn_gate_inp_shexp.weight")}):
            parts.append(GM.shared_gate(H, sty).replace("@g_moe_shared_gate(", f"@g_moe_shared_gate_{sty}(")
                         .replace('export("g_moe_shared_gate")', f'export("g_moe_shared_gate_{sty}")'))
    parts += [comb, GM.bucket(MOE["experts"], MOE["used"]), comb_s] + ([GM.wscale(MOE["experts"], MOE["used"])] if MOE.get("down_scale") else [])
    for fmt, K, N, mode in sorted(USED_MOE):
        gg = GM.grouped_gemm(fmt, "swiglu" if mode == "gu" else "store", N, K, MOE["experts"], MOE["used"], gemm_fn,
                             fused=bool(MOE.get("gate_up_fused")) and mode == "gu", **xb)
        parts.append(gelu_vectors(gg) if GELU else gg)
    return "".join(parts)


def gdn_layer(i, b, nm, sz, par):
    """Gated delta-net token mixer of layer i: h -> o (VD wide), then o16 for the output GEMM."""
    g = GDN
    C, VD, HV, HK, DV, K = g["C"], g["VD"], g["HV"], g["HK"], g["DV"], g["K"]
    P = [par("wqkv", "attn_qkv"), par("wz", "attn_gate"), par("walpha", "ssm_alpha"), par("wbeta", "ssm_beta"), par("wout", "ssm_out"),
         f'    %conv_{i} = command.parameter %weights, "{b}ssm_conv1d.weight" : view<{C * K}xf32>\n',
         f'    %ssm_a_{i} = command.parameter %weights, "{b}ssm_a" : view<{HV}xf32>\n',
         f'    %dt_bias_{i} = command.parameter %weights, "{b}ssm_dt.bias" : view<{HV}xf32>\n',
         f'    %ssm_norm_{i} = command.parameter %weights, "{b}ssm_norm.weight" : view<{DV}xf32>\n']
    T = P + [
        "  " + lin(fmt_of(nm("attn_qkv")), "%h", "%s_h", f"%wqkv_{i}", sz("attn_qkv"), "%gq", "%nH", "%nC", H) + "\n",
        "  " + lin(fmt_of(nm("attn_gate")), "%h", "%s_h", f"%wz_{i}", sz("attn_gate"), "%gz", "%nH", "%nVD", H) + "\n",
        "  " + lin_mv(fmt_of(nm("ssm_alpha")), "%h", f"%walpha_{i}", sz("ssm_alpha"), "%galpha", "%nHV", H) + "\n",
        "  " + lin_mv(fmt_of(nm("ssm_beta")), "%h", f"%wbeta_{i}", sz("ssm_beta"), "%gbeta", "%nHV", H) + "\n",
        f"""  kernel.launch @g_gdn_conv[%t](%t, %spans, %layer_{i}, %metadata, %info, %origins, %arena, %gq, %conv_{i}, %gs) : [index](index, index, index, buffer, buffer, buffer, buffer, buffer, view<{C * K}xf32>, buffer)
  kernel.launch @g_gdn_conv_commit[%spans](%t, %spans, %layer_{i}, %metadata, %origins, %arena, %gq) : [index](index, index, index, buffer, buffer, buffer, buffer)
  kernel.launch @g_gdn_qk_norm[%t](%t, %gs, %ginv) : [index](index, buffer, buffer)
  kernel.launch @g_gdn_recurrence[%spans](%t, %spans, %layer_{i}, %metadata, %origins, %arena, %gs, %ginv, %galpha, %gbeta, %ssm_a_{i}, %dt_bias_{i}, %graw) : [index](index, index, index, buffer, buffer, buffer, buffer, buffer, buffer, buffer, view<{HV}xf32>, view<{HV}xf32>, buffer)
  kernel.launch @g_gdn_norm_gate[%t](%t, %info, %graw, %ssm_norm_{i}, %gz, %o) : [index](index, buffer, buffer, view<{DV}xf32>, buffer, buffer)
  scf.if %gemm {{
    kernel.launch @g_row_scale_f16[%t](%t, %nVD, %o, %s_o, %o16) : [index](index, index, buffer, buffer, buffer)
  }}
""",
        "  " + lin(fmt_of(nm("ssm_out")), "%o", "%s_o", f"%wout_{i}", sz("ssm_out"), "%y", "%nVD", "%nH", VD) + "\n"]
    return "".join(T)


def gelu_scalar(o, x, ind="      "):
    """o = gelu_tanh(x) = x * logistic(2 sqrt(2/pi) (x + 0.044715 x^3)) (llama.cpp's GELU)."""
    return (f"{ind}%{o}_x2 = scalar.mulf %{x}, %{x} : f32\n{ind}%{o}_x3 = scalar.mulf %{o}_x2, %{x} : f32\n"
            f"{ind}%{o}_ca = scalar.constant 0.044715 : f32\n{ind}%{o}_cb = scalar.constant -1.5957691216057308 : f32\n"
            f"{ind}%{o}_in = scalar.fmaf %{o}_x3, %{o}_ca, %{x} : f32\n{ind}%{o}_ar = scalar.mulf %{o}_in, %{o}_cb : f32\n"
            f"{ind}%{o}_ex = scalar.expf<afn> %{o}_ar : f32\n{ind}%{o}_one = scalar.constant 1.0 : f32\n"
            f"{ind}%{o}_de = scalar.addf %{o}_ex, %{o}_one : f32\n{ind}%{o} = scalar.divf %{x}, %{o}_de : f32")


def gelu_vectors(text):
    """Replace `%o = vector.siluf<afn> %x : vector<NxT>` lines by gelu-tanh (GEGLU FFN)."""
    import re
    def sub(m):
        ind, o, x, ty = m.group(1), m.group(2), m.group(3), m.group(4)
        return (f"{ind}%{o}_x2 = vector.mulf %{x}, %{x} : {ty}\n{ind}%{o}_x3 = vector.mulf %{o}_x2, %{x} : {ty}\n"
                f"{ind}%{o}_ca = vector.constant 0.044715 : {ty}\n{ind}%{o}_cb = vector.constant 1.5957691216057308 : {ty}\n"
                f"{ind}%{o}_in = vector.fmaf %{o}_x3, %{o}_ca, %{x} : {ty}\n{ind}%{o}_ar = vector.mulf %{o}_in, %{o}_cb : {ty}\n"
                f"{ind}%{o}_sg = vector.logisticf<afn> %{o}_ar : {ty}\n{ind}%{o} = vector.mulf %{x}, %{o}_sg : {ty}")
    return re.sub(r"( *)%(\w+) = vector\.siluf<afn> %(\w+) : (vector<\w+>)", sub, text)


def split_qkv():
    """g_split_qkv: a fused attn_qkv projection's rows [q (QH heads) | k (KVH) | v (KVH)] -> q, k, v (QKV_FUSED)."""
    NHT = QH + 2 * KVH
    return f"""
// Fused attention projection (cfg attn_qkv, spark2_5): each token's attn_qkv row is [q ({QH} heads) | k ({KVH}) | v ({KVH})]
// of {D} values per head; copied into the q, k and v buffers the rest of the layer reads.
kernel.def export("g_split_qkv") @g_split_qkv(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %qkv: buffer, %q: buffer, %k: buffer, %v: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %thread_id = kernel.workitem.id<x> : index
  %thread = index.assume %thread_id [range(%thread_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %call = index.constant {NHT * D} : index
  %cd = index.constant {D} : index
  %cqh = index.constant {QH} : index
  %ckh = index.constant {QH + KVH} : index
  %c256 = index.constant 256 : index
  %in_view = buffer.view %qkv[%zero] : buffer -> view<[%tokens]x{NHT}x{D}xf32>
  %q_view = buffer.view %q[%zero] : buffer -> view<[%tokens]x{QH}x{D}xf32>
  %k_view = buffer.view %k[%zero] : buffer -> view<[%tokens]x{KVH}x{D}xf32>
  %v_view = buffer.view %v[%zero] : buffer -> view<[%tokens]x{KVH}x{D}xf32>
  scf.for %i = [%thread to %call step %c256] {{
    %bi = index.assume %i [range(%i, 0, {NHT * D - 1})] : index
    %h_value = index.div %bi, %cd : index
    %h = index.assume %h_value [range(%h_value, 0, {NHT - 1})] : index
    %d_value = index.rem %bi, %cd : index
    %d = index.assume %d_value [range(%d_value, 0, {D - 1})] : index
    %x = view.load %in_view[%token, %h, %d] : view<[%tokens]x{NHT}x{D}xf32> -> f32
    %is_q = index.cmp ult, %h, %cqh : index
    scf.if %is_q {{
      %qh = index.assume %h [range(%h, 0, {QH - 1})] : index
      view.store %x, %q_view[%token, %qh, %d] : f32, view<[%tokens]x{QH}x{D}xf32>
    }} else {{
      %is_k = index.cmp ult, %h, %ckh : index
      scf.if %is_k {{
        %kh_value = index.sub %h, %cqh : index
        %kh = index.assume %kh_value [range(%kh_value, 0, {KVH - 1})] : index
        view.store %x, %k_view[%token, %kh, %d] : f32, view<[%tokens]x{KVH}x{D}xf32>
      }} else {{
        %vh_value = index.sub %h, %ckh : index
        %vh = index.assume %vh_value [range(%vh_value, 0, {KVH - 1})] : index
        view.store %x, %v_view[%token, %vh, %d] : f32, view<[%tokens]x{KVH}x{D}xf32>
      }}
    }}
  }}
  kernel.return
}}
"""


def gate_heads():
    """g_gate_heads: a per-head gate logit (HEAD_GATE) broadcast over the head's D dims (attn_gate_act softplus, laguna:
    softplus(logit), which the combine / flash kernels multiply by as it is)."""
    return f"""
// Per-head attention gate (cfg attn_gate "per_head", spark2_5): gate[t, h, d] = gh[t, h], the head's pre-sigmoid logit;
// the attention combine / flash kernels then scale o[t, h, d] by sigmoid(gate[t, h, d]) as for an elementwise gate.
kernel.def export("g_gate_heads") @g_gate_heads(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %gh: buffer, %gate: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %thread_id = kernel.workitem.id<x> : index
  %thread = index.assume %thread_id [range(%thread_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %cqd = index.constant {QH * D} : index
  %cd = index.constant {D} : index
  %c256 = index.constant 256 : index
  %gh_view = buffer.view %gh[%zero] : buffer -> view<[%tokens]x{QH}xf32>
  %g_view = buffer.view %gate[%zero] : buffer -> view<[%tokens]x{QH}x{D}xf32>
  scf.for %i = [%thread to %cqd step %c256] {{
    %bi = index.assume %i [range(%i, 0, {QH * D - 1})] : index
    %h_value = index.div %bi, %cd : index
    %h = index.assume %h_value [range(%h_value, 0, {QH - 1})] : index
    %d_value = index.rem %bi, %cd : index
    %d = index.assume %d_value [range(%d_value, 0, {D - 1})] : index
    %x = view.load %gh_view[%token, %h] : view<[%tokens]x{QH}xf32> -> f32{"""
    %xs = scalar.softplusf<afn> %x : f32""" if GATE_ACT == "softplus" else ""}
    view.store %x{"s" if GATE_ACT == "softplus" else ""}, %g_view[%token, %h, %d] : f32, view<[%tokens]x{QH}x{D}xf32>
  }}
  kernel.return
}}
"""


def geglu_mul():
    """g_silu_mul for GEGLU models: SILU_MUL with the channels viewed as [n/4]x4 groups, so the vector footprint is
    in bounds by the view's static extent (SILU_MUL's is proven only from the launch's constant n, so it compiles in
    the model but not as a kernel-test root); gelu_vectors() then swaps silu for gelu-tanh."""
    t = SILU_MUL.replace("silu(f[t, c])", "gelu(f[t, c])")
    for old, new in (
        ("  %c1024 = index.constant 1024 : index\n", "  %c0 = index.constant 0 : index\n  %c256 = index.constant 256 : index\n"
         "  %n4_value = index.div %n, %c4 : index\n  %n4 = index.assume %n4_value [range(%n4_value, 1, 16384)] : index\n"),
        ("view<[%tokens]x[%n]xf32>", "view<[%tokens]x[%n4]x4xf32>"),
        ("  %first = index.mul %lane, %c4 : index\n  scf.for %c0v = [%first to %n step %c1024] {\n"
         "    %c = index.assume %c0v [range(%c0v, 0, 65532), multiple_of(%c0v, 4), lt(%c0v, %n)] : index\n",
         "  scf.for %c0v = [%lane to %n4 step %c256] {\n    %c = index.assume %c0v [range(%c0v, 0, 16383), lt(%c0v, %n4)] : index\n"),
        ("[%token, %c]", "[%token, %c, %c0]"),
        ("    %sg = vector.logisticf<afn> %g : vector<4xf32>\n    %s = vector.mulf %g, %sg : vector<4xf32>\n",
         "    %s = vector.siluf<afn> %g : vector<4xf32>\n")):
        assert old in t, old
        t = t.replace(old, new)
    return t


def with_heads(h, gen):
    """gen() with the query-head count h (a layer_heads layer narrower than QH), its kernels and functions renamed
    with the suffix _h<h>."""
    import re
    global QH, G
    saved = QH, G
    QH, G = h, h // KVH
    try:
        text = gen()
    finally:
        QH, G = saved
    names = set(re.findall(r"(?:kernel|func)\.def (?:export\(\"\w+\"\) |inline )?@(\w+)\(", text))
    return re.sub(r'(@|export\(")(' + "|".join(sorted(names, key=len, reverse=True)) + r')\b', rf"\1\2_h{h}", text)


NARROW_KERNELS = r"@(g_qk_rope_store(?:_swa)?|g_attention_flash2|g16_attention_split_listed|g16_attention_combine_listed|g_attention_split|g_attention_combine|g_gate_heads)\["


def qk_rope_swa():
    """g_qk_rope_store_swa: g_qk_rope_store with the sliding layers' rotary dims and base (SWA_ROPE), without the
    frequency factors (llama.cpp lfm3: rope_freqs belongs to the global layers) or YaRN (llama.cpp's swa rope is
    unscaled)."""
    return qk_rope(RD_SWA, THETA_SWA, "g_qk_rope_store_swa", None, None)


SILU_MUL = r"""
// f[t, c] = silu(f[t, c]) * u[t, c] for the first n channels of each token (gate/up in different formats).
kernel.def export("g_silu_mul") @g_silu_mul(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %n_arg: index, %f: buffer, %u: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %n = index.assume %n_arg [range(%n_arg, 4, 65536), multiple_of(%n_arg, 4)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c4 = index.constant 4 : index
  %c1024 = index.constant 1024 : index
  %f_view = buffer.view %f[%zero] : buffer -> view<[%tokens]x[%n]xf32>
  %u_view = buffer.view %u[%zero] : buffer -> view<[%tokens]x[%n]xf32>
  %first = index.mul %lane, %c4 : index
  scf.for %c0v = [%first to %n step %c1024] {
    %c = index.assume %c0v [range(%c0v, 0, 65532), multiple_of(%c0v, 4), lt(%c0v, %n)] : index
    %g = vector.load %f_view[%token, %c] : view<[%tokens]x[%n]xf32> -> vector<4xf32>
    %v = vector.load %u_view[%token, %c] : view<[%tokens]x[%n]xf32> -> vector<4xf32>
    %sg = vector.logisticf<afn> %g : vector<4xf32>
    %s = vector.mulf %g, %sg : vector<4xf32>
    %o = vector.mulf %s, %v : vector<4xf32>
    vector.store %o, %f_view[%token, %c] : vector<4xf32>, view<[%tokens]x[%n]xf32>
  }
  kernel.return
}
"""
BIAS_ADD = """
// y[t, c] += b[c] over the hidden row of each token (the attention output projection's bias, cfg attn_out_bias).
kernel.def export("g_bias_add") @g_bias_add(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %n_arg: index, %y: buffer, %b: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c4 = index.constant 4 : index
  %c1024 = index.constant 1024 : index
  %cH = index.constant @H@ : index
  %y_view = buffer.view %y[%zero] : buffer -> view<[%tokens]x@H@xf32>
  %b_view = buffer.view %b[%zero] : buffer -> view<@H@xf32>
  %first = index.mul %lane, %c4 : index
  scf.for %c0v = [%first to %cH step %c1024] {
    %c = index.assume %c0v [range(%c0v, 0, @H4@), multiple_of(%c0v, 4)] : index
    %v = vector.load %y_view[%token, %c] : view<[%tokens]x@H@xf32> -> vector<4xf32>
    %bv = vector.load %b_view[%c] : view<@H@xf32> -> vector<4xf32>
    %o = vector.addf %v, %bv : vector<4xf32>
    vector.store %o, %y_view[%token, %c] : vector<4xf32>, view<[%tokens]x@H@xf32>
  }
  kernel.return
}
""".replace("@H4@", str(H - 4)).replace("@H@", str(H))
# LAST_OUT=1 (opt-in, A/B it per port): the last layer's post-attention work runs only on the rows the head reads
# (llama.cpp's inp_out_ids). Measured: Qwen3.8-27B prefill +1 % (within noise), Muse-Glimmer-30B -3.5 %. Off for drafters that read the trunk's final hidden rows (MTP, Gemma-4 assistant) or a DFlash tap
# after the last layer.
LAST_OUT = os.environ.get("LAST_OUT", "0") == "1" and not (gen_draft.enabled() and gen_draft.DRAFT["kind"] in ("mtp", "assist"))
if LAST_OUT and gen_draft.enabled() and gen_draft.DRAFT["kind"] == "dflash":
    import gen_dflash as _gd
    LAST_OUT = CFG["layers"] not in _gd.config(gen_draft.DRAFT["gguf"])["target_layers"]   # a tap after the last layer

# llama.cpp's output-row selection (inp_out_ids / ggml_get_rows before the last layer's output projection and
# FFN): a prefill epoch only needs the last layer's post-attention rows that feed the head, one per span.
ROWS_SELECT = r"""
// dst[s] = src[row of span s's output token] (zeros for an unselected or absent span); one workgroup per span.
kernel.def export("g_rows_gather") @g_rows_gather(%spans: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%spans, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%spans_arg: index, %tokens_arg: index, %width_arg: index, %metadata: buffer, %src: buffer, %dst: buffer) {
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %width = index.assume %width_arg [range(%width_arg, 4, 65536), multiple_of(%width_arg, 4)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %c256 = index.constant 256 : index
  %c1_i32 = scalar.constant 1 : i32
  %c0_i32 = scalar.constant 0 : i32
  %span_id = kernel.workgroup.id<x> : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %row, %live = func.call @g_span_output_row(%metadata, %spans, %span, %tokens) : (buffer, index, index, index) -> (index, i1)
  %quads = index.div %width, %c4 : index
  %src_view = buffer.view %src[%zero] : buffer -> view<[%tokens]x[%quads]x4xf32>
  %dst_view = buffer.view %dst[%zero] : buffer -> view<[%spans]x[%quads]x4xf32>
  %z4 = vector.constant 0.0 : vector<4xf32>
  scf.for %q = [%lane to %quads step %c256] {
    %qb = index.assume %q [lt(%q, %quads)] : index
    %v = scf.if %live -> (vector<4xf32>) {
      %vv = vector.load %src_view[%row, %qb, %c0] : view<[%tokens]x[%quads]x4xf32> -> vector<4xf32>
      scf.yield %vv : vector<4xf32>
    } else {
      scf.yield %z4 : vector<4xf32>
    }
    vector.store %v, %dst_view[%span, %qb, %c0] : vector<4xf32>, view<[%spans]x[%quads]x4xf32>
  }
  kernel.return
}

// dst[row of span s's output token] = src[s] for every selected span.
kernel.def export("g_rows_scatter") @g_rows_scatter(%spans: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%spans, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%spans_arg: index, %tokens_arg: index, %width_arg: index, %metadata: buffer, %src: buffer, %dst: buffer) {
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %width = index.assume %width_arg [range(%width_arg, 4, 65536), multiple_of(%width_arg, 4)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %c256 = index.constant 256 : index
  %span_id = kernel.workgroup.id<x> : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %row, %live = func.call @g_span_output_row(%metadata, %spans, %span, %tokens) : (buffer, index, index, index) -> (index, i1)
  scf.if %live {
    %quads = index.div %width, %c4 : index
    %src_view = buffer.view %src[%zero] : buffer -> view<[%spans]x[%quads]x4xf32>
    %dst_view = buffer.view %dst[%zero] : buffer -> view<[%tokens]x[%quads]x4xf32>
    scf.for %q = [%lane to %quads step %c256] {
      %qb = index.assume %q [lt(%q, %quads)] : index
      %v = vector.load %src_view[%span, %qb, %c0] : view<[%spans]x[%quads]x4xf32> -> vector<4xf32>
      vector.store %v, %dst_view[%row, %qb, %c0] : vector<4xf32>, view<[%tokens]x[%quads]x4xf32>
    }
  }
  kernel.return
}

// The epoch row of span `span`'s last token (the one the head reads) and whether the span is selected.
func.def inline @g_span_output_row(%metadata: buffer, %spans: index, %span: index, %tokens: index) -> (index, i1) {
  %zero = index.constant 0 : offset
  %c16o = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1_i32 = scalar.constant 1 : i32
  %c0_i32 = scalar.constant 0 : i32
  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %count_bound = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count = index.cast %count_bound : i32 to index
  %in_count = index.cmp ult, %span, %count : index
  %packets = buffer.view %metadata[%c16o] : buffer -> view<[%spans]x9xi32>
  %bspan = index.assume %span [lt(%span, %spans)] : index
  %flags = view.load %packets[%bspan, 6] : view<[%spans]x9xi32> -> i32
  %bits = scalar.andi %flags, %c1_i32 : i32
  %selected = scalar.cmpi ne, %bits, %c0_i32 : i32
  %length32 = view.load %packets[%bspan, 0] : view<[%spans]x9xi32> -> i32
  %input32 = view.load %packets[%bspan, 4] : view<[%spans]x9xi32> -> i32
  %last32 = scalar.addi %input32, %length32 : i32
  %tok32 = scalar.subi %last32, %c1_i32 : i32
  %tok_ok = scalar.cmpi sge, %tok32, %c0_i32 : i32
  %tok_bound = scalar.assume %tok32 [range(%tok32, -1, 8191)] : i32
  %tok_value = index.cast %tok_bound : i32 to index
  %in_range = index.cmp ult, %tok_value, %tokens : index
  %a = scalar.andi %in_count, %selected : i1
  %b = scalar.andi %a, %tok_ok : i1
  %live = scalar.andi %b, %in_range : i1
  %row_value = scf.select %live, %tok_value, %c0 : index
  %row = index.assume %row_value [lt(%row_value, %tokens)] : index
  func.return %row, %live : index, i1
}
"""

# Prefill GEMM tile order per port (A/B it: rows fastest reuses each weight tile across a chunk's row tiles, columns
# fastest reuses each activation tile): cfg["gemm_n_fastest"] false = rows fastest. Muse-Glimmer-30B +4/+7/+6 % at
# 2k/8k/32k; Qwen3.8-27B and Bolt tie; Flint -4 % at 8k. An explicit GEMM_N_FASTEST still wins.
if "gemm_n_fastest" in CFG:
    os.environ.setdefault("GEMM_N_FASTEST", "1" if CFG["gemm_n_fastest"] else "0")
LAST_OUT_SAFE = r"g_row_scale_f16|g_gemm_\w+|g_mv_\w+|g_post_add_norm|g_silu_mul|g_bias_add"   # row-local post-attention kernels
ANCHOR_POST_ATTN = "  scf.if %gemm {\n    kernel.launch @g_row_scale_f16[%t](%t, %nQ, %o, %s_o, %o16)"


def last_out_select_program(program):
    """Apply last_out_select to the last trunk layer of the generated @llm_advance program (the text between the
    last `// ---- layer` marker and the head) and append the gather/scatter kernels."""
    end = program.find("\n  %head = command.parameter %weights") + 1
    start = program.rfind("  // ---- layer ", 0, end) if end else -1
    if start < 0 or ANCHOR_POST_ATTN not in program[start:end] or program[start:end].count(ANCHOR_POST_ATTN) != 1:
        return program   # a kit whose last layer is not the starter's attention layer: unchanged
    import re as _re
    tail = program[start:end].split(ANCHOR_POST_ATTN, 1)[1]
    other = {k for k in _re.findall(r"kernel\.launch @(\w+)", tail) if not _re.fullmatch(LAST_OUT_SAFE, k)}
    if other:   # kernels that may index other per-token state (MoE routing, per-layer embeddings): unchanged
        return program
    return program[:start] + last_out_select(program[start:end]) + program[end:] + "\n" + ROWS_SELECT


def last_out_select(text):
    """llama.cpp's output-row selection for the last layer: on prefill-shaped epochs (%gemm) its attention output
    projection, residual, FFN and final norm run on one row per span (the row the head reads) instead of every
    token: gather those rows of %o / %x, run a renamed copy of the post-attention text on them, scatter the final
    %h rows back. K/V of the last layer are still written for every token. The verify program (every drafted
    token feeds the head) sets %outsel_on false."""
    import re as _re
    anchor = ANCHOR_POST_ATTN
    head, tail = text.split(anchor, 1)
    tail = anchor + tail
    defs = set(_re.findall(r"(?m)^\s*((?:%\w+(?:, )?)+) = ", tail))
    names = set()
    for d in defs:
        names.update(n.strip() for n in d.split(","))
    ren = {n: n + "_os" for n in names}
    ren.update({"%o": "%os", "%o16": "%os16", "%s_o": "%s_os", "%x": "%xs", "%y": "%ys", "%h": "%hs", "%h16": "%hs16",
                "%s_h": "%s_hs", "%f": "%fs", "%f16": "%fs16", "%s_f": "%s_fs", "%fu": "%fus",
                "%o16q": "%os16q", "%h16q": "%hs16q", "%f16q": "%fs16q",
                "%t": "%ts", "%gemm": "%gemm_os", "%single": "%single_os", "%big": "%big_os", "%huge": "%huge_os"})
    comp = _re.sub(r"%\w+", lambda m: ren.get(m.group(0), m.group(0)), tail)
    sizes = (f"""  %outsel_on = scalar.constant true : i1
  %outsel = scalar.andi %gemm, %outsel_on : i1
  %ts = index.add %spans, %c0 : index
  %c64_os = index.constant 64 : index
  %c128_os = index.constant 128 : index
  %gemm_os = index.cmp uge, %ts, %c64_os : index
  %single_os = index.cmp eq, %ts, %c1 : index
  %big_os = index.cmp uge, %ts, %c128_os : index
  %c256_os = index.constant 256 : index
  %ts_rem256 = index.rem %ts, %c256_os : index
  %c0_os256 = index.constant 0 : index
  %ts_full256 = index.cmp eq, %ts_rem256, %c0_os256 : index
  %ts_ge256 = index.cmp uge, %ts, %c256_os : index
  %huge_os = scalar.andi %ts_full256, %ts_ge256 : i1
  %xs_size = index.scale %ts, %row_bytes : index, offset -> offset
  %os_size = index.scale %ts, %wide_bytes : index, offset -> offset
  %fs_size = index.scale %ts, %ffn_bytes : index, offset -> offset
  %ss_bytes = index.constant 8 : offset
  %ss_size = index.scale %ts, %ss_bytes : index, offset -> offset
  %h16s_bytes = index.constant {H * 2} : offset
  %o16s_bytes = index.constant {QH * D * 2} : offset
  %f16s_bytes = index.constant {FF * 2} : offset
  %hs16_size = index.scale %ts, %h16s_bytes : index, offset -> offset
  %os16_size = index.scale %ts, %o16s_bytes : index, offset -> offset
  %fs16_size = index.scale %ts, %f16s_bytes : index, offset -> offset
  %xs = buffer.alloca<global> align(256) %xs_size : buffer
  %ys = buffer.alloca<global> align(256) %xs_size : buffer
  %hs = buffer.alloca<global> align(256) %xs_size : buffer
  %os = buffer.alloca<global> align(256) %os_size : buffer
  %fs = buffer.alloca<global> align(256) %fs_size : buffer
""" + ("  %fus = buffer.alloca<global> align(256) %fs_size : buffer\n" if MIXED_FFN else "") + """  %s_hs = buffer.alloca<global> align(256) %ss_size : buffer
  %s_os = buffer.alloca<global> align(256) %ss_size : buffer
  %s_fs = buffer.alloca<global> align(256) %ss_size : buffer
  %hs16 = buffer.alloca<global> align(256) %hs16_size : buffer
  %os16 = buffer.alloca<global> align(256) %os16_size : buffer
  %fs16 = buffer.alloca<global> align(256) %fs16_size : buffer
""")
    gather = (f"""  %nQ_os = index.constant {QH * D} : index
  %nH_os = index.constant {H} : index
  kernel.launch @g_rows_gather[%spans](%spans, %t, %nQ_os, %metadata, %o, %os) : [index](index, index, index, buffer, buffer, buffer)
  kernel.launch @g_rows_gather[%spans](%spans, %t, %nH_os, %metadata, %x, %xs) : [index](index, index, index, buffer, buffer, buffer)
""")
    scatter = "  kernel.launch @g_rows_scatter[%spans](%spans, %t, %nH_os, %metadata, %hs, %h) : [index](index, index, index, buffer, buffer, buffer)\n"
    return (head + sizes + "  scf.if %outsel {\n" + gather + comp + scatter + "  } else {\n" + tail + "  }\n")


MODEL_PARTS = {}   # model() records its pieces here for the speculative-decoding programs (draft/gen_draft.py)


def rope_bias(block, tag, ind="  "):
    """(parameter lines, launch args, launch types) of block's attention projection biases (QKV_BIAS), which
    g_qk_rope_store takes after its other arguments. Empty for a model without biases."""
    width = {"q": QH * D, "k": KVH * D, "v": KVH * D}
    return ("".join(f'{ind}%{p}_bias_{tag} = command.parameter %weights, "{block}attn_{p}.bias" : view<{width[p]}xf32>\n' for p in QKV_BIAS),
            "".join(f", %{p}_bias_{tag}" for p in QKV_BIAS), "".join(f", view<{width[p]}xf32>" for p in QKV_BIAS))


def model():
    """Command program: embed, then a statically unrolled layer loop (each
    layer's tensor formats come from the GGUF), then the argmax head."""
    QD, KD = QH * D, KVH * D
    sliding = set(CFG.get("sliding_layers", []))
    global NARROW_ROPE
    NARROW_ROPE = {(LH[i], i in sliding) for i in range(L) if LH[i] != QH and i not in RECUR}
    nope = set(CFG.get("nope_layers", []))
    head_name = "output.weight" if "output.weight" in TENS else "token_embd.weight"
    head_fmt = fmt_of(head_name)
    hsize = tensor_bytes(head_name)
    esize = tensor_bytes("token_embd.weight")
    L_ = [f"""
config.decl @llm.context : %value: index where [range(%value, 16, 262144)]

config.decl @llm.spans : %value: index where [range(%value, 1, 8192)]

config.decl @llm.tokens : %value: index where [range(%value, 1, 8192)]

config.decl @llm.pool : %value: index where [range(%value, 16, 1048576)]

command.program.def public @llm_advance() launch(%weights: buffer, %residual: buffer, %metadata: buffer, %origins: buffer, %arena: buffer, %tokens: buffer, %outputs: buffer) {{
  %t = config.get @llm.tokens : index
  %spans = config.get @llm.spans : index
  %context = config.get @llm.context : index
  %pool = config.get @llm.pool : index
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %nH = index.constant {H} : index
  %nFF = index.constant {FF} : index
""" + ("" if FF_UNIFORM else "".join(f"  %nFF{w} = index.constant {w} : index\n" for w in sorted(set(FFL)))) + "".join(
        f"  %nQ{h} = index.constant {h * D} : index\n" + (f"  %nQH{h} = index.constant {h} : index\n" if HEAD_GATE else "")
        for h in NARROW) + f"""  %nQ = index.constant {QD} : index
  %nKV = index.constant {KD} : index
  %zero_i32 = scalar.constant 0 : i32
  %one_i32 = scalar.constant 1 : i32
  %row_bytes = index.constant {H * 4} : offset
  %wide_bytes = index.constant {max(QD, GDN["VD"] if GDN else 0) * 4} : offset
  %kv_bytes = index.constant {KD * 4} : offset
  %info_bytes = index.constant 16 : offset
  %ffn_bytes = index.constant {FF * 4} : offset
  %partial_bytes = index.constant {CHUNKS * 8} : offset
  %x_size = index.scale %t, %row_bytes : index, offset -> offset
  %q_size = index.scale %t, %wide_bytes : index, offset -> offset
  %kv_size = index.scale %t, %kv_bytes : index, offset -> offset
  %info_size = index.scale %t, %info_bytes : index, offset -> offset
  %ffn_size = index.scale %t, %ffn_bytes : index, offset -> offset
  %partial_size = index.scale %spans, %partial_bytes : index, offset -> offset
  %logit_bytes = index.constant {CHUNKS * 128 * 4} : offset
  %logit_size = index.scale %spans, %logit_bytes : index, offset -> offset
  %logits = buffer.alloca<global> align(256) %logit_size : buffer
  %x = buffer.alloca<global> align(256) %x_size : buffer
  %h = buffer.alloca<global> align(256) %x_size : buffer
  %y = buffer.alloca<global> align(256) %x_size : buffer
  %q = buffer.alloca<global> align(256) %q_size : buffer
  %o = buffer.alloca<global> align(256) %q_size : buffer
  %k = buffer.alloca<global> align(256) %kv_size : buffer
  %v = buffer.alloca<global> align(256) %kv_size : buffer
  %info = buffer.alloca<global> align(256) %info_size : buffer
  %f = buffer.alloca<global> align(256) %ffn_size : buffer
{"  %fu = buffer.alloca<global> align(256) %ffn_size : buffer" + chr(10) if MIXED_FFN else ""}  %partials = buffer.alloca<global> align(256) %partial_size : buffer
  // 64 splits per head for decode, 16 for the prefill fallback.
  %c64 = index.constant 64 : index
  // NO_GEMM=1 (diagnostic build): %c_never keeps every epoch on the decode kernels.
  %c_never = index.constant 1000000 : index
  %c_false = index.cmp ult, %c64, %c64 : index
  %c128 = index.constant 128 : index
  %prefill_shape = index.cmp uge, %t, {"%c_never" if os.environ.get("NO_GEMM") or "attn" in DIAG else "%c64"} : index
  %pacc_dec = index.constant {QH * 64 * D * 4} : offset
  %pacc_pre = index.constant {QH * 16 * D * 4} : offset
  %pml_dec = index.constant {QH * 64 * 2 * 4} : offset
  %pml_pre = index.constant {QH * 16 * 2 * 4} : offset
  %pacc_bytes = scf.select %prefill_shape, %pacc_pre, %pacc_dec : offset
  %pml_bytes = scf.select %prefill_shape, %pml_pre, %pml_dec : offset
  %pacc_size = index.scale %t, %pacc_bytes : index, offset -> offset
  %pml_size = index.scale %t, %pml_bytes : index, offset -> offset
  %pacc = buffer.alloca<global> align(256) %pacc_size : buffer
  %pml = buffer.alloca<global> align(256) %pml_size : buffer
  %gemm = index.cmp uge, %t, {"%c_never" if os.environ.get("NO_GEMM") else "%c64"} : index
  %single = index.cmp eq, %t, %c1 : index
  %big = index.cmp uge, %t, {"%c_never" if os.environ.get("GEMM_ONE") else "%c128"} : index
  %scale_entry = index.constant 8 : offset
  %scale_size = index.scale %t, %scale_entry : index, offset -> offset
  %s_h = buffer.alloca<global> align(256) %scale_size : buffer
  %s_o = buffer.alloca<global> align(256) %scale_size : buffer
  %s_f = buffer.alloca<global> align(256) %scale_size : buffer
  %half_row = index.constant {H * 2} : offset
  %half_wide = index.constant {max(QD, GDN["VD"] if GDN else 0) * 2} : offset
  %half_ffn = index.constant {FF * 2} : offset
  %h16_size = index.scale %t, %half_row : index, offset -> offset
  %o16_size = index.scale %t, %half_wide : index, offset -> offset
  %f16_size = index.scale %t, %half_ffn : index, offset -> offset
  // 256-row GEMM tiles (_m256) when the epoch's rows are a multiple of 256.
  %c256_rows = index.constant 256 : index
  %t_rem256 = index.rem %t, %c256_rows : index
  %t_full256 = index.cmp eq, %t_rem256, %c0 : index
  %t_ge256 = index.cmp uge, %t, %c256_rows : index
  %t_never = index.cmp ult, %t, %c0 : index
  %huge = scalar.andi %t_full256, {"%t_never" if os.environ.get("GEMM_ONE") else "%t_ge256"} : i1

  %h16 = buffer.alloca<global> align(256) %h16_size : buffer
  %o16 = buffer.alloca<global> align(256) %o16_size : buffer
  %f16 = buffer.alloca<global> align(256) %f16_size : buffer
"""]
    if GATE:
        L_.append("  %gate = buffer.alloca<global> align(256) %q_size : buffer\n")
    if HEAD_GATE:
        L_.append(f"  %nQH = index.constant {QH} : index\n  %gh_bytes = index.constant {QH * 4} : offset\n"
                  "  %gh_size = index.scale %t, %gh_bytes : index, offset -> offset\n  %gh = buffer.alloca<global> align(256) %gh_size : buffer\n")
    if QKV_FUSED:
        L_.append(f"  %nQKV = index.constant {(QH + 2 * KVH) * D} : index\n  %qkv_bytes = index.constant {(QH + 2 * KVH) * D * 4} : offset\n"
                  "  %qkv_size = index.scale %t, %qkv_bytes : index, offset -> offset\n  %qkv = buffer.alloca<global> align(256) %qkv_size : buffer\n")
    if FUSED_Q:
        L_.append(f"  %qg_bytes = index.constant {2 * QD * 4} : offset\n  %qg_size = index.scale %t, %qg_bytes : index, offset -> offset\n"
                  "  %qg = buffer.alloca<global> align(256) %qg_size : buffer\n"
                  f"  %nQ2 = index.constant {2 * QD} : index\n")
    if GDN:
        g = GDN
        for name, width in (("gq", g["C"]), ("gs", g["C"]), ("gz", g["VD"]), ("graw", g["VD"]), ("ginv", 2 * g["HK"]), ("galpha", g["HV"]), ("gbeta", g["HV"])):
            L_.append(f"  %{name}_bytes = index.constant {width * 4} : offset\n  %{name}_size = index.scale %t, %{name}_bytes : index, offset -> offset\n"
                      f"  %{name} = buffer.alloca<global> align(256) %{name}_size : buffer\n")
        L_.append(f"  %nC = index.constant {g['C']} : index\n  %nVD = index.constant {g['VD']} : index\n  %nHV = index.constant {g['HV']} : index\n")
    if not QK_NORM:
        L_.append("  %no_norm = buffer.alloca<global> align(256) %row_bytes : buffer\n")
    if MOE:   # its own part: the MTP drafter programs reuse the prologue (all parts but the last), MoE layer included
        L_.append("\n" + moe_alloc())
    L_.append(f"""{"" if MOE else chr(10)}  kernel.launch @g_token_info[%t](%t, %spans, %metadata, %info) : [index](index, index, buffer, buffer)
  %list_entry = index.constant 4 : offset
  %list_size = index.scale %t, %list_entry : index, offset -> offset
  %count_size = index.constant 256 : offset
  %flash_tiles = buffer.alloca<global> align(256) %list_size : buffer
  %fallback_list = buffer.alloca<global> align(256) %list_size : buffer
  %fallback_count = buffer.alloca<global> align(256) %count_size : buffer
  scf.if %gemm {{
    kernel.launch @g_attention_fallback_list[%t](%t, %info, %flash_tiles, %fallback_list, %fallback_count) : [index](index, buffer, buffer, buffer, buffer)
  }}
  %embedding = command.parameter %weights, "token_embd.weight" : view<{esize}xi8>
  %first_norm = command.parameter %weights, "blk.0.attn_norm.weight" : view<{H}xf32>
  kernel.launch @g_embed[%t](%t, %tokens{", %metadata, %info" if VISION else ""}, %embedding, %first_norm, %x, %h) : [index](index, buffer{", buffer, buffer" if VISION else ""}, view<{esize}xi8>, view<{H}xf32>, buffer, buffer)
  %output_norm = command.parameter %weights, "output_norm.weight" : view<{H}xf32>
""")
    gate0 = "%gate" if GATE else "%q"
    attn_ordinal = {i: n for n, i in enumerate(j for j in range(L) if j not in RECUR)}
    gdn_ordinal = {i: n for n, i in enumerate(sorted(RECUR))}
    def layer_text(i, ordinal=None, last_norm=None, kind_note=None):
        """Text of trunk layer i (tensors blk.i.*). ordinal overrides its attention KV-plane ordinal and
        last_norm names the norm applied to its output (default: next layer's attn_norm / output_norm)."""
        b = f"blk.{i}."
        nm = lambda n: b + n + ".weight"
        sz = lambda n: tensor_bytes(nm(n))
        par = lambda var, n, ty=None: f'    %{var}_{i} = command.parameter %weights, "{nm(n)}" : {ty or f"view<{sz(n)}xi8>"}\n'
        nq, nqh, qd = ("%nQ", "%nQH", QD) if LH[i] == QH else (f"%nQ{LH[i]}", f"%nQH{LH[i]}", LH[i] * D)
        kind = kind_note or (" (gated delta-net)" if i in RECUR else (" (sliding)" if i in sliding else "") + (" (no RoPE)" if i in nope else ""))
        T = [f"  // ---- layer {i}{kind}\n  scf.if %gemm {{\n    kernel.launch @g_row_scale_f16[%t](%t, %nH, %h, %s_h, %h16) : [index](index, index, buffer, buffer, buffer)\n  }}\n",
             f"  %layer_{i} = index.constant {attn_ordinal.get(i, gdn_ordinal.get(i)) if ordinal is None else ordinal} : index\n  %sliding_{i} = scalar.constant {1 if i in sliding else 0} : i32\n  %rope_{i} = scalar.constant {0 if i in nope else 1} : i32\n"]
        if i not in MOE_LAYERS:
            T += [par("w_gate", "ffn_gate"), par("w_up", "ffn_up"), par("w_down", "ffn_down")]
        T += [par("ffn_norm", FFN_NORM, f"view<{H}xf32>")]
        if SINKS:
            T.append(par("sinks", "attn_sinks", f"view<{QH}xf32>"))
        if i in RECUR:
            T.append(gdn_layer(i, b, nm, sz, par))
        elif QKV_FUSED:
            T += [par("wqkv", "attn_qkv"), par("wo", "attn_output")]
        else:
            T += [par("wq", "attn_q"), par("wk", "attn_k"), par("wv", "attn_v"), par("wo", "attn_output")]
        if i in RECUR:
            pass
        elif GATE and not FUSED_Q:
            T.append(par("wg", "attn_gate"))
        if QK_NORM and i not in RECUR:
            T += [par("q_norm", "attn_q_norm", f"view<{D}xf32>"), par("k_norm", "attn_k_norm", f"view<{D}xf32>")]
        bias_par, bias_args, bias_types = rope_bias(b, i, "    ") if i not in RECUR else ("", "", "")
        T.append(bias_par)
        if POST_NORM:
            T += [par("post_attn", "post_attention_norm", f"view<{H}xf32>"), par("post_ffn", "post_ffw_norm", f"view<{H}xf32>")]
        if FUSED_Q and i not in RECUR:
            T.append("  " + lin(fmt_of(nm("attn_q")), "%h", "%s_h", f"%wq_{i}", sz("attn_q"), "%qg", "%nH", "%nQ2", H) + "\n")
            T.append("  kernel.launch @g_split_qgate[%t](%t, %qg, %q, %gate) : [index](index, buffer, buffer, buffer)\n")
        # same format and size: one dual-weight launch at 8/16 tokens (PAIR_PROJ)
        same = lambda a, b: fmt_of(nm(a)) == fmt_of(nm(b)) and sz(a) == sz(b)
        qg = GATE and not FUSED_Q and not HEAD_GATE and not QKV_FUSED and i not in RECUR and same("attn_q", "attn_gate") and PAIR_PROJ in ("1", "qg")
        if i in RECUR or FUSED_Q:
            pass
        elif QKV_FUSED:
            T.append("  " + lin(fmt_of(nm("attn_qkv")), "%h", "%s_h", f"%wqkv_{i}", sz("attn_qkv"), "%qkv", "%nH", "%nQKV", H) + "\n")
            T.append("  kernel.launch @g_split_qkv[%t](%t, %qkv, %q, %k, %v) : [index](index, buffer, buffer, buffer, buffer)\n")
        elif qg:
            T.append("  " + lin2(fmt_of(nm("attn_q")), "%h", "%s_h", f"%wq_{i}", f"%wg_{i}", sz("attn_q"), "%q", "%gate", "%nH", "%nQ", H) + "\n")
        else:
            T.append("  " + lin(fmt_of(nm("attn_q")), "%h", "%s_h", f"%wq_{i}", sz("attn_q"), "%q", "%nH", nq, H) + "\n")
        if i in RECUR or QKV_FUSED:
          pass
        elif same("attn_k", "attn_v") and PAIR_PROJ in ("1", "kv"):
          T.append("  " + lin2(fmt_of(nm("attn_k")), "%h", "%s_h", f"%wk_{i}", f"%wv_{i}", sz("attn_k"), "%k", "%v", "%nH", "%nKV", H) + "\n")
        else:
          T.append("  " + lin(fmt_of(nm("attn_k")), "%h", "%s_h", f"%wk_{i}", sz("attn_k"), "%k", "%nH", "%nKV", H) + "\n")
          T.append("  " + lin(fmt_of(nm("attn_v")), "%h", "%s_h", f"%wv_{i}", sz("attn_v"), "%v", "%nH", "%nKV", H) + "\n")
        if HEAD_GATE and i not in RECUR:   # one logit per query head (n = QH: always the matvec kernels)
            T.append("  " + lin_mv(fmt_of(nm("attn_gate")), "%h", f"%wg_{i}", sz("attn_gate"), "%gh", nqh, H) + "\n")
            T.append("  kernel.launch @g_gate_heads[%t](%t, %gh, %gate) : [index](index, buffer, buffer)\n")
        elif GATE and not FUSED_Q and i not in RECUR and not qg:
            T.append("  " + lin(fmt_of(nm("attn_gate")), "%h", "%s_h", f"%wg_{i}", sz("attn_gate"), "%gate", "%nH", "%nQ", H) + "\n")
        qn, kn = (f"%q_norm_{i}", f"%k_norm_{i}") if QK_NORM else ("%no_norm", "%no_norm")
        gate, gty = (f"%sinks_{i}", f"view<{QH}xf32>") if SINKS else (gate0, "buffer")
        nty = f"view<{D}xf32>" if QK_NORM else "buffer"
        if i not in RECUR:
          rope_k = "g_qk_rope_store_swa" if SWA_ROPE and i in sliding else "g_qk_rope_store"
          T.append(f"""  kernel.launch @{rope_k}[%t](%t, %layer_{i}, %rope_{i}, %context, %pool, %q, %k, %v, {qn}, {kn}, %info, %origins, %arena{", %metadata, %tokens" if MROPE else ""}{bias_args}) : [index](index, index, i32, index, index, buffer, buffer, buffer, {nty}, {nty}, buffer, buffer, buffer{", buffer, buffer" if MROPE else ""}{bias_types})
  scf.if {"%c_false" if "attn" in DIAG else "%gemm"} {{
    kernel.launch @g_attention_flash2[%t](%t, %layer_{i}, %sliding_{i}, %context, %pool, %q, {gate}, %info, %flash_tiles, %origins, %arena, %o) : [index](index, index, i32, index, index, buffer, {gty}, buffer, buffer, buffer, buffer, buffer)
    kernel.launch @g16_attention_split_listed[%fallback_count](%t, %fallback_list, %layer_{i}, %sliding_{i}, %context, %pool, %q, %info, %origins, %arena, %pacc, %pml) : [buffer](index, buffer, index, i32, index, index, buffer, buffer, buffer, buffer, buffer, buffer)
    kernel.launch @g16_attention_combine_listed[%fallback_count](%t, %fallback_list, %sliding_{i}, %info, %pacc, %pml, {gate}, %o) : [buffer](index, buffer, i32, buffer, buffer, buffer, {gty}, buffer)
  }} else {{
    kernel.launch @g_attention_split[%t](%t, %layer_{i}, %sliding_{i}, %context, %pool, %q, %info, %origins, %arena, %pacc, %pml) : [index](index, index, i32, index, index, buffer, buffer, buffer, buffer, buffer, buffer)
    kernel.launch @g_attention_combine[%t](%t, %sliding_{i}, %info, %pacc, %pml, {gate}, %o) : [index](index, i32, buffer, buffer, buffer, {gty}, buffer)
  }}
  scf.if %gemm {{
    kernel.launch @g_row_scale_f16[%t](%t, {nq}, %o, %s_o, %o16) : [index](index, index, buffer, buffer, buffer)
  }}
""")

        if i not in RECUR:
          T.append("  " + lin(fmt_of(nm("attn_output")), "%o", "%s_o", f"%wo_{i}", sz("attn_output"), "%y", nq, "%nH", qd) + "\n")
          if OUT_BIAS:   # attention output projection bias (gpt-oss attn_output.bias)
            T.append(f'  %bo_{i} = command.parameter %weights, "{b}attn_output.bias" : view<{H}xf32>\n'
                     f"  kernel.launch @g_bias_add[%t](%t, %nH, %y, %bo_{i}) : [index](index, index, buffer, view<{H}xf32>)\n")
        pa = f"%post_attn_{i}" if POST_NORM else f"%ffn_norm_{i}"
        T.append(f"  kernel.launch @g_post_add_norm[%t](%t, %y, {pa}, %x, %ffn_norm_{i}, %h) : [index](index, buffer, view<{H}xf32>, buffer, view<{H}xf32>, buffer)\n")
        T.append(f"  scf.if %gemm {{\n    kernel.launch @g_row_scale_f16[%t](%t, %nH, %h, %s_h, %h16) : [index](index, index, buffer, buffer, buffer)\n  }}\n")
        if i in MOE_LAYERS:
          T.append(moe_layer(i, b, nm, sz, par))
        else:
          fg, fu = fmt_of(nm("ffn_gate")), fmt_of(nm("ffn_up"))
          if fg == fu:
              T.append("  " + lin(fg, "%h", "%s_h", (f"%w_gate_{i}", f"%w_up_{i}"), sz("ffn_gate"), "%f", "%nH", nff(i), H, mode="swiglu") + "\n")
          else:   # gate and up in different formats (unsloth UD mixes): two projections, then silu(gate) * up
              T.append("  " + lin(fg, "%h", "%s_h", f"%w_gate_{i}", sz("ffn_gate"), "%f", "%nH", nff(i), H) + "\n")
              T.append("  " + lin(fu, "%h", "%s_h", f"%w_up_{i}", sz("ffn_up"), "%fu", "%nH", nff(i), H) + "\n")
              T.append(f"  kernel.launch @g_silu_mul[%t](%t, {nff(i)}, %f, %fu) : [index](index, index, buffer, buffer)\n")
          T.append(f"  scf.if %gemm {{\n    kernel.launch @g_row_scale_f16[%t](%t, {nff(i)}, %f, %s_f, %f16) : [index](index, index, buffer, buffer, buffer)\n  }}\n")
          T.append("  " + lin(fmt_of(nm("ffn_down")), "%f", "%s_f", f"%w_down_{i}", sz("ffn_down"), "%y", nff(i), "%nH", FFL[i]) + "\n")
        pf = f"%post_ffn_{i}" if POST_NORM else f"%ffn_norm_{i}"
        if last_norm:
            nn = last_norm
        elif i == L - 1:
            nn = "%output_norm"
        else:
            T.append(f'  %next_norm_{i} = command.parameter %weights, "blk.{i + 1}.attn_norm.weight" : view<{H}xf32>\n')
            nn = f"%next_norm_{i}"
        T.append(f"  kernel.launch @g_post_add_norm[%t](%t, %y, {pf}, %x, {nn}, %h) : [index](index, buffer, view<{H}xf32>, buffer, view<{H}xf32>, buffer)\n")
        text = "".join(T).replace("\n    %", "\n  %").replace("\n      ", "\n    ")
        if LH[i] != QH:   # this layer's attention kernels: the variants for its head count
            import re
            text = re.sub(NARROW_KERNELS, rf"@\1_h{LH[i]}[", text)
        return text

    MODEL_PARTS.update(layer_text=layer_text, prologue="".join(L_[:-1]), attn_ordinal=attn_ordinal)  # [-1]: token info + embed
    for i in range(L):
        L_.append(layer_text(i))
    L_.append(f"""
  %head = command.parameter %weights, "{head_name}" : view<{hsize}xi8>
  %one_span = index.cmp eq, %spans, %c1 : index
  scf.if %one_span {{
    kernel.launch @g_head_mv[%spans](%spans, %t, %metadata, %h, %head, %partials, %logits) : [index](index, index, buffer, buffer, view<{hsize}xi8>, buffer, buffer)
  }} else {{
    kernel.launch @g_head_mv_t4[%spans](%spans, %t, %metadata, %h, %head, %partials, %logits) : [index](index, index, buffer, buffer, view<{hsize}xi8>, buffer, buffer)
  }}
  kernel.launch @g_head_select[%spans](%spans, %metadata, %partials, %logits, %outputs) : [index](index, buffer, buffer, buffer, buffer)
  command.return
}}
""")
    return "".join(L_), head_fmt


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.environ.get("LLM_OUT") or os.path.join(here, "out")
    os.makedirs(out_dir, exist_ok=True)
    program, head_fmt = model()  # first: records the mv / GEMM variants it launches
    program = deepstack_wiring(program) if DEEPSTACK else program
    program = last_out_select_program(program) if LAST_OUT else program
    spec_kernels = ""
    if gen_draft.enabled():   # before the kernel list is emitted: these programs record mv / GEMM variants too
        head_name = "output.weight" if "output.weight" in TENS else "token_embd.weight"
        gfacts = dict(H=H, HG=HG, EPS=EPS, D=D, KD=KVH * D, QK_NORM=QK_NORM, VISION=bool(VISION), MROPE=bool(MROPE),
                      GDN=GDN, NG=len(RECUR), gdn_layers=[(i, n) for n, i in enumerate(sorted(RECUR))],
                      head_name=head_name, head_size=tensor_bytes(head_name), esize=tensor_bytes("token_embd.weight"),
                      prologue=MODEL_PARTS.get("prologue"), layer_text=MODEL_PARTS.get("layer_text"), lin=lin, lin2=lin2, pair_kv=PAIR_PROJ in ("1", "kv"),
                      fmt=fmt_of, size=tensor_bytes, mtp_layer=MTP_LAYER, replay=MODEL_PARTS.get("replay", []), rope_bias=rope_bias)
        if gen_draft.DRAFT["kind"] == "dflash":
            import gen_dflash
            dcfg = gen_dflash.config(gen_draft.DRAFT["gguf"])
            assert dcfg["hidden"] == H, f"drafter hidden {dcfg['hidden']} != target {H}"
            # BF16/F16 drafter matrices are converted to Q8_0 at load (gen_dflash.prep_*, DRAFTER_PREP=0 keeps them)
            import prep_weights
            DPREP = gen_dflash.prep_tensors(dcfg["tensors"])
            DT = dcfg["tensors"] = gen_dflash.apply_prep(dcfg["tensors"], DPREP)
            if DPREP:
                open(os.path.join(out_dir, "draft_weights.loom"), "w", newline="\n").write(prep_weights.policy(DPREP, "the drafter"))
            # The proposals' head pass reads a load-time Q4_0 copy of the target's head (PROPOSE_HEAD=off: the head)
            HPREP = {}
            if os.environ.get("PROPOSE_HEAD", "q4_0") == "q4_0" and fmt_of(head_name) in prep_weights.SRC and ACT_Q8 and H % 32 == 0:
                HPREP = {head_name: (fmt_of(head_name), "q4_0", V * H // 32)}
                open(os.path.join(out_dir, "head_weights.loom"), "w", newline="\n").write(prep_weights.policy(HPREP, "the proposal head"))
                gfacts["phead"] = prep_weights.dst_bytes("q4_0", V * H // 32)
            def dsize(name):
                t = DT[name]
                n = 1
                for d in t["shape"]:
                    n *= d
                return n * 4 if t["type"] == "f32" else n // SB[t["type"]] * FMT[t["type"]]
            gfacts.update(dflash=dcfg, dflash_taps=[t - 1 for t in dcfg["target_layers"]], L=L, CHUNKS=CHUNKS, V=V,
                          allreduce=allreduce, ALLRED_SETUP=ALLRED_SETUP,
                          row_bytes=row_bytes, FMT=FMT, SB=SB, dfmt=lambda n: DT[n]["type"], dsize=dsize,
                          EFMT=EFMT, embed_plain=not EMBED_NORM and not _ES)
        if gen_draft.DRAFT["kind"] == "assist":
            import gen_assist
            acfg = gen_assist.config(gen_draft.DRAFT["gguf"])
            assert acfg["H"] == H and acfg["vocab"] == V, f"assistant out {acfg['H']} / vocab {acfg['vocab']} != target {H} / {V}"
            AT = acfg["tensors"]
            def asize(name):
                t = AT[name]
                n = 1
                for d in t["shape"]:
                    n *= d
                return n * 4 if t["type"] == "f32" else n // SB[t["type"]] * FMT[t["type"]]
            gfacts.update(assist=acfg, QH=QH, KVH=KVH, V=V, EFMT=EFMT, row_bytes=row_bytes, FMT=FMT, SB=SB,
                          afmt=lambda n: AT[n]["type"], asize=asize,
                          assist_attention=gen_assist.target_attention(program, L, L - 2, L - 1))
        spec = [gen_draft.advance_program(program, gfacts), gen_draft.verify_program(program, gfacts)]
        if MTP_LAYER is not None:
            spec.append(gen_draft.mtp_programs(gfacts))
        if gen_draft.DRAFT["kind"] == "dflash":
            spec.append(gen_dflash.programs(gfacts))
        if gen_draft.DRAFT["kind"] == "ngram":
            import gen_ngram
            spec.append(gen_ngram.programs(gfacts))
        if gen_draft.DRAFT["kind"] == "assist":
            spec.append(gen_assist.programs(gfacts))
        # 5-63-token epochs (depth > 3, or several drafting streams) take the f16 WMMA GEMM: one weight pass per step
        # (4 drafting streams at depth 3: 69 vs 146 ms per epoch on Bolt-E4B), at the price of rare near-tie flips vs
        # 1-token decode. SPEC_EXACT=1 keeps the 4-token matvec (decode's rounding) at every width instead.
        # ACT_Q8: the int8 matvec streams weights once per 16 tokens, so the GEMM takes over later (SPEC_MM_FROM).
        mm_from = int(os.environ.get("SPEC_MM_FROM", "17" if USED_MVQ else "5"))
        program = "\n".join(spec) if os.environ.get("SPEC_EXACT") == "1" else gen_draft.small_gemm("\n".join(spec), mm_from)
        # The drafter's catch-up (warm) program only feeds proposals, never the output, so it takes the GEMM from
        # DRAFT_MM_FROM tokens (4 drafting streams commit ~10 tokens per step: DFlash's 33280-wide fc on the f32 4-token
        # matvec re-read its weights per 4 tokens; Muse-Glimmer c4 epoch 161.5 -> 157.6 ms). Proposals keep the int8
        # matvec (the GEMM from 5 there: 161.5 -> 173.4 ms).
        import re as _re
        _wm = _re.search(r"command\.program\.def public @llm_\w+_warm\(\).*?\n\}\n", program, _re.S)
        if _wm and "%c5_mm = index.constant" in _wm.group(0):
            program = program[:_wm.start()] + _re.sub(r"%c5_mm = index\.constant \d+ : index", "%c5_mm = index.constant "
                      + os.environ.get("DRAFT_MM_FROM", "5") + " : index", _wm.group(0), count=1) + program[_wm.end():]
        spec_kernels = gen_draft.kernels(gfacts)
        if gen_draft.DRAFT["kind"] == "dflash":
            spec_kernels += gen_dflash.kernels(gfacts) + prep_weights.library({(a, b) for a, b, _ in list(DPREP.values()) + list(HPREP.values())})
        if gen_draft.DRAFT["kind"] == "ngram":
            spec_kernels += gen_ngram.kernels(gfacts)
        if gen_draft.DRAFT["kind"] == "assist":
            spec_kernels += gen_draft.mtp_kernels(gfacts) + gen_assist.kernels(gfacts)
    if any(t["type"] == "q5_0" for t in TENS.values()):
        COMMON += Q5_0
    if any(t["type"] == "mxfp4" for t in TENS.values()):
        COMMON += MXFP4
    if USED_MVQ:   # ACT_Q8: int8-activation matvecs, their weight-code helpers and g_quant_q8
        COMMON += ICODES + "".join(mv(fmt, K, R=R, TG=TG, swiglu=sw == 1, pair=sw == 2, suffix=f"_r{R}" if R != 16 else "", iq=True)
                                   for fmt, K, sw, R, TG in sorted(USED_MVQ))
        program = quant_wiring(program)
        if head_fmt in IQ_FMTS:
            COMMON += "".join(head_mv(tg, head_fmt, iq=True) for tg in (1, 4, 8, 16))
            program = head_wiring(program)
            if spec_kernels and "phead" in gfacts:   # the DFlash proposals' head: the Q4_0 copy's variants
                COMMON += "".join(head_mv(tg, "q4_0", iq=True, suffix="_p40") for tg in (1, 4, 8, 16))
                _pm = _re.search(r"command\.program\.def public @llm_dflash_propose\(\).*?\n\}\n", program, _re.S)
                program = program[:_pm.start()] + _re.sub(r"@(g_head_mv\w*_q8)\[", r"@\1_p40[", _pm.group(0)) + program[_pm.end():]
    if os.environ.get("ROW_SCALE_LIVE", "1") != "0":   # after quant_wiring (it matches g_row_scale_f16); =0 A/B it off
        program = live_row_scale(program)
        COMMON += ROW_SCALE_LIVE
    if os.environ.get("POST_ADD_ROW_SCALE", "1") != "0":   # after live_row_scale; =0 A/B it off
        program, _fused = fuse_post_add_row_scale(program)
        COMMON += _fused
    if any(f == "f32" for f, *_ in USED_MV | USED_MV_R) or "f32" in USED_GEMM:
        COMMON += F32_VALUES
    if any(f == "f16" for f, *_ in USED_MV | USED_MV_R) or "f16" in USED_GEMM:
        COMMON += F16_VALUES
    if (any(f == "bf16" for f, *_ in USED_MV | USED_MV_R) or "bf16" in USED_GEMM) and "@g_bf16_values(" not in COMMON:
        COMMON += BF16_VALUES
    if {f for f, *_ in USED_MV | USED_MV_R} & LOWBIT or set(USED_GEMM) & LOWBIT or any(t["type"] in LOWBIT for t in TENS.values()):
        COMMON += "\n" + open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "quant_values.loom")).read()
    parts = [f"// Generated by gen_llm.py from {os.environ.get('LLM_CONFIG') or sys.argv[1]}.\n", COMMON.replace("%c8_q6", "%c8q6").replace(
        "  %c64 = index.constant 64 : index\n  %qh_base", "  %c64 = index.constant 64 : index\n  %c8q6 = index.constant 8 : index\n  %qh_base")]
    for TG in (1, 4):
        for fmt, K, sw in sorted((f, k, w) for f, k, w, t in USED_MV if t == TG):
            parts.append(mv(fmt, K, TG=TG, swiglu=sw))
    for fmt, K, sw, R, TG in sorted(USED_MV_R):
        parts.append(mv(fmt, K, R=R, TG=TG, swiglu=sw, suffix=f"_r{R}"))
    parts += [NORMS, ROW_SCALE, qk_rope(), attention().replace("%not_uniform = scalar.xori %uniform, %true : i1", "%not_uniform = scalar.ori %uniform, %true : i1") if NO_FLASH else attention(),
              head(), head_mv(1, head_fmt), head_mv(4, head_fmt)]
    if FUSED_Q:
        parts.append(split_qgate(QH, D))
    if SWA_ROPE:
        parts.append(qk_rope_swa())
    if QKV_FUSED:
        parts.append(split_qkv())
    if HEAD_GATE:
        parts.append(gate_heads())
    for h in NARROW:
        parts.append(with_heads(h, lambda: "".join(qk_rope_swa() if swa and SWA_ROPE else qk_rope() for hh, swa in sorted(NARROW_ROPE) if hh == h)
                                + attention()[:-len(FALLBACK)] + (gate_heads() if HEAD_GATE else "")))
    if RECUR:
        parts.append(gdn_kernels(CFG, EPS))
    if spec_kernels:
        parts.append(spec_kernels)
    if MIXED_FFN:   # GEGLU: gelu(gate) * up, on 4-value groups (bounds provable without the launch: kernel-testable)
        parts.append(gelu_vectors(geglu_mul()) if GELU else SILU_MUL)
    if OUT_BIAS:
        parts.append(BIAS_ADD)
    if MOE:
        import importlib.util
        _spec = importlib.util.spec_from_file_location("gen_gemm_mod", os.path.join(here, "gen_gemm.py"))
        _gg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_gg)
        parts.append(moe_kernels(ROW_SCALE, _gg.gemm))
    env = dict(os.environ, GEMM_FORMATS=",".join(f"{f}:{'+'.join(sorted(m))}" for f, m in sorted(USED_GEMM.items())))
    # Prefill GEMM switches, ON by default; set GEMM_N_FASTEST=0 / GEMM_SKIP_PADDING=0 to A/B them off.
    env.setdefault("GEMM_SKIP_PADDING", "1"); env.setdefault("GEMM_N_FASTEST", "1")
    gemm_text = subprocess.run([sys.executable, os.path.join(here, "gen_gemm.py")], capture_output=True, text=True, check=True, env=env).stdout
    parts.append(gelu_vectors(gemm_text) if GELU else gemm_text)
    parts.append(open(os.path.join(here, "tile_uniform.loom")).read())
    if not NO_FLASH:
        flash2 = os.path.join(out_dir, "flash2.loom")
        subprocess.run([sys.executable, os.path.join(here, "gen_flash2.py")], check=True,
                       env=dict(os.environ, FLASH2_TARGET="llm", FLASH2_OUT=flash2, LLM_CONFIG=os.environ.get("LLM_CONFIG") or sys.argv[1]))
        parts.append(open(flash2).read())
        for h in NARROW:   # narrower layers' flash prefill: the same generator at their head count
            hcfg, hout = (os.path.join(out_dir, f"flash2_h{h}.{x}") for x in ("json", "loom"))
            json.dump(dict(CFG, heads=h), open(hcfg, "w"))
            subprocess.run([sys.executable, os.path.join(here, "gen_flash2.py")], check=True,
                           env=dict(os.environ, FLASH2_TARGET="llm", FLASH2_OUT=hout, LLM_CONFIG=hcfg))
            parts.append(with_heads(h, lambda: open(hout).read()))
            os.remove(hcfg)
    parts.append(program)
    text = "\n".join(parts)
    if "_attention_flash2_deep\")" in text:   # depth dispatch (gen_flash2.py): its deep pairs' kernel after each launch
        text = re.sub(r"^( *)kernel\.launch @(\w+)_attention_flash2(_h\d+)?\[(.*)$",
                      r"\g<0>\n\1kernel.launch @\2_attention_flash2_deep\3[\4",
                      text, flags=re.M)
    if os.environ.get("LIVE_ROW_SKIP", "1") != "0":   # =0 A/B it off
        text = live_row_skip(text)
    if seg_tokens():
        text = seg_attention(text)
    out = os.path.join(out_dir, "model.loom")
    open(out, "w", newline="\n").write(text)
    print("wrote", out)
