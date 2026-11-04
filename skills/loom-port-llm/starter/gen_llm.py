# Generates llm.loom: kernel library + forward-pass command program for a dense
# decoder-only LLM stored as GGUF, driven by a config JSON from inspect_gguf.py.
#
# Generalized from the Muse Glimmer 30B port; end-to-end validated (llama.cpp
# golden, 1k-32k prompts) on Qwen3-0.6B. Supported:
#   * head_dim 128 or 256, any query/KV head counts with heads % kv_heads == 0
#   * GGUF weight formats q4_k, q5_k, q6_k, q8_0 (mixed per tensor is fine)
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
    _w = CFG["tensors"][f"blk.{MTP_LAYER}.ffn_gate.weight"]["shape"][1]
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
assert D in (128, 256), "attention kernels support head_dim 128 or 256 (lane l owns dims l, l+32, ...)"
DL = D // 32                              # head dims per lane in the attention kernels
RD = CFG.get("rope_dims") or D            # rotary dims (partial RoPE rotates the first RD dims)
assert RD % 64 == 0 and RD <= D, "rope_dims must be a multiple of 64"
assert QH % KVH == 0 and G <= 16, "need heads % kv_heads == 0 and at most 16 query heads per KV head"
assert HG <= 256 and H % 32 == 0, "hidden must be a multiple of 32 and at most 8192"
UNROLL = 16
POS_BYTES = KVH * D * 2                   # F16 K (or V) bytes per position
WINDOW = CFG.get("sliding_window") or 0   # keys kept by sliding layers (0: none)
GATE = CFG.get("attn_gate", False)       # True: separate attn_gate tensor; "fused_q": attn_q packs [q | gate] per head
FUSED_Q = GATE == "fused_q"
RECUR = set(CFG.get("recurrent_layers", []))  # gated delta-net layers (hybrid models)
GDN = gdn_geometry(CFG) if RECUR else None
FFN_NORM = CFG.get("ffn_norm_tensor", "ffn_norm")
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
ROPE = CFG.get("rope_type", "neox")       # "neox": pairs (d, d+64); "norm": pairs (2d, 2d+1)
THETA = CFG["rope_theta"]
TENS = CFG["tensors"]                     # tensor name -> {"type": fmt, "shape": [...]}

FMT = {  # bytes per superblock of SB[fmt] values
    "q4_k": 144, "q5_k": 176, "q6_k": 210, "q8_0": 34, "f32": 128,
    # low-bit formats of unsloth UD / bartowski quants (helpers in quant_values.loom, appended when used)
    "q4_0": 18, "iq4_nl": 18, "iq4_xs": 136, "q3_k": 110, "iq3_s": 110,
    "f16": 64,   # small F16 projections of unsloth UD quants (Qwen3.5 ssm_alpha/beta): @g_f16_values, emitted when used
}
SB = {"q4_k": 256, "q5_k": 256, "q6_k": 256, "q8_0": 32, "f32": 32,
      "q4_0": 32, "iq4_nl": 32, "iq4_xs": 256, "q3_k": 256, "iq3_s": 256, "f16": 32}
LOWBIT = {"q4_0", "iq4_nl", "iq4_xs", "q3_k", "iq3_s"}


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
    if t["type"] == "f16":
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

# emitted only when an F32 matrix is used (see main)
F32_VALUES = r"""
// F32: 32 values (128 bytes) of a row (small projections some GGUFs keep unquantized).
func.def inline @g_f32_values(%w: buffer, %origin: offset, %group: index) -> (vector<32xf32>) {
  %view = buffer.view %w[%origin] : buffer -> view<32xf32>
  %values = vector.load %view[0] : view<32xf32> -> vector<32xf32>
  func.return %values : vector<32xf32>
}
"""


def mv(fmt, K, R=16, TG=1, swiglu=False, suffix=""):
    """Decode matrix-vector: a workgroup covers one K row with one lane per
    32-value group (lanes = groups rounded up to a wave), R output rows and TG
    tokens. Each lane keeps its x group for the TG tokens in registers and
    decodes each weight group once for all of them, so weights stream once per
    token group and activations once per workgroup. Waves reduce with shuffles;
    one barrier folds the wave partials."""
    groups = K // 32
    if groups > 1024:
        return mv_wide(fmt, K, R, TG, swiglu, suffix)
    lanes = (groups + 31) // 32 * 32
    waves = lanes // 32
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
  %c8 = index.constant 8 : index
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
  %ok = index.cmp ult, %workitem, %groups : index
  %g_last = index.constant {groups - 1} : index
  %g_c = index.min %workitem, %g_last : index
  %g = index.assume %g_c [range(%g_c, 0, {groups - 1})] : index
  %cgpb = index.constant {SB[fmt] // 32} : index
  %blk = index.div %g, %cgpb : index
  %within = index.rem %g, %cgpb : index
  %ba = index.scale %blk, %block_bytes : index, offset -> offset"""]
    for j in range(TG):
        L.append(f"""  %tj{j}_off = index.constant {j} : index
  %tok{j}_raw = index.add %token0, %tj{j}_off : index
  %live{j} = index.cmp ult, %tok{j}_raw, %tokens : index
  %use{j} = scalar.andi %live{j}, %ok : i1
  %x{j} = scf.if %use{j} -> (vector<32xf32>) {{
    %tk{j} = index.assume %tok{j}_raw [lt(%tok{j}_raw, %tokens)] : index
    %v = vector.load %x_view[%tk{j}, %g, %c0] : view<[%tokens]x{groups}x32xf32> -> vector<32xf32>
    scf.yield %v : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}
  %xs{j} = vector.reduce<addf> %x{j}, %zero_f : vector<32xf32>, f32""")
    L.append(f"""  %part_bytes = index.constant {waves * R * TG * nw * 4} : offset
  %parts = buffer.alloca<workgroup> align(16) %part_bytes : buffer
  %pv = buffer.view %parts[%zero] : buffer -> view<{nw}x{waves}x{R}x{TG}xf32>
  %first = index.cmp eq, %lane, %c0 : index
  scf.for %r = [%c0 to %rows_per step %c1] {{
    %row_value = index.add %row_base, %r : index
    %row = index.assume %row_value [lt(%row_value, %n)] : index
    %ro = index.scale %row, %row_bytes : index, offset -> offset
    %rb = index.assume %r [range(%r, 0, {R - 1})] : index
    %o = index.add %ro, %ba : offset""")
    for wi, w in enumerate(["w", "w2"] if swiglu else ["w"]):
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
    L.append("  }")
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
  %x_view = buffer.view %x[%zero] : buffer -> view<[%rows]x[%k]xf32>
  %h_view = buffer.view %x16[%zero] : buffer -> view<[%rows]x[%k]xf16>
  %local = scf.for %c = [%lane to %k step %c256](%m = %zero_f : f32) -> (f32) {
    %bc = index.assume %c [lt(%c, %k)] : index
    %v = view.load %x_view[%row, %bc] : view<[%rows]x[%k]xf32> -> f32
    %a = scalar.absf %v : f32
    %n = scalar.maxnumf %m, %a : f32
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
  scf.for %c = [%lane to %k step %c256] {
    %bc = index.assume %c [lt(%c, %k)] : index
    %v = view.load %x_view[%row, %bc] : view<[%rows]x[%k]xf32> -> f32
    %sv = scalar.mulf %v, %scale : f32
    %hv = scalar.fptrunc %sv : f32 to f16
    view.store %hv, %h_view[%row, %bc] : f16, view<[%rows]x[%k]xf16>
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


def qk_rope():
    """Per-head (optional) RMSNorm + RoPE for QH query and KVH key heads, with
    keys/values stored to the paged F16 KV planes: one 256-lane workgroup per
    token. Lane l of a wave owns D/32 of a head's dims: NeoX: l, l+32, ...
    with rotary pairs (d, d + RD/2) over the first RD (rope_dims) dims and the
    rest passed through; interleaved ("norm", D = 128 only): 4l..4l+3 with
    pairs (0,1),(2,3). Each pair's angle is computed once for every head."""
    n = DL
    if ROPE == "neox":
        dims = ["%lane"] + [f"%d{j}" for j in range(1, n)]
        dim_defs = "".join(f"  %d{j}c = index.constant {32 * j} : index\n  %d{j} = index.add %lane, %d{j}c : index\n" for j in range(1, n))
        half = RD // 64  # pair offset in lanes-of-32 units
        freq_idx = [dims[j] for j in range(half)]
        pairs = [(j, j + half, j) for j in range(half)]
        ln_scale = 2.0 * math.log(THETA) / RD   # angle = position * exp(-d * 2 ln(theta) / RD)
    else:
        assert D == 128 and RD == 128, "interleaved RoPE is implemented for head_dim 128"
        dims = ["%e0", "%e1", "%e2", "%e3"]
        dim_defs = "  %c1 = index.constant 1 : index\n  %e0 = index.mul %lane, %c4 : index\n  %e1 = index.add %e0, %c1 : index\n  %e2 = index.add %e0, %c2 : index\n  %e3 = index.add %e0, %c3 : index\n  %f0 = index.mul %lane, %c2 : index\n  %f1 = index.add %f0, %c1 : index\n"
        freq_idx = ["%f0", "%f1"]; pairs = [(0, 1, 0), (2, 3, 1)]
        ln_scale = math.log(THETA) / 64.0
    yt = ", ".join(["f32"] * n)
    xs = ", ".join(f"%x{i}" for i in range(n))
    t = [f"""
kernel.def export("g_qk_rope_store") @g_qk_rope_store(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %layer_arg: index, %rope: i32, %context_arg: index, %pool_arg: index, %q: buffer, %k: buffer, %v: buffer, %q_norm: buffer, %k_norm: buffer, %info: buffer, %origins: buffer, %arena: buffer{", %meta: buffer, %ids: buffer" if MROPE else ""}) {{
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
  %if{j} = scalar.expf<afn> %ex{j}b : f32
  %ra{j} = scalar.mulf {"%mr_p" + str(j) if MROPE else "%position_f"}, %if{j} : f32
  %tr{j} = scalar.mulf %ra{j}, %inv_two_pi : f32
  %ts{j} = scalar.addf %tr{j}, %half : f32
  %ti{j} = scalar.fptosi %ts{j} : f32 to i32
  %tf{j} = scalar.sitofp %ti{j} : i32 to f32
  %wr{j} = scalar.mulf %tf{j}, %two_pi : f32
  %an{j} = scalar.subf %ra{j}, %wr{j} : f32
  %cr{j} = scalar.cosf<afn> %an{j} : f32
  %sr{j} = scalar.sinf<afn> %an{j} : f32
  %cos{j} = scf.select %apply_rope, %cr{j}, %one_f : f32
  %sin{j} = scf.select %apply_rope, %sr{j}, %zero_f : f32""")
    t.append(f"""  %q_view = buffer.view %q[%zero] : buffer -> view<[%tokens]x{QH}x{D}xf32>
  %k_view = buffer.view %k[%zero] : buffer -> view<[%tokens]x{KVH}x{D}xf32>
  %v_view = buffer.view %v[%zero] : buffer -> view<[%tokens]x{KVH}x{D}xf32>
  %qn_view = buffer.view %q_norm[%zero] : buffer -> view<{D}xf32>
  %kn_view = buffer.view %k_norm[%zero] : buffer -> view<{D}xf32>""")
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
        t.append(f"      %x{i} = view.load %q_view[%token, %qh, {dims[i]}] : view<[%tokens]x{QH}x{D}xf32> -> f32")
    t.append(f"""      scf.yield {xs} : {yt}
    }} else {{
      %kh = index.assume %kvh_value [range(%kvh_value, 0, {KVH - 1})] : index""")
    for i in range(n):
        t.append(f"      %x{i} = view.load %k_view[%token, %kh, {dims[i]}] : view<[%tokens]x{KVH}x{D}xf32> -> f32")
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
        %vv{i} = view.load %v_view[%token, %kh, {dims[i]}] : view<[%tokens]x{KVH}x{D}xf32> -> f32
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


def attention_text(S, P):
    """Split-K causal GQA attention: workgroup (token, KV head, split) of 16
    waves, wave w = query head 16*kv + w, lane l = dims 4l..4l+3. Sliding
    layers see the last 2048 positions (including the current one). The
    combine kernel folds the splits and applies the sigmoid output gate."""
    U = UNROLL
    body = []
    for i in range(U):
        body.append(f"""      %key{i}_raw = index.add %j, %c{i}u : index
      %ok{i} = index.cmp ult, %key{i}_raw, %stop : index
      %key{i}_c = scf.select %ok{i}, %key{i}_raw, %j : index
      %kp{i} = index.add %window_low, %key{i}_c : index
      %ko{i} = func.call @g_kv_offset(%origins, %row, %kp{i}, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
      %kc{i} = buffer.view %arena[%ko{i}] : buffer -> view<{KVH}x32x4xf16>
      %kh{i} = vector.load %kc{i}[%kv_head, %lane, %c0] : view<{KVH}x32x4xf16> -> vector<4xf16>
      %k{i} = vector.extf %kh{i} : vector<4xf16> to vector<4xf32>
      %vo{i} = index.add %ko{i}, %plane_stride : offset
      %vc{i} = buffer.view %arena[%vo{i}] : buffer -> view<{KVH}x32x4xf16>
      %vh{i} = vector.load %vc{i}[%kv_head, %lane, %c0] : view<{KVH}x32x4xf16> -> vector<4xf16>
      %v{i} = vector.extf %vh{i} : vector<4xf16> to vector<4xf32>""")
    for i in range(U):
        body.append(f"""      %prod{i} = vector.mulf %q4, %k{i} : vector<4xf32>
      %part{i} = vector.reduce<addf> %prod{i}, %zero_f : vector<4xf32>, f32
      %sr{i} = kernel.subgroup.reduce<addf> %part{i} : f32
      %s{i} = scf.select %ok{i}, %sr{i}, %neg_big : f32""")
    prev = "%m"
    for i in range(U):
        body.append(f"      %mx{i} = scalar.maxnumf {prev}, %s{i} : f32"); prev = f"%mx{i}"
    body.append(f"""      %m_shift = scalar.subf %m, {prev} : f32
      %alpha = scalar.expf<afn> %m_shift : f32
      %l_scaled = scalar.mulf %l, %alpha : f32
      %alpha_v = vector.splat %alpha : vector<4xf32>
      %acc_s = vector.mulf %acc, %alpha_v : vector<4xf32>""")
    pl, pa = "%l_scaled", "%acc_s"
    for i in range(U):
        body.append(f"""      %d{i} = scalar.subf %s{i}, {prev} : f32
      %p{i} = scalar.expf<afn> %d{i} : f32
      %la{i} = scalar.addf {pl}, %p{i} : f32
      %pv{i} = vector.splat %p{i} : vector<4xf32>
      %aa{i} = vector.fmaf %pv{i}, %v{i}, {pa} : vector<4xf32>"""); pl, pa = f"%la{i}", f"%aa{i}"
    body.append(f"      scf.yield {prev}, {pl}, {pa} : f32, f32, vector<4xf32>")
    consts = "".join(f"  %c{i}u = index.constant {i} : index\n" for i in range(U))
    GATE_TEXT = """    %gate_view = buffer.view %gate[%zero] : buffer -> view<[%tokens]x{QH}x32x4xf32>
    %g = vector.load %gate_view[%token, %head, %lane, %c0] : view<[%tokens]x{QH}x32x4xf32> -> vector<4xf32>
    %sig = vector.logisticf<afn> %g : vector<4xf32>
    %result = vector.mulf %attended, %sig : vector<4xf32>""".replace("{QH}", str(QH)) if GATE else "    %result = vector.addf %attended, %zero4 : vector<4xf32>"
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
{consts}  %zero_f = scalar.constant 0.0 : f32
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
    %m_final, %l_final, %acc_final = scf.for %j = [%start to %stop step %cU](%m = %neg_big : f32, %l = %zero_f : f32, %acc = %zero4 : vector<4xf32>) -> (f32, f32, vector<4xf32>) {{
{chr(10).join(body)}
    }}
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
    %maximum = scf.for %s = [%c0 to %s_eff step %c1](%mx = %neg_big : f32) -> (f32) {{
      %bs = index.assume %s [range(%s, 0, {S - 1})] : index
      %ms = view.load %ml_view[%token, %head, %bs, 0] : view<[%tokens]x{QH}x{S}x2xf32> -> f32
      %next = scalar.maxnumf %mx, %ms : f32
      scf.yield %next : f32
    }}
    %den, %total = scf.for %s = [%c0 to %s_eff step %c1](%d = %zero_f : f32, %acc = %zero4 : vector<4xf32>) -> (f32, vector<4xf32>) {{
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


def attention():
    """Decode attention with up to 64 splits; the prefill fallback (listed,
    indexed by token over up to 1024 rows) keeps 16 splits."""
    dec, _, _ = attention_text(64, "g")
    pre, split_l, combine_l = attention_text(16, "g16")
    count16 = pre[pre.index("// Splits for"):pre.index('kernel.def export("g16_attention_split")')]
    return dec + "\n" + count16 + split_l + "\n" + combine_l + FALLBACK


FALLBACK = r"""
// Tokens of 16-token tiles that g_attention_flash does not cover (mixed rows,
// non-consecutive positions or padding), listed once per epoch; count[0] is
// the list length for indirect dispatch.
kernel.def export("g_attention_fallback_list") @g_attention_fallback_list(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %info: buffer, %list: buffer, %count: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 16, 8192), multiple_of(%tokens_arg, 16)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c16 = index.constant 16 : index
  %c256 = index.constant 256 : index
  %c0_i32 = scalar.constant 0 : i32
  %c16_i32 = scalar.constant 16 : i32
  %counter_bytes = index.constant 16 : offset
  %counter = buffer.alloca<workgroup> align(16) %counter_bytes : buffer
  %counter_view = buffer.view %counter[%zero] : buffer -> view<4xi32>
  %first = index.cmp eq, %lane, %c0 : index
  scf.if %first {
    view.store %c0_i32, %counter_view[%c0] : i32, view<4xi32>
  }
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %tiles = index.div %tokens, %c16 : index
  %list_view = buffer.view %list[%zero] : buffer -> view<[%tokens]xi32>
  scf.for %tile = [%lane to %tiles step %c256] {
    %tile_first = index.mul %tile, %c16 : index
    %uniform, %r_unused, %p_unused = func.call @g_tile_uniform(%info, %tokens, %tile_first) : (buffer, index, index) -> (i1, i32, i32)
    %true = scalar.constant true : i1
    %not_uniform = scalar.xori %uniform, %true : i1
    scf.if %not_uniform {
      %base32 = view.atomic.rmw<addi> %c16_i32, %counter_view[%c0] {ordering = relaxed, scope = workgroup} : i32, view<4xi32> -> i32
      %base_bound = scalar.assume %base32 [range(%base32, 0, 8176)] : i32
      %base = index.cast %base_bound : i32 to index
      scf.for %i = [%c0 to %c16 step %c1] {
        %slot_value = index.add %base, %i : index
        %slot = index.assume %slot_value [lt(%slot_value, %tokens)] : index
        %token_value = index.add %tile_first, %i : index
        %token32 = index.cast %token_value : index to i32
        view.store %token32, %list_view[%slot] : i32, view<[%tokens]xi32>
      }
    }
  }
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  scf.if %first {
    %total = view.load %counter_view[%c0] : view<4xi32> -> i32
    %count_view = buffer.view %count[%zero] : buffer -> view<1xi32>
    view.store %total, %count_view[0] : i32, view<1xi32>
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


def linear(fmt, x, s, w, wsize, y, k, n, kval, mode="store"):
    """Projection y = x W^T: WMMA GEMM for 64+ tokens (128-row tiles from 128),
    else the decode matrix-vector kernel (1-token or 4-token variant)."""
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
    return f"""scf.if %gemm {{
      scf.if %big {{
        kernel.launch @g_gemm_{fmt}_{mode}_m128[%t, {n}]({args}) : [index, index]({types})
      }} else {{
        kernel.launch @g_gemm_{fmt}_{mode}[%t, {n}]({args}) : [index, index]({types})
      }}
    }} else {{
      scf.if %single {{
        kernel.launch @{mv_name(fmt, kval, mode == "swiglu", n, 1)}[{n}, %t]{mv_args}
      }} else {{
        kernel.launch @{mv_name(fmt, kval, mode == "swiglu", n, 4)}[{n}, %t]{mv_args}
      }}
    }}"""

def head_mv(TG, fmt):
    """Output head argmax: workgroup = 128 vocabulary rows x TG spans,
    one lane per 32-value group of the 6656-wide row with each span's
    activations in registers; weights stream once per group of TG spans."""
    R = 128
    groups = HG
    lanes = (groups + 31) // 32 * 32
    waves = lanes // 32
    rb = row_bytes(fmt, H)
    name = "g_head_mv" + (f"_t{TG}" if TG > 1 else "")
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
  %x_view = buffer.view %h[%zero] : buffer -> view<[%tokens]x{groups}x32xf32>
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
  %use{j} = scalar.andi %live{j}, %ok : i1
  %x{j} = scf.if %use{j} -> (vector<32xf32>) {{
    %tk = index.assume %token{j} [lt(%token{j}, %tokens)] : index
    %v = vector.load %x_view[%tk, %g, %c0] : view<[%tokens]x{groups}x32xf32> -> vector<32xf32>
    scf.yield %v : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}""")
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
    %rb = index.assume %r [range(%r, 0, {R - 1})] : index
    %wv = scf.if %load -> (vector<32xf32>) {{
      %v = func.call @g_{fmt}_values(%w, %o, %within) : (buffer, offset, index) -> (vector<32xf32>)
      scf.yield %v : vector<32xf32>
    }} else {{
      scf.yield %zero32 : vector<32xf32>
    }}""")
    for j in range(TG):
        L.append(f"""    %pa{j} = vector.dotf<reassoc|contract> %wv, %x{j}, %zero_f : vector<32xf32>, vector<32xf32>, f32
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


USED_MV = set()      # (fmt, K, swiglu, TG) default-row (R=16) decode matrix-vector kernels the model launches
USED_MV_R = set()    # (fmt, K, swiglu, R, TG) adaptive-row variants
NVAL = {"%nH": H, "%nFF": FF, **({} if FF_UNIFORM else {f"%nFF{w}": w for w in set(FFL)}), "%nKV": KVH * D, "%nQ": QH * D, "%nQ2": 2 * QH * D}
if GDN:
    NVAL.update({"%nC": GDN["C"], "%nVD": GDN["VD"], "%nHV": GDN["HV"]})
from mv_rows import pick_r  # noqa: E402  (shared with make_kernel_tests.py)


def mv_name(fmt, kval, sw, n, TG):
    nv = NVAL.get(n, None) if isinstance(n, str) else n
    R = pick_r(nv, kval, TG)
    base = f"g_mv_{fmt}_k{kval}" + ("_swiglu" if sw else "") + (f"_t{TG}" if TG > 1 else "")
    if R == 16:
        USED_MV.add((fmt, kval, sw, TG))
        return base
    USED_MV_R.add((fmt, kval, sw, R, TG))
    return base + f"_r{R}"
USED_GEMM = {}       # fmt -> set of GEMM modes


def lin(fmt, x, s, w, wsize, y, k, n, kval, mode="store"):
    USED_GEMM.setdefault(fmt, set()).add(mode)
    return linear(fmt, x, s, w, wsize, y, k, n, kval, mode)


def lin_mv(fmt, x, w, wsize, y, n, kval):
    """Always the matrix-vector kernels (any token count): for tiny projections (n = 16)."""
    a = f"({n}, %t, %zero_i32, {x}, {w}, {y}) : [index, index](index, index, i32, buffer, view<{wsize}xi8>, buffer)"
    return f"""scf.if %single {{
    kernel.launch @{mv_name(fmt, kval, False, n, 1)}[{n}, %t]{a}
  }} else {{
    kernel.launch @{mv_name(fmt, kval, False, n, 4)}[{n}, %t]{a}
  }}"""


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
MODEL_PARTS = {}   # model() records its pieces here for the speculative-decoding programs (draft/gen_draft.py)


def model():
    """Command program: embed, then a statically unrolled layer loop (each
    layer's tensor formats come from the GGUF), then the argmax head."""
    QD, KD = QH * D, KVH * D
    sliding = set(CFG.get("sliding_layers", []))
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
""" + ("" if FF_UNIFORM else "".join(f"  %nFF{w} = index.constant {w} : index\n" for w in sorted(set(FFL)))) + f"""  %nQ = index.constant {QD} : index
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
  %c128 = index.constant 128 : index
  %prefill_shape = index.cmp uge, %t, {"%c_never" if os.environ.get("NO_GEMM") else "%c64"} : index
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
  %big = index.cmp uge, %t, %c128 : index
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
  %h16 = buffer.alloca<global> align(256) %h16_size : buffer
  %o16 = buffer.alloca<global> align(256) %o16_size : buffer
  %f16 = buffer.alloca<global> align(256) %f16_size : buffer
"""]
    if GATE:
        L_.append("  %gate = buffer.alloca<global> align(256) %q_size : buffer\n")
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
    L_.append(f"""
  kernel.launch @g_token_info[%t](%t, %spans, %metadata, %info) : [index](index, index, buffer, buffer)
  %list_entry = index.constant 4 : offset
  %list_size = index.scale %t, %list_entry : index, offset -> offset
  %count_size = index.constant 256 : offset
  %fallback_list = buffer.alloca<global> align(256) %list_size : buffer
  %fallback_count = buffer.alloca<global> align(256) %count_size : buffer
  scf.if %gemm {{
    kernel.launch @g_attention_fallback_list[%t](%t, %info, %fallback_list, %fallback_count) : [index](index, buffer, buffer, buffer)
  }}
  %embedding = command.parameter %weights, "token_embd.weight" : view<{esize}xi8>
  %first_norm = command.parameter %weights, "blk.0.attn_norm.weight" : view<{H}xf32>
  kernel.launch @g_embed[%t](%t, %tokens{", %metadata, %info" if VISION else ""}, %embedding, %first_norm, %x, %h) : [index](index, buffer{", buffer, buffer" if VISION else ""}, view<{esize}xi8>, view<{H}xf32>, buffer, buffer)
  %output_norm = command.parameter %weights, "output_norm.weight" : view<{H}xf32>
""")
    gate = "%gate" if GATE else "%q"
    attn_ordinal = {i: n for n, i in enumerate(j for j in range(L) if j not in RECUR)}
    gdn_ordinal = {i: n for n, i in enumerate(sorted(RECUR))}
    def layer_text(i, ordinal=None, last_norm=None, kind_note=None):
        """Text of trunk layer i (tensors blk.i.*). ordinal overrides its attention KV-plane ordinal and
        last_norm names the norm applied to its output (default: next layer's attn_norm / output_norm)."""
        b = f"blk.{i}."
        nm = lambda n: b + n + ".weight"
        sz = lambda n: tensor_bytes(nm(n))
        par = lambda var, n, ty=None: f'    %{var}_{i} = command.parameter %weights, "{nm(n)}" : {ty or f"view<{sz(n)}xi8>"}\n'
        kind = kind_note or (" (gated delta-net)" if i in RECUR else (" (sliding)" if i in sliding else "") + (" (no RoPE)" if i in nope else ""))
        T = [f"  // ---- layer {i}{kind}\n  scf.if %gemm {{\n    kernel.launch @g_row_scale_f16[%t](%t, %nH, %h, %s_h, %h16) : [index](index, index, buffer, buffer, buffer)\n  }}\n",
             f"  %layer_{i} = index.constant {attn_ordinal.get(i, gdn_ordinal.get(i)) if ordinal is None else ordinal} : index\n  %sliding_{i} = scalar.constant {1 if i in sliding else 0} : i32\n  %rope_{i} = scalar.constant {0 if i in nope else 1} : i32\n"]
        T += [par("w_gate", "ffn_gate"), par("w_up", "ffn_up"), par("w_down", "ffn_down"),
              par("ffn_norm", FFN_NORM, f"view<{H}xf32>")]
        if i in RECUR:
            T.append(gdn_layer(i, b, nm, sz, par))
        else:
            T += [par("wq", "attn_q"), par("wk", "attn_k"), par("wv", "attn_v"), par("wo", "attn_output")]
        if i in RECUR:
            pass
        elif GATE and not FUSED_Q:
            T.append(par("wg", "attn_gate"))
        if QK_NORM and i not in RECUR:
            T += [par("q_norm", "attn_q_norm", f"view<{D}xf32>"), par("k_norm", "attn_k_norm", f"view<{D}xf32>")]
        if POST_NORM:
            T += [par("post_attn", "post_attention_norm", f"view<{H}xf32>"), par("post_ffn", "post_ffw_norm", f"view<{H}xf32>")]
        if i in RECUR:
            pass
        elif FUSED_Q:
            T.append("  " + lin(fmt_of(nm("attn_q")), "%h", "%s_h", f"%wq_{i}", sz("attn_q"), "%qg", "%nH", "%nQ2", H) + "\n")
            T.append("  kernel.launch @g_split_qgate[%t](%t, %qg, %q, %gate) : [index](index, buffer, buffer, buffer)\n")
        else:
            T.append("  " + lin(fmt_of(nm("attn_q")), "%h", "%s_h", f"%wq_{i}", sz("attn_q"), "%q", "%nH", "%nQ", H) + "\n")
        if i not in RECUR:
          T.append("  " + lin(fmt_of(nm("attn_k")), "%h", "%s_h", f"%wk_{i}", sz("attn_k"), "%k", "%nH", "%nKV", H) + "\n")
        if i not in RECUR:
          T.append("  " + lin(fmt_of(nm("attn_v")), "%h", "%s_h", f"%wv_{i}", sz("attn_v"), "%v", "%nH", "%nKV", H) + "\n")
        if GATE and not FUSED_Q and i not in RECUR:
            T.append("  " + lin(fmt_of(nm("attn_gate")), "%h", "%s_h", f"%wg_{i}", sz("attn_gate"), "%gate", "%nH", "%nQ", H) + "\n")
        qn, kn = (f"%q_norm_{i}", f"%k_norm_{i}") if QK_NORM else ("%no_norm", "%no_norm")
        nty = f"view<{D}xf32>" if QK_NORM else "buffer"
        if i not in RECUR:
          T.append(f"""  kernel.launch @g_qk_rope_store[%t](%t, %layer_{i}, %rope_{i}, %context, %pool, %q, %k, %v, {qn}, {kn}, %info, %origins, %arena{", %metadata, %tokens" if MROPE else ""}) : [index](index, index, i32, index, index, buffer, buffer, buffer, {nty}, {nty}, buffer, buffer, buffer{", buffer, buffer" if MROPE else ""})
  scf.if %gemm {{
{"" if NO_FLASH else f"    kernel.launch @g_attention_flash2[%t](%t, %layer_{i}, %sliding_{i}, %context, %pool, %q, {gate}, %info, %origins, %arena, %o) : [index](index, index, i32, index, index, buffer, buffer, buffer, buffer, buffer, buffer)" + chr(10)}    kernel.launch @g16_attention_split_listed[%fallback_count](%t, %fallback_list, %layer_{i}, %sliding_{i}, %context, %pool, %q, %info, %origins, %arena, %pacc, %pml) : [buffer](index, buffer, index, i32, index, index, buffer, buffer, buffer, buffer, buffer, buffer)
    kernel.launch @g16_attention_combine_listed[%fallback_count](%t, %fallback_list, %sliding_{i}, %info, %pacc, %pml, {gate}, %o) : [buffer](index, buffer, i32, buffer, buffer, buffer, buffer, buffer)
  }} else {{
    kernel.launch @g_attention_split[%t](%t, %layer_{i}, %sliding_{i}, %context, %pool, %q, %info, %origins, %arena, %pacc, %pml) : [index](index, index, i32, index, index, buffer, buffer, buffer, buffer, buffer, buffer)
    kernel.launch @g_attention_combine[%t](%t, %sliding_{i}, %info, %pacc, %pml, {gate}, %o) : [index](index, i32, buffer, buffer, buffer, buffer, buffer)
  }}
  scf.if %gemm {{
    kernel.launch @g_row_scale_f16[%t](%t, %nQ, %o, %s_o, %o16) : [index](index, index, buffer, buffer, buffer)
  }}
""")
        if i not in RECUR:
          T.append("  " + lin(fmt_of(nm("attn_output")), "%o", "%s_o", f"%wo_{i}", sz("attn_output"), "%y", "%nQ", "%nH", QD) + "\n")
        pa = f"%post_attn_{i}" if POST_NORM else f"%ffn_norm_{i}"
        T.append(f"  kernel.launch @g_post_add_norm[%t](%t, %y, {pa}, %x, %ffn_norm_{i}, %h) : [index](index, buffer, view<{H}xf32>, buffer, view<{H}xf32>, buffer)\n")
        T.append(f"  scf.if %gemm {{\n    kernel.launch @g_row_scale_f16[%t](%t, %nH, %h, %s_h, %h16) : [index](index, index, buffer, buffer, buffer)\n  }}\n")
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
        return "".join(T).replace("\n    %", "\n  %").replace("\n      ", "\n    ")

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
    spec_kernels = ""
    if gen_draft.enabled():   # before the kernel list is emitted: these programs record mv / GEMM variants too
        head_name = "output.weight" if "output.weight" in TENS else "token_embd.weight"
        gfacts = dict(H=H, HG=HG, EPS=EPS, D=D, KD=KVH * D, QK_NORM=QK_NORM, VISION=bool(VISION), MROPE=bool(MROPE),
                      GDN=GDN, NG=len(RECUR), gdn_layers=[(i, n) for n, i in enumerate(sorted(RECUR))],
                      head_name=head_name, head_size=tensor_bytes(head_name), esize=tensor_bytes("token_embd.weight"),
                      prologue=MODEL_PARTS.get("prologue"), layer_text=MODEL_PARTS.get("layer_text"), lin=lin,
                      fmt=fmt_of, size=tensor_bytes, mtp_layer=MTP_LAYER, replay=MODEL_PARTS.get("replay", []))
        if gen_draft.DRAFT["kind"] == "dflash":
            import gen_dflash
            dcfg = gen_dflash.config(gen_draft.DRAFT["gguf"])
            assert dcfg["hidden"] == H, f"drafter hidden {dcfg['hidden']} != target {H}"
            DT = dcfg["tensors"]
            def dsize(name):
                t = DT[name]
                n = 1
                for d in t["shape"]:
                    n *= d
                return n * 4 if t["type"] == "f32" else n // SB[t["type"]] * FMT[t["type"]]
            gfacts.update(dflash=dcfg, dflash_taps=[t - 1 for t in dcfg["target_layers"]], L=L, CHUNKS=CHUNKS, V=V,
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
        program = "\n".join(spec) if os.environ.get("SPEC_EXACT") == "1" else gen_draft.small_gemm("\n".join(spec))
        spec_kernels = gen_draft.kernels(gfacts)
        if gen_draft.DRAFT["kind"] == "dflash":
            spec_kernels += gen_dflash.kernels(gfacts)
        if gen_draft.DRAFT["kind"] == "ngram":
            spec_kernels += gen_ngram.kernels(gfacts)
        if gen_draft.DRAFT["kind"] == "assist":
            spec_kernels += gen_draft.mtp_kernels(gfacts) + gen_assist.kernels(gfacts)
    if any(f == "f32" for f, *_ in USED_MV | USED_MV_R) or "f32" in USED_GEMM:
        COMMON += F32_VALUES
    if any(f == "f16" for f, *_ in USED_MV | USED_MV_R) or "f16" in USED_GEMM:
        COMMON += F16_VALUES
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
    if RECUR:
        parts.append(gdn_kernels(CFG, EPS))
    if spec_kernels:
        parts.append(spec_kernels)
    if MIXED_FFN:
        parts.append(SILU_MUL)
    env = dict(os.environ, GEMM_FORMATS=",".join(f"{f}:{'+'.join(sorted(m))}" for f, m in sorted(USED_GEMM.items())))
    # Prefill GEMM switches, ON by default; set GEMM_N_FASTEST=0 / GEMM_SKIP_PADDING=0 to A/B them off.
    env.setdefault("GEMM_SKIP_PADDING", "1"); env.setdefault("GEMM_N_FASTEST", "1")
    parts.append(subprocess.run([sys.executable, os.path.join(here, "gen_gemm.py")], capture_output=True, text=True, check=True, env=env).stdout)
    parts.append(open(os.path.join(here, "tile_uniform.loom")).read())
    if not NO_FLASH:
        flash2 = os.path.join(out_dir, "flash2.loom")
        subprocess.run([sys.executable, os.path.join(here, "gen_flash2.py")], check=True,
                       env=dict(os.environ, FLASH2_TARGET="llm", FLASH2_OUT=flash2, LLM_CONFIG=os.environ.get("LLM_CONFIG") or sys.argv[1]))
        parts.append(open(flash2).read())
    parts.append(program)
    out = os.path.join(out_dir, "model.loom")
    open(out, "w", newline="\n").write("\n".join(parts))
    print("wrote", out)
