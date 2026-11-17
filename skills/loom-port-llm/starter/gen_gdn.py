"""Gated DeltaNet (linear attention) kernels for hybrid models such as Qwen3.5 / Qwen3-Next
(llama.cpp: src/models/qwen35.cpp + delta-net-base.cpp). gen_llm.py includes them when the
config has "recurrent_layers".

Per layer and token: qkv = W_qkv h (C = 2*HK*DK + HV*DV channels), z = W_z h, alpha/beta = W h
(HV each). Causal conv1d (kernel K) over qkv + SiLU; q, k L2-normalized per head (q also scaled
by 1/sqrt(DK)); per value head with state S[a][b] (a: key dim, b: value dim):
    S *= exp(softplus(alpha + dt_bias) * ssm_a)
    d  = sigmoid(beta) * (v - S^T k);  S += k (x) d;  o = S^T q
then RMSNorm(o) * ssm_norm * silu(z) per head and the output projection. Value head h reads
key head h % HK (ggml_repeat tiles the key heads).

Recurrent state lives in the runtime's recurrent slots (storage.loom): the slot holds, per
recurrent layer, S (HV*DV*DK f32, column-major per head: index (h*DV + b)*DK + a) followed
by the conv history (K-1 previous qkv inputs, [K-1][C] f32, oldest first). Per row, the
origins record names a reader slot (state at the fork position, e.g. a reused prefix), a
writer slot (the row's private state) and the fork position: a span starting at the fork
reads the reader slot, otherwise the writer; a span at position 0 starts from zeros. Every
span publishes its final state to the writer. The register recurrence is adapted from
hrx-loom-library model/Qwen/Qwen3.8-27B/program/kernel/gdn_prefill.loom; by default it is replaced by
the row-split form (_recurrence_rows: 8 lanes per value column, per-lane dot products + an 8-lane
shuffle sum), GDN_ROW_LANES=0 keeps the original 32-lane form.
"""
import os


_V_GROUPED = [False, 1]


def kh_line():
    """The value head's key head. GGUF files hold the value heads tiled (llama.cpp's converter reorders them so ggml_repeat
    broadcasts the key heads: value head h reads key head h % HK); an HF safetensors checkpoint keeps them grouped (value heads
    h = r * k + j of key head k, repeat_interleave: key head h // (HV / HK)), which cfg gdn.v_order "grouped" reads natively
    (no reordering of any weight)."""
    if _V_GROUPED[0]:
        return "%vratio = index.constant {RATIO} : index\n  %kh = index.div %vh, %vratio : index".replace("{RATIO}", str(_V_GROUPED[1]))
    return "%kh = index.rem %vh, %chk : index"


def gdn_geometry(cfg):
    g = cfg["gdn"]
    _V_GROUPED[0] = g.get("v_order") == "grouped"
    if _V_GROUPED[0]:
        _V_GROUPED[1:] = [g["v_heads"] // g["k_heads"]]
    HK, HV, DK, DV, K = g["k_heads"], g["v_heads"], g["head_k"], g["head_v"], g["conv_kernel"]
    C = 2 * HK * DK + HV * DV
    state_bytes = HV * DV * DK * 4
    history_bytes = (K - 1) * C * 4
    layer_bytes = (state_bytes + history_bytes + 255) // 256 * 256
    NG = len(cfg["recurrent_layers"])
    pend_base = pend_stride = pend_bytes = 0
    if _defer_env() and _row_lanes(DK, DV):   # decode.spec.gdn_fold_defer: the slot also holds each layer's pending fold
        pend_bytes = (PEND_HEAD + FOLD_W * (HK * DK + HV * DV + HV) * 4 + 255) // 256 * 256
        pend_base, pend_stride = layer_bytes * NG, pend_bytes   # after every layer's state: the layer stride is that of the slot without it
    return dict(HK=HK, HV=HV, DK=DK, DV=DV, K=K, C=C, VD=HV * DV, state_bytes=state_bytes, pend_base=pend_base, pend_stride=pend_stride,
                pend_bytes=pend_bytes,
                layer_bytes=layer_bytes, slot_bytes=(layer_bytes + pend_bytes) * NG)


SLOT = r"""
// (reader, writer, fresh) for the span of `row` starting at `start`: the reader slot at the
// fork position, otherwise the row's own writer; position 0 starts from zeros.
func.def inline @g_gdn_slots(%origins: buffer, %row: index, %start32: i32, %layer_bytes: offset) -> (offset, offset, i1) {
  %record_stride = index.constant 40 : offset
  %bounded_row = index.assume %row [range(%row, 0, 2147483646)] : index
  %record_offset = index.scale %bounded_row, %record_stride : index, offset -> offset
  %records = buffer.view %origins[%record_offset] : buffer -> view<5xi64>
  %read64 = view.load %records[0] : view<5xi64> -> i64
  %write64 = view.load %records[1] : view<5xi64> -> i64
  %fork64 = view.load %records[3] : view<5xi64> -> i64
  %start64 = scalar.extui %start32 : i32 to i64
  %at_fork = scalar.cmpi eq, %start64, %fork64 : i64
  %reader64 = scf.select %at_fork, %read64, %write64 : i64
  %reader_bound = scalar.assume %reader64 [range(%reader64, 0, 9223372036854775807)] : i64
  %writer_bound = scalar.assume %write64 [range(%write64, 0, 9223372036854775807)] : i64
  %reader_base = index.cast %reader_bound : i64 to offset
  %writer_base = index.cast %writer_bound : i64 to offset
  %reader = index.add %reader_base, %layer_bytes : offset
  %writer = index.add %writer_base, %layer_bytes : offset
  %zero32 = scalar.constant 0 : i32
  %fresh = scalar.cmpi eq, %start32, %zero32 : i32
  func.return %reader, %writer, %fresh : offset, offset, i1
}

// Whether span `span` publishes its state (speculative verification spans carry flag bit 4 and do not).
func.def inline @g_gdn_publish(%metadata: buffer, %spans: index, %span: index) -> (i1) {
  %c16 = index.constant 16 : offset
  %bounded_span = index.assume %span [lt(%span, %spans)] : index
  %packets = buffer.view %metadata[%c16] : buffer -> view<[%spans]x9xi32>
  %flags = view.load %packets[%bounded_span, 6] : view<[%spans]x9xi32> -> i32
  %four = scalar.constant 4 : i32
  %bit = scalar.andi %flags, %four : i32
  %zero32 = scalar.constant 0 : i32
  %publish = scalar.cmpi eq, %bit, %zero32 : i32
  func.return %publish : i1
}

// Span `span` of the epoch: (active, length, start position, row, first input slot).
func.def inline @g_gdn_span(%metadata: buffer, %spans: index, %span: index) -> (i1, index, i32, index, index) {
  %zero = index.constant 0 : offset
  %c16 = index.constant 16 : offset
  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %count_bound = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count = index.cast %count_bound : i32 to index
  %active = index.cmp ult, %span, %count : index
  %bounded_span = index.assume %span [lt(%span, %spans)] : index
  %packets = buffer.view %metadata[%c16] : buffer -> view<[%spans]x9xi32>
  %length32 = view.load %packets[%bounded_span, 0] : view<[%spans]x9xi32> -> i32
  %start32 = view.load %packets[%bounded_span, 1] : view<[%spans]x9xi32> -> i32
  %row32 = view.load %packets[%bounded_span, 3] : view<[%spans]x9xi32> -> i32
  %first32 = view.load %packets[%bounded_span, 4] : view<[%spans]x9xi32> -> i32
  %length_bound = scalar.assume %length32 [range(%length32, 0, 8192)] : i32
  %row_bound = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
  %first_bound = scalar.assume %first32 [range(%first32, 0, 8191)] : i32
  %length = index.cast %length_bound : i32 to index
  %row = index.cast %row_bound : i32 to index
  %first = index.cast %first_bound : i32 to index
  %zero_index = index.constant 0 : index
  %nonempty = index.cmp ugt, %length, %zero_index : index
  %live = scalar.andi %active, %nonempty : i1
  func.return %live, %length, %start32, %row, %first : i1, index, i32, index, index
}
"""


def gdn_kernels(cfg, eps):
    G = gdn_geometry(cfg)
    HK, HV, DK, DV, K, C, VD = G["HK"], G["HV"], G["DK"], G["DV"], G["K"], G["C"], G["VD"]
    CB = int(os.environ.get("GDN_CB", str((C + 255) // 256)))  # channel blocks per token/span (1 = original serial grid)
    NG = len(cfg["recurrent_layers"])
    SB, LB = G["state_bytes"], G["layer_bytes"]
    RK, RV = DK // 32, DV // 32
    assert DK % 32 == 0 and DV % 32 == 0 and DV % 4 == 0 and HV % HK == 0
    # Speculative decoding: verification spans must not publish (draft/gen_draft.py replays them).
    spec = bool(os.environ.get("LLM_DRAFT")) and os.environ.get("LLM_DRAFT") != "null"
    a, b = SLOT.index("// Whether span `span` publishes"), SLOT.index("// Span `span` of the epoch")
    T = [SLOT if spec else SLOT[:a] + SLOT[b:]]
    guard_open = ("    %publish = func.call @g_gdn_publish(%metadata, %spans, %span) : (buffer, index, index) -> (i1)\n"
                  "    scf.if %publish {\n") if spec else ""
    guard_close = "\n    }" if spec else ""
    # ---- conv1d + SiLU over the span's inputs and the stored history
    taps = []
    for j in range(K):
        back = K - 1 - j
        taps.append(f"""      %w{j} = view.load %w_view[%ch, {j}] : view<{C}x{K}xf32> -> f32
      %back{j} = index.constant {back} : index
      %has_rel{j} = index.cmp uge, %rel, %back{j} : index
      %has_token{j} = index.cmp uge, %token, %back{j} : index
      %has{j} = scalar.andi %has_rel{j}, %has_token{j} : i1
      %x{j} = scf.if %has{j} -> (f32) {{
        %src{j} = index.sub %token, %back{j} : index
        %bsrc{j} = index.assume %src{j} [lt(%src{j}, %tokens)] : index
        %v{j} = view.load %qkv_view[%bsrc{j}, %ch] : view<[%tokens]x{C}xf32> -> f32
        scf.yield %v{j} : f32
      }} else {{
        %h{j} = scf.if %fresh -> (f32) {{
          scf.yield %zero_f : f32
        }} else {{
          %hk{j}a = index.constant {K - 1 - back} : index
          %hk{j}b = index.add %hk{j}a, %rel : index
          %hk{j} = index.assume %hk{j}b [range(%hk{j}b, 0, {K - 2})] : index
          %hv{j} = view.load %hist_view[%hk{j}, %ch] : view<{K - 1}x{C}xf32> -> f32
          scf.yield %hv{j} : f32
        }}
        scf.yield %h{j} : f32
      }}
      %acc{j} = scalar.fmaf %w{j}, %x{j}, {"%zero_f" if j == 0 else f"%acc{j - 1}"} : f32""")
    # CU channels per loop step (c, c + stride, ...), every load before any store: Loom orders a global load
    # after earlier global stores, so one channel per step paid a store -> load round trip per channel.
    CU = int(os.environ.get("GDN_CONV_UNROLL", "8"))
    tap_text = chr(10).join(taps) + f"""
      %sig = scalar.logisticf<afn> %acc{K - 1} : f32
      %silu = scalar.mulf %acc{K - 1}, %sig : f32"""
    import re as _re
    defined = set(_re.findall(r"(%\w+) = ", tap_text))
    cu_parts, cu_stores = [], []
    for u in range(CU):
        body = _re.sub(r"%\w+", lambda m: m.group(0) + f"_u{u}" if m.group(0) in defined or m.group(0) == "%ch" else m.group(0), tap_text)
        body = "\n".join("  " + l for l in body.split("\n"))
        cu_parts.append(f"""      %cu{u}_k = index.constant {u} : index
      %cu{u}_o = index.mul %cu{u}_k, %cstride : index
      %cu{u}_c = index.add %c, %cu{u}_o : index
      %cu{u}_in = index.cmp ult, %cu{u}_c, %cC : index
      %val_u{u} = scf.if %cu{u}_in -> (f32) {{
        %ch_u{u} = index.assume %cu{u}_c [range(%cu{u}_c, 0, {C - 1})] : index
{body}
        scf.yield %silu_u{u} : f32
      }} else {{
        scf.yield %zero_f : f32
      }}""")
        cu_stores.append(f"""      scf.if %cu{u}_in {{
        %cs{u} = index.assume %cu{u}_c [range(%cu{u}_c, 0, {C - 1})] : index
        view.store %val_u{u}, %out_view[%token, %cs{u}] : f32, view<[%tokens]x{C}xf32>
      }}""")
    conv_unrolled = "\n".join(cu_parts + cu_stores)
    T.append(f"""
// out[t, c] = silu(sum_j w[c, j] x[t - (K-1) + j, c]); inputs before the span come from the history.
kernel.def export("g_gdn_conv") @g_gdn_conv(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cblocks = index.constant {CB} : index
  %cmany = index.constant 64 : index
  %few = index.cmp ult, %tokens, %cmany : index
  %cbsel = scf.select %few, %cblocks, %c1 : index
  kernel.launch.config workgroups(%tokens, %cbsel, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %layer_arg: index, %metadata: buffer, %info: buffer, %origins: buffer, %arena: buffer, %qkv: buffer, %weight: buffer, %out: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {NG - 1})] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %thread_id = kernel.workitem.id<x> : index
  %thread = index.assume %thread_id [range(%thread_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0_i32 = scalar.constant 0 : i32
  %zero_f = scalar.constant 0.0 : f32
  %one_f = scalar.constant 1.0 : f32
  %cC = index.constant {C} : index
  %c256 = index.constant 256 : index
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %span32 = view.load %info_view[%token, 3] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  scf.if %live {{
    %span_bound = scalar.assume %span32 [range(%span32, 0, 8191)] : i32
    %span = index.cast %span_bound : i32 to index
    %active, %length, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
    %layer_stride = index.constant {LB} : offset
    %layer_bytes = index.scale %layer, %layer_stride : index, offset -> offset
    %reader, %writer, %fresh = func.call @g_gdn_slots(%origins, %row, %start32, %layer_bytes) : (buffer, index, i32, offset) -> (offset, offset, i1)
    %history_skip = index.constant {SB} : offset
    %history_at = index.add %reader, %history_skip : offset
    %hist_view = buffer.view %arena[%history_at] : buffer -> view<{K - 1}x{C}xf32>
    %qkv_view = buffer.view %qkv[%zero] : buffer -> view<[%tokens]x{C}xf32>
    %out_view = buffer.view %out[%zero] : buffer -> view<[%tokens]x{C}xf32>
    %w_view = buffer.view %weight[%zero] : buffer -> view<{C}x{K}xf32>
    %rel_value = index.sub %token, %first : index
    %rel = index.assume %rel_value [range(%rel_value, 0, 8191)] : index
    %cblk_id = kernel.workgroup.id<y> : index
    %cblk_off = index.mul %cblk_id, %c256 : index
    %cstart = index.add %thread, %cblk_off : index
    %cmany_b = index.constant 64 : index
    %few_b = index.cmp ult, %tokens, %cmany_b : index
    %cbk = index.constant {CB} : index
    %cone = index.constant 1 : index
    %cbn = scf.select %few_b, %cbk, %cone : index
    %cstride = index.mul %cbn, %c256 : index
    %cunroll = index.constant {CU} : index
    %cstride_u = index.mul %cstride, %cunroll : index
    scf.for %c = [%cstart to %cC step %cstride_u] {{
{conv_unrolled}
    }}
  }}
  kernel.return
}}
""")
    # ---- history commit: the last K-1 inputs of each span (after g_gdn_conv read the old ones)
    reads, writes = [], []
    for k in range(K - 1):
        reads.append(f"""      %need{k} = index.constant {K - 1 - k} : index
      %has{k} = index.cmp uge, %length, %need{k} : index
      %val{k} = scf.if %has{k} -> (f32) {{
        %off{k}a = index.add %first, %length : index
        %src{k} = index.sub %off{k}a, %need{k} : index
        %bsrc{k} = index.assume %src{k} [lt(%src{k}, %tokens)] : index
        %v{k} = view.load %qkv_view[%bsrc{k}, %ch] : view<[%tokens]x{C}xf32> -> f32
        scf.yield %v{k} : f32
      }} else {{
        %o{k} = scf.if %fresh -> (f32) {{
          scf.yield %zero_f : f32
        }} else {{
          %hk{k}a = index.constant {k} : index
          %hk{k}b = index.add %hk{k}a, %length : index
          %hk{k} = index.assume %hk{k}b [range(%hk{k}b, 0, {K - 2})] : index
          %hv{k} = view.load %old_view[%hk{k}, %ch] : view<{K - 1}x{C}xf32> -> f32
          scf.yield %hv{k} : f32
        }}
        scf.yield %o{k} : f32
      }}""")
        writes.append(f"      view.store %val{k}, %new_view[{k}, %ch] : f32, view<{K - 1}x{C}xf32>")
    T.append(f"""
kernel.def export("g_gdn_conv_commit") @g_gdn_conv_commit(%spans: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cblocks = index.constant {CB} : index
  kernel.launch.config workgroups(%spans, %cblocks, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %layer_arg: index, %metadata: buffer, %origins: buffer, %arena: buffer, %qkv: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {NG - 1})] : index
  %span_id = kernel.workgroup.id<x> : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
  %thread_id = kernel.workitem.id<x> : index
  %thread = index.assume %thread_id [range(%thread_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %zero_f = scalar.constant 0.0 : f32
  %cC = index.constant {C} : index
  %c256 = index.constant 256 : index
  %active, %length, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
  scf.if %active {{
    %layer_stride = index.constant {LB} : offset
    %layer_bytes = index.scale %layer, %layer_stride : index, offset -> offset
    %reader, %writer, %fresh = func.call @g_gdn_slots(%origins, %row, %start32, %layer_bytes) : (buffer, index, i32, offset) -> (offset, offset, i1)
    %history_skip = index.constant {SB} : offset
    %old_at = index.add %reader, %history_skip : offset
    %new_at = index.add %writer, %history_skip : offset
    %old_view = buffer.view %arena[%old_at] : buffer -> view<{K - 1}x{C}xf32>
    %new_view = buffer.view %arena[%new_at] : buffer -> view<{K - 1}x{C}xf32>
    %qkv_view = buffer.view %qkv[%zero] : buffer -> view<[%tokens]x{C}xf32>
    %cblk_id = kernel.workgroup.id<y> : index
    %cblk_off = index.mul %cblk_id, %c256 : index
    %cstart = index.add %thread, %cblk_off : index
    %cstride = index.constant {256 * CB} : index
    scf.for %c = [%cstart to %cC step %cstride] {{
      %ch = index.assume %c [range(%c, 0, {C - 1})] : index
{chr(10).join(reads)}
{guard_open}{chr(10).join(writes)}{guard_close}
    }}
  }}
  kernel.return
}}
""")
    # ---- q/k inverse L2 norms: inverse[t, 2*kh] (q), inverse[t, 2*kh + 1] (k)
    sq = "\n".join(f"""    %ix{m} = index.constant {32 * m} : index
    %d{m} = index.add %lane, %ix{m} : index
    %c{m}v = index.add %base, %d{m} : index
    %cc{m} = index.assume %c{m}v [range(%c{m}v, 0, {C - 1})] : index
    %e{m} = view.load %x_view[%token, %cc{m}] : view<[%tokens]x{C}xf32> -> f32
    %s{m} = scalar.fmaf %e{m}, %e{m}, {"%zero_f" if m == 0 else f"%s{m - 1}"} : f32""" for m in range(RK))
    T.append(f"""
kernel.def export("g_gdn_qk_norm") @g_gdn_qk_norm(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %x: buffer, %inverse: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, 7)] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c8 = index.constant 8 : index
  %chk = index.constant {HK} : index
  %heads = index.constant {2 * HK} : index
  %cdk = index.constant {DK} : index
  %zero_f = scalar.constant 0.0 : f32
  %eps = scalar.constant {eps!r} : f32
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{C}xf32>
  %inv_view = buffer.view %inverse[%zero] : buffer -> view<[%tokens]x{2 * HK}xf32>
  scf.for %head = [%wave to %heads step %c8] {{
    %h = index.assume %head [range(%head, 0, {2 * HK - 1})] : index
    %base = index.mul %h, %cdk : index
{sq}
    %sum = kernel.subgroup.reduce<addf> %s{RK - 1} : f32
    %stable = scalar.addf %sum, %eps : f32
    %inv = scalar.rsqrtf %stable : f32
    %is_q = index.cmp ult, %h, %chk : index
    %kh_k = index.sub %h, %chk : index
    %kh = scf.select %is_q, %h, %kh_k : index
    %c2 = index.constant 2 : index
    %slot_a = index.mul %kh, %c2 : index
    %c1 = index.constant 1 : index
    %slot_k = index.add %slot_a, %c1 : index
    %slot_v = scf.select %is_q, %slot_a, %slot_k : index
    %slot = index.assume %slot_v [range(%slot_v, 0, {2 * HK - 1})] : index
    %leader = index.cmp eq, %lane, %c0 : index
    scf.if %leader {{
      view.store %inv, %inv_view[%token, %slot] : f32, view<[%tokens]x{2 * HK}xf32>
    }}
  }}
  kernel.return
}}
""")
    # ---- recurrence: workgroup (value head, 4-column tile, span) of 32 lanes; lane owns key rows lane + 32m
    loads = []
    for m in range(RK):
        loads.append(f"""    %qr{m}c = index.constant {32 * m} : index
    %qr{m} = index.add %q_row0, %qr{m}c : index
    %kr{m} = index.add %k_row0, %qr{m}c : index
    %q{m}_raw = view.load %x_view[%tabs, %qr{m}] : view<[%tokens]x{C}xf32> -> f32
    %k{m}_raw = view.load %x_view[%tabs, %kr{m}] : view<[%tokens]x{C}xf32> -> f32
    %q{m} = scalar.mulf %q{m}_raw, %q_inverse : f32
    %k{m} = scalar.mulf %k{m}_raw, %k_inverse : f32
    %qv{m} = vector.splat %q{m} : vector<4xf32>
    %kv{m} = vector.splat %k{m} : vector<4xf32>
    %dec{m} = vector.mulf %old{m}, %decay_vector : vector<4xf32>
    %pr{m} = vector.mulf %dec{m}, %kv{m} : vector<4xf32>""")
    psum = "%pr0"
    for m in range(1, RK):
        loads.append(f"    %ps{m} = vector.addf {psum}, %pr{m} : vector<4xf32>"); psum = f"%ps{m}"
    upd = []
    for m in range(RK):
        upd.append(f"""    %up{m} = vector.mulf %kv{m}, %delta : vector<4xf32>
    %new{m} = vector.addf %dec{m}, %up{m} : vector<4xf32>
    %ot{m} = vector.mulf %new{m}, %qv{m} : vector<4xf32>""")
    osum = "%ot0"
    for m in range(1, RK):
        upd.append(f"    %os{m} = vector.addf {osum}, %ot{m} : vector<4xf32>"); osum = f"%os{m}"
    state_idx = "\n".join(f"""  %si{m}c = index.constant {32 * m} : index
  %si{m} = index.add %state_index0, %si{m}c : index""" for m in range(1, RK))
    state_idx = "  %si0 = index.add %state_index0, %c0 : index\n" + state_idx
    gathers = "\n".join(f"""      %g{m} = vector.gather %state_view[%si{m}][%column_offsets] : view<{HV * DV * DK}xf32>, vector<4xindex> -> vector<4xf32>""" for m in range(RK))
    scatters = "\n".join(f"""    vector.scatter %final{m}, %next_view[%si{m}][%column_offsets] : vector<4xf32>, view<{HV * DV * DK}xf32>, vector<4xindex>""" for m in range(RK))
    v4 = ", ".join(["vector<4xf32>"] * RK)
    T.append(f"""
kernel.def export("g_gdn_recurrence") @g_gdn_recurrence(%spans: index) {{
  %c1 = index.constant 1 : index
  %heads = index.constant {HV} : index
  %tiles = index.constant {DV // 4} : index
  %c32 = index.constant 32 : index
  kernel.launch.config workgroups(%heads, %tiles, %spans) workgroup_size(%c32, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %layer_arg: index, %metadata: buffer, %origins: buffer, %arena: buffer, %x: buffer, %inverse: buffer, %alpha: buffer, %beta: buffer, %ssm_a: buffer, %dt_bias: buffer, %out: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {NG - 1})] : index
  %head_id = kernel.workgroup.id<x> : index
  %vh = index.assume %head_id [range(%head_id, 0, {HV - 1})] : index
  %tile_id = kernel.workgroup.id<y> : index
  %tile = index.assume %tile_id [range(%tile_id, 0, {DV // 4 - 1})] : index
  %span_id = kernel.workgroup.id<z> : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1i = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %zero_f = scalar.constant 0.0 : f32
  %active, %length, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
  scf.if %active {{
  %layer_stride = index.constant {LB} : offset
  %layer_bytes = index.scale %layer, %layer_stride : index, offset -> offset
  %reader, %writer, %fresh = func.call @g_gdn_slots(%origins, %row, %start32, %layer_bytes) : (buffer, index, i32, offset) -> (offset, offset, i1)
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{C}xf32>
  %inv_view = buffer.view %inverse[%zero] : buffer -> view<[%tokens]x{2 * HK}xf32>
  %alpha_view = buffer.view %alpha[%zero] : buffer -> view<[%tokens]x{HV}xf32>
  %beta_view = buffer.view %beta[%zero] : buffer -> view<[%tokens]x{HV}xf32>
  %a_view = buffer.view %ssm_a[%zero] : buffer -> view<{HV}xf32>
  %bias_view = buffer.view %dt_bias[%zero] : buffer -> view<{HV}xf32>
  %out_view = buffer.view %out[%zero] : buffer -> view<[%tokens]x{VD}xf32>
  %state_view = buffer.view %arena[%reader] : buffer -> view<{HV * DV * DK}xf32>
  %next_view = buffer.view %arena[%writer] : buffer -> view<{HV * DV * DK}xf32>
  %cdk = index.constant {DK} : index
  %cdv = index.constant {DV} : index
  %chk = index.constant {HK} : index
  %column_offsets = vector.iota %c0 step %cdk : vector<4xindex>
  {kh_line()}
  %kh_base = index.mul %kh, %cdk : index
  %q_row0 = index.add %kh_base, %lane : index
  %k_skip = index.constant {HK * DK} : index
  %k_row0 = index.add %q_row0, %k_skip : index
  %q_norm_index = index.mul %kh, %c2 : index
  %k_norm_index = index.add %q_norm_index, %c1i : index
  %column_base = index.mul %tile, %c4 : index
  %head_columns = index.mul %vh, %cdv : index
  %state_column = index.add %head_columns, %column_base : index
  %state_column_base = index.mul %state_column, %cdk : index
  %state_index0 = index.add %state_column_base, %lane : index
{state_idx}
  %zero4 = vector.splat %zero_f : vector<4xf32>
  {", ".join(f"%init{m}" for m in range(RK))} = scf.if %fresh -> ({v4}) {{
    scf.yield {", ".join(["%zero4"] * RK)} : {v4}
  }} else {{
{gathers}
    scf.yield {", ".join(f"%g{m}" for m in range(RK))} : {v4}
  }}
  %a = view.load %a_view[%vh] : view<{HV}xf32> -> f32
  %dt_bias_value = view.load %bias_view[%vh] : view<{HV}xf32> -> f32
  %v_skip = index.constant {2 * HK * DK} : index
  %v_head = index.add %v_skip, %head_columns : index
  %v_channel = index.add %v_head, %column_base : index
  %out_channel = index.add %head_columns, %column_base : index
  %scale = scalar.constant {DK ** -0.5!r} : f32
  %scale_v = vector.splat %scale : vector<4xf32>
  {", ".join(f"%final{m}" for m in range(RK))} = scf.for %step = [%c0 to %length step %c1i]({", ".join(f"%old{m} = %init{m} : vector<4xf32>" for m in range(RK))}) -> ({v4}) {{
    %tabs_value = index.add %first, %step : index
    %tabs = index.assume %tabs_value [lt(%tabs_value, %tokens)] : index
    %q_inverse = view.load %inv_view[%tabs, %q_norm_index] : view<[%tokens]x{2 * HK}xf32> -> f32
    %k_inverse = view.load %inv_view[%tabs, %k_norm_index] : view<[%tokens]x{2 * HK}xf32> -> f32
    %alpha_raw = view.load %alpha_view[%tabs, %vh] : view<[%tokens]x{HV}xf32> -> f32
    %alpha_biased = scalar.addf %alpha_raw, %dt_bias_value : f32
    %alpha_positive = scalar.softplusf<afn> %alpha_biased : f32
    %gate = scalar.mulf %alpha_positive, %a : f32
    %decay = scalar.expf<afn> %gate : f32
    %decay_vector = vector.splat %decay : vector<4xf32>
{chr(10).join(loads)}
    %prediction = kernel.workgroup.reduce<addf> {psum} : vector<4xf32>
    %value = vector.load %x_view[%tabs, %v_channel] : view<[%tokens]x{C}xf32> -> vector<4xf32>
    %residual = vector.subf %value, %prediction : vector<4xf32>
    %beta_raw = view.load %beta_view[%tabs, %vh] : view<[%tokens]x{HV}xf32> -> f32
    %beta_value = scalar.logisticf<afn> %beta_raw : f32
    %beta_vector = vector.splat %beta_value : vector<4xf32>
    %delta = vector.mulf %residual, %beta_vector : vector<4xf32>
{chr(10).join(upd)}
    %output = kernel.workgroup.reduce<addf> {osum} : vector<4xf32>
    %leader = index.cmp eq, %lane, %c0 : index
    scf.if %leader {{
      %scaled = vector.mulf %output, %scale_v : vector<4xf32>
      vector.store %scaled, %out_view[%tabs, %out_channel] : vector<4xf32>, view<[%tokens]x{VD}xf32>
    }}
    scf.yield {", ".join(f"%new{m}" for m in range(RK))} : {v4}
  }}
{guard_open}{scatters}{guard_close}
  }}
  kernel.return
}}
""")
    RL = _row_lanes(DK, DV)
    if RL:
        # Spans of PF_SPAN+ tokens (prefill) keep the 8 lanes x 4 columns split that measured fastest there; shorter
        # spans (decode, verify) run g_gdn_recurrence (16 x 2 by default: +5 % decode on Qwen3.5-35B-A3B, -12 % prefill).
        # Each span runs in exactly one of the two kernels, chosen by its own length, so its numerics never depend on
        # what else shares the epoch.
        PF = (8, 4)
        T[-1] = _recurrence_rows(G, NG, LB, *RL, guard_open, guard_close, "short" if recurrence_pf(G) else None)
        if recurrence_pf(G):
            pf = _recurrence_rows(G, NG, LB, *PF, guard_open, guard_close, "long")
            T.append(pf.replace("g_gdn_recurrence", "g_gdn_recurrence_pf"))
    # ---- gated RMSNorm per value head: out = rmsnorm(o) * w * silu(z); padding tokens get zeros. One wave per head,
    # HPW heads per workgroup: 32-lane workgroups (tokens x HV of them) made this tiny kernel cost as much as the
    # recurrence at prefill (Qwen3.5-35B-A3B, 1k tokens: 32k workgroups per layer).
    HPW = max(h for h in (8, 4, 2, 1) if HV % h == 0)
    nl = "\n".join(f"""    %ni{m}c = index.constant {32 * m} : index
    %nd{m} = index.add %lane, %ni{m}c : index
    %nc{m}v = index.add %hbase, %nd{m} : index
    %nc{m} = index.assume %nc{m}v [range(%nc{m}v, 0, {VD - 1})] : index
    %r{m} = view.load %raw_view[%token, %nc{m}] : view<[%tokens]x{VD}xf32> -> f32
    %ss{m} = scalar.fmaf %r{m}, %r{m}, {"%zero_f" if m == 0 else f"%ss{m - 1}"} : f32""" for m in range(RV))
    nw = "\n".join(f"""    %wd{m} = view.load %w_view[%nd{m}] : view<{DV}xf32> -> f32
    %z{m} = view.load %z_view[%token, %nc{m}] : view<[%tokens]x{VD}xf32> -> f32
    %zs{m} = scalar.logisticf<afn> %z{m} : f32
    %zg{m} = scalar.mulf %z{m}, %zs{m} : f32
    %n{m}a = scalar.mulf %r{m}, %inv : f32
    %n{m}b = scalar.mulf %n{m}a, %wd{m} : f32
    %n{m} = scalar.mulf %n{m}b, %zg{m} : f32
    %res{m} = scf.select %live, %n{m}, %zero_f : f32""" for m in range(RV))
    # stores after every load: Loom orders each global load after earlier stores (a round trip per store/load pair)
    nw += "\n" + "\n".join(f"    view.store %res{m}, %o_view[%token, %nc{m}] : f32, view<[%tokens]x{VD}xf32>" for m in range(RV))
    T.append(f"""
kernel.def export("g_gdn_norm_gate") @g_gdn_norm_gate(%tokens: index) {{
  %c1 = index.constant 1 : index
  %groups = index.constant {HV // HPW} : index
  %lanes = index.constant {32 * HPW} : index
  kernel.launch.config workgroups(%tokens, %groups, %c1) workgroup_size(%lanes, %c1, %c1) : index
}} launch(%tokens_arg: index, %info: buffer, %raw: buffer, %weight: buffer, %z: buffer, %o: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %group_id = kernel.workgroup.id<y> : index
  %wave_id = kernel.subgroup.id : index
  %hpw = index.constant {HPW} : index
  %head_id = index.madd %group_id, %hpw, %wave_id : index
  %vh = index.assume %head_id [range(%head_id, 0, {HV - 1})] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0_i32 = scalar.constant 0 : i32
  %zero_f = scalar.constant 0.0 : f32
  %eps = scalar.constant {eps!r} : f32
  %count = scalar.constant {float(DV)!r} : f32
  %cdv = index.constant {DV} : index
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %raw_view = buffer.view %raw[%zero] : buffer -> view<[%tokens]x{VD}xf32>
  %z_view = buffer.view %z[%zero] : buffer -> view<[%tokens]x{VD}xf32>
  %o_view = buffer.view %o[%zero] : buffer -> view<[%tokens]x{VD}xf32>
  %w_view = buffer.view %weight[%zero] : buffer -> view<{DV}xf32>
  %hbase = index.mul %vh, %cdv : index
{nl}
    %sum = kernel.subgroup.reduce<addf> %ss{RV - 1} : f32
    %mean = scalar.divf %sum, %count : f32
    %stable = scalar.addf %mean, %eps : f32
    %inv = scalar.rsqrtf %stable : f32
{nw}
  kernel.return
}}
""")
    text = "\n".join(T)
    if not RL and os.environ.get("GDN_PREFETCH", "1") not in ("", "0"):
        text = _prefetch_recurrence(text, RK)
    fold = spec and fold_enabled(G)
    if fold:   # GDN_FOLD: the verification variants and the fold launches (draft/gen_draft.verify_program)
        main_class = "short" if recurrence_pf(G) else None
        text += "\n" + FOLD_RANK + "\n" + _conv_bank(text, G, CU) + "\n" + _recurrence_rows(G, NG, LB, *RL, guard_open, guard_close, main_class, record=True)
        text += "\n" + fold_kernels(G, NG, LB)
        if defer_enabled(G):
            text += "\n" + defer_kernels(G, NG)
    mode = chunk_mode(G)
    if mode:
        from gen_gdn_chunk import chunk_kernel
        ck = chunk_kernel(G, NG, LB, guard_open, guard_close)
        if mode == "pf":
            ck = ck.replace("g_gdn_chunk_rdna4", "g_gdn_recurrence_pf_rdna4")
        else:   # the sequential kernel's gfx12 twin runs the spans shorter than the chunk kernel's minimum
            twin = _recurrence_rows(G, NG, LB, *RL, guard_open, guard_close, "short")
            twin = twin.replace('kernel.def export("g_gdn_recurrence") @g_gdn_recurrence(', 'kernel.def target(@rdna4) export("g_gdn_recurrence_rdna4") @g_gdn_recurrence_rdna4(', 1)
            assert "g_gdn_recurrence_rdna4" in twin
            if fold:
                rt = _recurrence_rows(G, NG, LB, *RL, guard_open, guard_close, "short", record=True)
                twin += "\n" + rt.replace('kernel.def export("g_gdn_recurrence_rec") @g_gdn_recurrence_rec(', 'kernel.def target(@rdna4) export("g_gdn_recurrence_rec_rdna4") @g_gdn_recurrence_rec_rdna4(', 1)
            ck = twin + "\n" + ck
        text = text + "\n" + ck
    if fused_mode(G):   # gfx12 fused prefill (gen_gdn_fused.py): g_gdn_fa_rdna4 + g_gdn_fb_rdna4 replace conv .. norm_gate in GEMM-shape epochs
        from gen_gdn_fused import fused_kernels
        text = text + "\n" + fused_kernels(G, NG, LB, eps, _V_GROUPED[0], _V_GROUPED[1], guard_open, guard_close)
    return text


def fused_mode(G):
    """The two-launch fused gfx12 GDN prefill (choice prefill.gdn.fused2, env GDN_FUSED2): off unless the env says so, LOOM_RDNA4 is on and the
    heads are 128 x 128 with a 4-tap conv."""
    if os.environ.get("GDN_FUSED2", "") in ("", "0", "off", "no", "false") or os.environ.get("LOOM_RDNA4", "1") in ("", "0", "false", "no"):
        return False
    return G["DK"] == 128 and G["DV"] == 128 and G["K"] == 4 and G["HV"] % G["HK"] == 0


FOLD_W = 8   # bank rows per speculative span of the fold factors: the verify width's bound (config @llm.width <= 8; g_spec_accept: span length 1..8)
FOLD_SPANS = 32   # speculative spans per verification the banks hold (one per drafting row; build.py refuses GDN_FOLD with more rows). The banks are
                  # sized min(spans, FOLD_SPANS) * FOLD_W rows: the runtime prepares @llm_verify for every packing shape, (1024, 1024) included,
                  # so spans * FOLD_W rows would be 29 GB there (DevLab run 43821931908: RESOURCE_EXHAUSTED); this cap keeps the largest
                  # verification scratch below the capture it replaces.

PEND_HEAD = 256   # pending fold of a slot's layer (decode.spec.gdn_fold_defer): i32 step count, then [FOLD_W][HK*DK] keys, [FOLD_W][VD + HV] dk + decay

FOLD_RANK = r"""
// Rank of span `span` among the epoch's speculative spans (flag bit 4): its block of the fold banks (gen_gdn.fold_kernels).
func.def inline @g_gdn_spec_rank(%metadata: buffer, %spans: index, %span: index) -> (index) {
  %c16 = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %packets = buffer.view %metadata[%c16] : buffer -> view<[%spans]x9xi32>
  %four = scalar.constant 4 : i32
  %zero32 = scalar.constant 0 : i32
  %rank = scf.for %s = [%c0 to %span step %c1](%n = %c0 : index) -> (index) {
    %sb = index.assume %s [lt(%s, %spans)] : index
    %flags = view.load %packets[%sb, 6] : view<[%spans]x9xi32> -> i32
    %bit = scalar.andi %flags, %four : i32
    %is = scalar.cmpi ne, %bit, %zero32 : i32
    %inc = scf.select %is, %c1, %c0 : index
    %n1 = index.add %n, %inc : index
    scf.yield %n1 : index
  }
  func.return %rank : index
}
"""


def _rank_text(meta="%metadata", ok_extra=None):
    """%rank (the span's speculative rank) and %rank_ok (speculative and rank < FOLD_SPANS) for the kernel's %span."""
    return (f"  %rank_pub = func.call @g_gdn_publish({meta}, %spans, %span) : (buffer, index, index) -> (i1)\n"
            "  %rank_true = scalar.constant true : i1\n"
            "  %rank_spec = scalar.xori %rank_pub, %rank_true : i1\n"
            f"  %rank = func.call @g_gdn_spec_rank({meta}, %spans, %span) : (buffer, index, index) -> (index)\n"
            f"  %rank_cap = index.constant {FOLD_SPANS} : index\n"
            "  %rank_in = index.cmp ult, %rank, %rank_cap : index\n"
            "  %rank_ok = scalar.andi %rank_spec, %rank_in : i1\n")


def fold_enabled(G=None):
    """decode.spec.gdn_fold (GDN_FOLD): drafted verification records each step's fold factors instead of the replay inputs, and one launch
    folds the accepted steps into every layer's state (fold_kernels; draft/gen_draft.verify_program). Needs the row-split recurrence."""
    on = os.environ.get("GDN_FOLD", "").strip().lower() not in ("", "0", "false", "no", "off")
    return on and (G is None or bool(_row_lanes(G["DK"], G["DV"])))


def _defer_env():
    return os.environ.get("GDN_FOLD_DEFER", "").strip().lower() not in ("", "0", "false", "no", "off") and \
        os.environ.get("GDN_FOLD", "").strip().lower() not in ("", "0", "false", "no", "off")


def defer_enabled(G=None):
    """decode.spec.gdn_fold_defer (GDN_FOLD_DEFER, needs GDN_FOLD): the fold of a drafted verification is not applied after acceptance;
    the accepted steps' factors are stored in the writer slot as its pending fold (g_gdn_pend), and the next reader applies them:
    the next verification's recurrence in registers (it also writes the folded state back), every other span (advance epochs,
    publishing spans of a verification, forks from a retained slot) through g_gdn_flush + g_gdn_pend_clear at the program start.
    One state read per drafted step instead of two (gen_gdn.defer_kernels)."""
    return _defer_env() and (G is None or bool(G.get("pend_stride")))


def _bank_view(name, buf, width, define_rows=True):
    """`%{name}` = view<[%bank_rows]x{width}xf32> of layer %layer's block of a fold bank ([layer][rank * FOLD_W + step][width] f32);
    define_rows: also define %bank_rows = min(%spans, FOLD_SPANS) * FOLD_W (once per kernel)."""
    rows = (f"  %bank_rows_w = index.constant {FOLD_W} : index\n"
            f"  %bank_rows_c = index.constant {FOLD_SPANS} : index\n"
            f"  %bank_rows_s = index.min %spans, %bank_rows_c : index\n"
            f"  %bank_rows = index.mul %bank_rows_s, %bank_rows_w : index\n") if define_rows else ""
    return (rows + f"  %{name}_rb = index.constant {width * 4} : offset\n"
            f"  %{name}_lb = index.scale %bank_rows, %{name}_rb : index, offset -> offset\n"
            f"  %{name}_at = index.scale %layer, %{name}_lb : index, offset -> offset\n"
            f"  %{name} = buffer.view %{buf}[%{name}_at] : buffer -> view<[%bank_rows]x{width}xf32>\n")


def _bank_row(name, span, step, rows):
    """%{name} = span * FOLD_W + step, clamped below `rows` (the store/load using it is guarded by step < FOLD_W)."""
    return (f"  %{name}_w = index.constant {FOLD_W} : index\n"
            f"  %{name}_s = index.mul {span}, %{name}_w : index\n"
            f"  %{name}_v = index.add %{name}_s, {step} : index\n"
            f"  %{name}_one = index.constant 1 : index\n"
            f"  %{name}_max = index.sub {rows}, %{name}_one : index\n"
            f"  %{name}_c = index.min %{name}_v, %{name}_max : index\n"
            f"  %{name} = index.assume %{name}_c [range(%{name}_c, 0, 2147483646), lt(%{name}_c, {rows})] : index\n")


def _conv_bank(text, G, CU):
    """g_gdn_conv -> g_gdn_conv_bank (GDN_FOLD verification): the same outputs, plus each step's raw input (the conv history of the
    fold) in %bankx and its key channels' output (the fold's k) in %bankk, at row span * FOLD_W + step (steps < FOLD_W)."""
    C, K, HK, DK = G["C"], G["K"], G["HK"], G["DK"]
    a, b = text.index('kernel.def export("g_gdn_conv")'), text.index("kernel.return\n}", text.index('kernel.def export("g_gdn_conv")'))
    t = text[a:b + len("kernel.return\n}") + 1]

    def sub(old, new):
        nonlocal t
        assert t.count(old) == 1, old[:60]
        t = t.replace(old, new)
    sub('kernel.def export("g_gdn_conv") @g_gdn_conv(', 'kernel.def export("g_gdn_conv_bank") @g_gdn_conv_bank(')
    sub("%weight: buffer, %out: buffer) {", "%weight: buffer, %out: buffer, %bankx: buffer, %bankk: buffer) {")
    rel = "    %rel = index.assume %rel_value [range(%rel_value, 0, 8191)] : index\n"
    sub(rel, rel + "\n".join("  " + l for l in (_bank_view("bx_view", "bankx", C) + _bank_view("bk_view", "bankk", HK * DK, False)
                                                  + _rank_text() + _bank_row("bank_row", "%rank", "%rel", "%bank_rows")).rstrip("\n").split("\n"))
        + f"\n    %bank_cw = index.constant {FOLD_W} : index\n    %bank_rel = index.cmp ult, %rel, %bank_cw : index\n    %bank_ok = scalar.andi %bank_rel, %rank_ok : i1\n"
        f"    %bank_k0 = index.constant {HK * DK} : index\n    %bank_k1 = index.constant {2 * HK * DK} : index\n")
    stores = []
    for u in range(CU):
        sub(f"%val_u{u} = scf.if %cu{u}_in -> (f32) {{", f"%val_u{u}, %raw_u{u} = scf.if %cu{u}_in -> (f32, f32) {{")
        sub(f"        scf.yield %silu_u{u} : f32\n      }} else {{\n        scf.yield %zero_f : f32\n      }}",
            f"        scf.yield %silu_u{u}, %x{K - 1}_u{u} : f32, f32\n      }} else {{\n        scf.yield %zero_f, %zero_f : f32, f32\n      }}")
        stores.append(f"""      %bk{u}_ok = scalar.andi %cu{u}_in, %bank_ok : i1
      scf.if %bk{u}_ok {{
        %bc{u} = index.assume %cu{u}_c [range(%cu{u}_c, 0, {C - 1})] : index
        view.store %raw_u{u}, %bx_view[%bank_row, %bc{u}] : f32, view<[%bank_rows]x{C}xf32>
        %bk{u}_ge = index.cmp uge, %bc{u}, %bank_k0 : index
        %bk{u}_lt = index.cmp ult, %bc{u}, %bank_k1 : index
        %bk{u}_isk = scalar.andi %bk{u}_ge, %bk{u}_lt : i1
        scf.if %bk{u}_isk {{
          %bkc{u}_v = index.sub %bc{u}, %bank_k0 : index
          %bkc{u} = index.assume %bkc{u}_v [range(%bkc{u}_v, 0, {HK * DK - 1})] : index
          view.store %val_u{u}, %bk_view[%bank_row, %bkc{u}] : f32, view<[%bank_rows]x{HK * DK}xf32>
        }}
      }}""")
    last = f"        view.store %val_u{CU - 1}, %out_view[%token, %cs{CU - 1}] : f32, view<[%tokens]x{C}xf32>\n      }}\n"
    sub(last, last + "\n".join(stores) + "\n")
    return t


def _record_rows(text, G, NG, CW, RL):
    """g_gdn_recurrence -> g_gdn_recurrence_rec: the same arithmetic, plus each step's dk (per value column) and decay (per head) in
    %bank, stored with the step's outputs (the RL-step batches: no store between the loop's loads)."""
    VD, HV = G["VD"], G["HV"]
    VC = f"vector<{CW}xf32>"

    def sub(old, new, n=1):
        nonlocal text
        assert text.count(old) == n, (old[:60], text.count(old))
        text = text.replace(old, new)
    sub('kernel.def export("g_gdn_recurrence") @g_gdn_recurrence(', 'kernel.def export("g_gdn_recurrence_rec") @g_gdn_recurrence_rec(')
    sub("%dt_bias: buffer, %out: buffer) {", "%dt_bias: buffer, %out: buffer, %bank: buffer) {")
    view = f"  %out_view = buffer.view %out[%zero] : buffer -> view<[%tokens]x{VD}xf32>\n"
    sub(view, view + _bank_view("bd_view", "bank", VD + HV) + _rank_text())
    head = [l for l in text.split("\n") if "= scf.for %step = [%c0 to %length step %c1i](" in l]
    assert len(head) == 1
    h = head[0]
    assert h.endswith(f"%ob = %zero4 : {VC}) -> (" + h.split(") -> (")[1]) and h.endswith(f", {VC}) {{")
    nh = h.replace(", %ob_unused = scf.for", ", %ob_unused, %db_unused, %gb_unused = scf.for", 1)
    nh = nh.replace(f"%ob = %zero4 : {VC}) -> (", f"%ob = %zero4 : {VC}, %db = %zero4 : {VC}, %gb = %zero_f : f32) -> (", 1)
    nh = nh[:-len(") {")] + f", {VC}, f32) {{"
    sub(h, nh)
    sel = f"    %ob_next = scf.select %ob_mine, %outv, %ob : {VC}\n"
    sub(sel, sel + f"    %dkvec = vector.from_elements {', '.join(f'%dk{j}' for j in range(CW))} : {VC}\n"
                   f"    %db_next = scf.select %ob_mine, %dkvec, %db : {VC}\n"
                   f"    %gb_next = scf.select %ob_mine, %decay, %gb : f32\n")
    st = f"        vector.store %ob_next, %out_view[%ob_row, %out_channel] : {VC}, view<[%tokens]x{VD}xf32>\n"
    rec = (st + "        %rb_step = index.add %ob_step0, %r : index\n"
           f"        %rb_cw = index.constant {FOLD_W} : index\n"
           "        %rb_in = index.cmp ult, %rb_step, %rb_cw : index\n"
           "        %rb_ok = scalar.andi %rb_in, %rank_ok : i1\n"
           "        scf.if %rb_ok {\n"
           + "\n".join("  " + l for l in _bank_row("rb_row", "%rank", "%rb_step", "%bank_rows").rstrip("\n").split("\n")) + "\n"
           f"          vector.store %db_next, %bd_view[%rb_row, %out_channel] : {VC}, view<[%bank_rows]x{VD + HV}xf32>\n"
           "          %rb_t0 = index.cmp eq, %tile, %c0 : index\n"
           "          %rb_g0 = index.cmp eq, %cg, %c0 : index\n"
           "          %rb_dw = scalar.andi %rb_t0, %rb_g0 : i1\n"
           "          scf.if %rb_dw {\n"
           f"            %rb_dc0 = index.constant {VD} : index\n"
           "            %rb_dc1 = index.add %rb_dc0, %vh : index\n"
           f"            %rb_dc = index.assume %rb_dc1 [range(%rb_dc1, {VD}, {VD + HV - 1})] : index\n"
           f"            view.store %gb_next, %bd_view[%rb_row, %rb_dc] : f32, view<[%bank_rows]x{VD + HV}xf32>\n"
           "          }\n"
           "        }\n")
    sub(st, rec)
    y = [l for l in text.split("\n") if l.startswith("    scf.yield ") and "%ob_next : " in l]
    assert len(y) == 1
    sub(y[0], y[0].replace(", %ob_next : ", ", %ob_next, %db_next, %gb_next : ", 1) + f", {VC}, f32")
    return text


def fold_kernels(G, NG, LB):
    """The fold of a drafted verification (GDN_FOLD), one launch each for all layers after g_spec_accept:
    g_gdn_fold: every speculative span's accepted steps applied to its state, S = decay S + k dk^T per step, from the factors
    g_gdn_recurrence_rec recorded and the raw keys g_gdn_conv_bank kept: the replay recurrence's arithmetic per state element
    (mulf, then fmaf), so the published state is bit-identical to the replay's. Workgroup: (layer, value head, 32 columns), 256 lanes
    of 4 key rows x 8 columns per pass. g_gdn_fold_hist: the conv history (the last K-1 pre-conv inputs) from g_gdn_conv_bank's raw
    inputs, as g_gdn_conv_commit over the accepted length. Spans and lengths come from the committed packet (length = consumed),
    speculative spans from the epoch's metadata (flag bit 4). Launch shape [spans, layers]: layers = the model's gated delta-net layers
    (a kernel test folds layer 0 alone)."""
    HK, HV, DK, DV, K, C, VD, SB = G["HK"], G["HV"], G["DK"], G["DV"], G["K"], G["C"], G["VD"], G["state_bytes"]
    NS = HV * DV * DK
    assert DK == 128 and DV % 32 == 0
    CB = DV // 32   # 32-column blocks per head: 8 columns per pass x NP passes
    span_head = f"""  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %zero_f = scalar.constant 0.0 : f32
  %active, %length, %start32, %row, %first = func.call @g_gdn_span(%committed, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
  %publish = func.call @g_gdn_publish(%metadata, %spans, %span) : (buffer, index, index) -> (i1)
  %true = scalar.constant true : i1
  %spec = scalar.xori %publish, %true : i1
  %mine_s = scalar.andi %active, %spec : i1
  %rank = func.call @g_gdn_spec_rank(%metadata, %spans, %span) : (buffer, index, index) -> (index)
  %rank_cap = index.constant {FOLD_SPANS} : index
  %rank_in = index.cmp ult, %rank, %rank_cap : index
  %mine = scalar.andi %mine_s, %rank_in : i1
"""
    NP = 4   # 8-column passes per workgroup: every pass's state is loaded before the step loop, stored after it
    loads, stores, dks, upd = [], [], [], []
    for p in range(NP):
        loads.append(f"""    %b{p}_o = index.constant {8 * p} : index
    %b{p}_c = index.add %colbase, %b{p}_o : index
    %b{p} = index.assume %b{p}_c [range(%b{p}_c, 0, {DV - 1})] : index
    %hb{p} = index.add %head_cols, %b{p} : index
    %dc{p} = index.assume %hb{p} [range(%hb{p}, 0, {VD - 1})] : index
    %si{p}_r = index.mul %hb{p}, %c128 : index
    %si{p}_v = index.add %si{p}_r, %a4 : index
    %si{p} = index.assume %si{p}_v [range(%si{p}_v, 0, {NS - 4})] : index
    %s{p}_0 = scf.if %fresh -> (vector<4xf32>) {{
      scf.yield %zero_v : vector<4xf32>
    }} else {{
      %s{p}_l = vector.load %state_view[%si{p}] : view<{NS}xf32> -> vector<4xf32>
      scf.yield %s{p}_l : vector<4xf32>
    }}""")
        dks.append(f"      %dk{p} = view.load %bd_view[%br, %dc{p}] : view<[%bank_rows]x{VD + HV}xf32> -> f32")
        upd.append(f"""      %dkv{p} = vector.splat %dk{p} : vector<4xf32>
      %sd{p} = vector.mulf %s{p}, %decay_vector : vector<4xf32>
      %n{p} = vector.fmaf %kv, %dkv{p}, %sd{p} : vector<4xf32>""")
        stores.append(f"    vector.store %s{p}_f, %next_view[%si{p}] : vector<4xf32>, view<{NS}xf32>")
    V4 = ", ".join(["vector<4xf32>"] * NP)
    steps = f"""    %{"_f, %".join(f"s{p}" for p in range(NP))}_f = scf.for %step = [%c0 to %length step %c1]({", ".join(f"%s{p} = %s{p}_0 : vector<4xf32>" for p in range(NP))}) -> ({V4}) {{
{chr(10).join("    " + l for l in _bank_row("br", "%rank", "%step", "%bank_rows").rstrip(chr(10)).split(chr(10)))}
      %kv = vector.load %bk_view[%br, %krow] : view<[%bank_rows]x{HK * DK}xf32> -> vector<4xf32>
{chr(10).join(dks)}
      %dec = view.load %bd_view[%br, %dcol] : view<[%bank_rows]x{VD + HV}xf32> -> f32
      %decay_vector = vector.splat %dec : vector<4xf32>
{chr(10).join(upd)}
      scf.yield {", ".join(f"%n{p}" for p in range(NP))} : {V4}
    }}"""
    passes = loads + [steps] + stores
    fold = f"""
// Fold of a drafted verification (gen_gdn.fold_kernels): the accepted steps of every speculative span into each layer's state.
kernel.def export("g_gdn_fold") @g_gdn_fold(%spans: index, %layers: index) {{
  %c1 = index.constant 1 : index
  %blocks = index.constant {HV * CB} : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%blocks, %layers, %spans) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %metadata: buffer, %committed: buffer, %origins: buffer, %arena: buffer, %bankk: buffer, %bankd: buffer) {{
  %block_id = kernel.workgroup.id<x> : index
  %block = index.assume %block_id [range(%block_id, 0, {HV * CB - 1})] : index
  %layer_id = kernel.workgroup.id<y> : index
  %layer = index.assume %layer_id [range(%layer_id, 0, 65534)] : index
  %span_id = kernel.workgroup.id<z> : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
{span_head}  scf.if %mine {{
    %layer_stride = index.constant {LB} : offset
    %layer_bytes = index.scale %layer, %layer_stride : index, offset -> offset
    %reader, %writer, %fresh = func.call @g_gdn_slots(%origins, %row, %start32, %layer_bytes) : (buffer, index, i32, offset) -> (offset, offset, i1)
    %state_view = buffer.view %arena[%reader] : buffer -> view<{NS}xf32>
    %next_view = buffer.view %arena[%writer] : buffer -> view<{NS}xf32>
{_bank_view("bk_view", "bankk", HK * DK)}{_bank_view("bd_view", "bankd", VD + HV, False)}
    %ccb = index.constant {CB} : index
    %vh_v = index.div %block, %ccb : index
    %vh = index.assume %vh_v [range(%vh_v, 0, {HV - 1})] : index
    %cblk_v = index.rem %block, %ccb : index
    %cblk = index.assume %cblk_v [range(%cblk_v, 0, {CB - 1})] : index
    %chk = index.constant {HK} : index
    {kh_line()}
    %c4 = index.constant 4 : index
    %c32 = index.constant 32 : index
    %c128 = index.constant {DK} : index
    %cdv = index.constant {DV} : index
    %quad_v = index.rem %lane, %c32 : index
    %quad = index.assume %quad_v [range(%quad_v, 0, 31)] : index
    %a4_v = index.mul %quad, %c4 : index
    %a4 = index.assume %a4_v [range(%a4_v, 0, {DK - 4})] : index
    %sub_v = index.div %lane, %c32 : index
    %sub = index.assume %sub_v [range(%sub_v, 0, 7)] : index
    %cb32 = index.mul %cblk, %c32 : index
    %colbase = index.add %cb32, %sub : index
    %head_cols = index.mul %vh, %cdv : index
    %kh_rows = index.mul %kh, %c128 : index
    %krow_v = index.add %kh_rows, %a4 : index
    %krow = index.assume %krow_v [range(%krow_v, 0, {HK * DK - 4})] : index
    %dcol_c = index.constant {VD} : index
    %dcol_v = index.add %dcol_c, %vh : index
    %dcol = index.assume %dcol_v [range(%dcol_v, {VD}, {VD + HV - 1})] : index
    %zero_v = vector.splat %zero_f : vector<4xf32>
{chr(10).join(passes)}
  }}
  kernel.return
}}
"""
    # conv history: the last K-1 inputs of [old history, accepted raw inputs]
    reads, writes = [], []
    for k in range(K - 1):
        need = K - 1 - k
        reads.append(f"""      %need{k} = index.constant {need} : index
      %has{k} = index.cmp uge, %length, %need{k} : index
      %val{k} = scf.if %has{k} -> (f32) {{
        %st{k} = index.sub %length, %need{k} : index
{chr(10).join("    " + l for l in _bank_row(f"hr{k}", "%rank", f"%st{k}", "%bank_rows").rstrip(chr(10)).split(chr(10)))}
        %v{k} = view.load %bx_view[%hr{k}, %ch] : view<[%bank_rows]x{C}xf32> -> f32
        scf.yield %v{k} : f32
      }} else {{
        %o{k} = scf.if %fresh -> (f32) {{
          scf.yield %zero_f : f32
        }} else {{
          %hk{k}a = index.constant {k} : index
          %hk{k}b = index.add %hk{k}a, %length : index
          %hk{k} = index.assume %hk{k}b [range(%hk{k}b, 0, {K - 2})] : index
          %hv{k} = view.load %old_view[%hk{k}, %ch] : view<{K - 1}x{C}xf32> -> f32
          scf.yield %hv{k} : f32
        }}
        scf.yield %o{k} : f32
      }}""")
        writes.append(f"      view.store %val{k}, %new_view[{k}, %ch] : f32, view<{K - 1}x{C}xf32>")
    HCB = (C + 255) // 256
    hist = f"""
kernel.def export("g_gdn_fold_hist") @g_gdn_fold_hist(%spans: index, %layers: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cblocks = index.constant {HCB} : index
  kernel.launch.config workgroups(%spans, %cblocks, %layers) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %metadata: buffer, %committed: buffer, %origins: buffer, %arena: buffer, %bankx: buffer) {{
  %span_id = kernel.workgroup.id<x> : index
  %cblk_id = kernel.workgroup.id<y> : index
  %layer_id = kernel.workgroup.id<z> : index
  %layer = index.assume %layer_id [range(%layer_id, 0, 65534)] : index
  %thread_id = kernel.workitem.id<x> : index
  %thread = index.assume %thread_id [range(%thread_id, 0, 255)] : index
{span_head}  scf.if %mine {{
    %layer_stride = index.constant {LB} : offset
    %layer_bytes = index.scale %layer, %layer_stride : index, offset -> offset
    %reader, %writer, %fresh = func.call @g_gdn_slots(%origins, %row, %start32, %layer_bytes) : (buffer, index, i32, offset) -> (offset, offset, i1)
    %history_skip = index.constant {SB} : offset
    %old_at = index.add %reader, %history_skip : offset
    %new_at = index.add %writer, %history_skip : offset
    %old_view = buffer.view %arena[%old_at] : buffer -> view<{K - 1}x{C}xf32>
    %new_view = buffer.view %arena[%new_at] : buffer -> view<{K - 1}x{C}xf32>
{_bank_view("bx_view", "bankx", C)}    %c256 = index.constant 256 : index
    %cC = index.constant {C} : index
    %cblk_off = index.mul %cblk_id, %c256 : index
    %c = index.add %thread, %cblk_off : index
    %inside = index.cmp ult, %c, %cC : index
    scf.if %inside {{
      %ch = index.assume %c [range(%c, 0, {C - 1})] : index
{chr(10).join(reads)}
{chr(10).join(writes)}
    }}
  }}
  kernel.return
}}
"""
    return fold + hist


def _pend_views(G, which, pre):
    """Views of the pending fold of layer %layer of the span's `which` ("reader" / "writer") slot (the block after every layer's state):
    %{pre}_hdr (view<4xi32>: [0] = steps), %{pre}_k [FOLD_W][HK*DK] keys, %{pre}_d [FOLD_W][VD + HV] dk per value column + decay per head."""
    HK, DK, VD, HV, PB, PS = G["HK"], G["DK"], G["VD"], G["HV"], G["pend_base"], G["pend_stride"]
    return (f"  %{pre}_z = index.constant 0 : offset\n"
            f"  %{pre}_rb, %{pre}_wb, %{pre}_fr = func.call @g_gdn_slots(%origins, %row, %start32, %{pre}_z) : (buffer, index, i32, offset) -> (offset, offset, i1)\n"
            f"  %{pre}_lb = index.constant {PS} : offset\n"
            f"  %{pre}_lo = index.scale %layer, %{pre}_lb : index, offset -> offset\n"
            f"  %{pre}_pb = index.constant {PB} : offset\n"
            f"  %{pre}_a0 = index.add %{pre}_{which[0]}b, %{pre}_pb : offset\n"
            f"  %{pre}_at = index.add %{pre}_a0, %{pre}_lo : offset\n"
            f"  %{pre}_hdr = buffer.view %arena[%{pre}_at] : buffer -> view<4xi32>\n"
            f"  %{pre}_k_off = index.constant {PEND_HEAD} : offset\n"
            f"  %{pre}_k_at = index.add %{pre}_at, %{pre}_k_off : offset\n"
            f"  %{pre}_k = buffer.view %arena[%{pre}_k_at] : buffer -> view<{FOLD_W}x{HK * DK}xf32>\n"
            f"  %{pre}_d_off = index.constant {PEND_HEAD + FOLD_W * HK * DK * 4} : offset\n"
            f"  %{pre}_d_at = index.add %{pre}_at, %{pre}_d_off : offset\n"
            f"  %{pre}_d = buffer.view %arena[%{pre}_d_at] : buffer -> view<{FOLD_W}x{VD + HV}xf32>\n")


def _defer_rows(text, G, CW, RL):
    """g_gdn_recurrence_rec under decode.spec.gdn_fold_defer: the reader slot's pending fold (the previous verification's accepted
    steps) is applied to the state in registers first, S = fma(k, dk, S * decay) per step, g_gdn_fold's op order (bit-identical to
    the eager fold), and a speculative span writes that folded state to its writer slot (its pending fold is replaced after
    acceptance by g_gdn_pend). Publishing spans find no pending fold: g_gdn_flush ran at the program start."""
    HK, DK, VD, HV = G["HK"], G["DK"], G["VD"], G["HV"]
    RPL = DK // RL
    NS = HV * G["DV"] * DK
    VR = f"vector<{RPL}xf32>"
    J = range(CW)

    def sub(old, new):
        nonlocal text
        assert text.count(old) == 1, (old[:60], text.count(old))
        text = text.replace(old, new)
    anchor = f"  %a = view.load %a_view[%vh] : view<{HV}xf32> -> f32\n"
    V = ", ".join([VR] * CW)
    # the folded-state write is non-temporal: it is read again only by the next verification; written through the caches it slowed the
    # next weight matvecs by ~0.35 ms per drafted step (R9700 profile, defold exp4/exp5)
    nt = " {cache_scope = device, cache_temporal = non_temporal}"
    code = (_pend_views(G, "reader", "dfr")
            + "  %dfr_zero32 = scalar.constant 0 : i32\n"
            "  %dfr_n32 = scf.if %fresh -> (i32) {\n    scf.yield %dfr_zero32 : i32\n  } else {\n"
            "    %dfr_l = view.load %dfr_hdr[0] : view<4xi32> -> i32\n    scf.yield %dfr_l : i32\n  }\n"
            f"  %dfr_nb = scalar.assume %dfr_n32 [range(%dfr_n32, 0, {FOLD_W})] : i32\n"
            "  %dfr_n = index.cast %dfr_nb : i32 to index\n"
            f"  %dfr_dc_c = index.constant {VD} : index\n  %dfr_dc_v = index.add %dfr_dc_c, %vh : index\n"
            f"  %dfr_dc = index.assume %dfr_dc_v [range(%dfr_dc_v, {VD}, {VD + HV - 1})] : index\n"
            + "".join(f"  %dfr_c{j} = index.assume %sc{j} [range(%sc{j}, 0, {VD - 1})] : index\n" for j in J)
            + f"  {', '.join(f'%pinit{j}' for j in J)} = scf.for %dfr_p = [%c0 to %dfr_n step %c1i]({', '.join(f'%dfr_s{j} = %init{j} : {VR}' for j in J)}) -> ({V}) {{\n"
            f"    %dfr_pb = index.assume %dfr_p [range(%dfr_p, 0, {FOLD_W - 1})] : index\n"
            f"    %dfr_kv = vector.load %dfr_k[%dfr_pb, %q_row] : view<{FOLD_W}x{HK * DK}xf32> -> {VR}\n"
            f"    %dfr_dec = view.load %dfr_d[%dfr_pb, %dfr_dc] : view<{FOLD_W}x{VD + HV}xf32> -> f32\n"
            f"    %dfr_decv = vector.splat %dfr_dec : {VR}\n"
            + "".join(f"    %dfr_dk{j} = view.load %dfr_d[%dfr_pb, %dfr_c{j}] : view<{FOLD_W}x{VD + HV}xf32> -> f32\n"
                      f"    %dfr_dkv{j} = vector.splat %dfr_dk{j} : {VR}\n"
                      f"    %dfr_sd{j} = vector.mulf %dfr_s{j}, %dfr_decv : {VR}\n"
                      f"    %dfr_n{j} = vector.fmaf %dfr_kv, %dfr_dkv{j}, %dfr_sd{j} : {VR}\n" for j in J)
            + f"    scf.yield {', '.join(f'%dfr_n{j}' for j in J)} : {V}\n  }}\n"
            "  scf.if %rank_spec {\n"
            + "".join(f"    vector.store %pinit{j}, %next_view[%si{j}]{nt} : {VR}, view<{NS}xf32>\n" for j in J)
            + "  }\n")
    sub(anchor, code + anchor)
    for j in J:
        sub(f"%s{j} = %init{j} : {VR}", f"%s{j} = %pinit{j} : {VR}")
    return text


def defer_kernels(G, NG):
    """decode.spec.gdn_fold_defer (defer_enabled): the pending fold of a slot's layer = the accepted steps of its last drafted
    verification, applied by the slot's next reader.
    g_gdn_pend (after g_spec_accept, in place of g_gdn_fold): each speculative span's accepted steps' factors (rows of the verify banks)
    into its writer slot's pending block, and the step count (0 when nothing was accepted: the verification already wrote the folded
    state there). Launch [spans, layers, FOLD_W].
    g_gdn_flush (program start, @llm_advance and @llm_verify): every live publishing span's reader slot with a pending fold gets it
    applied in place (g_gdn_fold's op order; a slot read by several spans is flushed by the first). Launch [spans, layers, 1].
    g_gdn_pend_clear (right after): those reader slots' and every live publishing span's writer slot's step count = 0 (a writer is
    republished whole by the span). Launch [spans, layers]."""
    HK, HV, DK, DV, VD, NS, LB = G["HK"], G["HV"], G["DK"], G["DV"], G["VD"], G["HV"] * G["DV"] * G["DK"], G["layer_bytes"]
    KW, DW = HK * DK, VD + HV
    assert KW % 4 == 0 and DW % 4 == 0 and NS % (4 * 256 * 8) == 0
    head = """  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
  %layer = index.assume %layer_id [range(%layer_id, 0, 65534)] : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c4 = index.constant 4 : index
  %c256 = index.constant 256 : index
  %zero32 = scalar.constant 0 : i32
  %true = scalar.constant true : i1
  %active, %length, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
  %publish = func.call @g_gdn_publish(%metadata, %spans, %span) : (buffer, index, index) -> (i1)
  %layer_stride = index.constant {LB} : offset
  %layer_bytes = index.scale %layer, %layer_stride : index, offset -> offset
""".replace("{LB}", str(LB))

    def copy(src, dst, width, row_src):
        n4 = width // 4
        return f"""      %{dst}_n4 = index.constant {n4} : index
      scf.for %{dst}_i = [%lane to %{dst}_n4 step %c256] {{
        %{dst}_ib = index.assume %{dst}_i [range(%{dst}_i, 0, {n4 - 1})] : index
        %{dst}_e = index.mul %{dst}_ib, %c4 : index
        %{dst}_eb = index.assume %{dst}_e [range(%{dst}_e, 0, {width - 4})] : index
        %{dst}_v = vector.load %{src}[{row_src}, %{dst}_eb] : view<[%bank_rows]x{width}xf32> -> vector<4xf32>
        vector.store %{dst}_v, %{dst}[%step, %{dst}_eb] : vector<4xf32>, view<{FOLD_W}x{width}xf32>
      }}
"""
    pend = f"""
// decode.spec.gdn_fold_defer: the accepted steps' fold factors of every speculative span into its writer slot's pending fold (gen_gdn.defer_kernels).
kernel.def export("g_gdn_pend") @g_gdn_pend(%spans: index, %layers: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %steps = index.constant {FOLD_W} : index
  kernel.launch.config workgroups(%spans, %layers, %steps) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %metadata: buffer, %committed: buffer, %origins: buffer, %arena: buffer, %bankk: buffer, %bankd: buffer) {{
  %span_id = kernel.workgroup.id<x> : index
  %layer_id = kernel.workgroup.id<y> : index
  %step_id = kernel.workgroup.id<z> : index
  %step = index.assume %step_id [range(%step_id, 0, {FOLD_W - 1})] : index
  %lane_id = kernel.workitem.id<x> : index
{head}  %spec = scalar.xori %publish, %true : i1
  %mine = scalar.andi %active, %spec : i1
  scf.if %mine {{
    %cactive, %clength, %cstart32, %crow, %cfirst = func.call @g_gdn_span(%committed, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
    %rank = func.call @g_gdn_spec_rank(%metadata, %spans, %span) : (buffer, index, index) -> (index)
    %rank_cap = index.constant {FOLD_SPANS} : index
    %rank_in = index.cmp ult, %rank, %rank_cap : index
    %ok = scalar.andi %cactive, %rank_in : i1
    %fw = index.constant {FOLD_W} : index
    %clen = index.min %clength, %fw : index
    %n = scf.select %ok, %clen, %c0 : index
    %reader, %writer, %fresh = func.call @g_gdn_slots(%origins, %row, %start32, %layer_bytes) : (buffer, index, i32, offset) -> (offset, offset, i1)
{chr(10).join("  " + l for l in _pend_views(G, "writer", "pw").rstrip(chr(10)).split(chr(10)))}
    %s0 = index.cmp eq, %step, %c0 : index
    %l0 = index.cmp eq, %lane, %c0 : index
    %first_lane = scalar.andi %s0, %l0 : i1
    scf.if %first_lane {{
      %n_i = index.cast %n : index to i32
      view.store %n_i, %pw_hdr[0] : i32, view<4xi32>
    }}
    %in = index.cmp ult, %step, %n : index
    scf.if %in {{
{chr(10).join("    " + l for l in (_bank_view("bk_view", "bankk", KW) + _bank_view("bd_view", "bankd", DW, False) + _bank_row("br", "%rank", "%step", "%bank_rows")).rstrip(chr(10)).split(chr(10)))}
{copy("bk_view", "pw_k", KW, "%br")}{copy("bd_view", "pw_d", DW, "%br")}    }}
  }}
  kernel.return
}}
"""
    Z = 1   # one workgroup per (span, layer): the empty launch of every advance epoch stays small; a pending fold (once per turn) loops over the layer
    per = NS // (4 * 256 * Z)   # vec4 slots per lane
    flush = f"""
// The reader slot's base (g_gdn_slots without the layer) as i64, to compare spans' readers.
func.def inline @g_gdn_reader64(%origins: buffer, %row: index, %start32: i32) -> (i64) {{
  %record_stride = index.constant 40 : offset
  %bounded_row = index.assume %row [range(%row, 0, 2147483646)] : index
  %record_offset = index.scale %bounded_row, %record_stride : index, offset -> offset
  %records = buffer.view %origins[%record_offset] : buffer -> view<5xi64>
  %read64 = view.load %records[0] : view<5xi64> -> i64
  %write64 = view.load %records[1] : view<5xi64> -> i64
  %fork64 = view.load %records[3] : view<5xi64> -> i64
  %start64 = scalar.extui %start32 : i32 to i64
  %at_fork = scalar.cmpi eq, %start64, %fork64 : i64
  %reader64 = scf.select %at_fork, %read64, %write64 : i64
  func.return %reader64 : i64
}}

// decode.spec.gdn_fold_defer: a live publishing span's reader slot with a pending fold gets it applied in place (gen_gdn.defer_kernels).
kernel.def export("g_gdn_flush") @g_gdn_flush(%spans: index, %layers: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %parts = index.constant {Z} : index
  kernel.launch.config workgroups(%spans, %layers, %parts) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %metadata: buffer, %origins: buffer, %arena: buffer) {{
  %span_id = kernel.workgroup.id<x> : index
  %layer_id = kernel.workgroup.id<y> : index
  %part_id = kernel.workgroup.id<z> : index
  %part = index.assume %part_id [range(%part_id, 0, {Z - 1})] : index
  %lane_id = kernel.workitem.id<x> : index
{head}  %mine = scalar.andi %active, %publish : i1
  scf.if %mine {{
    %reader, %writer, %fresh = func.call @g_gdn_slots(%origins, %row, %start32, %layer_bytes) : (buffer, index, i32, offset) -> (offset, offset, i1)
    %stale = scalar.xori %fresh, %true : i1
    scf.if %stale {{
{chr(10).join("    " + l for l in _pend_views(G, "reader", "pf").rstrip(chr(10)).split(chr(10)))}
      %n32 = view.load %pf_hdr[0] : view<4xi32> -> i32
      %nb = scalar.assume %n32 [range(%n32, 0, {FOLD_W})] : i32
      %n = index.cast %nb : i32 to index
      %has = index.cmp ugt, %n, %c0 : index
      scf.if %has {{
        // the first live publishing span reading this slot flushes it
        %reader64 = func.call @g_gdn_reader64(%origins, %row, %start32) : (buffer, index, i32) -> (i64)
        %dup = scf.for %o = [%c0 to %span step %c1](%d = %false_init : i1) -> (i1) {{
          %ob = index.assume %o [lt(%o, %spans)] : index
          %oa, %ol, %os32, %orow, %of = func.call @g_gdn_span(%metadata, %spans, %ob) : (buffer, index, index) -> (i1, index, i32, index, index)
          %op = func.call @g_gdn_publish(%metadata, %spans, %ob) : (buffer, index, index) -> (i1)
          %or64 = func.call @g_gdn_reader64(%origins, %orow, %os32) : (buffer, index, i32) -> (i64)
          %olive = scalar.andi %oa, %op : i1
          %ozero = scalar.cmpi eq, %os32, %zero32 : i32
          %ostale = scalar.xori %ozero, %true : i1
          %oread = scalar.andi %olive, %ostale : i1
          %osame = scalar.cmpi eq, %or64, %reader64 : i64
          %ohit = scalar.andi %oread, %osame : i1
          %d1 = scalar.ori %d, %ohit : i1
          scf.yield %d1 : i1
        }}
        %go = scalar.xori %dup, %true : i1
        scf.if %go {{
          %state_view = buffer.view %arena[%reader] : buffer -> view<{NS}xf32>
          %cdk = index.constant {DK} : index
          %cdv = index.constant {DV} : index
          %chk = index.constant {HK} : index
          %per = index.constant {per} : index
          %part_base = index.constant {per * 256} : index
          %pb = index.mul %part, %part_base : index
          scf.for %it = [%c0 to %per step %c1] {{
            %iw = index.mul %it, %c256 : index
            %i0 = index.add %pb, %iw : index
            %i = index.add %i0, %lane : index
            %e_v = index.mul %i, %c4 : index
            %e = index.assume %e_v [range(%e_v, 0, {NS - 4})] : index
            %col_v = index.div %e, %cdk : index
            %col = index.assume %col_v [range(%col_v, 0, {VD - 1})] : index
            %a4_v = index.rem %e, %cdk : index
            %a4 = index.assume %a4_v [range(%a4_v, 0, {DK - 4})] : index
            %vh_v = index.div %col, %cdv : index
            %vh = index.assume %vh_v [range(%vh_v, 0, {HV - 1})] : index
            {kh_line()}
            %kb = index.mul %kh, %cdk : index
            %krow_v = index.add %kb, %a4 : index
            %krow = index.assume %krow_v [range(%krow_v, 0, {KW - 4})] : index
            %dcol_c = index.constant {VD} : index
            %dcol_v = index.add %dcol_c, %vh : index
            %dcol = index.assume %dcol_v [range(%dcol_v, {VD}, {DW - 1})] : index
            %s0 = vector.load %state_view[%e] : view<{NS}xf32> -> vector<4xf32>
            %sf = scf.for %p = [%c0 to %n step %c1](%s = %s0 : vector<4xf32>) -> (vector<4xf32>) {{
              %pp = index.assume %p [range(%p, 0, {FOLD_W - 1})] : index
              %kv = vector.load %pf_k[%pp, %krow] : view<{FOLD_W}x{KW}xf32> -> vector<4xf32>
              %dk = view.load %pf_d[%pp, %col] : view<{FOLD_W}x{DW}xf32> -> f32
              %dec = view.load %pf_d[%pp, %dcol] : view<{FOLD_W}x{DW}xf32> -> f32
              %dkv = vector.splat %dk : vector<4xf32>
              %decay_vector = vector.splat %dec : vector<4xf32>
              %sd = vector.mulf %s, %decay_vector : vector<4xf32>
              %nx = vector.fmaf %kv, %dkv, %sd : vector<4xf32>
              scf.yield %nx : vector<4xf32>
            }}
            vector.store %sf, %state_view[%e] : vector<4xf32>, view<{NS}xf32>
          }}
        }}
      }}
    }}
  }}
  kernel.return
}}
""".replace("%false_init", "%false_c").replace("  %true = scalar.constant true : i1\n", "  %true = scalar.constant true : i1\n  %false_c = scalar.constant false : i1\n", 1)
    clear = f"""
// decode.spec.gdn_fold_defer: after g_gdn_flush, the flushed reader slots' and the live publishing spans' writer slots' pending step count = 0.
kernel.def export("g_gdn_pend_clear") @g_gdn_pend_clear(%spans: index, %layers: index) {{
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  kernel.launch.config workgroups(%spans, %layers, %c1) workgroup_size(%c32, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %metadata: buffer, %origins: buffer, %arena: buffer) {{
  %span_id = kernel.workgroup.id<x> : index
  %layer_id = kernel.workgroup.id<y> : index
  %lane_id = kernel.workitem.id<x> : index
{head.replace("[range(%lane_id, 0, 255)]", "[range(%lane_id, 0, 31)]")}  %l0 = index.cmp eq, %lane, %c0 : index
  %live = scalar.andi %active, %publish : i1
  %mine = scalar.andi %live, %l0 : i1
  scf.if %mine {{
    %reader, %writer, %fresh = func.call @g_gdn_slots(%origins, %row, %start32, %layer_bytes) : (buffer, index, i32, offset) -> (offset, offset, i1)
{chr(10).join("  " + l for l in _pend_views(G, "writer", "cw").rstrip(chr(10)).split(chr(10)))}
    view.store %zero32, %cw_hdr[0] : i32, view<4xi32>
    %stale = scalar.xori %fresh, %true : i1
    scf.if %stale {{
{chr(10).join("    " + l for l in _pend_views(G, "reader", "cr").rstrip(chr(10)).split(chr(10)))}
      view.store %zero32, %cr_hdr[0] : i32, view<4xi32>
    }}
  }}
  kernel.return
}}
"""
    return pend + flush + clear


PF_SPAN = 64   # spans of at least this many tokens (the GEMM epoch shape) run the prefill form


def single_form(draft_kind):
    """MTP-drafted hybrids keep one recurrence form (8 x 4) for every span. With the decode/prefill split (#350) Sage-A1.5
    (MTP depth 3) failed the agent gate's history reuse at one turn ('cached 0'); the 8 x 4 form everywhere passes every
    gate (run 43351934089). The other drafters (DFlash, n-gram) passed with the split. The environment still wins."""
    if draft_kind == "mtp":
        os.environ.setdefault("GDN_ROW_LANES", "8")
        os.environ.setdefault("GDN_COLS", "4")


def recurrence_pf(G):
    """True when prefill epochs launch g_gdn_recurrence_pf (the 8 x 4 split) beside the decode form g_gdn_recurrence."""
    RL = _row_lanes(G["DK"], G["DV"])
    return bool(RL) and "GDN_ROW_LANES" not in os.environ and RL != (8, 4) and _row_fits(G["DK"], G["DV"], 8, 4)


def chunk_mode(G):
    """The gfx12 chunked prefill kernel (gen_gdn_chunk.py): None (off: env GDN_CHUNK unset/0, LOOM_RDNA4=0, other head dims), "pf" (it
    is the gfx12 twin of g_gdn_recurrence_pf) or "group" (single-form models: g_gdn_recurrence's gfx12 twin takes the spans shorter than
    PF_SPAN and a dual-group launch adds g_gdn_chunk_rdna4 for the longer ones)."""
    if os.environ.get("GDN_CHUNK", "") in ("", "0", "off", "no", "false") or os.environ.get("LOOM_RDNA4", "1") in ("", "0", "false", "no"):
        return None
    if G["DK"] != 128 or G["DV"] != 128 or G["HV"] % G["HK"] or not _row_lanes(G["DK"], G["DV"]):
        return None
    return "pf" if recurrence_pf(G) else "group"


def _row_fits(DK, DV, rl, cw):
    return rl in (1, 2, 4, 8, 16, 32) and cw in (2, 4, 8) and DK % rl == 0 and (DK // rl) % 4 == 0 and DV % (cw * 32 // rl) == 0


def _row_lanes(DK, DV):
    """Lanes per value column in the row-split recurrence (0: the original 32-lane form). GDN_ROW_LANES overrides."""
    RL = int(os.environ.get("GDN_ROW_LANES", "16"))
    CW = int(os.environ.get("GDN_COLS", "2"))
    if _row_fits(DK, DV, RL, CW):
        return (RL, CW)
    if "GDN_ROW_LANES" not in os.environ and "GDN_COLS" not in os.environ and _row_fits(DK, DV, 8, 4):
        return (8, 4)   # the former defaults, for head dims the new ones do not divide
    return 0


def _recurrence_rows(G, NG, LB, RL, CW, guard_open, guard_close, span_class=None, record=False):
    """g_gdn_recurrence, row-split form: a 32-lane workgroup owns CW * (32 / RL) value columns of one head; each
    group of RL lanes owns CW columns and lane r of the group the key rows [r * DK / RL, (r + 1) * DK / RL) of
    them (DK / RL * CW state values in registers). S^T k and S^T q are per-lane dot products over the lane's
    rows plus a log2(RL)-step xor-shuffle sum inside the group, instead of two 32-lane reductions per 4 columns;
    the key/query L2 norms and the decay fold into the per-column scalars (no per-element normalization), and
    the token's decay/beta are computed once per workgroup. The step's inputs are loaded one step ahead and the
    outputs stored in RL-step batches (lane r keeps step r's CW outputs), so no global store sits between the
    loop's loads. Same arithmetic per state element as the 32-lane form; the sums are ordered differently.
    GDN_ROW_LANES (default 16) / GDN_COLS (default 2) pick RL / CW.
    record=True: g_gdn_recurrence_rec, which also writes each step's fold factors to %bank (fold_kernels): the update scalar
    dk = beta (v - decay S^T k) / |k| per value column and the decay per head, at row span * FOLD_W + step (steps < FOLD_W)."""
    HK, HV, DK, DV, C, VD = G["HK"], G["HV"], G["DK"], G["DV"], G["C"], G["VD"]
    RPL, CPW = DK // RL, CW * (32 // RL)
    VC = f"vector<{CW}xf32>"
    VR = f"vector<{RPL}xf32>"
    NS = HV * DV * DK
    ld = [("qv", VR, f"vector.load %x_view[@T@, %q_row] : view<[%tokens]x{C}xf32> -> {VR}"),
          ("kv", VR, f"vector.load %x_view[@T@, %k_row] : view<[%tokens]x{C}xf32> -> {VR}"),
          ("vv", VC, f"vector.load %x_view[@T@, %v_channel] : view<[%tokens]x{C}xf32> -> {VC}"),
          ("qi", "f32", f"view.load %inv_view[@T@, %q_norm_index] : view<[%tokens]x{2 * HK}xf32> -> f32"),
          ("ki", "f32", f"view.load %inv_view[@T@, %k_norm_index] : view<[%tokens]x{2 * HK}xf32> -> f32"),
          ("al", "f32", f"view.load %alpha_view[@T@, %vh] : view<[%tokens]x{HV}xf32> -> f32"),
          ("be", "f32", f"view.load %beta_view[@T@, %vh] : view<[%tokens]x{HV}xf32> -> f32")]
    xs = [1 << i for i in range(RL.bit_length() - 1)]
    nl = "\n"
    J = range(CW)

    def group_sum(name):
        out, cur = [], name
        for d in xs:
            out.append(f"    {cur}_x{d}, {cur}_u{d} = kernel.subgroup.shuffle<xor> {cur}, %x{d}, %w32 : f32, i32, i32\n"
                       f"    {name}_s{d} = scalar.addf {cur}, {cur}_x{d} : f32")
            cur = f"{name}_s{d}"
        return nl.join(out), cur
    body = []
    for j in J:
        red, pj = group_sum(f"%p{j}")
        body.append(f"    %p{j} = vector.dotf<reassoc|contract> %s{j}, %kv, %zero_f : {VR}, {VR}, f32\n{red}\n"
                    f"    %v{j} = vector.extract %vv[{j}] : {VC} -> f32\n"
                    f"    %pred{j} = scalar.mulf {pj}, %kdecay : f32\n"
                    f"    %res{j} = scalar.subf %v{j}, %pred{j} : f32\n"
                    f"    %del{j} = scalar.mulf %res{j}, %beta_value : f32\n"
                    f"    %dk{j} = scalar.mulf %del{j}, %ki : f32\n"
                    f"    %dkv{j} = vector.splat %dk{j} : {VR}\n"
                    f"    %sd{j} = vector.mulf %s{j}, %decay_vector : {VR}\n"
                    f"    %n{j} = vector.fmaf %kv, %dkv{j}, %sd{j} : {VR}")
    for j in J:
        red, oj = group_sum(f"%o{j}")
        body.append(f"    %o{j} = vector.dotf<reassoc|contract> %n{j}, %qv, %zero_f : {VR}, {VR}, f32\n{red}\n"
                    f"    %out{j} = scalar.mulf {oj}, %qscale : f32")
    V4 = ", ".join([VR] * CW)
    fin = ", ".join(f"%final{j}" for j in J)
    lt = ", ".join(t for _, t, _ in ld)
    sidx = nl.join(f"  %sc{j}c = index.constant {j} : index\n  %sc{j} = index.add %out_channel, %sc{j}c : index\n"
                   f"  %sb{j} = index.mul %sc{j}, %cdk : index\n  %si{j}_value = index.add %sb{j}, %row_off : index\n"
                   f"  %si{j} = index.assume %si{j}_value [range(%si{j}_value, 0, {NS - RPL})] : index" for j in J)
    gl = nl.join(f"    %g{j} = vector.load %state_view[%si{j}] : view<{NS}xf32> -> {VR}" for j in J)
    pf = nl.join(f"  %pf_{n} = {rhs.replace('@T@', '%pf_first')}" for n, _, rhs in ld)
    nx = nl.join(f"    %nx_{n} = {rhs.replace('@T@', '%nx_tabs')}" for n, _, rhs in ld)
    iters = (", ".join(f"%s{j} = %init{j} : {VR}" for j in J) + ", " +
             ", ".join(f"%{n} = %pf_{n} : {t}" for n, t, _ in ld) + f", %ob = %zero4 : {VC}")
    stores = nl.join(f"    vector.store %final{j}, %next_view[%si{j}] : {VR}, view<{NS}xf32>" for j in J)
    shuf = nl.join(f"  %x{d} = scalar.constant {d} : i32" for d in xs)
    cls_flag, cls_text = "active", ""
    if span_class:   # a span runs in the kernel of its own length class: its numerics never depend on the epoch's other spans
        cmp_ = "ult" if span_class == "short" else "uge"
        cls_flag = "mine"
        cls_text = f"""  %cpf_len = index.constant {PF_SPAN} : index
  %in_class = index.cmp {cmp_}, %length, %cpf_len : index
  %mine = scalar.andi %active, %in_class : i1
"""
    text = f"""
kernel.def export("g_gdn_recurrence") @g_gdn_recurrence(%spans: index) {{
  %c1 = index.constant 1 : index
  %heads = index.constant {HV} : index
  %tiles = index.constant {DV // CPW} : index
  %c32 = index.constant 32 : index
  kernel.launch.config workgroups(%heads, %tiles, %spans) workgroup_size(%c32, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %layer_arg: index, %metadata: buffer, %origins: buffer, %arena: buffer, %x: buffer, %inverse: buffer, %alpha: buffer, %beta: buffer, %ssm_a: buffer, %dt_bias: buffer, %out: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {NG - 1})] : index
  %head_id = kernel.workgroup.id<x> : index
  %vh = index.assume %head_id [range(%head_id, 0, {HV - 1})] : index
  %tile_id = kernel.workgroup.id<y> : index
  %tile = index.assume %tile_id [range(%tile_id, 0, {DV // CPW - 1})] : index
  %span_id = kernel.workgroup.id<z> : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1i = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %zero_f = scalar.constant 0.0 : f32
  %w32 = scalar.constant 32 : i32
{shuf}
  %active, %length, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
{cls_text}  scf.if %{cls_flag} {{
  %layer_stride = index.constant {LB} : offset
  %layer_bytes = index.scale %layer, %layer_stride : index, offset -> offset
  %reader, %writer, %fresh = func.call @g_gdn_slots(%origins, %row, %start32, %layer_bytes) : (buffer, index, i32, offset) -> (offset, offset, i1)
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{C}xf32>
  %inv_view = buffer.view %inverse[%zero] : buffer -> view<[%tokens]x{2 * HK}xf32>
  %alpha_view = buffer.view %alpha[%zero] : buffer -> view<[%tokens]x{HV}xf32>
  %beta_view = buffer.view %beta[%zero] : buffer -> view<[%tokens]x{HV}xf32>
  %a_view = buffer.view %ssm_a[%zero] : buffer -> view<{HV}xf32>
  %bias_view = buffer.view %dt_bias[%zero] : buffer -> view<{HV}xf32>
  %out_view = buffer.view %out[%zero] : buffer -> view<[%tokens]x{VD}xf32>
  %state_view = buffer.view %arena[%reader] : buffer -> view<{NS}xf32>
  %next_view = buffer.view %arena[%writer] : buffer -> view<{NS}xf32>
  %cdk = index.constant {DK} : index
  %cdv = index.constant {DV} : index
  %chk = index.constant {HK} : index
  %crl = index.constant {RL} : index
  %crpl = index.constant {RPL} : index
  %ccpw = index.constant {CPW} : index
  %ccw = index.constant {CW} : index
  %r_value = index.rem %lane, %crl : index
  %r = index.assume %r_value [range(%r_value, 0, {RL - 1})] : index
  %cg_value = index.div %lane, %crl : index
  %cg = index.assume %cg_value [range(%cg_value, 0, {32 // RL - 1})] : index
  %row_off = index.mul %r, %crpl : index
  {kh_line()}
  %kh_base = index.mul %kh, %cdk : index
  %q_row_value = index.add %kh_base, %row_off : index
  %q_row = index.assume %q_row_value [range(%q_row_value, 0, {HK * DK - RPL})] : index
  %k_skip = index.constant {HK * DK} : index
  %k_row_value = index.add %q_row, %k_skip : index
  %k_row = index.assume %k_row_value [range(%k_row_value, {HK * DK}, {2 * HK * DK - RPL})] : index
  %q_norm_index = index.mul %kh, %c2 : index
  %k_norm_index = index.add %q_norm_index, %c1i : index
  %tile_col = index.mul %tile, %ccpw : index
  %grp_col = index.mul %cg, %ccw : index
  %col0 = index.add %tile_col, %grp_col : index
  %head_columns = index.mul %vh, %cdv : index
  %out_channel_value = index.add %head_columns, %col0 : index
  %out_channel = index.assume %out_channel_value [range(%out_channel_value, 0, {VD - CW})] : index
  %v_skip = index.constant {2 * HK * DK} : index
  %v_channel_value = index.add %v_skip, %out_channel : index
  %v_channel = index.assume %v_channel_value [range(%v_channel_value, {2 * HK * DK}, {C - CW})] : index
{sidx}
  %zero_r = vector.splat %zero_f : {VR}
  {", ".join(f"%init{j}" for j in J)} = scf.if %fresh -> ({V4}) {{
    scf.yield {", ".join(["%zero_r"] * CW)} : {V4}
  }} else {{
{gl}
    scf.yield {", ".join(f"%g{j}" for j in J)} : {V4}
  }}
  %a = view.load %a_view[%vh] : view<{HV}xf32> -> f32
  %dt_bias_value = view.load %bias_view[%vh] : view<{HV}xf32> -> f32
  %scale = scalar.constant {DK ** -0.5!r} : f32
  %zero4 = vector.splat %zero_f : {VC}
  %last_step = index.sub %length, %c1i : index
  %pf_first = index.assume %first [lt(%first, %tokens)] : index
{pf}
  {fin}, {", ".join(f"%pf_unused_{n}" for n, _, _ in ld)}, %ob_unused = scf.for %step = [%c0 to %length step %c1i]({iters}) -> ({V4}, {lt}, {VC}) {{
    %nx_step_raw = index.add %step, %c1i : index
    %nx_step = index.min %nx_step_raw, %last_step : index
    %nx_tabs_value = index.add %first, %nx_step : index
    %nx_tabs = index.assume %nx_tabs_value [lt(%nx_tabs_value, %tokens)] : index
{nx}
    %alpha_biased = scalar.addf %al, %dt_bias_value : f32
    %alpha_positive = scalar.softplusf<afn> %alpha_biased : f32
    %gate = scalar.mulf %alpha_positive, %a : f32
    %decay = scalar.expf<afn> %gate : f32
    %decay_vector = vector.splat %decay : {VR}
    %kdecay = scalar.mulf %ki, %decay : f32
    %beta_value = scalar.logisticf<afn> %be : f32
    %qscale = scalar.mulf %qi, %scale : f32
{nl.join(body)}
    %outv = vector.from_elements {", ".join(f"%out{j}" for j in J)} : {VC}
    %ob_slot = index.rem %step, %crl : index
    %ob_mine = index.cmp eq, %r, %ob_slot : index
    %ob_next = scf.select %ob_mine, %outv, %ob : {VC}
    %ob_cl = index.constant {RL - 1} : index
    %ob_full = index.cmp eq, %ob_slot, %ob_cl : index
    %ob_last = index.cmp eq, %step, %last_step : index
    %ob_flush = scalar.ori %ob_full, %ob_last : i1
    scf.if %ob_flush {{
      %ob_live = index.cmp ule, %r, %ob_slot : index
      scf.if %ob_live {{
        %ob_group = index.div %step, %crl : index
        %ob_step0 = index.mul %ob_group, %crl : index
        %ob_base = index.add %first, %ob_step0 : index
        %ob_row_value = index.add %ob_base, %r : index
        %ob_row = index.assume %ob_row_value [lt(%ob_row_value, %tokens)] : index
        vector.store %ob_next, %out_view[%ob_row, %out_channel] : {VC}, view<[%tokens]x{VD}xf32>
      }}
    }}
    scf.yield {", ".join(f"%n{j}" for j in J)}, {", ".join(f"%nx_{n}" for n, _, _ in ld)}, %ob_next : {V4}, {lt}, {VC}
  }}
{guard_open}{stores}{guard_close}
  }}
  kernel.return
}}
"""
    if record:
        text = _record_rows(text, G, NG, CW, RL)
        if defer_enabled(G):
            text = _defer_rows(text, G, CW, RL)
    return text


def _prefetch_recurrence(text, RK):
    """Software-pipeline g_gdn_recurrence: the step's inputs (q, k, v, the norms, alpha, beta) don't depend on
    the state, so step s+1's loads are issued during step s and carried in the loop. A step was otherwise one
    global-load latency plus the two reductions (the recurrence is a serial chain per workgroup)."""
    import re
    a = text.index('kernel.def export("g_gdn_recurrence")')
    b = text.index("kernel.def export", a + 10)
    k = text[a:b]
    loads = []   # (name, type, rhs with @T@ for the token index)
    for line in k.split("\n"):
        m = re.match(r"\s*(%(?:q_inverse|k_inverse|alpha_raw|beta_raw|value|[qk]\d+_raw)) = ((?:view|vector)\.load %\w+\[)%tabs(, .*-> (\S+))$", line)
        if m:
            loads.append((m.group(1), m.group(4), m.group(2) + "@T@" + m.group(3)))
    assert len(loads) == 5 + 2 * RK, loads
    names = {n for n, _, _ in loads}
    body = [l for l in k.split("\n") if not (re.match(r"\s*(%\S+) = ", l) and re.match(r"\s*(%\S+) = ", l).group(1) in names)]
    k = "\n".join(body)
    # Row indices of q/k (loop-invariant) move before the loop.
    rows = [l for l in k.split("\n") if re.match(r"\s*%(qr\d+c|qr\d+|kr\d+) = ", l)]
    k = "\n".join(l for l in k.split("\n") if l not in rows)
    fl = re.search(r"\n(  (%final\S*(?:, %final\S*)*) = scf\.for %step = \[%c0 to %length step %c1i\]\((.*)\) -> \((.*)\) \{)\n", k)
    head, finals, iters, types = fl.group(1), fl.group(2), fl.group(3), fl.group(4)
    pro = ["  " + l.strip() for l in rows]
    pro += ["  %last_step = index.sub %length, %c1i : index",
            "  %pf_first = index.assume %first [lt(%first, %tokens)] : index"]
    pro += [f"  %pf_{n[1:]} = {rhs.replace('@T@', '%pf_first')}" for n, _, rhs in loads]
    new_head = (f"  {finals}, " + ", ".join(f"%pf_unused_{i}" for i in range(len(loads))) +
                f" = scf.for %step = [%c0 to %length step %c1i]({iters}, " +
                ", ".join(f"{n} = %pf_{n[1:]} : {t}" for n, t, _ in loads) + f") -> ({types}, " + ", ".join(t for _, t, _ in loads) + ") {")
    k = k.replace(head, "\n".join(pro) + "\n" + new_head, 1)
    tabs = "    %tabs = index.assume %tabs_value [lt(%tabs_value, %tokens)] : index\n"
    assert tabs in k
    nxt = ["    %nx_step_raw = index.add %step, %c1i : index",
           "    %nx_step = index.min %nx_step_raw, %last_step : index",
           "    %nx_tabs_value = index.add %first, %nx_step : index",
           "    %nx_tabs = index.assume %nx_tabs_value [lt(%nx_tabs_value, %tokens)] : index"]
    nxt += [f"    %nx_{n[1:]} = {rhs.replace('@T@', '%nx_tabs')}" for n, _, rhs in loads]
    k = k.replace(tabs, tabs + "\n".join(nxt) + "\n", 1)
    # Output stores in batches of 32 steps: lane j keeps step j's output and the wave stores them together.
    # A store each step made every following global load wait for the store to complete (the compiler
    # cannot tell the output from the inputs), a memory round trip per step on the serial chain.
    st = re.search(r"    %leader = index\.cmp eq, %lane, %c0 : index\n    scf\.if %leader \{\n      %scaled = vector\.mulf %output, %scale_v : (vector<\d+xf32>)\n"
                   r"      vector\.store %scaled, %out_view\[%tabs, %out_channel\] : vector<\d+xf32>, (view<[^>]*>)\n    \}\n", k)
    vt, ot = st.group(1), st.group(2)
    k = k.replace(st.group(0), f"""    %scaled = vector.mulf %output, %scale_v : {vt}
    %ob_c32 = index.constant 32 : index
    %ob_slot = index.rem %step, %ob_c32 : index
    %ob_mine = index.cmp eq, %lane, %ob_slot : index
    %ob_next = scf.select %ob_mine, %scaled, %ob : {vt}
    %ob_c31 = index.constant 31 : index
    %ob_full = index.cmp eq, %ob_slot, %ob_c31 : index
    %ob_last = index.cmp eq, %step, %last_step : index
    %ob_flush = scalar.ori %ob_full, %ob_last : i1
    scf.if %ob_flush {{
      %ob_live = index.cmp ule, %lane, %ob_slot : index
      scf.if %ob_live {{
        %ob_group = index.div %step, %ob_c32 : index
        %ob_step0 = index.mul %ob_group, %ob_c32 : index
        %ob_base = index.add %first, %ob_step0 : index
        %ob_row_value = index.add %ob_base, %lane : index
        %ob_row = index.assume %ob_row_value [lt(%ob_row_value, %tokens)] : index
        vector.store %ob_next, %out_view[%ob_row, %out_channel] : {vt}, {ot}
      }}
    }}
""", 1)
    assert new_head.endswith(") {")
    k = k.replace(new_head, new_head[:-3].replace(") -> (", f", %ob = %zero4 : {vt}) -> (", 1).replace(
        " = scf.for %step", ", %ob_unused = scf.for %step", 1) + f", {vt}) {{", 1)
    y = re.search(r"\n    scf\.yield ((?:%new\d+(?:, )?)+) : (.*)\n  \}\n", k)
    k = k.replace(y.group(0), f"\n    scf.yield {y.group(1)}, " + ", ".join(f"%nx_{n[1:]}" for n, _, _ in loads) +
                  f", %ob_next : {y.group(2)}, " + ", ".join(t for _, t, _ in loads) + f", {vt}\n  }}\n", 1)
    return text[:a] + k + text[b:]


SPLIT_QG = r"""
// qwen35-style attention: attn_q packs [query (D) | gate (D)] per head; split into q and gate.
kernel.def export("g_split_qgate") @g_split_qgate(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %qg: buffer, %q: buffer, %gate: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %thread_id = kernel.workitem.id<x> : index
  %thread = index.assume %thread_id [range(%thread_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %cqd = index.constant @@QD@@ : index
  %cd = index.constant @@D@@ : index
  %c256 = index.constant 256 : index
  %qg_view = buffer.view %qg[%zero] : buffer -> view<[%tokens]x@@QH@@x2x@@D@@xf32>
  %q_view = buffer.view %q[%zero] : buffer -> view<[%tokens]x@@QH@@x@@D@@xf32>
  %g_view = buffer.view %gate[%zero] : buffer -> view<[%tokens]x@@QH@@x@@D@@xf32>
  scf.for %i = [%thread to %cqd step %c256] {
    %bi = index.assume %i [range(%i, 0, @@QD_LESS@@)] : index
    %h_value = index.div %bi, %cd : index
    %h = index.assume %h_value [range(%h_value, 0, @@QH_LESS@@)] : index
    %d_value = index.rem %bi, %cd : index
    %d = index.assume %d_value [range(%d_value, 0, @@D_LESS@@)] : index
    %qv = view.load %qg_view[%token, %h, 0, %d] : view<[%tokens]x@@QH@@x2x@@D@@xf32> -> f32
    %gv = view.load %qg_view[%token, %h, 1, %d] : view<[%tokens]x@@QH@@x2x@@D@@xf32> -> f32
    view.store %qv, %q_view[%token, %h, %d] : f32, view<[%tokens]x@@QH@@x@@D@@xf32>
    view.store %gv, %g_view[%token, %h, %d] : f32, view<[%tokens]x@@QH@@x@@D@@xf32>
  }
  kernel.return
}
"""


def split_qgate(QH, D):
    return (SPLIT_QG.replace("@@QD_LESS@@", str(QH * D - 1)).replace("@@QH_LESS@@", str(QH - 1)).replace("@@D_LESS@@", str(D - 1))
            .replace("@@QD@@", str(QH * D)).replace("@@QH@@", str(QH)).replace("@@D@@", str(D)))
