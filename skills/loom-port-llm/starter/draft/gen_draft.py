"""Speculative decoding for starter ports: target verification, device acceptance, and the MTP drafter.

gen_llm.py imports this module when LLM_DRAFT is set (build.py sets it; see draft/README.md). Everything
here is emitted into llm.loom next to the trunk, reusing the trunk's own kernels and layer text:

- @llm_advance (ordinary epochs) keeps the trunk, but its final normalized hidden rows land in the
  `residual` allocation so the drafter's catch-up (warm) stage can read them.
- @llm_verify runs the same trunk over mixed known and speculative spans. Speculative spans (device plan
  flag bit 4, credit in bits 8..15; draft/build_draft.py's encode fixup) do not publish recurrent state;
  gated delta-net layers capture their pre-conv inputs instead. The head predicts every speculative
  position (one synthetic record per position, so sampling keys on the true position and stays exact),
  g_spec_accept keeps the longest matching prefix (stopping at EOS and the output credit), and a replay
  pass re-runs the gated delta-net conv + recurrence over just the accepted inputs and publishes them.
  Attention K/V beyond the accepted frontier is unreachable and gets overwritten later.
- MTP (NextN block, cfg["mtp"]): @llm_mtp_warm rebuilds the block's K/V at every consumed position from
  (embedding of token P, target hidden P-1) and keeps the last hidden per row in the carry; it runs after
  every epoch, so drafting works from any position up to the full context. @llm_mtp_propose runs
  draft_depth autoregressive rounds of the block for the proposing spans and writes the proposals into
  the verifier's input slots on device (no host round trip).

Packet (starter format): metadata = 16-byte header {count, eos, mixed, 0} + count 9-word records
{length, position, temperature, row, cursor, top_p, flags, top_k, seed} (+ the M-RoPE offset table for
vision ports). Committed/replay/row/draft packets use the same format, so the trunk kernels read them.
Adapted from hrx-loom-library (kernel/sequence/qwen3_8/accept.loom, model/Qwen/Qwen3.8-27B/program/
{verify,mtp}.loom); Apache-2.0.
"""
import json
import os
import re

DRAFT = json.loads(os.environ.get("LLM_DRAFT") or "null")


def enabled():
    return bool(DRAFT) and DRAFT.get("kind") in ("mtp", "dflash", "ngram")


def _ids(view, idx, stride_mixed, dst):
    """%dst = token id at packed index %idx of buffer view `view` (flat i32, stride 8 when mixed)."""
    return f"""      {dst}_e = index.mul {idx}, %id_stride : index
      {dst}_b = index.assume {dst}_e [lt({dst}_e, %n_ids)] : index
      {dst} = view.load {view}[{dst}_b] : view<[%n_ids]xi32> -> i32"""


def kernels(g):
    """Device kernels. g: dict of trunk facts (H, HG, EPS, GDN geometry or None, MROPE flag, NG)."""
    H, HG, EPS, MROPE = g["H"], g["HG"], g["EPS"], g["MROPE"]
    T = []
    mrope_copy = ""
    # ---- verification rows: one synthetic head record per prediction
    T.append(r"""
// Expands the epoch into head records: a selected known span predicts after its last input; a
// speculative span (flags bit 4) predicts after each of its inputs. Record j of a speculative span keeps
// the span's cursor and position with length j + 1, so the head reads input cursor + j and draws with
// the counter of position + j + 1 (the same draw ordinary decode makes there).
kernel.def export("g_spec_rows") @g_spec_rows() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%spans_arg: index, %rows_arg: index, %metadata: buffer, %rows_meta: buffer) {
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 65536)] : index
  %zero = index.constant 0 : offset
  %c16 = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %c4_i32 = scalar.constant 4 : i32
  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %eos = view.load %header[1] : view<4xi32> -> i32
  %count_b = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count_v = index.cast %count_b : i32 to index
  %count = index.min %count_v, %spans : index
  %records = buffer.view %metadata[%c16] : buffer -> view<[%spans]x9xi32>
  %out = buffer.view %rows_meta[%c16] : buffer -> view<[%rows]x9xi32>
  %c9 = index.constant 9 : index
  %total = scf.for %s = [%c0 to %count step %c1](%r = %c0 : index) -> (index) {
    %sb = index.assume %s [range(%s, 0, 2147483646), lt(%s, %spans)] : index
    %length32 = view.load %records[%sb, 0] : view<[%spans]x9xi32> -> i32
    %flags = view.load %records[%sb, 6] : view<[%spans]x9xi32> -> i32
    %sel_bits = scalar.andi %flags, %c1_i32 : i32
    %selected = scalar.cmpi ne, %sel_bits, %c0_i32 : i32
    %spec_bits = scalar.andi %flags, %c4_i32 : i32
    %spec = scalar.cmpi ne, %spec_bits, %c0_i32 : i32
    %length_b = scalar.assume %length32 [range(%length32, 0, 8)] : i32
    %spec_rows32 = scf.select %spec, %length_b, %c1_i32 : i32
    %n32 = scf.select %selected, %spec_rows32, %c0_i32 : i32
    %n_b = scalar.assume %n32 [range(%n32, 0, 8)] : i32
    %n = index.cast %n_b : i32 to index
    %next = scf.for %j = [%c0 to %n step %c1](%rr = %r : index) -> (index) {
      %fits = index.cmp ult, %rr, %rows : index
      scf.if %fits {
        %rb = index.assume %rr [range(%rr, 0, 2147483646), lt(%rr, %rows)] : index
        scf.for %w = [%c0 to %c9 step %c1] {
          %wb = index.assume %w [range(%w, 0, 8)] : index
          %v = view.load %records[%sb, %wb] : view<[%spans]x9xi32> -> i32
          view.store %v, %out[%rb, %wb] : i32, view<[%rows]x9xi32>
        }
        %j32 = index.cast %j : index to i32
        %len_j = scalar.addi %j32, %c1_i32 : i32
        %len_row = scf.select %spec, %len_j, %length32 : i32
        view.store %len_row, %out[%rb, 0] : i32, view<[%rows]x9xi32>
        view.store %c1_i32, %out[%rb, 6] : i32, view<[%rows]x9xi32>
      }
      %rn = index.add %rr, %c1 : index
      scf.yield %rn : index
    }
    scf.yield %next : index
  }
  %bounded_total = index.min %total, %rows : index
  %total32 = index.cast %bounded_total : index to i32
  %out_header = buffer.view %rows_meta[%zero] : buffer -> view<4xi32>
  view.store %total32, %out_header[0] : i32, view<4xi32>
  view.store %eos, %out_header[1] : i32, view<4xi32>
  view.store %c0_i32, %out_header[2] : i32, view<4xi32>
  view.store %c0_i32, %out_header[3] : i32, view<4xi32>
  kernel.return
}
""")
    # ---- acceptance
    if MROPE:
        mrope_copy = r"""
    // M-RoPE offset tables follow each packet's records.
    %m_table = index.add %c16, %m_plan : offset
    %m_tv = buffer.view %metadata[%m_table] : buffer -> view<[%count]xi32>
    %delta = view.load %m_tv[%sc] : view<[%count]xi32> -> i32
    %c_tv = buffer.view %committed[%m_table] : buffer -> view<[%count]xi32>
    view.store %delta, %c_tv[%sc] : i32, view<[%count]xi32>"""
    T.append(f"""
// Greedy acceptance, the sole owner of speculative publication (one device thread). Result record per
// span: {{consumed, output_count, token[8]}}. A speculative span emits predictions while each matches the
// next proposal, stopping at EOS or its output credit; the last emitted token stays pending, so consumed
// == output_count. Known spans consume their whole length and emit their selected prediction. Committed
// (every span, length = consumed) feeds the drafter's catch-up; replay (speculative spans only) feeds the
// gated delta-net replay. Both keep the original packed cursors.
kernel.def export("g_spec_accept") @g_spec_accept() {{
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
}} launch(%spans_arg: index, %rows_arg: index, %tokens_arg: index, %metadata: buffer, %ids: buffer, %predictions: buffer, %results: buffer, %committed: buffer, %replay: buffer) {{
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 65536)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %zero = index.constant 0 : offset
  %c16 = index.constant 16 : offset
  %c36 = index.constant 36 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c8 = index.constant 8 : index
  %c9 = index.constant 9 : index
  %c10 = index.constant 10 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %c4_i32 = scalar.constant 4 : i32
  %c8_i32 = scalar.constant 8 : i32
  %c255_i32 = scalar.constant 255 : i32
  %true = scalar.constant true : i1
  %false = scalar.constant false : i1
  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %eos = view.load %header[1] : view<4xi32> -> i32
  %mixed_word = view.load %header[2] : view<4xi32> -> i32
  %mixed = scalar.cmpi ne, %mixed_word, %c0_i32 : i32
  %stride_one = index.constant 1 : index
  %id_stride = scf.select %mixed, %c8, %stride_one : index
  %n_ids = index.mul %tokens, %id_stride : index
  %ids_flat = buffer.view %ids[%zero] : buffer -> view<[%n_ids]xi32>
  %count_b = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count_v = index.cast %count_b : i32 to index
  %count = index.min %count_v, %spans : index
  %m_plan = index.scale %count, %c36 : index, offset -> offset
  %records = buffer.view %metadata[%c16] : buffer -> view<[%spans]x9xi32>
  %crecords = buffer.view %committed[%c16] : buffer -> view<[%spans]x9xi32>
  %rrecords = buffer.view %replay[%c16] : buffer -> view<[%spans]x9xi32>
  %pred = buffer.view %predictions[%zero] : buffer -> view<[%rows]xi32>
  %out = buffer.view %results[%zero] : buffer -> view<[%spans]x10xi32>
  %row_end, %replays = scf.for %s = [%c0 to %count step %c1](%r = %c0 : index, %k = %c0 : index) -> (index, index) {{
    %sc = index.assume %s [range(%s, 0, 2147483646), lt(%s, %count)] : index
    %sb = index.assume %s [range(%s, 0, 2147483646), lt(%s, %spans)] : index
    %length32 = view.load %records[%sb, 0] : view<[%spans]x9xi32> -> i32
    %cursor32 = view.load %records[%sb, 4] : view<[%spans]x9xi32> -> i32
    %flags = view.load %records[%sb, 6] : view<[%spans]x9xi32> -> i32
    %sel_bits = scalar.andi %flags, %c1_i32 : i32
    %selected = scalar.cmpi ne, %sel_bits, %c0_i32 : i32
    %spec_bits = scalar.andi %flags, %c4_i32 : i32
    %spec = scalar.cmpi ne, %spec_bits, %c0_i32 : i32
    %credit_raw = scalar.shrui %flags, %c8_i32 : i32
    %credit32 = scalar.andi %credit_raw, %c255_i32 : i32
    scf.for %w = [%c2 to %c10 step %c1] {{
      %wb = index.assume %w [range(%w, 2, 9)] : index
      view.store %c0_i32, %out[%sb, %wb] : i32, view<[%spans]x10xi32>
    }}
    %cursor_b = scalar.assume %cursor32 [range(%cursor32, 0, 8191)] : i32
    %cursor = index.cast %cursor_b : i32 to index
    %consumed, %emitted, %used = scf.if %spec -> (i32, i32, index) {{
      %length_b = scalar.assume %length32 [range(%length32, 1, 8)] : i32
      %width = index.cast %length_b : i32 to index
      %credit_b = scalar.assume %credit32 [range(%credit32, 0, 255)] : i32
      %credit = index.cast %credit_b : i32 to index
      %last_draft = index.sub %width, %c1 : index
      %n, %live_end = scf.for %j = [%c0 to %width step %c1](%cnt = %c0 : index, %live = %true : i1) -> (index, i1) {{
        %credited = index.cmp ult, %j, %credit : index
        %emit = scalar.andi %live, %credited : i1
        %rj = index.add %r, %j : index
        %row_ok = index.cmp ult, %rj, %rows : index
        %emit_ok = scalar.andi %emit, %row_ok : i1
        %nc, %nl = scf.if %emit_ok -> (index, i1) {{
          %rb = index.assume %rj [range(%rj, 0, 2147483646), lt(%rj, %rows)] : index
          %p = view.load %pred[%rb] : view<[%rows]xi32> -> i32
          %slot = index.add %j, %c2 : index
          %slot_b = index.assume %slot [range(%slot, 2, 9)] : index
          view.store %p, %out[%sb, %slot_b] : i32, view<[%spans]x10xi32>
          %cnt1 = index.add %cnt, %c1 : index
          %has_draft = index.cmp ult, %j, %last_draft : index
          %match = scf.if %has_draft -> (i1) {{
            %rel = index.add %j, %c1 : index
            %cand_i = index.add %cursor, %rel : index
{_ids("%ids_flat", "%cand_i", True, "%cand")}
            %same = scalar.cmpi eq, %p, %cand : i32
            scf.yield %same : i1
          }} else {{
            scf.yield %false : i1
          }}
          %not_eos = scalar.cmpi ne, %p, %eos : i32
          %go = scalar.andi %match, %not_eos : i1
          scf.yield %cnt1, %go : index, i1
        }} else {{
          scf.yield %cnt, %false : index, i1
        }}
        scf.yield %nc, %nl : index, i1
      }}
      %n32 = index.cast %n : index to i32
      scf.yield %n32, %n32, %width : i32, i32, index
    }} else {{
      %has_row = index.cmp ult, %r, %rows : index
      %take = scalar.andi %selected, %has_row : i1
      %e, %u = scf.if %take -> (i32, index) {{
        %rb = index.assume %r [range(%r, 0, 2147483646), lt(%r, %rows)] : index
        %p = view.load %pred[%rb] : view<[%rows]xi32> -> i32
        view.store %p, %out[%sb, 2] : i32, view<[%spans]x10xi32>
        scf.yield %c1_i32, %c1 : i32, index
      }} else {{
        scf.yield %c0_i32, %c0 : i32, index
      }}
      scf.yield %length32, %e, %u : i32, i32, index
    }}
    view.store %consumed, %out[%sb, 0] : i32, view<[%spans]x10xi32>
    view.store %emitted, %out[%sb, 1] : i32, view<[%spans]x10xi32>
    scf.for %w = [%c0 to %c9 step %c1] {{
      %wb = index.assume %w [range(%w, 0, 8)] : index
      %v = view.load %records[%sb, %wb] : view<[%spans]x9xi32> -> i32
      view.store %v, %crecords[%sb, %wb] : i32, view<[%spans]x9xi32>
    }}
    view.store %consumed, %crecords[%sb, 0] : i32, view<[%spans]x9xi32>
    view.store %sel_bits, %crecords[%sb, 6] : i32, view<[%spans]x9xi32>
    %k_now = index.assume %k [range(%k, 0, 2147483646), lt(%k, %spans)] : index
    scf.if %spec {{
      scf.for %w = [%c0 to %c9 step %c1] {{
        %wb = index.assume %w [range(%w, 0, 8)] : index
        %v = view.load %records[%sb, %wb] : view<[%spans]x9xi32> -> i32
        view.store %v, %rrecords[%k_now, %wb] : i32, view<[%spans]x9xi32>
      }}
      view.store %consumed, %rrecords[%k_now, 0] : i32, view<[%spans]x9xi32>
      view.store %c0_i32, %rrecords[%k_now, 6] : i32, view<[%spans]x9xi32>
    }}{mrope_copy}
    %kinc = scf.select %spec, %c1, %c0 : index
    %k_next = index.add %k, %kinc : index
    %r_next = index.add %r, %used : index
    scf.yield %r_next, %k_next : index, index
  }}
  %count_out = index.cast %count : index to i32
  %replay_count = index.cast %replays : index to i32
  %ch = buffer.view %committed[%zero] : buffer -> view<4xi32>
  view.store %count_out, %ch[0] : i32, view<4xi32>
  view.store %eos, %ch[1] : i32, view<4xi32>
  view.store %mixed_word, %ch[2] : i32, view<4xi32>
  view.store %c0_i32, %ch[3] : i32, view<4xi32>
  %rh = buffer.view %replay[%zero] : buffer -> view<4xi32>
  view.store %replay_count, %rh[0] : i32, view<4xi32>
  view.store %eos, %rh[1] : i32, view<4xi32>
  view.store %mixed_word, %rh[2] : i32, view<4xi32>
  view.store %c0_i32, %rh[3] : i32, view<4xi32>
  kernel.return
}}
""")
    # ---- plain rmsnorm: out[t] = rmsnorm(in[t]) * w
    T.append(f"""
kernel.def export("g_spec_rmsnorm") @g_spec_rmsnorm(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %input: buffer, %w: buffer, %output: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [range(%token_id, 0, 2147483646), lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %groups = index.constant {HG} : index
  %zero_f = scalar.constant 0.0 : f32
  %zero32 = vector.constant 0.0 : vector<32xf32>
  %count = scalar.constant {float(H)} : f32
  %active = index.cmp ult, %lane, %groups : index
  %in_view = buffer.view %input[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
  %xv = scf.if %active -> (vector<32xf32>) {{
    %bl = index.assume %lane [range(%lane, 0, {HG - 1})] : index
    %a = vector.load %in_view[%token, %bl, %c0] : view<[%tokens]x{HG}x32xf32> -> vector<32xf32>
    scf.yield %a : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}
  %sq = vector.mulf %xv, %xv : vector<32xf32>
  %local = vector.reduce<addf> %sq, %zero_f : vector<32xf32>, f32
  %sum = kernel.workgroup.reduce<addf> %local : f32
  %mean = scalar.divf %sum, %count : f32
  %eps = scalar.constant {EPS!r} : f32
  %stable = scalar.addf %mean, %eps : f32
  %inv = scalar.rsqrtf %stable : f32
  %invs = vector.splat %inv : vector<32xf32>
  scf.if %active {{
    %bl = index.assume %lane [range(%lane, 0, {HG - 1})] : index
    %w_view = buffer.view %w[%zero] : buffer -> view<{HG}x32xf32>
    %out_view = buffer.view %output[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
    %wv = vector.load %w_view[%bl, %c0] : view<{HG}x32xf32> -> vector<32xf32>
    %n = vector.mulf %xv, %invs : vector<32xf32>
    %o = vector.mulf %n, %wv : vector<32xf32>
    vector.store %o, %out_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x{HG}x32xf32>
  }}
  kernel.return
}}
""")
    if g["GDN"]:
        G = g["GDN"]
        C, HV, NG = G["C"], G["HV"], g["NG"]
        W = C + 2 * HV
        for name, save in (("g_gdn_capture", True), ("g_gdn_restore", False)):
            body = []
            for src, off, n in (("qkv", 0, C), ("alpha", C, HV), ("beta", C + HV, HV)):
                lv = f"view<[%tokens]x{n}xf32>"
                ld = f"%{src}_v[%token, %c]" if save else "%cap_v[%cc]"
                st = "%cap_v[%cc]" if save else f"%{src}_v[%token, %c]"
                lt, stt = (lv, "view<[%cap_rows]xf32>") if save else ("view<[%cap_rows]xf32>", lv)
                body.append(f"""  %{src}_v = buffer.view %{src}[%zero] : buffer -> {lv}
  %{src}_n = index.constant {n} : index
  %{src}_off = index.constant {off} : index
  scf.for %{src}_c = [%lane to %{src}_n step %c256] {{
    %c = index.assume %{src}_c [range(%{src}_c, 0, {n - 1})] : index
    %cc0 = index.add %row_base, %c : index
    %cc1 = index.add %cc0, %{src}_off : index
    %cc = index.assume %cc1 [range(%cc1, 0, 2147483646), lt(%cc1, %cap_rows)] : index
    %v = view.load {ld} : {lt} -> f32
    view.store %v, {st} : f32, {stt}
  }}""")
            T.append(f"""
// Gated delta-net replay inputs: [layer][token][qkv C | alpha HV | beta HV] f32.
kernel.def export("{name}") @{name}(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %layer_arg: index, %qkv: buffer, %alpha: buffer, %beta: buffer, %capture: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {NG - 1})] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [range(%token_id, 0, 2147483646), lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c256 = index.constant 256 : index
  %width = index.constant {W} : index
  %layers = index.constant {NG} : index
  %plane = index.mul %tokens, %width : index
  %cap_rows = index.mul %plane, %layers : index
  %layer_base = index.mul %layer, %plane : index
  %token_base = index.mul %token, %width : index
  %row_base = index.add %layer_base, %token_base : index
  %cap_v = buffer.view %capture[%zero] : buffer -> view<[%cap_rows]xf32>
{chr(10).join(body)}
  kernel.return
}}
""")
    for W in sorted({w for r in g.get("replay", []) for _, w in r["buffers"]}):
        T.append(replay_kernels(W))
    if DRAFT and DRAFT.get("kind") == "mtp":
        T.append(mtp_kernels(g))
    return "\n".join(T)


def replay_kernels(W):
    """Capture / restore of W-wide f32 token rows ([slot][token][W]) for the generic recurrent replay (verify_program)."""
    out = []
    for name, save in (("save", True), ("load", False)):
        ld, st = ("%src_v[%token, %c]", "%cap_v[%token, %c]") if save else ("%cap_v[%token, %c]", "%src_v[%token, %c]")
        out.append(f"""
kernel.def export("g_spec_{name}_w{W}") @g_spec_{name}_w{W}(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %slot_arg: index, %src: buffer, %capture: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %slot = index.assume %slot_arg [range(%slot_arg, 0, 1023)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [range(%token_id, 0, 2147483646), lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %c256 = index.constant 256 : index
  %row_bytes = index.constant {W * 4} : offset
  %plane = index.scale %tokens, %row_bytes : index, offset -> offset
  %at = index.scale %slot, %plane : index, offset -> offset
  %zero = index.constant 0 : offset
  %src_v = buffer.view %src[%zero] : buffer -> view<[%tokens]x{W}xf32>
  %cap_v = buffer.view %capture[%at] : buffer -> view<[%tokens]x{W}xf32>
  %width = index.constant {W} : index
  scf.for %c0v = [%lane to %width step %c256] {{
    %c = index.assume %c0v [range(%c0v, 0, {W - 1})] : index
    %v = view.load {ld} : view<[%tokens]x{W}xf32> -> f32
    view.store %v, {st} : f32, view<[%tokens]x{W}xf32>
  }}
  kernel.return
}}
""")
    return "".join(out)


def mtp_kernels(g):
    H, HG = g["H"], g["HG"]
    return f"""
// MTP input pairing for catch-up: prev[t] = target hidden of the input before t (the row's carry for a
// span's first input). Inactive tokens get zeros.
kernel.def export("g_mtp_pair") @g_mtp_pair(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %owners_arg: index, %metadata: buffer, %info: buffer, %hidden: buffer, %carry: buffer, %prev: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [range(%token_id, 0, 2147483646), lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c16 = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1i = index.constant 1 : index
  %groups = index.constant {HG} : index
  %c0_i32 = scalar.constant 0 : i32
  %zero32 = vector.constant 0.0 : vector<32xf32>
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %row32 = view.load %info_view[%token, 0] : view<[%tokens]x4xi32> -> i32
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %span32 = view.load %info_view[%token, 3] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %active = index.cmp ult, %lane, %groups : index
  %prev_view = buffer.view %prev[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
  %h_view = buffer.view %hidden[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
  scf.if %active {{
    %bl = index.assume %lane [range(%lane, 0, {HG - 1})] : index
    %value = scf.if %live -> (vector<32xf32>) {{
      %records = buffer.view %metadata[%c16] : buffer -> view<[%spans]x9xi32>
      %span_b = scalar.assume %span32 [range(%span32, 0, 8191)] : i32
      %span_v = index.cast %span_b : i32 to index
      %span = index.assume %span_v [range(%span_v, 0, 2147483646), lt(%span_v, %spans)] : index
      %cursor32 = view.load %records[%span, 4] : view<[%spans]x9xi32> -> i32
      %token32 = index.cast %token : index to i32
      %first = scalar.cmpi eq, %token32, %cursor32 : i32
      %v = scf.if %first -> (vector<32xf32>) {{
        %row_b = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
        %row_v = index.cast %row_b : i32 to index
        %row = index.assume %row_v [range(%row_v, 0, 2147483646)] : index
        %carry_stride = index.constant {4 * H} : offset
        %carry_at = index.scale %row, %carry_stride : index, offset -> offset
        %carry_row_view = buffer.view %carry[%carry_at] : buffer -> view<{HG}x32xf32>
        %c = vector.load %carry_row_view[%bl, %c0] : view<{HG}x32xf32> -> vector<32xf32>
        scf.yield %c : vector<32xf32>
      }} else {{
        %token_pos = index.max %token, %c1i : index
        %before = index.sub %token_pos, %c1i : index
        %bb = index.assume %before [range(%before, 0, 2147483646), lt(%before, %tokens)] : index
        %p = vector.load %h_view[%bb, %bl, %c0] : view<[%tokens]x{HG}x32xf32> -> vector<32xf32>
        scf.yield %p : vector<32xf32>
      }}
      scf.yield %v : vector<32xf32>
    }} else {{
      scf.yield %zero32 : vector<32xf32>
    }}
    vector.store %value, %prev_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x{HG}x32xf32>
  }}
  kernel.return
}}

// carry[row] = target hidden of the span's last consumed input (runs after g_mtp_pair read it).
kernel.def export("g_mtp_save") @g_mtp_save(%spans: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%spans, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %owners_arg: index, %metadata: buffer, %hidden: buffer, %carry: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %span_id = kernel.workgroup.id<x> : index
  %span = index.assume %span_id [range(%span_id, 0, 2147483646), lt(%span_id, %spans)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c16 = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %groups = index.constant {HG} : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %count_b = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count = index.cast %count_b : i32 to index
  %in_count = index.cmp ult, %span, %count : index
  %active = index.cmp ult, %lane, %groups : index
  %go = scalar.andi %in_count, %active : i1
  scf.if %go {{
    %records = buffer.view %metadata[%c16] : buffer -> view<[%spans]x9xi32>
    %length32 = view.load %records[%span, 0] : view<[%spans]x9xi32> -> i32
    %row32 = view.load %records[%span, 3] : view<[%spans]x9xi32> -> i32
    %cursor32 = view.load %records[%span, 4] : view<[%spans]x9xi32> -> i32
    %nonempty = scalar.cmpi sgt, %length32, %c0_i32 : i32
    scf.if %nonempty {{
      %end32 = scalar.addi %cursor32, %length32 : i32
      %last32 = scalar.subi %end32, %c1_i32 : i32
      %last_b = scalar.assume %last32 [range(%last32, 0, 8191)] : i32
      %last_v = index.cast %last_b : i32 to index
      %last = index.assume %last_v [range(%last_v, 0, 2147483646), lt(%last_v, %tokens)] : index
      %row_b = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
      %row_v = index.cast %row_b : i32 to index
      %row = index.assume %row_v [range(%row_v, 0, 2147483646)] : index
      %bl = index.assume %lane [range(%lane, 0, {HG - 1})] : index
      %h_view = buffer.view %hidden[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
          %v = vector.load %h_view[%last, %bl, %c0] : view<[%tokens]x{HG}x32xf32> -> vector<32xf32>
      %carry_stride = index.constant {4 * H} : offset
      %carry_at = index.scale %row, %carry_stride : index, offset -> offset
      %carry_row_view = buffer.view %carry[%carry_at] : buffer -> view<{HG}x32xf32>
      vector.store %v, %carry_row_view[%bl, %c0] : vector<32xf32>, view<{HG}x32xf32>
    }}
  }}
  kernel.return
}}

// out[t] = [a[t] | b[t]] (2H wide).
kernel.def export("g_mtp_concat") @g_mtp_concat(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %a: buffer, %b: buffer, %output: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
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
    %a_view = buffer.view %a[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
    %b_view = buffer.view %b[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
    %o_view = buffer.view %output[%zero] : buffer -> view<[%tokens]x2x{HG}x32xf32>
    %va = vector.load %a_view[%token, %bl, %c0] : view<[%tokens]x{HG}x32xf32> -> vector<32xf32>
    %vb = vector.load %b_view[%token, %bl, %c0] : view<[%tokens]x{HG}x32xf32> -> vector<32xf32>
    vector.store %va, %o_view[%token, 0, %bl, %c0] : vector<32xf32>, view<[%tokens]x2x{HG}x32xf32>
    vector.store %vb, %o_view[%token, 1, %bl, %c0] : vector<32xf32>, view<[%tokens]x2x{HG}x32xf32>
  }}
  kernel.return
}}

// Proposal batch: one draft record per PROPOSE span (length 1 at the anchor's position, greedy), its
// anchor token, and its verifier input slot.
kernel.def export("g_mtp_prepare") @g_mtp_prepare() {{
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
}} launch(%spans_arg: index, %drafts_arg: index, %tokens_arg: index, %metadata: buffer, %ids: buffer, %draft: buffer, %draft_ids: buffer, %slots: buffer) {{
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %drafts = index.assume %drafts_arg [range(%drafts_arg, 1, 8192)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %zero = index.constant 0 : offset
  %c16 = index.constant 16 : offset
  %c36 = index.constant 36 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c8 = index.constant 8 : index
  %c9 = index.constant 9 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %c2_i32 = scalar.constant 2 : i32
  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %eos = view.load %header[1] : view<4xi32> -> i32
  %mixed_word = view.load %header[2] : view<4xi32> -> i32
  %mixed = scalar.cmpi ne, %mixed_word, %c0_i32 : i32
  %stride_one = index.constant 1 : index
  %id_stride = scf.select %mixed, %c8, %stride_one : index
  %n_ids = index.mul %tokens, %id_stride : index
  %ids_flat = buffer.view %ids[%zero] : buffer -> view<[%n_ids]xi32>
  %count_b = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count_v = index.cast %count_b : i32 to index
  %count = index.min %count_v, %spans : index
  %records = buffer.view %metadata[%c16] : buffer -> view<[%spans]x9xi32>
  %out = buffer.view %draft[%c16] : buffer -> view<[%drafts]x9xi32>
  %dids = buffer.view %draft_ids[%zero] : buffer -> view<[%drafts]xi32>
  %slot_v = buffer.view %slots[%zero] : buffer -> view<[%drafts]xi32>
  %m_plan = index.scale %count, %c36 : index, offset -> offset
  %m_table = index.add %c16, %m_plan : offset
  %d_plan = index.scale %drafts, %c36 : index, offset -> offset
  %d_table = index.add %c16, %d_plan : offset
  %total = scf.for %s = [%c0 to %count step %c1](%k = %c0 : index) -> (index) {{
    %sb = index.assume %s [range(%s, 0, 2147483646), lt(%s, %spans)] : index
    %flags = view.load %records[%sb, 6] : view<[%spans]x9xi32> -> i32
    %prop_bits = scalar.andi %flags, %c2_i32 : i32
    %proposes = scalar.cmpi ne, %prop_bits, %c0_i32 : i32
    %fits = index.cmp ult, %k, %drafts : index
    %take = scalar.andi %proposes, %fits : i1
    scf.if %take {{
      %kb = index.assume %k [range(%k, 0, 2147483646), lt(%k, %drafts)] : index
      scf.for %w = [%c0 to %c9 step %c1] {{
        %wb = index.assume %w [range(%w, 0, 8)] : index
        %v = view.load %records[%sb, %wb] : view<[%spans]x9xi32> -> i32
        view.store %v, %out[%kb, %wb] : i32, view<[%drafts]x9xi32>
      }}
      %k32 = index.cast %kb : index to i32
      view.store %c1_i32, %out[%kb, 0] : i32, view<[%drafts]x9xi32>
      view.store %c0_i32, %out[%kb, 2] : i32, view<[%drafts]x9xi32>
      view.store %k32, %out[%kb, 4] : i32, view<[%drafts]x9xi32>
      view.store %c1_i32, %out[%kb, 6] : i32, view<[%drafts]x9xi32>
      %cursor32 = view.load %records[%sb, 4] : view<[%spans]x9xi32> -> i32
      %cursor_b = scalar.assume %cursor32 [range(%cursor32, 0, 8191)] : i32
      %cursor = index.cast %cursor_b : i32 to index
{_ids("%ids_flat", "%cursor", True, "%anchor")}
      view.store %anchor, %dids[%kb] : i32, view<[%drafts]xi32>
      view.store %cursor32, %slot_v[%kb] : i32, view<[%drafts]xi32>
      {"" if not g["MROPE"] else '''%sc = index.assume %s [range(%s, 0, 2147483646), lt(%s, %count)] : index
      %m_tv = buffer.view %metadata[%m_table] : buffer -> view<[%count]xi32>
      %delta = view.load %m_tv[%sc] : view<[%count]xi32> -> i32
      %d_tv = buffer.view %draft[%d_table] : buffer -> view<[%drafts]xi32>
      view.store %delta, %d_tv[%kb] : i32, view<[%drafts]xi32>'''}
    }}
    %inc = scf.select %take, %c1, %c0 : index
    %kn = index.add %k, %inc : index
    scf.yield %kn : index
  }}
  %total32 = index.cast %total : index to i32
  %dh = buffer.view %draft[%zero] : buffer -> view<4xi32>
  view.store %total32, %dh[0] : i32, view<4xi32>
  view.store %eos, %dh[1] : i32, view<4xi32>
  view.store %c0_i32, %dh[2] : i32, view<4xi32>
  view.store %c0_i32, %dh[3] : i32, view<4xi32>
  kernel.return
}}

// hidden[k] = carry[row of draft k] (zeros past the live count).
kernel.def export("g_mtp_gather") @g_mtp_gather(%drafts: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%drafts, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%drafts_arg: index, %owners_arg: index, %draft: buffer, %carry: buffer, %hidden: buffer) {{
  %drafts = index.assume %drafts_arg [range(%drafts_arg, 1, 8192)] : index
  %k_id = kernel.workgroup.id<x> : index
  %k = index.assume %k_id [range(%k_id, 0, 2147483646), lt(%k_id, %drafts)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c16 = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %groups = index.constant {HG} : index
  %zero32 = vector.constant 0.0 : vector<32xf32>
  %header = buffer.view %draft[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %count_b = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count = index.cast %count_b : i32 to index
  %live = index.cmp ult, %k, %count : index
  %active = index.cmp ult, %lane, %groups : index
  scf.if %active {{
    %bl = index.assume %lane [range(%lane, 0, {HG - 1})] : index
    %h_view = buffer.view %hidden[%zero] : buffer -> view<[%drafts]x{HG}x32xf32>
    %v = scf.if %live -> (vector<32xf32>) {{
      %records = buffer.view %draft[%c16] : buffer -> view<[%drafts]x9xi32>
      %row32 = view.load %records[%k, 3] : view<[%drafts]x9xi32> -> i32
      %row_b = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
      %row_v = index.cast %row_b : i32 to index
      %row = index.assume %row_v [range(%row_v, 0, 2147483646)] : index
          %carry_stride = index.constant {4 * H} : offset
          %carry_at = index.scale %row, %carry_stride : index, offset -> offset
          %carry_row_view = buffer.view %carry[%carry_at] : buffer -> view<{HG}x32xf32>
          %c = vector.load %carry_row_view[%bl, %c0] : view<{HG}x32xf32> -> vector<32xf32>
      scf.yield %c : vector<32xf32>
    }} else {{
      scf.yield %zero32 : vector<32xf32>
    }}
    vector.store %v, %h_view[%k, %bl, %c0] : vector<32xf32>, view<[%drafts]x{HG}x32xf32>
  }}
  kernel.return
}}

// After proposal round `round`: write each draft into its verifier slot (cursor + round + 1), make it
// the next round's input, and advance the draft position.
kernel.def export("g_mtp_advance") @g_mtp_advance() {{
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
}} launch(%round_arg: index, %drafts_arg: index, %tokens_arg: index, %metadata: buffer, %draft: buffer, %slots: buffer, %proposals: buffer, %draft_ids: buffer, %ids: buffer) {{
  %round = index.assume %round_arg [range(%round_arg, 0, 7)] : index
  %drafts = index.assume %drafts_arg [range(%drafts_arg, 1, 8192)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %zero = index.constant 0 : offset
  %c16 = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1i = index.constant 1 : index
  %c8 = index.constant 8 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %mheader = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %mixed_word = view.load %mheader[2] : view<4xi32> -> i32
  %mixed = scalar.cmpi ne, %mixed_word, %c0_i32 : i32
  %id_stride = scf.select %mixed, %c8, %c1i : index
  %n_ids = index.mul %tokens, %id_stride : index
  %ids_flat = buffer.view %ids[%zero] : buffer -> view<[%n_ids]xi32>
  %header = buffer.view %draft[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %count_b = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count_v = index.cast %count_b : i32 to index
  %count = index.min %count_v, %drafts : index
  %records = buffer.view %draft[%c16] : buffer -> view<[%drafts]x9xi32>
  %slot_v = buffer.view %slots[%zero] : buffer -> view<[%drafts]xi32>
  %prop = buffer.view %proposals[%zero] : buffer -> view<[%drafts]xi32>
  %dids = buffer.view %draft_ids[%zero] : buffer -> view<[%drafts]xi32>
  %step = index.add %round, %c1i : index
  scf.for %k = [%c0 to %count step %c1i] {{
    %kb = index.assume %k [range(%k, 0, 2147483646), lt(%k, %drafts)] : index
    %token = view.load %prop[%kb] : view<[%drafts]xi32> -> i32
    %slot32 = view.load %slot_v[%kb] : view<[%drafts]xi32> -> i32
    %slot_b = scalar.assume %slot32 [range(%slot32, 0, 8191)] : i32
    %slot0 = index.cast %slot_b : i32 to index
    %slot = index.add %slot0, %step : index
    %e0 = index.mul %slot, %id_stride : index
    %e = index.assume %e0 [range(%e0, 0, 2147483646), lt(%e0, %n_ids)] : index
    view.store %token, %ids_flat[%e] : i32, view<[%n_ids]xi32>
    view.store %token, %dids[%kb] : i32, view<[%drafts]xi32>
    %position = view.load %records[%kb, 1] : view<[%drafts]x9xi32> -> i32
    %next_position = scalar.addi %position, %c1_i32 : i32
    view.store %next_position, %records[%kb, 1] : i32, view<[%drafts]x9xi32>
  }}
  kernel.return
}}
"""


# ---------------------------------------------------------------------------------------------------
# Command programs (text transforms of the trunk program gen_llm.model() produced)

ADVANCE_SIG = "command.program.def public @llm_advance() launch(%weights: buffer, %residual: buffer, %metadata: buffer, %origins: buffer, %arena: buffer, %tokens: buffer, %outputs: buffer) {"
H_ALLOCA = "  %h = buffer.alloca<global> align(256) %x_size : buffer\n"


def _split_head(program):
    i = program.index("\n  %head = command.parameter %weights")
    return program[:i + 1], program[i + 1:]


def small_gemm(program):
    """Verification epochs carry width x spans tokens (e.g. 4 streams x 4 = 16). The 4-token matvec streams the
    weights once per 4 tokens, so their linears take the WMMA GEMM from 5 tokens on, over the token count padded
    to 64 rows (one weight pass per step); GEMM rows are independent, so the padded rows are never read.
    Attention keeps its decode path (%gemm still switches it at 64). Scratch is sized for the padded count."""
    import re as _re
    t_def = "  %t = config.get @llm.tokens : index\n"
    assert t_def in program
    program = program.replace(t_def, t_def + "  %c5_mm = index.constant 5 : index\n  %c63_tp = index.constant 63 : index\n"
                              "  %c64_tp = index.constant 64 : index\n  %tp_a = index.add %t, %c63_tp : index\n"
                              "  %tp_b = index.div %tp_a, %c64_tp : index\n  %tp = index.mul %tp_b, %c64_tp : index\n"
                              "  %mm = index.cmp uge, %t, %c5_mm : index\n")
    program = program.replace("index.scale %t, ", "index.scale %tp, ")
    program = _re.sub(r"scf\.if %gemm \{(\s*)kernel\.launch @g_row_scale_f16", r"scf.if %mm {\1kernel.launch @g_row_scale_f16", program)
    program = _re.sub(r"scf\.if %gemm \{(\s*)scf\.if %big \{", r"scf.if %mm {\1scf.if %big {", program)
    program = _re.sub(r"kernel\.launch @(g_gemm_\w+)\[%t, ([^\]]+)\]\(%t, ", r"kernel.launch @\1[%tp, \2](%tp, ", program)
    return program


def first_binding():
    """Name of the trunk's first launch buffer (the `residual` allocation): the final normalized hidden rows
    for MTP's catch-up, the captured target features for DFlash's."""
    return "%features" if DRAFT and DRAFT.get("kind") == "dflash" else "%h"


def _with_taps(program, g):
    """DFlash: capture the residual stream after each target layer the drafter reads."""
    import re as _re
    taps = g["dflash_taps"]
    for k, i in enumerate(taps):
        launch = (f"  %tap_slot_{k} = index.constant {k} : index\n"
                  f"  kernel.launch @g_dfl_tap[%t](%t, %tap_slot_{k}, %x, %features) : [index](index, index, buffer, buffer)\n")
        m = _re.search(rf"\n  // ---- layer {i + 1}[ \n]", program)
        at = m.start() + 1 if m else program.index("\n  %head = command.parameter %weights") + 1
        assert i + 1 < g["L"] or not m
        program = program[:at] + launch + program[at:]
    return program


def advance_program(program, g=None):
    """Ordinary epochs. MTP: the final normalized hidden rows stay in the `residual` allocation. DFlash: the
    trunk keeps its own hidden rows and captures the drafter's feature taps into that allocation."""
    assert ADVANCE_SIG in program and H_ALLOCA in program
    if first_binding() == "%features":
        return _with_taps(program.replace(ADVANCE_SIG, ADVANCE_SIG.replace("%residual: buffer", "%features: buffer")), g)
    return program.replace(ADVANCE_SIG, ADVANCE_SIG.replace("%residual: buffer", "%h: buffer")).replace(H_ALLOCA, "", 1)


def verify_program(program, g):
    body, _ = _split_head(advance_program(program, g))
    body = body[body.index("command.program.def public @llm_advance()"):]   # config decls are emitted once (advance)
    fb = first_binding()
    body = body.replace(f"command.program.def public @llm_advance() launch(%weights: buffer, {fb}: buffer, %metadata: buffer, %origins: buffer, %arena: buffer, %tokens: buffer, %outputs: buffer) {{",
                        f"command.program.def public @llm_verify() launch(%weights: buffer, {fb}: buffer, %metadata: buffer, %origins: buffer, %arena: buffer, %tokens: buffer, %committed: buffer, %results: buffer) {{", 1)
    assert "@llm_verify()" in body
    GDN = g["GDN"]
    if GDN:
        W = GDN["C"] + 2 * GDN["HV"]
        cap = (f"  %cap_token_bytes = index.constant {W * 4 * g['NG']} : offset\n"
               "  %cap_size = index.scale %t, %cap_token_bytes : index, offset -> offset\n"
               "  %capture = buffer.alloca<global> align(256) %cap_size : buffer\n")
        anchor = "  kernel.launch @g_token_info[%t]"
        body = body.replace(anchor, cap + anchor, 1)
        out = []
        for line in body.split("\n"):
            if line.startswith("  kernel.launch @g_gdn_conv[%t](%t, %spans, %layer_"):
                layer = line.split("%layer_")[1].split(",")[0]
                out.append(f"  kernel.launch @g_gdn_capture[%t](%t, %layer_{layer}, %gq, %galpha, %gbeta, %capture) : [index](index, index, buffer, buffer, buffer, buffer)")
            out.append(line)
        body = "\n".join(out)
    # Generic recurrent replay (kits register it in gen_llm.MODEL_PARTS["replay"]; e.g. LFM2 short-conv,
    # reference/lfm2moe-example/gen_sconv.py): {"name", "match": prefix of the trunk launch that reads the
    # span's inputs, "buffers": [(ssa, f32 width per token)], "replay": f(layer) -> text}. The buffers are
    # captured before every matching launch; after acceptance they are restored and replay(layer) re-runs the
    # state update over the accepted inputs (the %replay packet). The kit's state commit must skip speculative
    # spans (flag bit 4) itself, as gen_gdn.py's does.
    replays = []
    for spec in g.get("replay", []):
        hits = [line for line in body.split("\n") if line.startswith(spec["match"])]
        if not hits:
            continue
        layers = [line.split("%layer_")[1].split(",")[0] for line in hits]
        allocs = "".join(f"  %rc_{spec['name']}_{bi}_bytes = index.constant {w * 4 * len(hits)} : offset\n"
                         f"  %rc_{spec['name']}_{bi}_size = index.scale %t, %rc_{spec['name']}_{bi}_bytes : index, offset -> offset\n"
                         f"  %rc_{spec['name']}_{bi} = buffer.alloca<global> align(256) %rc_{spec['name']}_{bi}_size : buffer\n"
                         for bi, (_, w) in enumerate(spec["buffers"]))
        anchor = "  kernel.launch @g_token_info[%t]"
        body = body.replace(anchor, allocs + anchor, 1)
        out, k = [], 0
        for line in body.split("\n"):
            if line.startswith(spec["match"]):
                for bi, (buf, w) in enumerate(spec["buffers"]):
                    out.append(f"  %rs_{spec['name']}_{k}_{bi} = index.constant {k} : index\n"
                               f"  kernel.launch @g_spec_save_w{w}[%t](%t, %rs_{spec['name']}_{k}_{bi}, {buf}, %rc_{spec['name']}_{bi}) : [index](index, index, buffer, buffer)")
                k += 1
            out.append(line)
        body = "\n".join(out)
        replays.append((spec, layers))
    head_name, hsize = g["head_name"], g["head_size"]
    tail = f"""  %head = command.parameter %weights, "{head_name}" : view<{hsize}xi8>
  %width = config.get @llm.width : index
  %span_rows = index.mul %spans, %width : index
  %rows = index.min %span_rows, %t : index
  %c36_o = index.constant 36 : offset
  %c16_o = index.constant 16 : offset
  %rows_records = index.scale %rows, %c36_o : index, offset -> offset
  %rows_meta_size = index.add %rows_records, %c16_o : offset
  %rows_meta = buffer.alloca<global> align(256) %rows_meta_size : buffer
  %c4_o = index.constant 4 : offset
  %pred_size = index.scale %rows, %c4_o : index, offset -> offset
  %predictions = buffer.alloca<global> align(256) %pred_size : buffer
  %vpartial_size = index.scale %rows, %partial_bytes : index, offset -> offset
  %vpartials = buffer.alloca<global> align(256) %vpartial_size : buffer
  %vlogit_size = index.scale %rows, %logit_bytes : index, offset -> offset
  %vlogits = buffer.alloca<global> align(256) %vlogit_size : buffer
  %replay_records = index.scale %spans, %c36_o : index, offset -> offset
  %replay_tables = index.scale %spans, %c4_o : index, offset -> offset
  %replay_body = index.add %replay_records, %replay_tables : offset
  %replay_size = index.add %replay_body, %c16_o : offset
  %replay = buffer.alloca<global> align(256) %replay_size : buffer
  kernel.launch @g_spec_rows(%spans, %rows, %metadata, %rows_meta) : (index, index, buffer, buffer)
  %one_row = index.cmp eq, %rows, %c1 : index
  scf.if %one_row {{
    kernel.launch @g_head_mv[%rows](%rows, %t, %rows_meta, %h, %head, %vpartials, %vlogits) : [index](index, index, buffer, buffer, view<{hsize}xi8>, buffer, buffer)
  }} else {{
    kernel.launch @g_head_mv_t4[%rows](%rows, %t, %rows_meta, %h, %head, %vpartials, %vlogits) : [index](index, index, buffer, buffer, view<{hsize}xi8>, buffer, buffer)
  }}
  kernel.launch @g_head_select[%rows](%rows, %rows_meta, %vpartials, %vlogits, %predictions) : [index](index, buffer, buffer, buffer, buffer)
  kernel.launch @g_spec_accept(%spans, %rows, %t, %metadata, %tokens, %predictions, %results, %committed, %replay) : (index, index, index, buffer, buffer, buffer, buffer, buffer, buffer)
"""
    if GDN:
        tail += "  // ---- gated delta-net replay: conv + recurrence over the accepted speculative inputs, published\n"
        tail += "  kernel.launch @g_token_info[%t](%t, %spans, %replay, %info) : [index](index, index, buffer, buffer)\n"
        C, HV, DV, K = GDN["C"], GDN["HV"], GDN["DV"], GDN["K"]
        for i, ordinal in g["gdn_layers"]:
            tail += f"""  kernel.launch @g_gdn_restore[%t](%t, %layer_{i}, %gq, %galpha, %gbeta, %capture) : [index](index, index, buffer, buffer, buffer, buffer)
  kernel.launch @g_gdn_conv[%t](%t, %spans, %layer_{i}, %replay, %info, %origins, %arena, %gq, %conv_{i}, %gs) : [index](index, index, index, buffer, buffer, buffer, buffer, buffer, view<{C * K}xf32>, buffer)
  kernel.launch @g_gdn_conv_commit[%spans](%t, %spans, %layer_{i}, %replay, %origins, %arena, %gq) : [index](index, index, index, buffer, buffer, buffer, buffer)
  kernel.launch @g_gdn_qk_norm[%t](%t, %gs, %ginv) : [index](index, buffer, buffer)
  kernel.launch @g_gdn_recurrence[%spans](%t, %spans, %layer_{i}, %replay, %origins, %arena, %gs, %ginv, %galpha, %gbeta, %ssm_a_{i}, %dt_bias_{i}, %graw) : [index](index, index, index, buffer, buffer, buffer, buffer, buffer, buffer, buffer, view<{HV}xf32>, view<{HV}xf32>, buffer)
"""
    for spec, layers in replays:
        tail += f"  // ---- {spec['name']} replay over the accepted speculative inputs, published\n"
        for k, layer in enumerate(layers):
            for bi, (buf, w) in enumerate(spec["buffers"]):
                tail += (f"  %rl_{spec['name']}_{k}_{bi} = index.constant {k} : index\n"
                         f"  kernel.launch @g_spec_load_w{w}[%t](%t, %rl_{spec['name']}_{k}_{bi}, {buf}, %rc_{spec['name']}_{bi}) : [index](index, index, buffer, buffer)\n")
            tail += spec["replay"](layer)
    tail += "  command.return\n}\n"
    return "config.decl @llm.width : %value: index where [range(%value, 2, 8)]\n\n" + body + tail


def _prologue_allocs(prologue):
    """The trunk's scratch allocations (everything after the @llm_advance signature line)."""
    return prologue.split(ADVANCE_SIG, 1)[1]


def _token_setup(meta, tokens_buf, embed_norm, x, h, g):
    """Token info, the attention fallback list (prefill shapes), and the embedding (h = rmsnorm(emb) * w)."""
    vis = f", {meta}, %info" if g["VISION"] else ""
    vty = ", buffer, buffer" if g["VISION"] else ""
    return f"""  kernel.launch @g_token_info[%t](%t, %spans, {meta}, %info) : [index](index, index, buffer, buffer)
  %list_entry = index.constant 4 : offset
  %list_size = index.scale %t, %list_entry : index, offset -> offset
  %count_size = index.constant 256 : offset
  %fallback_list = buffer.alloca<global> align(256) %list_size : buffer
  %fallback_count = buffer.alloca<global> align(256) %count_size : buffer
  scf.if %gemm {{
    kernel.launch @g_attention_fallback_list[%t](%t, %info, %fallback_list, %fallback_count) : [index](index, buffer, buffer, buffer)
  }}
  %embedding = command.parameter %w_embed, "token_embd.weight" : view<{g['esize']}xi8>
  kernel.launch @g_embed[%t](%t, {tokens_buf}{vis}, %embedding, {embed_norm}, {x}, {h}) : [index](index, buffer{vty}, view<{g['esize']}xi8>, view<{g['H']}xf32>, buffer, buffer)
"""


def _mtp_input(g, prev_norm_src):
    """x = eh_proj([enorm(emb) | hnorm(prev)]), h = rmsnorm(x) * attn_norm of the NextN block."""
    H, M = g["H"], g["mtp_layer"]
    lin = g["lin"]
    eh = f"blk.{M}.nextn.eh_proj.weight"
    return f"""  %enorm = command.parameter %weights, "blk.{M}.nextn.enorm.weight" : view<{H}xf32>
  %hnorm = command.parameter %weights, "blk.{M}.nextn.hnorm.weight" : view<{H}xf32>
  %w_eh = command.parameter %weights, "{eh}" : view<{g['size'](eh)}xi8>
  %mtp_norm = command.parameter %weights, "blk.{M}.attn_norm.weight" : view<{H}xf32>
  %n2H = index.constant {2 * H} : index
  %cat_bytes = index.constant {8 * H} : offset
  %cat_size = index.scale %t, %cat_bytes : index, offset -> offset
  %cat = buffer.alloca<global> align(256) %cat_size : buffer
  %cat16_bytes = index.constant {4 * H} : offset
  %cat16_size = index.scale %t, %cat16_bytes : index, offset -> offset
  %cat16 = buffer.alloca<global> align(256) %cat16_size : buffer
  %s_cat = buffer.alloca<global> align(256) %scale_size : buffer
  %hn = buffer.alloca<global> align(256) %x_size : buffer
  kernel.launch @g_spec_rmsnorm[%t](%t, {prev_norm_src}, %hnorm, %hn) : [index](index, buffer, view<{H}xf32>, buffer)
  kernel.launch @g_mtp_concat[%t](%t, %h, %hn, %cat) : [index](index, buffer, buffer, buffer)
  scf.if %gemm {{
    kernel.launch @g_row_scale_f16[%t](%t, %n2H, %cat, %s_cat, %cat16) : [index](index, index, buffer, buffer, buffer)
  }}
  {lin(g['fmt'](eh), "%cat", "%s_cat", "%w_eh", g['size'](eh), "%x", "%n2H", "%nH", 2 * H)}
  kernel.launch @g_spec_rmsnorm[%t](%t, %x, %mtp_norm, %h) : [index](index, buffer, view<{H}xf32>, buffer)
  scf.if %gemm {{
    kernel.launch @g_row_scale_f16[%t](%t, %nH, %h, %s_h, %h16) : [index](index, index, buffer, buffer, buffer)
  }}
"""


def warm_root_count(g):
    """MTP warm's own-tensor roots: enorm, hnorm, eh_proj, attn_norm, attn_k, attn_v (+ q/k norms)."""
    return 6 + (2 if g["QK_NORM"] else 0)


def mtp_programs(g):
    """@llm_mtp_warm (catch-up K/V + carry) and @llm_mtp_propose (draft_depth rounds)."""
    assert g["prologue"] and g["layer_text"], ("MTP drafting reuses the trunk's layer text: this kit's model() does not record "
                                               "MODEL_PARTS (prologue, layer_text, attn_ordinal); add the update there, or build with --draft dflash/ngram/none")
    P = _prologue_allocs(g["prologue"])
    H, M, lin, fmt, size = g["H"], g["mtp_layer"], g["lin"], g["fmt"], g["size"]
    D, KD = g["D"], g["KD"]
    qn = f"%q_norm_m" if g["QK_NORM"] else "%no_norm"
    nty = f"view<{D}xf32>" if g["QK_NORM"] else "buffer"
    mr = ", %metadata, %tokens" if g["MROPE"] else ""
    mrt = ", buffer, buffer" if g["MROPE"] else ""
    wk, wv = f"blk.{M}.attn_k.weight", f"blk.{M}.attn_v.weight"
    norms = (f'  %q_norm_m = command.parameter %weights, "blk.{M}.attn_q_norm.weight" : view<{D}xf32>\n'
             f'  %k_norm_m = command.parameter %weights, "blk.{M}.attn_k_norm.weight" : view<{D}xf32>\n') if g["QK_NORM"] else ""
    kn = "%k_norm_m" if g["QK_NORM"] else "%no_norm"
    warm = (f"""
// MTP catch-up: K/V of the NextN block at every consumed position, from (embedding of token P, target
// hidden P-1). Query, attention output and FFN have no consumer here. Also saves the per-row carry.
command.program.def public @llm_mtp_warm() launch(%w_embed: buffer, %weights: buffer, %target: buffer, %metadata: buffer, %origins: buffer, %arena: buffer, %carry: buffer, %tokens: buffer) {{
{P}  %owners = index.constant 1 : index
  %prev = buffer.alloca<global> align(256) %x_size : buffer
""" + _token_setup("%metadata", "%tokens", "%enorm_e", "%y", "%h", g).replace(
        "  %embedding = command.parameter", f'  %enorm_e = command.parameter %weights, "blk.{M}.nextn.enorm.weight" : view<{H}xf32>\n  %embedding = command.parameter') + f"""  kernel.launch @g_mtp_pair[%t](%t, %spans, %owners, %metadata, %info, %target, %carry, %prev) : [index](index, index, index, buffer, buffer, buffer, buffer, buffer)
  kernel.launch @g_mtp_save[%spans](%t, %spans, %owners, %metadata, %target, %carry) : [index](index, index, index, buffer, buffer, buffer)
""" + _mtp_input(g, "%prev") + f"""  %wk_m = command.parameter %weights, "{wk}" : view<{size(wk)}xi8>
  %wv_m = command.parameter %weights, "{wv}" : view<{size(wv)}xi8>
{norms}  {lin(fmt(wk), "%h", "%s_h", "%wk_m", size(wk), "%k", "%nH", "%nKV", H)}
  {lin(fmt(wv), "%h", "%s_h", "%wv_m", size(wv), "%v", "%nH", "%nKV", H)}
  %mtp_layer = index.constant 0 : index
  %rope_on = scalar.constant 1 : i32
  kernel.launch @g_qk_rope_store[%t](%t, %mtp_layer, %rope_on, %context, %pool, %q, %k, %v, {qn}, {kn}, %info, %origins, %arena{mr}) : [index](index, index, i32, index, index, buffer, buffer, buffer, {nty}, {nty}, buffer, buffer, buffer{mrt})
  command.return
}}
""")
    # Warm reads a subset of the block's tensors, which the proposal stages (declared first) already placed
    # in one root: give each its own root so it shares that placement (draft/build_draft.py binds them).
    names = list(dict.fromkeys(re.findall(r'command\.parameter %weights, "([^"]+)"', warm)))
    assert len(names) == warm_root_count(g), names
    for n, name in enumerate(names):
        warm = warm.replace(f'command.parameter %weights, "{name}"', f'command.parameter %wr{n}, "{name}"')
    warm = warm.replace("launch(%w_embed: buffer, %weights: buffer,", "launch(%w_embed: buffer, " + "".join(f"%wr{n}: buffer, " for n in range(len(names))), 1)
    layer = g["layer_text"](M, ordinal=0, last_norm="%head_norm", kind_note=" (MTP NextN block)")
    # Proposals are one token per span: attention always takes the decode (split-K) path, whatever the
    # cohort size, and the split buffers are sized for it.
    for k in ("@g_attention_flash2", "@g16_attention_split_listed"):   # flash2 is absent for non-power-of-two groups
        a = f"  scf.if %gemm {{\n    kernel.launch {k}"
        if a in layer:
            layer = layer.replace(a, f"  scf.if %attn_tiles {{\n    kernel.launch {k}")
            break
    else:
        raise AssertionError("attention block not found in the MTP layer text")
    Pp = P.replace("%prefill_shape = index.cmp uge, %t, %c64 : index", "%prefill_shape = index.cmp uge, %t, %c_never : index", 1)
    assert Pp != P
    Pp += "  %attn_tiles = index.cmp uge, %t, %c_never : index\n"
    head_name, hsize = g["head_name"], g["head_size"]
    head_root = "" if head_name == "token_embd.weight" else "%w_head: buffer, "
    propose = (f"""
// MTP proposals: draft_depth autoregressive rounds of the NextN block for the PROPOSE spans; each round's
// greedy token goes straight into the verifier's input slot and feeds the next round.
command.program.def public @llm_mtp_propose() launch(%w_embed: buffer, {head_root}%weights: buffer, %epoch: buffer, %target_tokens: buffer, %origins: buffer, %arena: buffer, %carry: buffer) {{
{Pp}  %owners = index.constant 1 : index
  %epoch_spans = config.get @llm.epoch_spans : index
  %epoch_tokens = config.get @llm.epoch_tokens : index
  %width = config.get @llm.width : index
  %depth = index.sub %width, %c1 : index
  %meta_bytes = index.constant {16 + 40 * 8192} : offset
  %metadata = buffer.alloca<global> align(256) %meta_bytes : buffer
  %c4_o = index.constant 4 : offset
  %word_size = index.scale %t, %c4_o : index, offset -> offset
  %tokens = buffer.alloca<global> align(256) %word_size : buffer
  %slots = buffer.alloca<global> align(256) %word_size : buffer
  %proposals = buffer.alloca<global> align(256) %word_size : buffer
  %hidden = buffer.alloca<global> align(256) %x_size : buffer
  %head = command.parameter {"%w_embed" if head_name == "token_embd.weight" else "%w_head"}, "{head_name}" : view<{hsize}xi8>
  %head_norm = command.parameter %weights, "blk.{M}.nextn.shared_head_norm.weight" : view<{H}xf32>
  kernel.launch @g_mtp_prepare(%epoch_spans, %t, %epoch_tokens, %epoch, %target_tokens, %metadata, %tokens, %slots) : (index, index, index, buffer, buffer, buffer, buffer, buffer)
  kernel.launch @g_mtp_gather[%t](%t, %owners, %metadata, %carry, %hidden) : [index](index, index, buffer, buffer, buffer)
  scf.for %round = [%c0 to %depth step %c1] unroll {{
""" + _token_setup("%metadata", "%tokens", "%enorm_e", "%y", "%h", g).replace(
        "  %embedding = command.parameter", f'  %enorm_e = command.parameter %weights, "blk.{M}.nextn.enorm.weight" : view<{H}xf32>\n  %embedding = command.parameter')
        + _mtp_input(g, "%hidden") + layer + f"""  kernel.launch @g_mtp_copy[%t](%t, %h, %hidden) : [index](index, buffer, buffer)
  %one_draft = index.cmp eq, %t, %c1 : index
  scf.if %one_draft {{
    kernel.launch @g_head_mv[%t](%t, %t, %metadata, %h, %head, %partials, %logits) : [index](index, index, buffer, buffer, view<{hsize}xi8>, buffer, buffer)
  }} else {{
    kernel.launch @g_head_mv_t4[%t](%t, %t, %metadata, %h, %head, %partials, %logits) : [index](index, index, buffer, buffer, view<{hsize}xi8>, buffer, buffer)
  }}
  kernel.launch @g_head_select[%t](%t, %metadata, %partials, %logits, %proposals) : [index](index, buffer, buffer, buffer, buffer)
  kernel.launch @g_mtp_advance(%round, %t, %epoch_tokens, %epoch, %metadata, %slots, %proposals, %tokens, %target_tokens) : (index, index, index, buffer, buffer, buffer, buffer, buffer, buffer)
  }}
  command.return
}}
""")
    copy = f"""
kernel.def export("g_mtp_copy") @g_mtp_copy(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %input: buffer, %output: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [range(%token_id, 0, 2147483646), lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %groups = index.constant {g['HG']} : index
  %active = index.cmp ult, %lane, %groups : index
  scf.if %active {{
    %bl = index.assume %lane [range(%lane, 0, {g['HG'] - 1})] : index
    %i_view = buffer.view %input[%zero] : buffer -> view<[%tokens]x{g['HG']}x32xf32>
    %o_view = buffer.view %output[%zero] : buffer -> view<[%tokens]x{g['HG']}x32xf32>
    %v = vector.load %i_view[%token, %bl, %c0] : view<[%tokens]x{g['HG']}x32xf32> -> vector<32xf32>
    vector.store %v, %o_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x{g['HG']}x32xf32>
  }}
  kernel.return
}}
"""
    decls = ("config.decl @llm.epoch_spans : %value: index where [range(%value, 1, 8192)]\n\n"
             "config.decl @llm.epoch_tokens : %value: index where [range(%value, 1, 8192)]\n")
    return decls + copy + warm + propose
