"""LFM2 short-conv token mixer kernels (recurrent conv history in the runtime's recurrent slots).

in_proj gives bcx[t] = [B | C | X] (3 x H). bx = B * X is the conv input; the depthwise causal
conv (kernel K = l_cache) runs over the stored history (the last K-1 bx rows of the span's
reader slot) and the span's own rows; out = C * conv. The commit kernel publishes the last K-1
bx rows of each span to its writer slot. Same slot protocol as gen_gdn.py (g_gdn_slots/g_gdn_span).
Speculative decoding (LLM_DRAFT): verification spans (flag bit 4) do not commit; replay_spec() registers the
capture of bcx and a commit over the accepted inputs with draft/gen_draft.py's generic recurrent replay.
"""
import os
from gen_gdn import SLOT


def sconv_geometry(cfg):
    H, K = cfg["hidden"], cfg["shortconv"]["l_cache"]
    layer_bytes = ((K - 1) * H * 4 + 255) // 256 * 256
    return dict(H=H, K=K, layer_bytes=layer_bytes, slot_bytes=layer_bytes * len(cfg["shortconv_layers"]))


def replay_spec(cfg):
    """gen_draft.verify_program's recurrent replay: capture bcx (3H f32 per token) before each short-conv layer's
    conv, then commit the accepted inputs' history (the %replay packet) after acceptance."""
    W = 3 * cfg["hidden"]
    return {"name": "sconv", "match": "  kernel.launch @g_sc_conv[%t](%t, %spans, %layer_", "buffers": [("%bcx", W)],
            "replay": lambda layer: (f"  kernel.launch @g_sc_commit[%spans](%t, %spans, %layer_{layer}, %replay, %origins, %arena, %bcx)"
                                     " : [index](index, index, index, buffer, buffer, buffer, buffer)\n")}


def sconv_kernels(cfg, with_slot=True):
    G = sconv_geometry(cfg)
    C, K, LB = G["H"], G["K"], G["layer_bytes"]
    NG = len(cfg["shortconv_layers"])
    CB = (C + 255) // 256
    T = [SLOT] if with_slot else []
    taps = []
    for j in range(K):
        back = K - 1 - j
        taps.append(f"""      %w{j} = view.load %w_view[%ch, {j}] : view<{C}x{K}xf32> -> f32
      %back{j} = index.constant {back} : index
      %has{j} = index.cmp uge, %rel, %back{j} : index
      %x{j} = scf.if %has{j} -> (f32) {{
        %src{j} = index.sub %token, %back{j} : index
        %bsrc{j} = index.assume %src{j} [lt(%src{j}, %tokens)] : index
        %bb{j} = view.load %bcx_view[%bsrc{j}, 0, %ch] : view<[%tokens]x3x{C}xf32> -> f32
        %xx{j} = view.load %bcx_view[%bsrc{j}, 2, %ch] : view<[%tokens]x3x{C}xf32> -> f32
        %v{j} = scalar.mulf %bb{j}, %xx{j} : f32
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
    T.append(f"""
// out[t, c] = C[t, c] * sum_j w[c, j] bx[t - (K-1) + j, c]; bx before the span comes from the history.
kernel.def export("g_sc_conv") @g_sc_conv(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cblocks = index.constant {CB} : index
  kernel.launch.config workgroups(%tokens, %cblocks, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %layer_arg: index, %metadata: buffer, %info: buffer, %origins: buffer, %arena: buffer, %bcx: buffer, %weight: buffer, %out: buffer) {{
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
  %c256 = index.constant 256 : index
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %out_view = buffer.view %out[%zero] : buffer -> view<[%tokens]x{C}xf32>
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %span32 = view.load %info_view[%token, 3] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %cblk_id = kernel.workgroup.id<y> : index
  %cblk_off = index.mul %cblk_id, %c256 : index
  %c_raw = index.add %thread, %cblk_off : index
  %cC = index.constant {C} : index
  %in_c = index.cmp ult, %c_raw, %cC : index
  %go = scalar.andi %live, %in_c : i1
  scf.if %go {{
    %ch = index.assume %c_raw [range(%c_raw, 0, {C - 1})] : index
    %span_bound = scalar.assume %span32 [range(%span32, 0, 8191)] : i32
    %span = index.cast %span_bound : i32 to index
    %active, %length, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
    %layer_stride = index.constant {LB} : offset
    %layer_bytes = index.scale %layer, %layer_stride : index, offset -> offset
    %reader, %writer, %fresh = func.call @g_gdn_slots(%origins, %row, %start32, %layer_bytes) : (buffer, index, i32, offset) -> (offset, offset, i1)
    %hist_view = buffer.view %arena[%reader] : buffer -> view<{K - 1}x{C}xf32>
    %bcx_view = buffer.view %bcx[%zero] : buffer -> view<[%tokens]x3x{C}xf32>
    %w_view = buffer.view %weight[%zero] : buffer -> view<{C}x{K}xf32>
    %rel_value = index.sub %token, %first : index
    %rel = index.assume %rel_value [range(%rel_value, 0, 8191)] : index
{chr(10).join(taps)}
    %cc = view.load %bcx_view[%token, 1, %ch] : view<[%tokens]x3x{C}xf32> -> f32
    %y = scalar.mulf %cc, %acc{K - 1} : f32
    view.store %y, %out_view[%token, %ch] : f32, view<[%tokens]x{C}xf32>
  }} else {{
    scf.if %in_c {{
      %chz = index.assume %c_raw [range(%c_raw, 0, {C - 1})] : index
      view.store %zero_f, %out_view[%token, %chz] : f32, view<[%tokens]x{C}xf32>
    }}
  }}
  kernel.return
}}
""")
    reads, writes = [], []
    for k in range(K - 1):
        reads.append(f"""      %need{k} = index.constant {K - 1 - k} : index
      %has{k} = index.cmp uge, %length, %need{k} : index
      %val{k} = scf.if %has{k} -> (f32) {{
        %off{k}a = index.add %first, %length : index
        %src{k} = index.sub %off{k}a, %need{k} : index
        %bsrc{k} = index.assume %src{k} [lt(%src{k}, %tokens)] : index
        %bb{k} = view.load %bcx_view[%bsrc{k}, 0, %ch] : view<[%tokens]x3x{C}xf32> -> f32
        %xx{k} = view.load %bcx_view[%bsrc{k}, 2, %ch] : view<[%tokens]x3x{C}xf32> -> f32
        %v{k} = scalar.mulf %bb{k}, %xx{k} : f32
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
    spec = bool(os.environ.get("LLM_DRAFT")) and os.environ.get("LLM_DRAFT") != "null"
    guard = ("\n  %publish = func.call @g_gdn_publish(%metadata, %spans, %span) : (buffer, index, index) -> (i1)"
             "\n  %go_live = scalar.andi %go, %publish : i1") if spec else ""
    go = "%go_live" if spec else "%go"
    T.append(f"""
kernel.def export("g_sc_commit") @g_sc_commit(%spans: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cblocks = index.constant {CB} : index
  kernel.launch.config workgroups(%spans, %cblocks, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %layer_arg: index, %metadata: buffer, %origins: buffer, %arena: buffer, %bcx: buffer) {{
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
  %cblk_id = kernel.workgroup.id<y> : index
  %cblk_off = index.mul %cblk_id, %c256 : index
  %c_raw = index.add %thread, %cblk_off : index
  %in_c = index.cmp ult, %c_raw, %cC : index
  %go = scalar.andi %active, %in_c : i1{guard}
  scf.if {go} {{
    %ch = index.assume %c_raw [range(%c_raw, 0, {C - 1})] : index
    %layer_stride = index.constant {LB} : offset
    %layer_bytes = index.scale %layer, %layer_stride : index, offset -> offset
    %reader, %writer, %fresh = func.call @g_gdn_slots(%origins, %row, %start32, %layer_bytes) : (buffer, index, i32, offset) -> (offset, offset, i1)
    %old_view = buffer.view %arena[%reader] : buffer -> view<{K - 1}x{C}xf32>
    %new_view = buffer.view %arena[%writer] : buffer -> view<{K - 1}x{C}xf32>
    %bcx_view = buffer.view %bcx[%zero] : buffer -> view<[%tokens]x3x{C}xf32>
{chr(10).join(reads)}
{chr(10).join(writes)}
  }}
  kernel.return
}}
""")
    return "\n".join(T)
