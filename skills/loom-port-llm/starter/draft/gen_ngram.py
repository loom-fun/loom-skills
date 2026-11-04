"""Prompt-lookup (n-gram) drafting for starter ports: speculative decoding with no drafter weights.

Works for every model family the starter ports (no MTP block, no DFlash drafter needed): the proposals come
from the row's own token history, so it pays off where the output repeats its context (tool calls quoting
arguments, code edits, JSON, summaries that quote) and is a cheap guess elsewhere. draft/tune.py measures it
like the other drafters and keeps it only when decoding gets faster.

- History: every consumed input's token id at its position, 4 bytes per position in the draft cache (one plane
  over the target pool's page ids, so prefix reuse, checkpoints and the full context come for free).
  @llm_ngram_warm stores the committed inputs after every epoch (prefill included).
- Proposal (@llm_ngram_propose, one workgroup per PROPOSE span): the anchor (the span's pending input at
  position P) and up to NGRAM - 1 history tokens before it form the query. Lanes scan the last WINDOW positions
  for the longest suffix match (ties: the newest), then the draft_depth tokens that followed the match are
  written into the verifier's input slots. A continuation that runs into position P continues with the anchor
  and the proposals already written (a repeating pattern drafts its own period). No match proposes token 0,
  which verification simply rejects. Acceptance (draft/gen_draft.py) is unchanged: output stays exactly the
  target's.
Same idea as hrx-loom-library model/Qwen/Qwen3.8-Flash-Next/program/kernel/lookup.loom (paged accepted
history) and llama.cpp's ngram speculative decoding.
"""
import os

NGRAM = int(os.environ.get("NGRAM_MAX", "4"))          # longest suffix compared (anchor included)
WINDOW = int(os.environ.get("NGRAM_WINDOW", "32768"))  # history positions scanned per proposal (newest first)
LANES = 256


def kernels(g):
    assert 1 <= NGRAM <= 8 and WINDOW % LANES == 0
    keys = "\n".join(f"""  %kp{j}_ok = index.cmp uge, %position, %c{j}_i : index
  %k{j} = scf.if %kp{j}_ok -> (i32) {{
    %kp{j} = index.sub %position, %c{j}_i : index
    %kv{j} = func.call @g_ng_load(%origins, %cache, %row, %kp{j}, %blocks, %pool) : (buffer, buffer, index, index, index, index) -> (i32)
    scf.yield %kv{j} : i32
  }} else {{
    scf.yield %none : i32
  }}""" for j in range(1, NGRAM))
    consts = "\n".join(f"  %c{j}_i = index.constant {j} : index" for j in range(1, NGRAM))
    # match length at candidate q: 1 + #consecutive j with hist[q - j] == k_j
    match = []
    prev = "%c1_i32"
    alive = "%true"
    for j in range(1, NGRAM):
        match.append(f"""      %has{j} = index.cmp uge, %q, %c{j}_i : index
      %chk{j} = scalar.andi {alive}, %has{j} : i1
      %eq{j} = scf.if %chk{j} -> (i1) {{
        %qp{j} = index.sub %q, %c{j}_i : index
        %hv{j} = func.call @g_ng_load(%origins, %cache, %row, %qp{j}, %blocks, %pool) : (buffer, buffer, index, index, index, index) -> (i32)
        %same{j} = scalar.cmpi eq, %hv{j}, %k{j} : i32
        scf.yield %same{j} : i1
      }} else {{
        scf.yield %false : i1
      }}
      %inc{j} = scalar.extui %eq{j} : i1 to i32
      %m{j} = scalar.addi {prev}, %inc{j} : i32""")
        prev, alive = f"%m{j}", f"%eq{j}"
    match = "\n".join(match)
    return f"""
// Token history: one i32 per position in the draft cache, same 16-position pages as the target pool.
func.def inline @g_ng_offset(%origins: buffer, %row: index, %position: index, %blocks: index) -> (offset) {{
  %c1 = index.constant 1 : index
  %c16 = index.constant 16 : index
  %record_stride = index.constant 40 : offset
  %position_bytes = index.constant 4 : offset
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
  %bytes = index.scale %slot, %position_bytes : index, offset -> offset
  func.return %bytes : offset
}}

func.def inline @g_ng_load(%origins: buffer, %cache: buffer, %row: index, %position: index, %blocks: index, %pool: index) -> (i32) {{
  %at = func.call @g_ng_offset(%origins, %row, %position, %blocks) : (buffer, index, index, index) -> (offset)
  %view = buffer.view %cache[%at] : buffer -> view<1xi32>
  %v = view.load %view[0] : view<1xi32> -> i32
  func.return %v : i32
}}

// history[row][position] = id of every live input of the (committed) packet.
kernel.def export("g_ng_store") @g_ng_store(%tokens: index) {{
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
}} launch(%tokens_arg: index, %context_arg: index, %pool_arg: index, %metadata: buffer, %info: buffer, %ids: buffer, %origins: buffer, %cache: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %context = index.assume %context_arg [range(%context_arg, 16, 262144)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 1048576)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [range(%token_id, 0, 2147483646), lt(%token_id, %tokens)] : index
  %zero = index.constant 0 : offset
  %c1i = index.constant 1 : index
  %c8 = index.constant 8 : index
  %c15 = index.constant 15 : index
  %c16 = index.constant 16 : index
  %c0_i32 = scalar.constant 0 : i32
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %row32 = view.load %info_view[%token, 0] : view<[%tokens]x4xi32> -> i32
  %position32 = view.load %info_view[%token, 1] : view<[%tokens]x4xi32> -> i32
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  scf.if %live {{
    %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
    %mixed_word = view.load %header[2] : view<4xi32> -> i32
    %mixed = scalar.cmpi ne, %mixed_word, %c0_i32 : i32
    %id_stride = scf.select %mixed, %c8, %c1i : index
    %n_ids = index.mul %tokens, %id_stride : index
    %ids_flat = buffer.view %ids[%zero] : buffer -> view<[%n_ids]xi32>
    %e0 = index.mul %token, %id_stride : index
    %e = index.assume %e0 [range(%e0, 0, 65535), lt(%e0, %n_ids)] : index
    %id = view.load %ids_flat[%e] : view<[%n_ids]xi32> -> i32
    %row_b = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
    %row = index.cast %row_b : i32 to index
    %pos_b = scalar.assume %position32 [range(%position32, 0, 262143)] : i32
    %position = index.cast %pos_b : i32 to index
    %padded = index.add %context, %c15 : index
    %blocks = index.div %padded, %c16 : index
    %at = func.call @g_ng_offset(%origins, %row, %position, %blocks) : (buffer, index, index, index) -> (offset)
    %slot = buffer.view %cache[%at] : buffer -> view<1xi32>
    view.store %id, %slot[0] : i32, view<1xi32>
  }}
  kernel.return
}}

// One workgroup per epoch span: PROPOSE spans look their anchor's suffix up in the row's history (newest
// {WINDOW} positions, up to {NGRAM} tokens compared) and write draft_depth proposals into the input slots
// cursor + 1 .. cursor + depth.
kernel.def export("g_ng_propose") @g_ng_propose(%spans: index) {{
  %c1 = index.constant 1 : index
  %lanes = index.constant {LANES} : index
  kernel.launch.config workgroups(%spans, %c1, %c1) workgroup_size(%lanes, %c1, %c1) : index
}} launch(%spans_arg: index, %tokens_arg: index, %depth_arg: index, %context_arg: index, %pool_arg: index, %metadata: buffer, %ids: buffer, %origins: buffer, %cache: buffer) {{
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %depth = index.assume %depth_arg [range(%depth_arg, 1, 7)] : index
  %context = index.assume %context_arg [range(%context_arg, 16, 262144)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 1048576)] : index
  %span_id = kernel.workgroup.id<x> : index
  %span = index.assume %span_id [range(%span_id, 0, 2147483646), lt(%span_id, %spans)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, {LANES - 1})] : index
  %zero = index.constant 0 : offset
  %c16o = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1i = index.constant 1 : index
  %c8 = index.constant 8 : index
  %c15 = index.constant 15 : index
  %c16 = index.constant 16 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %c2_i32 = scalar.constant 2 : i32
  %c20_i32 = scalar.constant 20 : i32
  %none = scalar.constant -1 : i32
  %true = scalar.constant true : i1
  %false = scalar.constant false : i1
{consts}
  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %mixed_word = view.load %header[2] : view<4xi32> -> i32
  %mixed = scalar.cmpi ne, %mixed_word, %c0_i32 : i32
  %id_stride = scf.select %mixed, %c8, %c1i : index
  %n_ids = index.mul %tokens, %id_stride : index
  %ids_flat = buffer.view %ids[%zero] : buffer -> view<[%n_ids]xi32>
  %count_b = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count = index.cast %count_b : i32 to index
  %in_count = index.cmp ult, %span, %count : index
  %records = buffer.view %metadata[%c16o] : buffer -> view<[%spans]x9xi32>
  %flags = scf.if %in_count -> (i32) {{
    %f = view.load %records[%span, 6] : view<[%spans]x9xi32> -> i32
    scf.yield %f : i32
  }} else {{
    scf.yield %c0_i32 : i32
  }}
  %prop_bits = scalar.andi %flags, %c2_i32 : i32
  %proposes = scalar.cmpi ne, %prop_bits, %c0_i32 : i32
  scf.if %proposes {{
    %position32 = view.load %records[%span, 1] : view<[%spans]x9xi32> -> i32
    %row32 = view.load %records[%span, 3] : view<[%spans]x9xi32> -> i32
    %cursor32 = view.load %records[%span, 4] : view<[%spans]x9xi32> -> i32
    %row_b = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
    %row = index.cast %row_b : i32 to index
    %pos_b = scalar.assume %position32 [range(%position32, 0, 262143)] : i32
    %position = index.cast %pos_b : i32 to index
    %cursor_b = scalar.assume %cursor32 [range(%cursor32, 0, 8191)] : i32
    %cursor = index.cast %cursor_b : i32 to index
    %padded = index.add %context, %c15 : index
    %blocks = index.div %padded, %c16 : index
    %ae0 = index.mul %cursor, %id_stride : index
    %ae = index.assume %ae0 [range(%ae0, 0, 65535), lt(%ae0, %n_ids)] : index
    %k0 = view.load %ids_flat[%ae] : view<[%n_ids]xi32> -> i32
{keys}
    // newest-first scan of positions position-1 .. position-window; key = match length << 20 | q
    %window = index.constant {WINDOW} : index
    %span_len = index.min %position, %window : index
    %lanes_c = index.constant {LANES} : index
    %best = scf.for %i = [%lane to %span_len step %lanes_c](%acc = %c0_i32 : i32) -> (i32) {{
      %back = index.add %i, %c1i : index
      %q = index.sub %position, %back : index
      %h0 = func.call @g_ng_load(%origins, %cache, %row, %q, %blocks, %pool) : (buffer, buffer, index, index, index, index) -> (i32)
      %hit = scalar.cmpi eq, %h0, %k0 : i32
      %score = scf.if %hit -> (i32) {{
{match}
        %len_bits = scalar.shli {prev}, %c20_i32 : i32
        %q32 = index.cast %q : index to i32
        %key = scalar.ori %len_bits, %q32 : i32
        scf.yield %key : i32
      }} else {{
        scf.yield %c0_i32 : i32
      }}
      %better = scalar.maxui %acc, %score : i32
      scf.yield %better : i32
    }}
    %winner = kernel.workgroup.reduce<maxui> %best : i32
    %first = index.cmp eq, %lane, %c0 : index
    scf.if %first {{
      %found = scalar.cmpi ne, %winner, %c0_i32 : i32
      %qmask = scalar.constant 1048575 : i32
      %q32w = scalar.andi %winner, %qmask : i32
      %q_b = scalar.assume %q32w [range(%q32w, 0, 262143)] : i32
      %qw = index.cast %q_b : i32 to index
      scf.for %j = [%c0 to %depth step %c1i] {{
        %step = index.add %j, %c1i : index
        %x = index.add %qw, %step : index
        %before = index.cmp ult, %x, %position : index
        %token = scf.if %found -> (i32) {{
          %t = scf.if %before -> (i32) {{
            %hx = func.call @g_ng_load(%origins, %cache, %row, %x, %blocks, %pool) : (buffer, buffer, index, index, index, index) -> (i32)
            scf.yield %hx : i32
          }} else {{
            // the match runs into the span: anchor (x == position), then the proposals already written
            %rel = index.sub %x, %position : index
            %s0 = index.add %cursor, %rel : index
            %s1 = index.mul %s0, %id_stride : index
            %sb = index.assume %s1 [range(%s1, 0, 65535), lt(%s1, %n_ids)] : index
            %pv = view.load %ids_flat[%sb] : view<[%n_ids]xi32> -> i32
            scf.yield %pv : i32
          }}
          scf.yield %t : i32
        }} else {{
          scf.yield %c0_i32 : i32
        }}
        %d0 = index.add %cursor, %step : index
        %d1 = index.mul %d0, %id_stride : index
        %db = index.assume %d1 [range(%d1, 0, 65535), lt(%d1, %n_ids)] : index
        view.store %token, %ids_flat[%db] : i32, view<[%n_ids]xi32>
      }}
    }}
  }}
  kernel.return
}}
"""


def programs(g):
    common = """  %context = config.get @llm.context : index
  %pool = config.get @llm.pool : index
"""
    warm = f"""
// N-gram drafting catch-up: the committed inputs' ids into the token history.
command.program.def public @llm_ngram_warm() launch(%residual: buffer, %metadata: buffer, %origins: buffer, %arena: buffer, %tokens: buffer) {{
  %t = config.get @llm.tokens : index
  %spans = config.get @llm.spans : index
{common}  %info_bytes = index.constant 16 : offset
  %info_size = index.scale %t, %info_bytes : index, offset -> offset
  %info = buffer.alloca<global> align(256) %info_size : buffer
  kernel.launch @g_token_info[%t](%t, %spans, %metadata, %info) : [index](index, index, buffer, buffer)
  kernel.launch @g_ng_store[%t](%t, %context, %pool, %metadata, %info, %tokens, %origins, %arena) : [index](index, index, index, buffer, buffer, buffer, buffer, buffer)
  command.return
}}
"""
    propose = f"""
// N-gram proposals for the epoch's PROPOSE spans, written into the verifier's input slots.
command.program.def public @llm_ngram_propose() launch(%epoch: buffer, %target_tokens: buffer, %origins: buffer, %arena: buffer) {{
{common}  %epoch_spans = config.get @llm.epoch_spans : index
  %epoch_tokens = config.get @llm.epoch_tokens : index
  %width = config.get @llm.width : index
  %c1 = index.constant 1 : index
  %depth = index.sub %width, %c1 : index
  kernel.launch @g_ng_propose[%epoch_spans](%epoch_spans, %epoch_tokens, %depth, %context, %pool, %epoch, %target_tokens, %origins, %arena) : [index](index, index, index, index, index, buffer, buffer, buffer, buffer)
  command.return
}}
"""
    decls = ("config.decl @llm.epoch_spans : %value: index where [range(%value, 1, 8192)]\n\n"
             "config.decl @llm.epoch_tokens : %value: index where [range(%value, 1, 8192)]\n")
    return decls + warm + propose
