"""Serving-glue patches for speculative decoding (build.py applies them when a drafter is selected).

prepare.loom: one verify and one drafter catch-up (warm) stage per packing shape next to the ordinary
  advance stage, and proposal stages for power-of-two cohorts (1, 2, 4, ... up to the largest span count).
storage.loom: the drafter's allocations: carry (MTP hidden per resident/checkpoint owner), committed
  packet, verification results (two feedback banks), draft K/V cache (same page ids as the target pool),
  draft page table; final hidden rows in the residual allocation.
control.loom: control state grows to 2048 header bytes (sampling records move from 640 to 2048);
  encode_epoch places speculative spans (anchor + placeholder slots) at packed cursors and marks them in the
  device plan copy (flags bit 4, credit in bits 8..15); publish_epoch reads verification results; epoch
  runs propose -> verify (or advance) -> warm.
Native contract: hrx-system loom/src/loom/serve/service/text/{model,storage,epoch}.h (hrx-loom 0.1.0).
"""
import re


def hx(s):
    return s.encode().hex()


SAMPLING = 2048          # control-state byte offset of the per-row sampling records
VERIFY_TABLE, WARM_TABLE = 640, 1152
PROPOSE_FIRST, PROPOSE_LEVELS, DRAFT_KIND = 1664, 1668, 1672


def _sub(text, old, new, count=1):
    assert text.count(old) >= 1, f"draft patch anchor not found: {old[:80]!r}"
    return text.replace(old, new, count)


# ---------------------------------------------------------------------------------------------------
def warm_roots(cfg, draft):
    """Catch-up stage roots: the embedding plus one per drafter tensor it reads (draft/gen_draft.py)."""
    return 1 + 6 + (2 if cfg.get("qk_norm") else 0)


def drafter_roots(cfg, draft):
    """Parameter roots of the drafter stages: tensors the target already places (embedding, head) each get
    their own root so they share the target's storage; the drafter's own tensors form the last root."""
    return 2 if cfg.get("tied_embeddings") else 3


def patch_prepare(text, draft, cfg):
    kind = draft["kind"]
    roots = drafter_roots(cfg, draft)
    wroots = warm_roots(cfg, draft)
    warm_root = {"mtp": "llm_mtp_warm"}[kind]
    propose_root = {"mtp": "llm_mtp_propose"}[kind]
    text = _sub(text, '''// "weights.loom"''', f'''// "llm_verify", "{warm_root}", "{propose_root}", "llm.width", "llm.epoch_spans", "llm.epoch_tokens"
global.rodata.def @verify_root = bytes("{hx("llm_verify")}")

global.rodata.def @warm_root = bytes("{hx(warm_root)}")

global.rodata.def @propose_root = bytes("{hx(propose_root)}")

global.rodata.def @width_key = bytes("{hx("llm.width")}")

global.rodata.def @epoch_spans_key = bytes("{hx("llm.epoch_spans")}")

global.rodata.def @epoch_tokens_key = bytes("{hx("llm.epoch_tokens")}")

// "weights.loom"''')
    # draft depth 1..7 is supported (0 = ordinary decoding)
    text = _sub(text, "  %mode_ok = scalar.andi %has_epochs, %no_draft : i1\n",
                "  %c7_i32 = scalar.constant 7 : i32\n  %depth_ok = scalar.cmpi ule, %draft_depth, %c7_i32 : i32\n"
                "  %mode_ok = scalar.andi %has_epochs, %depth_ok : i1\n")
    text = _sub(text, "  %control_bytes = index.add %header_bytes, %stage_bytes : offset\n",
                f"  %control_bytes = index.constant {SAMPLING} : offset\n")
    text = _sub(text, "  %stage_view = buffer.view %state[%header_bytes] : buffer -> view<[%count]xi32>\n",
                f"""  %stage_view = buffer.view %state[%header_bytes] : buffer -> view<[%count]xi32>
  %verify_at = index.constant {VERIFY_TABLE} : offset
  %warm_at = index.constant {WARM_TABLE} : offset
  %verify_view = buffer.view %state[%verify_at] : buffer -> view<[%count]xi32>
  %warm_view = buffer.view %state[%warm_at] : buffer -> view<[%count]xi32>
  %drafting = scalar.cmpi ne, %draft_depth, %c0_i32 : i32
  %depth64 = scalar.extui %draft_depth : i32 to i64
  %c1_w = scalar.constant 1 : i64
  %width = scalar.addi %depth64, %c1_w : i64
  %verify_root = global.load @verify_root : buffer
  %warm_root = global.load @warm_root : buffer
  %width_key = global.load @width_key : buffer
""")
    text = _sub(text, "  %smaller = scalar.cmpi ult, %maximum, %prefill : i64\n", f"""  // Verify and catch-up stages per shape, declared after every advance stage (the first shape_count
  // stages are the shared target stages), then proposal stages for cohorts of 1, 2, 4, ... spans.
  %draft_roots = index.constant {roots} : index
  %warm_roots = index.constant {wroots} : index
  scf.if %drafting {{
    %propose_root = global.load @propose_root : buffer
    %epoch_spans_key = global.load @epoch_spans_key : buffer
    %epoch_tokens_key = global.load @epoch_tokens_key : buffer
    %span32 = scalar.trunci %span_capacity : i64 to i32
    %one32 = scalar.constant 1 : i32
    %bits32 = scalar.constant 32 : i32
    %below = scalar.subi %span32, %one32 : i32
    %leading = scalar.ctlzi %below : i32
    %last_level = scalar.subi %bits32, %leading : i32
    %levels32 = scalar.addi %last_level, %one32 : i32
    %levels_b = scalar.assume %levels32 [range(%levels32, 1, 14)] : i32
    %levels = index.cast %levels_b : i32 to index
    %c1_64 = scalar.constant 1 : i64
    %first_propose = scf.for %level = [%c0 to %levels step %c1](%first = %c0_i32 : i32) -> (i32) {{
      %shift = index.cast %level : index to i64
      %cap = scalar.shli %c1_64, %shift : i64
      %pstage = func.call @stage(%propose_root, %shift) : (buffer, i64) -> (i32)
      func.call @config_i64(%pstage, %context_key, %context) : (i32, buffer, i64)
      func.call @config_i64(%pstage, %pool_key, %pool) : (i32, buffer, i64)
      func.call @config_i64(%pstage, %tokens_key, %cap) : (i32, buffer, i64)
      func.call @config_i64(%pstage, %spans_key, %cap) : (i32, buffer, i64)
      func.call @config_i64(%pstage, %width_key, %width) : (i32, buffer, i64)
      func.call @config_i64(%pstage, %epoch_spans_key, %span_capacity) : (i32, buffer, i64)
      func.call @config_i64(%pstage, %epoch_tokens_key, %maximum) : (i32, buffer, i64)
      scf.for %root = [%c0 to %draft_roots step %c1] {{
        %binding = index.cast %root : index to i32
        func.call @parameter(%pstage, %binding, %c0_source, %checkpoint, %policy) : (i32, i32, i32, i32, buffer)
      }}
      %initial = index.cmp eq, %level, %c0 : index
      %next_first = scf.select %initial, %pstage, %first : i32
      scf.yield %next_first : i32
    }}
    scf.for %i = [%c0 to %count step %c1] {{
      %tokens = view.load %shape_view[%i, 0] : view<[%count]x2xi64> -> i64
      %spans = view.load %shape_view[%i, 1] : view<[%count]x2xi64> -> i64
      %tag = index.cast %i : index to i64
      %vstage = func.call @stage(%verify_root, %tag) : (buffer, i64) -> (i32)
      func.call @config_i64(%vstage, %context_key, %context) : (i32, buffer, i64)
      func.call @config_i64(%vstage, %pool_key, %pool) : (i32, buffer, i64)
      func.call @config_i64(%vstage, %tokens_key, %tokens) : (i32, buffer, i64)
      func.call @config_i64(%vstage, %spans_key, %spans) : (i32, buffer, i64)
      func.call @config_i64(%vstage, %width_key, %width) : (i32, buffer, i64)
      func.call @parameter(%vstage, %c0_i32, %c0_source, %checkpoint, %policy) : (i32, i32, i32, i32, buffer)
      view.store %vstage, %verify_view[%i] : i32, view<[%count]xi32>
      %wstage = func.call @stage(%warm_root, %tag) : (buffer, i64) -> (i32)
      func.call @config_i64(%wstage, %context_key, %context) : (i32, buffer, i64)
      func.call @config_i64(%wstage, %pool_key, %pool) : (i32, buffer, i64)
      func.call @config_i64(%wstage, %tokens_key, %tokens) : (i32, buffer, i64)
      func.call @config_i64(%wstage, %spans_key, %spans) : (i32, buffer, i64)
      scf.for %root = [%c0 to %warm_roots step %c1] {{
        %binding = index.cast %root : index to i32
        func.call @parameter(%wstage, %binding, %c0_source, %checkpoint, %policy) : (i32, i32, i32, i32, buffer)
      }}
      view.store %wstage, %warm_view[%i] : i32, view<[%count]xi32>
    }}
    %draft_at = index.constant {PROPOSE_FIRST} : offset
    %draft_view = buffer.view %state[%draft_at] : buffer -> view<3xi32>
    view.store %first_propose, %draft_view[0] : i32, view<3xi32>
    view.store %levels32, %draft_view[1] : i32, view<3xi32>
    view.store %one32, %draft_view[2] : i32, view<3xi32>
  }}
  %smaller = scalar.cmpi ult, %maximum, %prefill : i64
""")
    text = _sub(text, "  %allocations, %views, %origins, %draft, %geometry = func.call @llm_storage(%maximum, %span_capacity, %context, %pool, %rows, %pins) : (i64, i64, i64, i64, i64, i64) -> (buffer, buffer, buffer, buffer, buffer)\n  %terminal = global.load @terminal : buffer\n  func.return %effective, %stage_count32, %state, %terminal, %allocations, %views, %origins, %empty, %geometry",
                "  %allocations, %views, %origins, %draft, %geometry = func.call @llm_storage(%maximum, %span_capacity, %context, %pool, %rows, %pins) : (i64, i64, i64, i64, i64, i64) -> (buffer, buffer, buffer, buffer, buffer)\n  %terminal = global.load @terminal : buffer\n  func.return %effective, %stage_count32, %state, %terminal, %allocations, %views, %origins, %draft, %geometry")
    return text


# ---------------------------------------------------------------------------------------------------
def patch_storage(text, cfg, draft):
    """Draft allocations 6..10, the draft cache region and the draft page table."""
    H = cfg["hidden"]
    pos_bytes = cfg["kv_heads"] * cfg["head_dim"] * 2
    planes = {"mtp": 2 * len(cfg["mtp"]["layers"])}[draft["kind"]]
    text = _sub(text, "  %residual_bytes = scalar.muli %span_capacity, %c4_i64 : i64\n",
                f"""  // Final normalized hidden rows (the drafter's catch-up input) live in the residual allocation.
  %hidden_row = scalar.constant {4 * H} : i64
  %residual_bytes = scalar.muli %maximum, %hidden_row : i64
  %span_floor = scalar.muli %span_capacity, %c4_i64 : i64
  %feedback_bytes = scalar.maxui %residual_bytes, %span_floor : i64
""")
    text = _sub(text, "  func.call @allocation(%allocations, %c5, %residual_bytes, %c0_i64) : (buffer, index, i64, i64)\n",
                f"""  func.call @allocation(%allocations, %c5, %span_floor, %c0_i64) : (buffer, index, i64, i64)
  // Drafter: carry (one hidden row per resident row and checkpoint), committed packet, verification
  // results (first bank 40 B/span, second bank 44 B/span), draft K/V ({planes} planes over the pool's
  // page ids), and the draft page table (same layout as the target table).
  %c6 = index.constant 6 : index
  %c7 = index.constant 7 : index
  %c8 = index.constant 8 : index
  %c9 = index.constant 9 : index
  %c10 = index.constant 10 : index
  %owners = scalar.addi %rows, %pins : i64
  %carry_bytes = scalar.muli %owners, %hidden_row : i64
  %committed_stride = scalar.constant 40 : i64
  %committed_records = scalar.muli %span_capacity, %committed_stride : i64
  %committed_bytes = scalar.addi %committed_records, %header_bytes : i64
  %result_stride = scalar.constant 84 : i64
  %results_bytes = scalar.muli %span_capacity, %result_stride : i64
  %first_bank_stride = scalar.constant 40 : i64
  %feedback_split = scalar.muli %span_capacity, %first_bank_stride : i64
  %draft_planes = scalar.constant {planes} : i64
  %draft_pos_bytes = scalar.constant {pos_bytes} : i64
  %draft_plane_bytes = scalar.muli %pool, %draft_pos_bytes : i64
  %draft_bytes = scalar.muli %draft_plane_bytes, %draft_planes : i64
  func.call @allocation(%allocations, %c6, %carry_bytes, %carry_bytes) : (buffer, index, i64, i64)
  func.call @allocation(%allocations, %c7, %committed_bytes, %c0_i64) : (buffer, index, i64, i64)
  func.call @allocation(%allocations, %c8, %results_bytes, %c0_i64) : (buffer, index, i64, i64)
  func.call @allocation(%allocations, %c9, %draft_bytes, %c0_i64) : (buffer, index, i64, i64)
  func.call @allocation(%allocations, %c10, %table_bytes, %c0_i64) : (buffer, index, i64, i64)
""")
    text = _sub(text, "  %geometry_bytes = index.constant 104 : offset\n  %geometry = buffer.alloca<private> align(8) %geometry_bytes : buffer\n  %geometry_view = buffer.view %geometry[%origin] : buffer -> view<13xi64>\n",
                "  %geometry_bytes = index.constant 144 : offset\n  %geometry = buffer.alloca<private> align(8) %geometry_bytes : buffer\n  %geometry_view = buffer.view %geometry[%origin] : buffer -> view<18xi64>\n")
    text = re.sub(r"view<13xi64>", "view<18xi64>", text)
    text = _sub(text, "  view.store %c0_i64, %geometry_view[2] : i64, view<18xi64>\n  view.store %c4_i64, %geometry_view[3] : i64, view<18xi64>\n",
                "  view.store %hidden_row, %geometry_view[2] : i64, view<18xi64>\n  view.store %feedback_split, %geometry_view[3] : i64, view<18xi64>\n")
    text = _sub(text, "  view.store %block_bytes, %geometry_view[12] : i64, view<18xi64>\n",
                f"""  view.store %block_bytes, %geometry_view[12] : i64, view<18xi64>
  %c9_i64 = scalar.constant 9 : i64
  %draft_block_bytes = scalar.constant {16 * pos_bytes} : i64
  view.store %c9_i64, %geometry_view[13] : i64, view<18xi64>
  view.store %c0_i64, %geometry_view[14] : i64, view<18xi64>
  view.store %draft_planes, %geometry_view[15] : i64, view<18xi64>
  view.store %draft_plane_bytes, %geometry_view[16] : i64, view<18xi64>
  view.store %draft_block_bytes, %geometry_view[17] : i64, view<18xi64>
  // The draft page table starts as a copy of the target's (page maps at the same offsets).
  %draft_origins = buffer.alloca<private> align(8) %origin_bytes : buffer
  buffer.copy %origins[%origin], %draft_origins[%origin], %origin_bytes
""")
    text = _sub(text, "  func.call @allocation(%allocations, %c0, %residual_bytes, %c0_i64) : (buffer, index, i64, i64)\n",
                "  func.call @allocation(%allocations, %c0, %feedback_bytes, %c0_i64) : (buffer, index, i64, i64)\n")
    text = _sub(text, "  func.return %allocations, %views, %origins, %empty, %geometry : buffer, buffer, buffer, buffer, buffer",
                "  func.return %allocations, %views, %origins, %draft_origins, %geometry : buffer, buffer, buffer, buffer, buffer")
    return text


# ---------------------------------------------------------------------------------------------------
def patch_control(text, cfg, draft):
    mrope = bool((cfg.get("vision") or {}).get("mrope"))
    source_step = 1 if mrope else 2   # text-only ports advance the opaque source position by 2 per input
    # sampling records move behind the larger control header
    n = text.count("index.constant 640 : offset")
    assert n == 3, n
    text = text.replace("index.constant 640 : offset", f"index.constant {SAMPLING} : offset")
    text = _sub(text, "func.def public @encode_epoch(", "func.def @plain_encode_epoch(")
    text = _sub(text, "func.def public @publish_epoch(", "func.def @plain_publish_epoch(")
    text = _sub(text, "func.def public @epoch(", "func.def @plain_epoch(")
    text = _sub(text, "func.decl import(\"runner\") @execute_6(", """func.decl import("runner") @execute_5(%stage: i32, %b0: hal.buffer, %b1: hal.buffer, %b2: hal.buffer, %b3: hal.buffer, %b4: hal.buffer) -> (i64)

func.decl import("runner") @execute_7(%stage: i32, %b0: hal.buffer, %b1: hal.buffer, %b2: hal.buffer, %b3: hal.buffer, %b4: hal.buffer, %b5: hal.buffer, %b6: hal.buffer) -> (i64)

func.decl import("runner") @execute_6(""")
    text += f"""
// ---- Speculative decoding (generated by draft/build_draft.py) ----
// Speculative spans reserve draft_depth + 1 packed inputs but the host supplies only the anchor; inputs
// therefore move to packed cursors (the sum of earlier span lengths), placeholders are zeroed, and the
// device plan copy carries the cursor in word 4, flag bit 4 (speculative: verify, do not publish
// recurrent state) and the output credit in bits 8..15. Spans are moved last to first, so a move never
// overwrites inputs not yet moved (cursors never precede their host offsets).
func.def public @encode_epoch(%control_state: buffer, %plan: buffer, %inputs: buffer, %media_plan: buffer, %first_count: i32, %next_count: i32, %input_count: i32, %first_media_count: i32, %next_media_count: i32, %first_packet_capacity: i32, %next_packet_capacity: i32, %eos: i32, %metadata: buffer, %tokens: buffer) -> (i32, buffer, i64, i64) {{
  %count, %state, %transfer_begin, %transfer_length = func.call @plain_encode_epoch(%control_state, %plan, %inputs, %media_plan, %first_count, %next_count, %input_count, %first_media_count, %next_media_count, %first_packet_capacity, %next_packet_capacity, %eos, %metadata, %tokens) : (buffer, buffer, buffer, buffer, i32, i32, i32, i32, i32, i32, i32, i32, buffer, buffer) -> (i32, buffer, i64, i64)
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c0_i32 = scalar.constant 0 : i32
  %c4_i32 = scalar.constant 4 : i32
  %c8_i32 = scalar.constant 8 : i32
  %header_bytes = index.constant 16 : offset
  %bounded_first = scalar.assume %first_count [range(%first_count, 1, 8192)] : i32
  %spans = index.cast %bounded_first : i32 to index
  %plan_extent = func.call @plan_capacity(%plan) : (buffer) -> (index)
  %requests = buffer.view %plan[%origin] : buffer -> view<[%plan_extent]x9xi32>
  %device = buffer.view %metadata[%header_bytes] : buffer -> view<[%spans]x9xi32>
  %media_total = scalar.addi %first_media_count, %next_media_count : i32
  %mixed = scalar.cmpi ne, %media_total, %c0_i32 : i32
  %one = index.constant 1 : index
  %eight = index.constant 8 : index
  %stride = scf.select %mixed, %eight, %one : index
  %token_bytes = buffer.length %tokens
  %token_words = index.cast %token_bytes : offset to index
  %four = index.constant 4 : index
  %words = index.div %token_words, %four : index
  %flat = buffer.view %tokens[%origin] : buffer -> view<[%words]xi32>
  // cursors, flags and the total packed length
  %total = scf.for %s = [%c0 to %spans step %c1](%cursor = %c0_i32 : i32) -> (i32) {{
    %sr = index.assume %s [lt(%s, %plan_extent)] : index
    %length = view.load %requests[%sr, 0] : view<[%plan_extent]x9xi32> -> i32
    %flags = view.load %requests[%sr, 6] : view<[%plan_extent]x9xi32> -> i32
    %credit = view.load %requests[%sr, 7] : view<[%plan_extent]x9xi32> -> i32
    %speculative = scalar.cmpi ne, %credit, %c0_i32 : i32
    %credit_bits = scalar.shli %credit, %c8_i32 : i32
    %spec_bits = scalar.ori %credit_bits, %c4_i32 : i32
    %marked = scalar.ori %flags, %spec_bits : i32
    %device_flags = scf.select %speculative, %marked, %flags : i32
    view.store %cursor, %device[%s, 4] : i32, view<[%spans]x9xi32>
    view.store %device_flags, %device[%s, 6] : i32, view<[%spans]x9xi32>
    %next = scalar.addi %cursor, %length : i32
    scf.yield %next : i32
  }}
  // move inputs to their cursors (last span first), zero placeholders
  scf.for %r = [%c0 to %spans step %c1] {{
    %last = index.sub %spans, %c1 : index
    %s = index.sub %last, %r : index
    %sr = index.assume %s [lt(%s, %plan_extent)] : index
    %sd = index.assume %s [lt(%s, %spans)] : index
    %length32 = view.load %requests[%sr, 0] : view<[%plan_extent]x9xi32> -> i32
    %begin32 = view.load %requests[%sr, 4] : view<[%plan_extent]x9xi32> -> i32
    %known32 = view.load %requests[%sr, 5] : view<[%plan_extent]x9xi32> -> i32
    %cursor32 = view.load %device[%sd, 4] : view<[%spans]x9xi32> -> i32
    %moves = scalar.cmpi ne, %begin32, %cursor32 : i32
    %length_b = scalar.assume %length32 [range(%length32, 0, 8192)] : i32
    %known_b = scalar.assume %known32 [range(%known32, 0, 8192)] : i32
    %begin_b = scalar.assume %begin32 [range(%begin32, 0, 8191)] : i32
    %cursor_b = scalar.assume %cursor32 [range(%cursor32, 0, 8191)] : i32
    %length = index.cast %length_b : i32 to index
    %known = index.cast %known_b : i32 to index
    %begin = index.cast %begin_b : i32 to index
    %cursor = index.cast %cursor_b : i32 to index
    scf.if %moves {{
      scf.for %q = [%c0 to %known step %c1] {{
        %kl = index.sub %known, %c1 : index
        %j = index.sub %kl, %q : index
        %src_t = index.add %begin, %j : index
        %dst_t = index.add %cursor, %j : index
        %src0 = index.mul %src_t, %stride : index
        %dst0 = index.mul %dst_t, %stride : index
        scf.for %w = [%c0 to %stride step %c1] {{
          %si = index.add %src0, %w : index
          %di = index.add %dst0, %w : index
          %sb = index.assume %si [lt(%si, %words)] : index
          %db = index.assume %di [lt(%di, %words)] : index
          %v = view.load %flat[%sb] : view<[%words]xi32> -> i32
          view.store %v, %flat[%db] : i32, view<[%words]xi32>
        }}
      }}
    }}
    // placeholders after the known inputs (speculative spans): zero ids; media records keep text
    // positions continuing the anchor's (t = h = w)
    scf.for %j = [%known to %length step %c1] {{
      %dst_t = index.add %cursor, %j : index
      %dst0 = index.mul %dst_t, %stride : index
      scf.for %w = [%c0 to %stride step %c1] {{
        %di = index.add %dst0, %w : index
        %db = index.assume %di [lt(%di, %words)] : index
        view.store %c0_i32, %flat[%db] : i32, view<[%words]xi32>
      }}
      scf.if %mixed {{
        %anchor0 = index.mul %cursor, %stride : index
        %anchor4 = index.add %anchor0, %four : index
        %ab = index.assume %anchor4 [lt(%anchor4, %words)] : index
        %t0 = view.load %flat[%ab] : view<[%words]xi32> -> i32
        %j32 = index.cast %j : index to i32
        %tj = scalar.addi %t0, %j32 : i32
        %five = index.constant 5 : index
        %six = index.constant 6 : index
        %p4 = index.add %dst0, %four : index
        %p5 = index.add %dst0, %five : index
        %p6 = index.add %dst0, %six : index
        %b4 = index.assume %p4 [lt(%p4, %words)] : index
        %b5 = index.assume %p5 [lt(%p5, %words)] : index
        %b6 = index.assume %p6 [lt(%p6, %words)] : index
        view.store %tj, %flat[%b4] : i32, view<[%words]xi32>
        view.store %tj, %flat[%b5] : i32, view<[%words]xi32>
        view.store %tj, %flat[%b6] : i32, view<[%words]xi32>
      }}
    }}
  }}
  // the input transfer covers every packed slot
  %total64 = scalar.extui %total : i32 to i64
  %stride_index = index.cast %stride : index to i64
  %word64 = scalar.constant 4 : i64
  %record_bytes = scalar.muli %stride_index, %word64 : i64
  %input_length = scalar.muli %total64, %record_bytes : i64
  %transfer_offset = index.constant 8 : offset
  %transfers = buffer.view %state[%transfer_offset] : buffer -> view<2x5xi64>
  view.store %input_length, %transfers[1, 4] : i64, view<2x5xi64>
  func.return %count, %state, %transfer_begin, %transfer_length : i32, buffer, i64, i64
}}

// Verified epochs: semantic progress comes from device acceptance (records of 10 i32:
// {{consumed, output_count, token[8]}}); a speculative span's known count is zero.
func.def public @publish_epoch(%plan: buffer, %media_plan: buffer, %count: i32, %first_media_count: i32, %next_media_count: i32, %verified: i32, %continues: i32, %next_speculative: buffer, %ordinary: buffer, %first_results: buffer, %next_results: buffer, %progress: buffer) {{
  func.call @plain_publish_epoch(%plan, %media_plan, %count, %first_media_count, %next_media_count, %verified, %continues, %next_speculative, %ordinary, %first_results, %next_results, %progress) : (buffer, buffer, i32, i32, i32, i32, i32, buffer, buffer, buffer, buffer, buffer)
  %c0_i32 = scalar.constant 0 : i32
  %verifies = scalar.cmpi ne, %verified, %c0_i32 : i32
  scf.if %verifies {{
    %origin = index.constant 0 : offset
    %c0 = index.constant 0 : index
    %c1 = index.constant 1 : index
    %c1_i32 = scalar.constant 1 : i32
    %step = scalar.constant {source_step} : i32
    %bounded_count = scalar.assume %count [range(%count, 1, 2147483647)] : i32
    %length = index.cast %bounded_count : i32 to index
    %plan_extent = func.call @plan_capacity(%plan) : (buffer) -> (index)
    %requests = buffer.view %plan[%origin] : buffer -> view<[%plan_extent]x9xi32>
    %records = buffer.view %first_results[%origin] : buffer -> view<[%length]x10xi32>
    %published = buffer.view %progress[%origin] : buffer -> view<[%length]x21xi32>
    scf.for %i = [%c0 to %length step %c1] {{
      %ir = index.assume %i [lt(%i, %plan_extent)] : index
      %consumed = view.load %records[%i, 0] : view<[%length]x10xi32> -> i32
      %outputs = view.load %records[%i, 1] : view<[%length]x10xi32> -> i32
      %credit = view.load %requests[%ir, 7] : view<[%plan_extent]x9xi32> -> i32
      %speculative = scalar.cmpi ne, %credit, %c0_i32 : i32
      %known = scf.select %speculative, %c0_i32, %consumed : i32
      %verification = scalar.extui %speculative : i1 to i32
      view.store %consumed, %published[%i, 0] : i32, view<[%length]x21xi32>
      view.store %known, %published[%i, 1] : i32, view<[%length]x21xi32>
      view.store %outputs, %published[%i, 2] : i32, view<[%length]x21xi32>
      view.store %verification, %published[%i, 3] : i32, view<[%length]x21xi32>
      scf.if %speculative {{
        %source = view.load %requests[%ir, 2] : view<[%plan_extent]x9xi32> -> i32
        %advance = scalar.muli %consumed, %step : i32
        %next_source = scalar.addi %source, %advance : i32
        view.store %next_source, %published[%i, 4] : i32, view<[%length]x21xi32>
      }}
      %bounded_outputs = scalar.assume %outputs [range(%outputs, 0, 8)] : i32
      %output_extent = index.cast %bounded_outputs : i32 to index
      %c2 = index.constant 2 : index
      %c5 = index.constant 5 : index
      scf.for %j = [%c0 to %output_extent step %c1] {{
        %src = index.add %j, %c2 : index
        %dst = index.add %j, %c5 : index
        %src_b = index.assume %src [range(%src, 2, 9)] : index
        %dst_b = index.assume %dst [range(%dst, 5, 12)] : index
        %token = view.load %records[%i, %src_b] : view<[%length]x10xi32> -> i32
        view.store %token, %published[%i, %dst_b] : i32, view<[%length]x21xi32>
      }}
    }}
  }}
  func.return
}}

// propose (when any span proposes) -> verify or advance -> drafter catch-up. Stage tables: advance at
// 128, verify at {VERIFY_TABLE}, warm at {WARM_TABLE}; first proposal stage, level count and draft kind at {PROPOSE_FIRST}.
func.def public @epoch(%control_state: buffer, %shape: i32, %proposes: i32, %verifies: i32, %span_count: i32, %output_count: i32, %epoch_count: i32, %continuation_shape: i32, %continuation_proposes: i32, %first_media_count: i32, %next_media_count: i32, %residual: hal.buffer, %metadata: hal.buffer, %origins: hal.buffer, %state: hal.buffer, %tokens: hal.buffer, %outputs: hal.buffer, %carry: hal.buffer, %committed: hal.buffer, %results: hal.buffer, %draft_cache: hal.buffer, %draft_origins: hal.buffer, %next_results: hal.buffer) -> (i64) {{
  %c0 = scalar.constant 0 : i32
  %c1 = scalar.constant 1 : i32
  %c0_i64 = scalar.constant 0 : i64
  %bounded_shape = scalar.assume %shape [range(%shape, 0, 127)] : i32
  %shape_index = index.cast %bounded_shape : i32 to index
  %advance_at = index.constant 128 : offset
  %verify_at = index.constant {VERIFY_TABLE} : offset
  %warm_at = index.constant {WARM_TABLE} : offset
  %draft_at = index.constant {PROPOSE_FIRST} : offset
  %advance_view = buffer.view %control_state[%advance_at] : buffer -> view<128xi32>
  %verify_view = buffer.view %control_state[%verify_at] : buffer -> view<128xi32>
  %warm_view = buffer.view %control_state[%warm_at] : buffer -> view<128xi32>
  %draft_view = buffer.view %control_state[%draft_at] : buffer -> view<3xi32>
  %propose_first = view.load %draft_view[0] : view<3xi32> -> i32
  %levels = view.load %draft_view[1] : view<3xi32> -> i32
  %kind = view.load %draft_view[2] : view<3xi32> -> i32
  %drafting = scalar.cmpi ne, %kind, %c0 : i32
  %propose = scalar.cmpi ne, %proposes, %c0 : i32
  %verify = scalar.cmpi ne, %verifies, %c0 : i32
  scf.if %propose {{
    %bits = scalar.constant 32 : i32
    %below = scalar.subi %span_count, %c1 : i32
    %leading = scalar.ctlzi %below : i32
    %level = scalar.subi %bits, %leading : i32
    %last = scalar.subi %levels, %c1 : i32
    %chosen = scalar.minui %level, %last : i32
    %pstage = scalar.addi %propose_first, %chosen : i32
    %proposal = func.call @execute_5(%pstage, %metadata, %tokens, %draft_origins, %draft_cache, %carry) : (i32, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer) -> (i64)
  }}
  %target = scf.if %verify -> (i64) {{
    %vstage = view.load %verify_view[%shape_index] : view<128xi32> -> i32
    %completion = func.call @execute_7(%vstage, %residual, %metadata, %origins, %state, %tokens, %committed, %results) : (i32, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer) -> (i64)
    %count64 = scalar.extui %span_count : i32 to i64
    %record = scalar.constant 40 : i64
    %length = scalar.muli %count64, %record : i64
    %feedback = func.call @feedback(%c1, %results, %c0_i64, %length) : (i32, hal.buffer, i64, i64) -> (i64)
    scf.yield %completion : i64
  }} else {{
    %astage = view.load %advance_view[%shape_index] : view<128xi32> -> i32
    %completion = func.call @execute_6(%astage, %residual, %metadata, %origins, %state, %tokens, %outputs) : (i32, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer) -> (i64)
    %count64 = scalar.extui %span_count : i32 to i64
    %word = scalar.constant 4 : i64
    %length = scalar.muli %count64, %word : i64
    %feedback = func.call @feedback(%c0, %outputs, %c0_i64, %length) : (i32, hal.buffer, i64, i64) -> (i64)
    scf.yield %completion : i64
  }}
  %done = scf.if %drafting -> (i64) {{
    %accepted = scf.if %verify -> (hal.buffer) {{
      scf.yield %committed : hal.buffer
    }} else {{
      scf.yield %metadata : hal.buffer
    }}
    %wstage = view.load %warm_view[%shape_index] : view<128xi32> -> i32
    %warm = func.call @execute_6(%wstage, %residual, %accepted, %draft_origins, %draft_cache, %carry, %tokens) : (i32, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer) -> (i64)
    scf.yield %warm : i64
  }} else {{
    scf.yield %target : i64
  }}
  func.return %done : i64
}}
"""
    return text


def card_execution(execution, draft):
    """model.json execution settings for the selected drafter (continuation stays at one epoch)."""
    execution = dict(execution)
    execution["draft_depth"] = int(draft.get("depth", 3))
    return execution
