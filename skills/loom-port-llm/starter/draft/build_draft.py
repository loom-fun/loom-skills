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
import os
import re


def hx(s):
    return s.encode().hex()


SAMPLING = 2048          # control-state byte offset of the per-row sampling records
VERIFY_TABLE, WARM_TABLE = 640, 1152
PROPOSE_FIRST, PROPOSE_LEVELS, DRAFT_KIND = 1664, 1668, 1672
# Occupancy gate (SPEC_OCC_MIN=N, default 2; 0 = off): an epoch whose spans are ALL speculative and number >= N verifies only
# the anchors (1 token per span, the plain decode shape) instead of depth+1 tokens per span. encode_epoch decides and leaves
# the shape index + 1 at GATE_FLAG; GATE_TABLE[s] holds the index + 1 of the packing shape with s tokens in s spans.
GATE_FLAG, GATE_TABLE, GATE_SLOTS = 1676, 1700, 32
# DFlash gated epochs run @llm_gate (the plain advance trunk + g_gate_results) with no drafter catch-up; GATE_STAGES[s] is its
# stage for s spans (gen_draft.gate_advance; SPEC_GATE_VERIFY=1 keeps the one-token verify)
GATE_STAGES = GATE_TABLE + 4 * GATE_SLOTS


def gate_advance(kind):
    return kind == "dflash" and os.environ.get("SPEC_GATE_VERIFY", "0") != "1"
OCC_MIN, CTX_MAX = 2, 0


def set_gates(cfg):
    """The two gates, each an env switch (SPEC_OCC_MIN, SPEC_CTX_MAX) that wins over cfg["drafter"] ("occ_min", "ctx_max",
    written by tune.py) and defaults to occupancy 2 / no context limit. 0 turns a gate off. Occupancy: epochs with >= occ_min
    streams verify anchors only. Context: an epoch with any span at causal position >= ctx_max does too (long-context
    verification costs depth + 1 attention passes per span, which low acceptance does not repay)."""
    global OCC_MIN, CTX_MAX
    d = cfg.get("drafter") or {}
    OCC_MIN = int(os.environ["SPEC_OCC_MIN"] or 0) if "SPEC_OCC_MIN" in os.environ else int(d.get("occ_min", 2) or 0)
    CTX_MAX = int(os.environ["SPEC_CTX_MAX"] or 0) if "SPEC_CTX_MAX" in os.environ else int(d.get("ctx_max", 0) or 0)


def _sub(text, old, new, count=1):
    assert text.count(old) >= 1, f"draft patch anchor not found: {old[:80]!r}"
    return text.replace(old, new, count)


# ---------------------------------------------------------------------------------------------------
def root_sources(cfg, draft):
    """Parameter-root sources of the (propose, warm) stages: "model" (the target checkpoint) or "draft" (a
    separate drafter checkpoint). Tensors the target already places (embedding, head) get roots of their own
    so they share its storage; a root either matches an existing placement exactly or is entirely new.
    MTP: propose = [embedding, (head), NextN block]; warm = [embedding, one root per block tensor it reads].
    DFlash: propose = [embedding, (head), drafter, drafter context projections]; warm = [fc + norm, context]."""
    head = [] if cfg.get("tied_embeddings") else ["model"]
    if draft["kind"] == "mtp":
        return ["model"] + head + ["model"], ["model"] * (1 + 6 + (2 if cfg.get("qk_norm") else 0))
    if draft["kind"] == "ngram":   # prompt lookup reads no weights
        return [], []
    if draft["kind"] == "assist":  # propose = [target embedding, assistant checkpoint]; warm reads no weights
        return ["model", "draft"], []
    # "phead": the target's head once more, through its load-time Q4_0 policy (head_weights.loom, prep_weights.py)
    # With "phead" the proposals never read the head root, so an untied head is not bound (gen_dflash.py: same rule).
    if draft.get("head_policy"):
        return ["model", "draft", "draft", "phead"], ["draft", "draft"]
    return ["model"] + head + ["draft", "draft"], ["draft", "draft"]


def _bind(stage, sources, draft_policy=False):
    """Bind each parameter root; the drafter's roots use its own weight policy (draft_weights.loom) when it has one,
    the proposals' head root ("phead") the target checkpoint through head_weights.loom."""
    pol = lambda src: "%head_policy" if src == "phead" else "%draft_policy" if src != "model" and draft_policy else "%policy"
    return "".join(f"      %{stage}_b{i} = scalar.constant {i} : i32\n"
                   f"      func.call @parameter(%{stage}, %{stage}_b{i}, %c0_source, {'%checkpoint' if src in ('model', 'phead') else '%draft_source'}, {pol(src)}) : (i32, i32, i32, i32, buffer)\n"
                   for i, src in enumerate(sources))


def patch_prepare(text, draft, cfg):
    set_gates(cfg)
    gate_fill = ""
    gate_stage = ""
    if (OCC_MIN or CTX_MAX) and gate_advance(draft["kind"]):
        gate_stage = f"""        %gate_root = global.load @gate_root : buffer
        %gstage = func.call @stage(%gate_root, %tag) : (buffer, i64) -> (i32)
        func.call @config_i64(%gstage, %context_key, %context) : (i32, buffer, i64)
        func.call @config_i64(%gstage, %pool_key, %pool) : (i32, buffer, i64)
        func.call @config_i64(%gstage, %tokens_key, %tokens) : (i32, buffer, i64)
        func.call @config_i64(%gstage, %spans_key, %spans) : (i32, buffer, i64)
        func.call @parameter(%gstage, %c0_i32, %c0_source, %checkpoint, %policy) : (i32, i32, i32, i32, buffer)
        %gs_at = index.constant {GATE_STAGES} : offset
        %gs_view = buffer.view %state[%gs_at] : buffer -> view<{GATE_SLOTS}xi32>
        %gs_slot = index.cast %spans : i64 to index
        view.store %gstage, %gs_view[%gs_slot] : i32, view<{GATE_SLOTS}xi32>
"""
    if OCC_MIN or CTX_MAX:   # shape index + 1 of the (s tokens, s spans) packing, for the occupancy gate
        gate_fill = f"""      %same_shape = scalar.cmpi eq, %tokens, %spans : i64
      %gate_slots = scalar.constant {GATE_SLOTS} : i64
      %few_spans = scalar.cmpi ult, %spans, %gate_slots : i64
      %gate_ok = scalar.andi %same_shape, %few_spans : i1
      scf.if %gate_ok {{
{gate_stage}        %gate_at = index.constant {GATE_TABLE} : offset
        %gate_view = buffer.view %state[%gate_at] : buffer -> view<{GATE_SLOTS}xi32>
        %gate_slot = index.cast %spans : i64 to index
        %gate_i32 = index.cast %i : index to i32
        %gate_one = scalar.constant 1 : i32
        %gate_tag = scalar.addi %gate_i32, %gate_one : i32
        view.store %gate_tag, %gate_view[%gate_slot] : i32, view<{GATE_SLOTS}xi32>
      }}
"""
    kind = draft["kind"]
    psrc, wsrc = root_sources(cfg, draft)
    warm_root = {"mtp": "llm_mtp_warm", "dflash": "llm_dflash_warm", "ngram": "llm_ngram_warm", "assist": "llm_as_warm"}[kind]
    propose_root = {"mtp": "llm_mtp_propose", "dflash": "llm_dflash_propose", "ngram": "llm_ngram_propose",
                    "assist": "llm_as_propose"}[kind]
    block = draft.get("block_size", 1) if kind == "dflash" else 1
    text = _sub(text, '''// "weights.loom"''', f'''// "llm_verify", "{warm_root}", "{propose_root}", "llm.width", "llm.epoch_spans", "llm.epoch_tokens"
global.rodata.def @verify_root = bytes("{hx("llm_verify")}")
{f'{chr(10)}// "llm_gate"{chr(10)}global.rodata.def @gate_root = bytes("{hx("llm_gate")}"){chr(10)}' if (OCC_MIN or CTX_MAX) and gate_advance(kind) else ""}

global.rodata.def @warm_root = bytes("{hx(warm_root)}")

global.rodata.def @propose_root = bytes("{hx(propose_root)}")

global.rodata.def @width_key = bytes("{hx("llm.width")}")

global.rodata.def @epoch_spans_key = bytes("{hx("llm.epoch_spans")}")

global.rodata.def @epoch_tokens_key = bytes("{hx("llm.epoch_tokens")}")

// "draft"
global.rodata.def @draft_scope = bytes("{hx("draft")}")
{f'{chr(10)}// "{draft["policy"]}"{chr(10)}global.rodata.def @draft_policy_name = bytes("{hx(draft["policy"])}"){chr(10)}' if draft.get("policy") else ""}{f'{chr(10)}// "{draft["head_policy"]}"{chr(10)}global.rodata.def @head_policy_name = bytes("{hx(draft["head_policy"])}"){chr(10)}' if draft.get("head_policy") else ""}
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
    draft_source = ("  // the drafter checkpoint (card parameter scope \"draft\")\n  %draft_scope = global.load @draft_scope : buffer\n"
                    "  %draft_source = func.call @source_scope(%draft_scope) : (buffer) -> (i32)\n"
                    + ("  %draft_policy = global.load @draft_policy_name : buffer\n" if draft.get("policy") else "")
                    + ("  %head_policy = global.load @head_policy_name : buffer\n" if draft.get("head_policy") else "")) if "draft" in psrc + wsrc else ""
    text = _sub(text, "  %smaller = scalar.cmpi ult, %maximum, %prefill : i64\n", f"""  // Verify and catch-up stages per shape, declared after every advance stage (the first shape_count
  // stages are the shared target stages), then proposal stages for cohorts of 1, 2, 4, ... spans.
{draft_source}  scf.if %drafting {{
    %propose_root = global.load @propose_root : buffer
    %epoch_spans_key = global.load @epoch_spans_key : buffer
    %epoch_tokens_key = global.load @epoch_tokens_key : buffer
    %span32 = scalar.trunci %span_capacity : i64 to i32
    %one32 = scalar.constant 1 : i32
    %bits32 = scalar.constant 32 : i32
    %below = scalar.subi %span32, %one32 : i32
    %leading = scalar.ctlzi %below : i32
    %last_level = scalar.subi %bits32, %leading : i32
    %levels_all = scalar.addi %last_level, %one32 : i32
    %level_cap = scalar.constant {(8192 // block).bit_length()} : i32
    %levels32 = scalar.minui %levels_all, %level_cap : i32
    %levels_b = scalar.assume %levels32 [range(%levels32, 1, 14)] : i32
    %levels = index.cast %levels_b : i32 to index
    %c1_64 = scalar.constant 1 : i64
    %first_propose = scf.for %level = [%c0 to %levels step %c1](%first = %c0_i32 : i32) -> (i32) {{
      %shift = index.cast %level : index to i64
      %cap = scalar.shli %c1_64, %shift : i64
      %pstage = func.call @stage(%propose_root, %shift) : (buffer, i64) -> (i32)
      func.call @config_i64(%pstage, %context_key, %context) : (i32, buffer, i64)
      func.call @config_i64(%pstage, %pool_key, %pool) : (i32, buffer, i64)
      // A DFlash block is [anchor, <mask> x depth] (llama.cpp's n_max + 1 tokens), capped at the trained block size:
      // 4 streams at depth 3 propose 16 tokens (one int8 matvec pass per drafter weight) instead of 64 (GEMM).
      %block_full = scalar.constant {block} : i64
      %block_tokens = scalar.minui %block_full, %width : i64
      %cap_tokens = scalar.muli %cap, %block_tokens : i64
      func.call @config_i64(%pstage, %tokens_key, %cap_tokens) : (i32, buffer, i64)
      func.call @config_i64(%pstage, %spans_key, %cap) : (i32, buffer, i64)
      func.call @config_i64(%pstage, %width_key, %width) : (i32, buffer, i64)
      func.call @config_i64(%pstage, %epoch_spans_key, %span_capacity) : (i32, buffer, i64)
      func.call @config_i64(%pstage, %epoch_tokens_key, %maximum) : (i32, buffer, i64)
{_bind("pstage", psrc, bool(draft.get("policy")))}
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
{_bind("wstage", wsrc, bool(draft.get("policy")))}
      view.store %wstage, %warm_view[%i] : i32, view<[%count]xi32>
{gate_fill}    }}
    %draft_at = index.constant {PROPOSE_FIRST} : offset
    %draft_view = buffer.view %state[%draft_at] : buffer -> view<3xi32>
    view.store %first_propose, %draft_view[0] : i32, view<3xi32>
    view.store %levels32, %draft_view[1] : i32, view<3xi32>
    %kind32 = scalar.constant {dict(mtp=1, dflash=2, ngram=3, assist=4)[kind]} : i32
    view.store %kind32, %draft_view[2] : i32, view<3xi32>
  }}
  %smaller = scalar.cmpi ult, %maximum, %prefill : i64
""")
    decl = "func.decl @llm_storage(%maximum: i64, %span_capacity: i64, %context: i64, %pool: i64, %rows: i64, %pins: i64) -> (buffer, buffer, buffer, buffer, buffer)"
    text = _sub(text, decl, decl.replace("%pins: i64)", "%pins: i64, %drafting: i64)"))
    text = _sub(text, "  %allocations, %views, %origins, %draft, %geometry = func.call @llm_storage(%maximum, %span_capacity, %context, %pool, %rows, %pins) : (i64, i64, i64, i64, i64, i64) -> (buffer, buffer, buffer, buffer, buffer)\n  %terminal = global.load @terminal : buffer\n  func.return %effective, %stage_count32, %state, %terminal, %allocations, %views, %origins, %empty, %geometry",
                "  %drafting64 = scalar.extui %drafting : i1 to i64\n  %allocations, %views, %origins, %draft, %geometry = func.call @llm_storage(%maximum, %span_capacity, %context, %pool, %rows, %pins, %drafting64) : (i64, i64, i64, i64, i64, i64, i64) -> (buffer, buffer, buffer, buffer, buffer)\n  %terminal = global.load @terminal : buffer\n  func.return %effective, %stage_count32, %state, %terminal, %allocations, %views, %origins, %draft, %geometry")
    return text


# ---------------------------------------------------------------------------------------------------
def patch_storage(text, cfg, draft):
    """Draft allocations 6..10, the draft cache region and the draft page table."""
    H = cfg["hidden"]
    if draft["kind"] == "mtp":
        pos_bytes, planes, rows_per_token, carry_stride = cfg["kv_heads"] * cfg["head_dim"] * 2, 2 * len(cfg["mtp"]["layers"]), 1, 4 * H
    elif draft["kind"] == "ngram":   # token history: one i32 per position, no carry
        pos_bytes, planes, rows_per_token, carry_stride = 4, 1, 1, 0
    elif draft["kind"] == "assist":  # the target's hidden per row (carry); a 4-byte placeholder draft plane
        pos_bytes, planes, rows_per_token, carry_stride = 4, 1, 1, 4 * H
    else:   # DFlash: features of every tap per token, no carry, two planes per draft layer
        pos_bytes, planes, rows_per_token, carry_stride = draft["kv_heads"] * draft["head_dim"] * 2, 2 * draft["layers"], draft["taps"], 0
    sig_old = "func.def public @llm_storage(%maximum: i64, %span_capacity: i64, %context: i64, %pool: i64, %rows: i64, %pins: i64) -> (buffer, buffer, buffer, buffer, buffer) {\n"
    text = _sub(text, sig_old, sig_old.replace("%pins: i64)", "%pins: i64, %drafting: i64)") +
                "  // drafting = 0 (draft_depth 0 at serve time): no drafter allocations, one cache region, no draft table.\n"
                "  %zero_flag = scalar.constant 0 : i64\n  %on = scalar.cmpi ne, %drafting, %zero_flag : i64\n")
    text = _sub(text, "  %residual_bytes = scalar.muli %span_capacity, %c4_i64 : i64\n",
                f"""  // Final normalized hidden rows (the drafter's catch-up input) live in the residual allocation.
  %hidden_row = scalar.constant {4 * H * rows_per_token} : i64
  %carry_stride = scalar.constant {carry_stride} : i64
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
  %carry_bytes = scalar.muli %owners, %carry_stride : i64
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
  %carry_on = scf.select %on, %carry_bytes, %c0_i64 : i64
  %committed_on = scf.select %on, %committed_bytes, %c0_i64 : i64
  %results_on = scf.select %on, %results_bytes, %c0_i64 : i64
  %draft_on = scf.select %on, %draft_bytes, %c0_i64 : i64
  %table_on = scf.select %on, %table_bytes, %c0_i64 : i64
  %carry_stride_on = scf.select %on, %carry_stride, %c0_i64 : i64
  %split_on = scf.select %on, %feedback_split, %c4_i64 : i64
  func.call @allocation(%allocations, %c6, %carry_on, %carry_on) : (buffer, index, i64, i64)
  func.call @allocation(%allocations, %c7, %committed_on, %c0_i64) : (buffer, index, i64, i64)
  func.call @allocation(%allocations, %c8, %results_on, %c0_i64) : (buffer, index, i64, i64)
  func.call @allocation(%allocations, %c9, %draft_on, %c0_i64) : (buffer, index, i64, i64)
  func.call @allocation(%allocations, %c10, %table_on, %c0_i64) : (buffer, index, i64, i64)
""")
    text = _sub(text, "  %geometry_bytes = index.constant 104 : offset\n",
                "  %geometry_one = index.constant 104 : offset\n  %geometry_two = index.constant 144 : offset\n"
                "  %geometry_bytes = scf.select %on, %geometry_two, %geometry_one : offset\n")
    text = _sub(text, "  view.store %c0_i64, %geometry_view[2] : i64, view<13xi64>\n  view.store %c4_i64, %geometry_view[3] : i64, view<13xi64>\n",
                "  view.store %carry_stride_on, %geometry_view[2] : i64, view<13xi64>\n  view.store %split_on, %geometry_view[3] : i64, view<13xi64>\n")
    text = _sub(text, "  view.store %block_bytes, %geometry_view[12] : i64, view<13xi64>\n",
                f"""  view.store %block_bytes, %geometry_view[12] : i64, view<13xi64>
  // second cache region: the drafter's K/V planes (allocation 9), same page geometry
  %draft_origins = scf.if %on -> (buffer) {{
    %region_at = index.constant 104 : offset
    %region = buffer.view %geometry[%region_at] : buffer -> view<5xi64>
    %c9_i64 = scalar.constant 9 : i64
    %draft_block_bytes = scalar.constant {16 * pos_bytes} : i64
    view.store %c9_i64, %region[0] : i64, view<5xi64>
    view.store %c0_i64, %region[1] : i64, view<5xi64>
    view.store %draft_planes, %region[2] : i64, view<5xi64>
    view.store %draft_plane_bytes, %region[3] : i64, view<5xi64>
    view.store %draft_block_bytes, %region[4] : i64, view<5xi64>
    // the draft page table starts as a copy of the target's (page maps at the same offsets)
    %copy = buffer.alloca<private> align(8) %origin_bytes : buffer
    buffer.copy %origins[%origin], %copy[%origin], %origin_bytes
    scf.yield %copy : buffer
  }} else {{
    %none_bytes = index.constant 0 : offset
    %none = buffer.alloca<private> align(1) %none_bytes : buffer
    scf.yield %none : buffer
  }}
""")
    text = _sub(text, "  func.call @allocation(%allocations, %c0, %residual_bytes, %c0_i64) : (buffer, index, i64, i64)\n",
                "  func.call @allocation(%allocations, %c0, %feedback_bytes, %c0_i64) : (buffer, index, i64, i64)\n")
    text = _sub(text, "  func.return %allocations, %views, %origins, %empty, %geometry : buffer, buffer, buffer, buffer, buffer",
                "  func.return %allocations, %views, %origins, %draft_origins, %geometry : buffer, buffer, buffer, buffer, buffer")
    return text


# ---------------------------------------------------------------------------------------------------
def patch_control(text, cfg, draft):
    set_gates(cfg)
    gate_pre = gate_epoch = ""
    L1 = "    %length = view.load %requests[%sr, 0] : view<[%plan_extent]x9xi32> -> i32\n"
    L2 = "    %length32 = view.load %requests[%sr, 0] : view<[%plan_extent]x9xi32> -> i32\n"
    propose_flag, eff = "propose", "shape_index"
    if OCC_MIN or CTX_MAX:
        occupied_ir = (f"  %occ_below = scalar.constant {OCC_MIN - 1} : i32\n  %occupied = scalar.cmpi ugt, %first_count, %occ_below : i32\n"
                       if OCC_MIN else "  %occupied = scalar.cmpi ne, %c0_i32, %c0_i32 : i32\n")
        ctx_ir = (f"""  %ctx_max = scalar.constant {CTX_MAX} : i32
  %deepest = scf.for %g = [%c0 to %spans step %c1](%acc = %c0_i32 : i32) -> (i32) {{
    %gr = index.assume %g [lt(%g, %plan_extent)] : index
    %gpos = view.load %requests[%gr, 1] : view<[%plan_extent]x9xi32> -> i32
    %gbig = scalar.cmpi ugt, %gpos, %acc : i32
    %gmax = scf.select %gbig, %gpos, %acc : i32
    scf.yield %gmax : i32
  }}
  %ctx_over = scalar.cmpi uge, %deepest, %ctx_max : i32
""" if CTX_MAX else "  %ctx_over = scalar.cmpi ne, %c0_i32, %c0_i32 : i32\n")
        # a context gate must look at every epoch; the occupancy gate alone cannot fire below occ_min spans, so a
        # single stream skips the whole pre-pass (the 1-row path stays free)
        try_ir = (f"  %try_below = scalar.constant {OCC_MIN - 1} : i32\n  %try = scalar.cmpi ugt, %first_count, %try_below : i32\n"
                  if OCC_MIN and not CTX_MAX else "  %try = scalar.cmpi eq, %c0_i32, %c0_i32 : i32\n")
        gate_pre = f"""  // occupancy / context gate: all spans speculative and at least {OCC_MIN} of them -> verify the anchors only
  %c1_i32 = scalar.constant 1 : i32
{try_ir}  %flag_value = scf.if %try -> (i32) {{
  %nspec = scf.for %g = [%c0 to %spans step %c1](%acc = %c0_i32 : i32) -> (i32) {{
    %gr = index.assume %g [lt(%g, %plan_extent)] : index
    %gcredit = view.load %requests[%gr, 7] : view<[%plan_extent]x9xi32> -> i32
    %gspec = scalar.cmpi ne, %gcredit, %c0_i32 : i32
    %gadd = scalar.extui %gspec : i1 to i32
    %gnext = scalar.addi %acc, %gadd : i32
    scf.yield %gnext : i32
  }}
  %all_spec = scalar.cmpi eq, %nspec, %first_count : i32
{occupied_ir}  %gate_limit = scalar.constant {GATE_SLOTS} : i32
  %few_spans = scalar.cmpi ult, %first_count, %gate_limit : i32
  %gate_max = scalar.constant {GATE_SLOTS - 1} : i32
  %gate_slot32 = scalar.minui %first_count, %gate_max : i32
  %gate_slot_b = scalar.assume %gate_slot32 [range(%gate_slot32, 0, {GATE_SLOTS - 1})] : i32
  %gate_slot = index.cast %gate_slot_b : i32 to index
  %gate_at = index.constant {GATE_TABLE} : offset
  %gate_view = buffer.view %control_state[%gate_at] : buffer -> view<{GATE_SLOTS}xi32>
  %gate_raw = view.load %gate_view[%gate_slot] : view<{GATE_SLOTS}xi32> -> i32
  %gate_shape = scf.select %few_spans, %gate_raw, %c0_i32 : i32
  %has_shape = scalar.cmpi ne, %gate_shape, %c0_i32 : i32
{ctx_ir}  %g0 = scalar.ori %occupied, %ctx_over : i1
  %g1 = scalar.andi %all_spec, %g0 : i1
  %gate_in = scalar.andi %g1, %has_shape : i1
  %flag_sel = scf.select %gate_in, %gate_shape, %c0_i32 : i32
  scf.yield %flag_sel : i32
  }} else {{
  scf.yield %c0_i32 : i32
  }}
  %gate = scalar.cmpi ne, %flag_value, %c0_i32 : i32
  %flag_at = index.constant {GATE_FLAG} : offset
  %flag_view = buffer.view %control_state[%flag_at] : buffer -> view<1xi32>
  view.store %flag_value, %flag_view[0] : i32, view<1xi32>
"""
        L1 = ("    %length_host = view.load %requests[%sr, 0] : view<[%plan_extent]x9xi32> -> i32\n"
              "    %length = scf.select %gate, %c1_i32, %length_host : i32\n"
              "    scf.if %gate {\n      view.store %length, %device[%s, 0] : i32, view<[%spans]x9xi32>\n    }\n")
        L2 = ("    %length32_host = view.load %requests[%sr, 0] : view<[%plan_extent]x9xi32> -> i32\n"
              "    %length32 = scf.select %gate, %c1_i32, %length32_host : i32\n")
        propose_flag, eff = "propose_on", "eff_index"
        gate_epoch = f"""  %flag_at = index.constant {GATE_FLAG} : offset
  %flag_view = buffer.view %control_state[%flag_at] : buffer -> view<1xi32>
  %flag = view.load %flag_view[0] : view<1xi32> -> i32
  %gated = scalar.cmpi ne, %flag, %c0 : i32
  %flag_nz = scf.select %gated, %flag, %c1 : i32
  %flag_idx = scalar.subi %flag_nz, %c1 : i32
  %flag_b = scalar.assume %flag_idx [range(%flag_idx, 0, 127)] : i32
  %flag_shape = index.cast %flag_b : i32 to index
  %eff_index = scf.select %gated, %flag_shape, %shape_index : index
  %ungated = scalar.cmpi eq, %flag, %c0 : i32
  %propose_on = scalar.andi %propose, %ungated : i1
"""
    exe7 = "(i32, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer) -> (i64)"
    verify_run = f"""    %vstage = view.load %verify_view[%{eff}] : view<128xi32> -> i32
    %VAR = func.call @execute_7(%vstage, %residual, %metadata, %origins, %state, %tokens, %committed, %results) : {exe7}
"""
    feedback_ir = """    %count64 = scalar.extui %span_count : i32 to i64
    %record = scalar.constant 40 : i64
    %length = scalar.muli %count64, %record : i64
    %feedback = func.call @feedback(%c1, %results, %c0_i64, %length) : (i32, hal.buffer, i64, i64) -> (i64)
"""
    flags_sel = "    %device_flags = scf.select %speculative, %marked, %flags : i32\n"
    warm_ir, warm_on = "", "drafting"
    if (OCC_MIN or CTX_MAX) and gate_advance(draft["kind"]):
        # a gated span is an ordinary selected decode span: no speculative mark, no credit
        flags_sel = ("    %plain_flags = scalar.ori %flags, %c1_i32 : i32\n"
                     "    %spec_flags = scf.select %speculative, %marked, %flags : i32\n"
                     "    %device_flags = scf.select %gate, %plain_flags, %spec_flags : i32\n")
        # gated epoch: the plain advance trunk (@llm_gate) writes the verification records; nothing proposed, so no catch-up
        verify_body = f"""    %completion = scf.if %gated -> (i64) {{
      %gs_max = scalar.constant {GATE_SLOTS - 1} : i32
      %gs_slot32 = scalar.minui %span_count, %gs_max : i32
      %gs_slot_b = scalar.assume %gs_slot32 [range(%gs_slot32, 0, {GATE_SLOTS - 1})] : i32
      %gs_slot = index.cast %gs_slot_b : i32 to index
      %gs_at = index.constant {GATE_STAGES} : offset
      %gs_view = buffer.view %control_state[%gs_at] : buffer -> view<{GATE_SLOTS}xi32>
      %gstage = view.load %gs_view[%gs_slot] : view<{GATE_SLOTS}xi32> -> i32
      %gcompletion = func.call @execute_7(%gstage, %residual, %metadata, %origins, %state, %tokens, %outputs, %results) : {exe7}
      scf.yield %gcompletion : i64
    }} else {{
{verify_run.replace("%VAR", "%vcompletion")}      scf.yield %vcompletion : i64
    }}
{feedback_ir}"""
        warm_ir = "  %warm_on = scalar.andi %drafting, %ungated : i1\n"
        warm_on = "warm_on"
    else:
        verify_body = verify_run.replace("%VAR", "%completion") + feedback_ir
    dflash = draft["kind"] in ("dflash", "ngram")   # stage signatures: no MTP carry
    mrope = bool((cfg.get("vision") or {}).get("mrope"))
    source_step = 1 if mrope else 2   # text-only ports advance the opaque source position by 2 per input
    # sampling records move behind the larger control header
    n = text.count("index.constant 640 : offset")
    assert n == 3, n
    text = text.replace("index.constant 640 : offset", f"index.constant {SAMPLING} : offset")
    decl = "func.decl @llm_storage(%maximum: i64, %span_capacity: i64, %context: i64, %pool: i64, %rows: i64, %pins: i64) -> (buffer, buffer, buffer, buffer, buffer)"
    text = _sub(text, decl, decl.replace("%pins: i64)", "%pins: i64, %drafting: i64)"))
    text = _sub(text, "  %allocations, %views, %origins, %draft, %geometry = func.call @llm_storage(%maximum, %span_capacity, %context, %pool, %rows, %pins) : (i64, i64, i64, i64, i64, i64) -> (buffer, buffer, buffer, buffer, buffer)\n  func.return %state,",
                f"""  %kind_at = index.constant {DRAFT_KIND} : offset
  %kind_view = buffer.view %state[%kind_at] : buffer -> view<1xi32>
  %kind_word = view.load %kind_view[0] : view<1xi32> -> i32
  %kind_zero = scalar.constant 0 : i32
  %drafting_on = scalar.cmpi ne, %kind_word, %kind_zero : i32
  %drafting64 = scalar.extui %drafting_on : i1 to i64
  %allocations, %views, %origins, %draft, %geometry = func.call @llm_storage(%maximum, %span_capacity, %context, %pool, %rows, %pins, %drafting64) : (i64, i64, i64, i64, i64, i64, i64) -> (buffer, buffer, buffer, buffer, buffer)
  func.return %state,""")
    text = _sub(text, "func.def public @encode_epoch(", "func.def @plain_encode_epoch(")
    text = _sub(text, "func.def public @publish_epoch(", "func.def @plain_publish_epoch(")
    text = _sub(text, "func.def public @epoch(", "func.def @plain_epoch(")
    text = _sub(text, "func.decl import(\"runner\") @execute_6(", """func.decl import("runner") @execute_5(%stage: i32, %b0: hal.buffer, %b1: hal.buffer, %b2: hal.buffer, %b3: hal.buffer, %b4: hal.buffer) -> (i64)

func.decl import("runner") @execute_7(%stage: i32, %b0: hal.buffer, %b1: hal.buffer, %b2: hal.buffer, %b3: hal.buffer, %b4: hal.buffer, %b5: hal.buffer, %b6: hal.buffer) -> (i64)

func.decl import("runner") @execute_4(%stage: i32, %b0: hal.buffer, %b1: hal.buffer, %b2: hal.buffer, %b3: hal.buffer) -> (i64)

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
{gate_pre}  // cursors, flags and the total packed length
  %total = scf.for %s = [%c0 to %spans step %c1](%cursor = %c0_i32 : i32) -> (i32) {{
    %sr = index.assume %s [lt(%s, %plan_extent)] : index
{L1}    %flags = view.load %requests[%sr, 6] : view<[%plan_extent]x9xi32> -> i32
    %credit = view.load %requests[%sr, 7] : view<[%plan_extent]x9xi32> -> i32
    %speculative = scalar.cmpi ne, %credit, %c0_i32 : i32
    %credit_bits = scalar.shli %credit, %c8_i32 : i32
    %spec_bits = scalar.ori %credit_bits, %c4_i32 : i32
    %marked = scalar.ori %flags, %spec_bits : i32
{flags_sel}    view.store %cursor, %device[%s, 4] : i32, view<[%spans]x9xi32>
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
{L2}    %begin32 = view.load %requests[%sr, 4] : view<[%plan_extent]x9xi32> -> i32
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
{gate_epoch}  scf.if %{propose_flag} {{
    %bits = scalar.constant 32 : i32
    %below = scalar.subi %span_count, %c1 : i32
    %leading = scalar.ctlzi %below : i32
    %level = scalar.subi %bits, %leading : i32
    %last = scalar.subi %levels, %c1 : i32
    %chosen = scalar.minui %level, %last : i32
    %pstage = scalar.addi %propose_first, %chosen : i32
{"    %proposal = func.call @execute_4(%pstage, %metadata, %tokens, %draft_origins, %draft_cache) : (i32, hal.buffer, hal.buffer, hal.buffer, hal.buffer) -> (i64)" if dflash else ("    %proposal = func.call @execute_5(%pstage, %metadata, %tokens, %origins, %state, %carry) : (i32, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer) -> (i64)" if draft["kind"] == "assist" else "    %proposal = func.call @execute_5(%pstage, %metadata, %tokens, %draft_origins, %draft_cache, %carry) : (i32, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer) -> (i64)")}
  }}
  %target = scf.if %verify -> (i64) {{
{verify_body}    scf.yield %completion : i64
  }} else {{
    %astage = view.load %advance_view[%{eff}] : view<128xi32> -> i32
    %completion = func.call @execute_6(%astage, %residual, %metadata, %origins, %state, %tokens, %outputs) : (i32, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer) -> (i64)
    %count64 = scalar.extui %span_count : i32 to i64
    %word = scalar.constant 4 : i64
    %length = scalar.muli %count64, %word : i64
    %feedback = func.call @feedback(%c0, %outputs, %c0_i64, %length) : (i32, hal.buffer, i64, i64) -> (i64)
    scf.yield %completion : i64
  }}
{warm_ir}  %done = scf.if %{warm_on} -> (i64) {{
    %accepted = scf.if %verify -> (hal.buffer) {{
      scf.yield %committed : hal.buffer
    }} else {{
      scf.yield %metadata : hal.buffer
    }}
    %wstage = view.load %warm_view[%{eff}] : view<128xi32> -> i32
{"    %warm = func.call @execute_5(%wstage, %residual, %accepted, %draft_origins, %draft_cache, %tokens) : (i32, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer) -> (i64)" if dflash else "    %warm = func.call @execute_6(%wstage, %residual, %accepted, %draft_origins, %draft_cache, %carry, %tokens) : (i32, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer, hal.buffer) -> (i64)"}
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


def card_drafter(card, pkg, draft):
    """DFlash: the drafter checkpoint is a second parameter scope ("draft") of the chat variant."""
    import os
    if draft["kind"] not in ("dflash", "assist"):
        return card
    name = os.path.basename(draft["gguf"])
    link = os.path.join(pkg, "parameter", name)
    if os.path.lexists(link):
        os.remove(link)
    try:
        os.symlink(os.path.abspath(draft["gguf"]), link)
    except OSError:
        os.link(os.path.realpath(draft["gguf"]), link)
    params = card["variants"]["gguf"]["services"]["chat"]["parameters"]
    params["draft"] = {"files": [{"path": f"parameter/{name}", "format": "gguf"}]}
    return card
