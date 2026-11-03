"""Adds image input to a package built by build.py (cfg["vision"] present). Called by build.py:
patch_control() edits the control program text, write_component() writes the image component programs,
links the mmproj and extends the card. Text-only builds never call this (their programs are unchanged).

cfg["vision"]: {"mmproj": path, "family": projector type, "img_begin": "...", "img_end": "...",
                "end_states": [s0, s1] (kits with a custom chat: generic chat_end state -> the kit's
                output state) or "format" (call the kit's own chat_end)}"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import encoder as EN               # noqa: E402
import gen_chat_vision as GCV      # noqa: E402
import gen_component as GC         # noqa: E402
import preprocess as PP            # noqa: E402

VBIT = 1 << 61   # chat state bit: this request renders through the generic per-message functions


def _sub(text, old, new, count=1):
    assert text.count(old) == count, (old[:120], text.count(old))
    return text.replace(old, new)


def patch_control(text, custom_chat, vcfg, vc):
    """vc: encoder config (encoder.config). Returns the patched control program."""
    fam = vc["family"]; F = PP.FAMILY[fam]
    geo = GCV.geometry(fam, vc["patch"], F["merge"], F["tokens"][0], F["tokens"][1], F["half_even"])
    fn = "generic_chat_message" if custom_chat else "chat_message"
    # 1. content: text parts + image markers (llama.cpp's join) instead of text-only parts
    text = _sub(text, '''  %raw_text = scf.if %has_parts -> (buffer) {
    %joined = func.call @text_parts(%content, %parts, %part_count) : (buffer, buffer, i64) -> (buffer)
    scf.yield %joined : buffer
  } else {
    scf.yield %content : buffer
  }''', '''  %vision_none = global.load @empty : buffer
  %vision_zero = scalar.constant 0 : i64
  %raw_text, %vision_plan, %vision_count = scf.if %has_parts -> (buffer, buffer, i64) {
    %joined, %vplan, %vcount = func.call @vision_content(%content, %parts, %frames, %part_count) : (buffer, buffer, buffer, i64) -> (buffer, buffer, i64)
    scf.yield %joined, %vplan, %vcount : buffer, buffer, i64
  } else {
    scf.yield %content, %vision_none, %vision_zero : buffer, buffer, i64
  }''')
    text = _sub(text, "  func.return %fragment, %next, %empty, %zero64 : buffer, i64, buffer, i64",
                '''  func.call @vision_locate(%fragment, %vision_plan, %vision_count) : (buffer, buffer, i64)
  func.return %fragment, %next, %vision_plan, %vision_count : buffer, i64, buffer, i64''')
    # 2. prepare_input with markers
    text = _sub(text, "func.def public @prepare_input(", "func.def @plain_prepare_input(")
    text += '''
func.def public @prepare_input(%text: buffer, %markers: buffer, %marker_count: i64, %format: i32, %boundary: i32, %capacity: i64) -> (buffer, i64, buffer) {
  %zero = scalar.constant 0 : i64
  %plain = scalar.cmpi eq, %marker_count, %zero : i64
  %tokens, %count, %spans = scf.if %plain -> (buffer, i64, buffer) {
    %t, %n, %s = func.call @plain_prepare_input(%text, %markers, %marker_count, %format, %boundary, %capacity) : (buffer, buffer, i64, i32, i32, i64) -> (buffer, i64, buffer)
    scf.yield %t, %n, %s : buffer, i64, buffer
  } else {
    %t, %n, %s = func.call @vision_prepare_input(%text, %markers, %marker_count, %capacity) : (buffer, buffer, i64, i64) -> (buffer, i64, buffer)
    scf.yield %t, %n, %s : buffer, i64, buffer
  }
  func.return %tokens, %count, %spans : buffer, i64, buffer
}
'''
    # 3. epochs with media: metadata word 2 = 1 (the device reads 8-word input records)
    text = _sub(text, '''  view.store %c0, %header[2] : i32, view<4xi32>
  view.store %c0, %header[3] : i32, view<4xi32>
  buffer.copy %plan[%origin], %metadata[%header_bytes], %plan_bytes''', '''  %vision_media = scalar.addi %first_media_count, %next_media_count : i32
  %vision_has = scalar.cmpi ne, %vision_media, %c0 : i32
  %vision_flag = scalar.extui %vision_has : i1 to i32
  view.store %vision_flag, %header[2] : i32, view<4xi32>
  view.store %c0, %header[3] : i32, view<4xi32>
  buffer.copy %plan[%origin], %metadata[%header_bytes], %plan_bytes''')
    # 4. kits with a whole-conversation chat program: image requests use the generic per-message path
    if custom_chat:
        for f in ("chat_begin", "chat_message", "chat_end"):
            text = _sub(text, f"func.def public @{f}(", f"func.def @format_{f}(")
        end = vcfg.get("end_states", [0, 1])
        if end == "format":
            end_body = '''    %p, %s = func.call @format_chat_end(%plain_state) : (i64) -> (buffer, i64)
    scf.yield %p, %s : buffer, i64'''
        else:
            end_body = f'''    %p, %g = func.call @generic_chat_end(%plain_state) : (i64) -> (buffer, i64)
    %zero_e = scalar.constant 0 : i64
    %is0 = scalar.cmpi eq, %g, %zero_e : i64
    %s0 = scalar.constant {end[0]} : i64
    %s1 = scalar.constant {end[1]} : i64
    %s = scf.select %is0, %s0, %s1 : i64
    scf.yield %p, %s : buffer, i64'''
        text += f'''
global.rodata.def @vision_image_key = bytes("{GCV.hexs('"image_url"')}")
func.def @vision_request(%request: buffer) -> (i1) {{
  %origin = index.constant 0 : offset
  %end = buffer.length %request
  %key = global.load @vision_image_key : buffer
  %at = func.call @find(%request, %origin, %end, %key) : (buffer, offset, offset, buffer) -> (offset)
  %found = index.cmp ult, %at, %end : offset
  func.return %found : i1
}}

func.def public @chat_begin(%request: buffer, %tools: buffer) -> (buffer, i64) {{
  %images = func.call @vision_request(%request) : (buffer) -> (i1)
  %bit = scalar.constant {VBIT} : i64
  %text, %state = scf.if %images -> (buffer, i64) {{
    %t, %s = func.call @generic_chat_begin(%request, %tools) : (buffer, buffer) -> (buffer, i64)
    %sv = scalar.ori %s, %bit : i64
    scf.yield %t, %sv : buffer, i64
  }} else {{
    %t, %s = func.call @format_chat_begin(%request, %tools) : (buffer, buffer) -> (buffer, i64)
    scf.yield %t, %s : buffer, i64
  }}
  func.return %text, %state : buffer, i64
}}

func.def public @chat_message(%raw: buffer, %role: buffer, %content: buffer, %parts: buffer, %frames: buffer, %reasoning: buffer, %calls: buffer, %state: i64, %ordinal: i64, %fields: i64, %part_count: i64) -> (buffer, i64, buffer, i64) {{
  %bit = scalar.constant {VBIT} : i64
  %zero = scalar.constant 0 : i64
  %masked = scalar.andi %state, %bit : i64
  %images = scalar.cmpi ne, %masked, %zero : i64
  %mask = scalar.constant {~VBIT} : i64
  %plain_state = scalar.andi %state, %mask : i64
  %f, %n, %p, %c = scf.if %images -> (buffer, i64, buffer, i64) {{
    %f1, %n1, %p1, %c1 = func.call @generic_chat_message(%raw, %role, %content, %parts, %frames, %reasoning, %calls, %plain_state, %ordinal, %fields, %part_count) : (buffer, buffer, buffer, buffer, buffer, buffer, buffer, i64, i64, i64, i64) -> (buffer, i64, buffer, i64)
    %nv = scalar.ori %n1, %bit : i64
    scf.yield %f1, %nv, %p1, %c1 : buffer, i64, buffer, i64
  }} else {{
    %f2, %n2, %p2, %c2 = func.call @format_chat_message(%raw, %role, %content, %parts, %frames, %reasoning, %calls, %state, %ordinal, %fields, %part_count) : (buffer, buffer, buffer, buffer, buffer, buffer, buffer, i64, i64, i64, i64) -> (buffer, i64, buffer, i64)
    scf.yield %f2, %n2, %p2, %c2 : buffer, i64, buffer, i64
  }}
  func.return %f, %n, %p, %c : buffer, i64, buffer, i64
}}

func.def public @chat_end(%state: i64) -> (buffer, i64) {{
  %bit = scalar.constant {VBIT} : i64
  %zero = scalar.constant 0 : i64
  %masked = scalar.andi %state, %bit : i64
  %images = scalar.cmpi ne, %masked, %zero : i64
  %mask = scalar.constant {~VBIT} : i64
  %plain_state = scalar.andi %state, %mask : i64
  %prompt, %out = scf.if %images -> (buffer, i64) {{
{end_body}
  }} else {{
    %p2, %s2 = func.call @format_chat_end(%state) : (i64) -> (buffer, i64)
    scf.yield %p2, %s2 : buffer, i64
  }}
  func.return %prompt, %out : buffer, i64
}}
'''
    # 5. image limits: 4 images of up to 4096 x 4096
    text = _sub(text, "  %compressed_bytes = scalar.constant 1048576 : i64", "  %compressed_bytes = scalar.constant 67108864 : i64")
    text += GCV.chat_lib(vcfg["img_begin"], vcfg["img_end"], geo)
    if vcfg.get("mrope"):
        text = mrope_control(text, vc["patch"] * F["merge"])
    return text


def write_component(pkg, card, vcfg, vc, max_rows=None):
    """Writes program/vision_*.loom, links the mmproj, adds the image component to the card."""
    fam = vc["family"]; F = PP.FAMILY[fam]; f = vc["patch"] * F["merge"]
    max_rows = max_rows or F["tokens"][1]
    prog = os.path.join(pkg, "program")
    open(os.path.join(prog, "vision_encoder.loom"), "w").write(GC.encoder_program(vc))
    # default stage size (cold preparation compiles one): a square of the median token budget
    side = int(round((F["tokens"][1] ** 0.5) / 2)) * f
    open(os.path.join(prog, "vision_prepare.loom"), "w").write(GC.prepare_program(vc, (side, side), vc["out"], max_rows))
    open(os.path.join(prog, "vision_stages.loom"), "w").write(GC.stages_library())
    # scratch at the largest size (bytes), for the memory forecast
    T = max_rows * F["merge"] ** 2; Tp = -(-T // 256) * 256; Mp = -(-max_rows // 64) * 64
    scratch = sum((Tp if r == "Tp" else Mp) * w * b for r, w, b in EN.buffer_exprs(vc).values()) + 4096 * (f * max_rows ** 0.5 + 64) * 3 * 2
    open(os.path.join(prog, "vision_control.loom"), "w").write(GC.control_program(vc, vc["out"], max_rows, int(scratch)))
    name = os.path.basename(vcfg["mmproj"])
    link = os.path.join(pkg, "parameter", name)
    if os.path.lexists(link):
        os.remove(link)
    try:
        os.symlink(os.path.abspath(vcfg["mmproj"]), link)
    except OSError:
        os.link(os.path.realpath(vcfg["mmproj"]), link)
    src = card["programs"]["source"]
    src["inputs"] += ["program/vision_prepare.loom", "program/vision_control.loom"]
    src["libraries"] += ["program/vision_stages.loom", "program/vision_encoder.loom"]
    card["services"]["chat"]["components"] = {"image": {"version": 1}}
    svc = card["variants"]["gguf"]["services"]["chat"]
    svc["parameters"]["vision"] = {"files": [{"path": f"parameter/{name}", "format": "gguf"}]}
    return card


# ---- M-RoPE (Qwen-VL): rotary positions follow the host's source position, not the KV position ----
# Text advances the source position by 1 per input; an image row advances it only when it extends the
# image's grid frontier (llama.cpp: the image occupies max(rows, cols) positions; row r, col c of an image
# starting at p is (t, h, w) = (p, p + r, p + c)). encode_epoch gives the device a per-span offset
# (source - KV position) after the plan, and, in media epochs, explicit (t, h, w) in record words 4..6.
def mrope_control(text, merge_px):
    text = _sub(text, '''  %transfer_offset = index.constant 8 : offset
  %transfers = buffer.view %control_state[%transfer_offset] : buffer -> view<2x5xi64>''', f'''  // M-RoPE: per-span source-position offset after the plan; media records carry (t, h, w)
  %mr_total = scalar.addi %first_count, %next_count : i32
  %mr_total_b = scalar.assume %first_count [range(%first_count, 0, 8192)] : i32
  %mr_spans = index.cast %mr_total_b : i32 to index
  %mr_plan_extent = func.call @plan_capacity(%plan) : (buffer) -> (index)
  %mr_requests = buffer.view %plan[%origin] : buffer -> view<[%mr_plan_extent]x9xi32>
  %mr_table_at = index.add %header_bytes, %plan_bytes : offset
  %mr_table = buffer.view %metadata[%mr_table_at] : buffer -> view<[%mr_spans]xi32>
  %mr_c0 = index.constant 0 : index
  %mr_c1 = index.constant 1 : index
  scf.for %mr_s = [%mr_c0 to %mr_spans step %mr_c1] {{
    %mr_r = index.assume %mr_s [lt(%mr_s, %mr_plan_extent)] : index
    %mr_pos = view.load %mr_requests[%mr_r, 1] : view<[%mr_plan_extent]x9xi32> -> i32
    %mr_src = view.load %mr_requests[%mr_r, 2] : view<[%mr_plan_extent]x9xi32> -> i32
    %mr_delta = scalar.subi %mr_src, %mr_pos : i32
    view.store %mr_delta, %mr_table[%mr_s] : i32, view<[%mr_spans]xi32>
  }}
  scf.if %has_media {{
    %mr_cap = scalar.addi %first_packet_capacity, %next_packet_capacity : i32
    func.call @vision_mrope_records(%plan, %media_plan, %media_count, %mr_cap, %tokens, %mr_total) : (buffer, buffer, i32, i32, buffer, i32)
  }}
  %transfer_offset = index.constant 8 : offset
  %transfers = buffer.view %control_state[%transfer_offset] : buffer -> view<2x5xi64>''')
    text = _sub(text, '''  %metadata_length = scalar.addi %header_length, %plan_length : i64''', '''  %mr_table_length = scalar.muli %count64, %mr_four64 : i64
  %metadata_plan_length = scalar.addi %header_length, %plan_length : i64
  %metadata_length = scalar.addi %metadata_plan_length, %mr_table_length : i64''')
    text = _sub(text, '''  %header_length = scalar.constant 16 : i64''', '''  %header_length = scalar.constant 16 : i64
  %mr_four64 = scalar.constant 4 : i64''')
    # publish: source advance = consumed minus image rows that do not extend their grid's frontier
    text = _sub(text, '''    %source_advance = scalar.muli %consumed, %c2_i32 : i32
    %next_source_position = scalar.addi %source_position, %source_advance : i32
    %flags = view.load %requests[%request, 6] : view<[%plan_extent]x9xi32> -> i32''', '''    %mr_saved = func.call @vision_mrope_saved(%media_plan, %first_media_count, %next_media_count, %i) : (buffer, i32, i32, index) -> (i32)
    %source_advance = scalar.subi %consumed, %mr_saved : i32
    %next_source_position = scalar.addi %source_position, %source_advance : i32
    %flags = view.load %requests[%request, 6] : view<[%plan_extent]x9xi32> -> i32''')
    text += f'''
// ---- M-RoPE positions (generated by build_vision.py) ----
// Grid frontier after `consumed` rows of a row-major grid `width` wide: one past the largest coordinate.
func.def @vision_frontier(%consumed: i32, %width: i32) -> (i32) {{
  %one = scalar.constant 1 : i32
  %bias = scalar.subi %width, %one : i32
  %rounded = scalar.addi %consumed, %bias : i32
  %rows = scalar.divui %rounded, %width : i32
  %cols = scalar.minui %consumed, %width : i32
  %frontier = scalar.maxui %rows, %cols : i32
  func.return %frontier : i32
}}

// Media entries [plan index, relative input, feature offset, feature bytes, metadata offset, metadata bytes,
// feature index]; metadata = [processed width, processed height].
func.def @vision_entry_width(%media_plan: buffer, %meta_offset: i32) -> (i32) {{
  %zero = scalar.constant 0 : i32
  %mo_c = scalar.maxsi %meta_offset, %zero : i32
  %mo64 = scalar.extui %mo_c : i32 to i64
  %mo_b = scalar.assume %mo64 [range(%mo64, 0, 4294967295)] : i64
  %mo = index.cast %mo_b : i64 to offset
  %meta = buffer.view %media_plan[%mo] : buffer -> view<2xi32>
  %width_px = view.load %meta[0] : view<2xi32> -> i32
  %px = scalar.constant {merge_px} : i32
  %width = scalar.divui %width_px, %px : i32
  %one = scalar.constant 1 : i32
  %safe = scalar.maxui %width, %one : i32
  func.return %safe : i32
}}

// Rows of span `span` that did not advance the source position (1 - frontier advance each).
func.def @vision_mrope_saved(%media_plan: buffer, %first: i32, %next: i32, %span: index) -> (i32) {{
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %zero = scalar.constant 0 : i32
  %one = scalar.constant 1 : i32
  %total = scalar.addi %first, %next : i32
  %tb = scalar.assume %total [range(%total, 0, 524288)] : i32
  %count = index.cast %tb : i32 to index
  %entries = buffer.view %media_plan[%origin] : buffer -> view<[%count]x7xi32>
  %span32a = index.cast %span : index to i64
  %span32 = scalar.trunci %span32a : i64 to i32
  %saved = scf.for %e = [%c0 to %count step %c1](%acc = %zero : i32) -> (i32) {{
    %owner = view.load %entries[%e, 0] : view<[%count]x7xi32> -> i32
    %mine = scalar.cmpi eq, %owner, %span32 : i32
    %next_acc = scf.if %mine -> (i32) {{
      %meta_offset = view.load %entries[%e, 4] : view<[%count]x7xi32> -> i32
      %fi = view.load %entries[%e, 6] : view<[%count]x7xi32> -> i32
      %width = func.call @vision_entry_width(%media_plan, %meta_offset) : (buffer, i32) -> (i32)
      %before = func.call @vision_frontier(%fi, %width) : (i32, i32) -> (i32)
      %fi1 = scalar.addi %fi, %one : i32
      %after = func.call @vision_frontier(%fi1, %width) : (i32, i32) -> (i32)
      %advance = scalar.subi %after, %before : i32
      %not = scalar.subi %one, %advance : i32
      %sum = scalar.addi %acc, %not : i32
      scf.yield %sum : i32
    }} else {{
      scf.yield %acc : i32
    }}
    scf.yield %next_acc : i32
  }}
  func.return %saved : i32
}}

// Record words 4..6 = (t, h, w) for every input of every span (media or text).
func.def @vision_mrope_records(%plan: buffer, %media_plan: buffer, %media_count: i32, %packet_capacity: i32, %packet: buffer, %span_count: i32) {{
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %one = scalar.constant 1 : i32
  %mb = scalar.assume %media_count [range(%media_count, 1, 524288)] : i32
  %media_n = index.cast %mb : i32 to index
  %pb = scalar.assume %packet_capacity [range(%packet_capacity, 1, 524288)] : i32
  %packet_n = index.cast %pb : i32 to index
  %sb = scalar.assume %span_count [range(%span_count, 0, 8192)] : i32
  %spans = index.cast %sb : i32 to index
  %plan_extent = func.call @plan_capacity(%plan) : (buffer) -> (index)
  %requests = buffer.view %plan[%origin] : buffer -> view<[%plan_extent]x9xi32>
  %entries = buffer.view %media_plan[%origin] : buffer -> view<[%media_n]x7xi32>
  %records = buffer.view %packet[%origin] : buffer -> view<[%packet_n]x8xi32>
  %final = scf.for %s = [%c0 to %spans step %c1](%cursor = %c0 : index) -> (index) {{
    %r = index.assume %s [lt(%s, %plan_extent)] : index
    %length32 = view.load %requests[%r, 0] : view<[%plan_extent]x9xi32> -> i32
    %source0 = view.load %requests[%r, 2] : view<[%plan_extent]x9xi32> -> i32
    %begin32 = view.load %requests[%r, 4] : view<[%plan_extent]x9xi32> -> i32
    %lb = scalar.assume %length32 [range(%length32, 0, 524288)] : i32
    %bb = scalar.assume %begin32 [range(%begin32, 0, 524287)] : i32
    %length = index.cast %lb : i32 to index
    %begin = index.cast %bb : i32 to index
    %s64 = index.cast %s : index to i64
    %s32 = scalar.trunci %s64 : i64 to i32
    %src_end, %cur_end = scf.for %j = [%c0 to %length step %c1](%source = %source0 : i32, %cur = %cursor : index) -> (i32, index) {{
      %at0 = index.add %begin, %j : index
      %at = index.assume %at0 [lt(%at0, %packet_n)] : index
      %j64 = index.cast %j : index to i64
      %j32 = scalar.trunci %j64 : i64 to i32
      %has = index.cmp ult, %cur, %media_n : index
      %match = scf.if %has -> (i1) {{
        %cb = index.assume %cur [lt(%cur, %media_n)] : index
        %owner = view.load %entries[%cb, 0] : view<[%media_n]x7xi32> -> i32
        %rel = view.load %entries[%cb, 1] : view<[%media_n]x7xi32> -> i32
        %m0 = scalar.cmpi eq, %owner, %s32 : i32
        %m1 = scalar.cmpi eq, %rel, %j32 : i32
        %m = scalar.andi %m0, %m1 : i1
        scf.yield %m : i1
      }} else {{
        %f = scalar.constant false : i1
        scf.yield %f : i1
      }}
      %next_source, %next_cur = scf.if %match -> (i32, index) {{
        %cb = index.assume %cur [lt(%cur, %media_n)] : index
        %meta_offset = view.load %entries[%cb, 4] : view<[%media_n]x7xi32> -> i32
        %fi = view.load %entries[%cb, 6] : view<[%media_n]x7xi32> -> i32
        %width = func.call @vision_entry_width(%media_plan, %meta_offset) : (buffer, i32) -> (i32)
        %prefix = func.call @vision_frontier(%fi, %width) : (i32, i32) -> (i32)
        %base = scalar.subi %source, %prefix : i32
        %row = scalar.divui %fi, %width : i32
        %col = scalar.remui %fi, %width : i32
        %ph = scalar.addi %base, %row : i32
        %pw = scalar.addi %base, %col : i32
        view.store %base, %records[%at, 4] : i32, view<[%packet_n]x8xi32>
        view.store %ph, %records[%at, 5] : i32, view<[%packet_n]x8xi32>
        view.store %pw, %records[%at, 6] : i32, view<[%packet_n]x8xi32>
        %fi1 = scalar.addi %fi, %one : i32
        %suffix = func.call @vision_frontier(%fi1, %width) : (i32, i32) -> (i32)
        %adv = scalar.subi %suffix, %prefix : i32
        %ns = scalar.addi %source, %adv : i32
        %nc = index.add %cur, %c1 : index
        scf.yield %ns, %nc : i32, index
      }} else {{
        view.store %source, %records[%at, 4] : i32, view<[%packet_n]x8xi32>
        view.store %source, %records[%at, 5] : i32, view<[%packet_n]x8xi32>
        view.store %source, %records[%at, 6] : i32, view<[%packet_n]x8xi32>
        %ns = scalar.addi %source, %one : i32
        scf.yield %ns, %cur : i32, index
      }}
      scf.yield %next_source, %next_cur : i32, index
    }}
    scf.yield %cur_end : index
  }}
  func.return
}}
'''
    return text


def mrope_storage(text):
    """Metadata: the device plan copy (36 bytes per span) plus the span offset table (4 bytes per span)."""
    return _sub(text, "  %record_bytes = scalar.constant 36 : i64", "  %record_bytes = scalar.constant 40 : i64")
