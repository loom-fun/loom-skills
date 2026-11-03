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
