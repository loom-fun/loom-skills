"""Choice chat.input_measure: control.loom's prepare_input follows the SDK 0.3.0 complete-conversation count contract.

SDK 0.3.0 (hrx-system #3512, library #125) lets a client count a whole conversation, beyond the deployment's window, with
`POST /v1/chat/completions/input_tokens`. The server asks the model's prepare_input for the input at its capacity; when the
model answers with NO token storage and a count above the capacity, the server (a) rejects a generation request itself
(HTTP 400 "input exceeds the available context size of N tokens") and (b) for a count request calls prepare_input again with
capacity -1, the measuring probe: the tokenizer import `encode(text, flags, -1)` returns only the count (no IDs). A
prepare_input that fails with @require instead (the legacy kit) makes the count route answer HTTP 400 for every conversation
above the window. The same shape as the SDK's text fixture (loom/src/loom/serve/fixture/text/control.loom at v0.3.0):

  capacity == -1        measure: encode(.., -1), return (empty, count, spans)
  count > capacity      return (empty, count, empty spans): the server rejects or re-measures
  otherwise             unchanged

Generation, tokens and every device program are unchanged (prepare_input runs on the host before admission).
"""

PLAIN_OLD = """  %probe = scalar.addi %capacity, %c1_i64 : i64
  %tokens, %count = func.call @encode(%text, %c0_i32, %probe) : (buffer, i32, i64) -> (buffer, i64)
  %fits = scalar.cmpi ule, %count, %capacity : i64
  %fits32 = scalar.extui %fits : i1 to i32
  %error = global.load @overflow : buffer
  func.call @require(%fits32, %error) : (i32, buffer)
  %spans = buffer.alloca<private> align(4) %c0 : buffer
  func.return %tokens, %count, %spans : buffer, i64, buffer
"""
PLAIN_NEW = """  // SDK 0.3.0 count contract (choice chat.input_measure): capacity -1 measures; above the capacity: no tokens, the count
  %measure = scalar.constant -1 : i64
  %measuring = scalar.cmpi eq, %capacity, %measure : i64
  %bounded_probe = scalar.addi %capacity, %c1_i64 : i64
  %probe = scf.select %measuring, %measure, %bounded_probe : i64
  %tokens, %count = func.call @encode(%text, %c0_i32, %probe) : (buffer, i32, i64) -> (buffer, i64)
  %fits = scalar.cmpi ule, %count, %capacity : i64
  %empty = buffer.alloca<private> align(4) %c0 : buffer
  %output = scf.select %fits, %tokens, %empty : buffer
  %spans = buffer.alloca<private> align(4) %c0 : buffer
  func.return %output, %count, %spans : buffer, i64, buffer
"""

# vision/gen_chat_vision.py's @vision_prepare_input (present when the package takes image input)
VISION_PROBE_OLD = """  %probe = scalar.addi %capacity, %c1_i64 : i64
  // pass 1: total inputs
"""
VISION_PROBE_NEW = """  // SDK 0.3.0 count contract (choice chat.input_measure): capacity -1 measures (encode returns counts, no IDs)
  %measure = scalar.constant -1 : i64
  %measuring = scalar.cmpi eq, %capacity, %measure : i64
  %bounded_probe = scalar.addi %capacity, %c1_i64 : i64
  %probe = scf.select %measuring, %measure, %bounded_probe : i64
  %keep_ids = scalar.cmpi ne, %capacity, %measure : i64
  // pass 1: total inputs
"""
VISION_FIT_OLD = """  %fits = scalar.cmpi ule, %total, %capacity : i64
  %fits32 = scalar.extui %fits : i1 to i32
  %error = global.load @overflow : buffer
  func.call @require(%fits32, %error) : (i32, buffer)
  %tb = scalar.assume %total [range(%total, 0, 1073741823)] : i64
  %total_index = index.cast %tb : i64 to index
  %token_bytes = index.scale %total_index, %four : index, offset -> offset
"""
VISION_FIT_NEW = """  %fits = scalar.cmpi ule, %total, %capacity : i64
  %tb = scalar.assume %total [range(%total, 0, 1073741823)] : i64
  %total_index = index.cast %tb : i64 to index
  %all_bytes = index.scale %total_index, %four : index, offset -> offset
  %token_bytes = scf.select %measuring, %origin, %all_bytes : offset
"""
VISION_COPY_OLD = """    %id_bytes = index.scale %ni, %four : index, offset -> offset
    buffer.copy %ids[%origin], %tokens[%out], %id_bytes
"""
VISION_COPY_NEW = """    %id_bytes = index.scale %ni, %four : index, offset -> offset
    scf.if %keep_ids {{
      buffer.copy %ids[%origin], %tokens[%out], %id_bytes
    }}
"""
VISION_TAIL_OLD = """  buffer.copy %tail_ids[%origin], %tokens[%out_end], %tail_bytes
  func.return %tokens, %total, %spans : buffer, i64, buffer
"""
VISION_TAIL_NEW = """  scf.if %keep_ids {{
    buffer.copy %tail_ids[%origin], %tokens[%out_end], %tail_bytes
  }}
  %no_tokens = buffer.alloca<private> align(4) %origin : buffer
  %no_spans = buffer.alloca<private> align(4) %origin : buffer
  %output = scf.select %fits, %tokens, %no_tokens : buffer
  %output_spans = scf.select %fits, %spans, %no_spans : buffer
  func.return %output, %total, %output_spans : buffer, i64, buffer
"""


def _sub(text, old, new, what):
    assert text.count(old) == 1, f"input_measure: {what} not found exactly once ({text.count(old)})"
    return text.replace(old, new)


def patch_control(text):
    """control.loom TEXT with prepare_input (and the vision variant when present) on the 0.3.0 count contract."""
    text = _sub(text, PLAIN_OLD, PLAIN_NEW, "prepare_input body")
    if "func.def @vision_prepare_input(" in text:
        text = _sub(text, VISION_PROBE_OLD, VISION_PROBE_NEW, "vision probe")
        text = _sub(text, VISION_FIT_OLD, VISION_FIT_NEW, "vision fit check")
        # the vision template is an f-string in gen_chat_vision.py: its braces are doubled there, single in the emitted text
        text = _sub(text, VISION_COPY_OLD, VISION_COPY_NEW.replace("{{", "{").replace("}}", "}"), "vision id copy")
        text = _sub(text, VISION_TAIL_OLD, VISION_TAIL_NEW.replace("{{", "{").replace("}}", "}"), "vision tail copy")
    return text
