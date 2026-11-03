"""VM (control program) code for image input: geometry, content rendering with markers, marker location,
and marker-aware prepare_input. Appended to control.loom by build.py when the config has a vision block."""

def hexs(s): return s.encode().hex()

def geometry(fam, P, m, tmin, tmax, half_even):
    """@vision_geometry(w, h) -> (tw, th, rows): llama.cpp mtmd target size for this projector family."""
    f = P * m
    if fam == "muse-glimmer":
        # muse_glimmer_grid_size (doubles): nearest-aspect grid under max tokens
        return f'''func.def @vision_geometry(%w: i32, %h: i32) -> (i32, i32, i32) {{
  %fd = scalar.constant {float(f)} : f64
  %maxd = scalar.constant {float(tmax)} : f64
  %oned = scalar.constant 1.0 : f64
  %zerod = scalar.constant 0.0 : f64
  %wd = scalar.uitofp %w : i32 to f64
  %hd = scalar.uitofp %h : i32 to f64
  %nph0 = scalar.divf %hd, %fd : f64
  %npw0 = scalar.divf %wd, %fd : f64
  %ratio = scalar.divf %npw0, %nph0 : f64
  %area = scalar.mulf %nph0, %npw0 : f64
  %over = scalar.cmpf ogt, %area, %maxd : f64
  %q = scalar.divf %maxd, %ratio : f64
  // sqrt in F64 (the VM has F32 sqrt only): an F32 estimate, two Newton steps
  %q32 = scalar.fptrunc %q : f64 to f32
  %r32 = scalar.sqrtf %q32 : f32
  %r0 = scalar.extf %r32 : f32 to f64
  %halfd = scalar.constant 0.5 : f64
  %qd0 = scalar.divf %q, %r0 : f64
  %s0 = scalar.addf %r0, %qd0 : f64
  %r1 = scalar.mulf %s0, %halfd : f64
  %qd1 = scalar.divf %q, %r1 : f64
  %s1 = scalar.addf %r1, %qd1 : f64
  %nph1 = scalar.mulf %s1, %halfd : f64
  %npw1 = scalar.mulf %nph1, %ratio : f64
  %nph = scf.select %over, %nph1, %nph0 : f64
  %npw = scf.select %over, %npw1, %npw0 : f64
  %hf = scalar.floorf %nph : f64
  %hc = scalar.ceilf %nph : f64
  %wf = scalar.floorf %npw : f64
  %wc = scalar.ceilf %npw : f64
  %ar = scalar.divf %hd, %wd : f64
  %best_h, %best_w, %best_d, %found = func.call @vision_glimmer_pick(%hf, %hc, %wf, %wc, %ar, %maxd) : (f64, f64, f64, f64, f64, f64) -> (f64, f64, f64, i1)
  %rh0 = scalar.roundf %nph : f64
  %rw0 = scalar.roundf %npw : f64
  %rh = scalar.maxnumf %rh0, %oned : f64
  %rw = scalar.maxnumf %rw0, %oned : f64
  %gh = scf.select %found, %best_h, %rh : f64
  %gw = scf.select %found, %best_w, %rw : f64
  %ghi = scalar.fptoui %gh : f64 to i32
  %gwi = scalar.fptoui %gw : f64 to i32
  %fi = scalar.constant {f} : i32
  %tw = scalar.muli %gwi, %fi : i32
  %th = scalar.muli %ghi, %fi : i32
  %rows = scalar.muli %ghi, %gwi : i32
  func.return %tw, %th, %rows : i32, i32, i32
}}

func.def @vision_glimmer_try(%nph: f64, %npw: f64, %ar: f64, %maxd: f64, %bh: f64, %bw: f64, %bd: f64, %found: i1) -> (f64, f64, f64, i1) {{
  %oned = scalar.constant 1.0 : f64
  %ok_h = scalar.cmpf oge, %nph, %oned : f64
  %ok_w = scalar.cmpf oge, %npw, %oned : f64
  %n = scalar.mulf %nph, %npw : f64
  %ok_n = scalar.cmpf ole, %n, %maxd : f64
  %ok0 = scalar.andi %ok_h, %ok_w : i1
  %ok = scalar.andi %ok0, %ok_n : i1
  %r = scalar.divf %nph, %npw : f64
  %dd = scalar.subf %r, %ar : f64
  %d = scalar.absf %dd : f64
  %bn = scalar.mulf %bh, %bw : f64
  %less = scalar.cmpf olt, %d, %bd : f64
  %eq = scalar.cmpf oeq, %d, %bd : f64
  %more = scalar.cmpf ogt, %n, %bn : f64
  %tie = scalar.andi %eq, %more : i1
  %better0 = scalar.ori %less, %tie : i1
  %true = scalar.constant true : i1
  %first = scalar.xori %found, %true : i1
  %better1 = scalar.ori %first, %better0 : i1
  %take = scalar.andi %ok, %better1 : i1
  %nh = scf.select %take, %nph, %bh : f64
  %nw = scf.select %take, %npw, %bw : f64
  %nd = scf.select %take, %d, %bd : f64
  %nf = scalar.ori %found, %take : i1
  func.return %nh, %nw, %nd, %nf : f64, f64, f64, i1
}}

func.def @vision_glimmer_pick(%hf: f64, %hc: f64, %wf: f64, %wc: f64, %ar: f64, %maxd: f64) -> (f64, f64, f64, i1) {{
  %zerod = scalar.constant 0.0 : f64
  %false = scalar.constant false : i1
  %h1, %w1, %d1, %f1 = func.call @vision_glimmer_try(%hf, %wf, %ar, %maxd, %zerod, %zerod, %zerod, %false) : (f64, f64, f64, f64, f64, f64, f64, i1) -> (f64, f64, f64, i1)
  %h2, %w2, %d2, %f2 = func.call @vision_glimmer_try(%hf, %wc, %ar, %maxd, %h1, %w1, %d1, %f1) : (f64, f64, f64, f64, f64, f64, f64, i1) -> (f64, f64, f64, i1)
  %h3, %w3, %d3, %f3 = func.call @vision_glimmer_try(%hc, %wf, %ar, %maxd, %h2, %w2, %d2, %f2) : (f64, f64, f64, f64, f64, f64, f64, i1) -> (f64, f64, f64, i1)
  %h4, %w4, %d4, %f4 = func.call @vision_glimmer_try(%hc, %wc, %ar, %maxd, %h3, %w3, %d3, %f3) : (f64, f64, f64, f64, f64, f64, f64, i1) -> (f64, f64, f64, i1)
  func.return %h4, %w4, %d4, %f4 : f64, f64, f64, i1
}}
'''
    rnd = "scalar.roundevenf" if half_even else "scalar.roundf"
    return f'''func.def @vision_geometry(%w: i32, %h: i32) -> (i32, i32, i32) {{
  %ff = scalar.constant {float(f)} : f32
  %fi = scalar.constant {f} : i32
  %mx = scalar.constant {tmax * f * f} : i32
  %mn = scalar.constant {tmin * f * f} : i32
  %mxf = scalar.constant {float(tmax * f * f)} : f32
  %mnf = scalar.constant {float(tmin * f * f)} : f32
  %wf = scalar.uitofp %w : i32 to f32
  %hf = scalar.uitofp %h : i32 to f32
  %wq = scalar.divf %wf, %ff : f32
  %hq = scalar.divf %hf, %ff : f32
  %wr = {rnd} %wq : f32
  %hr = {rnd} %hq : f32
  %wri = scalar.fptoui %wr : f32 to i32
  %hri = scalar.fptoui %hr : f32 to i32
  %wb0 = scalar.muli %wri, %fi : i32
  %hb0 = scalar.muli %hri, %fi : i32
  %wb = scalar.maxui %wb0, %fi : i32
  %hb = scalar.maxui %hb0, %fi : i32
  %area = scalar.muli %wb, %hb : i32
  %over = scalar.cmpi ugt, %area, %mx : i32
  %under = scalar.cmpi ult, %area, %mn : i32
  %hw = scalar.mulf %hf, %wf : f32
  %r_over = scalar.divf %hw, %mxf : f32
  %b_over = scalar.sqrtf %r_over : f32
  %ho0 = scalar.divf %hf, %b_over : f32
  %wo0 = scalar.divf %wf, %b_over : f32
  %ho1 = scalar.divf %ho0, %ff : f32
  %wo1 = scalar.divf %wo0, %ff : f32
  %ho2 = scalar.floorf %ho1 : f32
  %wo2 = scalar.floorf %wo1 : f32
  %hoi = scalar.fptoui %ho2 : f32 to i32
  %woi = scalar.fptoui %wo2 : f32 to i32
  %ho3 = scalar.muli %hoi, %fi : i32
  %wo3 = scalar.muli %woi, %fi : i32
  %ho = scalar.maxui %ho3, %fi : i32
  %wo = scalar.maxui %wo3, %fi : i32
  %r_under = scalar.divf %mnf, %hw : f32
  %b_under = scalar.sqrtf %r_under : f32
  %hu0 = scalar.mulf %hf, %b_under : f32
  %wu0 = scalar.mulf %wf, %b_under : f32
  %hu1 = scalar.divf %hu0, %ff : f32
  %wu1 = scalar.divf %wu0, %ff : f32
  %hu2 = scalar.ceilf %hu1 : f32
  %wu2 = scalar.ceilf %wu1 : f32
  %hui = scalar.fptoui %hu2 : f32 to i32
  %wui = scalar.fptoui %wu2 : f32 to i32
  %hu = scalar.muli %hui, %fi : i32
  %wu = scalar.muli %wui, %fi : i32
  %th1 = scf.select %under, %hu, %hb : i32
  %tw1 = scf.select %under, %wu, %wb : i32
  %th = scf.select %over, %ho, %th1 : i32
  %tw = scf.select %over, %wo, %tw1 : i32
  %gh = scalar.divui %th, %fi : i32
  %gw = scalar.divui %tw, %fi : i32
  %rows = scalar.muli %gh, %gw : i32
  func.return %tw, %th, %rows : i32, i32, i32
}}
'''

def chat_lib(img_beg, img_end, geometry_src):
    S = "\x1aLOOM_IMAGE\x1a"
    return f'''
// ---- image input (generated by gen_chat_vision.py) ----
// An image part renders as BEGIN + SENTINEL + END. The plan's marker covers only the sentinel, which
// prepare_input replaces with one placeholder token (id 0) per projected row; the projected rows replace
// those inputs' embeddings on the device (llama.cpp: img_beg + embeddings + img_end).
global.rodata.def @vision_begin = bytes("{hexs(img_beg)}")
global.rodata.def @vision_end = bytes("{hexs(img_end)}")
global.rodata.def @vision_sentinel = bytes("{hexs(S)}")
global.rodata.def @vision_newline = bytes("0a")

{geometry_src}
// Text parts joined as llama.cpp's server does: a newline between text parts, none next to an image.
// Returns (text, plan, image count). Plan: per image [image_begin, image_count, marker offset (set by
// vision_locate), marker length, rows, metadata offset, metadata length] then metadata [width, height].
func.def @vision_content(%content: buffer, %parts: buffer, %frames: buffer, %part_count: i64) -> (buffer, buffer, i64) {{
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c4 = index.constant 4 : index
  %zero32 = scalar.constant 0 : i32
  %text_kind = scalar.constant 0 : i32
  %bounded_count = scalar.assume %part_count [range(%part_count, 1, 4096)] : i64
  %count = index.cast %bounded_count : i64 to index
  %records = buffer.view %parts[%origin] : buffer -> view<[%count]x6xi32>
  %begin = global.load @vision_begin : buffer
  %end = global.load @vision_end : buffer
  %sentinel = global.load @vision_sentinel : buffer
  %newline = global.load @vision_newline : buffer
  %begin_length = buffer.length %begin
  %end_length = buffer.length %end
  %sentinel_length = buffer.length %sentinel
  %marker_bytes0 = index.add %begin_length, %end_length : offset
  %marker_bytes = index.add %marker_bytes0, %sentinel_length : offset
  %images, %text_bytes = scf.for %i = [%c0 to %count step %c1](%n = %c0 : index, %b = %origin : offset) -> (index, offset) {{
    %kind = view.load %records[%i, 0] : view<[%count]x6xi32> -> i32
    %is_text = scalar.cmpi eq, %kind, %text_kind : i32
    %length32 = view.load %records[%i, 2] : view<[%count]x6xi32> -> i32
    %length64 = scalar.extui %length32 : i32 to i64
    %lb = scalar.assume %length64 [range(%length64, 0, 4294967295)] : i64
    %length = index.cast %lb : i64 to offset
    %one_byte = index.constant 1 : offset
    %with_nl = index.add %length, %one_byte : offset
    %grow = scf.select %is_text, %with_nl, %marker_bytes : offset
    %nb = index.add %b, %grow : offset
    %inc = scf.select %is_text, %c0, %c1 : index
    %nn = index.add %n, %inc : index
    scf.yield %nn, %nb : index, offset
  }}
  %scratch = buffer.alloca<private> align(1) %text_bytes : buffer
  %c28 = index.constant 28 : offset
  %c8 = index.constant 8 : offset
  %record_bytes = index.scale %images, %c28 : index, offset -> offset
  %meta_bytes = index.scale %images, %c8 : index, offset -> offset
  %plan_bytes = index.add %record_bytes, %meta_bytes : offset
  %plan = buffer.alloca<private> align(4) %plan_bytes : buffer
  %frame_bytes = buffer.length %frames
  %frame_words0 = index.cast %frame_bytes : offset to index
  %frame_words = index.div %frame_words0, %c4 : index
  %frame_view = buffer.view %frames[%origin] : buffer -> view<[%frame_words]xi32>
  %false = scalar.constant false : i1
  %used, %image_n, %last_media = scf.for %i = [%c0 to %count step %c1](%cursor = %origin : offset, %image = %c0 : index, %was_media = %false : i1) -> (offset, index, i1) {{
    %kind = view.load %records[%i, 0] : view<[%count]x6xi32> -> i32
    %is_text = scalar.cmpi eq, %kind, %text_kind : i32
    %nc, %ni, %nm = scf.if %is_text -> (offset, index, i1) {{
      %source32 = view.load %records[%i, 1] : view<[%count]x6xi32> -> i32
      %length32 = view.load %records[%i, 2] : view<[%count]x6xi32> -> i32
      %source64 = scalar.extui %source32 : i32 to i64
      %length64 = scalar.extui %length32 : i32 to i64
      %sb = scalar.assume %source64 [range(%source64, 0, 4294967295)] : i64
      %lb = scalar.assume %length64 [range(%length64, 0, 4294967295)] : i64
      %source = index.cast %sb : i64 to offset
      %length = index.cast %lb : i64 to offset
      %nonempty = index.cmp ne, %cursor, %origin : offset
      %true = scalar.constant true : i1
      %not_media = scalar.xori %was_media, %true : i1
      %add_nl = scalar.andi %nonempty, %not_media : i1
      %after_nl = scf.if %add_nl -> (offset) {{
        %one_byte = index.constant 1 : offset
        buffer.copy %newline[%origin], %scratch[%cursor], %one_byte
        %c = index.add %cursor, %one_byte : offset
        scf.yield %c : offset
      }} else {{
        scf.yield %cursor : offset
      }}
      buffer.copy %content[%source], %scratch[%after_nl], %length
      %after = index.add %after_nl, %length : offset
      scf.yield %after, %image, %false : offset, index, i1
    }} else {{
      %frame32 = view.load %records[%i, 3] : view<[%count]x6xi32> -> i32
      %frame_count32 = view.load %records[%i, 4] : view<[%count]x6xi32> -> i32
      %frame_bound = scalar.assume %frame32 [range(%frame32, 0, 4095)] : i32
      %frame = index.cast %frame_bound : i32 to index
      %wi0 = index.mul %frame, %c4 : index
      %hi0 = index.add %wi0, %c1 : index
      %wi = index.assume %wi0 [lt(%wi0, %frame_words)] : index
      %hi = index.assume %hi0 [lt(%hi0, %frame_words)] : index
      %width = view.load %frame_view[%wi] : view<[%frame_words]xi32> -> i32
      %height = view.load %frame_view[%hi] : view<[%frame_words]xi32> -> i32
      %tw, %th, %rows = func.call @vision_geometry(%width, %height) : (i32, i32) -> (i32, i32, i32)
      buffer.copy %begin[%origin], %scratch[%cursor], %begin_length
      %c_a = index.add %cursor, %begin_length : offset
      buffer.copy %sentinel[%origin], %scratch[%c_a], %sentinel_length
      %c_b = index.add %c_a, %sentinel_length : offset
      buffer.copy %end[%origin], %scratch[%c_b], %end_length
      %c_c = index.add %c_b, %end_length : offset
      %bi = index.assume %image [range(%image, 0, 4095), lt(%image, %images)] : index
      %rec = buffer.view %plan[%origin] : buffer -> view<[%images]x7xi32>
      %sl64 = index.cast %sentinel_length : offset to i64
      %sl32 = scalar.trunci %sl64 : i64 to i32
      %mo0 = index.scale %bi, %c8 : index, offset -> offset
      %mo1 = index.add %record_bytes, %mo0 : offset
      %mo64 = index.cast %mo1 : offset to i64
      %mo32 = scalar.trunci %mo64 : i64 to i32
      %eight32 = scalar.constant 8 : i32
      view.store %frame32, %rec[%bi, 0] : i32, view<[%images]x7xi32>
      view.store %frame_count32, %rec[%bi, 1] : i32, view<[%images]x7xi32>
      view.store %zero32, %rec[%bi, 2] : i32, view<[%images]x7xi32>
      view.store %sl32, %rec[%bi, 3] : i32, view<[%images]x7xi32>
      view.store %rows, %rec[%bi, 4] : i32, view<[%images]x7xi32>
      view.store %mo32, %rec[%bi, 5] : i32, view<[%images]x7xi32>
      view.store %eight32, %rec[%bi, 6] : i32, view<[%images]x7xi32>
      %meta = buffer.view %plan[%mo1] : buffer -> view<2xi32>
      view.store %tw, %meta[0] : i32, view<2xi32>
      view.store %th, %meta[1] : i32, view<2xi32>
      %ni2 = index.add %image, %c1 : index
      %true2 = scalar.constant true : i1
      scf.yield %c_c, %ni2, %true2 : offset, index, i1
    }}
    scf.yield %nc, %ni, %nm : offset, index, i1
  }}
  %text = buffer.alloca<private> align(1) %used : buffer
  buffer.copy %scratch[%origin], %text[%origin], %used
  %images64 = index.cast %images : index to i64
  func.return %text, %plan, %images64 : buffer, buffer, i64
}}

// Fills each image record's marker offset with the position of its sentinel in the final fragment.
func.def @vision_locate(%fragment: buffer, %plan: buffer, %count64: i64) {{
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %bounded = scalar.assume %count64 [range(%count64, 0, 4096)] : i64
  %count = index.cast %bounded : i64 to index
  %sentinel = global.load @vision_sentinel : buffer
  %sentinel_length = buffer.length %sentinel
  %end = buffer.length %fragment
  %rec = buffer.view %plan[%origin] : buffer -> view<[%count]x7xi32>
  %final = scf.for %i = [%c0 to %count step %c1](%from = %origin : offset) -> (offset) {{
    %at = func.call @find(%fragment, %from, %end, %sentinel) : (buffer, offset, offset, buffer) -> (offset)
    %found = index.cmp ult, %at, %end : offset
    %f32 = scalar.extui %found : i1 to i32
    %error = global.load @vision_lost : buffer
    func.call @require(%f32, %error) : (i32, buffer)
    %at64 = index.cast %at : offset to i64
    %at32 = scalar.trunci %at64 : i64 to i32
    view.store %at32, %rec[%i, 2] : i32, view<[%count]x7xi32>
    %next = index.add %at, %sentinel_length : offset
    scf.yield %next : offset
  }}
  func.return
}}
global.rodata.def @vision_lost = bytes("{hexs("image marker lost while rendering the message (an image part inside text the chat format rewrites?)")}")

func.def @vision_slice(%text: buffer, %begin: offset, %end: offset) -> (buffer) {{
  %origin = index.constant 0 : offset
  %length = index.sub %end, %begin : offset
  %out = buffer.alloca<private> align(1) %length : buffer
  buffer.copy %text[%begin], %out[%origin], %length
  func.return %out : buffer
}}

// prepare_input with image markers [projection, byte offset, byte length, rows, metadata offset, length]:
// text between markers is encoded as usual; each marker becomes rows placeholder inputs (id 0), and a
// span [projection, input offset, rows] tells the host where its projected rows go.
func.def @vision_prepare_input(%text: buffer, %markers: buffer, %marker_count: i64, %capacity: i64) -> (buffer, i64, buffer) {{
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c0_i32 = scalar.constant 0 : i32
  %c0_i64 = scalar.constant 0 : i64
  %c1_i64 = scalar.constant 1 : i64
  %four = index.constant 4 : offset
  %bounded = scalar.assume %marker_count [range(%marker_count, 1, 4096)] : i64
  %count = index.cast %bounded : i64 to index
  %marks = buffer.view %markers[%origin] : buffer -> view<[%count]x6xi32>
  %text_end = buffer.length %text
  %probe = scalar.addi %capacity, %c1_i64 : i64
  // pass 1: total inputs
  %last, %total0 = scf.for %i = [%c0 to %count step %c1](%cursor = %origin : offset, %sum = %c0_i64 : i64) -> (offset, i64) {{
    %off32 = view.load %marks[%i, 1] : view<[%count]x6xi32> -> i32
    %len32 = view.load %marks[%i, 2] : view<[%count]x6xi32> -> i32
    %rows32 = view.load %marks[%i, 3] : view<[%count]x6xi32> -> i32
    %off64 = scalar.extui %off32 : i32 to i64
    %len64 = scalar.extui %len32 : i32 to i64
    %ob = scalar.assume %off64 [range(%off64, 0, 4294967295)] : i64
    %lb = scalar.assume %len64 [range(%len64, 0, 4294967295)] : i64
    %off = index.cast %ob : i64 to offset
    %len = index.cast %lb : i64 to offset
    %segment = func.call @vision_slice(%text, %cursor, %off) : (buffer, offset, offset) -> (buffer)
    %ids, %n = func.call @encode(%segment, %c0_i32, %probe) : (buffer, i32, i64) -> (buffer, i64)
    %rows64 = scalar.extui %rows32 : i32 to i64
    %s1 = scalar.addi %sum, %n : i64
    %s2 = scalar.addi %s1, %rows64 : i64
    %next = index.add %off, %len : offset
    scf.yield %next, %s2 : offset, i64
  }}
  %tail = func.call @vision_slice(%text, %last, %text_end) : (buffer, offset, offset) -> (buffer)
  %tail_ids, %tail_n = func.call @encode(%tail, %c0_i32, %probe) : (buffer, i32, i64) -> (buffer, i64)
  %total = scalar.addi %total0, %tail_n : i64
  %fits = scalar.cmpi ule, %total, %capacity : i64
  %fits32 = scalar.extui %fits : i1 to i32
  %error = global.load @overflow : buffer
  func.call @require(%fits32, %error) : (i32, buffer)
  %tb = scalar.assume %total [range(%total, 0, 1073741823)] : i64
  %total_index = index.cast %tb : i64 to index
  %token_bytes = index.scale %total_index, %four : index, offset -> offset
  %tokens = buffer.alloca<private> align(4) %token_bytes : buffer
  buffer.fill %c0_i32, %tokens[%origin], %token_bytes : i32
  %c12 = index.constant 12 : offset
  %span_bytes = index.scale %count, %c12 : index, offset -> offset
  %spans = buffer.alloca<private> align(4) %span_bytes : buffer
  %span_view = buffer.view %spans[%origin] : buffer -> view<[%count]x3xi32>
  // pass 2: write ids and spans
  %last2, %out_end = scf.for %i = [%c0 to %count step %c1](%cursor = %origin : offset, %out = %origin : offset) -> (offset, offset) {{
    %proj = view.load %marks[%i, 0] : view<[%count]x6xi32> -> i32
    %off32 = view.load %marks[%i, 1] : view<[%count]x6xi32> -> i32
    %len32 = view.load %marks[%i, 2] : view<[%count]x6xi32> -> i32
    %rows32 = view.load %marks[%i, 3] : view<[%count]x6xi32> -> i32
    %off64 = scalar.extui %off32 : i32 to i64
    %len64 = scalar.extui %len32 : i32 to i64
    %ob = scalar.assume %off64 [range(%off64, 0, 4294967295)] : i64
    %lb = scalar.assume %len64 [range(%len64, 0, 4294967295)] : i64
    %off = index.cast %ob : i64 to offset
    %len = index.cast %lb : i64 to offset
    %segment = func.call @vision_slice(%text, %cursor, %off) : (buffer, offset, offset) -> (buffer)
    %ids, %n = func.call @encode(%segment, %c0_i32, %probe) : (buffer, i32, i64) -> (buffer, i64)
    %nb = scalar.assume %n [range(%n, 0, 1073741823)] : i64
    %ni = index.cast %nb : i64 to index
    %id_bytes = index.scale %ni, %four : index, offset -> offset
    buffer.copy %ids[%origin], %tokens[%out], %id_bytes
    %image_at = index.add %out, %id_bytes : offset
    %image_at64 = index.cast %image_at : offset to i64
    %four64 = scalar.constant 4 : i64
    %input_offset64 = scalar.divui %image_at64, %four64 : i64
    %input_offset = scalar.trunci %input_offset64 : i64 to i32
    view.store %proj, %span_view[%i, 0] : i32, view<[%count]x3xi32>
    view.store %input_offset, %span_view[%i, 1] : i32, view<[%count]x3xi32>
    view.store %rows32, %span_view[%i, 2] : i32, view<[%count]x3xi32>
    %rows64 = scalar.extui %rows32 : i32 to i64
    %rb = scalar.assume %rows64 [range(%rows64, 0, 1073741823)] : i64
    %ri = index.cast %rb : i64 to index
    %row_bytes = index.scale %ri, %four : index, offset -> offset
    %after = index.add %image_at, %row_bytes : offset
    %next = index.add %off, %len : offset
    scf.yield %next, %after : offset, offset
  }}
  %tail_b = scalar.assume %tail_n [range(%tail_n, 0, 1073741823)] : i64
  %tail_i = index.cast %tail_b : i64 to index
  %tail_bytes = index.scale %tail_i, %four : index, offset -> offset
  buffer.copy %tail_ids[%origin], %tokens[%out_end], %tail_bytes
  func.return %tokens, %total, %spans : buffer, i64, buffer
}}
'''
