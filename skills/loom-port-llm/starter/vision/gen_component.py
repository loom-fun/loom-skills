"""Image component of a text package: the image service exports (text_image_*_v1), the encoder command
program (vision_encode) specialized per processed image size, and the kernels it launches.

Packet (host, v0.1.0): header u32 [version, plane_count, output_rows, metadata_offset, metadata_length,
planes_offset, pixel_offset, packet_length]; 48-byte plane records [codec, width, height, display_w,
display_h, orientation, pixel_offset, pixel_length, pixel_format (3 RGB8 | 4 RGBA8)]; our metadata is
[processed width, processed height] (8 bytes, written by the chat program's vision_content).
prepare_batch prepends a 64-byte device header [src_w, src_h, channels, pixel byte offset, dst_w, dst_h]
to the packed cohort (one image per run). EXIF orientation is ignored and alpha dropped, like
llama.cpp's stb_image RGB load."""
import encoder as EN
import gen_vision as G


def hexs(s):
    return s.encode().hex()


def resize_kernels(filt):
    """Header-driven passes: the source size comes from the device header, the processed size is static."""
    out = []
    for axis in ("h", "v"):
        name = f"v_resize_{filt}_{axis}_hdr"
        if axis == "h":   # [src_h, src_w, cin] (payload @ pixel offset) -> [src_h, dst_w, 3]
            dims = '''  %in_w32 = view.load %hv[0] : view<16xi32> -> i32
  %rows32 = view.load %hv[1] : view<16xi32> -> i32
  %cin32 = view.load %hv[2] : view<16xi32> -> i32
  %pix32 = view.load %hv[3] : view<16xi32> -> i32'''
            count = "%rows_i * %out_n"
        else:             # [src_h, dst_w, 3] -> [dst_h, dst_w, 3]
            dims = '''  %in_w32 = view.load %hv[1] : view<16xi32> -> i32
  %rows32 = scalar.constant 1 : i32
  %cin32 = scalar.constant 3 : i32
  %pix32 = scalar.constant 0 : i32'''
            count = None
        _, base = G.resize_pass(filt, axis, 3)
        base = base.replace(f"v_resize_{filt}_{axis}3", name)
        # replace the launch ABI: (count, in_n, out_n, other, src, dst) -> (out_n, other, hdr, src, dst)
        if axis == "h":   # the header is at the start of the source payload: one binding
            base = base.replace("} launch(%count_arg: index, %in_n_arg: index, %out_n_arg: index, %other_arg: index, %src: buffer, %dst: buffer) {",
                                "} launch(%out_n_arg: index, %other_arg: index, %src: buffer, %dst: buffer) {")
        else:
            base = base.replace("} launch(%count_arg: index, %in_n_arg: index, %out_n_arg: index, %other_arg: index, %src: buffer, %dst: buffer) {",
                                "} launch(%out_n_arg: index, %other_arg: index, %hdr: buffer, %src: buffer, %dst: buffer) {")
        hdr = f'''  %zero_h = index.constant 0 : offset
  %hv = buffer.view {"%src" if axis == "h" else "%hdr"}[%zero_h] : buffer -> view<16xi32>
{dims}
  %in_b = scalar.assume %in_w32 [range(%in_w32, 1, 16384)] : i32
  %in_n = index.cast %in_b : i32 to index
  %rows_b = scalar.assume %rows32 [range(%rows32, 1, 16384)] : i32
  %rows_i = index.cast %rows_b : i32 to index
  %cin_b = scalar.assume %cin32 [range(%cin32, 3, 4)] : i32
  %cin_i = index.cast %cin_b : i32 to index
  %pix_b = scalar.assume %pix32 [range(%pix32, 0, 2147483647)] : i32
  %pix_i = index.cast %pix_b : i32 to index
  %out_n = index.assume %out_n_arg [range(%out_n_arg, 1, 65536)] : index
  %other = index.assume %other_arg [range(%other_arg, 1, 65536)] : index'''
        base = base.replace('''  %count = index.assume %count_arg [range(%count_arg, 1, 268435456)] : index
  %in_n = index.assume %in_n_arg [range(%in_n_arg, 1, 65536)] : index
  %out_n = index.assume %out_n_arg [range(%out_n_arg, 1, 65536)] : index
  %other = index.assume %other_arg [range(%other_arg, 1, 65536)] : index''', hdr)
        if axis == "h":
            base = base.replace("  %c256 = index.constant 256 : index\n  %gid", "  %count = index.mul %rows_i, %out_n : index\n  %c256 = index.constant 256 : index\n  %gid", 1)
            base = base.replace("%ccin = index.constant 3 : index", "%ccin = index.add %cin_i, %c0 : index")
            base = base.replace("%el0 = index.mul %src_idx, %ccin : index", "%el_rel = index.mul %src_idx, %ccin : index\n      %el0 = index.add %el_rel, %pix_i : index")
            # grid: config region gets (out_n, other=max rows) -> workgroups over other*out_n
            base = base.replace("kernel.def export(\"" + name + "\") @" + name + "(%count: index) {",
                                "kernel.def export(\"" + name + "\") @" + name + "(%out_w: index, %max_rows: index) {\n  %count = index.mul %out_w, %max_rows : index")
        else:
            base = base.replace("  %c256 = index.constant 256 : index\n  %gid", "  %count = index.mul %other, %out_n : index\n  %c256 = index.constant 256 : index\n  %gid", 1)
            base = base.replace("kernel.def export(\"" + name + "\") @" + name + "(%count: index) {",
                                "kernel.def export(\"" + name + "\") @" + name + "(%out_h: index, %cols: index) {\n  %count = index.mul %out_h, %cols : index")
        out.append((name, base))
    return out


def copy_kernel():
    return "v_copy", '''kernel.def export("v_copy") @v_copy(%n: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %c255 = index.constant 255 : index
  %n_up = index.add %n, %c255 : index
  %groups = index.div %n_up, %c256 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%n_arg: index, %x: buffer, %y: buffer) {
  %n = index.assume %n_arg [range(%n_arg, 1, 268435456)] : index
  %c256 = index.constant 256 : index
  %gid = kernel.workgroup.id<x> : index
  %lid = kernel.workitem.id<x> : index
  %i = index.madd %gid, %c256, %lid : index
  %live = index.cmp ult, %i, %n : index
  scf.if %live {
    %bi = index.assume %i [lt(%i, %n)] : index
    %zero = index.constant 0 : offset
    %xv = buffer.view %x[%zero] : buffer -> view<[%n]xf32>
    %a = view.load %xv[%bi] : view<[%n]xf32> -> f32
    %yv = buffer.view %y[%zero] : buffer -> view<[%n]xf32>
    view.store %a, %yv[%bi] : f32, view<[%n]xf32>
  }
  kernel.return
}
'''


FILTER = {"gemma4v": "bicubic", "qwen3vl_merger": "bicubic", "muse-glimmer": "lanczos"}


def encoder_program(c, max_side=4096):
    """vision_encode(weights, input, output): config @vision.height / @vision.width = processed size."""
    p = EN.build(c)
    kernels = dict(p.kernels)
    rk = resize_kernels(FILTER[c["family"]])
    for n, s in rk:
        kernels[n] = s
    ck = copy_kernel(); kernels[ck[0]] = ck[1]
    P, m, E = c["patch"], c["merge"], c["E"]
    L = ['''config.decl @vision.height : %value: index where [range(%value, 1, 16384)]
config.decl @vision.width : %value: index where [range(%value, 1, 16384)]

command.program.def public @vision_encode() launch(%weights: buffer, %input: buffer, %output: buffer) {''',
         f'''  %img_h = config.get @vision.height : index
  %img_w = config.get @vision.width : index
  %cP = index.constant {P} : index
  %cm = index.constant {m} : index
  %c1 = index.constant 1 : index
  %c3 = index.constant 3 : index
  %c63 = index.constant 63 : index
  %c64 = index.constant 64 : index
  %c255 = index.constant 255 : index
  %c256 = index.constant 256 : index
  %gh = index.div %img_h, %cP : index
  %gw = index.div %img_w, %cP : index
  %T = index.mul %gh, %gw : index
  %mh = index.div %gh, %cm : index
  %mw = index.div %gw, %cm : index
  %M = index.mul %mh, %mw : index
  %T_up = index.add %T, %c255 : index
  %T_t = index.div %T_up, %c256 : index
  %Tp = index.mul %T_t, %c256 : index
  %M_up = index.add %M, %c63 : index
  %M_t = index.div %M_up, %c64 : index
  %Mp = index.mul %M_t, %c64 : index
  %NH = index.constant {c["NH"]} : index
  %img_px = index.mul %img_h, %img_w : index
  %n_pix = index.mul %img_px, %c3 : index
  %max_rows = index.constant {max_side} : index
  %hdr_bytes = index.constant 64 : offset
  %zero_o = index.constant 0 : offset''']
    sizes = EN.buffer_exprs(c)
    for name, (rows, width, bytes_per) in sizes.items():
        L.append(f"  %{name}_w = index.constant {width} : index")
        L.append(f"  %{name}_n = index.mul %{rows}, %{name}_w : index")
        L.append(f"  %{name}_bpe = index.constant {bytes_per} : offset")
        L.append(f"  %{name}_size = index.scale %{name}_n, %{name}_bpe : index, offset -> offset")
        L.append(f"  %{name} = buffer.alloca<global> align(256) %{name}_size : buffer")
    # resize: input payload -> wide (src_h x dst_w x 3) -> pix (dst_h x dst_w x 3)
    L.append(f'''  %wide_n = index.mul %max_rows, %img_w : index
  %wide_n3 = index.mul %wide_n, %c3 : index
  %wide_bpe = index.constant 1 : offset
  %wide_size = index.scale %wide_n3, %wide_bpe : index, offset -> offset
  %wide = buffer.alloca<global> align(256) %wide_size : buffer
  %pix_size = index.scale %n_pix, %wide_bpe : index, offset -> offset
  %pix = buffer.alloca<global> align(256) %pix_size : buffer
  %pixf_bpe = index.constant 4 : offset
  %pixf_size = index.scale %n_pix, %pixf_bpe : index, offset -> offset
  %pixf = buffer.alloca<global> align(256) %pixf_size : buffer
  kernel.launch @{rk[0][0]}[%img_w, %max_rows](%img_w, %max_rows, %input, %wide) : [index, index](index, index, buffer, buffer)
  kernel.launch @{rk[1][0]}[%img_h, %img_w](%img_h, %img_w, %input, %wide, %pix) : [index, index](index, index, buffer, buffer, buffer)''')
    # weights
    wdecl = {}
    for _, grid, args in p.ops:
        for kind, n in args:
            if kind == "W" and n not in wdecl:
                var = "%w_" + n.replace(".", "_")
                wdecl[n] = var
                L.append(f'  {var} = command.parameter %weights, "{n}" : view<{c["tensors"][n]["bytes"]}xi8>')
    consts = {}
    def scalar(expr):
        """index SSA for a plan scalar expression."""
        if expr in ("T", "Tp", "M", "Mp", "gw", "gh", "n_pix", "NH"):
            return "%" + expr
        if expr == "img_w":
            return "%img_w"
        if expr.isdigit():
            if expr not in consts:
                consts[expr] = f"%k{expr}"
                L.append(f"  %k{expr} = index.constant {expr} : index")
            return consts[expr]
        a, b = expr.split("*")
        key = f"{a}x{b}"
        if key not in consts:
            consts[key] = f"%p_{a}_{b}"
            L.append(f"  %p_{a}_{b} = index.mul {scalar(a)}, {scalar(b)} : index")
        return consts[key]
    for kern, grid, args in p.ops:
        g = [scalar(n) for _, n in grid]
        av, at = [], []
        for kind, n in args:
            if kind == "S":
                av.append(scalar(n)); at.append("index")
            elif kind == "B":
                av.append("%" + n); at.append("buffer")
            else:
                av.append(wdecl[n]); at.append(f'view<{c["tensors"][n]["bytes"]}xi8>')
        L.append(f"  kernel.launch @{kern}[{', '.join(g)}]({', '.join(av)}) : [{', '.join('index' for _ in g)}]({', '.join(at)})")
    L.append(f'''  %out_rows = index.constant {c["out"]} : index
  %final_n = index.mul %M, %out_rows : index
  kernel.launch @v_copy[%final_n](%final_n, %out, %output) : [index](index, buffer, buffer)
  command.return
}}''')
    return "\n".join(kernels.values()) + "\n" + "\n".join(L) + "\n"


def prepare_program(c, default_hw, out_dim, max_rows):
    """Cold preparation: binds the 'vision' parameter scope to the encoder stage."""
    return f'''// Image component preparation (generated by gen_component.py).
func.decl import("prepare") @source_scope(%scope: buffer) -> (i32)
func.decl import("prepare") @parameter(%stage: i32, %binding: i32, %input: i32, %source: i32, %policy: buffer)
func.decl import("prepare") @memory_forecast(%root: buffer)

global.rodata.def @vision_forecast_root = bytes("{hexs("vision_forecast_memory")}")
global.rodata.def @vision_scope = bytes("{hexs("vision")}")
global.rodata.def @vision_policy = bytes("{hexs("weights.loom")}")
global.rodata.def @vision_name = bytes("{hexs(c["family"] + "-projector")}")
global.rodata.def @vision_empty = bytes("")

func.decl @vision_stage(%height: i64, %width: i64) -> (i32)

func.def public @text_image_prepare_v1(%directory: buffer, %options: buffer) -> (i64, i64, i64, i64, buffer, buffer, buffer) {{
  %forecast = global.load @vision_forecast_root : buffer
  func.call @memory_forecast(%forecast) : (buffer)
  %h = scalar.constant {default_hw[1]} : i64
  %w = scalar.constant {default_hw[0]} : i64
  %stage = func.call @vision_stage(%h, %w) : (i64, i64) -> (i32)
  %scope = global.load @vision_scope : buffer
  %checkpoint = func.call @source_scope(%scope) : (buffer) -> (i32)
  %policy = global.load @vision_policy : buffer
  %c0 = scalar.constant 0 : i32
  func.call @parameter(%stage, %c0, %c0, %checkpoint, %policy) : (i32, i32, i32, i32, buffer)
  %units = scalar.constant 1 : i64
  %upload = scalar.constant {4096 * 4096 * 4 + 65536} : i64
  %output = scalar.constant {max_rows * out_dim * 4} : i64
  %state_bytes = index.constant 1 : offset
  %state = buffer.alloca<private> align(1) %state_bytes : buffer
  %empty = global.load @vision_empty : buffer
  %name = global.load @vision_name : buffer
  func.return %units, %upload, %output, %output, %empty, %name, %state : i64, i64, i64, i64, buffer, buffer, buffer
}}
'''


def control_program(c, out_dim, max_rows, scratch_bytes):
    """Request/batch/run callbacks plus the memory forecast."""
    return f'''// Image component control (generated by gen_component.py).
func.decl import("input") @require(%condition: i32, %diagnostic: buffer)
func.decl import("runner") @execute_2(%stage: i32, %input: hal.buffer, %output: hal.buffer) -> (i64)
func.decl @vision_stage(%height: i64, %width: i64) -> (i32)
global.rodata.def @vision_bad_rows = bytes("{hexs("image row count does not match the processed size")}")

// Memory: private = largest packet + encoder scratch at the largest size; retained = output rows.
func.def public @vision_forecast_memory(%state: buffer, %ranges: buffer, %descriptors: buffer) -> (i64, i64, i64) {{
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c0_i64 = scalar.constant 0 : i64
  %c16 = scalar.constant 16 : i64
  %range_bytes = buffer.length %ranges
  %range_length = index.cast %range_bytes : offset to i64
  %count_value = scalar.divui %range_length, %c16 : i64
  %bounded = scalar.assume %count_value [range(%count_value, 1, 2147483647)] : i64
  %count = index.cast %bounded : i64 to index
  %records = buffer.view %ranges[%origin] : buffer -> view<[%count]x2xi64>
  %peak, %rows = scf.for %i = [%c0 to %count step %c1](%p = %c0_i64 : i64, %r = %c0_i64 : i64) -> (i64, i64) {{
    %off = view.load %records[%i, 0] : view<[%count]x2xi64> -> i64
    %ob = scalar.assume %off [range(%off, 0, 9223372036854775807)] : i64
    %o = index.cast %ob : i64 to offset
    %header = buffer.view %descriptors[%o] : buffer -> view<8xi32>
    %len32 = view.load %header[7] : view<8xi32> -> i32
    %rows32 = view.load %header[2] : view<8xi32> -> i32
    %len = scalar.extui %len32 : i32 to i64
    %rr = scalar.extui %rows32 : i32 to i64
    %bigger = scalar.cmpi ugt, %len, %p : i64
    %np = scf.select %bigger, %len, %p : i64
    %nr = scalar.addi %r, %rr : i64
    scf.yield %np, %nr : i64, i64
  }}
  %scratch = scalar.constant {scratch_bytes} : i64
  %private = scalar.addi %peak, %scratch : i64
  %row_bytes = scalar.constant {out_dim * 4} : i64
  %retained = scalar.muli %rows, %row_bytes : i64
  func.return %private, %retained, %retained : i64, i64, i64
}}

// One image per run: the tag is the processed size (metadata [width, height]).
func.def public @text_image_prepare_request_v1(%state: buffer, %request: buffer) -> (buffer, i64, i64) {{
  %origin = index.constant 0 : offset
  %header = buffer.view %request[%origin] : buffer -> view<8xi32>
  %meta32 = view.load %header[3] : view<8xi32> -> i32
  %meta64 = scalar.extui %meta32 : i32 to i64
  %mb = scalar.assume %meta64 [range(%meta64, 0, 4294967295)] : i64
  %mo = index.cast %mb : i64 to offset
  %meta = buffer.view %request[%mo] : buffer -> view<2xi32>
  %w = view.load %meta[0] : view<2xi32> -> i32
  %h = view.load %meta[1] : view<2xi32> -> i32
  %w64 = scalar.extui %w : i32 to i64
  %h64 = scalar.extui %h : i32 to i64
  %shift = scalar.constant 32 : i64
  %wbits = scalar.shli %w64, %shift : i64
  %tag = scalar.ori %wbits, %h64 : i64
  %units = scalar.constant 1 : i64
  func.return %request, %units, %tag : buffer, i64, i64
}}

func.def public @text_image_prepare_batch_v1(%state: buffer, %tags: buffer, %ranges: buffer, %packed: buffer) -> (i32, buffer, buffer) {{
  %origin = index.constant 0 : offset
  %range = buffer.view %ranges[%origin] : buffer -> view<1x2xi64>
  %source64 = view.load %range[0, 0] : view<1x2xi64> -> i64
  %sb = scalar.assume %source64 [range(%source64, 0, 4294967295)] : i64
  %source = index.cast %sb : i64 to offset
  %header = buffer.view %packed[%source] : buffer -> view<8xi32>
  %rows32 = view.load %header[2] : view<8xi32> -> i32
  %meta32 = view.load %header[3] : view<8xi32> -> i32
  %planes32 = view.load %header[5] : view<8xi32> -> i32
  %meta64 = scalar.extui %meta32 : i32 to i64
  %planes64 = scalar.extui %planes32 : i32 to i64
  %mb = scalar.assume %meta64 [range(%meta64, 0, 4294967295)] : i64
  %pb = scalar.assume %planes64 [range(%planes64, 0, 4294967295)] : i64
  %mo0 = index.cast %mb : i64 to offset
  %po0 = index.cast %pb : i64 to offset
  %mo = index.add %source, %mo0 : offset
  %po = index.add %source, %po0 : offset
  %meta = buffer.view %packed[%mo] : buffer -> view<2xi32>
  %dst_w = view.load %meta[0] : view<2xi32> -> i32
  %dst_h = view.load %meta[1] : view<2xi32> -> i32
  %plane = buffer.view %packed[%po] : buffer -> view<12xi32>
  %src_w = view.load %plane[1] : view<12xi32> -> i32
  %src_h = view.load %plane[2] : view<12xi32> -> i32
  %pix_rel = view.load %plane[6] : view<12xi32> -> i32
  %channels = view.load %plane[8] : view<12xi32> -> i32
  // rows check: the chat program declared rows from the same size rule
  %pf = scalar.constant {c["patch"] * c["merge"]} : i32
  %gw = scalar.divui %dst_w, %pf : i32
  %gh = scalar.divui %dst_h, %pf : i32
  %expect = scalar.muli %gw, %gh : i32
  %ok = scalar.cmpi eq, %expect, %rows32 : i32
  %ok32 = scalar.extui %ok : i1 to i32
  %bad = global.load @vision_bad_rows : buffer
  func.call @require(%ok32, %bad) : (i32, buffer)
  // payload: 64-byte device header + the packed cohort
  %hdr_bytes = index.constant 64 : offset
  %packed_bytes = buffer.length %packed
  %payload_bytes = index.add %hdr_bytes, %packed_bytes : offset
  %payload = buffer.alloca<private> align(16) %payload_bytes : buffer
  buffer.copy %packed[%origin], %payload[%hdr_bytes], %packed_bytes
  %hdr = buffer.view %payload[%origin] : buffer -> view<16xi32>
  %source64b = index.cast %source : offset to i64
  %source32 = scalar.trunci %source64b : i64 to i32
  %sixty4 = scalar.constant 64 : i32
  %pix0 = scalar.addi %pix_rel, %source32 : i32
  %pix_abs = scalar.addi %pix0, %sixty4 : i32
  %zero32 = scalar.constant 0 : i32
  view.store %src_w, %hdr[0] : i32, view<16xi32>
  view.store %src_h, %hdr[1] : i32, view<16xi32>
  view.store %channels, %hdr[2] : i32, view<16xi32>
  view.store %pix_abs, %hdr[3] : i32, view<16xi32>
  view.store %dst_w, %hdr[4] : i32, view<16xi32>
  view.store %dst_h, %hdr[5] : i32, view<16xi32>
  // plan: [output bytes, result bytes, 1 request, rows] [first row, rows, 0] then [byte offset, width] per row
  %rows64 = scalar.extui %rows32 : i32 to i64
  %rb = scalar.assume %rows64 [range(%rows64, 1, {max_rows})] : i64
  %rows = index.cast %rb : i64 to index
  %c16 = index.constant 16 : offset
  %c56 = index.constant 56 : offset
  %row_bytes_total = index.scale %rows, %c16 : index, offset -> offset
  %plan_bytes = index.add %c56, %row_bytes_total : offset
  %plan = buffer.alloca<private> align(8) %plan_bytes : buffer
  %ph = buffer.view %plan[%origin] : buffer -> view<7xi64>
  %width = scalar.constant {out_dim} : i64
  %row_bytes = scalar.constant {out_dim * 4} : i64
  %out_bytes = scalar.muli %rb, %row_bytes : i64
  %one = scalar.constant 1 : i64
  %zero = scalar.constant 0 : i64
  view.store %out_bytes, %ph[0] : i64, view<7xi64>
  view.store %out_bytes, %ph[1] : i64, view<7xi64>
  view.store %one, %ph[2] : i64, view<7xi64>
  view.store %rb, %ph[3] : i64, view<7xi64>
  view.store %zero, %ph[4] : i64, view<7xi64>
  view.store %rb, %ph[5] : i64, view<7xi64>
  view.store %zero, %ph[6] : i64, view<7xi64>
  %records = buffer.view %plan[%c56] : buffer -> view<[%rows]x2xi64>
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  scf.for %i = [%c0 to %rows step %c1] {{
    %i64 = index.cast %i : index to i64
    %offset = scalar.muli %i64, %row_bytes : i64
    view.store %offset, %records[%i, 0] : i64, view<[%rows]x2xi64>
    view.store %width, %records[%i, 1] : i64, view<[%rows]x2xi64>
  }}
  %h64 = scalar.extui %dst_h : i32 to i64
  %w64 = scalar.extui %dst_w : i32 to i64
  %stage = func.call @vision_stage(%h64, %w64) : (i64, i64) -> (i32)
  func.return %stage, %payload, %plan : i32, buffer, buffer
}}

func.def public @text_image_run_v1(%stage: i32, %input: hal.buffer, %output: hal.buffer) {{
  %work = func.call @execute_2(%stage, %input, %output) : (i32, hal.buffer, hal.buffer) -> (i64)
  func.return
}}
'''


def stages_library():
    return f'''// Encoder stage per processed image size (shared by preparation and batching).
func.decl import("prepare") @stage(%root: buffer, %tag: i64) -> (i32)
func.decl import("prepare") @config_i64(%stage: i32, %key: buffer, %value: i64)
global.rodata.def @vision_root = bytes("{hexs("vision_encode")}")
global.rodata.def @vision_height_key = bytes("{hexs("vision.height")}")
global.rodata.def @vision_width_key = bytes("{hexs("vision.width")}")

func.def public @vision_stage(%height: i64, %width: i64) -> (i32) {{
  %root = global.load @vision_root : buffer
  %one = scalar.constant 1 : i64
  %stage = func.call @stage(%root, %one) : (buffer, i64) -> (i32)
  %hk = global.load @vision_height_key : buffer
  %wk = global.load @vision_width_key : buffer
  func.call @config_i64(%stage, %hk, %height) : (i32, buffer, i64)
  func.call @config_i64(%stage, %wk, %width) : (i32, buffer, i64)
  func.return %stage : i32
}}

'''
