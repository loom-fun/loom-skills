"""Load-time weight preparation for drafting: loom-serve converts a tensor of the published checkpoint into another
format once, at load, through a weight policy (a `data` input of prepare.loom bound to one parameter root). No
converted file is written, so a card keeps referencing the published GGUFs.

Two uses (gen_llm.py wires them, build_draft.py binds the policies):
- drafter (draft_weights.loom, DRAFTER_PREP=1): a DFlash drafter published in BF16/F16 (NVIDIA's Nemotron drafters)
  becomes Q8_0, so its matvecs take the int8-activation path and stream half the bytes.
- proposal head (head_weights.loom, PROPOSE_HEAD=q4_0): the drafter's proposals pick the argmax of the target's head
  over every proposed position, a full extra pass over the head (Nemotron-3.5-Bolt-E4B: 262144 x 2560 Q8_0, 0.71 GB
  of ~4.4 GB per drafting step). A Q4_0 copy halves it. Proposals only feed verification, so their rounding never
  reaches the output (verification runs the target's own head); it can only change which drafts are accepted.

A policy's prepare_weight(key) returns the prepared byte length (0 = keep the file's bytes) and stages one command
program, llm_prep_<src>_<dst> (llm.loom), that reads the staged source tensor (llm.prep_input) and writes the prepared
one: one lane per 32-value block, llama.cpp's quantize_row_q8_0 / quantize_row_q4_0 numerics."""

# - MTP proposal head (head_weights.loom, PROPOSE_HEAD_ROWS=N): the proposals' head pass reads a Q4_0 copy of the Q6_K head
#   restricted to rows [0, N) plus every row of the added tokens (ids >= the tokenizer's base vocabulary): a reduced draft
#   vocabulary (FR-Spec / EAGLE-3 style). The converter dequantizes Q6_K (llama.cpp's dequantize_row_q6_K) and re-encodes
#   Q4_0; the output block g reads input block g (g < split) or g + skip (the tail), so one pass copies both segments.
SRC = {"bf16": 64, "f16": 64, "q8_0": 34, "bf16e": 2, "q2_0x": 36}   # q2_0x: two Q2_0 blocks per 128 values (ternary.prep_kernel)     # bytes per 32 values of a source the converter reads (bf16e: per element)
DST = {"q8_0": 34, "q4_0": 18, "q6_k": 210, "f32_copy": 4, "f32_add1": 4, "f32_negexp": 4, "ptq1_0": 28}   # ptq1_0: per 128 values   # ... of a prepared format (f32_*: per element)
SRC_SUPER = {"q6_k": 210, "bf16r": 512}       # bytes per 256 values of a source with a row subset (see subset_entry); bf16r: a BF16 head (32-value units)

# Elementwise bf16 -> f32 conversions of a safetensors checkpoint's small tensors (one lane per element; safetensors.py's name map
# says which tensor takes which op): copy (conv1d, dt_bias, the gated norm), add1 (the zero-centred RMSNorm weights: w + 1 is
# what llama.cpp's converter writes to the GGUF), negexp (A_log -> -exp(A_log)).
ELEMENT_OPS = {
    "f32_copy": "    %r = scalar.addf %v, %zero_f : f32\n",
    "f32_add1": "    %one_f = scalar.constant 1.0 : f32\n    %r = scalar.addf %v, %one_f : f32\n",
    "f32_negexp": "    %ex = scalar.expf<afn> %v : f32\n    %r = scalar.negf %ex : f32\n",
}


def dst_bytes(dst, blocks):
    return blocks * DST[dst]


def subset_rows(N, base, V):
    """Source rows copied for a proposal head of N leading rows: [0, N) plus [max(base, N), V) (the added tokens);
    N >= V or N >= base copies everything from N on. Returns (head rows, tail first row, rows)."""
    N = min(N, V)
    tail = max(base, N)
    return N, tail, N + V - tail


def subset_entry(src, dst, N, base, V, H):
    """A policy entry (src, dst, out blocks, extra) of a row-subset copy: extra = (source bytes, split block, skip blocks); bf16r:
    (bytes of the head-rows range, split block, 0, byte offset of the tail range, its bytes)."""
    n, tail, rows = subset_rows(N, base, V)
    if src == "bf16r":   # BF16 rows are staged as two ranges (head rows, tail rows): (bytes of the first, split, 0, offset and bytes of the second)
        return (src, dst, rows * H // 32, (n * H * 2, n * H // 32, 0, tail * H * 2, (V - tail) * H * 2))
    unit = 256 if dst == src else 32   # a same-format copy moves whole superblocks
    split = n * H // unit
    skip = tail * H // unit - split
    return (src, dst, rows * H // unit, (V * H // 256 * SRC_SUPER[src], split, skip))


def piece_entries(dst, key, N, base, V, H, piece_rows, prefix):
    """A BF16 head's row subset (subset_rows: [0, N) + the added-token rows; N >= V: every row) copied as several prepared
    tensors of piece_rows rows each (keys prefix0, prefix1, ...; the last piece holds the remainder). hrx-loom stages each prepared
    tensor's input ranges whole through four load-time lanes (weights.h: the staging extent is the LARGEST tensor's staged bytes),
    so one copy of N rows costs 4 x N x 2H bytes of device budget while loading; pieces cost 4 x piece_rows x 2H whatever N is.
    Each piece reads `key` explicitly (its own parameter key is not a checkpoint tensor): entry (bf16r, dst, blocks,
    (len1, split, 0, off2, len2, off1, key)); a piece that straddles the head/tail boundary stages two ranges. Returns
    ({piece key: entry}, [rows per piece])."""
    assert piece_rows % 128 == 0 and H % 32 == 0
    n, tail, rows = subset_rows(N, base, V)
    entries, sizes = {}, []
    for k, a in enumerate(range(0, rows, piece_rows)):
        b = min(rows, a + piece_rows)
        segs = []   # source row ranges of combined rows [a, b): head rows map to themselves, rows >= n to tail + (r - n)
        if a < n:
            segs.append((a, min(b, n)))
        if b > n:
            segs.append((tail + max(a, n) - n, tail + b - n))
        (s0, e0), *rest = segs
        o2, l2 = (rest[0][0] * H * 2, (rest[0][1] - rest[0][0]) * H * 2) if rest else (0, 0)
        entries[f"{prefix}{k}"] = ("bf16r", dst, (b - a) * H // 32,
                                   ((e0 - s0) * H * 2, (e0 - s0) * H // 32, 0, o2, l2, s0 * H * 2, key))
        sizes.append(b - a)
    return entries, sizes


def _decode(src, g="%g", origin="%in_origin"):
    if src in ("bf16", "f16"):
        return f"""    %ib = index.constant 64 : offset
    %io = index.scale {g}, %ib : index, offset -> offset
    %ia = index.add {origin}, %io : offset
    %src_view = buffer.view %input[%ia] : buffer -> view<32x{src}>
    %raw = vector.load %src_view[0] : view<32x{src}> -> vector<32x{src}>
    %v = vector.extf %raw : vector<32x{src}> to vector<32xf32>
"""
    return f"""    %ib = index.constant 34 : offset
    %two_i = index.constant 2 : offset
    %io = index.scale {g}, %ib : index, offset -> offset
    %ia = index.add {origin}, %io : offset
    %sd_view = buffer.view %input[%ia] : buffer -> view<1xf16>
    %sdh = view.load %sd_view[0] : view<1xf16> -> f16
    %sd = scalar.extf %sdh : f16 to f32
    %qa = index.add %ia, %two_i : offset
    %sq_view = buffer.view %input[%qa] : buffer -> view<32xi8>
    %sq = vector.load %sq_view[0] : view<32xi8> -> vector<32xi8>
    %sqf = vector.sitofp %sq : vector<32xi8> to vector<32xf32>
    %sdv = vector.splat %sd : vector<32xf32>
    %v = vector.mulf %sqf, %sdv : vector<32xf32>
"""


def _encode(dst):
    if dst == "q8_0":   # d = amax / 127 (F16), q = round(x / d)
        return """    %zero_f = scalar.constant 0.0 : f32
    %c127 = scalar.constant 127.0 : f32
    %zero32 = vector.splatf 0.0 : vector<32xf32>
    %a = vector.absf %v : vector<32xf32>
    %amax = vector.reduce<maxnumf> %a, %zero_f : vector<32xf32>, f32
    %d = scalar.divf %amax, %c127 : f32
    %nz = scalar.cmpf one, %amax, %zero_f : f32
    %dv = vector.splat %d : vector<32xf32>
    %sc = vector.divf %v, %dv : vector<32xf32>
    %rd = vector.roundf %sc : vector<32xf32>
    %nzv = vector.splat %nz : vector<32xi1>
    %rs = vector.select %nzv, %rd, %zero32 : vector<32xf32>
    %q = vector.fptosi %rs : vector<32xf32> to vector<32xi8>
    %dh = scalar.fptrunc %d : f32 to f16
    %ob = index.constant 34 : offset
    %two_o = index.constant 2 : offset
    %oo = index.scale %g, %ob : index, offset -> offset
    %d_view = buffer.view %output[%oo] : buffer -> view<1xf16>
    view.store %dh, %d_view[0] : f16, view<1xf16>
    %qo = index.add %oo, %two_o : offset
    %q_view = buffer.view %output[%qo] : buffer -> view<32xi8>
    vector.store %q, %q_view[0] : vector<32xi8>, view<32xi8>
"""
    # Q4_0: m = the value of largest magnitude, d = m / -8, q = min(15, floor(x / d + 8.5)); byte j holds values
    # j (low nibble) and j + 16 (high nibble)
    return """    %zero_f = scalar.constant 0.0 : f32
    %ninf = scalar.constant -3.0e38 : f32
    %pinf = scalar.constant 3.0e38 : f32
    %cm8 = scalar.constant -8.0 : f32
    %one_f = scalar.constant 1.0 : f32
    %hi = vector.reduce<maxnumf> %v, %ninf : vector<32xf32>, f32
    %lo = vector.reduce<minnumf> %v, %pinf : vector<32xf32>, f32
    %nlo = scalar.negf %lo : f32
    %pos = scalar.cmpf oge, %hi, %nlo : f32
    %m = scf.select %pos, %hi, %lo : f32
    %d = scalar.divf %m, %cm8 : f32
    %nz = scalar.cmpf one, %d, %zero_f : f32
    %inv = scalar.divf %one_f, %d : f32
    %id = scf.select %nz, %inv, %zero_f : f32
    %idv = vector.splat %id : vector<32xf32>
    %off = vector.splatf 8.5 : vector<32xf32>
    %top = vector.splatf 15.0 : vector<32xf32>
    %sv = vector.fmaf %v, %idv, %off : vector<32xf32>
    %cl = vector.minnumf %sv, %top : vector<32xf32>
    %q = vector.fptosi %cl : vector<32xf32> to vector<32xi8>
    %ql = vector.slice %q[0] : vector<32xi8> -> vector<16xi8>
    %qh = vector.slice %q[16] : vector<32xi8> -> vector<16xi8>
    %four = vector.splati 4 : vector<16xi8>
    %qhs = vector.shli %qh, %four : vector<16xi8>
    %packed = vector.ori %ql, %qhs : vector<16xi8>
    %dh = scalar.fptrunc %d : f32 to f16
    %ob = index.constant 18 : offset
    %two_o = index.constant 2 : offset
    %oo = index.scale %g, %ob : index, offset -> offset
    %d_view = buffer.view %output[%oo] : buffer -> view<1xf16>
    view.store %dh, %d_view[0] : f16, view<1xf16>
    %qo = index.add %oo, %two_o : offset
    %q_view = buffer.view %output[%qo] : buffer -> view<16xi8>
    vector.store %packed, %q_view[0] : vector<16xi8>, view<16xi8>
"""


def _element_decode_encode(dst):
    return f"""    %zero_f = scalar.constant 0.0 : f32
    %ib = index.constant 2 : offset
    %io = index.scale %g, %ib : index, offset -> offset
    %ia = index.add %in_origin, %io : offset
    %src_view = buffer.view %input[%ia] : buffer -> view<1xbf16>
    %raw = view.load %src_view[0] : view<1xbf16> -> bf16
    %v = scalar.extf %raw : bf16 to f32
{ELEMENT_OPS[dst]}    %ob = index.constant 4 : offset
    %oo = index.scale %g, %ob : index, offset -> offset
    %o_view = buffer.view %output[%oo] : buffer -> view<1xf32>
    view.store %r, %o_view[0] : f32, view<1xf32>
"""


def _kernel_super(src, dst):
    """g_prep_<src>_<dst> for a superblock source (Q6_K) with a row subset: output unit g reads source unit g below
    llm.prep_split, g + llm.prep_skip from there on. dst Q4_0: a unit is a 32-value block (g_q6_k_values, the trunk's
    dequantizer in llm.loom, then quantize_row_q4_0); dst == src: a 256-value superblock, copied byte for byte."""
    n = f"{src}_{dst}"
    same = dst == src
    unit = "256-value superblock" if same else "32-value block"
    if same:
        body = """    %src_view = buffer.view %input[%origin] : buffer -> view<210xi8>
    %oo = index.scale %g, %sb_bytes : index, offset -> offset
    %dst_view = buffer.view %output[%oo] : buffer -> view<210xi8>
    %c0 = index.constant 0 : index
    %c210 = index.constant 210 : index
    scf.for %k = [%c0 to %c210 step %c1] {
      %kb = index.assume %k [range(%k, 0, 209)] : index
      %byte = view.load %src_view[%kb] : view<210xi8> -> i8
      view.store %byte, %dst_view[%kb] : i8, view<210xi8>
    }
"""
        locate = """    %sb = %sg
"""
    else:
        body = f"""    %v = func.call @g_q6_k_values(%input, %origin, %grp) : (buffer, offset, index) -> (vector<32xf32>)
{_encode(dst)}"""
    sbuf = "%sg" if same else "%sb"
    locate = ("" if same else "    %sb = index.div %sg, %c8 : index\n    %grp = index.rem %sg, %c8 : index\n")
    return f"""
// Load-time weight preparation (draft/prep_weights.py): staged {src.upper()} tensor (input + llm.prep_input) ->
// {dst.upper()} in output, one lane per {unit}; output unit g reads source unit g (g < split) or g + skip.
kernel.def export("g_prep_{n}") @g_prep_{n}(%blocks: index) {{
  %c1 = index.constant 1 : index
  %c63 = index.constant 63 : index
  %c64 = index.constant 64 : index
  %rounded = index.add %blocks, %c63 : index
  %groups = index.div %rounded, %c64 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c64, %c1, %c1) : index
}} launch(%blocks_arg: index, %in64: i64, %split64: i64, %skip64: i64, %output: buffer, %input: buffer) {{
  %blocks = index.assume %blocks_arg [range(%blocks_arg, 1, 1073741824)] : index
  %in_bound = scalar.assume %in64 [range(%in64, 0, 9223372036854775807)] : i64
  %in_origin = index.cast %in_bound : i64 to offset
  %split_bound = scalar.assume %split64 [range(%split64, 0, 1073741824)] : i64
  %split = index.cast %split_bound : i64 to index
  %skip_bound = scalar.assume %skip64 [range(%skip64, 0, 1073741824)] : i64
  %skip = index.cast %skip_bound : i64 to index
  %c1 = index.constant 1 : index
  %c8 = index.constant 8 : index
  %c64 = index.constant 64 : index
  %sb_bytes = index.constant 210 : offset
  %wg = kernel.workgroup.id<x> : index
  %lane = kernel.workitem.id<x> : index
  %base = index.mul %wg, %c64 : index
  %g = index.add %base, %lane : index
  %live = index.cmp ult, %g, %blocks : index
  scf.if %live {{
    %in_tail = index.cmp uge, %g, %split : index
    %shifted = index.add %g, %skip : index
    %sg = scf.select %in_tail, %shifted, %g : index
{locate}    %sbo = index.scale {sbuf}, %sb_bytes : index, offset -> offset
    %origin = index.add %in_origin, %sbo : offset
{body}  }}
  kernel.return
}}

command.def public @llm_prep_{n}() launch(%output: buffer, %input: buffer) {{
  %b64 = config.get @llm.prep_blocks : i64
  %blocks = index.cast %b64 : i64 to index
  %in64 = config.get @llm.prep_input : i64
  %split64 = config.get @llm.prep_split : i64
  %skip64 = config.get @llm.prep_skip : i64
  kernel.launch @g_prep_{n}[%blocks](%blocks, %in64, %split64, %skip64, %output, %input) : [index](index, i64, i64, i64, buffer, buffer)
  command.return
}}
"""


def _kernel_bf16r(dst):
    """g_prep_bf16r_<dst>: a row subset of a BF16 tensor staged as two ranges (the head rows [0, N) at llm.prep_input, the added-token
    rows at the origin llm.prep_skip carries): output block g reads block g of the first range below llm.prep_split, block g - split of
    the second from there on. Only the bytes the copy needs are staged (0.85 GB of the 2.54 GB head at N = 81920)."""
    n = f"bf16r_{dst}"
    return f"""
// Load-time weight preparation (draft/prep_weights.py): rows [0, N) and the tail rows of a staged BF16 tensor (two staged ranges:
// input + llm.prep_input, and the origin in llm.prep_skip) -> {dst.upper()} in output, one lane per 32-value block.
kernel.def export("g_prep_{n}") @g_prep_{n}(%blocks: index) {{
  %c1 = index.constant 1 : index
  %c63 = index.constant 63 : index
  %c64 = index.constant 64 : index
  %rounded = index.add %blocks, %c63 : index
  %groups = index.div %rounded, %c64 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c64, %c1, %c1) : index
}} launch(%blocks_arg: index, %in64: i64, %split64: i64, %in2_64: i64, %output: buffer, %input: buffer) {{
  %blocks = index.assume %blocks_arg [range(%blocks_arg, 1, 1073741824)] : index
  %in_bound = scalar.assume %in64 [range(%in64, 0, 9223372036854775807)] : i64
  %in2_bound = scalar.assume %in2_64 [range(%in2_64, 0, 9223372036854775807)] : i64
  %split_bound = scalar.assume %split64 [range(%split64, 0, 1073741824)] : i64
  %split = index.cast %split_bound : i64 to index
  %c64 = index.constant 64 : index
  %wg = kernel.workgroup.id<x> : index
  %lane = kernel.workitem.id<x> : index
  %base = index.mul %wg, %c64 : index
  %g = index.add %base, %lane : index
  %live = index.cmp ult, %g, %blocks : index
  scf.if %live {{
    %in_tail = index.cmp uge, %g, %split : index
    %gs = scf.select %in_tail, %g, %split : index
    %rel = index.sub %gs, %split : index
    %tg = scf.select %in_tail, %rel, %g : index
    %o64 = scf.select %in_tail, %in2_bound, %in_bound : i64
    %in_origin = index.cast %o64 : i64 to offset
{_decode("bf16", "%tg") + _encode(dst)}  }}
  kernel.return
}}

command.def public @llm_prep_{n}() launch(%output: buffer, %input: buffer) {{
  %b64 = config.get @llm.prep_blocks : i64
  %blocks = index.cast %b64 : i64 to index
  %in64 = config.get @llm.prep_input : i64
  %split64 = config.get @llm.prep_split : i64
  %in2_64 = config.get @llm.prep_skip : i64
  kernel.launch @g_prep_{n}[%blocks](%blocks, %in64, %split64, %in2_64, %output, %input) : [index](index, i64, i64, i64, buffer, buffer)
  command.return
}}
"""


def kernel(src, dst):
    """g_prep_<src>_<dst> (one lane per 32-value block; per element for bf16e) and its stage program llm_prep_<src>_<dst>."""
    if src == "bf16r":
        return _kernel_bf16r(dst)
    if (src, dst) == ("q2_0x", "ptq1_0"):   # Ternary Bonsai: Q2_0 file -> dense trits on the device
        import ternary
        return ternary.prep_kernel()
    if src in SRC_SUPER:
        return _kernel_super(src, dst)
    n = f"{src}_{dst}"
    return f"""
// Load-time weight preparation (draft/prep_weights.py): staged {src.upper()} tensor (input + llm.prep_input) ->
// {dst.upper()} blocks in output, one lane per 32-value block.
kernel.def export("g_prep_{n}") @g_prep_{n}(%blocks: index) {{
  %c1 = index.constant 1 : index
  %c63 = index.constant 63 : index
  %c64 = index.constant 64 : index
  %rounded = index.add %blocks, %c63 : index
  %groups = index.div %rounded, %c64 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c64, %c1, %c1) : index
}} launch(%blocks_arg: index, %in64: i64, %output: buffer, %input: buffer) {{
  %blocks = index.assume %blocks_arg [range(%blocks_arg, 1, 1073741824)] : index
  %in_bound = scalar.assume %in64 [range(%in64, 0, 9223372036854775807)] : i64
  %in_origin = index.cast %in_bound : i64 to offset
  %c64 = index.constant 64 : index
  %wg = kernel.workgroup.id<x> : index
  %lane = kernel.workitem.id<x> : index
  %base = index.mul %wg, %c64 : index
  %g = index.add %base, %lane : index
  %live = index.cmp ult, %g, %blocks : index
  scf.if %live {{
{_element_decode_encode(dst) if src == "bf16e" else _decode(src) + _encode(dst)}  }}
  kernel.return
}}

command.def public @llm_prep_{n}() launch(%output: buffer, %input: buffer) {{
  %b64 = config.get @llm.prep_blocks : i64
  %blocks = index.cast %b64 : i64 to index
  %in64 = config.get @llm.prep_input : i64
  kernel.launch @g_prep_{n}[%blocks](%blocks, %in64, %output, %input) : [index](index, i64, buffer, buffer)
  command.return
}}
"""


def library(pairs, have=""):
    """Config declarations + converters for the (src, dst) pairs any policy of the package uses (llm.loom). have: text that
    already holds some of them (another policy's library): those are not emitted twice."""
    if not pairs:
        return ""
    sub = "config.decl @llm.prep_split : i64\n\nconfig.decl @llm.prep_skip : i64\n" if any(s in SRC_SUPER for s, _ in pairs) else ""
    text = ("config.decl @llm.prep_blocks : i64\n\nconfig.decl @llm.prep_input : i64\n" + sub
            + "".join(kernel(s, d) for s, d in sorted(pairs) if f'export("g_prep_{s}_{d}")' not in have))
    for c in ("blocks", "input", "split", "skip"):
        if f"config.decl @llm.prep_{c} : i64" in have:
            text = text.replace(f"config.decl @llm.prep_{c} : i64\n\n", "", 1).replace(f"config.decl @llm.prep_{c} : i64\n", "", 1)
    return text


def policy(entries, what):
    """A weight policy file: entries {key: (src, dst, blocks)}; every other key keeps its file bytes. A row-subset entry
    (subset_entry) is (src, dst, blocks, (source bytes, split, skip)): the staged length and llm.prep_split / llm.prep_skip."""
    hx = lambda s: s.encode().hex()
    SUB = any(len(e) > 3 for e in entries.values())
    SUB2 = any(len(e) > 3 and len(e[3]) > 3 for e in entries.values())   # bf16r: a second staged range (llm.prep_skip carries its origin)
    SUB3 = any(len(e) > 3 and len(e[3]) > 5 for e in entries.values())   # piece_entries: the first range's offset + an explicit source key
    skeys = sorted({e[3][6] for e in entries.values() if len(e) > 3 and len(e[3]) > 5})
    pairs = sorted({e[:2] for e in entries.values()})
    names = sorted(entries)
    rod = "".join(f'global.rodata.def @root{j} = bytes("{hx(f"llm_prep_{s}_{d}")}")\n\n' for j, (s, d) in enumerate(pairs))
    if SUB:
        rod += f'global.rodata.def @split_key = bytes("{hx("llm.prep_split")}")\n\nglobal.rodata.def @skip_key = bytes("{hx("llm.prep_skip")}")\n\n'
    rod += f'global.rodata.def @blocks_key = bytes("{hx("llm.prep_blocks")}")\n\nglobal.rodata.def @input_key = bytes("{hx("llm.prep_input")}")\n\n'
    rod += "".join(f'// "{n}"\nglobal.rodata.def @k{i} = bytes("{hx(n)}")\n\n' for i, n in enumerate(names))
    rod += "".join(f'// source "{n}"\nglobal.rodata.def @s{i} = bytes("{hx(n)}")\n\n' for i, n in enumerate(skeys))
    body = ["  %zero = scalar.constant 0 : i64\n  %blocks_0 = scalar.constant 0 : i64\n  %kind_0 = scalar.constant 0 : i64\n"
            "  %inb_0 = scalar.constant 0 : i64\n  %outb_0 = scalar.constant 0 : i64\n"
            + ("  %len_0 = scalar.constant 0 : i64\n  %split_0 = scalar.constant 0 : i64\n  %skip_0 = scalar.constant 0 : i64\n" if SUB else "")
            + ("  %off2_0 = scalar.constant 0 : i64\n  %len2_0 = scalar.constant 0 : i64\n" if SUB2 else "")
            + ("  %off1_0 = scalar.constant 0 : i64\n" + "".join(f"  %sk{i} = global.load @s{i} : buffer\n" for i in range(len(skeys))) if SUB3 else "")]
    prev_src = "%key"
    o2n, l2n = ("%nso_", "%nsl_") if SUB3 else ("%no2", "%nl2")   # %no2<i> / %nl2<i> collide with %no<2x> beyond 20 entries (only one-entry policies used them)
    for i, n in enumerate(names):
        s, d, b, *ex = entries[n]
        ln, sp, sk, *r2 = ex[0] if ex else (b * SRC.get(s, 0), b, 0)
        o2, l2, *r3 = r2 if r2 else (0, 0)
        o1, skey = r3 if r3 else (0, None)
        body.append(f"  %name{i} = global.load @k{i} : buffer\n"
                    f"  %is{i} = func.call @same(%key, %name{i}) : (buffer, buffer) -> (i1)\n"
                    f"  %nb{i} = scalar.constant {b} : i64\n  %nk{i} = scalar.constant {pairs.index((s, d))} : i64\n"
                    f"  %ni{i} = scalar.constant {SRC.get(s, SRC_SUPER.get(s, 0))} : i64\n  %no{i} = scalar.constant {DST[d]} : i64\n"
                    f"  %blocks_{i + 1} = scf.select %is{i}, %nb{i}, %blocks_{i} : i64\n"
                    f"  %kind_{i + 1} = scf.select %is{i}, %nk{i}, %kind_{i} : i64\n"
                    f"  %inb_{i + 1} = scf.select %is{i}, %ni{i}, %inb_{i} : i64\n"
                    f"  %outb_{i + 1} = scf.select %is{i}, %no{i}, %outb_{i} : i64\n"
                    + (f"  %nl{i} = scalar.constant {ln} : i64\n  %nsp{i} = scalar.constant {sp} : i64\n  %nsk{i} = scalar.constant {sk} : i64\n"
                       f"  %len_{i + 1} = scf.select %is{i}, %nl{i}, %len_{i} : i64\n  %split_{i + 1} = scf.select %is{i}, %nsp{i}, %split_{i} : i64\n"
                       f"  %skip_{i + 1} = scf.select %is{i}, %nsk{i}, %skip_{i} : i64\n" if SUB else "")
                    + (f"  {o2n}{i} = scalar.constant {o2} : i64\n  {l2n}{i} = scalar.constant {l2} : i64\n"
                       f"  %off2_{i + 1} = scf.select %is{i}, {o2n}{i}, %off2_{i} : i64\n  %len2_{i + 1} = scf.select %is{i}, {l2n}{i}, %len2_{i} : i64\n" if SUB2 else "")
                    + (f"  %nfo_{i} = scalar.constant {o1} : i64\n  %off1_{i + 1} = scf.select %is{i}, %nfo_{i}, %off1_{i} : i64\n"
                       + (f"  %src_{i + 1} = scf.select %is{i}, %sk{skeys.index(skey)}, {prev_src} : buffer\n" if skey else "") if SUB3 else ""))
        if SUB3 and skey:
            prev_src = f"%src_{i + 1}"
    N = len(names)
    stage = []
    for j, _ in enumerate(pairs):
        stage.append(f"""    %want{j} = scalar.constant {j} : i64
    %this{j} = scalar.cmpi eq, %kind_{N}, %want{j} : i64
    scf.if %this{j} {{
      %root{j} = global.load @root{j} : buffer
      %stage{j} = func.call @stage(%root{j}, %zero) : (buffer, i64) -> (i32)
      %bk{j} = global.load @blocks_key : buffer
      %ik{j} = global.load @input_key : buffer
      func.call @config_i64(%stage{j}, %bk{j}, %blocks_{N}) : (i32, buffer, i64)
      func.call @config_i64(%stage{j}, %ik{j}, %first) : (i32, buffer, i64)
""" + (f"""      %spk{j} = global.load @split_key : buffer
      %skk{j} = global.load @skip_key : buffer
      func.call @config_i64(%stage{j}, %spk{j}, %split_{N}) : (i32, buffer, i64)
      func.call @config_i64(%stage{j}, %skk{j}, {"%second" if pairs[j][0] == "bf16r" else "%skip_" + str(N)}) : (i32, buffer, i64)
""" if pairs[j][0] in SRC_SUPER else "") + "    }\n")
    second = f"""    %has2 = scalar.cmpi ne, %len2_{N}, %zero : i64
    %second = scf.if %has2 -> (i64) {{
      %s2 = func.call @input(%source, {prev_src}, %zero, %key_length, %off2_{N}, %len2_{N}, %align) : (i32, buffer, i64, i64, i64, i64, i64) -> (i64)
      scf.yield %s2 : i64
    }} else {{
      scf.yield %zero : i64
    }}
""" if SUB2 else ""
    return f"""// Weight policy ({what}; draft/prep_weights.py): the listed tensors of the published checkpoint are staged and
// converted at load by llm_prep_<src>_<dst> (llm.loom); every other tensor keeps its file bytes. No converted
// checkpoint is written or published.

func.decl import("weights") @input(%source: i32, %key: buffer, %key_offset: i64, %key_length: i64, %offset: i64, %length: i64, %alignment: i64) -> (i64)

func.decl import("prepare") @stage(%root: buffer, %tag: i64) -> (i32)

func.decl import("prepare") @config_i64(%stage: i32, %key: buffer, %value: i64)

{rod}func.def @same(%key: buffer, %name: buffer) -> (i1) {{
  %a = buffer.length %key
  %b = buffer.length %name
  %sized = index.cmp eq, %a, %b : offset
  %result = scf.if %sized -> (i1) {{
    %o = index.constant 0 : offset
    %order = buffer.compare %key[%o], %name[%o], %b
    %z = scalar.constant 0 : i32
    %same = scalar.cmpi eq, %order, %z : i32
    scf.yield %same : i1
  }} else {{
    %no = scalar.constant false : i1
    scf.yield %no : i1
  }}
  func.return %result : i1
}}

func.def public @prepare_weight(%key: buffer) -> (i64) {{
{"".join(body)}  %found = scalar.cmpi ne, %blocks_{N}, %zero : i64
  %bytes = scf.if %found -> (i64) {{
    %source = scalar.constant 0 : i32
    %align = scalar.constant 256 : i64
    {"%length = scalar.muli %blocks_" + str(N) + ", %inb_" + str(N) + " : i64" if not SUB else "%length = scalar.addi %len_" + str(N) + ", %zero : i64"}
    %key_bytes = buffer.length {prev_src}
    %key_length = index.cast %key_bytes : offset to i64
    %first = func.call @input(%source, {prev_src}, %zero, %key_length, {"%off1_" + str(N) if SUB3 else "%zero"}, %length, %align) : (i32, buffer, i64, i64, i64, i64, i64) -> (i64)
{second}{"".join(stage)}    %prepared = scalar.muli %blocks_{N}, %outb_{N} : i64
    scf.yield %prepared : i64
  }} else {{
    scf.yield %zero : i64
  }}
  func.return %bytes : i64
}}
"""
