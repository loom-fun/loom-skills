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

SRC = {"bf16": 64, "f16": 64, "q8_0": 34}     # bytes per 32 values of a source the converter reads
DST = {"q8_0": 34, "q4_0": 18}                # ... of a prepared format


def dst_bytes(dst, blocks):
    return blocks * DST[dst]


def _decode(src):
    if src in ("bf16", "f16"):
        return f"""    %ib = index.constant 64 : offset
    %io = index.scale %g, %ib : index, offset -> offset
    %ia = index.add %in_origin, %io : offset
    %src_view = buffer.view %input[%ia] : buffer -> view<32x{src}>
    %raw = vector.load %src_view[0] : view<32x{src}> -> vector<32x{src}>
    %v = vector.extf %raw : vector<32x{src}> to vector<32xf32>
"""
    return """    %ib = index.constant 34 : offset
    %two_i = index.constant 2 : offset
    %io = index.scale %g, %ib : index, offset -> offset
    %ia = index.add %in_origin, %io : offset
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
    %zero32 = vector.constant 0.0 : vector<32xf32>
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
    %off = vector.constant 8.5 : vector<32xf32>
    %top = vector.constant 15.0 : vector<32xf32>
    %sv = vector.fmaf %v, %idv, %off : vector<32xf32>
    %cl = vector.minnumf %sv, %top : vector<32xf32>
    %q = vector.fptosi %cl : vector<32xf32> to vector<32xi8>
    %ql = vector.slice %q[0] : vector<32xi8> -> vector<16xi8>
    %qh = vector.slice %q[16] : vector<32xi8> -> vector<16xi8>
    %four = vector.constant 4 : vector<16xi8>
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


def kernel(src, dst):
    """g_prep_<src>_<dst> (one lane per 32-value block) and its stage program llm_prep_<src>_<dst>."""
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
{_decode(src)}{_encode(dst)}  }}
  kernel.return
}}

command.program.def public @llm_prep_{n}() launch(%output: buffer, %input: buffer) {{
  %b64 = config.get @llm.prep_blocks : i64
  %blocks = index.cast %b64 : i64 to index
  %in64 = config.get @llm.prep_input : i64
  kernel.launch @g_prep_{n}[%blocks](%blocks, %in64, %output, %input) : [index](index, i64, buffer, buffer)
  command.return
}}
"""


def library(pairs):
    """Config declarations + converters for the (src, dst) pairs any policy of the package uses (llm.loom)."""
    if not pairs:
        return ""
    return ("config.decl @llm.prep_blocks : i64\n\nconfig.decl @llm.prep_input : i64\n"
            + "".join(kernel(s, d) for s, d in sorted(pairs)))


def policy(entries, what):
    """A weight policy file: entries {key: (src, dst, blocks)}; every other key keeps its file bytes."""
    hx = lambda s: s.encode().hex()
    pairs = sorted({(s, d) for s, d, _ in entries.values()})
    names = sorted(entries)
    rod = "".join(f'global.rodata.def @root{j} = bytes("{hx(f"llm_prep_{s}_{d}")}")\n\n' for j, (s, d) in enumerate(pairs))
    rod += f'global.rodata.def @blocks_key = bytes("{hx("llm.prep_blocks")}")\n\nglobal.rodata.def @input_key = bytes("{hx("llm.prep_input")}")\n\n'
    rod += "".join(f'// "{n}"\nglobal.rodata.def @k{i} = bytes("{hx(n)}")\n\n' for i, n in enumerate(names))
    body = ["  %zero = scalar.constant 0 : i64\n  %blocks_0 = scalar.constant 0 : i64\n  %kind_0 = scalar.constant 0 : i64\n"
            "  %inb_0 = scalar.constant 0 : i64\n  %outb_0 = scalar.constant 0 : i64\n"]
    for i, n in enumerate(names):
        s, d, b = entries[n]
        body.append(f"  %name{i} = global.load @k{i} : buffer\n"
                    f"  %is{i} = func.call @same(%key, %name{i}) : (buffer, buffer) -> (i1)\n"
                    f"  %nb{i} = scalar.constant {b} : i64\n  %nk{i} = scalar.constant {pairs.index((s, d))} : i64\n"
                    f"  %ni{i} = scalar.constant {SRC[s]} : i64\n  %no{i} = scalar.constant {DST[d]} : i64\n"
                    f"  %blocks_{i + 1} = scf.select %is{i}, %nb{i}, %blocks_{i} : i64\n"
                    f"  %kind_{i + 1} = scf.select %is{i}, %nk{i}, %kind_{i} : i64\n"
                    f"  %inb_{i + 1} = scf.select %is{i}, %ni{i}, %inb_{i} : i64\n"
                    f"  %outb_{i + 1} = scf.select %is{i}, %no{i}, %outb_{i} : i64\n")
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
    }}
""")
    return f"""// Weight policy ({what}; draft/prep_weights.py): the listed tensors of the published checkpoint are staged and
// converted at load by llm_prep_<src>_<dst> (llm.loom); every other tensor keeps its file bytes. No converted
// checkpoint is written or published.

func.decl import("weights") @input(%source: i32, %key: buffer, %offset: i64, %length: i64, %alignment: i64) -> (i64)

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
    %length = scalar.muli %blocks_{N}, %inb_{N} : i64
    %first = func.call @input(%source, %key, %zero, %length, %align) : (i32, buffer, i64, i64, i64) -> (i64)
{"".join(stage)}    %prepared = scalar.muli %blocks_{N}, %outb_{N} : i64
    scf.yield %prepared : i64
  }} else {{
    scf.yield %zero : i64
  }}
  func.return %bytes : i64
}}
"""
