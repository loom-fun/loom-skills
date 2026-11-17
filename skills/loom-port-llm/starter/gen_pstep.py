"""Persistent / prologue-fused variants of the int8-activation decode matvec (gen_llm._mv, iq=True) for the drafted verify step.

Opt-in (choice decode.mv.persist, env MV_PERSIST; the AMD MXFP4 node on rdna4); the GGUF fleet never sees these kernels.
Every kernel here is a text transform of the kernel gen_llm._mv generates (the same weight decode, integer dots, butterfly
and partial fold, so each output row is summed exactly as the source kernel sums it), with three independent changes:

  persistent row loop (G)   the grid is min(G, row groups) workgroups and each workgroup loops over row groups
                            vg = id, id + G, ...: the activation is loaded (and, with a prologue, produced) once per
                            workgroup instead of once per row group, and the dispatch has G workgroups instead of thousands.
  several matrices (mats)   row groups [0, rg1) are matrix 1, [rg1, rg1 + rg2) matrix 2 (same input, e.g. gate + up,
                            qkv + z): one launch for projections that read one activation.
  activation prologue       the int8 codes (g_quant_q8's layout and arithmetic) are produced in the lane that consumes
                            them, from:  f32  an f32 row (replaces g_quant_q8),
                                         silu silu(f) * u of two f32 rows (replaces g_silu_mul + g_quant_q8),
                                         norm rmsnorm(x) * w of the residual row (replaces g_post_add_norm's h + g_quant_q8;
                                              the residual add moves into the producing matvec's accumulate store).
                            The prologue ops are g_quant_q8's / g_silu_mul's / g_post_add_norm's ops in the same order, so
                            the codes are bit-identical (kernel tests compare against the launch chain with equality).

Names: g_pmv_<fmt>_k<K>_t<TG>_<prologue>_r<R>[_m2][_p<G>]; launch args (n, [n2,] tokens, accumulate, x, [u | nw], w, ws,
[w2, ws2,] y [, y2]).
"""
import re

PROLOGUES = ("q8", "f32", "silu", "norm")


def _one(text, anchor, new, what):
    assert text.count(anchor) == 1, f"gen_pstep {what}: the matvec template moved ({anchor.strip()[:60]!r} x{text.count(anchor)})"
    return text.replace(anchor, new, 1)


ROW_LOOP = "  scf.for %r = [%c0 to %rows_per step %c1] {\n"
TAIL = "  kernel.return\n}\n"
ROW_BASE = "  %row_base = index.mul %group_id, %rows_per : index\n"
TOKEN_BLOCK = re.compile(
    r"  %q(\d+), %da\1, %xs\1 = scf\.if %use\1 -> \(vector<32xi8>, f32, f32\) \{\n.*?\n  \} else \{\n"
    r"    scf\.yield %zero32i8, %zero_f, %zero_f : vector<32xi8>, f32, f32\n  \}", re.S)


def quant_lines(j, v, halves=4):
    """g_quant_q8's arithmetic on this lane's 32 values %{v}: codes %q{j}, F16-rounded scale %da{j}, code sum %xs{j}. The
    elementwise part runs on `halves` slices (the same per-element ops, so the same codes) to bound the registers the f32
    division needs; amax (maxnum) and the integer code sum are order-free."""
    n = 32 // halves
    t = [f"""  %qa{j} = vector.absf %{v} : vector<32xf32>
  %qamax{j} = vector.reduce<maxnumf> %qa{j}, %zero_f : vector<32xf32>, f32
  %qd{j} = scalar.divf %qamax{j}, %pq_c127 : f32
  %qnz{j} = scalar.cmpf one, %qamax{j}, %zero_f : f32
  %qdv{j} = vector.splat %qd{j} : vector<{n}xf32>
  %qnzv{j} = vector.splat %qnz{j} : vector<{n}xi1>
  %qz{j} = vector.splatf 0.0 : vector<{n}xf32>"""]
    parts = []
    for k in range(halves):
        t.append(f"""  %qv{j}_{k} = vector.slice %{v}[{k * n}] : vector<32xf32> -> vector<{n}xf32>
  %qsc{j}_{k} = vector.divf %qv{j}_{k}, %qdv{j} : vector<{n}xf32>
  %qrd{j}_{k} = vector.roundf %qsc{j}_{k} : vector<{n}xf32>
  %qrs{j}_{k} = vector.select %qnzv{j}, %qrd{j}_{k}, %qz{j} : vector<{n}xf32>
  %qi{j}_{k} = vector.fptosi %qrs{j}_{k} : vector<{n}xf32> to vector<{n}xi8>
  scf.schedule.fence""")
        parts.append(f"%qi{j}_{k}")
    while len(parts) > 1:
        w = 32 // len(parts)
        nxt = []
        for i in range(0, len(parts), 2):
            c = f"%qc{j}_{len(parts)}_{i}"
            t.append(f"  {c} = vector.concat<0> {parts[i]}, {parts[i + 1]} : vector<{w}xi8>, vector<{w}xi8> -> vector<{2 * w}xi8>")
            nxt.append(c)
        parts = nxt
    t.append(f"""  %q{j} = vector.bitcast {parts[0]} : vector<32xi8> to vector<32xi8>
  %qqs{j} = vector.dot4i<s8s8> %q{j}, %pq_ones8, %zero8i : vector<32xi8>, vector<32xi8>, vector<8xi32>
  %qqsi{j} = vector.reduce<addi> %qqs{j}, %zero_i : vector<8xi32>, i32
  %xs{j} = scalar.sitofp %qqsi{j} : i32 to f32
  %qdh{j} = scalar.fptrunc %qd{j} : f32 to f16
  %da{j} = scalar.extf %qdh{j} : f16 to f32
  scf.schedule.fence""")
    return "\n".join(t)


def _load(j, view, groups, name):
    return f"""  %{name}{j} = scf.if %use{j} -> (vector<32xf32>) {{
    %tk{j} = index.assume %tok{j}_raw [lt(%tok{j}_raw, %tokens)] : index
    %v = vector.load %{view}[%tk{j}, %g, %c0] : view<[%tokens]x{groups}x32xf32> -> vector<32xf32>
    scf.yield %v : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}"""


def prologue(text, mode, groups, eps=None, count=None, hout=False):
    """Replace the int8 activation loads of an iq matvec by the producer computed in the lane (module doc)."""
    if mode == "q8":
        return text
    assert mode in PROLOGUES, mode
    views = ("  %xq_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x" + str(groups) + "x8xi32>\n"
             f"  %xm_k = index.constant {groups * 32} : offset\n"
             "  %xm_off = index.scale %tokens, %xm_k : index, offset -> offset\n"
             "  %xm_view = buffer.view %x[%xm_off] : buffer -> view<[%tokens]x" + str(groups) + "x2xf32>\n")
    new = (f"  %xf_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{groups}x32xf32>\n"
           "  %pq_c127 = scalar.constant 127.0 : f32\n  %pq_ones8 = vector.splati 1 : vector<32xi8>\n")
    if mode == "silu":
        new += f"  %uf_view = buffer.view %u[%zero] : buffer -> view<[%tokens]x{groups}x32xf32>\n"
    if mode == "norm":
        new += (f"  %nw_view = buffer.view %nw[%zero] : buffer -> view<{groups}x32xf32>\n"
                f"  %pn_count = scalar.constant {float(count)!r} : f32\n  %pn_eps = scalar.constant {eps!r} : f32\n")
        if hout:
            new += (f"  %hout_view = buffer.view %hout[%zero] : buffer -> view<[%tokens]x{groups}x32xf32>\n"
                    "  %pn_first = index.cmp eq, %group_id, %c0 : index\n")
    text = _one(text, views, new, "activation views")
    # the zero constants the quant needs exist in every iq kernel (bfly / iq setup)
    for c in ("  %zero_i = scalar.constant 0 : i32\n", "  %zero8i = vector.splati 0 : vector<8xi32>\n"):
        assert c in text, c
    blocks = []

    def sub(m):
        j = int(m.group(1))
        blocks.append(j)
        if mode == "f32":
            return _load(j, "xf_view", groups, "pv_x") + "\n" + quant_lines(j, f"pv_x{j}")
        if mode == "silu":
            return (_load(j, "xf_view", groups, "pv_f") + "\n" + _load(j, "uf_view", groups, "pv_u") + f"""
  %pv_sg{j} = vector.logisticf<afn> %pv_f{j} : vector<32xf32>
  %pv_s{j} = vector.mulf %pv_f{j}, %pv_sg{j} : vector<32xf32>
  %pv_o{j} = vector.mulf %pv_s{j}, %pv_u{j} : vector<32xf32>
""" + quant_lines(j, f"pv_o{j}"))
        # norm: g_post_add_norm's second half (the residual row is final: the producer accumulated into it)
        w = "" if blocks[:-1] else f"""  %pn_w = scf.if %ok -> (vector<32xf32>) {{
    %v = vector.load %nw_view[%g, %c0] : view<{groups}x32xf32> -> vector<32xf32>
    scf.yield %v : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}
"""
        return (_load(j, "xf_view", groups, "pv_x") + "\n" + w + f"""  %pn_sq{j} = vector.mulf %pv_x{j}, %pv_x{j} : vector<32xf32>
  %pn_lo{j} = vector.reduce<addf> %pn_sq{j}, %zero_f : vector<32xf32>, f32
  %pn_sum{j} = kernel.workgroup.reduce<addf> %pn_lo{j} : f32
  %pn_mean{j} = scalar.divf %pn_sum{j}, %pn_count : f32
  %pn_st{j} = scalar.addf %pn_mean{j}, %pn_eps : f32
  %pn_inv{j} = scalar.rsqrtf %pn_st{j} : f32
  %pn_invs{j} = vector.splat %pn_inv{j} : vector<32xf32>
  %pn_xn{j} = vector.mulf %pv_x{j}, %pn_invs{j} : vector<32xf32>
  %pn_h{j} = vector.mulf %pn_xn{j}, %pn_w : vector<32xf32>
""" + (f"""  %pn_hs{j} = scalar.andi %use{j}, %pn_first : i1
  scf.if %pn_hs{j} {{
    %tkh{j} = index.assume %tok{j}_raw [lt(%tok{j}_raw, %tokens)] : index
    vector.store %pn_h{j}, %hout_view[%tkh{j}, %g, %c0] : vector<32xf32>, view<[%tokens]x{groups}x32xf32>
  }}
""" if hout else "") + quant_lines(j, f"pn_h{j}"))
    text = TOKEN_BLOCK.sub(sub, text)
    assert blocks == sorted(blocks) and blocks and blocks[0] == 0, f"gen_pstep prologue: token blocks {blocks}"
    extra = {"silu": "%u: buffer, ", "norm": "%nw: buffer, "}.get(mode, "")
    if extra:
        text = _one(text, "%x: buffer, %w: buffer", "%x: buffer, " + extra + "%w: buffer", "launch args")
    return text


def pmv(text, prologue_mode="q8", mats=1, G=0, groups=None, eps=None, count=None, hout=False):
    """The kernel text of gen_llm._mv (iq, one weight, no token chunks) -> its persistent / multi-matrix / prologue form.
    mats > 1: matrices 1..mats share the input and R (rows per row group; R does not change a row's sum), launch args
    (n, n2.., tokens, accumulate, x, [u | nw], w, ws, w2, ws2, .., y, y2, ..). hout (norm prologue): workgroup 0 also
    stores the normalized f32 rows to a last argument %hout (the f32-activation projections of the layer read them)."""
    name = re.search(r'kernel\.def export\("([^"]+)"\) @', text).group(1)
    m = re.match(r"g_mv_(\w+?)_k(\d+)_t(\d+)_q8_r(\d+)$", name)
    assert m, f"gen_pstep: {name} is not an int8 token-group matvec"
    fmt, K, TG, R = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    groups = groups or K // 32
    assert "%nch = index.constant" not in text and "%w2: buffer" not in text, f"{name}: chunked / two-weight kernels are not handled"
    assert not hout or prologue_mode == "norm"
    new = (f"g_pmv_{fmt}_k{K}_t{TG}_{prologue_mode}_r{R}" + (f"_m{mats}" if mats > 1 else "") + ("_h" if hout else "")
           + (f"_p{G}" if G else ""))
    text = text.replace(f'kernel.def export("{name}") @{name}(', f'kernel.def export("{new}") @{new}(', 1)
    text = prologue(text, prologue_mode, groups, eps, count, hout)
    sfx = [""] + [str(i) for i in range(2, mats + 1)]
    # launch config: the row groups of every matrix, at most G workgroups
    cfg_old = "  %row_groups = index.div %n, %rows_per : index\n"
    if mats > 1:
        text = _one(text, f"@{new}(%n: index, %tokens: index)", f"@{new}(" + "".join(f"%n{s}: index, " for s in sfx) + "%tokens: index)", "config signature")
    rg = "".join(f"  %rg{i + 1} = index.div %n{s}, %rows_per : index\n" for i, s in enumerate(sfx))
    acc = "%rg1"
    for i in range(2, mats + 1):
        rg += f"  %rgs{i} = index.add {acc}, %rg{i} : index\n"
        acc = f"%rgs{i}"
    cfg_new = rg + f"  %row_groups = index.add {acc}, %c0_rg : index\n"
    cfg_new = "  %c0_rg = index.constant 0 : index\n" + cfg_new
    if G:
        cfg_new += f"  %cGp = index.constant {G} : index\n  %grid_rows = index.min %row_groups, %cGp : index\n"
        text = _one(text, "workgroups(%token_groups, %row_groups, %c1)", "workgroups(%token_groups, %grid_rows, %c1)", "grid")
    text = _one(text, cfg_old, cfg_new, "config row groups")
    if mats > 1:
        text = _one(text, "launch(%n_arg: index, %tokens_arg: index,", "launch(" + "".join(f"%n{s}_arg: index, " for s in sfx) + "%tokens_arg: index,", "launch signature")
        text = _one(text, "%w: buffer, %ws: buffer, %y: buffer)", "".join(f"%w{s}: buffer, %ws{s}: buffer, " for s in sfx) + ", ".join(f"%y{s}: buffer" for s in sfx) + ")", "weights")
        nas = re.search(r"  %n = index\.assume %n_arg \[([^\n]*)\] : index\n", text)
        text = text.replace(nas.group(0), nas.group(0) + "".join(f"  %n{s} = index.assume %n{s}_arg [" + nas.group(1).replace("%n_arg", f"%n{s}_arg") + "] : index\n" for s in sfx[1:]), 1)
    if hout:
        text = _one(text, ", %y: buffer)" if mats == 1 else ", %y" + sfx[-1] + ": buffer)", (", %y: buffer" if mats == 1 else ", %y" + sfx[-1] + ": buffer") + ", %hout: buffer)", "hout arg")
    # the row work (row loop, partial fold barrier, stores) becomes the body of the row-group loop
    text = _one(text, ROW_BASE, "", "row base")
    head, rest = text.split(ROW_LOOP, 1)
    body, tail = rest.rsplit(TAIL, 1)
    assert not tail.strip(), f"{name}: text after the kernel"
    body = ROW_LOOP + body

    def ind(s, k=1):
        return "".join("  " * k + ln if ln.strip() else ln for ln in s.splitlines(True))
    loop = "  %c0_rg = index.constant 0 : index\n" + rg + f"  %grid_y = index.add {acc}, %c0_rg : index\n"
    if G:
        loop += f"  %cGs = index.constant {G} : index\n"
    loop += f"  scf.for %vg = [%group_id to %grid_y step {'%cGs' if G else '%grid_y'}] {{\n"

    def mat_body(i):
        s = sfx[i]
        b = body
        if s:
            for a, c in (("@g_mxfp4s_icodes(%w, %ws, ", f"@g_mxfp4s_icodes(%w{s}, %ws{s}, "), ("[lt(%row_value, %n)]", f"[lt(%row_value, %n{s})]"),
                         ("func.call @g_store_row(%y, %tokens, %n, ", f"func.call @g_store_row(%y{s}, %tokens, %n{s}, ")):
                assert b.count(a) == 1, f"{name}: matrix {i + 1} anchor {a}"
                b = b.replace(a, c)
        return b

    def chain(i, depth):
        """row groups of matrices i.. : scf.if vg < end of matrix i { matrix i } else { chain(i + 1) }"""
        base = "%vg" if i == 0 else f"%vgo{i}"
        if i == mats - 1:
            return ind(f"%row_base = index.mul {base}, %rows_per : index\n", depth) + ind(mat_body(i), depth - 1)
        nxt = f"%vgo{i + 1}"
        return (ind(f"%in{i} = index.cmp ult, {base}, %rg{i + 1} : index\nscf.if %in{i} {{\n", depth)
                + ind(f"%row_base = index.mul {base}, %rows_per : index\n", depth + 1) + ind(mat_body(i), depth)
                + ind("} else {\n", depth) + ind(f"{nxt} = index.sub {base}, %rg{i + 1} : index\n", depth + 1)
                + chain(i + 1, depth + 1) + ind("}\n", depth))
    loop += chain(0, 2)
    # the next row group's wave partials overwrite the LDS the stores just read
    loop += "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n  }\n"
    return head + loop + TAIL


# ======================================================================================================================
# The verify program pass (choice decode.verify.fuse, env VERIFY_FUSE): epochs of at most 4 tokens on the int8 matvec path
# take a fused copy of every layer; every other epoch keeps the layer as generated.
# ======================================================================================================================
_PROJ = re.compile(r"kernel\.launch @g_mv_(\w+?)_k(\d+)_t4_q8_r(\d+)\[(%\w+), %t\]\(\4, %t, %zero_i32, (%\w+), (%\w+), (%\w+), (%\w+)\) : "
                   r"\[index, index\]\(index, index, i32, buffer, (view<\d+xi8>), (view<\d+xi8>), buffer\)")
_QUANT = re.compile(r"kernel\.launch @g_quant_q8\[%t, (%\w+)\]\(%t, \1, (%\w+), (%\w+)\)")
_SILU = re.compile(r"kernel\.launch @g_silu_mul\[%t\]\(%t, (%\w+), (%\w+), (%\w+)\)")
_PRE = re.compile(r"kernel\.launch @g_post_add_norm\[%t\]\(%t, %y, %\w+, %x, (%\w+), %h\)")
_DEF = re.compile(r"  %[\w.]+ = (?!.*kernel\.launch)")


def _statements(lines):
    """Top-level statements of a layer's text: single lines, or a line opening regions up to the line that closes them
    (brace count: the generated GEMM branches close at uneven indentation)."""
    out, cur, depth = [], [], 0
    for ln in lines:
        cur.append(ln)
        depth += ln.count("{") - ln.count("}")
        assert depth >= 0, ln
        if depth == 0:
            out.append("".join(cur))
            cur = []
    assert not cur, "unbalanced layer text"
    return out


def fuse_verify(program, src, eps, hidden, G=0, n_of=None, mode="full", norm_src=None):
    """llm_verify with every layer duplicated: scf.if %pz {fused} else {as generated}, %pz = an epoch of at most 4 tokens
    off the GEMM path and not a single token. The fused layer:
      input projections that read the int8 rows of h (qkv + z, q|gate + k + v, gate + up): one g_pmv_*_norm_* launch whose
          prologue is rmsnorm(x) * w + the int8 quant (the g_post_add_norm / g_quant_q8 launches are gone); workgroup 0
          writes h when the layer's f32-activation projections (alpha / beta) read it;
      a projection of a g_quant_q8 output (attention / GDN out_proj): the f32 prologue, accumulate into x (x += o W^T is
          g_post_add_norm's x += y with the same f32 add);
      the down projection: the silu prologue (g_silu_mul + g_quant_q8 gone), accumulate into x; the last layer stores y
          (the program's final g_post_add_norm adds it).
    mode "swq": the layer stays as generated except gate + up + g_silu_mul + g_quant_q8 of the FFN, which become one
    swiglu_q8 launch (src must also give the 32-row kernel) writing the down projection's int8 input; with norm_src (the
    text of g_post_add_norm) every g_post_add_norm + g_quant_q8(h) pair becomes one g_post_add_norm_q8 launch.
    src(name) -> kernel text of the int8 matvec `name` (gen_llm.mv). n_of(view type, K) -> rows. Returns (program, kernels)."""
    start = program.index("@llm_verify()")
    end = program.index("\n}\n", start)
    body = program[start:end]
    marks = list(re.finditer(r"^  // ---- layer (\d+)[^\n]*\n", body, re.M))
    assert marks, "llm_verify has no layer markers"
    last_layer = len(marks) - 1
    # the final norm (x += y of the last layer, h = rmsnorm(x) * output norm) is the first top-level post_add_norm after the last layer
    tail_at = re.compile(r"^  kernel\.launch @g_post_add_norm\[%t\]\(%t, %y, ", re.M).search(body, marks[-1].end()).start()
    kernels = {}
    first_norm = re.search(r"kernel\.launch @g_embed\[%t\]\([^)]*?, (%\w+), %x, %h\)", body)
    assert first_norm, "g_embed launch"
    out = [body[:marks[0].start()]]
    out.append("  %pz_c4 = index.constant 4 : index\n  %pz_le4 = index.cmp ule, %t, %pz_c4 : index\n"
               "  %pz_true = scalar.constant true : i1\n  %pz_a = scalar.xori %mm, %pz_true : i1\n"
               "  %pz_b = scalar.xori %single, %pz_true : i1\n  %pz_ab = scalar.andi %pz_a, %pz_b : i1\n"
               "  %pz = scalar.andi %pz_ab, %pz_le4 : i1\n  %pz_one = scalar.constant 1 : i32\n")
    for li, mk in enumerate(marks):
        seg_end = marks[li + 1].start() if li + 1 < len(marks) else tail_at
        text = body[mk.end():seg_end]
        stmts = _statements(text.splitlines(True))
        hoist = [s for s in stmts if "\n" == s[-1] and s.count("\n") == 1 and _DEF.match(s)]
        rest = [s for s in stmts if s not in hoist]
        if mode == "swq":
            new = _swq_layer(rest, src, kernels, li, norm_src)
            ind = lambda s: "".join("  " + ln if ln.strip() else ln for ln in s.splitlines(True))
            out.append(mk.group(0) + "".join(hoist) + "  scf.if %pz {\n" + ind("".join(new)) + "  } else {\n" + ind("".join(rest)) + "  }\n")
            continue
        norm_w, quant_of, silu, fused, groups = (first_norm.group(1) if li == 0 else None), {}, None, [], {}
        reads_h = any(re.search(r"%zero_i32, %h, ", s) for s in rest)
        for s in rest:
            pm, qm, sm, mm = _PRE.search(s), _QUANT.search(s), _SILU.search(s), _PROJ.search(s)
            if pm:
                norm_w = pm.group(1)
                continue
            if qm and "g_mv_" not in s:
                if qm.group(2) == "%h":
                    continue
                quant_of[qm.group(3)] = qm.group(2)
                continue
            if sm:
                silu = (sm.group(1), sm.group(2), sm.group(3))
                continue
            if mm:
                fmt, K, R, n, xin, w, ws, y, wt, wst = mm.groups()
                if xin == "%h16q" or xin == "%hq":
                    assert norm_w, f"layer {li}: a projection of h without its norm"
                    group = groups.get(norm_w)
                    if group is None or group["K"] != int(K):
                        group = groups[norm_w] = {"norm": norm_w, "K": int(K), "R": int(R), "fmt": fmt, "mats": []}
                        fused.append(group)
                    group["mats"].append((n, w, ws, y, wt, wst))
                    continue
                src_x = quant_of.get(xin)
                assert src_x, f"layer {li}: {xin} has no producer in the layer"
                if silu and src_x == silu[1]:
                    mode, xs = "silu", f"{silu[1]}, {silu[2]}"
                else:
                    mode, xs = "f32", src_x
                down_last = mode == "silu" and li == last_layer
                kname, ktext = _kernel(src, fmt, int(K), int(R), mode, 1, G, eps, hidden, False)
                kernels[kname] = ktext
                acc, dst = ("%zero_i32", y) if down_last else ("%pz_one", "%x")
                xt = "buffer, buffer" if mode == "silu" else "buffer"
                fused.append(f"  kernel.launch @{kname}[{n}, %t]({n}, %t, {acc}, {xs}, {w}, {ws}, {dst}) : [index, index](index, index, i32, {xt}, {wt}, {wst}, buffer)\n")
                continue
            fused.append(s)
        new = []
        for f in fused:
            if isinstance(f, str):
                new.append(f)
                continue
            mats = f["mats"]
            for n, w, ws, y, wt, wst in mats:
                if n_of:
                    assert n_of(wt, f["K"]) % f["R"] == 0, f"layer {li}: {n} rows are not a multiple of R={f['R']}"
            hout = reads_h and f is next(g for g in fused if not isinstance(g, str))
            kname, ktext = _kernel(src, f["fmt"], f["K"], f["R"], "norm", len(mats), G, eps, hidden, hout)
            kernels[kname] = ktext
            ns = ", ".join(m[0] for m in mats)
            args = ", ".join(f"{m[1]}, {m[2]}" for m in mats) + ", " + ", ".join(m[3] for m in mats) + (", %h" if hout else "")
            tys = ", ".join(f"{m[4]}, {m[5]}" for m in mats) + ", " + ", ".join("buffer" for _ in mats) + (", buffer" if hout else "")
            idx = ", ".join("index" for _ in range(len(mats) + 1))
            new.append(f"  kernel.launch @{kname}[{ns}, %t]({ns}, %t, %zero_i32, %x, {f['norm']}, {args}) : [{idx}]({idx}, i32, buffer, view<{hidden}xf32>, {tys})\n")
        ind = lambda s: "".join("  " + ln if ln.strip() else ln for ln in s.splitlines(True))
        out.append(mk.group(0) + "".join(hoist) + "  scf.if %pz {\n" + ind("".join(new)) + "  } else {\n" + ind("".join(rest)) + "  }\n")
    out.append(body[tail_at:])
    return program[:start] + "".join(out) + program[end:], kernels


def _kernel(src, fmt, K, R, mode, mats, G, eps, hidden, hout):
    base = f"g_mv_{fmt}_k{K}_t4_q8_r{R}"
    t = pmv(src(base), mode, mats=mats, G=G, eps=eps if mode == "norm" else None, count=float(hidden) if mode == "norm" else None, hout=hout)
    return re.search(r'export\("([^"]+)"', t).group(1), t


def swiglu_q8(text):
    """gen_llm._mv text of an int8 matvec with R = 32 rows (one q8 group of the output) -> gate + up of the same rows in one
    workgroup, whose epilogue writes silu(gate) * up (g_silu_mul's ops) straight as g_quant_q8 codes + (d, code sum) of the
    32-row group: g_mv(gate) + g_mv(up) + g_silu_mul + g_quant_q8 in one launch, every code bit-identical.
    Launch args (n, tokens, accumulate (unused), x, w, ws, w2, ws2, fq)."""
    name = re.search(r'kernel\.def export\("([^"]+)"\) @', text).group(1)
    m = re.match(r"g_mv_(\w+?)_k(\d+)_t(\d+)_q8_r32$", name)
    assert m, f"gen_pstep.swiglu_q8: {name} is not a 32-row int8 matvec"
    fmt, K, TG = m.group(1), int(m.group(2)), int(m.group(3))
    new = f"g_pmv_{fmt}_k{K}_t{TG}_swq_r32"
    text = text.replace(f'kernel.def export("{name}") @{name}(', f'kernel.def export("{new}") @{new}(', 1)
    text = _one(text, "%w: buffer, %ws: buffer, %y: buffer)", "%w: buffer, %ws: buffer, %w2: buffer, %ws2: buffer, %y: buffer)", "weights")
    pb = re.search(r"  %part_bytes = index\.constant (\d+) : offset\n", text)
    text = text.replace(pb.group(0), f"  %part_bytes = index.constant {2 * int(pb.group(1))} : offset\n", 1)
    W = re.search(r"view<1x(\d+)x32x" + str(TG) + r"xf32>", text).group(1)
    text = text.replace(f"view<1x{W}x32x{TG}xf32>", f"view<2x{W}x32x{TG}xf32>")
    head, rest = text.split(ROW_LOOP, 1)
    bar = "  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"
    loop, epi = rest.split(bar, 1)
    loop2 = loop
    for a, b in (("@g_mxfp4s_icodes(%w, %ws, ", "@g_mxfp4s_icodes(%w2, %ws2, "), ("%pv[0, ", "%pv[1, ")):
        assert loop2.count(a) == 1, a
        loop2 = loop2.replace(a, b)
    groups = None
    epi_new = f"""  %fl_bytes = index.constant {32 * TG * 4} : offset
  %fl = buffer.alloca<workgroup> align(16) %fl_bytes : buffer
  %flv = buffer.view %fl[%zero] : buffer -> view<{TG}x32xf32>
  %outputs = index.constant {32 * TG} : index
  %lanes_step = index.constant {(K // 32 + 31) // 32 * 32} : index
  scf.for %wr0 = [%workitem to %outputs step %lanes_step] {{
    %wr = index.assume %wr0 [range(%wr0, 0, {32 * TG - 1})] : index
    %rr_v = index.div %wr, %tg : index
    %rr = index.assume %rr_v [range(%rr_v, 0, 31)] : index
    %jj_v = index.rem %wr, %tg : index
    %jj = index.assume %jj_v [range(%jj_v, 0, {TG - 1})] : index
    %t0 = scf.for %ww = [%c0 to %cW step %c1](%acc = %zero_f : f32) -> (f32) {{
      %bw = index.assume %ww [range(%ww, 0, {int(W) - 1})] : index
      %p = view.load %pv[0, %bw, %rr, %jj] : view<2x{W}x32x{TG}xf32> -> f32
      %nx = scalar.addf %acc, %p : f32
      scf.yield %nx : f32
    }}
    %t1 = scf.for %ww = [%c0 to %cW step %c1](%acc = %zero_f : f32) -> (f32) {{
      %bw = index.assume %ww [range(%ww, 0, {int(W) - 1})] : index
      %p = view.load %pv[1, %bw, %rr, %jj] : view<2x{W}x32x{TG}xf32> -> f32
      %nx = scalar.addf %acc, %p : f32
      scf.yield %nx : f32
    }}
    %g1 = vector.from_elements %t0 : vector<1xf32>
    %u1 = vector.from_elements %t1 : vector<1xf32>
    %sg1 = vector.logisticf<afn> %g1 : vector<1xf32>
    %s1 = vector.mulf %g1, %sg1 : vector<1xf32>
    %o1 = vector.mulf %s1, %u1 : vector<1xf32>
    %o = vector.extract %o1[0] : vector<1xf32> -> f32
    view.store %o, %flv[%jj, %rr] : f32, view<{TG}x32xf32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %ctg = index.constant {TG} : index
  %qlane = index.cmp ult, %workitem, %ctg : index
  scf.if %qlane {{
    %jq = index.assume %workitem [range(%workitem, 0, {TG - 1})] : index
    %tokq = index.add %token0, %jq : index
    %liveq = index.cmp ult, %tokq, %tokens : index
    scf.if %liveq {{
      %tq = index.assume %tokq [lt(%tokq, %tokens)] : index
      %fvals = vector.load %flv[%jq, %c0] : view<{TG}x32xf32> -> vector<32xf32>
      %pq_c127 = scalar.constant 127.0 : f32
      %pq_ones8 = vector.splati 1 : vector<32xi8>
{quant_lines(0, "fvals")}
      %c32q = index.constant 32 : index
      %gout = index.div %n, %c32q : index
      %gq = index.div %row_base, %c32q : index
      %gqa = index.assume %gq [lt(%gq, %gout)] : index
      %qv = buffer.view %y[%zero] : buffer -> view<[%tokens]x[%gout]x8xi32>
      %nb = index.cast %n : index to offset
      %moff = index.scale %tokens, %nb : index, offset -> offset
      %mv = buffer.view %y[%moff] : buffer -> view<[%tokens]x[%gout]x2xf32>
      %qw = vector.bitcast %q0 : vector<32xi8> to vector<8xi32>
      vector.store %qw, %qv[%tq, %gqa, %c0] : vector<8xi32>, view<[%tokens]x[%gout]x8xi32>
      %meta = vector.from_elements %da0, %xs0 : vector<2xf32>
      vector.store %meta, %mv[%tq, %gqa, %c0] : vector<2xf32>, view<[%tokens]x[%gout]x2xf32>
    }}
  }}
  kernel.return
}}
"""
    return head + ROW_LOOP + loop + ROW_LOOP + loop2 + bar + epi_new


def _swq_layer(rest, src, kernels, li, norm_src=None):
    """mode swq of fuse_verify: the FFN's gate + up + silu + quant statements -> one swiglu_q8 launch; the rest verbatim."""
    sil = [_SILU.search(s) for s in rest]
    si = [i for i, m in enumerate(sil) if m]
    assert len(si) == 1, f"layer {li}: {len(si)} g_silu_mul launches"
    _, f, fu = sil[si[0]].groups()
    projs = {}
    for i, s in enumerate(rest):
        m = _PROJ.search(s)
        if m and m.group(8) in (f, fu):
            projs[m.group(8)] = (i, m)
    qi = [i for i, s in enumerate(rest) if (q := _QUANT.search(s)) and "g_mv_" not in s and q.group(2) == f]
    assert set(projs) == {f, fu} and len(qi) == 1, f"layer {li}: the FFN's gate / up / quant statements"
    (gi, g), (ui, u) = projs[f], projs[fu]
    assert g.group(5) == u.group(5) and g.group(4) == u.group(4), f"layer {li}: gate and up read different inputs"
    fq = _QUANT.search(rest[qi[0]]).group(3)
    fmt, K = g.group(1), int(g.group(2))
    t = swiglu_q8(src(f"g_mv_{fmt}_k{K}_t4_q8_r32"))
    kname = re.search(r'export\("([^"]+)"', t).group(1)
    kernels[kname] = t
    n, xin = g.group(4), g.group(5)
    launch = (f"  kernel.launch @{kname}[{n}, %t]({n}, %t, %zero_i32, {xin}, {g.group(6)}, {g.group(7)}, {u.group(6)}, {u.group(7)}, {fq}) : "
              f"[index, index](index, index, i32, buffer, {g.group(9)}, {g.group(10)}, {u.group(9)}, {u.group(10)}, buffer)\n")
    drop = {ui, si[0], qi[0]}
    out = [launch if i == gi else s for i, s in enumerate(rest) if i not in drop]
    if norm_src:   # the residual norms write their int8 rows themselves (norm_q8)
        nk = norm_q8(norm_src)
        kernels["g_post_add_norm_q8"] = nk
        pat = re.compile(r"(kernel\.launch @g_post_add_norm)(\[%t\]\(%t, %y, %\w+, %x, %\w+, %h)\) : \[index\]\(index, buffer, (view<\d+xf32>), buffer, (view<\d+xf32>), buffer\)\n"
                         r"(\s*)kernel\.launch @g_quant_q8\[%t, %nH\]\(%t, %nH, %h, (%\w+)\) : [^\n]*\n")
        out = [pat.sub(lambda m: f"{m.group(1)}_q8{m.group(2)}, {m.group(6)}) : [index](index, buffer, {m.group(3)}, buffer, {m.group(4)}, buffer, buffer)\n", s) for s in out]
    return out


def norm_q8(text):
    """g_post_add_norm (x += y; h = rmsnorm(x) * w) -> g_post_add_norm_q8: the lane that writes its 32 values of h also writes
    their g_quant_q8 codes + (d, code sum) to a last argument %hq (the int8 matvecs' input): the g_quant_q8 launch after
    it goes away, every code bit-identical (the same per-lane ops on the same f32 values)."""
    name = "g_post_add_norm"
    text = _one(text, f'kernel.def export("{name}") @{name}(', f'kernel.def export("{name}_q8") @{name}_q8(', "name")
    text = _one(text, "%w_next: buffer, %h: buffer)", "%w_next: buffer, %h: buffer, %hq: buffer)", "args")
    G = int(re.search(r"%h_view = buffer\.view %h\[%zero\] : buffer -> view<\[%tokens\]x(\d+)x32xf32>", text).group(1))
    st = f"    vector.store %hv, %h_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x{G}x32xf32>\n"
    q = "\n".join("    " + ln[2:] if ln.startswith("  ") else ln for ln in quant_lines(0, "hv").splitlines())
    text = _one(text, st, st + f"""    %pq_c127 = scalar.constant 127.0 : f32
    %pq_ones8 = vector.splati 1 : vector<32xi8>
    %zero_i = scalar.constant 0 : i32
    %zero8i = vector.splati 0 : vector<8xi32>
{q}
    %hq_view = buffer.view %hq[%zero] : buffer -> view<[%tokens]x{G}x8xi32>
    %hm_k = index.constant {G * 32} : offset
    %hm_off = index.scale %tokens, %hm_k : index, offset -> offset
    %hm_view = buffer.view %hq[%hm_off] : buffer -> view<[%tokens]x{G}x2xf32>
    %qw0 = vector.bitcast %q0 : vector<32xi8> to vector<8xi32>
    vector.store %qw0, %hq_view[%token, %bl, %c0] : vector<8xi32>, view<[%tokens]x{G}x8xi32>
    %meta0 = vector.from_elements %da0, %xs0 : vector<2xf32>
    vector.store %meta0, %hm_view[%token, %bl, %c0] : vector<2xf32>, view<[%tokens]x{G}x2xf32>
""", "h store")
    return text
