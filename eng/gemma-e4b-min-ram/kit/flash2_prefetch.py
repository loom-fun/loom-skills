"""Software prefetch of the next K/V tile for the gfx12 flash2s kernels (gen_flash2s.py, FLASH2_PF=1, gfx12 twin only).

The tile loop of gen_flash2s.py stages a tile right before it uses it: every wave of the workgroup runs the paged-KV lookup
(two dependent global loads), the K/V loads, a wait, the LDS stores, a barrier, and only then the matrix work. The 16 waves
of a workgroup are barrier-locked, so the whole workgroup waits on memory together (about 650 instructions and two or three
global round trips per 16-key tile at head_dim 512, 16 of them WMMA).

This pass moves the loads of tile kt + 1 behind the first barrier of tile kt (the tile's own K/V are already in workgroup
memory), keeping the loaded vectors in two loop-carried values that the next iteration stores. The loads are in flight
during the S / softmax / O work of tile kt. The prologue loads tile 0; the peeled last tile (flash2_peel.py) takes its
values from the loop's results; the last loop iteration's prefetch is predicated off (no tile kt + 1).

Applied to the generated kernel text (before the peel), so the gfx11 text is untouched.
"""
import re

HEADER = re.compile(r"^(?P<ind> *)%m_final, %l_final, (?P<outs>[^=]*) = scf\.for %kt = \[%c0 to %key_tiles step %c1\]\((?P<args>.*)\) -> \((?P<types>.*)\) \{$")
INNER = ("kvals", "vvals", "kp", "k_offset", "v_offset", "k_cache", "v_cache", "kh", "vh")


def _suffix(lines, tag):
    names = "|".join(INNER)
    return [re.sub("%(" + names + r")\b", lambda m: "%" + m.group(1) + tag, ln) for ln in lines]


def prefetch(text):
    lines = text.split("\n")
    hi = next(i for i, ln in enumerate(lines) if HEADER.match(ln))
    m = HEADER.match(lines[hi])
    ind = m.group("ind")
    assert lines[hi + 1].strip().startswith("%tile_first = index.madd %kt"), lines[hi + 1]
    assert lines[hi + 2].strip() == "scf.if %st_live {", lines[hi + 2]
    ib = next(i for i in range(hi + 3, len(lines)) if lines[i].strip().startswith("kernel.barrier"))
    assert lines[ib - 1] == ind + "  }", repr(lines[ib - 1])
    stage = lines[hi + 2:ib]                      # scf.if %st_live { ... }
    vt = re.search(r"vector<(\d+)xf16>", "\n".join(stage)).group(0)
    kp = next(i for i, ln in enumerate(stage) if ln.strip().startswith("%key_pos = "))
    kv = next(i for i, ln in enumerate(stage) if ln.strip().startswith("%key_valid = "))
    l0 = next(i for i, ln in enumerate(stage) if ln.strip().startswith("%kvals, %vvals = scf.if %key_valid"))
    l1 = next(i for i in range(l0 + 1, len(stage)) if stage[i] == ind + "    }")
    limit = re.search(r"index\.cmp ule, %key_pos, (%\w+)", stage[kv]).group(1)
    zeros = re.search(r"scf\.yield (%kz), (%kz)", "\n".join(stage[l0:l1 + 1])).group(1)
    load = stage[l0:l1 + 1]
    rest = [ln.replace("%kvals", "%pfk").replace("%vvals", "%pfv") for ln in stage[l1 + 1:]]   # the stores
    new_stage = [stage[0]] + rest

    def emit(tag, first, extra, pad):
        o = [f"{pad}%key_pos{tag} = index.add {first}, %st_key : index",
             f"{pad}%key_valid{tag}_0 = index.cmp ule, %key_pos{tag}, {limit} : index"]
        valid = f"%key_valid{tag}_0"
        if extra:
            o.append(f"{pad}%key_valid{tag} = scalar.andi %key_valid{tag}_0, {extra} : i1")
            valid = f"%key_valid{tag}"
        body = _suffix(load, tag)
        body = [re.sub(r"%key_pos\b", f"%key_pos{tag}", ln).replace("scf.if %key_valid ", f"scf.if {valid} ") for ln in body]
        o += body
        return o

    # the prologue: tile 0
    pro = [f"{ind}%pfk_p0, %pfv_p0 = scf.if %st_live -> ({vt}, {vt}) {{"]
    pro += ["  " + ln for ln in emit("_p0", "%key_low", None, ind + "  ")]
    pro += [f"{ind}  scf.yield %kvals_p0, %vvals_p0 : {vt}, {vt}", f"{ind}}} else {{", f"{ind}  scf.yield {zeros}, {zeros} : {vt}, {vt}", f"{ind}}}"]
    # the next tile's loads, behind the first barrier
    nxt = [f"{ind}%pf_nt = index.add %kt, %c1 : index",
           f"{ind}%pf_has = index.cmp ult, %pf_nt, %key_tiles : index",
           f"{ind}%pf_first = index.madd %pf_nt, %ckt, %key_low : index",
           f"{ind}%pfk_n, %pfv_n = scf.if %st_live -> ({vt}, {vt}) {{"]
    nxt += ["  " + ln for ln in emit("_n", "%pf_first", "%pf_has", ind + "  ")]
    nxt += [f"{ind}  scf.yield %kvals_n, %vvals_n : {vt}, {vt}", f"{ind}}} else {{", f"{ind}  scf.yield {zeros}, {zeros} : {vt}, {vt}", f"{ind}}}"]
    # the loop
    head = (f"{ind}%m_final, %l_final, {m.group('outs').strip()}, %pfk_f, %pfv_f = scf.for %kt = [%c0 to %key_tiles step %c1]"
            f"({m.group('args')}, %pfk = %pfk_p0 : {vt}, %pfv = %pfv_p0 : {vt}) -> ({m.group('types')}, {vt}, {vt}) {{")
    yi = next(i for i in range(ib, len(lines)) if lines[i].lstrip().startswith("scf.yield %m_new, %l_new,"))
    y = lines[yi]
    y = re.sub(r"^(.*?) : (f32, f32, .*)$", lambda mm: f"{mm.group(1)}, %pfk_n, %pfv_n : {mm.group(2)}, {vt}, {vt}", y)
    out = lines[:hi] + pro + [head, lines[hi + 1]] + new_stage + lines[ib:ib + 1] + nxt + lines[ib + 1:yi] + [y] + lines[yi + 1:]
    return "\n".join(out)
