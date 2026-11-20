"""Double-buffered K/V staging with a one-tile-ahead prefetch for the flash2s kernels (gen_flash2s.py, FLASH2_G11=db; 16-key tiles).

The plain tile loop stages tile t right before it uses it: a paged-KV lookup (map pointer, page entry: two dependent global loads),
the K/V loads, a wait, the workgroup-memory stores, a barrier, and only then the matrix work, so every tile exposes two or three global
round trips to a barrier-locked workgroup; a third barrier at the end of the tile keeps the next tile's stores off the buffers the
slowest wave still reads. Here K and V^T have two buffers (tile t in buffer t % 2):
  * the loads of tile t + 1 are issued right after tile t's first barrier, UNCONDITIONALLY (the key position clamped to the last
    written one, the value selected to zero afterwards for keys past it), so no branch result (phi) forces the wait at once; the
    data is waited for and stored into buffer (t + 1) % 2 at the END of the iteration, after tile t's S, softmax and PV work;
  * the page entry of tile t + 1 is a loop-carried value loaded one iteration earlier (the map pointer is hoisted out of the loop);
  * two barriers per tile instead of three: buffer (t + 1) % 2 was last read in iteration t - 1, before every wave passed tile t's
    first barrier, and the S partial exchange slots are written after the first barrier and read after the second.
The prologue stages tile 0 (the original staging code) and loads tile 1's page entry. The peeled last tile (flash2_peel.py, applied
after this) prefetches a clamped, unused tile. Needs 16-key tiles (K and V^T of two 32-key tiles plus the S slots exceed 64 KiB at
head_dim 256) and the plain paged KV layout (g_kv_offset). Same arithmetic in the same order as the plain loop: bit-identical results.
"""
import re

HEADER = re.compile(r"^(?P<ind> *)%m_final, %l_final, (?P<outs>[^=]*) = scf\.for %kt = \[%c0 to %key_tiles step %c1\]\((?P<args>.*)\) -> \((?P<types>.*)\) \{$")


def double_buffer(text, at="end"):
    lines = text.split("\n")
    hi = next((i for i, ln in enumerate(lines) if HEADER.match(ln)), None)
    assert hi is not None, "flash2_db: no tile loop"
    m = HEADER.match(lines[hi])
    ind = m.group("ind")
    kt = int(re.search(r"%ckt = index\.constant (\d+) : index", text).group(1))
    assert kt == 16, "flash2_db: 16-key tiles only (FLASH2_KT256=16)"
    assert lines[hi + 1].strip().startswith("%tile_first = index.madd %kt"), lines[hi + 1]
    assert lines[hi + 2].strip() == "scf.if %st_live {", lines[hi + 2]
    ib = next(i for i in range(hi + 3, len(lines)) if lines[i].strip().startswith("kernel.barrier"))
    stage = lines[hi + 2:ib]
    blk = "\n".join(stage)
    fn = re.search(r"func\.call @(\w+)\(%origins, %row, %kp, %blocks, %k_plane, %pool\)", blk)
    assert fn, "flash2_db: plain paged KV lookup expected"
    limit = re.search(r"%key_valid = index\.cmp ule, %key_pos, (%\w+) : index", blk).group(1)
    vec = re.search(r"%kvals, %vvals = scf\.if %key_valid -> \((vector<\d+xf16>),", blk).group(1)
    cache_ty = re.search(r"%k_cache = buffer\.view %arena\[%k_offset\] : buffer -> (view<[^>]*>)", blk).group(1)
    kload = re.search(r"%kh = vector\.load %k_cache\[([^\]]*)\]", blk).group(1)
    yi = next(i for i in range(ib, len(lines)) if lines[i].lstrip().startswith("scf.yield %m_new, %l_new,"))
    assert lines[yi - 1].strip().startswith("kernel.barrier"), lines[yi - 1]
    pb = int(re.search(r"%pos_bytes = index\.constant (\d+) : offset", text).group(1))
    kv = re.search(r"%k_view = buffer\.view %k_tile\[%origin\] : buffer -> (view<(\d+)x(\d+)xf16>)", text)
    vv = re.search(r"%vt_view = buffer\.view %vt_tile\[%origin\] : buffer -> (view<(\d+)x(\d+)xf16>)", text)
    kbytes = int(kv.group(2)) * int(kv.group(3)) * 2
    vbytes = int(vv.group(2)) * int(vv.group(3)) * 2
    ktype, vtype = kv.group(1), vv.group(1)

    # the staging stores of a value pair (KV, VV) into views (KVIEW, VTVIEW)
    stores = [ln for ln in stage[stage.index(next(s for s in stage if s.strip().startswith("vector.store %kvals"))):-1]]

    def store_lines(kvn, vvn, kview, vtview, tag):
        out = []
        for ln in stores:
            ln = ln.replace("%kvals", kvn).replace("%vvals", vvn).replace("%k_view[", kview + "[").replace("%vt_view[", vtview + "[")
            ln = re.sub(r"%(j|vj|vt_row_value|vt_row)\b", lambda mm: "%" + mm.group(1) + tag, ln)
            out.append(ln)
        return out

    # --- definitions before the loop: buffer offsets, the hoisted page map, the prologue (tile 0) and tile 1's page entry
    pre = [f"{ind}%db_c2 = index.constant 2 : index",
           f"{ind}%db_kbytes = index.constant {kbytes} : offset",
           f"{ind}%db_vbytes = index.constant {vbytes} : offset",
           f"{ind}%k_view0 = buffer.view %k_tile[%origin] : buffer -> {ktype}",
           f"{ind}%vt_view0 = buffer.view %vt_tile[%origin] : buffer -> {vtype}",
           f"{ind}%db_rec_stride = index.constant 40 : offset",
           f"{ind}%db_row = index.assume %row [range(%row, 0, 2147483646)] : index",
           f"{ind}%db_rec_off = index.scale %db_row, %db_rec_stride : index, offset -> offset",
           f"{ind}%db_records = buffer.view %origins[%db_rec_off] : buffer -> view<5xi64>",
           f"{ind}%db_map64 = view.load %db_records[4] : view<5xi64> -> i64",
           f"{ind}%db_map_b = scalar.assume %db_map64 [range(%db_map64, 0, 9223372036854775807)] : i64",
           f"{ind}%db_map_origin = index.cast %db_map_b : i64 to offset",
           f"{ind}%db_map = buffer.view %origins[%db_map_origin] : buffer -> view<[%blocks]xi32>",
           f"{ind}%db_last_blk = index.sub %blocks, %c1 : index",
           f"{ind}%db_pp = index.mul %k_plane, %pool : index",
           f"{ind}%db_lim_pg = index.div {limit}, %c16 : index"]

    def entry(tile, tag):   # page entry (i32) of the lane's key in tile TILE (clamped to the last written page)
        return [f"{ind}%db_pf{tag} = index.madd {tile}, %ckt, %key_low : index",
                f"{ind}%db_pk{tag} = index.add %db_pf{tag}, %st_key : index",
                f"{ind}%db_pgr{tag} = index.div %db_pk{tag}, %c16 : index",
                f"{ind}%db_pgc{tag} = index.min %db_pgr{tag}, %db_lim_pg : index",
                f"{ind}%db_pg{tag} = index.assume %db_pgc{tag} [range(%db_pgc{tag}, 0, %db_last_blk), lt(%db_pgc{tag}, %blocks)] : index",
                f"{ind}%db_ent{tag} = view.load %db_map[%db_pg{tag}] : view<[%blocks]xi32> -> i32"]

    # prologue: tile 0 staged with the original code (its names suffixed), into buffer 0
    defined = set()
    for ln in stage:
        d = re.match(r"\s*(%\w+(?:, %\w+)*) = ", ln)
        if d:
            defined.update(re.findall(r"%\w+", d.group(1)))
        defined.update(re.findall(r"scf\.for (%\w+) = ", ln))
    mp = {n: n + "_db0" for n in defined}
    mp.update({"%tile_first": "%key_low", "%k_view": "%k_view0", "%vt_view": "%vt_view0"})
    pro = [re.sub(r"%\w+", lambda mm: mp.get(mm.group(0), mm.group(0)), ln) for ln in stage]
    pre += pro
    pre += entry("%c1", "_i")

    # --- the loop body
    body_top = [f"{ind}  %db_par = index.rem %kt, %db_c2 : index",
                f"{ind}  %db_parb = index.assume %db_par [range(%db_par, 0, 1)] : index",
                f"{ind}  %db_parn = index.sub %c1, %db_parb : index",
                f"{ind}  %db_parnb = index.assume %db_parn [range(%db_parn, 0, 1)] : index",
                f"{ind}  %db_ko = index.scale %db_parb, %db_kbytes : index, offset -> offset",
                f"{ind}  %db_vo = index.scale %db_parb, %db_vbytes : index, offset -> offset",
                f"{ind}  %db_kon = index.scale %db_parnb, %db_kbytes : index, offset -> offset",
                f"{ind}  %db_von = index.scale %db_parnb, %db_vbytes : index, offset -> offset",
                f"{ind}  %k_view = buffer.view %k_tile[%db_ko] : buffer -> {ktype}",
                f"{ind}  %vt_view = buffer.view %vt_tile[%db_vo] : buffer -> {vtype}",
                f"{ind}  %db_kvn = buffer.view %k_tile[%db_kon] : buffer -> {ktype}",
                f"{ind}  %db_vvn = buffer.view %vt_tile[%db_von] : buffer -> {vtype}"]
    # after the first barrier: tile kt + 1's K/V loads (unconditional, clamped) from the carried page entry, then tile kt + 2's entry
    loads = [f"{ind}  %db_nt = index.add %kt, %c1 : index",
             f"{ind}  %db_nfirst = index.madd %db_nt, %ckt, %key_low : index",
             f"{ind}  %db_kpos = index.add %db_nfirst, %st_key : index",
             f"{ind}  %db_ok = index.cmp ule, %db_kpos, {limit} : index",
             f"{ind}  %db_kpc = index.min %db_kpos, {limit} : index",
             f"{ind}  %db_entb = scalar.assume %db_ent [range(%db_ent, 0, 134217727)] : i32",
             f"{ind}  %db_ph = index.cast %db_entb : i32 to index",
             f"{ind}  %db_sb = index.mul %db_ph, %c16 : index",
             f"{ind}  %db_wi = index.rem %db_kpc, %c16 : index",
             f"{ind}  %db_sl = index.add %db_sb, %db_wi : index",
             f"{ind}  %db_ps = index.add %db_pp, %db_sl : index",
             f"{ind}  %db_pos_bytes = index.constant {pb} : offset",
             f"{ind}  %db_koff = index.scale %db_ps, %db_pos_bytes : index, offset -> offset",
             f"{ind}  %db_voff = index.add %db_koff, %v_plane_delta : offset",
             f"{ind}  %db_kc = buffer.view %arena[%db_koff] : buffer -> {cache_ty}",
             f"{ind}  %db_vc = buffer.view %arena[%db_voff] : buffer -> {cache_ty}",
             f"{ind}  %db_kh = vector.load %db_kc[{kload}] : {cache_ty} -> {vec}",
             f"{ind}  %db_vh = vector.load %db_vc[{kload}] : {cache_ty} -> {vec}",
             f"{ind}  %db_n2 = index.add %kt, %c2 : index"]
    loads += [ln.replace(ind, ind + "  ", 1) for ln in entry("%db_n2", "_n")]
    tail = [f"{ind}  %db_kz = scf.select %db_ok, %db_kh, %kz : {vec}",
            f"{ind}  %db_vz = scf.select %db_ok, %db_vh, %kz : {vec}",
            f"{ind}  scf.if %st_live {{"]
    tail += ["  " + ln for ln in store_lines("%db_kz", "%db_vz", "%db_kvn", "%db_vvn", "_dbn")[0:]]
    tail += [f"{ind}  }}"]
    # fix the indentation of the stores (they came from inside scf.if %st_live: same depth)
    if at == "b2":   # the stores right after the second barrier (the S exchange): the loads hide behind the S phase only
        ib2 = next(i for i in range(ib + 1, yi - 1) if lines[i].strip().startswith("kernel.barrier"))
        body = lines[hi + 1:hi + 2] + body_top + lines[ib:ib + 1] + loads + lines[ib + 1:ib2 + 1] + tail + lines[ib2 + 1:yi - 1]
    else:            # at the end of the iteration, after the PV work
        body = lines[hi + 1:hi + 2] + body_top + lines[ib:ib + 1] + loads + lines[ib + 1:yi - 1] + tail
    y = lines[yi]
    y = re.sub(r"^(.*?) : (f32, f32, .*)$", lambda mm: f"{mm.group(1)}, %db_ent_n : {mm.group(2)}, i32", y)
    head = (f"{ind}%m_final, %l_final, {m.group('outs').strip()}, %db_ent_f = scf.for %kt = [%c0 to %key_tiles step %c1]"
            f"({m.group('args')}, %db_ent = %db_ent_i : i32) -> ({m.group('types')}, i32) {{")
    out = lines[:hi] + pre + [head] + body + [y] + lines[yi + 1:]
    text = "\n".join(out)
    # the two staging buffers
    text = re.sub(r"(%k_bytes = index\.constant )(\d+)( : offset)", lambda mm: f"{mm.group(1)}{2 * int(mm.group(2))}{mm.group(3)}", text, count=1)
    text = re.sub(r"(%vt_bytes = index\.constant )(\d+)( : offset)", lambda mm: f"{mm.group(1)}{2 * int(mm.group(2))}{mm.group(3)}", text, count=1)
    text = text.replace(kv.group(0) + "\n", "", 1).replace(vv.group(0) + "\n", "", 1)
    return text
