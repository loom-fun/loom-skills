"""Instruction trims of the gfx12 flash2s twin (gen_flash2s.py; knobs FLASH2_SX / FLASH2_HOT, gfx12 twin only).

The d512 tile loop issues ~650 instructions per wave and tile for 16 WMMAs, and WMMA does not overlap with VALU on RDNA4, so the
kernel is issue bound (3200 cycles per tile and workgroup measured, 4 waves per SIMD).

FLASH2_SX=1 (head_dim 512, one key block): the S partials the four waves of a head exchange go through workgroup memory as the
lanes' own 8-vectors (one 32-byte record per lane: two 128-bit stores, two 128-bit loads per partial) instead of
fragment.store/load<result> (eight scalar 32-bit accesses per fragment and the address math): all waves have the same lane
layout, so the layout of the 16x16 tile does not matter.

FLASH2_HOT=1 (in the peeled loop's hot copy only: every key tile but the last): no key-validity select on the staged K/V (all
keys of a non-last tile are written positions) and, at head_dim 512 (global layers only), no causal / window select on S and P
(a non-last tile lies fully below the query tile's first position).
"""
import re


def sx(text):
    m = re.search(r"%s_view = buffer\.view %s_tile\[%origin\] : buffer -> view<(\d+)x16xf32>", text)
    if not m or "%part3_s" not in text or "%part4_s" in text:
        return text
    rows = int(m.group(1))
    waves = rows // 16
    text = text.replace(m.group(0), m.group(0) + f"\n  %sx_view = buffer.view %s_tile[%origin] : buffer -> view<{waves}x32x8xf32>"
                        "\n  %sx_c4 = index.constant 4 : index\n  %sx_hw = index.mul %head_in_group, %sx_c4 : index"
                        "\n  %sx_hw1 = index.add %sx_hw, %c1 : index\n  %sx_c2 = index.constant 2 : index\n  %sx_hw2 = index.add %sx_hw, %sx_c2 : index"
                        "\n  %sx_c3 = index.constant 3 : index\n  %sx_hw3 = index.add %sx_hw, %sx_c3 : index", 1)
    text = re.sub(r"vector\.fragment\.store<result> (%\w+), %s_view\[%my_s, %c0\] shape \[%c16, %c16\] : vector<8xf32>, view<\d+x16xf32>",
                  lambda mm: f"vector.store {mm.group(1)}, %sx_view[%wave, %lane, %c0] : vector<8xf32>, view<{waves}x32x8xf32>", text)
    hw = {0: "%sx_hw", 1: "%sx_hw1", 2: "%sx_hw2", 3: "%sx_hw3"}
    text = re.sub(r"(%sp\d+_(\d)(?:_pt)?) = vector\.fragment\.load<result> %s_view\[%part\d_s, %c0\] shape \[%c16, %c16\] : view<\d+x16xf32> -> vector<8xf32>",
                  lambda mm: f"{mm.group(1)} = vector.load %sx_view[{hw[int(mm.group(2))]}, %lane, %c0] : view<{waves}x32x8xf32> -> vector<8xf32>", text)
    return text


def hot(lines, global_only):
    """The hot copy's lines (flash2_peel.py) with the trims; GLOBAL_ONLY: the masks go too."""
    out = []
    for ln in lines:
        ln = re.sub(r"^(\s*%key_valid) = index\.cmp ule, .*$", r"\1 = scalar.constant true : i1", ln)
        if global_only:
            ln = re.sub(r"^(\s*%masked\d+) = vector\.select %valid\d+, (%sc\d+), %neg_v : vector<8xf32>$", r"\1 = vector.addf \2, %zero8 : vector<8xf32>", ln)
            ln = re.sub(r"^(\s*%p\d+) = vector\.select %valid\d+, (%pe\d+), %zero8 : vector<8xf32>$", r"\1 = vector.addf \2, %zero8 : vector<8xf32>", ln)
        out.append(ln)
    return out


def lazy(text):
    """FLASH2_LAZY=1: the O^T rescale (8 fragments x 8 elements: 64 multiplies per wave and tile) runs only when some query column of
    the wave changed its running max (kernel.subgroup.vote.any of m_new != m). A column whose max is unchanged has alpha == 1 exactly,
    so the multiplies the skipped branch drops are multiplications by 1.0: same results, bit for bit."""
    pat = re.compile(r"^( *)%os(\d+) = vector\.mulf %a\2, %alpha_v : vector<8xf32>$", re.M)
    ms = list(pat.finditer(text))
    if not ms:
        return text
    n = len(ms)
    ind = ms[0].group(1)
    names = [f"%os{m.group(2)}" for m in ms]
    avs = [f"%a{m.group(2)}" for m in ms]
    tys = ", ".join(["vector<8xf32>"] * n)
    blk = [f"{ind}%lz_chg = scalar.cmpf one, %m_new, %m : f32",
           f"{ind}%lz_any = kernel.subgroup.vote.any %lz_chg : i1",
           f"{ind}{', '.join(names)} = scf.if %lz_any -> ({tys}) {{"]
    blk += [f"{ind}  %lzs{m.group(2)} = vector.mulf %a{m.group(2)}, %alpha_v : vector<8xf32>" for m in ms]
    blk += [f"{ind}  scf.yield {', '.join(f'%lzs{m.group(2)}' for m in ms)} : {tys}", f"{ind}}} else {{",
            f"{ind}  scf.yield {', '.join(avs)} : {tys}", f"{ind}}}"]
    first = ms[0]
    out = text[:first.start()] + "\n".join(blk) + text[first.end():]
    for m in ms[1:]:
        out = out.replace(m.group(0) + "\n", "", 1)
    return out


def vtflat(text):
    """FLASH2_VTF=1: the V^T staging stores take a flat index (base + j * pitch) so the address math of the unrolled stores folds
    into the store offsets."""
    m = re.search(r"%vt_view = buffer\.view %vt_tile\[%origin\] : buffer -> view<(\d+)x(\d+)xf16>", text)
    if not m:
        return text
    d, vtc = int(m.group(1)), int(m.group(2))
    text = text.replace(m.group(0), m.group(0) + f"\n  %vtf_view = buffer.view %vt_tile[%origin] : buffer -> view<{d * vtc}xf16>\n  %vtf_c = index.constant {vtc} : index", 1)
    pat = re.compile(r"( *)%vt_row_value = index\.add %st_dim, %j : index\n *%vt_row = index\.assume %vt_row_value \[range\(%vt_row_value, 0, \d+\)\] : index\n *view\.store (%\w+), %vt_view\[%vt_row, %st_key\] : f16, view<\d+x\d+xf16>")
    def rep(mm):
        i = mm.group(1)
        return (f"{i}%vtf_b0 = index.mul %st_dim, %vtf_c : index\n{i}%vtf_b = index.add %vtf_b0, %st_key : index\n{i}%vtf_jo = index.mul %j, %vtf_c : index\n"
                f"{i}%vtf_iv = index.add %vtf_b, %vtf_jo : index\n{i}%vtf_i = index.assume %vtf_iv [range(%vtf_iv, 0, {d * vtc - 1})] : index\n"
                f"{i}view.store {mm.group(2)}, %vtf_view[%vtf_i] : f16, view<{d * vtc}xf16>")
    text, n = pat.subn(rep, text)
    return text


def kvhoist(text):
    """FLASH2_KVH=1: the paged-KV lookup of the tile loop without its loop-invariant half. g_kv_offset (inline) reloads the row's
    page-map pointer (a 64-bit global load) on every call, and the compiler does not hoist a load out of a loop that has barriers:
    three dependent global loads per tile (map pointer, page entry, K/V). Here the map view and the plane base are made once before
    the loop; a tile does the page-entry load and the address math."""
    if "func.call @g_kv_offset(%origins, %row, %kp, %blocks, %k_plane, %pool)" not in text:
        return text
    pb = re.search(r"%pos_bytes = index\.constant (\d+) : offset", text)
    hdr = re.compile(r"^( *)%m_final, %l_final, [^=]*= scf\.for %kt = ", re.M)
    mh = hdr.search(text)
    ind = mh.group(1)
    defs = f"""{ind}%kh_c1 = index.constant 1 : index
{ind}%kh_c16 = index.constant 16 : index
{ind}%kh_rec_stride = index.constant 40 : offset
{ind}%kh_row = index.assume %row [range(%row, 0, 2147483646)] : index
{ind}%kh_rec_off = index.scale %kh_row, %kh_rec_stride : index, offset -> offset
{ind}%kh_records = buffer.view %origins[%kh_rec_off] : buffer -> view<5xi64>
{ind}%kh_map64 = view.load %kh_records[4] : view<5xi64> -> i64
{ind}%kh_map_b = scalar.assume %kh_map64 [range(%kh_map64, 0, 9223372036854775807)] : i64
{ind}%kh_map_origin = index.cast %kh_map_b : i64 to offset
{ind}%kh_map = buffer.view %origins[%kh_map_origin] : buffer -> view<[%blocks]xi32>
{ind}%kh_last = index.sub %blocks, %kh_c1 : index
{ind}%kh_pp = index.mul %k_plane, %pool : index
"""
    call = re.compile(r"^( *)%k_offset = func\.call @g_kv_offset\(%origins, %row, %kp, %blocks, %k_plane, %pool\) : \(buffer, index, index, index, index, index\) -> \(offset\)$", re.M)
    def rep(m):
        i = m.group(1)
        return (f"{i}%kh_pg_v = index.div %kp, %kh_c16 : index\n"
                f"{i}%kh_pg = index.assume %kh_pg_v [range(%kh_pg_v, 0, %kh_last), lt(%kh_pg_v, %blocks)] : index\n"
                f"{i}%kh_ph32 = view.load %kh_map[%kh_pg] : view<[%blocks]xi32> -> i32\n"
                f"{i}%kh_phb = scalar.assume %kh_ph32 [range(%kh_ph32, 0, 1048575)] : i32\n"
                f"{i}%kh_ph = index.cast %kh_phb : i32 to index\n"
                f"{i}%kh_sb = index.mul %kh_ph, %kh_c16 : index\n"
                f"{i}%kh_wi = index.rem %kp, %kh_c16 : index\n"
                f"{i}%kh_sl = index.add %kh_sb, %kh_wi : index\n"
                f"{i}%kh_ps = index.add %kh_pp, %kh_sl : index\n"
                f"{i}%k_offset = index.scale %kh_ps, %pos_bytes : index, offset -> offset")
    new, n = call.subn(rep, text)
    assert n >= 1
    mh = hdr.search(new)
    return new[:mh.start()] + defs + new[mh.start():]
