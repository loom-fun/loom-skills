"""Paired K/V staging for the gfx12 flash2s kernels (gen_flash2s.py, FLASH2_VP=1, gfx12 twin only).

gen_flash2s stages a K/V tile with one key per lane (workitem % KT) and SEG_W dims per lane: K is one vector store, but V^T
([dim][key], the lhs layout of O^T += V^T P^T) takes SEG_W scalar 16-bit workgroup-memory stores (extract, address, ds_store_b16).
Here a lane stages two neighbouring keys (2p, 2p + 1) and SEG_W / 2 dims: K is two vector stores, V^T takes one 32-bit store of the
two keys' values per dim (vector<2xf16> at an even column): half the stores, the same global bytes per lane. The two keys are in
one page (a tile is one page at head_dim 512; two keys of an even-aligned pair always are), so the second key's offset is the
first's plus one position (the plain KV layout g_kv_offset; a dense layout is left to the generic staging).

Applied to the generated gfx12 twin text before the peel (flash2_peel.py); a kernel it does not fit is returned unchanged:
lanes must equal (KT / 2) * (D / SEG_W2) with SEG_W2 = 8 or 16, and the KV function must be g_kv_offset.
"""
import re

HEADER = re.compile(r"^(?P<ind> *)%m_final, %l_final, (?P<outs>[^=]*) = scf\.for %kt = \[%c0 to %key_tiles step %c1\]\((?P<args>.*)\) -> \((?P<types>.*)\) \{$")


def vpack(text):
    lines = text.split("\n")
    hi = next((i for i, ln in enumerate(lines) if HEADER.match(ln)), None)
    if hi is None:
        return text
    ind = HEADER.match(lines[hi]).group("ind")
    assert lines[hi + 2].strip() == "scf.if %st_live {", lines[hi + 2]
    ib = next(i for i in range(hi + 3, len(lines)) if lines[i].strip().startswith("kernel.barrier"))
    stage = lines[hi + 2:ib]
    blk = "\n".join(stage)
    fn = re.search(r"func\.call @(\w+)\(%origins", blk).group(1)
    if fn != "g_kv_offset":
        return text
    kt = int(re.search(r"%ckt = index\.constant (\d+) : index", text).group(1))
    seg_w = int(re.search(r"%cseg_w = index\.constant (\d+) : index", text).group(1))
    kv_view = re.search(r"%k_view = buffer\.view %k_tile\[%origin\] : buffer -> view<(\d+)x(\d+)xf16>", text)
    vt_view = re.search(r"%vt_view = buffer\.view %vt_tile\[%origin\] : buffer -> view<(\d+)x(\d+)xf16>", text)
    d = int(vt_view.group(1))
    lanes = int(re.search(r"%clanes = index\.constant (\d+) : index", text).group(1))
    pairs = kt // 2
    segs = lanes // pairs
    segw = d // segs
    if pairs * segs != lanes or segw * segs != d or segw not in (8, 16):
        return text
    limit = re.search(r"index\.cmp ule, %key_pos, (%\w+) : index", blk).group(1)
    vec = f"vector<{segw}xf16>"
    pb = re.search(r"%pos_bytes = index\.constant (\d+) : offset", text)
    assert pb, "no %pos_bytes"
    # the lookup is made once per lane (the pair's page); the second key is one position on
    defs = f"""{ind}%vp_c2 = index.constant 2 : index
{ind}%vp_pairs = index.constant {pairs} : index
{ind}%vp_pair = index.rem %workitem, %vp_pairs : index
{ind}%vp_key0_v = index.mul %vp_pair, %vp_c2 : index
{ind}%vp_key0 = index.assume %vp_key0_v [range(%vp_key0_v, 0, {kt - 2}), multiple_of(%vp_key0_v, 2)] : index
{ind}%vp_key1 = index.add %vp_key0, %c1 : index
{ind}%vp_seg = index.div %workitem, %vp_pairs : index
{ind}%vp_segw = index.constant {segw} : index
{ind}%vp_dim_v = index.mul %vp_seg, %vp_segw : index
{ind}%vp_dim = index.assume %vp_dim_v [range(%vp_dim_v, 0, {d - segw})] : index
{ind}%vp_kz = vector.constant 0.0 : {vec}
"""
    new = f"""{ind}  %vp_pos0 = index.add %tile_first, %vp_key0 : index
{ind}  %vp_pos1 = index.add %vp_pos0, %c1 : index
{ind}  %vp_ok0 = index.cmp ule, %vp_pos0, {limit} : index
{ind}  %vp_ok1 = index.cmp ule, %vp_pos1, {limit} : index
{ind}  %vk0, %vv0, %vk1, %vv1 = scf.if %vp_ok0 -> ({vec}, {vec}, {vec}, {vec}) {{
{ind}    %vp_kp = index.assume %vp_pos0 [range(%vp_pos0, 0, 262142)] : index
{ind}    %vp_koff0 = func.call @g_kv_offset(%origins, %row, %vp_kp, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
{ind}    %vp_voff0 = index.add %vp_koff0, %v_plane_delta : offset
{ind}    %vp_kc0 = buffer.view %arena[%vp_koff0] : buffer -> view<{re.search(r'view<(\d+x\d+)xf16>', blk).group(1)}xf16>
{ind}    %vp_vc0 = buffer.view %arena[%vp_voff0] : buffer -> view<{re.search(r'view<(\d+x\d+)xf16>', blk).group(1)}xf16>
{ind}    %vp_kh0 = vector.load %vp_kc0[%kv, %vp_dim] : view<{re.search(r'view<(\d+x\d+)xf16>', blk).group(1)}xf16> -> {vec}
{ind}    %vp_vh0 = vector.load %vp_vc0[%kv, %vp_dim] : view<{re.search(r'view<(\d+x\d+)xf16>', blk).group(1)}xf16> -> {vec}
{ind}    %vp_koff1 = index.add %vp_koff0, %pos_bytes : offset
{ind}    %vp_voff1 = index.add %vp_voff0, %pos_bytes : offset
{ind}    %vp_kc1 = buffer.view %arena[%vp_koff1] : buffer -> view<{re.search(r'view<(\d+x\d+)xf16>', blk).group(1)}xf16>
{ind}    %vp_vc1 = buffer.view %arena[%vp_voff1] : buffer -> view<{re.search(r'view<(\d+x\d+)xf16>', blk).group(1)}xf16>
{ind}    %vp_kh1, %vp_vh1 = scf.if %vp_ok1 -> ({vec}, {vec}) {{
{ind}      %vp_kl1 = vector.load %vp_kc1[%kv, %vp_dim] : view<{re.search(r'view<(\d+x\d+)xf16>', blk).group(1)}xf16> -> {vec}
{ind}      %vp_vl1 = vector.load %vp_vc1[%kv, %vp_dim] : view<{re.search(r'view<(\d+x\d+)xf16>', blk).group(1)}xf16> -> {vec}
{ind}      scf.yield %vp_kl1, %vp_vl1 : {vec}, {vec}
{ind}    }} else {{
{ind}      scf.yield %vp_kz, %vp_kz : {vec}, {vec}
{ind}    }}
{ind}    scf.yield %vp_kh0, %vp_vh0, %vp_kh1, %vp_vh1 : {vec}, {vec}, {vec}, {vec}
{ind}  }} else {{
{ind}    scf.yield %vp_kz, %vp_kz, %vp_kz, %vp_kz : {vec}, {vec}, {vec}, {vec}
{ind}  }}
{ind}  vector.store %vk0, %k_view[%vp_key0, %vp_dim] : {vec}, view<{kv_view.group(1)}x{kv_view.group(2)}xf16>
{ind}  vector.store %vk1, %k_view[%vp_key1, %vp_dim] : {vec}, view<{kv_view.group(1)}x{kv_view.group(2)}xf16>
{ind}  scf.for %vp_j = [%c0 to %vp_segw step %c1] unroll {{
{ind}    %vp_a = vector.extract %vv0[%vp_j] : {vec} -> f16
{ind}    %vp_b = vector.extract %vv1[%vp_j] : {vec} -> f16
{ind}    %vp_pr = vector.from_elements %vp_a, %vp_b : vector<2xf16>
{ind}    %vp_row_v = index.add %vp_dim, %vp_j : index
{ind}    %vp_row = index.assume %vp_row_v [range(%vp_row_v, 0, {d - 1})] : index
{ind}    vector.store %vp_pr, %vt_view[%vp_row, %vp_key0] : vector<2xf16>, view<{d}x{vt_view.group(2)}xf16>
{ind}  }}"""
    # one tile (a page) per iteration: all lanes live, no %st_live guard
    out = lines[:hi] + defs.rstrip("\n").split("\n") + [lines[hi], lines[hi + 1]] + new.split("\n") + lines[ib:]
    return "\n".join(out)
