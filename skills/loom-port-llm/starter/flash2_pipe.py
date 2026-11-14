"""Two barriers per key tile instead of three, with the next tile's global loads in flight (gfx12 flash2s twins, FLASH2_PIPE=1).

gen_flash2s stages a tile (K and V^T into workgroup memory), barrier A, runs S = K Q^T and stores the S partials, barrier B, then
softmax and O^T += V^T P^T, barrier C. Each step reads what the previous one wrote, so every phase waits for all 16 waves, and the
global loads of the staging sit between C and A with nothing else to run. Barriers cost 30 % of the d512 kernel (ablation), and the
loop is issue bound.

Order of tile t here (K_t and V^T_t of tile t are in workgroup memory when it starts):
    issue the global loads of tile t + 1 (K and V rows into registers)         <- in flight during the math
    S MMAs of tile t, store the S partials
    barrier B                  (every wave is done reading K_t; the partials are visible)
    store K_{t+1}              (K is free; visible to everyone after C)
    softmax, O^T += V^T_t P^T
    barrier C                  (every wave is done with V^T_t and the partials)
    store V^T_{t+1}            (visible to the PV MMAs after the next barrier B)
so barrier A is gone: K_{t+1} piggybacks on C and V^T_{t+1} on the next B. The loads are unconditional (no scf.if phi that would make the
compiler wait for them at once): a key past the last live position is read from the last live position (finite data; its softmax
weight is masked to exactly 0 and V^T of masked keys is zeroed by the masked path), the page of tile t + 2 is looked up one iteration
early (loop-carried i32), and the stores of a non-existent next tile are skipped by a uniform branch. Tile 0 is staged by the original
code before the loop.

Needs the hoisted page-map view of flash2_trim.kvhoist (it is applied first); returns the text unchanged when the loop does not have
the expected shape.
"""
import re

HEADER = re.compile(r"^(?P<ind> *)%m_final, %l_final, (?P<outs>[^=]*) = scf\.for %kt = \[%c0 to %key_tiles step %c1\]\((?P<args>.*)\) -> \((?P<types>.*)\) \{$")


def pipe(text):
    if "%kh_map = buffer.view" not in text:
        return text
    lines = text.split("\n")
    hi = next((i for i, ln in enumerate(lines) if HEADER.match(ln)), None)
    if hi is None:
        return text
    m = HEADER.match(lines[hi])
    ind = m.group("ind")
    if lines[hi + 2].strip() != "scf.if %st_live {":
        return text
    bars = [i for i in range(hi + 3, len(lines)) if lines[i].strip().startswith("kernel.barrier")]
    yi = next(i for i in range(hi + 3, len(lines)) if lines[i].lstrip().startswith("scf.yield %m_new, %l_new,"))
    if len(bars) < 3 or bars[2] > yi:
        return text
    bA, bB, bC = bars[:3]
    stage = lines[hi + 2:bA]
    blk = "\n".join(stage)
    if stage[-1] != ind + "  }":
        return text
    limit = re.search(r"index\.cmp ule, %key_pos, (%\w+) : index", blk).group(1)
    kidx = next(i for i, ln in enumerate(stage) if ln.strip().startswith("vector.store %kvals, %k_view"))
    # stage block pieces: [0: scf.if] [loads ... up to the K store) [K store] [V^T store loop ... } of the scf.if)
    loads_lines = stage[1:kidx]
    k_store = stage[kidx]
    v_lines = stage[kidx + 1:-1]
    vec = re.search(r"vector<(\d+)xf16>", k_store).group(0)
    viewk = re.search(r"%k_view\[[^\]]*\]", k_store).group(0)
    # the prologue: tile 0, original code with %tile_first = %key_low
    pro_stage = [ln.replace("%tile_first", "%key_low") for ln in stage]
    names = re.findall(r"^\s*(%\w+(?:, %\w+)*) = ", "\n".join(stage), re.M)
    defined = {n for g in names for n in re.findall(r"%\w+", g)}
    defined |= set(re.findall(r"scf\.for (%\w+) = ", blk))
    defined |= {"%j"} if "%j" in blk else set()
    sfx = lambda t: re.sub(r"%\w+", lambda mm: mm.group(0) + "_p0" if mm.group(0) in defined else mm.group(0), t)
    pro = [sfx(ln) for ln in pro_stage]
    if lines[hi + 1].strip().startswith("%tile_first = "):
        pass
    # page lookup of tile 1 (blocking is fine: once per workgroup) for the carried entry
    pq = f"""{ind}%pn_pg1_v = index.add %key_low_page, %c1 : index
{ind}%pn_pg1 = index.min %pn_pg1_v, %kh_last : index
{ind}%pn_pg1b = index.assume %pn_pg1 [range(%pn_pg1, 0, %kh_last), lt(%pn_pg1, %blocks)] : index
{ind}%pn_phys0 = view.load %kh_map[%pn_pg1b] : view<[%blocks]xi32> -> i32
"""
    # loads of tile t + 1 (unconditional)
    if int(re.search(r"%ckt = index\.constant (\d+) : index", text).group(1)) != 16:   # one page per tile (head_dim 512)
        return text
    kview = re.search(r'%k_cache = buffer\.view %arena\[%k_offset\] : buffer -> (view<[^>]*>)', blk).group(1)
    vview = re.search(r'%v_cache = buffer\.view %arena\[%v_offset\] : buffer -> (view<[^>]*>)', blk).group(1)
    loads = f"""{ind}%pn_first = index.add %tile_first, %ckt : index
{ind}%pn_pos_v = index.add %pn_first, %st_key : index
{ind}%pn_pos_c = index.min %pn_pos_v, {limit} : index
{ind}%pn_pos = index.assume %pn_pos_c [range(%pn_pos_c, 0, 262143)] : index
{ind}%pn_phb = scalar.assume %pn_phys [range(%pn_phys, 0, 1048575)] : i32
{ind}%pn_ph = index.cast %pn_phb : i32 to index
{ind}%pn_sb = index.mul %pn_ph, %kh_c16 : index
{ind}%pn_wi = index.rem %pn_pos, %kh_c16 : index
{ind}%pn_sl = index.add %pn_sb, %pn_wi : index
{ind}%pn_ps = index.add %kh_pp, %pn_sl : index
{ind}%pn_koff = index.scale %pn_ps, %pos_bytes : index, offset -> offset
{ind}%pn_voff = index.add %pn_koff, %v_plane_delta : offset
{ind}%pn_kc = buffer.view %arena[%pn_koff] : buffer -> {kview}
{ind}%pn_vc = buffer.view %arena[%pn_voff] : buffer -> {vview}
{ind}%pn_kh = vector.load %pn_kc[%kv, %st_dim] : {kview} -> {vec}
{ind}%pn_vh = vector.load %pn_vc[%kv, %st_dim] : {vview} -> {vec}
{ind}%pn_t2 = index.add %pn_first, %ckt : index
{ind}%pn_pg2_v = index.div %pn_t2, %kh_c16 : index
{ind}%pn_pg2 = index.min %pn_pg2_v, %kh_last : index
{ind}%pn_pg2b = index.assume %pn_pg2 [range(%pn_pg2, 0, %kh_last), lt(%pn_pg2, %blocks)] : index
{ind}%pn_phys_n = view.load %kh_map[%pn_pg2b] : view<[%blocks]xi32> -> i32
{ind}%pn_nt = index.add %kt, %c1 : index
{ind}%pn_has = index.cmp ult, %pn_nt, %key_tiles : index
{ind}%pn_do = scalar.andi %st_live, %pn_has : i1"""
    kst = f"""{ind}scf.if %pn_do {{
{ind}  vector.store %pn_kh, {viewk} : {vec}, {re.search(r'vector<\d+xf16>, (view<[^>]*>)', k_store).group(1)}
{ind}}}"""
    vst = [f"{ind}scf.if %pn_do {{"] + [ln.replace("%vvals", "%pn_vh") for ln in v_lines] + [f"{ind}}}"]
    # assemble
    head = (f"{ind}%m_final, %l_final, {m.group('outs').strip()}, %pn_phys_f = scf.for %kt = [%c0 to %key_tiles step %c1]"
            f"({m.group('args')}, %pn_phys = %pn_phys0 : i32) -> ({m.group('types')}, i32) {{")
    y = lines[yi]
    y = re.sub(r"^(.*?) : (f32, f32, .*)$", lambda mm: f"{mm.group(1)}, %pn_phys_n : {mm.group(2)}, i32", y)
    body = [lines[hi + 1]] + loads.split("\n") + lines[bA + 1:bB + 1] + kst.split("\n") + lines[bB + 1:bC + 1] + vst + [y]
    out = lines[:hi] + pro + [ln for ln in lines[bA:bA + 1]] + pq.rstrip("\n").split("\n") + [head] + body + lines[yi + 1:]
    return "\n".join(out)
