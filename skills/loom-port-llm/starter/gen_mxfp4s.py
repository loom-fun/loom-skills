"""MXFP4 split-plane (format mxfp4s: nibble plane U8 [rows, K/2] + E8M0 scale plane U8 [rows, K/32], the AMD/Quark safetensors layout) fast kernels.

gfx12 prefill: MXFP4 weights x fp8 activations on fp8 WMMA (v_wmma_f32_16x16x16_fp8_bf8). Opt-in: choice prefill.gemm.mxfp4s_f8 (env GEMM_MXFP4S_F8).
The weights are read as stored (no repack): the kernel's staging thread loads 32 nibble bytes and 2 scale bytes of its row per 128-value step.
Nibble order inside a 32-value block (order=): "pairs" (OCP MX / Quark: byte i holds element 2i in the low nibble, 2i+1 in the high one) or
"split16" (ggml: element i low, i+16 high). The producers' fp8 twins, the fuse pass and the exactness rules are those of the ggml-layout fp8 kernel
(skills/loom-kb/device/rdna4/prefill.md, "MXFP4 weights on fp8 WMMA"), with the weight operand split in two buffers."""
import os, re
from gen_gemm_rdna4 import enabled

# ======================================================================================================================
# MXFP4 weights x fp8 activations on fp8 WMMA (R9700, opt-in: choice prefill.gemm.mxfp4s_f8, env GEMM_MXFP4S_F8)
# ======================================================================================================================
# A weight is fp4 (E2M1: 0 .5 1 1.5 2 3 4 6, sign) * 2^(e - 127), e the block's E8M0 byte. Every such value is a bf8 (e5m2) number
# as long as its exponent field fits (1..30): the kernel decodes a weight nibble to the byte
#     sign | (base[m] + 4 * (e - 107))        base = 8 12 14 16 18 20 22 for the magnitude codes m = 1..7, 0 for m = 0
# (a v_perm_b32 table lookup per four weights, the table = base + the block's offset, zero stays zero), which is the e5m2 value
# fp4 * 2^(e - 127 + 8): the weights are held at 2^8 times their value and the epilogue's row factor carries 2^-8 (the producers write
# it into scales[row, 1]). Exact for every block with 107 <= e <= 132 (field 2..30 with the +8 shift); a GGUF is checked by
# mxfp4_f8_range() (scan_mxfp4s_exponents.py reports a checkpoint's range). The activations are fp8 e4m3 with a per-row power-of-two scale
# 2^(7 - exponent of the row max) (the row max lands in [128, 256)), so a row's only loss is the e4m3 rounding of its values.
MXFP4_F8_SHIFT = 8
MXFP4_F8_E_MIN, MXFP4_F8_E_MAX = 107, 132


def f8_enabled():
    """GEMM_MXFP4S_F8 (choice prefill.gemm.mxfp4s_f8) on, and the gfx12 twins on."""
    return enabled() and os.environ.get("GEMM_MXFP4S_F8", "0").lower() not in ("", "0", "false", "no", "off")


# kernel designs of the fp8 GEMM (choice prefill.gemm.mxfp4s_f8_kernel, env GEMM_MXFP4S_F8_KERNEL): name -> (store kwargs, swiglu kwargs) of gemm_mxfp4s_f8_m256
F8_KERNELS = {"v1": ({}, {}),
              "narrow": ({"tn": 64, "rf": 4, "cf": 2}, {"tn": 32, "rf": 4, "cf": 1}),
              "narrow_swp": ({"tn": 64, "rf": 4, "cf": 2, "swp": True}, {"tn": 32, "rf": 4, "cf": 1, "swp": True}),
              "fenced": ({"fence_k": 4}, {"fence_k": 4}),
              "fenced_swp": ({"fence_k": 4, "swp": True}, {"fence_k": 4}),
              "lvs": ({"lv": True, "lvs": 1}, {}),     # ogemm: store GEMMs -5..-7 % at 1024 rows (kbench 43869957328, 43875719428); the SwiGLU pair keeps v1; on the card +1..1.6 % prefill but the consistency gate fails (device/rdna4/prefill.md)
              "lv": ({"lv": True}, {"lv": True, "rf": 4}),
              "ws": ({"ws": 4}, {"ws": 4})}


def f8_kernel_kwargs(mode):
    """the generator kwargs of the selected fp8 GEMM design for `mode` (store / swiglu)"""
    name = os.environ.get("GEMM_MXFP4S_F8_KERNEL", "") or "v1"
    return dict(F8_KERNELS[name][mode == "swiglu"])


def mxfp4_f8_range(blocks_e):
    """(lowest, highest) block exponent byte of an iterable/ndarray of E8M0 bytes, and whether every block fits the kernel's exact range."""
    lo, hi = int(min(blocks_e)), int(max(blocks_e))
    return lo, hi, MXFP4_F8_E_MIN <= lo and hi <= MXFP4_F8_E_MAX


def gemm_mxfp4s_f8_m256(mode, prefix="g", order="pairs", a_direct=True, dbuf=False, kmult=1, rf=None, a_pf=False, nt=256, bm=None, tn=None, cf=None, swp=False, swz=False, pfa=0, yf16=False, pad=16, pair=False, lv=False, lvpf=0, lvs=False, ws=0, fence_k=0):
    """The 256 x 128 MXFP4 (split planes) prefill GEMM of gfx12 on fp8 WMMA (v_wmma_f32_16x16x16_fp8_bf8): fp8 e4m3 activations (the row-scale
    producers' x8, scales s8) times MXFP4 weights decoded to bf8 e5m2 in LDS, exactly (see MXFP4_F8 below). Weights: nibble plane
    %w [n, k/2] and scale plane %ws [n, k/32] (swiglu: %w, %ws, %w2, %ws2). y = acc * s8[row, 1].
    store: 256 x 128 outputs, swiglu: 256 x 64 of each of gate and up. K advances 128 values (four MXFP4 blocks) per stage.
    dbuf (with a_direct): the weight stage is double-buffered, one barrier per step: the next step is decoded into the other buffer while this one is multiplied.
    a_pf (with a_direct, not dbuf): the next step's activation fragments are loaded during this step (carried in registers).
    a_direct (default): the activation fragments are loaded from global memory straight into registers (no LDS staging of the activations): -11 % GEMM time on
    the R9700 against the staged variant (a_direct=False). rf: 16-row activation fragments per wave (default 4 for store, 2 for swiglu: measured best, see
    device/rdna4/prefill.md); kmult 2 doubles the K step; nt 128 halves the workgroup; a_pf / dbuf are measured non-wins kept for the record.
    pair (store only): two projections of the same activation in one launch: output tiles [0, n / tn) multiply %w into %y, tiles
    [n / tn, 2 n / tn) multiply %w2 into %y2 (both n rows; the skinny GDN alpha / beta projections, ab_f8 below).
    fence_k: bound compiler A-load lookahead to groups of this many K16 fragments. A subgroup buffer fence is a compiler scheduling
    fence on SDK0.2.0/gfx12, with no hardware rendezvous/cache operation. Accumulation, addresses and tile geometry stay unchanged."""
    if fence_k:
        assert fence_k in (1, 2, 4) and a_direct and not (a_pf or lv or lvs or ws), "K-group fences require the direct original-K fragment path"
    if ws:   # the wave-specialized design (ws = producer waves)
        assert not pair
        return gemm_mxfp4s_f8_ws(mode, prefix, order, npw=ws, rf=rf, pad=pad, lvs=lvs)
    two_w = mode == "swiglu"
    assert not (pair and two_w)
    BM = bm or nt                 # rows of the workgroup tile (default 256; 128 / 256 with a wider tn: the tile shapes of the prefill GEMM sweep)
    TN = tn or (64 if two_w else 128)     # output columns per tile (swiglu: of each of gate and up)
    TW = TN * (2 if two_w else 1)         # weight rows staged per tile (swiglu: TN gate + TN up)
    RF = rf or (2 if two_w else 4)   # 16-row A fragments per wave (rf = 4: 64 rows); the accumulators are rf * cfr * (1 or 2 weights) = 16 fragments
    cfr = cf or (8 if two_w else 16) // RF   # 16-column B fragments per wave and weight
    CBM = 'c256' if BM == 256 else 'cbm'   # the row-tile constant (gen_gemm's GEMM_SKIP_PADDING looks for %c256 in `%row_origin = index.mul %row_group, %c256`)
    WR = BM // (16 * RF)          # waves stacked along the rows, WC along the columns
    WC = TN // (16 * cfr)
    NWV = WR * WC
    NT = NWV * 32                 # threads = waves * 32 (256 for the 256 x 128 tile)
    HALVES = NT // TW             # staging threads per weight row
    assert RF in (2, 4, 8) and WR * 16 * RF == BM and WC * 16 * cfr == TN and HALVES in (1, 2, 4) and NT <= 1024
    assert a_direct or (NT == BM == 256 and TW == 128)
    wave_cols = 16 * cfr
    YT = 'f16' if yf16 else 'f32'   # the output element type (f16: a measured option for the SwiGLU output, whose consumer is the down projection's row-scale producer)
    # the workgroup's thread index folded onto the BM rows of the tile (when the workgroup has more threads than the tile has rows)
    BMI, BMDEF = ("%workitem", "") if NT == BM else ("%bm_item", f"  %bm_item = index.rem %workitem, %{CBM} : index\n")
    KS = 128 * kmult              # K per step: 4 * kmult 32-value blocks (kmult > 1 needs a_direct)
    PX = KS + pad                 # LDS pitch (bytes) of both stages: KS data + pad (16)
    NB = (KS // 32) // HALVES     # blocks one staging thread decodes per step (a half of a row's step; the whole step with 128 threads)
    assert kmult == 1 or a_direct
    name = f"{prefix}_gemm_mxfp4s_{mode}{'_pair' if pair else ''}_f8_m256_rdna4"
    weights = ["w", "w2"] if two_w else ["w"]
    accs = [f"{wi}r{r}c{c}" for wi in range(len(weights)) for r in range(RF) for c in range(cfr)]
    L = []
    A = L.append
    A(f'''kernel.def target(@rdna4) export("{name}") @{name}(%rows: index, %n: index) {{
  %c1 = index.constant 1 : index
  %{CBM} = index.constant {BM} : index
  %cnt = index.constant {NT} : index
  %tn = index.constant {TN} : index
  %row_tiles = index.div %rows, %{CBM} : index
  %out_tiles = index.div %n, %tn : index{chr(10) + "  %out_tiles2 = index.add %out_tiles, %out_tiles : index" if pair else ""}
  kernel.launch.config workgroups({"%out_tiles, %row_tiles" if swz else "%row_tiles, %out_tiles"}{"2" if pair else ""}, %c1) workgroup_size(%cnt, %c1, %c1) : index
}} launch(%rows_arg: index, %k_arg: index, %n_arg: index, %x: buffer, %scales: buffer, {", ".join(f"%{w}: buffer, %{w}s: buffer" for w in weights)}{", %w2: buffer, %w2s: buffer" if pair else ""}, %y: buffer{", %y2: buffer" if pair else ""}) {{
  %rows = index.assume %rows_arg [range(%rows_arg, {BM}, 8192), multiple_of(%rows_arg, {BM})] : index
  %k = index.assume %k_arg [range(%k_arg, {KS}, 65536), multiple_of(%k_arg, {KS})] : index
  %n = index.assume %n_arg [range(%n_arg, {TN}, 262144), multiple_of(%n_arg, {TN})] : index
  %origin = index.constant 0 : offset
  %two = index.constant 2 : offset
  %four = index.constant 4 : offset
  %one_off = index.constant 1 : offset
  %o0 = index.constant 0 : offset
{"".join(f"  %o{16 * j} = index.constant {16 * j} : offset" + chr(10) for j in range(1, NB))}{"".join(f"  %cj{j} = index.constant {j} : index" + chr(10) for j in range(NB))}  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %cks2 = index.constant {KS // 2} : index
  %cnb16 = index.constant {NB * 16} : index
  %cks32 = index.constant {KS // 32} : index
  %cnb = index.constant {NB} : index
  %c128 = index.constant {KS} : index
  %{CBM} = index.constant {BM} : index
  %tn = index.constant {TN} : index
  %row_tile_count = index.div %rows, %{CBM} : index
  %out_tile_count = index.div %n, %tn : index
  %row_group_id = kernel.workgroup.id<{"y" if swz else "x"}> : index
  %out_group_id = kernel.workgroup.id<{"x" if swz else "y"}> : index
  %row_group = index.assume %row_group_id [lt(%row_group_id, %row_tile_count)] : index
{PAIR_TEXT if pair else "  %out_group = index.assume %out_group_id [lt(%out_group_id, %out_tile_count)] : index"}
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, {NT - 1})] : index
  %subgroup = kernel.subgroup.id : index
  %bounded_subgroup = index.assume %subgroup [range(%subgroup, 0, {NT // 32 - 1})] : index
  %cwc = index.constant {WC} : index
  %cwr = index.constant {16 * RF} : index
  %wave_row_group = index.div %bounded_subgroup, %cwc : index
  %wave_col_group = index.rem %bounded_subgroup, %cwc : index
  %wave_row_value = index.mul %wave_row_group, %cwr : index
  %wave_row = index.assume %wave_row_value [range(%wave_row_value, 0, {BM - 16 * RF})] : index''')
    for r in range(1, RF):
        A(f"  %wave_row{r}_c = index.constant {16 * r} : index\n  %wave_row{r} = index.add %wave_row, %wave_row{r}_c : index")
    A(f'''  %wave_cols = index.constant {wave_cols} : index
  %wave_col_value = index.mul %wave_col_group, %wave_cols : index
  %wave_col = index.assume %wave_col_value [range(%wave_col_value, 0, {TN - wave_cols})] : index
  %row_origin = index.mul %row_group, %{CBM} : index
  %out_origin = index.mul %out_group, %tn : index
  %blocks = index.div %k, %c32 : index
  %k_half = index.div %k, %c2 : index
  %k_half_off = index.scale %k_half, %one_off : index, offset -> offset
  %ks_off = index.scale %blocks, %one_off : index, offset -> offset
  %k16 = index.div %k, %c16 : index
  %x_view = buffer.view %x[%origin] : buffer -> view<[%rows]x[%k16]x16xi8>
  %s_view = buffer.view %scales[%origin] : buffer -> view<[%rows]x2xf32>
  %x_bytes = index.constant {BM * 64 if a_direct else 256 * PX} : offset
  %x_stage = buffer.alloca<workgroup> align(16) %x_bytes : buffer
{"" if a_direct else f"  %x_stage_st = buffer.view %x_stage[%origin] : buffer -> view<256x{PX}xi8>"}
  %w_bytes = index.constant {TW * PX} : offset
  %w_alloc = index.constant {TW * PX * (2 if dbuf else 1)} : offset
  %w_stage = buffer.alloca<workgroup> align(16) %w_alloc : buffer
  %w_stage_st = buffer.view %w_stage[%origin] : buffer -> view<{TW}x{PX}xi8>
  %a_schema = encoding.define #encoding.operand<element_format=f8e4m3, payload_elements=8, payload_registers=2> : encoding<schema>
  %b_schema = encoding.define #encoding.operand<element_format=f8e5m2, payload_elements=8, payload_registers=2> : encoding<schema>
  %a_layout = encoding.layout.dense : encoding<layout>
  %b_layout = encoding.layout.strided [1, {PX}] : encoding<layout>
  %a_storage = encoding.define #encoding.storage {{layout = %a_layout : encoding<layout>, schema = %a_schema : encoding<schema>}} : encoding<storage>
  %b_storage = encoding.define #encoding.storage {{layout = %b_layout : encoding<layout>, schema = %b_schema : encoding<schema>}} : encoding<storage>
{"  %x_g8 = buffer.view %x[%origin] : buffer -> view<[%rows]x[%k]xf8E4M3, %a_storage>" if a_direct else f"  %x_stage_f8 = buffer.view %x_stage[%origin] : buffer -> view<256x{PX}xf8E4M3, %a_storage>"}
{"" if dbuf else f"  %w_stage_f8 = buffer.view %w_stage[%origin] : buffer -> view<{PX}x{TW}xf8E5M2, %b_storage>"}
  // A staging: thread t copies the 128 activation bytes of row t of this step
{BMDEF}  %x_row_value = index.add %row_origin, {BMI} : index
  %x_row = index.assume %x_row_value [lt(%x_row_value, %rows)] : index
  // W staging: thread t decodes two MXFP4 blocks (64 weights) of weight row t/2: columns (t%2) * 64
  %chalves = index.constant {HALVES} : index
  %stage_row = index.div %workitem, %chalves : index
  %stage_half = index.rem %workitem, %chalves : index
{'' if HALVES < 4 else f'  %cnb32 = index.constant {NB * 32} : index' + chr(10)}  %stage_col = index.mul %stage_half, {'%cks2' if HALVES < 4 else '%cnb32'} : index''')
    if two_w:
        A(f'''  %half_tw = index.constant {TN} : index
  %w_is_up = index.cmp uge, %stage_row, %half_tw : index
  %w_local_up = index.sub %stage_row, %half_tw : index
  %w_local = scf.select %w_is_up, %w_local_up, %stage_row : index
  %w_out_row_value = index.add %out_origin, %w_local : index
  %wsel = scf.select %w_is_up, %w2, %w : buffer
  %wssel = scf.select %w_is_up, %w2s, %ws : buffer''')
    else:
        A("  %w_out_row_value = index.add %out_origin, %stage_row : index")
    A(f'''  %w_out_row = index.assume %w_out_row_value [lt(%w_out_row_value, %n)] : index
  %w_row_origin = index.scale %w_out_row, %k_half_off : index, offset -> offset
  %ws_row_origin = index.scale %w_out_row, %ks_off : index, offset -> offset
  %zero8 = vector.splatf 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %k_steps = index.div %k, %c128 : index
  %k_steps_m1 = index.sub %k_steps, %c1 : index
  %c107 = scalar.constant 107 : i32
  %cs2 = scalar.constant 2 : i32
  %ones = scalar.constant 16843009 : i32
  %lo_mask = scalar.constant -256 : i32
  %tab_lo_c = scalar.constant 235669504 : i32
  %tab_hi_c = scalar.constant 370414096 : i32
  %m7 = vector.splati 7 : vector<4xi8>
  %m4 = vector.splati 4 : vector<4xi8>
  %m80 = vector.splati -128 : vector<4xi8>
  %m4i = vector.splati 4 : vector<1xi32>
  %m80i = vector.splati -2139062144 : vector<1xi32>''')
    wsrc = "%wsel" if two_w else "%wpsel" if pair else "%w"
    ssrc = "%wssel" if two_w else "%wspsel" if pair else "%ws"

    def loads(sfx, kstep):
        """global loads of step `kstep` (an index value name): activation 8 x vector<16xi8>, weights 2 blocks (e, q)"""
        t = [f"    %kcol{sfx} = index.mul {kstep}, %c8 : index"]
        for i in range(0 if a_direct else 8):
            t.append(f"    %xc{sfx}{i}_v = index.add %kcol{sfx}, %xi{i} : index")
            t.append(f"    %xc{sfx}{i} = index.assume %xc{sfx}{i}_v [lt(%xc{sfx}{i}_v, %k16)] : index")
            t.append(f"    %xr{sfx}{i} = vector.load %x_view[%x_row, %xc{sfx}{i}, %c0] : view<[%rows]x[%k16]x16xi8> -> vector<16xi8>")
        # weights: nibble bytes [kstep*64 + half*32 + 16j, +16) of the row, scale byte kstep*4 + half*2 + j
        t.append(f"    %wk{sfx}_a = index.mul {kstep}, %cks2 : index")
        t.append(f"    %wk{sfx}_b = index.mul %stage_half, %cnb16 : index")
        t.append(f"    %wk{sfx} = index.add %wk{sfx}_a, %wk{sfx}_b : index")
        t.append(f"    %wk{sfx}_o = index.scale %wk{sfx}, %one_off : index, offset -> offset")
        t.append(f"    %wkb{sfx} = index.add %w_row_origin, %wk{sfx}_o : offset")
        t.append(f"    %sb{sfx}_a = index.mul {kstep}, %cks32 : index")
        t.append(f"    %sb{sfx}_b = index.mul %stage_half, %cnb : index")
        t.append(f"    %sb{sfx} = index.add %sb{sfx}_a, %sb{sfx}_b : index")
        for j in range(NB):
            t.append(f"    %wqo{j}{sfx} = index.add %wkb{sfx}, %o{16 * j} : offset")
            t.append(f"    %wq{j}{sfx}_v = buffer.view {wsrc}[%wqo{j}{sfx}] : buffer -> view<4xi32>")
            t.append(f"    %wq{j}{sfx} = vector.load %wq{j}{sfx}_v[0] : view<4xi32> -> vector<4xi32>")
            t.append(f"    %sbj{j}{sfx} = index.add %sb{sfx}, %cj{j} : index")
            t.append(f"    %sbo{j}{sfx}_a = index.scale %sbj{j}{sfx}, %one_off : index, offset -> offset")
            t.append(f"    %sbo{j}{sfx} = index.add %ws_row_origin, %sbo{j}{sfx}_a : offset")
            t.append(f"    %we{j}{sfx}_v = buffer.view {ssrc}[%sbo{j}{sfx}] : buffer -> view<1xi8>")
            t.append(f"    %we{j}{sfx}_b = view.load %we{j}{sfx}_v[0] : view<1xi8> -> i8")
            t.append(f"    %we{j}{sfx} = scalar.extui %we{j}{sfx}_b : i8 to i32")
        return "\n".join(t)

    # constant column offsets of the activation loads
    for i in range(8):
        A(f"  %xo{i} = index.constant {16 * i} : index\n  %xi{i} = index.constant {i} : index")
    def stage_text(wview="%w_stage_st", raw=("%cr_q{j}", "%cr_e{j}"), sfx=""):
        """activation stores (staged variant) and the decode of the weights' raw registers (names raw % j) into the LDS view `wview`; sfx renames
        every value the text defines (a second copy of it in one scope)"""
        L2 = []
        A2 = L2.append
        for i in range(0 if a_direct else 8):
            A2(f"    vector.store %cr_x{i}, %x_stage_st[%workitem, %xo{i}] : vector<16xi8>, view<256x{PX}xi8>")
        for j in range(NB):
            A2(f'''    %em{j} = scalar.subi %cr_e{j}, %c107 : i32
        %ea{j} = scalar.shli %em{j}, %cs2 : i32
        %ea4_{j} = scalar.muli %ea{j}, %ones : i32
        %ea4z_{j} = scalar.andi %ea4_{j}, %lo_mask : i32
        %tlo{j} = scalar.addi %ea4z_{j}, %tab_lo_c : i32
        %thi{j} = scalar.addi %ea4_{j}, %tab_hi_c : i32
        %tab2_{j} = vector.from_elements %tlo{j}, %thi{j} : vector<2xi32>
        %tab{j} = vector.bitcast %tab2_{j} : vector<2xi32> to vector<8xi8>''')
            for i in range(4):
                A2(f'''    %qw{j}{i}_1 = vector.slice %cr_q{j}[{i}] : vector<4xi32> -> vector<1xi32>
        %qw{j}{i} = vector.bitcast %qw{j}{i}_1 : vector<1xi32> to vector<4xi8>
        %lo_i{j}{i} = vector.andi %qw{j}{i}, %m7 : vector<4xi8>
        %hs_i{j}{i} = vector.shrui %qw{j}{i}_1, %m4i : vector<1xi32>
        %hs_b{j}{i} = vector.bitcast %hs_i{j}{i} : vector<1xi32> to vector<4xi8>
        %hi_i{j}{i} = vector.andi %hs_b{j}{i}, %m7 : vector<4xi8>
        %lo_m{j}{i} = vector.table.lookup %tab{j}[%lo_i{j}{i}] : vector<8xi8>, vector<4xi8> -> vector<4xi8>
        %hi_m{j}{i} = vector.table.lookup %tab{j}[%hi_i{j}{i}] : vector<8xi8>, vector<4xi8> -> vector<4xi8>
        %ls_i{j}{i} = vector.shli %qw{j}{i}_1, %m4i : vector<1xi32>
        %ls_m{j}{i} = vector.andi %ls_i{j}{i}, %m80i : vector<1xi32>
        %ls_b{j}{i} = vector.bitcast %ls_m{j}{i} : vector<1xi32> to vector<4xi8>
        %hs_m{j}{i} = vector.andi %qw{j}{i}_1, %m80i : vector<1xi32>
        %hs_s{j}{i} = vector.bitcast %hs_m{j}{i} : vector<1xi32> to vector<4xi8>
        %lo_f{j}{i} = vector.ori %lo_m{j}{i}, %ls_b{j}{i} : vector<4xi8>
        %hi_f{j}{i} = vector.ori %hi_m{j}{i}, %hs_s{j}{i} : vector<4xi8>
        %lo_w{j}{i} = vector.bitcast %lo_f{j}{i} : vector<4xi8> to vector<1xi32>
        %hi_w{j}{i} = vector.bitcast %hi_f{j}{i} : vector<4xi8> to vector<1xi32>''')
            A2(f'''    %lo4_{j} = vector.concat<0> %lo_w{j}0, %lo_w{j}1, %lo_w{j}2, %lo_w{j}3 : vector<1xi32>, vector<1xi32>, vector<1xi32>, vector<1xi32> -> vector<4xi32>
        %hi4_{j} = vector.concat<0> %hi_w{j}0, %hi_w{j}1, %hi_w{j}2, %hi_w{j}3 : vector<1xi32>, vector<1xi32>, vector<1xi32>, vector<1xi32> -> vector<4xi32>
        %lo16_{j} = vector.bitcast %lo4_{j} : vector<4xi32> to vector<16xi8>
        %hi16_{j} = vector.bitcast %hi4_{j} : vector<4xi32> to vector<16xi8>
        %wcl{j}_a = index.constant {32 * j} : index
        %wcl{j} = index.add %stage_col, %wcl{j}_a : index''')
            if order == "pairs":   # byte i = elements 2i (low nibble) and 2i + 1 (high): one 32-byte block in element order
                A2(f'''    %blk{j} = vector.interleave<0> %lo16_{j}, %hi16_{j} : vector<16xi8>, vector<16xi8> -> vector<32xi8>
        vector.store %blk{j}, %w_stage_st[%stage_row, %wcl{j}] : vector<32xi8>, view<{TW}x{PX}xi8>''')
            else:                  # ggml: element i low, i + 16 high
                A2(f'''    %wch{j} = index.add %wcl{j}, %c16 : index
        vector.store %lo16_{j}, %w_stage_st[%stage_row, %wcl{j}] : vector<16xi8>, view<{TW}x{PX}xi8>
        vector.store %hi16_{j}, %w_stage_st[%stage_row, %wch{j}] : vector<16xi8>, view<{TW}x{PX}xi8>''')

        t = "\n".join(L2).replace("%cr_q{j}", "@").replace("%w_stage_st[", wview + "[")
        t = re.sub(r"%cr_q(\d)", lambda m: raw[0].format(j=m.group(1)), t)
        t = re.sub(r"%cr_e(\d)", lambda m: raw[1].format(j=m.group(1)), t)
        if sfx:
            for nm in sorted(set(re.findall(r"^\s*(%\w+) = ", t, flags=re.M)), key=len, reverse=True):
                t = re.sub(re.escape(nm) + r"\b", nm + sfx, t)
        return t
    if a_direct:
        for r in range(RF):
            if r == 0:
                A("  %c15 = index.constant 15 : index\n  %rows_m15 = index.sub %rows, %c15 : index")
            A(f"  %xrow{r}_v = index.add %row_origin, {'%wave_row' if r == 0 else f'%wave_row{r}'} : index\n  %xrow{r} = index.assume %xrow{r}_v [lt(%xrow{r}_v, %rows_m15)] : index")
    if lv:   # lane-vector operand loads: one 16-byte load per lane covers two K substeps (A from global, B from LDS); both operands take K in the same
        # permuted order (substep 2p: bytes [32p + 16h, +8) of lane half h; substep 2p + 1: the next 8), so the products are those of the plain layout
        assert a_direct and not (a_pf or swp or pfa) and lvpf <= KS // 32
        A("""  %lv_c32 = index.constant 32 : index
  %lv_lane = index.rem %workitem, %lv_c32 : index
  %lv_l16 = index.rem %lv_lane, %c16 : index
  %lv_h = index.div %lv_lane, %c16 : index
  %lv_h16 = index.mul %lv_h, %c16 : index""")
        for r in range(RF):
            A(f"""  %lv_xr{r} = index.add %xrow{r}, %lv_l16 : index
  %lv_xm{r} = index.mul %lv_xr{r}, %k : index
  %lv_xo{r}_i = index.add %lv_xm{r}, %lv_h16 : index
  %lv_xo{r} = index.scale %lv_xo{r}_i, %one_off : index, offset -> offset""")
        for p in range(KS // 32):
            A(f"  %lv_kc{p} = index.constant {32 * p} : index\n  %lv_bk{p} = index.add %lv_h16, %lv_kc{p} : index")
        for wi in range(len(weights)):
            for c in range(cfr):
                A(f"  %lv_bn{wi}{c}_c = index.constant {wi * TN + c * 16} : index\n  %lv_bn{wi}{c}_a = index.add %wave_col, %lv_bn{wi}{c}_c : index\n  %lv_bn{wi}{c} = index.add %lv_bn{wi}{c}_a, %lv_l16 : index")
    def lv_aload(dst, r, p, base):
        """the 16-byte activation load of row fragment r, K pair p, at the step whose byte offset is `base` (an offset value)"""
        return (f"    %{dst}_xp = index.add %lv_xo{r}, {base} : offset\n    %{dst}_xv = buffer.view %x[%{dst}_xp] : buffer -> view<{KS}xi8>\n"
                f"    %{dst} = vector.load %{dst}_xv[%lv_kc{p}] : view<{KS}xi8> -> vector<16xi8>")
    if lvpf:
        for r in range(RF):
            for p in range(lvpf):
                A(lv_aload(f"lv_pa{r}_{p}", r, p, "%o0"))
    A(loads("p", "%c0"))
    if a_pf:
        assert a_direct and not dbuf
        for kk in range(KS // 16):
            A(f"  %kp{kk} = index.constant {16 * kk} : index")
            for r in range(RF):
                A(f"  %lp{r}_{kk} = vector.fragment.load<lhs> %x_g8[%xrow{r}, %kp{kk}] shape [%c16, %c16] : view<[%rows]x[%k]xf8E4M3, %a_storage> -> vector<2xi32>")
    if dbuf:
        assert a_direct
        A(stage_text(raw=("%wq{j}p", "%we{j}p"), sfx="_pro"))
        A("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        A("    %k_one = index.min %c1, %k_steps_m1 : index")
        A(loads("q", "%k_one"))
    carried = [(f"%cr_x{i}", f"%xrp{i}", f"%xrn{i}", "vector<16xi8>") for i in range(0 if a_direct else 8)]
    for j in range(NB):
        carried.append((f"%cr_e{j}", f"%we{j}{'q' if dbuf else 'p'}", f"%we{j}n", "i32"))
        carried.append((f"%cr_q{j}", f"%wq{j}{'q' if dbuf else 'p'}", f"%wq{j}n", "vector<4xi32>"))
    carried += [(f"%lv_ca{r}_{p}", f"%lv_pa{r}_{p}", f"%lv_na{r}_{p}", "vector<16xi8>") for r in range(RF) for p in range(lvpf)]
    if a_pf:
        carried += [(f"%ca{r}_{kk}", f"%lp{r}_{kk}", f"%ln{r}_{kk}", "vector<2xi32>") for kk in range(KS // 16) for r in range(RF)]
    A("  " + ", ".join(f"%o{a}" for a in accs) + ", " + ", ".join(f"%unused_{i}" for i in range(len(carried))) +
      " = scf.for %step = [%c0 to %k_steps step %c1](" +
      ", ".join(f"%i{a} = %init : vector<8xf32>" for a in accs) + ", " + ", ".join(f"{c[0]} = {c[1]} : {c[3]}" for c in carried) + ") -> (" +
      ", ".join("vector<8xf32>" for _ in accs) + ", " + ", ".join(c[3] for c in carried) + ") {")
    if dbuf:
        A("    %par = index.rem %step, %c2 : index\n    %npar = index.sub %c1, %par : index")
        A("    %cur_off = index.scale %par, %w_bytes : index, offset -> offset\n    %nxt_off = index.scale %npar, %w_bytes : index, offset -> offset")
        A(f"    %w_cur_f8 = buffer.view %w_stage[%cur_off] : buffer -> view<{PX}x{TW}xf8E5M2, %b_storage>")
        A(f"    %w_nxt_st = buffer.view %w_stage[%nxt_off] : buffer -> view<{TW}x{PX}xi8>")
        A("    %k_next_raw = index.add %step, %c2 : index\n    %k_next = index.min %k_next_raw, %k_steps_m1 : index")
    else:
        A(stage_text())
        A("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        # --- prefetch the next step (clamped to the last step: the extra loads are unused)
        A("    %k_next_raw = index.add %step, %c1 : index\n    %k_next = index.min %k_next_raw, %k_steps_m1 : index")
    A(loads("n", "%k_next"))
    cur = {a: f"%i{a}" for a in accs}
    rowv = ["%wave_row"] + [f"%wave_row{r}" for r in range(1, RF)]
    if a_direct:
        A("    %kbase = index.mul %step, %c128 : index")
    if pfa:   # L2 prefetch of the activation lines `pfa` steps ahead: this wave's share of the tile's rows (one 128-byte line per row and step)
        assert a_direct
        npf = 16 * RF // WC
        A(f"""    %pf_d = index.constant {pfa} : index
    %pf_k_raw = index.add %step, %pf_d : index
    %pf_k = index.min %pf_k_raw, %k_steps_m1 : index
    %pf_kc = index.mul %pf_k, %c8 : index
    %pf_npf = index.constant {npf} : index
    %pf_cg = index.mul %wave_col_group, %pf_npf : index
    %pf_rr = index.add %row_origin, %wave_row : index
    %pf_base = index.add %pf_rr, %pf_cg : index""")
        for i in range(npf):
            A(f"""    %pf_i{i} = index.constant {i} : index
    %pf_rv{i} = index.add %pf_base, %pf_i{i} : index
    %pf_r{i} = index.assume %pf_rv{i} [lt(%pf_rv{i}, %rows)] : index
    view.prefetch %x_view[%pf_r{i}, %pf_kc, %c0] {{intent = read, locality = l2}} : view<[%rows]x[%k16]x16xi8>""")
    if a_pf:
        A("    %k_next_a = index.assume %k_next [lt(%k_next, %k_steps)] : index\n    %kbn = index.mul %k_next_a, %c128 : index")
    def frag_loads(kk):
        if fence_k and kk % fence_k == 0:
            A("    buffer.fence scope(subgroup) ordering(acq_rel)")
        A(f"    %fk{kk} = index.constant {kk * 16} : index")
        for r in range(RF):
            if a_pf:
                if r == 0:
                    A(f"    %kgn{kk} = index.add %kbn, %fk{kk} : index")
                A(f"    %ln{r}_{kk} = vector.fragment.load<lhs> %x_g8[%xrow{r}, %kgn{kk}] shape [%c16, %c16] : view<[%rows]x[%k]xf8E4M3, %a_storage> -> vector<2xi32>")
                A(f"    %l{r}_{kk} = vector.bitcast %ca{r}_{kk} : vector<2xi32> to vector<2xi32>")
            elif a_direct:
                if r == 0:
                    A(f"    %kg{kk} = index.add %kbase, %fk{kk} : index")
                A(f"    %l{r}_{kk} = vector.fragment.load<lhs> %x_g8[%xrow{r}, %kg{kk}] shape [%c16, %c16] : view<[%rows]x[%k]xf8E4M3, %a_storage> -> vector<2xi32>")
            else:
                A(f"    %l{r}_{kk} = vector.fragment.load<lhs> %x_stage_f8[{rowv[r]}, %fk{kk}] shape [%c16, %c16] : view<256x{PX}xf8E4M3, %a_storage> -> vector<2xi32>")
        for wi in range(len(weights)):
            for c in range(cfr):
                A(f'''    %wr{wi}{c}_{kk}_c = index.constant {wi * TN + c * 16} : index
    %wr{wi}{c}_{kk} = index.add %wave_col, %wr{wi}{c}_{kk}_c : index
    %rf{wi}{c}_{kk} = vector.fragment.load<rhs> {"%w_cur_f8" if dbuf else "%w_stage_f8"}[%fk{kk}, %wr{wi}{c}_{kk}] shape [%c16, %c16] : view<{PX}x{TW}xf8E5M2, %b_storage> -> vector<2xi32>''')

    def frag_mma(kk):
        for wi in range(len(weights)):
            for c in range(cfr):
                for r in range(RF):
                    a = f"{wi}r{r}c{c}"
                    A(f"    %m{a}_{kk} = vector.mma %l{r}_{kk}, %rf{wi}{c}_{kk}, {cur[a]} : vector<2xi32>, vector<2xi32>, vector<8xf32>")
                    cur[a] = f"%m{a}_{kk}"

    def lv_loads(p, parts="ab"):
        if p == 0 and "a" in parts:
            A("    %lv_kb = index.scale %kbase, %one_off : index, offset -> offset")
        for r in range(RF if "a" in parts else 0):
            if p == 0:
                A(f"    %lv_xp{r} = index.add %lv_xo{r}, %lv_kb : offset\n    %lv_xv{r} = buffer.view %x[%lv_xp{r}] : buffer -> view<{KS}xi8>")
            if p < lvpf:
                A(f"    %lv_a{r}_{p} = vector.bitcast %lv_ca{r}_{p} : vector<16xi8> to vector<16xi8>")
            else:
                A(f"    %lv_a{r}_{p} = vector.load %lv_xv{r}[%lv_kc{p}] : view<{KS}xi8> -> vector<16xi8>")
            A(f"""    %lv_aw{r}_{p} = vector.bitcast %lv_a{r}_{p} : vector<16xi8> to vector<4xi32>""")
            for q in range(2):
                A(f"""    %lv_ar{r}_{2 * p + q} = vector.slice %lv_aw{r}_{p}[{2 * q}] : vector<4xi32> -> vector<2xi32>
    %l{r}_{2 * p + q} = vector.fragment<lhs> %lv_ar{r}_{2 * p + q} shape [%c16, %c16] using {{schema = %a_schema : encoding<schema>}} : vector<2xi32>""")
        for wi in range(len(weights) if "b" in parts else 0):
            for c in range(cfr):
                A(f"""    %lv_b{wi}{c}_{p} = vector.load {"%w_cur_st" if dbuf else "%w_stage_st"}[%lv_bn{wi}{c}, %lv_bk{p}] : view<{TW}x{PX}xi8> -> vector<16xi8>
    %lv_bw{wi}{c}_{p} = vector.bitcast %lv_b{wi}{c}_{p} : vector<16xi8> to vector<4xi32>""")
                for q in range(2):
                    A(f"""    %lv_br{wi}{c}_{2 * p + q} = vector.slice %lv_bw{wi}{c}_{p}[{2 * q}] : vector<4xi32> -> vector<2xi32>
    %rf{wi}{c}_{2 * p + q} = vector.fragment<rhs> %lv_br{wi}{c}_{2 * p + q} shape [%c16, %c16] using {{schema = %b_schema : encoding<schema>}} : vector<2xi32>""")

    NKK = KS // 16
    if lv:
        if dbuf:
            A(f"    %w_cur_st = buffer.view %w_stage[%cur_off] : buffer -> view<{TW}x{PX}xi8>")
        # lvs 1: software-pipelined, the loads of pair p + 1 are issued before the WMMAs of pair p; lvs 2: only the activation loads run ahead
        pre = "ab" if lvs == 1 else "a"
        if lvs:
            lv_loads(0, pre)
        for p in range(KS // 32):
            if not lvs:
                lv_loads(p)
            else:
                if lvs == 2:
                    lv_loads(p, "b")
                if p + 1 < KS // 32:
                    lv_loads(p + 1, pre)
            frag_mma(2 * p)
            frag_mma(2 * p + 1)
        if lvpf:   # the next step's first activation pairs, before the barrier (carried)
            A("    %lv_kn = index.mul %k_next, %c128 : index\n    %lv_kno = index.scale %lv_kn, %one_off : index, offset -> offset")
            for r in range(RF):
                for p in range(lvpf):
                    A(lv_aload(f"lv_na{r}_{p}", r, p, "%lv_kno"))
    elif swp:
        frag_loads(0)
    for kk in range(0 if lv else NKK):
        if swp:
            if kk + 1 < NKK:
                frag_loads(kk + 1)
        else:
            frag_loads(kk)
        frag_mma(kk)
    if dbuf:
        A(stage_text(wview="%w_nxt_st"))
    A("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    A("    scf.yield " + ", ".join(cur[a] for a in accs) + ", " + ", ".join(c[2] for c in carried) + " : " +
      ", ".join("vector<8xf32>" for _ in accs) + ", " + ", ".join(c[3] for c in carried) + "\n  }")
    # --- epilogue: scale rows by scales[row, 1] (through LDS: the lane holding result row i needs row i's value)
    A(f'''  %inv_view = buffer.view %x_stage[%origin] : buffer -> view<{BM}x16xf32>
  %inv_row_value = index.add %row_origin, {BMI} : index
  %inv_row = index.assume %inv_row_value [lt(%inv_row_value, %rows)] : index
  %inverse = view.load %s_view[%inv_row, %c1] : view<[%rows]x2xf32> -> f32
  %inverse16 = vector.splat %inverse : vector<16xf32>
  vector.store %inverse16, %inv_view[{BMI}, %c0] : vector<16xf32>, view<{BM}x16xf32>
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)''')
    for r in range(RF):
        A(f"  %inv{r} = vector.fragment.load<result> %inv_view[{rowv[r]}, %c0] shape [%c16, %c16] : view<{BM}x16xf32> -> vector<8xf32>")
    A(f'''  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %tile_first = index.madd %row_origin, %n, %out_origin : index
  %tile_offset = index.scale %tile_first, {'%two' if yf16 else '%four'} : index, offset -> offset
  %y_view = buffer.view {"%ypsel" if pair else "%y"}[%tile_offset] : buffer -> view<{BM}x{TN}x{YT}, %y_layout>''')
    for r in range(RF):
        for c in range(cfr):
            row = rowv[r]
            A(f"  %oc{r}{c}_c = index.constant {c * 16} : index\n  %oc{r}{c} = index.add %wave_col, %oc{r}{c}_c : index")
            if two_w:
                A(f'''  %g{r}{c} = vector.mulf %o0r{r}c{c}, %inv{r} : vector<8xf32>
  %u{r}{c} = vector.mulf %o1r{r}c{c}, %inv{r} : vector<8xf32>
  %sg{r}{c} = vector.siluf<afn> %g{r}{c} : vector<8xf32>
  %v{r}{c} = vector.mulf %sg{r}{c}, %u{r}{c} : vector<8xf32>''')
            else:
                A(f"  %v{r}{c}_s = vector.mulf %o0r{r}c{c}, %inv{r} : vector<8xf32>\n  %v{r}{c} = vector.addf %v{r}{c}_s, %zero8 : vector<8xf32>")
            if yf16:
                A(f"  %vh{r}{c} = vector.fptrunc %v{r}{c} : vector<8xf32> to vector<8xf16>")
            A(f"  vector.fragment.store<result> %{'vh' if yf16 else 'v'}{r}{c}, %y_view[{row}, %oc{r}{c}] shape [%c16, %c16] : vector<8x{YT}>, view<{BM}x{TN}x{YT}, %y_layout>")
    A("  kernel.return\n}\n")
    text = "\n".join(L)
    from mxfp4s_atiled import enabled as atiled_enabled, gemm as atiled_gemm
    if atiled_enabled():
        assert a_direct and not (lv or a_pf or swp), "A-tiled requires the original K-order direct-load GEMM"
        text = atiled_gemm(text)
    return text


def _bf8_decode_text(j, q, e, row, col, view, TW, PX, order, sfx, ind="    "):
    """the bf8 decode of one MXFP4 block (raw nibble words `q` vector<4xi32>, E8M0 `e` i32) stored at view[row, col + 32 j] (view<TW x PX x i8>)"""
    t = [f"%em{j}{sfx} = scalar.subi {e}, %c107 : i32",
         f"%ea{j}{sfx} = scalar.shli %em{j}{sfx}, %cs2 : i32",
         f"%ea4_{j}{sfx} = scalar.muli %ea{j}{sfx}, %ones : i32",
         f"%ea4z_{j}{sfx} = scalar.andi %ea4_{j}{sfx}, %lo_mask : i32",
         f"%tlo{j}{sfx} = scalar.addi %ea4z_{j}{sfx}, %tab_lo_c : i32",
         f"%thi{j}{sfx} = scalar.addi %ea4_{j}{sfx}, %tab_hi_c : i32",
         f"%tab2_{j}{sfx} = vector.from_elements %tlo{j}{sfx}, %thi{j}{sfx} : vector<2xi32>",
         f"%tab{j}{sfx} = vector.bitcast %tab2_{j}{sfx} : vector<2xi32> to vector<8xi8>"]
    for i in range(4):
        n = f"{j}{i}{sfx}"
        t += [f"%qw{n}_1 = vector.slice {q}[{i}] : vector<4xi32> -> vector<1xi32>",
              f"%qw{n} = vector.bitcast %qw{n}_1 : vector<1xi32> to vector<4xi8>",
              f"%lo_i{n} = vector.andi %qw{n}, %m7 : vector<4xi8>",
              f"%hs_i{n} = vector.shrui %qw{n}_1, %m4i : vector<1xi32>",
              f"%hs_b{n} = vector.bitcast %hs_i{n} : vector<1xi32> to vector<4xi8>",
              f"%hi_i{n} = vector.andi %hs_b{n}, %m7 : vector<4xi8>",
              f"%lo_m{n} = vector.table.lookup %tab{j}{sfx}[%lo_i{n}] : vector<8xi8>, vector<4xi8> -> vector<4xi8>",
              f"%hi_m{n} = vector.table.lookup %tab{j}{sfx}[%hi_i{n}] : vector<8xi8>, vector<4xi8> -> vector<4xi8>",
              f"%ls_i{n} = vector.shli %qw{n}_1, %m4i : vector<1xi32>",
              f"%ls_m{n} = vector.andi %ls_i{n}, %m80i : vector<1xi32>",
              f"%ls_b{n} = vector.bitcast %ls_m{n} : vector<1xi32> to vector<4xi8>",
              f"%hs_m{n} = vector.andi %qw{n}_1, %m80i : vector<1xi32>",
              f"%hs_s{n} = vector.bitcast %hs_m{n} : vector<1xi32> to vector<4xi8>",
              f"%lo_f{n} = vector.ori %lo_m{n}, %ls_b{n} : vector<4xi8>",
              f"%hi_f{n} = vector.ori %hi_m{n}, %hs_s{n} : vector<4xi8>",
              f"%lo_w{n} = vector.bitcast %lo_f{n} : vector<4xi8> to vector<1xi32>",
              f"%hi_w{n} = vector.bitcast %hi_f{n} : vector<4xi8> to vector<1xi32>"]
    js = f"{j}{sfx}"
    t += [f"%lo4_{js} = vector.concat<0> %lo_w{j}0{sfx}, %lo_w{j}1{sfx}, %lo_w{j}2{sfx}, %lo_w{j}3{sfx} : vector<1xi32>, vector<1xi32>, vector<1xi32>, vector<1xi32> -> vector<4xi32>",
          f"%hi4_{js} = vector.concat<0> %hi_w{j}0{sfx}, %hi_w{j}1{sfx}, %hi_w{j}2{sfx}, %hi_w{j}3{sfx} : vector<1xi32>, vector<1xi32>, vector<1xi32>, vector<1xi32> -> vector<4xi32>",
          f"%lo16_{js} = vector.bitcast %lo4_{js} : vector<4xi32> to vector<16xi8>",
          f"%hi16_{js} = vector.bitcast %hi4_{js} : vector<4xi32> to vector<16xi8>",
          f"%wcl{js}_a = index.constant {32 * j} : index",
          f"%wcl{js} = index.add {col}, %wcl{js}_a : index"]
    if order == "pairs":
        t += [f"%blk{js} = vector.interleave<0> %lo16_{js}, %hi16_{js} : vector<16xi8>, vector<16xi8> -> vector<32xi8>",
              f"vector.store %blk{js}, {view}[{row}, %wcl{js}] : vector<32xi8>, view<{TW}x{PX}xi8>"]
    else:
        t += [f"%wch{js} = index.add %wcl{js}, %c16 : index",
              f"vector.store %lo16_{js}, {view}[{row}, %wcl{js}] : vector<16xi8>, view<{TW}x{PX}xi8>",
              f"vector.store %hi16_{js}, {view}[{row}, %wch{js}] : vector<16xi8>, view<{TW}x{PX}xi8>"]
    return "\n".join(ind + x for x in t)


def gemm_mxfp4s_f8_ws(mode, prefix="g", order="pairs", npw=4, rf=None, pad=16, kmult=1, lvs=False):
    """Wave-specialized variant of gemm_mxfp4s_f8_m256 (same launch ABI, name and results): the workgroup has the WMMA waves of the 256 x 128
    (swiglu 256 x 64 x 2) tile plus `npw` producer waves. The weight stage is double-buffered in LDS: during K step s the WMMA waves multiply
    buffer s % 2 (activation fragments from global memory) while the producer waves decode step s + 1 into the other buffer (their raw nibble and
    scale loads run one more step ahead); one workgroup barrier per step. The WMMA waves never decode, the producers never multiply."""
    two_w = mode == "swiglu"
    BM = 256
    TN = 64 if two_w else 128
    TW = TN * (2 if two_w else 1)
    RF = rf or (2 if two_w else 4)
    cfr = (8 if two_w else 16) // RF
    WR, WC = BM // (16 * RF), TN // (16 * cfr)
    NWV = WR * WC
    NM = NWV * 32
    NPT = npw * 32
    NT = NM + NPT
    HALVES = NPT // TW
    KS = 128 * kmult
    PX = KS + pad
    NB = (KS // 32) // HALVES
    assert WR * 16 * RF == BM and WC * 16 * cfr == TN and HALVES in (1, 2) and HALVES * TW == NPT and NM % NPT == 0 and NB >= 1 and NT <= 1024
    WB = TW * PX
    assert 2 * WB >= BM * 64
    name = f"{prefix}_gemm_mxfp4s_{mode}_f8_m256_rdna4"
    weights = ["w", "w2"] if two_w else ["w"]
    accs = [f"{wi}r{r}c{c}" for wi in range(len(weights)) for r in range(RF) for c in range(cfr)]
    L = []
    A = L.append
    A(f'''kernel.def target(@rdna4) export("{name}") @{name}(%rows: index, %n: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant {BM} : index
  %cnt = index.constant {NT} : index
  %tn = index.constant {TN} : index
  %row_tiles = index.div %rows, %c256 : index
  %out_tiles = index.div %n, %tn : index
  kernel.launch.config workgroups(%row_tiles, %out_tiles, %c1) workgroup_size(%cnt, %c1, %c1) : index
}} launch(%rows_arg: index, %k_arg: index, %n_arg: index, %x: buffer, %scales: buffer, {", ".join(f"%{w}: buffer, %{w}s: buffer" for w in weights)}, %y: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, {BM}, 8192), multiple_of(%rows_arg, {BM})] : index
  %k = index.assume %k_arg [range(%k_arg, {KS}, 65536), multiple_of(%k_arg, {KS})] : index
  %n = index.assume %n_arg [range(%n_arg, {TN}, 262144), multiple_of(%n_arg, {TN})] : index
  %origin = index.constant 0 : offset
  %four = index.constant 4 : offset
  %one_off = index.constant 1 : offset
  %o0 = index.constant 0 : offset''')
    for j in range(1, NB):
        A(f"  %o{16 * j} = index.constant {16 * j} : offset")
    for j in range(NB):
        A(f"  %cj{j} = index.constant {j} : index")
    A(f'''  %c0 = index.constant 0 : index
  %c2 = index.constant 2 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %cks2 = index.constant {KS // 2} : index
  %cnb16 = index.constant {NB * 16} : index
  %cks32 = index.constant {KS // 32} : index
  %cnb = index.constant {NB} : index
  %c128 = index.constant {KS} : index
  %c1 = index.constant 1 : index
  %c256 = index.constant {BM} : index
  %tn = index.constant {TN} : index
  %row_tile_count = index.div %rows, %c256 : index
  %out_tile_count = index.div %n, %tn : index
  %row_group_id = kernel.workgroup.id<x> : index
  %out_group_id = kernel.workgroup.id<y> : index
  %row_group = index.assume %row_group_id [lt(%row_group_id, %row_tile_count)] : index
  %out_group = index.assume %out_group_id [lt(%out_group_id, %out_tile_count)] : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, {NT - 1})] : index
  %subgroup = kernel.subgroup.id : index
  %bounded_subgroup = index.assume %subgroup [range(%subgroup, 0, {NT // 32 - 1})] : index
  %cnwv = index.constant {NWV} : index
  %is_mma = index.cmp ult, %bounded_subgroup, %cnwv : index
  %is_prod = index.cmp uge, %bounded_subgroup, %cnwv : index
  %cnwv1 = index.constant {NWV - 1} : index
  %msub_v = index.min %bounded_subgroup, %cnwv1 : index
  %msub = index.assume %msub_v [range(%msub_v, 0, {NWV - 1})] : index
  %cwc = index.constant {WC} : index
  %cwr = index.constant {16 * RF} : index
  %wave_row_group = index.div %msub, %cwc : index
  %wave_col_group = index.rem %msub, %cwc : index
  %wave_row_value = index.mul %wave_row_group, %cwr : index
  %wave_row = index.assume %wave_row_value [range(%wave_row_value, 0, {BM - 16 * RF})] : index
  %wave_cols = index.constant {16 * cfr} : index
  %wave_col_value = index.mul %wave_col_group, %wave_cols : index
  %wave_col = index.assume %wave_col_value [range(%wave_col_value, 0, {TN - 16 * cfr})] : index''')
    for r in range(1, RF):
        A(f"  %wave_row{r}_c = index.constant {16 * r} : index\n  %wave_row{r} = index.add %wave_row, %wave_row{r}_c : index")
    A(f'''  %row_origin = index.mul %row_group, %c256 : index
  %out_origin = index.mul %out_group, %tn : index
  %blocks = index.div %k, %c32 : index
  %k_half = index.div %k, %c2 : index
  %k_half_off = index.scale %k_half, %one_off : index, offset -> offset
  %ks_off = index.scale %blocks, %one_off : index, offset -> offset
  %s_view = buffer.view %scales[%origin] : buffer -> view<[%rows]x2xf32>
  %w_bytes = index.constant {WB} : offset
  %w_alloc = index.constant {2 * WB} : offset
  %w_stage = buffer.alloca<workgroup> align(16) %w_alloc : buffer
  %a_schema = encoding.define #encoding.operand<element_format=f8e4m3, payload_elements=8, payload_registers=2> : encoding<schema>
  %b_schema = encoding.define #encoding.operand<element_format=f8e5m2, payload_elements=8, payload_registers=2> : encoding<schema>
  %a_layout = encoding.layout.dense : encoding<layout>
  %b_layout = encoding.layout.strided [1, {PX}] : encoding<layout>
  %a_storage = encoding.define #encoding.storage {{layout = %a_layout : encoding<layout>, schema = %a_schema : encoding<schema>}} : encoding<storage>
  %b_storage = encoding.define #encoding.storage {{layout = %b_layout : encoding<layout>, schema = %b_schema : encoding<schema>}} : encoding<storage>
  %x_g8 = buffer.view %x[%origin] : buffer -> view<[%rows]x[%k]xf8E4M3, %a_storage>
  // producer thread p = workitem mod {NPT} decodes weight row p / {HALVES}, columns (p mod {HALVES}) * {KS // 2} of each step
  %cnpt = index.constant {NPT} : index
  %pitem = index.rem %workitem, %cnpt : index
  %chalves = index.constant {HALVES} : index
  %stage_row = index.div %pitem, %chalves : index
  %stage_half = index.rem %pitem, %chalves : index
  %stage_col = index.mul %stage_half, %cks2 : index''')
    if two_w:
        A(f'''  %half_tw = index.constant {TN} : index
  %w_is_up = index.cmp uge, %stage_row, %half_tw : index
  %w_local_up = index.sub %stage_row, %half_tw : index
  %w_local = scf.select %w_is_up, %w_local_up, %stage_row : index
  %w_out_row_value = index.add %out_origin, %w_local : index
  %wsel = scf.select %w_is_up, %w2, %w : buffer
  %wssel = scf.select %w_is_up, %w2s, %ws : buffer''')
    else:
        A("  %w_out_row_value = index.add %out_origin, %stage_row : index")
    A(f'''  %w_out_row = index.assume %w_out_row_value [lt(%w_out_row_value, %n)] : index
  %w_row_origin = index.scale %w_out_row, %k_half_off : index, offset -> offset
  %ws_row_origin = index.scale %w_out_row, %ks_off : index, offset -> offset
  %zero8 = vector.splatf 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %k_steps = index.div %k, %c128 : index
  %k_steps_m1 = index.sub %k_steps, %c1 : index
  %c107 = scalar.constant 107 : i32
  %cs2 = scalar.constant 2 : i32
  %ones = scalar.constant 16843009 : i32
  %lo_mask = scalar.constant -256 : i32
  %tab_lo_c = scalar.constant 235669504 : i32
  %tab_hi_c = scalar.constant 370414096 : i32
  %m7 = vector.splati 7 : vector<4xi8>
  %m80i = vector.splati -2139062144 : vector<1xi32>
  %m4i = vector.splati 4 : vector<1xi32>
  %c15 = index.constant 15 : index
  %rows_m15 = index.sub %rows, %c15 : index''')
    for r in range(RF):
        A(f"  %xrow{r}_v = index.add %row_origin, {'%wave_row' if r == 0 else f'%wave_row{r}'} : index\n  %xrow{r} = index.assume %xrow{r}_v [lt(%xrow{r}_v, %rows_m15)] : index")
    if lvs:   # lane-vector operand loads (see gemm_mxfp4s_f8_m256 lv), the next K pair's loads issued before this pair's WMMAs
        A("""  %lv_c32 = index.constant 32 : index
  %lv_lane = index.rem %workitem, %lv_c32 : index
  %lv_l16 = index.rem %lv_lane, %c16 : index
  %lv_h = index.div %lv_lane, %c16 : index
  %lv_h16 = index.mul %lv_h, %c16 : index""")
        for r in range(RF):
            A(f"""  %lv_xr{r} = index.add %xrow{r}, %lv_l16 : index
  %lv_xm{r} = index.mul %lv_xr{r}, %k : index
  %lv_xo{r}_i = index.add %lv_xm{r}, %lv_h16 : index
  %lv_xo{r} = index.scale %lv_xo{r}_i, %one_off : index, offset -> offset""")
        for p in range(KS // 32):
            A(f"  %lv_kc{p} = index.constant {32 * p} : index\n  %lv_bk{p} = index.add %lv_h16, %lv_kc{p} : index")
        for wi in range(len(weights)):
            for c in range(cfr):
                A(f"  %lv_bn{wi}{c}_c = index.constant {wi * TN + c * 16} : index\n  %lv_bn{wi}{c}_a = index.add %wave_col, %lv_bn{wi}{c}_c : index\n  %lv_bn{wi}{c} = index.add %lv_bn{wi}{c}_a, %lv_l16 : index")
    wsrc = "%wsel" if two_w else "%w"
    ssrc = "%wssel" if two_w else "%ws"

    def loads(sfx, kstep, ind):
        t = [f"%wk{sfx}_a = index.mul {kstep}, %cks2 : index", f"%wk{sfx}_b = index.mul %stage_half, %cnb16 : index",
             f"%wk{sfx} = index.add %wk{sfx}_a, %wk{sfx}_b : index", f"%wk{sfx}_o = index.scale %wk{sfx}, %one_off : index, offset -> offset",
             f"%wkb{sfx} = index.add %w_row_origin, %wk{sfx}_o : offset", f"%sb{sfx}_a = index.mul {kstep}, %cks32 : index",
             f"%sb{sfx}_b = index.mul %stage_half, %cnb : index", f"%sb{sfx} = index.add %sb{sfx}_a, %sb{sfx}_b : index"]
        for j in range(NB):
            t += [f"%wqo{j}{sfx} = index.add %wkb{sfx}, %o{16 * j} : offset",
                  f"%wq{j}{sfx}_v = buffer.view {wsrc}[%wqo{j}{sfx}] : buffer -> view<4xi32>",
                  f"%wq{j}{sfx} = vector.load %wq{j}{sfx}_v[0] : view<4xi32> -> vector<4xi32>",
                  f"%sbj{j}{sfx} = index.add %sb{sfx}, %cj{j} : index",
                  f"%sbo{j}{sfx}_a = index.scale %sbj{j}{sfx}, %one_off : index, offset -> offset",
                  f"%sbo{j}{sfx} = index.add %ws_row_origin, %sbo{j}{sfx}_a : offset",
                  f"%we{j}{sfx}_v = buffer.view {ssrc}[%sbo{j}{sfx}] : buffer -> view<1xi8>",
                  f"%we{j}{sfx}_b = view.load %we{j}{sfx}_v[0] : view<1xi8> -> i8",
                  f"%we{j}{sfx} = scalar.extui %we{j}{sfx}_b : i8 to i32"]
        return "\n".join(ind + x for x in t)

    # prologue: every thread loads steps 0 and 1 (the WMMA waves' copies are unused); the producers decode step 0 into buffer 0
    A(loads("p0", "%c0", "  "))
    A("  %k_one = index.min %c1, %k_steps_m1 : index")
    A(loads("p1", "%k_one", "  "))
    A(f"  %w_buf0 = buffer.view %w_stage[%origin] : buffer -> view<{TW}x{PX}xi8>\n  scf.if %is_prod {{")
    for j in range(NB):
        A(_bf8_decode_text(j, f"%wq{j}p0", f"%we{j}p0", "%stage_row", "%stage_col", "%w_buf0", TW, PX, order, "_pro"))
    A("  }\n  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    carried = []
    for j in range(NB):
        carried += [(f"%cr_q{j}", f"%wq{j}p1", "vector<4xi32>"), (f"%cr_e{j}", f"%we{j}p1", "i32")]
    acc_t = ", ".join("vector<8xf32>" for _ in accs)
    car_t = ", ".join(c[2] for c in carried)
    A("  " + ", ".join(f"%o{a}" for a in accs) + ", " + ", ".join(f"%unused_{i}" for i in range(len(carried))) +
      " = scf.for %step = [%c0 to %k_steps step %c1](" + ", ".join(f"%i{a} = %init : vector<8xf32>" for a in accs) + ", " +
      ", ".join(f"{c[0]} = {c[1]} : {c[2]}" for c in carried) + f") -> ({acc_t}, {car_t}) {{")
    A('''    %par = index.rem %step, %c2 : index
    %npar = index.sub %c1, %par : index
    %cur_off = index.scale %par, %w_bytes : index, offset -> offset
    %nxt_off = index.scale %npar, %w_bytes : index, offset -> offset''')
    A("    " + ", ".join(f"%f{a}" for a in accs) + ", " + ", ".join(f"%fc{i}" for i in range(len(carried))) + f" = scf.if %is_mma -> ({acc_t}, {car_t}) {{")
    A(f"      %w_cur_f8 = buffer.view %w_stage[%cur_off] : buffer -> view<{PX}x{TW}xf8E5M2, %b_storage>\n      %kbase = index.mul %step, %c128 : index")
    cur = {a: f"%i{a}" for a in accs}

    def mma(kk):
        for wi in range(len(weights)):
            for c in range(cfr):
                for r in range(RF):
                    a = f"{wi}r{r}c{c}"
                    A(f"      %m{a}_{kk} = vector.mma %l{r}_{kk}, %rf{wi}{c}_{kk}, {cur[a]} : vector<2xi32>, vector<2xi32>, vector<8xf32>")
                    cur[a] = f"%m{a}_{kk}"

    def lv_loads(p):
        for r in range(RF):
            A(f"""      %lv_a{r}_{p} = vector.load %lv_xv{r}[%lv_kc{p}] : view<{KS}xi8> -> vector<16xi8>
      %lv_aw{r}_{p} = vector.bitcast %lv_a{r}_{p} : vector<16xi8> to vector<4xi32>""")
            for q in range(2):
                A(f"""      %lv_ar{r}_{2 * p + q} = vector.slice %lv_aw{r}_{p}[{2 * q}] : vector<4xi32> -> vector<2xi32>
      %l{r}_{2 * p + q} = vector.fragment<lhs> %lv_ar{r}_{2 * p + q} shape [%c16, %c16] using {{schema = %a_schema : encoding<schema>}} : vector<2xi32>""")
        for wi in range(len(weights)):
            for c in range(cfr):
                A(f"""      %lv_b{wi}{c}_{p} = vector.load %w_cur_st[%lv_bn{wi}{c}, %lv_bk{p}] : view<{TW}x{PX}xi8> -> vector<16xi8>
      %lv_bw{wi}{c}_{p} = vector.bitcast %lv_b{wi}{c}_{p} : vector<16xi8> to vector<4xi32>""")
                for q in range(2):
                    A(f"""      %lv_br{wi}{c}_{2 * p + q} = vector.slice %lv_bw{wi}{c}_{p}[{2 * q}] : vector<4xi32> -> vector<2xi32>
      %rf{wi}{c}_{2 * p + q} = vector.fragment<rhs> %lv_br{wi}{c}_{2 * p + q} shape [%c16, %c16] using {{schema = %b_schema : encoding<schema>}} : vector<2xi32>""")
    if lvs:
        A(f"      %w_cur_st = buffer.view %w_stage[%cur_off] : buffer -> view<{TW}x{PX}xi8>\n      %lv_kb = index.scale %kbase, %one_off : index, offset -> offset")
        for r in range(RF):
            A(f"      %lv_xp{r} = index.add %lv_xo{r}, %lv_kb : offset\n      %lv_xv{r} = buffer.view %x[%lv_xp{r}] : buffer -> view<{KS}xi8>")
        lv_loads(0)
        for p in range(KS // 32):
            if p + 1 < KS // 32:
                lv_loads(p + 1)
            mma(2 * p)
            mma(2 * p + 1)
    for kk in range(0 if lvs else KS // 16):
        A(f"      %fk{kk} = index.constant {kk * 16} : index\n      %kg{kk} = index.add %kbase, %fk{kk} : index")
        for r in range(RF):
            A(f"      %l{r}_{kk} = vector.fragment.load<lhs> %x_g8[%xrow{r}, %kg{kk}] shape [%c16, %c16] : view<[%rows]x[%k]xf8E4M3, %a_storage> -> vector<2xi32>")
        for wi in range(len(weights)):
            for c in range(cfr):
                A(f'''      %wr{wi}{c}_{kk}_c = index.constant {wi * TN + c * 16} : index
      %wr{wi}{c}_{kk} = index.add %wave_col, %wr{wi}{c}_{kk}_c : index
      %rf{wi}{c}_{kk} = vector.fragment.load<rhs> %w_cur_f8[%fk{kk}, %wr{wi}{c}_{kk}] shape [%c16, %c16] : view<{PX}x{TW}xf8E5M2, %b_storage> -> vector<2xi32>''')
        mma(kk)
    A("      scf.yield " + ", ".join(cur[a] for a in accs) + ", " + ", ".join(c[0] for c in carried) + f" : {acc_t}, {car_t}\n    }} else {{")
    A(f"      %w_nxt = buffer.view %w_stage[%nxt_off] : buffer -> view<{TW}x{PX}xi8>")
    for j in range(NB):
        A(_bf8_decode_text(j, f"%cr_q{j}", f"%cr_e{j}", "%stage_row", "%stage_col", "%w_nxt", TW, PX, order, "", ind="      "))
    A("      %k_n2_raw = index.add %step, %c2 : index\n      %k_n2 = index.min %k_n2_raw, %k_steps_m1 : index")
    A(loads("n", "%k_n2", "      "))
    nxt = []
    for j in range(NB):
        nxt += [f"%wq{j}n", f"%we{j}n"]
    A("      scf.yield " + ", ".join(f"%i{a}" for a in accs) + ", " + ", ".join(nxt) + f" : {acc_t}, {car_t}\n    }}")
    A("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    A("    scf.yield " + ", ".join(f"%f{a}" for a in accs) + ", " + ", ".join(f"%fc{i}" for i in range(len(carried))) + f" : {acc_t}, {car_t}\n  }}")
    # epilogue (WMMA waves): rows scaled by scales[row, 1] through LDS (buffer 0 of the weight stage)
    rowv = ["%wave_row"] + [f"%wave_row{r}" for r in range(1, RF)]
    A(f'''  %cbm = index.constant {BM} : index
  %mitem_v = index.rem %workitem, %cbm : index
  %mitem = index.assume %mitem_v [range(%mitem_v, 0, {BM - 1})] : index
  %inv_view = buffer.view %w_stage[%origin] : buffer -> view<{BM}x16xf32>
  scf.if %is_mma {{
    %inv_row_value = index.add %row_origin, %mitem : index
    %inv_row = index.assume %inv_row_value [lt(%inv_row_value, %rows)] : index
    %inverse = view.load %s_view[%inv_row, %c1] : view<[%rows]x2xf32> -> f32
    %inverse16 = vector.splat %inverse : vector<16xf32>
    vector.store %inverse16, %inv_view[%mitem, %c0] : vector<16xf32>, view<{BM}x16xf32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  scf.if %is_mma {{''')
    for r in range(RF):
        A(f"    %inv{r} = vector.fragment.load<result> %inv_view[{rowv[r]}, %c0] shape [%c16, %c16] : view<{BM}x16xf32> -> vector<8xf32>")
    A(f'''    %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
    %tile_first = index.madd %row_origin, %n, %out_origin : index
    %tile_offset = index.scale %tile_first, %four : index, offset -> offset
    %y_view = buffer.view %y[%tile_offset] : buffer -> view<{BM}x{TN}xf32, %y_layout>''')
    for r in range(RF):
        for c in range(cfr):
            A(f"    %oc{r}{c}_c = index.constant {c * 16} : index\n    %oc{r}{c} = index.add %wave_col, %oc{r}{c}_c : index")
            if two_w:
                A(f'''    %g{r}{c} = vector.mulf %o0r{r}c{c}, %inv{r} : vector<8xf32>
    %u{r}{c} = vector.mulf %o1r{r}c{c}, %inv{r} : vector<8xf32>
    %sg{r}{c} = vector.siluf<afn> %g{r}{c} : vector<8xf32>
    %v{r}{c} = vector.mulf %sg{r}{c}, %u{r}{c} : vector<8xf32>''')
            else:
                A(f"    %v{r}{c} = vector.mulf %o0r{r}c{c}, %inv{r} : vector<8xf32>")
            A(f"    vector.fragment.store<result> %v{r}{c}, %y_view[{rowv[r]}, %oc{r}{c}] shape [%c16, %c16] : vector<8xf32>, view<{BM}x{TN}xf32, %y_layout>")
    A("  }\n  kernel.return\n}\n")
    return "\n".join(L)


PAIR_TEXT = """  %out_tile_count2 = index.add %out_tile_count, %out_tile_count : index
  %out_group_p = index.assume %out_group_id [lt(%out_group_id, %out_tile_count2)] : index
  %second = index.cmp uge, %out_group_p, %out_tile_count : index
  %out_group_s = index.sub %out_group_p, %out_tile_count : index
  %out_group_v = scf.select %second, %out_group_s, %out_group_p : index
  %out_group = index.assume %out_group_v [lt(%out_group_v, %out_tile_count)] : index
  %wpsel = scf.select %second, %w2, %w : buffer
  %wspsel = scf.select %second, %w2s, %ws : buffer
  %ypsel = scf.select %second, %y2, %y : buffer"""


# -- the producers' fp8 twins ------------------------------------------------------------------------------------------
F8_SCALE = """  // fp8 activations: per-row power-of-two scale 2^(7 - e) (e: exponent of the row max, so the max lands in [128, 256)), the inverse
  // 2^(e - 7) carries the weights' 2^-%(shift)d too; a dead row (f16 scale 0) gets (0, 0)
  %%f8_c261 = scalar.constant 261 : i32
  %%f8_c15 = scalar.constant %(inv_c)d : i32
  %%f8_c1 = scalar.constant 1 : i32
  %%f8_c254 = scalar.constant 254 : i32
  %%f8_se = scalar.subi %%f8_c261, %%biased : i32
  %%f8_sl = scalar.maxsi %%f8_se, %%f8_c1 : i32
  %%f8_sf = scalar.minsi %%f8_sl, %%f8_c254 : i32
  %%f8_sb = scalar.shli %%f8_sf, %%c23 : i32
  %%f8_sv = scalar.bitcast %%f8_sb : i32 to f32
  %%f8_ie = scalar.subi %%biased, %%f8_c15 : i32
  %%f8_il = scalar.maxsi %%f8_ie, %%f8_c1 : i32
  %%f8_if = scalar.minsi %%f8_il, %%f8_c254 : i32
  %%f8_ib = scalar.shli %%f8_if, %%c23 : i32
  %%f8_iv = scalar.bitcast %%f8_ib : i32 to f32
  %%f8_alive = scalar.cmpf one, %%scale, %%zero_f : f32
  %%f8_scale = scf.select %%f8_alive, %%f8_sv, %%zero_f : f32
  %%f8_inv = scf.select %%f8_alive, %%f8_iv, %%zero_f : f32
""" % {"shift": MXFP4_F8_SHIFT, "inv_c": 7 + MXFP4_F8_SHIFT}

_PRODUCER = re.compile(r"^g_(row_scale_f16|post_add_norm_rs)(_live)?(_lv)?$")


def _sub1(t, old, new, name):
    assert t.count(old) == 1, (name, old, t.count(old))
    return t.replace(old, new, 1)


def _w16_guard(store):
    """The f16 operand store of a producer twin, only when the launch's %w16 flag is set."""
    return ("    %w16_zero = index.constant 0 : index\n    %w16_on = index.cmp ne, %w16, %w16_zero : index\n    scf.if %w16_on {\n  "
            + store + "    }\n")


def f8_producer(text):
    """The `<K>_f8_rdna4` kernel of a row-scale producer K (g_row_scale_f16[_live][_lv], g_post_add_norm_rs[_live][_lv]; text: one
    kernel.def .. kernel.return block): K plus two outputs, x8 (the row as fp8 e4m3, scaled by F8_SCALE's scale) and s8 ((scale, inverse)
    of it): launch(..., %x16: buffer) -> launch(..., %x16: buffer, %x8: buffer, %s8: buffer, %w16: index). The scales stay as K writes them;
    x16 is written only when %w16 != 0 (the epochs that do not take the fp8 GEMM, whose f16 GEMMs read it)."""
    m = re.search(r'kernel\.def export\("(\w+)"\) @(\w+)\(', text)
    old = m.group(1)
    assert _PRODUCER.match(old), old
    new = old + "_f8_rdna4"
    t = text.replace(f'kernel.def export("{old}") @{old}(', f'kernel.def target(@rdna4) export("{new}") @{new}(', 1)
    row_scale = "%x16: buffer) {" in t
    t = _sub1(t, "%x16: buffer) {" if row_scale else "%h16: buffer) {",
              "%x16: buffer, %x8: buffer, %s8: buffer, %w16: index) {" if row_scale else "%h16: buffer, %x8: buffer, %s8: buffer, %w16: index) {", old)
    if row_scale:
        view = re.search(r"  %h4_view = buffer\.view %x16\[%zero\] : buffer -> view<\[%rows\]x\[%k4\]x4xf16>\n", t)
        t = _sub1(t, view.group(0), view.group(0) + "  %x8_view = buffer.view %x8[%zero] : buffer -> view<[%rows]x[%k4]x4xf8E4M3>\n", old)
        t = _sub1(t, "  %scale4 = vector.splat %scale : vector<4xf32>\n",
                  F8_SCALE + "  %scale4 = vector.splat %scale : vector<4xf32>\n  %f8_scale4 = vector.splat %f8_scale : vector<4xf32>\n", old)
        st = "    vector.store %hv, %h4_view[%row, %bc, %c0] : vector<4xf16>, view<[%rows]x[%k4]x4xf16>\n"
        t = _sub1(t, st, _w16_guard(st) + """    %sv8 = vector.mulf %v, %f8_scale4 : vector<4xf32>
    %q8 = vector.fptrunc %sv8 : vector<4xf32> to vector<4xf8E4M3>
    vector.store %q8, %x8_view[%row, %bc, %c0] : vector<4xf8E4M3>, view<[%rows]x[%k4]x4xf8E4M3>
""", old)
        row, c1 = "%row", "%c1"
        s_store = "    view.store %inverse, %s_view[%row, %c1] : f32, view<[%rows]x2xf32>\n"
        rows = "rows"
    else:
        t = _sub1(t, "  %scale32 = vector.splat %scale : vector<32xf32>\n",
                  F8_SCALE + "  %scale32 = vector.splat %scale : vector<32xf32>\n  %f8_scale32 = vector.splat %f8_scale : vector<32xf32>\n", old)
        hval = re.search(r"%hsv = vector\.mulf (%\w+), %scale32 : vector<32xf32>\n", t).group(1)
        hg = re.search(r"%h16_view = buffer\.view %h16\[%zero\] : buffer -> view<\[%tokens\]x(\d+)x32xf16>", t).group(1)
        st = re.search(r"    vector\.store %h16v, %h16_view\[%token, %bl16, %c0\] : vector<32xf16>, view<\[%tokens\]x\d+x32xf16>\n", t).group(0)
        t = _sub1(t, st, _w16_guard(st) + f"""    %hsv8 = vector.mulf {hval}, %f8_scale32 : vector<32xf32>
    %h8v = vector.fptrunc %hsv8 : vector<32xf32> to vector<32xf8E4M3>
    %h8_view = buffer.view %x8[%zero] : buffer -> view<[%tokens]x{hg}x32xf8E4M3>
    vector.store %h8v, %h8_view[%token, %bl16, %c0] : vector<32xf8E4M3>, view<[%tokens]x{hg}x32xf8E4M3>
""", old)
        row, c1 = "%token", "%c1s"
        s_store = "    view.store %inverse, %s_view[%token, %c1s] : f32, view<[%tokens]x2xf32>\n"
        rows = "tokens"
    t = _sub1(t, s_store, s_store + f"""    %s8_view = buffer.view %s8[%zero] : buffer -> view<[%{rows}]x2xf32>
    view.store %f8_scale, %s8_view[{row}, %c0] : f32, view<[%{rows}]x2xf32>
    view.store %f8_inv, %s8_view[{row}, {c1}] : f32, view<[%{rows}]x2xf32>
""", old)
    if "%lv_zf" in t:   # the live-row-skip variant: a padding row writes scale 0 and nothing else
        z = re.search(r"      view\.store %lv_zf, %lv_s\[(%\w+), %lv_c1\] : f32, view<\[%(\w+)\]x2xf32>\n", t)
        t = _sub1(t, z.group(0), z.group(0) + f"""      %lv_s8 = buffer.view %s8[%lv_zero] : buffer -> view<[%{z.group(2)}]x2xf32>
      view.store %lv_zf, %lv_s8[{z.group(1)}, %lv_c0] : f32, view<[%{z.group(2)}]x2xf32>
      view.store %lv_zf, %lv_s8[{z.group(1)}, %lv_c1] : f32, view<[%{z.group(2)}]x2xf32>
""", old)
    from mxfp4s_atiled import enabled as atiled_enabled, producer as atiled_producer
    return atiled_producer(t, row_scale) if atiled_enabled() else t


# -- the program pass ----------------------------------------------------------------------------------------------------
_LAUNCH = re.compile(r"^(?P<ind>\s*)kernel\.launch @(?P<k>\w+)\[(?P<grid>[^\]]*)\]\((?P<args>.*)\) : \[(?P<gt>[^\]]*)\]\((?P<at>.*)\)$")
_F8_OPERANDS = {"%h16": ("%h8", "%s_h8", "%s_h"), "%o16": ("%o8", "%s_o8", "%s_o"), "%f16": ("%f8", "%s_f8", "%s_f")}   # f16 operand -> (fp8, its scales, f16's scales)
_F8_LEAF = re.compile(r"^g_gemm_mxfp4s_(store)_m128$")


def _split(s):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "<[(":
            depth += 1
        elif ch in ">])":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _kernel_def(text, name):
    a = text.index(f'kernel.def export("{name}")')
    end = "  kernel.return\n}\n"
    return text[a:text.index(end, a) + len(end)]


def fuse_mxfp4s_f8(text, prefix="g"):
    """gfx12: the prefill MXFP4 GEMMs of @llm_advance run on fp8 WMMA when the epoch's rows are a multiple of 256 (%huge).
    Every producer of an f16 GEMM operand buffer (%h16 / %o16 / %f16: g_row_scale_f16*, g_post_add_norm_rs*) also writes its fp8 copy
    (%h8 / %o8 / %f8 and scales %s_h8 / %s_o8 / %s_f8) in a dual-group whose gfx12 provider is the `_f8_rdna4` twin of the producer;
    each `scf.if %big { g_gemm_mxfp4_{store,swiglu}_m128 } else { ... }` becomes `scf.if %huge { group: gfx11 the same m128 launch, gfx12 the fp8
    256-row kernel } else { as before }`. The gfx11 programs keep every launch they had. An operand buffer with a producer or a reader this
    pass does not know stays as it is (f16 path). Returns (text, number of fp8 GEMM groups)."""
    a = text.find("command.def public @llm_advance(")
    if a < 0:
        return text, 0
    b = text.find("\ncommand.def ", a + 10)
    b = len(text) if b < 0 else b + 1
    lines = text[a:b].split("\n")
    producers, readers, others = {}, {}, set()
    for i, l in enumerate(lines):
        m = _LAUNCH.match(l)
        if not m:
            continue
        k, args = m.group("k"), _split(m.group("args"))
        if _PRODUCER.match(k) and len(args) >= 2 and args[-1] in _F8_OPERANDS and args[-2] == _F8_OPERANDS[args[-1]][2]:
            producers.setdefault(args[-1], []).append(i)
        elif k.startswith("g_gemm_") and len(args) >= 5 and args[3] in _F8_OPERANDS and args[4] == _F8_OPERANDS[args[3]][2]:
            readers.setdefault(args[3], []).append(i)
        else:
            others.update(x for x in args if x in _F8_OPERANDS)
    consts = {m.group(1): int(m.group(2)) for l in lines if (m := re.match(r"\s*(%\w+) = index\.constant (\d+) : index", l))}
    # a GEMM leaf: scf.if %big { <m128 launch> } else { <base launch> } with the m128 launch an MXFP4 kernel whose K the fp8 kernel's 128-value step divides
    leaves = []
    for x, idx in readers.items():
        for i in idx:
            m = _LAUNCH.match(lines[i])
            if (x in producers and x not in others and _F8_LEAF.match(m.group("k")) and consts.get(_split(m.group("args"))[1], 1) % 128 == 0
                    and lines[i - 1].strip() == "scf.if %big {"
                    and lines[i + 1].strip() == "} else {" and lines[i + 3].strip() == "}"
                    and (n := _LAUNCH.match(lines[i + 2])) and n.group("k") == m.group("k")[:-len("_m128")] and n.group("args") == m.group("args")):
                leaves.append((i, x))
    if not leaves:
        return text, 0
    # SwiGLU triples: gate leaf, up leaf (two projections of one operand, outputs YA / YB) and the silu_mul that joins them -> one fused fp8 kernel
    def block_end(i):
        depth = 0
        for j in range(i, len(lines)):
            depth += lines[j].count("{") - lines[j].count("}")
            if depth == 0:
                return j
    leaf_at = dict(leaves)
    triples = []   # (gate leaf line, up leaf line, silu block first line, silu block last line)
    for j, l in enumerate(lines):
        m = _LAUNCH.match(l)
        if not (m and m.group("k") == "g_silu_mul_lv" and lines[j - 1].strip() == "scf.if %gemm {" and lines[j + 1].strip() == "} else {"
                and lines[j + 3].strip() == "}" and (n := _LAUNCH.match(lines[j + 2])) and n.group("k") == "g_silu_mul"):
            continue
        ya, yb = _split(m.group("args"))[2:4]
        up = [i for i in leaf_at if _split(_LAUNCH.match(lines[i]).group("args"))[-1] == yb and i < j]
        gate = [i for i in leaf_at if _split(_LAUNCH.match(lines[i]).group("args"))[-1] == ya and i < j]
        if not up or not gate:
            continue
        g, u = max(gate), max(up)   # the nearest ones before the silu_mul
        if leaf_at[u] != leaf_at[g] or g > u:
            continue
        if lines[g - 2].strip() == "scf.if %mm {" and lines[u - 2].strip() == "scf.if %mm {" and block_end(g - 2) + 1 == u - 2 and block_end(u - 2) + 1 == j - 1:
            ga, ua = _split(_LAUNCH.match(lines[g]).group("args")), _split(_LAUNCH.match(lines[u]).group("args"))
            if ga[:5] == ua[:5]:
                triples.append((g, u, j - 1, j + 3))
    in_triple = {i for g, u, _, _ in triples for i in (g, u)}
    leaves = [(i, x) for i, x in leaves if i not in in_triple]
    used = sorted({x for _, x in leaves} | {leaf_at[g] for g, _, _, _ in triples})
    # an operand whose every reader launch is replaced by an fp8 group needs its f16 copy only in the epochs that do not take the fp8 GEMM
    converted = {x: 0 for x in used}
    for _, x in leaves:
        converted[x] += 2   # the m128 launch and its else-branch twin
    for g, _, _, _ in triples:
        converted[leaf_at[g]] += 4   # gate and up leaves
    f16_optional = {x: converted[x] == len(readers[x]) for x in used}
    kernels = {}
    for x in used:
        for i in producers[x]:
            k = _LAUNCH.match(lines[i]).group("k")
            if k not in kernels:
                kernels[k] = f8_producer(_kernel_def(text, k))
    repl, drop = {}, set()   # line index -> its replacement (several lines); lines left out
    for x in used:
        x8, s8, _ = _F8_OPERANDS[x]
        for i in producers[x]:
            m = _LAUNCH.match(lines[i])
            args, at = _split(m.group("args")), _split(m.group("at"))
            ind = m.group("ind")
            w16 = "%w16_flag" if f16_optional[x] else "%w16_one"
            repl[i] = (f"{ind}// dual-group: rdna4 kernel.launch @{m.group('k')}_f8_rdna4[{m.group('grid')}]({', '.join(args + [x8, s8, w16])}) "
                       f": [{m.group('gt')}]({', '.join(at + ['buffer', 'buffer', 'index'])})\n{lines[i]}\n{ind}// dual-group: end")
    for i, x in leaves:
        m = _LAUNCH.match(lines[i])
        x8, s8, _ = _F8_OPERANDS[x]
        args = _split(m.group("args"))
        args[3], args[4] = x8, s8
        mode = _F8_LEAF.match(m.group("k")).group(1)
        ind = lines[i - 1][:len(lines[i - 1]) - len(lines[i - 1].lstrip())]   # the scf.if's indentation
        repl[i - 1] = "\n".join([
            f"{ind}scf.if %huge {{",
            f"{ind}  // dual-group: rdna4 kernel.launch @{prefix}_gemm_mxfp4s_{mode}_f8_m256_rdna4[{m.group('grid')}]({', '.join(args)}) : [{m.group('gt')}]({m.group('at')})",
            f"{ind}  {lines[i].strip()}",
            f"{ind}  // dual-group: end",
            f"{ind}}} else {{",
            f"{ind}  scf.if %big {{", f"{ind}    {lines[i].strip()}", f"{ind}  }} else {{", f"{ind}    {lines[i + 2].strip()}", f"{ind}  }}",
            f"{ind}}}"])
        drop.update(range(i, i + 4))
    for g, u, s0, s1 in triples:
        mg, mu, ms = _LAUNCH.match(lines[g]), _LAUNCH.match(lines[u]), _LAUNCH.match(lines[s0 + 1])
        x8, s8, _ = _F8_OPERANDS[leaf_at[g]]
        ga, ua = _split(mg.group("args")), _split(mu.group("args"))
        gt, ut = _split(mg.group("at")), _split(mu.group("at"))
        args = ga[:3] + [x8, s8] + ga[5:7] + ua[5:7] + [ga[7]]
        at = gt[:7] + ut[5:7] + [gt[7]]
        ind = lines[g - 2][:len(lines[g - 2]) - len(lines[g - 2].lstrip())]
        region = lines[g - 2:s1 + 1]
        repl[g - 2] = "\n".join([
            f"{ind}scf.if %huge {{",
            f"{ind}  // dual-group: rdna4 kernel.launch @{prefix}_gemm_mxfp4s_swiglu_f8_m256_rdna4[{mg.group('grid')}]({', '.join(args)}) : [{mg.group('gt')}]({', '.join(at)})",
            f"{ind}  {lines[g].strip()}", f"{ind}  {lines[u].strip()}", f"{ind}  {lines[s0 + 1].strip()}",
            f"{ind}  // dual-group: end",
            f"{ind}}} else {{"] + region + [f"{ind}}}"])
        drop.update(range(g - 1, s1 + 1))
    # the operand buffers: after the %f16 allocation of @llm_advance's prologue
    j = next(i for i, l in enumerate(lines) if re.match(r"\s*%f16 = buffer\.alloca<global> align\(256\) %f16_size : buffer", l))
    widths = {}
    for key, nm in (("%h16", "%half_row"), ("%o16", "%half_wide"), ("%f16", "%half_ffn")):
        widths[key] = next(int(mm.group(1)) // 2 for l in lines if (mm := re.match(rf"\s*{nm} = index\.constant (\d+) : offset", l)))
    alloc = []
    for x in used:
        x8, s8, _ = _F8_OPERANDS[x]
        alloc += [f"  {x8}_row = index.constant {widths[x]} : offset", f"  {x8}_size = index.scale %t, {x8}_row : index, offset -> offset",
                  f"  {x8} = buffer.alloca<global> align(256) {x8}_size : buffer", f"  {s8} = buffer.alloca<global> align(256) %scale_size : buffer"]
    alloc += ["  %w16_zero = index.constant 0 : index", "  %w16_one = index.constant 1 : index", "  %w16_flag = scf.select %huge, %w16_zero, %w16_one : index"]
    repl[j] = lines[j] + "\n" + "\n".join(alloc)
    body = "\n".join(repl.get(i, l) for i, l in enumerate(lines) if i not in drop)
    text = text[:a] + body + text[b:]
    first = text.index("command.def")   # the twins precede every launch (dual_target.py: a launch of K comes after the end of the K_rdna4 kernels)
    first = text.rfind("\n", 0, first) + 1
    return text[:first] + "".join(t.rstrip("\n") + "\n\n" for t in kernels.values()) + text[first:], len(leaves) + 2 * len(triples)


# ======================================================================================================================
# int8-activation matvec weight codes (decode, verify widths): the split-plane twin of gen_llm.MXFP4_ICODES
# ======================================================================================================================
def mxfp4s_icodes(order="pairs"):
    """@g_mxfp4s_icodes(%w, %ws, %origin, %scale_origin, %group): the signed codes 2 * fp4 (kvalues_mxfp4: 0 1 2 3 4 6 8 12 and negatives) of the 32-value block
    whose 16 nibble bytes start at byte %origin of %w and whose E8M0 byte is at %scale_origin of %ws (%group unused: the Q8_0-style call shape of gen_llm.iq_decode),
    and the block scale 2^(e - 128) (the codes carry the factor 2): eight dot4 steps per block and token, as the Q8_0 path. Nibble order `order`."""
    join = ("vector.interleave<0> %lo_codes, %hi_codes : vector<16xi8>, vector<16xi8> -> vector<32xi8>" if order == "pairs"
            else "vector.concat<0> %lo_codes, %hi_codes : vector<16xi8>, vector<16xi8> -> vector<32xi8>")
    return f"""
func.def inline @g_mxfp4s_icodes(%w: buffer, %ws: buffer, %origin: offset, %scale_origin: offset, %group: index) -> (vector<32xi8>, f32) {{
  %q_view = buffer.view %w[%origin] : buffer -> view<16xi8>
  %e_view = buffer.view %ws[%scale_origin] : buffer -> view<1xi8>
  %e8 = view.load %e_view[0] : view<1xi8> -> i8
  %e = scalar.extui %e8 : i8 to i32
  %c1_i = scalar.constant 1 : i32
  %c2_i = scalar.constant 2 : i32
  %c23_i = scalar.constant 23 : i32
  %tiny_base = scalar.constant 2097152 : i32
  %e_low = scalar.cmpi ult, %e, %c2_i : i32
  %tiny = scalar.shli %tiny_base, %e : i32
  %em1 = scalar.subi %e, %c1_i : i32
  %normal = scalar.shli %em1, %c23_i : i32
  %bits = scf.select %e_low, %tiny, %normal : i32
  %d = scalar.bitcast %bits : i32 to f32
  %payload = vector.load %q_view[0] : view<16xi8> -> vector<16xi8>
  %mask = vector.splati 15 : vector<16xi8>
  %four = vector.splati 4 : vector<16xi8>
  %lower = vector.andi %payload, %mask : vector<16xi8>
  %shifted = vector.shrui %payload, %four : vector<16xi8>
  %upper = vector.andi %shifted, %mask : vector<16xi8>
  %k0 = scalar.constant 0 : i8
  %k1 = scalar.constant 1 : i8
  %k2 = scalar.constant 2 : i8
  %k3 = scalar.constant 3 : i8
  %k4 = scalar.constant 4 : i8
  %k5 = scalar.constant 6 : i8
  %k6 = scalar.constant 8 : i8
  %k7 = scalar.constant 12 : i8
  %k9 = scalar.constant -1 : i8
  %k10 = scalar.constant -2 : i8
  %k11 = scalar.constant -3 : i8
  %k12 = scalar.constant -4 : i8
  %k13 = scalar.constant -6 : i8
  %k14 = scalar.constant -8 : i8
  %k15 = scalar.constant -12 : i8
  %codebook = vector.from_elements %k0, %k1, %k2, %k3, %k4, %k5, %k6, %k7, %k0, %k9, %k10, %k11, %k12, %k13, %k14, %k15 : vector<16xi8>
  %lo_codes = vector.table.lookup %codebook[%lower] : vector<16xi8>, vector<16xi8> -> vector<16xi8>
  %hi_codes = vector.table.lookup %codebook[%upper] : vector<16xi8>, vector<16xi8> -> vector<16xi8>
  %codes = {join}
  func.return %codes, %d : vector<32xi8>, f32
}}
"""


# -- skinny alpha / beta projections of the gated delta-net on fp8 WMMA ---------------------------------------------------
def ab_f8_enabled():
    """GDN_AB_F8 (choice prefill.gdn.ab_f8): with prefill.gdn.ab_q8 and the fp8 GEMMs, the 48-row alpha / beta projections of GEMM-shape epochs
    run as one paired skinny fp8 GEMM on the producer's fp8 activations (%h8), not g_quant_q8 + two int8 16-token matvecs."""
    return f8_enabled() and os.environ.get("GDN_AB_F8", "0").lower() not in ("", "0", "false", "no", "off")


def ab_f8(text, hv, k):
    """gfx12, @llm_advance after fuse_mxfp4s_f8: each GDN layer's GEMM-epoch alpha block (gen_llm.ab_q8: g_quant_q8 + g_mv_.._t16_q8 into %galpha)
    and its beta matvec (into %gbeta) become, in epochs whose rows are a multiple of 32, a dual group whose gfx12 provider is
    g_ab_gemm_mxfp4s_store_pair_f8_m256_rdna4 (32 x hv tiles, both weights in one launch, K step 256) over %h8 / %s_h8, the fp8 rows the
    producer twin already wrote for the qkv GEMM. gfx11 and the other epochs keep the int8 launches. Returns (text, layers rewritten)."""
    a = text.find("command.def public @llm_advance(")
    if a < 0 or "%h8 = buffer.alloca" not in text[a:]:
        return text, 0
    b = text.find("\ncommand.def ", a + 10)
    b = len(text) if b < 0 else b + 1
    body = text[a:b]
    pat = re.compile(r"(?m)^(?P<ind> *)scf\.if %gemm \{\n(?P=ind)(?P<q>kernel\.launch @g_quant_q8\[%t, %nH\]\(%t, %nH, %h, %h16q\) : [^\n]*)\n"
                     r"(?P=ind)(?P<al>kernel\.launch @g_mv_mxfp4s_k\d+_t16_q8\[%nHV, %t\]\(%nHV, %t, %zero_i32, %h16q, (?P<wa>%walpha_\d+), (?P=wa)_sc, %galpha\) : \[index, index\]\(index, index, i32, buffer, (?P<va>view<\d+xi8>), (?P<vs>view<\d+xi8>), buffer\))\n")
    n = 0
    out, pos = [], 0
    for m in pat.finditer(body):
        layer = m.group("wa")[len("%walpha_"):]
        bl = re.compile(r"(?m)^(?P<ind> *)(?P<bt>kernel\.launch @g_mv_mxfp4s_k\d+_t16_q8\[%nHV, %t\]\(%nHV, %t, %zero_i32, %h16q, %wbeta_" + layer + r", %wbeta_" + layer + r"_sc, %gbeta\) : [^\n]*)\n")
        bm = bl.search(body, m.end())
        assert bm and body[bm.start() - len(m.group("ind")) - len("scf.if %gemm {\n"):bm.start()].endswith("scf.if %gemm {\n"), f"ab_f8: layer {layer}: beta block"
        ind = m.group("ind")
        pair = (f"kernel.launch @g_ab_gemm_mxfp4s_store_pair_f8_m256_rdna4[%t, %nHV](%t, %k_ab, %nHV, %h8, %s_h8, {m.group('wa')}, {m.group('wa')}_sc, %wbeta_{layer}, "
                f"%wbeta_{layer}_sc, %galpha, %gbeta) : [index, index](index, index, index, buffer, buffer, {m.group('va')}, {m.group('vs')}, {m.group('va')}, {m.group('vs')}, buffer, buffer)")
        chain = [m.group("q"), m.group("al"), bm.group("bt")]
        new = (f"{ind}scf.if %gemm {{\n{ind}scf.if %ab_t32 {{\n{ind}  // dual-group: rdna4 {pair}\n" + "".join(f"{ind}  {c}\n" for c in chain)
               + f"{ind}  // dual-group: end\n{ind}}} else {{\n" + "".join(f"{ind}  {c}\n" for c in chain) + f"{ind}}}\n")
        out.append(body[pos:m.start()] + new + body[m.end():bm.start()])
        pos = bm.end()
        n += 1
    if not n:
        return text, 0
    body = "".join(out) + body[pos:]
    anchor = re.search(r"(?m)^  %gemm = index\.cmp uge, %t, %c64 : index\n", body)
    body = (body[:anchor.end()] + f"  %ab_c32 = index.constant 32 : index\n  %ab_rem = index.rem %t, %ab_c32 : index\n  %ab_t32 = index.cmp eq, %ab_rem, %c0 : index\n"
            f"  %k_ab = index.constant {k} : index\n" + body[anchor.end():])
    kern = gemm_mxfp4s_f8_m256("store", prefix="g_ab", bm=32, tn=hv, rf=2, cf=1, kmult=2, pair=True)
    text = text[:a] + body + text[b:]
    first = text.index("command.def ")   # kernels before the programs (dual_target: a twin is defined before its launches)
    return text[:first] + kern + "\n" + text[first:], n
