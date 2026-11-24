"""MLA prefill attention in the EXPANDED form (gen_llm.py, MLA_X=1, the default for cfg "mla"; MLA_X=0 keeps the absorbed
flash kernels of gen_flash2s.py for every tile).

Absorbed MLA prefill (gen_flash2s.py MLA mode) scores 576-dim keys and sums 512-dim values per query head: 32 x (576 + 512)
x 2 = 69.6 KFLOP per (query, key) per layer. Expanded, each head has its own keys [k_nope (128) = k_b[h]^T c | k_pe (64,
shared)] and values v (128) = v_b[h] c from the latent row c (512): 32 x (192 + 128) x 2 = 20.5 KFLOP, 3.4x less, plus the
expansion of every key the span reads (2 x 512 x 32 x 256 = 8.4 MFLOP per key per layer, redone per prefill chunk because
the layers share the scratch). Expanded wins once a span has more than ~170 queries per epoch (8.4M / (69.6K - 20.5K)):
spans with fewer whole tiles than MLA_X_MIN_TILES stay absorbed.

Per epoch (g_mla_xlist, once): the flash2 whole-tile list (tiles[0] = count, tiles[1 + i] = first token of a 16-token tile
inside one run, in token order) is cut into SEGMENTS (consecutive tiles of one row, consecutive positions). A segment of
>= MLA_X_MIN_TILES tiles whose keys 0 .. last tile position + 15 (nk, rounded up to 64) fit the scratch (XCAP positions, a
running base) is EXPANDED: its tiles go to the quad list xq (up to 4 consecutive tiles per entry: first token, tile count,
scratch base) and its keys to the expansion block list xb (64 keys per entry: row, first key, scratch position, nk). Every
other tile is compacted back into tiles[] (count rewritten) for the absorbed g_attention_flash2. Counts: xcnt[0] quads,
xcnt[1] blocks (indirect dispatch).
Per layer:
  g_mla_xdeq   W[h] (256 x 512 f16) = [k_b[h]^T (128 rows: dim d over latent j) ; v_b[h] (128 rows)] from the IQ4 weights.
  g_mla_xexp   per (64-key block, head): [k_nope | v] = c W[h]^T on WMMA, c read from the paged latent cache; k_nope stored
               key-major xk[h][pos][128], v stored TRANSPOSED xv[h][dim][pos] (dims contiguous over keys: the V^T operand
               of the flash kernel loads straight from it); head 0's workgroups copy k_pe to xpe[pos][64]. Keys >= nk: zero.
  g_mla_xflash per (quad, head), 4 waves (scores in the log2 domain: scale * log2 e, exp2) (one 16-query tile each, no workgroup memory, no barriers): S^T = K Q^T over 12
               16-dim chunks (8 from xk, 4 from xpe; Q^T fragments [q_nope from qb | rope(q_pe) from q] in registers), causal
               mask, online softmax (gen_mla_dec.py's transposed form: a lane's column is one query), O^T += V^T P^T (8 dim
               tiles); vo[token][h][128] = O / l (the v_b product of the absorbed path is already in it).
The absorbed kernels still run for the remaining tiles, the partial blocks and short runs; their v_b GEMM writes vo for every
token, so g_mla_xflash runs after it and overwrites the expanded tiles' rows.
Numerics: Q, K, V, P rounded to f16 (the expansion output is f16), f32 accumulation and softmax.
"""
import os

XCAP = int(os.environ.get("MLA_XCAP", "131072"))         # scratch positions (sum over the epoch's expanded segments, 64-rounded)
MIN_TILES = int(os.environ.get("MLA_X_MIN_TILES", "16"))  # whole 16-token tiles a segment needs to be expanded
MIN_KEYS = int(os.environ.get("MLA_X_MIN_KEYS", "1024"))  # and keys (kbench mla_xpf: with the LDS expansion a 1k prompt is already faster expanded)
XB_MAX = XCAP // 64
QW = int(os.environ.get("MLA_XQ", "4"))   # 16-query tiles per g_mla_xflash workgroup (one wave each; a quad entry of xq)


def on():
    return os.environ.get("MLA_X", "1") == "1"


def scratch_bytes(cfg):
    """(xk, xv, xpe, w) bytes."""
    QH = cfg["heads"]
    return QH * XCAP * 128 * 2, QH * XCAP * 128 * 2, XCAP * 64 * 2, QH * 256 * 512 * 2


def xlist():
    return f"""
// Expanded-MLA work lists of the epoch (gen_mla_x.py): segments of the flash2 whole-tile list -> quads xq / expansion blocks
// xb (expanded) or back into tiles[] (absorbed). One lane walks the list in order.
kernel.def export("g_mla_xlist") @g_mla_xlist(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c32, %c1, %c1) : index
}} launch(%tokens_arg: index, %info: buffer, %tiles: buffer, %xq: buffer, %xb: buffer, %xcnt: buffer, %xskip: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 16, 8192), multiple_of(%tokens_arg, 16)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c4 = index.constant 4 : index
  %c16 = index.constant 16 : index
  %c63 = index.constant 63 : index
  %c64 = index.constant 64 : index
  %cxcap = index.constant {XCAP} : index
  %cmin = index.constant {MIN_TILES} : index
  %c0_i32 = scalar.constant 0 : i32
  %first = index.cmp eq, %lane, %c0 : index
  scf.if %first {{
    %tiles_view = buffer.view %tiles[%zero] : buffer -> view<[%tokens]xi32>
    %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
    %xq_view = buffer.view %xq[%zero] : buffer -> view<512x4xi32>
    %xb_view = buffer.view %xb[%zero] : buffer -> view<{XB_MAX}x4xi32>
    %cnt_view = buffer.view %xcnt[%zero] : buffer -> view<4xi32>
    %n32 = view.load %tiles_view[%c0] : view<[%tokens]xi32> -> i32
    %n_b = scalar.assume %n32 [range(%n32, 0, 512)] : i32
    %n = index.cast %n_b : i32 to index
    %n1 = index.add %n, %c1 : index
    // state: segment start (list index), its row, previous tile position / token, scratch base, quads, blocks, kept tiles
    %f_s, %f_row, %f_pp, %f_pt, %f_base, %f_nq, %f_nb, %f_nc = scf.for %i = [%c0 to %n1 step %c1](%s_a = %c0 : index, %row_a = %c0 : index, %pp_a = %c0 : index, %pt_a = %c0 : index, %base_a = %c0 : index, %nq_a = %c0 : index, %nb_a = %c0 : index, %nc_a = %c0 : index) -> (index, index, index, index, index, index, index, index) {{
      %s = index.assume %s_a [range(%s_a, 0, 512)] : index
      %row = index.assume %row_a [range(%row_a, 0, 2147483646)] : index
      %pp = index.assume %pp_a [range(%pp_a, 0, 262143)] : index
      %pt = index.assume %pt_a [range(%pt_a, 0, 8191)] : index
      %base = index.assume %base_a [range(%base_a, 0, {XCAP})] : index
      %nq = index.assume %nq_a [range(%nq_a, 0, 512)] : index
      %nb = index.assume %nb_a [range(%nb_a, 0, {XB_MAX})] : index
      %nc = index.assume %nc_a [range(%nc_a, 0, 512)] : index
      %in = index.cmp ult, %i, %n : index
      %slot_v = index.add %i, %c1 : index
      %slot_c = index.min %slot_v, %n : index
      %slot_1 = index.max %slot_c, %c1 : index
      %slot = index.assume %slot_1 [range(%slot_1, 1, 512), lt(%slot_1, %tokens)] : index
      %tok32 = view.load %tiles_view[%slot] : view<[%tokens]xi32> -> i32
      %tok_b = scalar.assume %tok32 [range(%tok32, 0, 8191)] : i32
      %tok_v = index.cast %tok_b : i32 to index
      %tok = index.assume %tok_v [lt(%tok_v, %tokens)] : index
      %row32 = view.load %info_view[%tok, 0] : view<[%tokens]x4xi32> -> i32
      %pos32 = view.load %info_view[%tok, 1] : view<[%tokens]x4xi32> -> i32
      %row_b = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
      %row_i = index.cast %row_b : i32 to index
      %pos_b = scalar.assume %pos32 [range(%pos32, 0, 262143)] : i32
      %pos_i = index.cast %pos_b : i32 to index
      %not0 = index.cmp ne, %i, %c0 : index
      %same_row = index.cmp eq, %row_i, %row : index
      %pp16 = index.add %pp, %c16 : index
      %pt16 = index.add %pt, %c16 : index
      %next_pos = index.cmp eq, %pos_i, %pp16 : index
      %next_tok = index.cmp eq, %tok, %pt16 : index
      %c_a = scalar.andi %in, %not0 : i1
      %c_b = scalar.andi %c_a, %same_row : i1
      %c_c = scalar.andi %c_b, %next_pos : i1
      %cont = scalar.andi %c_c, %next_tok : i1
      %true = scalar.constant true : i1
      %not_cont = scalar.xori %cont, %true : i1
      %close = scalar.andi %not0, %not_cont : i1
      %base2, %nq2, %nb2, %nc2 = scf.if %close -> (index, index, index, index) {{
        %nk = index.add %pp, %c16 : index
        %nk63 = index.add %nk, %c63 : index
        %nkb = index.div %nk63, %c64 : index
        %nk64 = index.mul %nkb, %c64 : index
        %end = index.add %base, %nk64 : index
        %ntl_v = index.sub %i, %s : index
        %ntl = index.assume %ntl_v [range(%ntl_v, 0, 512)] : index
        %fits = index.cmp ule, %end, %cxcap : index
        %long_q = index.cmp uge, %ntl, %cmin : index
        %cmink = index.constant {MIN_KEYS} : index
        %long_k = index.cmp uge, %nk, %cmink : index
        %long = scalar.andi %long_q, %long_k : i1
        %expand = scalar.andi %fits, %long : i1
        %r_base, %r_nq, %r_nb, %r_nc = scf.if %expand -> (index, index, index, index) {{
          %cqwl = index.constant {QW - 1} : index
          %cqw = index.constant {QW} : index
          %ntl3 = index.add %ntl, %cqwl : index
          %nqa = index.div %ntl3, %cqw : index
          %row_w = index.cast %row : index to i32
          %base_w = index.cast %base : index to i32
          %nk_w = index.cast %nk : index to i32
          scf.for %qq = [%c0 to %nqa step %c1] {{
            %q4 = index.mul %qq, %cqw : index
            %qs = index.add %s, %q4 : index
            %qslot_v = index.add %qs, %c1 : index
            %qslot = index.assume %qslot_v [range(%qslot_v, 1, 512), lt(%qslot_v, %tokens)] : index
            %qtok = view.load %tiles_view[%qslot] : view<[%tokens]xi32> -> i32
            %left = index.sub %ntl, %q4 : index
            %qn = index.min %left, %cqw : index
            %qn_w = index.cast %qn : index to i32
            %qe_v = index.add %nq, %qq : index
            %qe = index.assume %qe_v [range(%qe_v, 0, 511)] : index
            view.store %qtok, %xq_view[%qe, 0] : i32, view<512x4xi32>
            view.store %qn_w, %xq_view[%qe, 1] : i32, view<512x4xi32>
            view.store %base_w, %xq_view[%qe, 2] : i32, view<512x4xi32>
            view.store %row_w, %xq_view[%qe, 3] : i32, view<512x4xi32>
          }}
          scf.for %bb = [%c0 to %nkb step %c1] {{
            %k0 = index.mul %bb, %c64 : index
            %xp = index.add %base, %k0 : index
            %k0_w = index.cast %k0 : index to i32
            %xp_w = index.cast %xp : index to i32
            %be_v = index.add %nb, %bb : index
            %be = index.assume %be_v [range(%be_v, 0, {XB_MAX - 1})] : index
            view.store %row_w, %xb_view[%be, 0] : i32, view<{XB_MAX}x4xi32>
            view.store %k0_w, %xb_view[%be, 1] : i32, view<{XB_MAX}x4xi32>
            view.store %xp_w, %xb_view[%be, 2] : i32, view<{XB_MAX}x4xi32>
            view.store %nk_w, %xb_view[%be, 3] : i32, view<{XB_MAX}x4xi32>
          }}
          %nq_n = index.add %nq, %nqa : index
          %nb_n = index.add %nb, %nkb : index
          scf.yield %end, %nq_n, %nb_n, %nc : index, index, index, index
        }} else {{
          scf.for %j = [%c0 to %ntl step %c1] {{
            %src_v0 = index.add %s, %j : index
            %src_v = index.add %src_v0, %c1 : index
            %src = index.assume %src_v [range(%src_v, 1, 512), lt(%src_v, %tokens)] : index
            %dst_v0 = index.add %nc, %j : index
            %dst_v = index.add %dst_v0, %c1 : index
            %dst = index.assume %dst_v [range(%dst_v, 1, 512), lt(%dst_v, %tokens)] : index
            %tv = view.load %tiles_view[%src] : view<[%tokens]xi32> -> i32
            view.store %tv, %tiles_view[%dst] : i32, view<[%tokens]xi32>
          }}
          %nc_n = index.add %nc, %ntl : index
          scf.yield %base, %nq, %nb, %nc_n : index, index, index, index
        }}
        scf.yield %r_base, %r_nq, %r_nb, %r_nc : index, index, index, index
      }} else {{
        scf.yield %base, %nq, %nb, %nc : index, index, index, index
      }}
      %s2 = scf.select %cont, %s, %i : index
      %row2 = scf.select %cont, %row, %row_i : index
      scf.yield %s2, %row2, %pos_i, %tok, %base2, %nq2, %nb2, %nc2 : index, index, index, index, index, index, index, index
    }}
    %nc32 = index.cast %f_nc : index to i32
    %nq32 = index.cast %f_nq : index to i32
    %nb32 = index.cast %f_nb : index to i32
    view.store %nc32, %tiles_view[%c0] : i32, view<[%tokens]xi32>
    view.store %nq32, %cnt_view[0] : i32, view<4xi32>
    view.store %nb32, %cnt_view[1] : i32, view<4xi32>
  }}
  // xskip[b] = 1 when tokens 64b .. 64b + 63 all lie in expanded tiles or are padding (not live): the absorbed k_b / v_b
  // GEMMs skip that block
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %m_bytes = index.constant 32768 : offset
  %m_tile = buffer.alloca<workgroup> align(16) %m_bytes : buffer
  %m_view = buffer.view %m_tile[%zero] : buffer -> view<8192xi32>
  %c32x = index.constant 32 : index
  %c1_i32 = scalar.constant 1 : i32
  %mi_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  scf.for %z = [%lane to %tokens step %c32x] {{
    %zb = index.assume %z [range(%z, 0, 8191)] : index
    %zt = index.assume %z [lt(%z, %tokens)] : index
    %zlive = view.load %mi_view[%zt, 2] : view<[%tokens]x4xi32> -> i32
    %zdead = scalar.cmpi eq, %zlive, %c0_i32 : i32
    %zinit = scf.select %zdead, %c1_i32, %c0_i32 : i32
    view.store %zinit, %m_view[%zb] : i32, view<8192xi32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %cv = buffer.view %xcnt[%zero] : buffer -> view<4xi32>
  %nqw = view.load %cv[0] : view<4xi32> -> i32
  %nqw_b = scalar.assume %nqw [range(%nqw, 0, 512)] : i32
  %nqi = index.cast %nqw_b : i32 to index
  %xqv = buffer.view %xq[%zero] : buffer -> view<512x4xi32>
  scf.for %e = [%lane to %nqi step %c32x] {{
    %eb = index.assume %e [range(%e, 0, 511)] : index
    %et32 = view.load %xqv[%eb, 0] : view<512x4xi32> -> i32
    %en32 = view.load %xqv[%eb, 1] : view<512x4xi32> -> i32
    %et_b = scalar.assume %et32 [range(%et32, 0, 8191)] : i32
    %en_b = scalar.assume %en32 [range(%en32, 0, {QW})] : i32
    %et = index.cast %et_b : i32 to index
    %en = index.cast %en_b : i32 to index
    %en16 = index.mul %en, %c16 : index
    %eend = index.add %et, %en16 : index
    scf.for %k = [%et to %eend step %c1] {{
      %kb = index.assume %k [range(%k, 0, 8191)] : index
      view.store %c1_i32, %m_view[%kb] : i32, view<8192xi32>
    }}
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %nblk64 = index.div %tokens, %c64 : index
  %skip_view = buffer.view %xskip[%zero] : buffer -> view<128xi32>
  scf.for %bk = [%lane to %nblk64 step %c32x] {{
    %bkb = index.assume %bk [range(%bk, 0, 127)] : index
    %t0 = index.mul %bkb, %c64 : index
    %t1 = index.add %t0, %c64 : index
    %all = scf.for %k = [%t0 to %t1 step %c1](%acc = %c1_i32 : i32) -> (i32) {{
      %kb = index.assume %k [range(%k, 0, 8191)] : index
      %mv = view.load %m_view[%kb] : view<8192xi32> -> i32
      %a2 = scalar.andi %acc, %mv : i32
      scf.yield %a2 : i32
    }}
    view.store %all, %skip_view[%bkb] : i32, view<128xi32>
  }}
  kernel.return
}}
"""


def xdeq(QH, kb_fmt, vb_fmt, kb_row, vb_row, kb_block, vb_block):
    """W[h] = [k_b[h]^T ; v_b[h]] as f16 (256 x 512 per head). k_b rows (h, j) hold 128 dims in 4 IQ4_NL blocks; v_b rows
    (h, d) hold 512 latent values in 2 IQ4_XS superblocks (8 groups of 32 each)."""
    assert kb_fmt == "iq4_nl" and vb_fmt == "iq4_xs", (kb_fmt, vb_fmt)
    nk = QH * 512 * 4        # (h, j, block of 32 dims)
    nv = QH * 128 * 16       # (h, d, group of 32 latent values)
    wg = (nk + nv) // 256
    ext = "\n".join(f"""      %e{i} = vector.extract %kh[%x{i}] : vector<32xf16> -> f16
      %r{i} = index.add %krow0, %x{i} : index
      %rb{i} = index.assume %r{i} [range(%r{i}, 0, {QH * 256 - 1})] : index
      view.store %e{i}, %w_view[%rb{i}, %j] : f16, view<{QH * 256}x512xf16>""" for i in range(32))
    return f"""
// Expanded MLA: dequantized per-head expansion weights W[h] = [k_b[h]^T ; v_b[h]] (f16, {QH} x 256 x 512).
kernel.def export("g_mla_xdeq") @g_mla_xdeq(%unused: index) {{
  %c1 = index.constant 1 : index
  %cwg = index.constant {wg} : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%cwg, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%unused_arg: index, %kb: buffer, %vb: buffer, %w: buffer) {{
  %wg_id = kernel.workgroup.id<x> : index
  %wi_id = kernel.workitem.id<x> : index
  %wi = index.assume %wi_id [range(%wi_id, 0, 255)] : index
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c128 = index.constant 128 : index
  %c256 = index.constant 256 : index
  %c512 = index.constant 512 : index
  %c2048 = index.constant 2048 : index
  %cnk = index.constant {nk} : index
  %zero = index.constant 0 : offset
""" + "".join(f"  %x{i} = index.constant {i} : index\n" for i in range(32)) + f"""  %gid_v = index.madd %wg_id, %c256, %wi : index
  %gid = index.assume %gid_v [range(%gid_v, 0, {nk + nv - 1})] : index
  %w_view = buffer.view %w[%zero] : buffer -> view<{QH * 256}x512xf16>
  %is_k = index.cmp ult, %gid, %cnk : index
  scf.if %is_k {{
      %h = index.div %gid, %c2048 : index
      %hj = index.div %gid, %c4 : index
      %j_v = index.rem %hj, %c512 : index
      %j = index.assume %j_v [range(%j_v, 0, 511)] : index
      %b = index.rem %gid, %c4 : index
      %kbrow = index.constant {kb_row} : offset
      %kbblk = index.constant {kb_block} : offset
      %ro = index.scale %hj, %kbrow : index, offset -> offset
      %bo = index.scale %b, %kbblk : index, offset -> offset
      %origin = index.add %ro, %bo : offset
      %kv = func.call @g_iq4_nl_values(%kb, %origin, %c0) : (buffer, offset, index) -> (vector<32xf32>)
      %kh = vector.fptrunc %kv : vector<32xf32> to vector<32xf16>
      %hrow = index.mul %h, %c256 : index
      %b32 = index.mul %b, %c32 : index
      %krow0 = index.add %hrow, %b32 : index
{ext}
  }} else {{
      %g2 = index.sub %gid, %cnk : index
      %h = index.div %g2, %c2048 : index
      %hd = index.div %g2, %c16 : index
      %d = index.rem %hd, %c128 : index
      %g = index.rem %g2, %c16 : index
      %blk = index.div %g, %c8 : index
      %within = index.rem %g, %c8 : index
      %vbrow = index.constant {vb_row} : offset
      %vbblk = index.constant {vb_block} : offset
      %ro = index.scale %hd, %vbrow : index, offset -> offset
      %bo = index.scale %blk, %vbblk : index, offset -> offset
      %origin = index.add %ro, %bo : offset
      %vv = func.call @g_iq4_xs_values(%vb, %origin, %within) : (buffer, offset, index) -> (vector<32xf32>)
      %vh = vector.fptrunc %vv : vector<32xf32> to vector<32xf16>
      %vlo = vector.slice %vh[0] : vector<32xf16> -> vector<16xf16>
      %vhi = vector.slice %vh[16] : vector<32xf16> -> vector<16xf16>
      %hrow = index.mul %h, %c256 : index
      %vrow0 = index.add %hrow, %c128 : index
      %vrow_v = index.add %vrow0, %d : index
      %vrow = index.assume %vrow_v [range(%vrow_v, 0, {QH * 256 - 1})] : index
      %j0_v = index.mul %g, %c32 : index
      %j0 = index.assume %j0_v [range(%j0_v, 0, 480)] : index
      %j1 = index.add %j0, %c16 : index
      vector.store %vlo, %w_view[%vrow, %j0] : vector<16xf16>, view<{QH * 256}x512xf16>
      vector.store %vhi, %w_view[%vrow, %j1] : vector<16xf16>, view<{QH * 256}x512xf16>
  }}
  kernel.return
}}
"""


PE_BODY = '''    %pk = index.div %wi, %c4 : index
    %pq = index.rem %wi, %c4 : index
    %pkey = index.add %key0, %pk : index
    %pvalid = index.cmp ult, %pkey, %nk : index
    %pkc = scf.select %pvalid, %pkey, %c0 : index
    %po = func.call @{kvo}(%origins, %row, %pkc, %blocks, %layer, %pool) : (buffer, index, index, index, index, index) -> (offset)
    %pview = buffer.view %arena[%po] : buffer -> view<1x{D}xf16>
    %pin = index.constant 0 : index
    %pq16 = index.mul %pq, %c16 : index
    %pcol_v = index.add %pq16, %c512 : index
    %pcol = index.assume %pcol_v [range(%pcol_v, 512, 560)] : index
    %praw = vector.load %pview[%pin, %pcol] : view<1x{D}xf16> -> vector<16xf16>
    %pval = scf.select %pvalid, %praw, %zero16h : vector<16xf16>
    %prow_v = index.add %xpos, %pk : index
    %prow = index.assume %prow_v [range(%prow_v, 0, {XCAP - 1})] : index
    %pqc = index.assume %pq16 [range(%pq16, 0, 48)] : index
    vector.store %pval, %xpe_view[%prow, %pqc] : vector<16xf16>, view<{XCAP}x64xf16>
'''   # g_mla_xexp's k_pe copy step (template: {kvo}, {D}, {XCAP})


def xexp(QH, layer_max, kvo="g_kv_offset"):
    """[k_nope | v] of 64 keys x one head per workgroup: 8 waves = (32-key half kg, 64-column quarter qd of the head's 256
    W rows); per 16-value latent step a wave loads 2 key fragments (one 16-position page each) and 4 W fragments: 8 MMAs."""
    D = 576
    NT = int(os.environ.get("MLA_XE_NT", "4"))   # 16-column W tiles per wave (4: 8 waves of 32 keys x 64 cols; 8: 4 waves of 32 x 128)
    QDN = 256 // (16 * NT)                       # column groups of the head's 256 W rows (K: the first half)
    WT = 2 * QDN * 32                            # threads
    A = []
    a = A.append
    a(f"""
// Expanded MLA: k_nope / v (and k_pe) of the expansion blocks into the scratch (indirect: xcnt[1] blocks).
kernel.def export("g_mla_xexp") @g_mla_xexp(%count_buffer: buffer) {{
  %origin_c = index.constant 0 : offset
  %counts = buffer.view %count_buffer[%origin_c] : buffer -> view<4xi32>
  %word = view.load %counts[1] : view<4xi32> -> i32
  %bounded = scalar.assume %word [range(%word, 0, {XB_MAX})] : i32
  %count = index.cast %bounded : i32 to index
  %c1 = index.constant 1 : index
  %cqh = index.constant {QH} : index
  %cwt = index.constant {WT} : index
  kernel.launch.config workgroups(%count, %cqh, %c1) workgroup_size(%cwt, %c1, %c1) : index
}} launch(%tokens_arg: index, %layer_arg: index, %context_arg: index, %pool_arg: index, %xb: buffer, %origins: buffer, %arena: buffer, %w: buffer, %xk: buffer, %xv: buffer, %xpe: buffer) {{
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {layer_max})] : index
  %context = index.assume %context_arg [range(%context_arg, 16, 262144)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 2147483584)] : index
  %blk_id = kernel.workgroup.id<x> : index
  %blk = index.assume %blk_id [range(%blk_id, 0, {XB_MAX - 1})] : index
  %head_id = kernel.workgroup.id<y> : index
  %head = index.assume %head_id [range(%head_id, 0, {QH - 1})] : index
  %wi_id = kernel.workitem.id<x> : index
  %wi = index.assume %wi_id [range(%wi_id, 0, {WT - 1})] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, {WT // 32 - 1})] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c15 = index.constant 15 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %c128 = index.constant 128 : index
  %c256 = index.constant 256 : index
  %c512 = index.constant 512 : index
  %cxcap = index.constant {XCAP} : index
  %xb_view = buffer.view %xb[%zero] : buffer -> view<{XB_MAX}x4xi32>
  %row32 = view.load %xb_view[%blk, 0] : view<{XB_MAX}x4xi32> -> i32
  %k0_32 = view.load %xb_view[%blk, 1] : view<{XB_MAX}x4xi32> -> i32
  %xp32 = view.load %xb_view[%blk, 2] : view<{XB_MAX}x4xi32> -> i32
  %nk32 = view.load %xb_view[%blk, 3] : view<{XB_MAX}x4xi32> -> i32
  %row_b = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
  %row = index.cast %row_b : i32 to index
  %k0_b = scalar.assume %k0_32 [range(%k0_32, 0, 262080)] : i32
  %key0 = index.cast %k0_b : i32 to index
  %xp_b = scalar.assume %xp32 [range(%xp32, 0, {XCAP - 64})] : i32
  %xpos = index.cast %xp_b : i32 to index
  %nk_b = scalar.assume %nk32 [range(%nk32, 16, 262144)] : i32
  %nk = index.cast %nk_b : i32 to index
  %padded = index.add %context, %c15 : index
  %blocks = index.div %padded, %c16 : index
  %cqdn = index.constant {QDN} : index
  %ccols = index.constant {16 * NT} : index
  %kg = index.div %wave, %cqdn : index
  %qd = index.rem %wave, %cqdn : index
  %lane_col = index.rem %lane, %c16 : index
  %kg32 = index.mul %kg, %c32 : index
  %wkey0 = index.add %key0, %kg32 : index
  %hrow = index.mul %head, %c256 : index
  %qd64 = index.mul %qd, %ccols : index
  %nrow0 = index.add %hrow, %qd64 : index
  %w_view = buffer.view %w[%zero] : buffer -> view<{QH * 256}x512xf16>
  %zero16h = vector.splatf 0.0 : vector<16xf16>
  %zero8 = vector.splatf 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>""")
    for pp in range(2):
        a(f"""  %pf{pp}_o = index.constant {16 * pp} : index
  %pf{pp} = index.add %wkey0, %pf{pp}_o : index
  %pv{pp} = index.cmp ult, %pf{pp}, %nk : index
  %pc{pp} = scf.select %pv{pp}, %pf{pp}, %c0 : index
  %ko{pp} = func.call @{kvo}(%origins, %row, %pc{pp}, %blocks, %layer, %pool) : (buffer, index, index, index, index, index) -> (offset)
  %kv{pp} = buffer.view %arena[%ko{pp}] : buffer -> view<16x{D}xf16>
  %kk{pp} = index.add %pf{pp}, %lane_col : index
  %kval{pp} = index.cmp ult, %kk{pp}, %nk : index""")
    for t in range(NT):
        a(f"""  %nr{t}_o = index.constant {16 * t} : index
  %nr{t}_a = index.add %nrow0, %nr{t}_o : index
  %nr{t}_b = index.add %nr{t}_a, %lane_col : index
  %nr{t} = index.assume %nr{t}_b [range(%nr{t}_b, 0, {QH * 256 - 1})] : index""")
    accs = [f"a{pp}{t}" for pp in range(2) for t in range(NT)]

    def loop(vt):
        """the 32-step latent loop: K waves S = c W^T (lhs latent, rhs W); V waves (vt) V^T = W c^T (lhs W, rhs latent),
        so V^T lands in its natural row-major layout xv[h][dim][pos]."""
        a(f"""    {", ".join("%o" + x[1:] for x in accs)} = scf.for %j = [%c0 to %c32 step %c1]({", ".join(f"%{x} = %init : vector<8xf32>" for x in accs)}) -> ({", ".join(["vector<8xf32>"] * (2 * NT))}) {{
      %jc_v = index.mul %j, %c16 : index
      %jc = index.assume %jc_v [range(%jc_v, 0, 496)] : index""")
        for pp in range(2):
            a(f"""      %lr{pp} = vector.load %kv{pp}[%lane_col, %jc] : view<16x{D}xf16> -> vector<16xf16>
      %lh{pp} = scf.select %kval{pp}, %lr{pp}, %zero16h : vector<16xf16>
      %lf{pp} = vector.fragment<{"rhs" if vt else "lhs"}> %lh{pp} shape [%c16, %c16] : vector<16xf16>""")
        for tt in range(NT):
            a(f"""      %wr{tt} = vector.load %w_view[%nr{tt}, %jc] : view<{QH * 256}x512xf16> -> vector<16xf16>
      %wf{tt} = vector.fragment<{"lhs" if vt else "rhs"}> %wr{tt} shape [%c16, %c16] : vector<16xf16>""")
        for pp in range(2):
            for tt in range(NT):
                x, y = (f"%wf{tt}", f"%lf{pp}") if vt else (f"%lf{pp}", f"%wf{tt}")
                a(f"""      %m{pp}{tt} = vector.mma {x}, {y}, %a{pp}{tt} : vector<16xf16>, vector<16xf16>, vector<8xf32>""")
        a(f"""      scf.yield {", ".join(f"%m{pp}{tt}" for pp in range(2) for tt in range(NT))} : {", ".join(["vector<8xf32>"] * (2 * NT))}
    }}""")
    XL = os.environ.get("MLA_XE_LDS", "1") == "1"   # stage each 32-value latent step of the 64 keys and 256 W rows in workgroup memory
    assert not XL or NT == 4
    a(f"""  %two = index.constant 2 : offset
  %chalf = index.constant {QDN // 2} : index
  %is_k = index.cmp ult, %qd, %chalf : index
  %xrow0 = index.add %xpos, %kg32 : index""")
    if XL:
        TY8 = ", ".join(["vector<8xf32>"] * 8)
        a(f"""  %ls_bytes = index.constant {64 * 40 * 2} : offset
  %ls_tile = buffer.alloca<workgroup> align(16) %ls_bytes : buffer
  %ls_view = buffer.view %ls_tile[%zero] : buffer -> view<64x40xf16>
  %ws_bytes = index.constant {256 * 40 * 2} : offset
  %ws_tile = buffer.alloca<workgroup> align(16) %ws_bytes : buffer
  %ws_view = buffer.view %ws_tile[%zero] : buffer -> view<256x40xf16>
  // staging: thread wi < 128 loads latent key wi / 2 (16 values at 16 (wi % 2)), every thread one W row (32 values)
  %sl_key = index.div %wi, %c2 : index
  %sl_h = index.rem %wi, %c2 : index
  %sl_h16 = index.mul %sl_h, %c16 : index
  %sl_lk = index.rem %sl_key, %c64 : index
  %sl_pos = index.add %key0, %sl_lk : index
  %sl_valid0 = index.cmp ult, %sl_pos, %nk : index
  %sl_lo = index.cmp ult, %wi, %c128 : index
  %sl_pc = scf.select %sl_valid0, %sl_pos, %c0 : index
  %sl_o = func.call @{kvo}(%origins, %row, %sl_pc, %blocks, %layer, %pool) : (buffer, index, index, index, index, index) -> (offset)
  %sl_view = buffer.view %arena[%sl_o] : buffer -> view<1x{D}xf16>
  %sw_row_v = index.add %hrow, %wi : index
  %sw_row = index.assume %sw_row_v [range(%sw_row_v, 0, {QH * 256 - 1})] : index
  %c0i = index.constant 0 : index
  %lrow0 = index.add %kg32, %lane_col : index
  %c16i = index.constant 16 : index
  %lrow1 = index.add %lrow0, %c16i : index
  %wrow_b = index.add %qd64, %lane_col : index
  {", ".join(f"%o{pp}{tt}" for pp in range(2) for tt in range(4))} = scf.for %j = [%c0 to %c16 step %c1]({", ".join(f"%a{pp}{tt} = %init : vector<8xf32>" for pp in range(2) for tt in range(4))}) -> ({TY8}) {{
    %j32_v = index.mul %j, %c32 : index
    %j32 = index.assume %j32_v [range(%j32_v, 0, 480)] : index
    scf.if %sl_lo {{
      %sl_col_v = index.add %j32, %sl_h16 : index
      %sl_col = index.assume %sl_col_v [range(%sl_col_v, 0, 496)] : index
      %sl_raw = vector.load %sl_view[%c0i, %sl_col] : view<1x{D}xf16> -> vector<16xf16>
      %sl_val = scf.select %sl_valid0, %sl_raw, %zero16h : vector<16xf16>
      vector.store %sl_val, %ls_view[%sl_lk, %sl_h16] : vector<16xf16>, view<64x40xf16>
    }}
    %j32b = index.add %j32, %c16i : index
    %sw0 = vector.load %w_view[%sw_row, %j32] : view<{QH * 256}x512xf16> -> vector<16xf16>
    %sw1 = vector.load %w_view[%sw_row, %j32b] : view<{QH * 256}x512xf16> -> vector<16xf16>
    vector.store %sw0, %ws_view[%wi, %c0i] : vector<16xf16>, view<256x40xf16>
    vector.store %sw1, %ws_view[%wi, %c16i] : vector<16xf16>, view<256x40xf16>
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)""")
        cur = {(pp, tt): f"%a{pp}{tt}" for pp in range(2) for tt in range(4)}
        for s in range(2):
            a(f"""    %ss{s} = index.constant {16 * s} : index""")
            for pp in range(2):
                a(f"""    %lv{s}{pp} = vector.load %ls_view[%lrow{pp}, %ss{s}] : view<64x40xf16> -> vector<16xf16>""")
            for tt in range(4):
                a(f"""    %wo{s}{tt} = index.constant {16 * tt} : index
    %wrr{s}{tt} = index.add %wrow_b, %wo{s}{tt} : index
    %wv{s}{tt} = vector.load %ws_view[%wrr{s}{tt}, %ss{s}] : view<256x40xf16> -> vector<16xf16>""")
            names = [f"%n{s}{pp}{tt}" for pp in range(2) for tt in range(4)]
            a(f"""    {", ".join(names)} = scf.if %is_k -> ({TY8}) {{""")
            for pp in range(2):
                a(f"""      %klf{s}{pp} = vector.fragment<lhs> %lv{s}{pp} shape [%c16, %c16] : vector<16xf16>""")
            for tt in range(4):
                a(f"""      %kwf{s}{tt} = vector.fragment<rhs> %wv{s}{tt} shape [%c16, %c16] : vector<16xf16>""")
            for pp in range(2):
                for tt in range(4):
                    a(f"""      %km{s}{pp}{tt} = vector.mma %klf{s}{pp}, %kwf{s}{tt}, {cur[(pp, tt)]} : vector<16xf16>, vector<16xf16>, vector<8xf32>""")
            a(f"""      scf.yield {", ".join(f"%km{s}{pp}{tt}" for pp in range(2) for tt in range(4))} : {TY8}
    }} else {{""")
            for pp in range(2):
                a(f"""      %vlf{s}{pp} = vector.fragment<rhs> %lv{s}{pp} shape [%c16, %c16] : vector<16xf16>""")
            for tt in range(4):
                a(f"""      %vwf{s}{tt} = vector.fragment<lhs> %wv{s}{tt} shape [%c16, %c16] : vector<16xf16>""")
            for pp in range(2):
                for tt in range(4):
                    a(f"""      %vm{s}{pp}{tt} = vector.mma %vwf{s}{tt}, %vlf{s}{pp}, {cur[(pp, tt)]} : vector<16xf16>, vector<16xf16>, vector<8xf32>""")
            a(f"""      scf.yield {", ".join(f"%vm{s}{pp}{tt}" for pp in range(2) for tt in range(4))} : {TY8}
    }}""")
            cur = {(pp, tt): f"%n{s}{pp}{tt}" for pp in range(2) for tt in range(4)}
        a(f"""    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    scf.yield {", ".join(cur[(pp, tt)] for pp in range(2) for tt in range(4))} : {TY8}
  }}""")
    a("  scf.if %is_k {")
    if not XL:
        loop(False)
    a(f"""    %hk_e = index.mul %head, %cxcap : index
    %hk_e2 = index.mul %hk_e, %c128 : index
    %hk_o = index.scale %hk_e2, %two : index, offset -> offset
    %xk_view = buffer.view %xk[%hk_o] : buffer -> view<{XCAP}x128xf16>""")
    for pp in range(2):
        for tt in range(NT):
            a(f"""    %sk{pp}{tt}_r0 = index.constant {16 * pp} : index
    %sk{pp}{tt}_r1 = index.add %xrow0, %sk{pp}{tt}_r0 : index
    %sk{pp}{tt}_r = index.assume %sk{pp}{tt}_r1 [range(%sk{pp}{tt}_r1, 0, {XCAP - 16})] : index
    %sk{pp}{tt}_c0 = index.constant {16 * tt} : index
    %sk{pp}{tt}_c1 = index.add %qd64, %sk{pp}{tt}_c0 : index
    %sk{pp}{tt}_c = index.assume %sk{pp}{tt}_c1 [range(%sk{pp}{tt}_c1, 0, 112)] : index
    vector.fragment.store<result> %o{pp}{tt}, %xk_view[%sk{pp}{tt}_r, %sk{pp}{tt}_c] shape [%c16, %c16] : vector<8xf32>, view<{XCAP}x128xf16>""")
    a(f"""  }} else {{""")
    if not XL:
        loop(True)
    a(f"""    %cnb = index.constant {XCAP // 32} : index
    %c4096v = index.constant 4096 : index
    %vblk = index.div %xrow0, %c32 : index
    %hv_e = index.madd %head, %cnb, %vblk : index
    %hv_e2 = index.mul %hv_e, %c4096v : index
    %hv_o = index.scale %hv_e2, %two : index, offset -> offset
    %xv_view = buffer.view %xv[%hv_o] : buffer -> view<128x32xf16>
    %qdv = index.sub %qd, %chalf : index
    %qdv64 = index.mul %qdv, %ccols : index""")
    for pp in range(2):
        for tt in range(NT):
            a(f"""    %sv{pp}{tt}_r0 = index.constant {16 * pp} : index
    %sv{pp}{tt}_r1 = index.add %xrow0, %sv{pp}{tt}_r0 : index
    %sv{pp}{tt}_r = index.assume %sv{pp}{tt}_r1 [range(%sv{pp}{tt}_r1, 0, {XCAP - 16})] : index
    %sv{pp}{tt}_c0 = index.constant {16 * tt} : index
    %sv{pp}{tt}_c1 = index.add %qdv64, %sv{pp}{tt}_c0 : index
    %sv{pp}{tt}_c = index.assume %sv{pp}{tt}_c1 [range(%sv{pp}{tt}_c1, 0, 112)] : index
    vector.fragment.store<result> %o{pp}{tt}, %xv_view[%sv{pp}{tt}_c, %sv{pp}{tt}_r0] shape [%c16, %c16] : vector<8xf32>, view<128x32xf16>""")
    a(f"""  }}
  // k_pe (shared by the heads): head 0's workgroups copy it, (key, 16-dim quarter) per thread and step
  %h0 = index.cmp eq, %head, %c0 : index
  scf.if %h0 {{
    %xpe_view = buffer.view %xpe[%zero] : buffer -> view<{XCAP}x64xf16>
""")
    import re as _re
    for r in range(256 // WT):
        b = PE_BODY.replace("%wi, %c4", f"%pv_{r}, %c4")
        b = _re.sub(r"%(pk|pq|pkey|pvalid|pkc|po|pview|pin|pq16|pcol_v|pcol|praw|pval|prow_v|prow|pqc)\b", lambda m: f"%{m.group(1)}_{r}", b)
        a(f"""    %pvo_{r} = index.constant {WT * r} : index
    %pv_{r} = index.add %wi, %pvo_{r} : index
""" + b.rstrip("\n").replace("{kvo}", kvo).replace("{D}", str(D)).replace("{XCAP - 1}", str(XCAP - 1)).replace("{XCAP}", str(XCAP)))
    a(f"""  }}
  kernel.return
}}
""")
    text = "\n".join(A)
    dbg = int(os.environ.get("MLA_XE_DBG", "0"))   # bisection of device faults (kbench only): 1 no K/V stores, 2 no k_pe copy, 4 no latent loads, 8 no W loads
    import re as _re
    if dbg & 1:
        text = _re.sub(r"\n    vector\.fragment\.store<result> [^\n]*", "", text)
    if dbg & 2:
        text = _re.sub(r"\n    vector\.store %pval[^\n]*", "", text)
    if dbg & 4:
        text = _re.sub(r"(%lr\d) = vector\.load %kv\d\[%lane_col, %jc\] : view<16x576xf16> -> vector<16xf16>", r"\1 = vector.splatf 0.5 : vector<16xf16>", text)
    if dbg & 8:
        text = _re.sub(r"(%wr\d) = vector\.load %w_view\[[^\n]*", r"\1 = vector.splatf 0.25 : vector<16xf16>", text)
    return text


def xflash(QH, scale, QB=192, QD=576, VD=128, lds=None):
    """Expanded flash attention: workgroup (quad, head) of 4 waves, wave = one 16-query tile. lds (MLA_XF=lds): the four
    waves stage each 32-key block's K [32 x 192] and V^T [128 x 32] in workgroup memory once (128 threads, 5 16-value loads
    each) and read their fragments from there; the loop runs to the quad's last key block, a wave past its own causal
    range keeps its state."""
    if lds is None:
        lds = os.environ.get("MLA_XF", "lds") == "lds"
    NCH = 12   # 16-dim chunks of the 192-dim key: 8 nope (xk) + 4 rope (xpe)
    A = []
    a = A.append
    a(f"""
// Expanded MLA flash prefill (gen_mla_x.py): vo[token][h] = softmax(Q_h K_h^T scale) V_h for the quads' tiles (indirect: xcnt[0]).
kernel.def export("g_mla_xflash") @g_mla_xflash(%count_buffer: buffer) {{
  %origin_c = index.constant 0 : offset
  %counts = buffer.view %count_buffer[%origin_c] : buffer -> view<4xi32>
  %word = view.load %counts[0] : view<4xi32> -> i32
  %bounded = scalar.assume %word [range(%word, 0, 512)] : i32
  %count = index.cast %bounded : i32 to index
  %c1 = index.constant 1 : index
  %cqh = index.constant {QH} : index
  %cwgs = index.constant {32 * QW} : index
  kernel.launch.config workgroups(%count, %cqh, %c1) workgroup_size(%cwgs, %c1, %c1) : index
}} launch(%tokens_arg: index, %xq: buffer, %info: buffer, %qb: buffer, %q: buffer, %xk: buffer, %xv: buffer, %xpe: buffer, %vo: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 16, 8192), multiple_of(%tokens_arg, 16)] : index
  %quad_id = kernel.workgroup.id<x> : index
  %quad = index.assume %quad_id [range(%quad_id, 0, 511)] : index
  %head_id = kernel.workgroup.id<y> : index
  %head = index.assume %head_id [range(%head_id, 0, {QH - 1})] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, {QW - 1})] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c16 = index.constant 16 : index
  %c31 = index.constant 31 : index
  %c32 = index.constant 32 : index
  %c128 = index.constant 128 : index
  %c64x = index.constant 64 : index
  %cxcap = index.constant {XCAP} : index
""" + "".join(f"  %ex{i} = index.constant {i} : index\n" for i in range(8)) + f"""  %xq_view = buffer.view %xq[%zero] : buffer -> view<512x4xi32>
  %tok32 = view.load %xq_view[%quad, 0] : view<512x4xi32> -> i32
  %nt32 = view.load %xq_view[%quad, 1] : view<512x4xi32> -> i32
  %base32 = view.load %xq_view[%quad, 2] : view<512x4xi32> -> i32
  %nt_b = scalar.assume %nt32 [range(%nt32, 0, {QW})] : i32
  %ntiles = index.cast %nt_b : i32 to index
  %mine = index.cmp ult, %wave, %ntiles : index
  {"%all_waves = index.cmp ult, %c0, %c1 : index" + chr(10) + "  scf.if %all_waves {" if lds else "scf.if %mine {"}
    %tok_b = scalar.assume %tok32 [range(%tok32, 0, 8191)] : i32
    %tok_f = index.cast %tok_b : i32 to index
    %nt_last = index.sub %ntiles, %c1 : index
    %w16 = index.mul %wave, %c16 : index
    %tok0_w = index.add %tok_f, %w16 : index
    %tok_hi = index.sub %tokens, %c16 : index
    %tok0_v = {"index.min %tok0_w, %tok_hi" if lds else "index.add %tok0_w, %c0"} : index
    %tok0 = index.assume %tok0_v [range(%tok0_v, 0, 8176), lt(%tok0_v, %tokens)] : index
    %base_b = scalar.assume %base32 [range(%base32, 0, {XCAP - 64})] : i32
    %base = index.cast %base_b : i32 to index
    %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
    %pos32 = view.load %info_view[%tok0, 1] : view<[%tokens]x4xi32> -> i32
    %pos_b = scalar.assume %pos32 [range(%pos32, 0, 262128)] : i32
    %pos0 = index.cast %pos_b : i32 to index
    %lane_col = index.rem %lane, %c16 : index
    %lane_half = index.div %lane, %c16 : index
    %hi_half = index.cmp ne, %lane_half, %c0 : index
    %qtok_v = index.add %tok0, %lane_col : index
    %qtok = index.assume %qtok_v [lt(%qtok_v, %tokens)] : index
    %qb_view = buffer.view %qb[%zero] : buffer -> view<[%tokens]x{QH}x{QB}xf32>
    %q_view = buffer.view %q[%zero] : buffer -> view<[%tokens]x{QH}x{QD}xf32>""")
    QL = int(os.environ.get("MLA_XF_QL", "0")) if lds else 0   # the last QL 16-dim Q chunks live in workgroup memory, not VGPRs
    QLW = 16 * QL + 8
    if QL:
        a(f"""    %ql_bytes = index.constant {QW * 16 * QLW * 2} : offset
    %ql_tile = buffer.alloca<workgroup> align(16) %ql_bytes : buffer
    %ql_view = buffer.view %ql_tile[%zero] : buffer -> view<{QW * 16}x{QLW}xf16>
    %ql_r_v = index.add %w16, %lane_col : index
    %ql_r = index.assume %ql_r_v [range(%ql_r_v, 0, {QW * 16 - 1})] : index""")
    for c in range(NCH):
        if c < 8:
            a(f"""    %qraw{c} = vector.load %qb_view[%qtok, %head, {16 * c}] : view<[%tokens]x{QH}x{QB}xf32> -> vector<16xf32>""")
        else:
            a(f"""    %qraw{c} = vector.load %q_view[%qtok, %head, {512 + 16 * (c - 8)}] : view<[%tokens]x{QH}x{QD}xf32> -> vector<16xf32>""")
        a(f"""    %qh{c} = vector.fptrunc %qraw{c} : vector<16xf32> to vector<16xf16>""")
        if c >= NCH - QL:
            a(f"""    vector.store %qh{c}, %ql_view[%ql_r, {16 * (c - (NCH - QL))}] : vector<16xf16>, view<{QW * 16}x{QLW}xf16>""")
        else:
            a(f"""    %qf{c} = vector.fragment<rhs> %qh{c} shape [%c16, %c16] : vector<16xf16>""")
    a(f"""    %two = index.constant 2 : offset
    %hk_e = index.mul %head, %cxcap : index
    %hk_e2 = index.mul %hk_e, %c128 : index
    %hk_o = index.scale %hk_e2, %two : index, offset -> offset
    %xk_view = buffer.view %xk[%hk_o] : buffer -> view<{XCAP}x128xf16>
    %xv_view = buffer.view %xk[%zero] : buffer -> view<1xf16>
    %cnb = index.constant {XCAP // 32} : index
    %c4096v = index.constant 4096 : index
    %hv_b = index.mul %head, %cnb : index
    %pe_view = buffer.view %xpe[%zero] : buffer -> view<{XCAP}x64xf16>
    %zero8 = vector.splatf 0.0 : vector<8xf32>
    %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
    %zero_f = scalar.constant 0.0 : f32
    %neg_big = scalar.constant -3.0e38 : f32
    %scale = scalar.constant {scale * 1.4426950408889634!r} : f32
    %scale_v = vector.splat %scale : vector<8xf32>
    %neg_v = vector.splat %neg_big : vector<8xf32>
    %exchange = scalar.constant 16 : i32
    %width = scalar.constant 32 : i32
    %two_i32 = scalar.constant 2 : i32
    %half32 = index.cast %lane_half : index to i32
    %row_off = vector.iota %half32 step %two_i32 : vector<8xi32>
    %qpos_i = index.add %pos0, %lane_col : index
    %qpos32 = index.cast %qpos_i : index to i32
    %qpos_v = vector.splat %qpos32 : vector<8xi32>
    %nkw = index.add %pos0, %c16 : index
    %nkw31 = index.add %nkw, %c31 : index
    %nblk = index.div %nkw31, %c32 : index""")
    if lds:
        a(f"""    %ntl16 = index.mul %nt_last, %c16 : index
    %tokl_v = index.add %tok_f, %ntl16 : index
    %tokl = index.assume %tokl_v [range(%tokl_v, 0, 8176), lt(%tokl_v, %tokens)] : index
    %posl32 = view.load %info_view[%tokl, 1] : view<[%tokens]x4xi32> -> i32
    %posl_b = scalar.assume %posl32 [range(%posl32, 0, 262128)] : i32
    %posl = index.cast %posl_b : i32 to index
    %nkq = index.add %posl, %c16 : index
    %nkq31 = index.add %nkq, %c31 : index
    %nblk_q = index.div %nkq31, %c32 : index
    %ks_bytes = index.constant {32 * 200 * 2} : offset
    %ks_tile = buffer.alloca<workgroup> align(16) %ks_bytes : buffer
    %ks_view = buffer.view %ks_tile[%zero] : buffer -> view<32x200xf16>
    %vs_bytes = index.constant {128 * 40 * 2} : offset
    %vs_tile = buffer.alloca<workgroup> align(16) %vs_bytes : buffer
    %vs_view = buffer.view %vs_tile[%zero] : buffer -> view<128x40xf16>
    %wi_id = kernel.workitem.id<x> : index
    %wi_all = index.assume %wi_id [range(%wi_id, 0, {32 * QW - 1})] : index
    %c127 = index.constant 127 : index
    %stager = index.cmp ult, %wi_all, %c128 : index
    %wi_m = index.min %wi_all, %c127 : index
    %wi = index.assume %wi_m [range(%wi_m, 0, 127)] : index
    %c2 = index.constant 2 : index
    %c4 = index.constant 4 : index
    %c8 = index.constant 8 : index
    %st_kk0 = index.div %wi, %c8 : index
    %st_kc0_v = index.rem %wi, %c8 : index
    %st_kc0 = index.mul %st_kc0_v, %c16 : index
    %st_kk1 = index.add %st_kk0, %c16 : index
    %st_pk = index.div %wi, %c4 : index
    %st_pc_v = index.rem %wi, %c4 : index
    %st_pc = index.mul %st_pc_v, %c16 : index
    %st_pc_l = index.add %st_pc, %c128 : index
    %st_vd0 = index.div %wi, %c2 : index
    %st_vh_v = index.rem %wi, %c2 : index
    %st_vh = index.mul %st_vh_v, %c16 : index
    %st_vd1 = index.add %st_vd0, %c64x : index""")
    for d in range(8):
        a(f"""    %vd{d} = index.constant {16 * d} : index
    %vr{d}_v = index.add %vd{d}, %lane_col : index
    %vr{d} = index.assume %vr{d}_v [range(%vr{d}_v, 0, 127)] : index""")
    TR = set(t for t in os.environ.get("MLA_XF_T", "").split(",") if t)   # xflash trims (see xflash docstring)
    assert TR <= {"diag", "fma", "lazy"}, TR
    diag, fma, lazy = "diag" in TR, "fma" in TR, "lazy" in TR
    ABL = set(t for t in os.environ.get("MLA_XF_ABL", "").split(",") if t)   # kbench ablations (wrong results): nores, nox, nosm
    assert ABL <= {"nores", "nox", "nosm"}, ABL
    TY = ", ".join(["vector<8xf32>"] * 8)
    if fma or lazy:
        a(f"""    %scale_s = scalar.constant {scale * 1.4426950408889634!r} : f32
    %lazy_tau = scalar.constant 8.0 : f32
    %one_f = scalar.constant 1.0 : f32""")
    if diag:   # blocks whose 32 keys all precede the quad's first query: no causal mask (every wave sees every key)
        a(f"""    %tok_q0 = index.assume %tok_f [range(%tok_f, 0, 8176), lt(%tok_f, %tokens)] : index
    %pq0_32 = view.load %info_view[%tok_q0, 1] : view<[%tokens]x4xi32> -> i32
    %pq0_b = scalar.assume %pq0_32 [range(%pq0_32, 0, 262128)] : i32
    %pq0 = index.cast %pq0_b : i32 to index
    %pq0_1 = index.add %pq0, %c1 : index
    %nfull_v = index.div %pq0_1, %c32 : index
    %nfull = index.min %nfull_v, {"%nblk_q" if lds else "%nblk"} : index""")

    def body(masked, sfx):
        a(f"""      %kf = index.mul %b, %c32 : index
      %vb_pos = index.add %base, %kf : index
      %vb_i = index.div %vb_pos, %c32 : index
      %vb_e = index.add %hv_b, %vb_i : index
      %vb_e2 = index.mul %vb_e, %c4096v : index
      %vb_o = index.scale %vb_e2, %two : index, offset -> offset
      %vt_view = buffer.view %xv[%vb_o] : buffer -> view<128x32xf16>""")
        if lds:
            a(f"""      %sk_base = index.add %base, %kf : index
      %sk_r0_v = index.add %sk_base, %st_kk0 : index
      %sk_r0 = index.assume %sk_r0_v [range(%sk_r0_v, 0, {XCAP - 1})] : index
      %sk_r1_v = index.add %sk_base, %st_kk1 : index
      %sk_r1 = index.assume %sk_r1_v [range(%sk_r1_v, 0, {XCAP - 1})] : index
      %sk_rp_v = index.add %sk_base, %st_pk : index
      %sk_rp = index.assume %sk_rp_v [range(%sk_rp_v, 0, {XCAP - 1})] : index
      %sv_c_v = index.add %sk_base, %st_vh : index
      %sv_c = index.assume %sv_c_v [range(%sv_c_v, 0, {XCAP - 16}), multiple_of(%sv_c_v, 16)] : index
      scf.if %stager {{
      %gk0 = vector.load %xk_view[%sk_r0, %st_kc0] : view<{XCAP}x128xf16> -> vector<16xf16>
      %gk1 = vector.load %xk_view[%sk_r1, %st_kc0] : view<{XCAP}x128xf16> -> vector<16xf16>
      %gkp = vector.load %pe_view[%sk_rp, %st_pc] : view<{XCAP}x64xf16> -> vector<16xf16>
      %gv0 = vector.load %vt_view[%st_vd0, %st_vh] : view<128x32xf16> -> vector<16xf16>
      %gv1 = vector.load %vt_view[%st_vd1, %st_vh] : view<128x32xf16> -> vector<16xf16>
      vector.store %gk0, %ks_view[%st_kk0, %st_kc0] : vector<16xf16>, view<32x200xf16>
      vector.store %gk1, %ks_view[%st_kk1, %st_kc0] : vector<16xf16>, view<32x200xf16>
      vector.store %gkp, %ks_view[%st_pk, %st_pc_l] : vector<16xf16>, view<32x200xf16>
      vector.store %gv0, %vs_view[%st_vd0, %st_vh] : vector<16xf16>, view<128x40xf16>
      vector.store %gv1, %vs_view[%st_vd1, %st_vh] : vector<16xf16>, view<128x40xf16>
      }}
      kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
      // a wave past its causal range (or without a tile) runs the block anyway: every key is masked, so its state stays
      // (alpha = 1, p = 0); no branch keeps the accumulators from being duplicated across it""")
        for hh in range(2):
            a(f"""      %kh{hh}_o = index.constant {16 * hh} : index
      %kf{hh} = index.add %kf, %kh{hh}_o : index
      %xr{hh}_a = index.add %base, %kf{hh} : index
      %xr{hh}_b = index.add %xr{hh}_a, %lane_col : index
      %xr{hh} = index.assume %xr{hh}_b [range(%xr{hh}_b, 0, {XCAP - 1})] : index
      %xc{hh} = index.assume %xr{hh}_a [range(%xr{hh}_a, 0, {XCAP - 16}), multiple_of(%xr{hh}_a, 16)] : index
      %lr{hh}_v = index.add %kh{hh}_o, %lane_col : index
      %lr{hh} = index.assume %lr{hh}_v [range(%lr{hh}_v, 0, 31)] : index""")
            prev = "%init"
            for c in range(NCH):
                if lds:
                    a(f"""      %kr{hh}_{c} = vector.load %ks_view[%lr{hh}, {16 * c}] : view<32x200xf16> -> vector<16xf16>""")
                elif c < 8:
                    a(f"""      %kr{hh}_{c} = vector.load %xk_view[%xr{hh}, {16 * c}] : view<{XCAP}x128xf16> -> vector<16xf16>""")
                else:
                    a(f"""      %kr{hh}_{c} = vector.load %pe_view[%xr{hh}, {16 * (c - 8)}] : view<{XCAP}x64xf16> -> vector<16xf16>""")
                qf = f"%qf{c}"
                if c >= NCH - QL:
                    qf = f"%qlf{hh}_{c}"
                    a(f"""      %qlr{hh}_{c} = vector.load %ql_view[%ql_r, {16 * (c - (NCH - QL))}] : view<{QW * 16}x{QLW}xf16> -> vector<16xf16>
      {qf} = vector.fragment<rhs> %qlr{hh}_{c} shape [%c16, %c16] : vector<16xf16>""")
                a(f"""      %kfr{hh}_{c} = vector.fragment<lhs> %kr{hh}_{c} shape [%c16, %c16] : vector<16xf16>
      %s{hh}_{c} = vector.mma %kfr{hh}_{c}, {qf}, {prev} : vector<16xf16>, vector<16xf16>, vector<8xf32>""")
                prev = f"%s{hh}_{c}"
            raw = f"%s{hh}_{NCH - 1}"
            if fma or lazy:   # max over raw scores (scale > 0), scaled once as a scalar
                src = raw
            else:
                a(f"""      %sc{hh} = vector.mulf {raw}, %scale_v : vector<8xf32>""")
                src = f"%sc{hh}"
            if masked:
                a(f"""      %kf{hh}_32 = index.cast %kf{hh} : index to i32
      %kb{hh}_v = vector.splat %kf{hh}_32 : vector<8xi32>
      %kpos{hh} = vector.addi %kb{hh}_v, %row_off : vector<8xi32>
      %valid{hh} = vector.cmpi sle, %kpos{hh}, %qpos_v : vector<8xi32> -> vector<8xi1>
      %masked{hh} = vector.select %valid{hh}, {src}, %neg_v : vector<8xf32>""")
                src = f"%masked{hh}"
            a(f"""      %lmax{hh} = vector.reduce<maxnumf> {src}, %neg_big : vector<8xf32>, f32""")
        a(f"""      %lmax = scalar.maxnumf %lmax0, %lmax1 : f32
      %omax, %omax_u = kernel.subgroup.shuffle<xor> %lmax, %exchange, %width : f32, i32, i32
      %tmax{"_r" if (fma or lazy) else ""} = scalar.maxnumf %lmax, %omax : f32""")
        if fma or lazy:
            a("      %tmax = scalar.mulf %tmax_r, %scale_s : f32")
        if lazy:   # FA4-style: keep the running max unless some query of the wave grew by more than tau (2^8 headroom in P)
            os_ = ", ".join(f"%os{d}" for d in range(8))
            a(f"""      %m_tau = scalar.addf %m, %lazy_tau : f32
      %lz_need = scalar.cmpf ogt, %tmax, %m_tau : f32
      %lz_any = kernel.subgroup.vote.any %lz_need : i1
      %m_new, %alpha, {os_} = scf.if %lz_any -> (f32, f32, {TY}) {{
        %lm_n = scalar.maxnumf %m, %tmax : f32
        %lm_shift = scalar.subf %m, %lm_n : f32
        %lalpha = scalar.exp2f<afn> %lm_shift : f32
        %lalpha_v = vector.splat %lalpha : vector<8xf32>""")
            for d in range(8):
                a(f"""        %lzs{d} = vector.mulf %a{d}, %lalpha_v : vector<8xf32>""")
            a(f"""        scf.yield %lm_n, %lalpha, {", ".join(f"%lzs{d}" for d in range(8))} : f32, f32, {TY}
      }} else {{
        scf.yield %m, %one_f, {", ".join(f"%a{d}" for d in range(8))} : f32, f32, {TY}
      }}
      %m_new_v = vector.splat %m_new : vector<8xf32>""")
        else:
            a(f"""      %m_new = scalar.maxnumf %m, %tmax : f32
      %m_shift = scalar.subf %m, %m_new : f32
      %alpha = scalar.exp2f<afn> %m_shift : f32
      %m_new_v = vector.splat %m_new : vector<8xf32>""")
        if fma or lazy:
            a("      %m_neg = scalar.negf %m_new : f32\n      %m_neg_v = vector.splat %m_neg : vector<8xf32>")
        for hh in range(2):
            if fma or lazy:
                a(f"""      %sh{hh} = vector.fmaf %s{hh}_{NCH - 1}, %scale_v, %m_neg_v : vector<8xf32>""")
            else:
                a(f"""      %sh{hh} = vector.subf {"%masked" if masked else "%sc"}{hh}, %m_new_v : vector<8xf32>""")
            a(f"""      %pe{hh} = vector.exp2f<afn> %sh{hh} : vector<8xf32>""")
            pn = f"%s{hh}_{NCH - 1}" if "nosm" in ABL else (f"%p{hh}" if masked else f"%pe{hh}")
            if masked and "nosm" not in ABL:
                a(f"""      %p{hh} = vector.select %valid{hh}, %pe{hh}, %zero8 : vector<8xf32>""")
            a(f"""      %ls{hh} = vector.reduce<addf> {pn}, %zero_f : vector<8xf32>, f32""")
            for i in (range(8) if "nox" not in ABL else ()):
                a(f"""      %pi{hh}_{i} = vector.extract {pn}[%ex{i}] : vector<8xf32> -> f32
      %po{hh}_{i}, %pu{hh}_{i} = kernel.subgroup.shuffle<xor> %pi{hh}_{i}, %exchange, %width : f32, i32, i32""")
            if "nox" in ABL:
                a(f"""      %pt{hh}_f = vector.interleave<0> {pn}, {pn} : vector<8xf32>, vector<8xf32> -> vector<16xf32>
      %pt{hh}_h = vector.fptrunc %pt{hh}_f : vector<16xf32> to vector<16xf16>
      %pt{hh} = vector.fragment<rhs> %pt{hh}_h shape [%c16, %c16] : vector<16xf16>""")
                continue
            a(f"""      %partner{hh} = vector.from_elements {", ".join(f"%po{hh}_{i}" for i in range(8))} : vector<8xf32>
      %even{hh} = scf.select %hi_half, %partner{hh}, {pn} : vector<8xf32>
      %odd{hh} = scf.select %hi_half, {pn}, %partner{hh} : vector<8xf32>
      %pt{hh}_f = vector.interleave<0> %even{hh}, %odd{hh} : vector<8xf32>, vector<8xf32> -> vector<16xf32>
      %pt{hh}_h = vector.fptrunc %pt{hh}_f : vector<16xf32> to vector<16xf16>
      %pt{hh} = vector.fragment<rhs> %pt{hh}_h shape [%c16, %c16] : vector<16xf16>""")
        a(f"""      %ls = scalar.addf %ls0, %ls1 : f32
      %osum, %osum_u = kernel.subgroup.shuffle<xor> %ls, %exchange, %width : f32, i32, i32
      %tsum = scalar.addf %ls, %osum : f32
      %l_scaled = scalar.mulf %l, %alpha : f32
      %l_new = scalar.addf %l_scaled, %tsum : f32""")
        if not lazy:
            a("      %alpha_v = vector.splat %alpha : vector<8xf32>")
        for d in range(8):
            if not lazy:
                if "nores" not in ABL:
                    a(f"""      %os{d} = vector.mulf %a{d}, %alpha_v : vector<8xf32>""")
            cur = f"%os{d}" if "nores" not in ABL or lazy else f"%a{d}"
            for hh in range(2):
                a(f"""      %vl{d}_{hh} = vector.load %vs_view[%vr{d}, {16 * hh}] : view<128x40xf16> -> vector<16xf16>""" if lds else f"""      %vl{d}_{hh} = vector.load %vt_view[%vr{d}, {16 * hh}] : view<128x32xf16> -> vector<16xf16>""")
                a(f"""
      %vf{d}_{hh} = vector.fragment<lhs> %vl{d}_{hh} shape [%c16, %c16] : vector<16xf16>
      %an{d}_{hh} = vector.mma %vf{d}_{hh}, %pt{hh}, {cur} : vector<16xf16>, vector<16xf16>, vector<8xf32>""")
                cur = f"%an{d}_{hh}"
        if lds:
            a("      kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        a(f"""      scf.yield %m_new, %l_new, {", ".join(f"%an{d}_1" for d in range(8))} : f32, f32, {TY}
    }}""")

    acc = ", ".join(f"%a{d} = %init : vector<8xf32>" for d in range(8))
    nb = "%nblk_q" if lds else "%nblk"
    if diag:
        a(f"""    %m_d, %l_d, {", ".join(f"%od{d}" for d in range(8))} = scf.for %b = [%c0 to %nfull step %c1](%m = %neg_big : f32, %l = %zero_f : f32, {acc}) -> (f32, f32, {TY}) {{""")
        body(False, "")
        acc2 = ", ".join(f"%a{d} = %od{d} : vector<8xf32>" for d in range(8))
        a(f"""    %m_final, %l_final, {", ".join(f"%o{d}" for d in range(8))} = scf.for %b = [%nfull to {nb} step %c1](%m = %m_d : f32, %l = %l_d : f32, {acc2}) -> (f32, f32, {TY}) {{""")
        body(True, "")
    else:
        a(f"""    %m_final, %l_final, {", ".join(f"%o{d}" for d in range(8))} = scf.for %b = [%c0 to {nb} step %c1](%m = %neg_big : f32, %l = %zero_f : f32, {acc}) -> (f32, f32, {TY}) {{""")
        body(True, "")
    a(f"""{chr(10) + "    scf.if %mine {" if lds else ""}
    %one = scalar.constant 1.0 : f32
    %inv = scalar.divf %one, %l_final : f32
    %inv_v = vector.splat %inv : vector<8xf32>
    %cvo = index.constant {QH * VD} : index
    %c1_vo = index.constant 1 : index
    %vo_layout = encoding.layout.strided [%c1_vo, %cvo] : encoding<layout>
    %tok0_s = index.assume %tok0_w [range(%tok0_w, 0, 8176), lt(%tok0_w, %tokens)] : index
    %vo_e0 = index.mul %tok0_s, %cvo : index
    %hvd = index.mul %head, %c128 : index
    %vo_e = index.add %vo_e0, %hvd : index
    %four = index.constant 4 : offset
    %vo_o = index.scale %vo_e, %four : index, offset -> offset
    %vo_view = buffer.view %vo[%vo_o] : buffer -> view<128x16xf32, %vo_layout>""")
    for d in range(8):
        a(f"""    %on{d} = vector.mulf %o{d}, %inv_v : vector<8xf32>
    vector.fragment.store<result> %on{d}, %vo_view[%vd{d}, %c0] shape [%c16, %c16] : vector<8xf32>, view<128x16xf32, %vo_layout>""")
    if lds:
        a("    }")
    a("""  }
  kernel.return
}
""")
    t = "\n".join(A)
    return t.replace("    %xv_view = buffer.view %xk[%zero] : buffer -> view<1xf16>\n", "")


def direct(text, name):
    """The kernel tests' twin of an indirect kernel (loom-test cannot launch from a device count): `<name>_d`, whose grid's x
    is a launch index (the count the host expects) instead of the count word."""
    import re
    a = text.index(f'kernel.def export("{name}")')
    b = text.index("} launch(", a)
    body = text[a:b]
    m = re.search(r"  %count = index.cast %bounded : i32 to index\n", body)
    head = body[:body.index("(%count_buffer: buffer) {")].replace(f'export("{name}") @{name}', f'export("{name}_d") @{name}_d')
    rest = body[m.end():]
    hi = re.search(r"\[range\(%word, 0, (\d+)\)\]", body).group(1)
    return head + f"(%count_arg: index) {{\n  %count = index.assume %count_arg [range(%count_arg, 0, {hi})] : index\n" + rest + text[b:text.index("\n}\n", text.index("kernel.return", b)) + 3]


def skip_twin(text, name):
    """`<name>_x` of a g_mla_kbm / g_mla_vbm kernel (gen_mla.bmm / bmm2, with or without MLA_BMM_LIVE): an extra %xskip
    argument (after %info when the kernel is the MLA_BMM_LIVE form); a workgroup whose 64-token block is all expanded or
    padding (xskip[block] = 1) does nothing. Under MLA_BMM_LIVE the kernel's own all-padding test stays nested inside
    (workgroup-uniform both): a block is skipped when it is all padding OR all expanded/padding."""
    a = text.index(f'kernel.def export("{name}")')
    b = text.index("\n}\n", text.index("kernel.return", a)) + 3
    k = text[a:b].replace(f'export("{name}") @{name}(', f'export("{name}_x") @{name}_x(')
    n0 = k.count(", %y: buffer) {") + k.count(", %y: buffer, %info: buffer) {")
    assert n0 == 1, (name, n0)
    k = k.replace(", %y: buffer) {", ", %y: buffer, %xskip: buffer) {", 1)
    k = k.replace(", %y: buffer, %info: buffer) {", ", %y: buffer, %info: buffer, %xskip: buffer) {", 1)
    anchor = "  %tblk = index.assume %tb_id [lt(%tb_id, %tblocks)] : index\n"
    assert k.count(anchor) == 1 and k.count("  kernel.return\n") == 1
    k = k.replace(anchor, anchor + """  %xs_zero = index.constant 0 : offset
  %xs_view = buffer.view %xskip[%xs_zero] : buffer -> view<128xi32>
  %xs_b = index.assume %tblk [range(%tblk, 0, 127)] : index
  %xs_word = view.load %xs_view[%xs_b] : view<128xi32> -> i32
  %xs_zero_i32 = scalar.constant 0 : i32
  %xs_run = scalar.cmpi eq, %xs_word, %xs_zero_i32 : i32
  scf.if %xs_run {
""").replace("  kernel.return\n", "  }\n  kernel.return\n")
    return k


def kernels(cfg, layer_max, kb_fmt, vb_fmt, kb_row, vb_row, kb_block, vb_block, tests=False):
    m = cfg["mla"]
    QH = cfg["heads"]
    assert QH == 32 and m["kv_lora"] == 512 and m["rope_dims"] == 64 and m["nope_dims"] == 128 and m["v_dims"] == 128, \
        "gen_mla_x: 32 heads, latent 512 + rope 64, nope 128, v 128"
    assert not cfg.get("sliding_layers"), "gen_mla_x: no sliding-window layers"
    e, f = xexp(QH, layer_max), xflash(QH, m["scale"])
    return (xlist() + xdeq(QH, kb_fmt, vb_fmt, kb_row, vb_row, kb_block, vb_block) + e + f
            + (direct(e, "g_mla_xexp") + direct(f, "g_mla_xflash") if tests else ""))


if __name__ == "__main__":
    import json
    import sys
    c = json.load(open(sys.argv[1]))
    print(kernels(c, c["layers"] - 1, "iq4_nl", "iq4_xs", 72, 272, 18, 136, tests=len(sys.argv) > 2))
