"""Decode / verify-span split-K attention on WMMA (head_dim 128): g_attention_split_mma.

Drafting packages verify a span of up to T = depth + 1 consecutive tokens of one row per epoch. The scalar split kernel
(gen_llm.attention_text) spends ~30 wave instructions per (query row, key): lanes over dims, a butterfly reduction per
key and one exp per key. At 32k a depth-3 verify (4 tokens x G heads per KV head) is ALU bound there, not memory bound.
This kernel reads each key's K and V once per KV head for all T tokens x G query heads and does the QK^T and PV
products on the matrix cores (the arithmetic of gen_flash2.py, transposed form: S^T = K Q^T, one query column per
(token, head) pair).

Workgroup (token, KV head, split), 4 waves. The workgroup of the first token of a *run* computes the run: the
consecutive same-row tokens of one aligned group of T token slots that are all past ATTN_MMA_FROM keys (full-attention
layers); the other tokens of a run exit, and a token outside any run is a run of one. Every token's keys, split range
([start, stop) of g_split_count), 32-key tile grid and 16-key blocks are its own, and a column that has no key in a tile
keeps its state, so a token computes the same bits alone, in any run, or in any epoch (consistency / spec exact). The
partials keep the scalar kernel's layout: g_attention_combine is unchanged. g_attention_split exits for the tokens
this kernel takes (gen_llm.py).

Staging: 32-key tiles of K and V^T in workgroup memory (all 4 waves), then ceil(T * G / 16) waves compute 16 columns
each. A block of keys past a column's position gets V zeroed for that column (the loop over the run's tokens)."""
import os
import re

D, NF, NO, KT, KB, DP8, VTC = 128, 8, 4, 32, 2, 136, 40   # NF: 16-dim fragments of a head, NO: of them one wave accumulates
W, LANES, TPK, DPT = 4, 128, 4, 32


def fill(text, **kw):
    for k, v in kw.items():
        text = text.replace(f"<<{k}>>", str(v))
    assert "<<" not in text, re.findall(r"<<\w+>>", text)[:3]
    return text


def kernel(S, KVH, G, T, FROM, POS_BYTES, scale):
    QH = KVH * G
    NC = T * G
    assert NC <= 32 and 2 <= T <= 8
    WC = 2 * -(-NC // 16)   # computing waves: a pair per 16 columns, each accumulating half of the dims
    o = []
    A = o.append
    A(fill("""
// Decode / verify-span attention on WMMA (gen_attn_mma.py): runs of up to <<T>> consecutive tokens at >= <<FROM>> keys.
kernel.def export("g_attention_split_mma") @g_attention_split_mma(%tokens: index) {
  %c1 = index.constant 1 : index
  %ckvh = index.constant <<KVH>> : index
  %cS = index.constant <<S>> : index
  %clanes = index.constant <<LANES>> : index
  kernel.launch.config workgroups(%tokens, %ckvh, %cS) workgroup_size(%clanes, %c1, %c1) : index
} launch(%tokens_arg: index, %layer_arg: index, %sliding: i32, %context_arg: index, %pool_arg: index, %q: buffer, %info: buffer, %origins: buffer, %arena: buffer, %part_acc: buffer, %part_ml: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, 51)] : index
  %context = index.assume %context_arg [range(%context_arg, 16, 262144)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 1048576)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %kv_id = kernel.workgroup.id<y> : index
  %kv_head = index.assume %kv_id [range(%kv_id, 0, <<KVH1>>)] : index
  %split_id = kernel.workgroup.id<z> : index
  %split = index.assume %split_id [range(%split_id, 0, <<S1>>)] : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, <<LANES1>>)] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, <<W1>>)] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c31x = index.constant 31 : index
  %c15 = index.constant 15 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %c2_i32 = scalar.constant 2 : i32
  %zero_f = scalar.constant 0.0 : f32
  %neg_big = scalar.constant -3.0e38 : f32
  %true = scalar.constant true : i1
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %from32 = scalar.constant <<FROM1>> : i32
  %is_sliding = scalar.cmpi ne, %sliding, %c0_i32 : i32
  %not_sliding = scalar.xori %is_sliding, %true : i1
  %tokens_less = index.sub %tokens, %c1 : index
  %cT = index.constant <<T>> : index
  %t_rem = index.rem %token, %cT : index
  %rem_nz = index.cmp ne, %t_rem, %c0 : index
""", KVH=KVH, KVH1=KVH - 1, S=S, S1=S - 1, LANES=LANES, LANES1=LANES - 1, W1=W - 1, T=T, FROM=FROM, FROM1=FROM - 1))
    # info of the T slots from this token on
    for k in range(T):
        A(f"""  %ck{k} = index.constant {k} : index
  %ck{k}_i32 = scalar.constant {k} : i32
  %ta{k} = index.add %token, %ck{k} : index
  %inn{k} = index.cmp ult, %ta{k}, %tokens : index
  %tc{k} = index.min %ta{k}, %tokens_less : index
  %tb{k} = index.assume %tc{k} [lt(%tc{k}, %tokens)] : index
  %row{k} = view.load %info_view[%tb{k}, 0] : view<[%tokens]x4xi32> -> i32
  %pos{k} = view.load %info_view[%tb{k}, 1] : view<[%tokens]x4xi32> -> i32
  %lvv{k} = view.load %info_view[%tb{k}, 2] : view<[%tokens]x4xi32> -> i32
  %liv{k} = scalar.cmpi ne, %lvv{k}, %c0_i32 : i32
  %pf{k} = scalar.cmpi sge, %pos{k}, %from32 : i32
  %fa{k} = scalar.andi %liv{k}, %pf{k} : i1
  %fb{k} = scalar.andi %fa{k}, %not_sliding : i1
  %far{k} = scalar.andi %fb{k}, %inn{k} : i1""")
    A("""  // the previous token: does it continue into this one (same aligned group, same row, previous position, both far)?
  %has_prev = index.cmp ugt, %token, %c0 : index
  %tp_s = index.sub %token, %c1 : index
  %tp_c = scf.select %has_prev, %tp_s, %c0 : index
  %tpb = index.assume %tp_c [lt(%tp_c, %tokens)] : index
  %row_p = view.load %info_view[%tpb, 0] : view<[%tokens]x4xi32> -> i32
  %pos_p = view.load %info_view[%tpb, 1] : view<[%tokens]x4xi32> -> i32
  %lv_p = view.load %info_view[%tpb, 2] : view<[%tokens]x4xi32> -> i32
  %liv_p = scalar.cmpi ne, %lv_p, %c0_i32 : i32
  %pf_p = scalar.cmpi sge, %pos_p, %from32 : i32
  %fa_p = scalar.andi %liv_p, %pf_p : i1
  %far_p = scalar.andi %fa_p, %not_sliding : i1
  %pc_a = scalar.andi %has_prev, %rem_nz : i1
  %same_p = scalar.cmpi eq, %row_p, %row0 : i32
  %pos_p1 = scalar.addi %pos_p, %c1_i32 : i32
  %seq_p = scalar.cmpi eq, %pos0, %pos_p1 : i32
  %pc_b = scalar.andi %pc_a, %same_p : i1
  %pc_c = scalar.andi %pc_b, %seq_p : i1
  %pc_d = scalar.andi %pc_c, %far_p : i1
  %prev_cont = scalar.andi %pc_d, %far0 : i1
  %not_cont = scalar.xori %prev_cont, %true : i1
  %leader = scalar.andi %far0, %not_cont : i1""")
    # members
    for k in range(1, T):
        prev = "%leader" if k == 1 else f"%mem{k - 1}"
        A(f"""  %sr{k} = scalar.cmpi eq, %row{k}, %row0 : i32
  %pex{k} = scalar.addi %pos0, %ck{k}_i32 : i32
  %sq{k} = scalar.cmpi eq, %pos{k}, %pex{k} : i32
  %al{k} = index.add %t_rem, %ck{k} : index
  %ina{k} = index.cmp ult, %al{k}, %cT : index
  %ma{k} = scalar.andi %far{k}, %sr{k} : i1
  %mb{k} = scalar.andi %ma{k}, %sq{k} : i1
  %mc{k} = scalar.andi %mb{k}, %ina{k} : i1
  %mem{k} = scalar.andi {prev}, %mc{k} : i1""")
    live = ["%leader"] + [f"%mem{k}" for k in range(1, T)]
    A(f"""  %pos0_bound = scalar.assume %pos0 [range(%pos0, 0, 262143)] : i32
  %pos0i = index.cast %pos0_bound : i32 to index
  %lcount0 = scf.select %leader, %c1_i32, %c0_i32 : i32""")
    prev = "%lcount0"
    for k in range(1, T):
        A(f"  %lstep{k} = scf.select %mem{k}, %c1_i32, %c0_i32 : i32\n  %lcount{k} = scalar.addi {prev}, %lstep{k} : i32")
        prev = f"%lcount{k}"
    A(f"  %nlive = scalar.addi {prev}, %c0_i32 : i32")
    A("  %big = index.constant 1073741824 : index")
    # per-token split ranges
    for j in range(T):
        A(f"""  %cj{j} = index.constant {j} : index
  %visa{j} = index.add %pos0i, %ck{j} : index
  %visb{j} = index.add %visa{j}, %c1 : index
  %vis{j} = index.assume %visb{j} [range(%visb{j}, 1, 262144)] : index
  %se{j} = func.call @g_split_count(%vis{j}) : (index) -> (index)
  %ins{j} = index.cmp ult, %split, %se{j} : index
  %act{j} = scalar.andi {live[j]}, %ins{j} : i1
  %sl{j} = index.sub %se{j}, %c1 : index
  %pp{j} = index.add %vis{j}, %sl{j} : index
  %per{j} = index.div %pp{j}, %se{j} : index
  %srr{j} = index.mul %split, %per{j} : index
  %lo{j} = index.min %srr{j}, %vis{j} : index
  %spr{j} = index.add %srr{j}, %per{j} : index
  %hi{j} = index.min %spr{j}, %vis{j} : index
  %los{j} = scf.select %act{j}, %lo{j}, %big : index
  %his{j} = scf.select %act{j}, %hi{j}, %c0 : index""")
    A("  %lo_u0 = index.add %los0, %c0 : index\n  %hi_u0 = index.add %his0, %c0 : index\n  %any0 = scalar.andi %act0, %true : i1")
    for j in range(1, T):
        A(f"""  %lo_u{j} = index.min %lo_u{j - 1}, %los{j} : index
  %hi_u{j} = index.max %hi_u{j - 1}, %his{j} : index
  %any{j} = scalar.ori %any{j - 1}, %act{j} : i1""")
    A(f"  scf.if %any{T - 1} {{")
    # ---- the run: columns, constants, Q fragments
    A(fill("""    %row_bound = scalar.assume %row0 [range(%row0, 0, 2147483646)] : i32
    %row = index.cast %row_bound : i32 to index
    %origin = index.constant 0 : offset
    %lane_col = index.rem %lane, %c16 : index
    %lane_half = index.div %lane, %c16 : index
    %hi_half = index.cmp ne, %lane_half, %c0 : index
    %half32 = index.cast %lane_half : index to i32
    %c4i = index.constant 4 : index
    %c64i = index.constant 64 : index
    %cgrp = index.div %wave, %c2 : index
    %dh = index.rem %wave, %c2 : index
    %dh64 = index.mul %dh, %c64i : index
    %w16 = index.mul %cgrp, %c16 : index
    %col = index.add %w16, %lane_col : index
    %cG = index.constant <<G>> : index
    %jc = index.div %col, %cG : index
    %hc = index.rem %col, %cG : index
    %cNC = index.constant <<NC>> : index
    %col_in = index.cmp ult, %col, %cNC : index
    %cWC = index.constant <<WC>> : index
    %is_compute = index.cmp ult, %wave, %cWC : index
    %padded = index.add %context, %c15 : index
    %blocks = index.div %padded, %c16 : index
    %k_plane = index.mul %layer, %c2 : index
    %v_plane = index.add %k_plane, %c1 : index
    %nlive_idx = index.cast %nlive : i32 to index
    %pos_max_a = index.add %pos0i, %nlive_idx : index
    %pos_max = index.sub %pos_max_a, %c1 : index
""", G=G, NC=NC, WC=WC))
    # per column range/active (select over the token of the column)
    A("""    %lo_c_s = index.cast %big : index to i32
    %zero_i32 = scalar.constant 0 : i32
    %false = scalar.constant false : i1""")
    for j in range(T):
        pl, ph, pa = ("%lo_c_s", "%zero_i32", "%false") if j == 0 else (f"%loc{j - 1}", f"%hic{j - 1}", f"%actc{j - 1}")
        A(f"""    %isj{j} = index.cmp eq, %jc, %cj{j} : index
    %los{j}_i = index.cast %los{j} : index to i32
    %his{j}_i = index.cast %his{j} : index to i32
    %loc{j} = scf.select %isj{j}, %los{j}_i, {pl} : i32
    %hic{j} = scf.select %isj{j}, %his{j}_i, {ph} : i32
    %actc{j} = scf.select %isj{j}, %act{j}, {pa} : i1""")
    A(f"""    %lo_c = scalar.addi %loc{T - 1}, %zero_i32 : i32
    %hi_c = scalar.addi %hic{T - 1}, %zero_i32 : i32
    %act_c0 = scalar.andi %actc{T - 1}, %true : i1
    %act_c = scalar.andi %act_c0, %col_in : i1
    %lo_v = vector.splat %lo_c : vector<8xi32>
    %hi_v = vector.splat %hi_c : vector<8xi32>
    %tq_a = index.add %token, %jc : index
    %tq_c = index.min %tq_a, %tokens_less : index
    %tq = index.assume %tq_c [lt(%tq_c, %tokens)] : index
    %cgq = index.constant {G} : index
    %hd_a = index.madd %kv_head, %cgq, %hc : index
    %hd = index.assume %hd_a [range(%hd_a, 0, {QH - 1})] : index
    %q_view = buffer.view %q[%zero] : buffer -> view<[%tokens]x{QH}x{NF}x16xf32>""")
    for c in range(NF):
        if c == 4:   # the second half of Q loads after the first is converted: 8 raw f32 vectors at once would hold 128 VGPRs
            A("""    %qdep_e = vector.extract %qh3[%c0] : vector<16xf16> -> f16
    %qdep_f = scalar.extf %qdep_e : f16 to f32
    %qdep_m = scalar.mulf %qdep_f, %zero_f : f32
    %qdep_n = scalar.cmpf uno, %qdep_m, %qdep_m : f32
    %qdep = scf.select %qdep_n, %c1, %c0 : index""")
        A(f"""    %qc{c} = index.constant {c} : index
{"    %qcr" + str(c) + " = index.add %qc" + str(c) + ", %qdep : index" + chr(10) + "    %qcl" + str(c) + " = index.assume %qcr" + str(c) + " [range(%qcr" + str(c) + ", 0, 7)] : index" if c >= 4 else ""}
    %qraw{c} = vector.load %q_view[%tq, %hd, {"%qcl" + str(c) if c >= 4 else "%qc" + str(c)}, %c0] : view<[%tokens]x{QH}x{NF}x16xf32> -> vector<16xf32>
    %qh{c} = vector.fptrunc %qraw{c} : vector<16xf32> to vector<16xf16>
    %qf{c} = vector.fragment<rhs> %qh{c} shape [%c16, %c16] : vector<16xf16>""")
    A(fill("""    %zero8 = vector.constant 0.0 : vector<8xf32>
    %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
    %scale = scalar.constant <<SCALE>> : f32
    %scale_v = vector.splat %scale : vector<8xf32>
    %neg_v = vector.splat %neg_big : vector<8xf32>
    %zero_v8 = vector.constant 0.0 : vector<8xf32>
    %exchange = scalar.constant 16 : i32
    %width = scalar.constant 32 : i32
    %row_off = vector.iota %half32 step %c2_i32 : vector<8xi32>
    %pos0_i32 = scalar.addi %pos0, %c0_i32 : i32
    %zero16h = vector.constant 0.0 : vector<16xf16>
    %iota16 = vector.iota %c0_i32 step %c1_i32 : vector<16xi32>
    %c15_i32 = scalar.constant 15 : i32
    %c32_i32 = scalar.constant 32 : i32
    %ctpk = index.constant <<TPK>> : index
    %cdpt = index.constant <<DPT>> : index
    %st_key_value = index.div %workitem, %ctpk : index
    %st_key = index.assume %st_key_value [range(%st_key_value, 0, 31)] : index
    %st_seg = index.rem %workitem, %ctpk : index
    %st_dim_value = index.mul %st_seg, %cdpt : index
    %st_dim = index.assume %st_dim_value [range(%st_dim_value, 0, <<DIMMAX>>)] : index
    %kz = vector.constant 0.0 : vector<<<DPT>>xf16>
    %k_bytes = index.constant <<KBYTES>> : offset
    %k_tile = buffer.alloca<workgroup> align(16) %k_bytes : buffer
    %k_view = buffer.view %k_tile[%origin] : buffer -> view<<<KT>>x<<DP8>>xf16>
    %vt_bytes = index.constant <<VTBYTES>> : offset
    %vt_tile = buffer.alloca<workgroup> align(16) %vt_bytes : buffer
    %vt_view = buffer.view %vt_tile[%origin] : buffer -> view<<<D>>x<<VTC>>xf16>
    %lo_ub = index.assume %lo_u<<T1>> [range(%lo_u<<T1>>, 0, 262143)] : index
    %hi_ub = index.assume %hi_u<<T1>> [range(%hi_u<<T1>>, 0, 262144)] : index
    %key_low_q = index.div %lo_ub, %c32 : index
    %key_low = index.mul %key_low_q, %c32 : index
    %key_span_v = index.sub %hi_ub, %key_low : index
    %key_span = index.assume %key_span_v [range(%key_span_v, 0, 262144)] : index
    %span_round = index.add %key_span, %c31x : index
    %key_tiles = index.div %span_round, %c32 : index
""", SCALE=scale, TPK=TPK, DPT=DPT, DIMMAX=D - DPT, KBYTES=KT * DP8 * 2, VTBYTES=D * VTC * 2, KT=KT, DP8=DP8,
        D=D, VTC=VTC, T1=T - 1))
    for c in range(NF):
        A(f"    %dc{c} = index.constant {c * 16} : index")
    for i in range(8):
        A(f"    %di{i} = index.constant {i} : index")
    for d in range(NO):
        A(f"    %drow{d}_c = index.constant {d * 16} : index\n    %drow{d} = index.add %dh64, %drow{d}_c : index")
    for kb in range(KB):
        A(f"    %kb{kb}_row = index.constant {kb * 16} : index\n    %kbo{kb} = scalar.constant {kb * 16} : i32")
    acc0 = ", ".join(f"%a{d} = %init : vector<8xf32>" for d in range(NO))
    vt8 = ", ".join(["vector<8xf32>"] * NO)
    A(f"""    %m_final, %l_final, {", ".join(f"%o{d}" for d in range(NO))} = scf.for %kt = [%c0 to %key_tiles step %c1](%m = %neg_big : f32, %l = %zero_f : f32, {acc0}) -> (f32, f32, {vt8}) {{
      %tile_first = index.madd %kt, %c32, %key_low : index
      %key_pos = index.add %tile_first, %st_key : index
      %key_valid = index.cmp ule, %key_pos, %pos_max : index
      %kvals, %vvals = scf.if %key_valid -> (vector<{DPT}xf16>, vector<{DPT}xf16>) {{
        %kp = index.assume %key_pos [range(%key_pos, 0, 262143)] : index
        %k_offset = func.call @g_kv_offset(%origins, %row, %kp, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
        %v_offset = func.call @g_kv_offset(%origins, %row, %kp, %blocks, %v_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
        %k_cache = buffer.view %arena[%k_offset] : buffer -> view<{KVH}x{D}xf16>
        %v_cache = buffer.view %arena[%v_offset] : buffer -> view<{KVH}x{D}xf16>
        %kh = vector.load %k_cache[%kv_head, %st_dim] : view<{KVH}x{D}xf16> -> vector<{DPT}xf16>
        %vh = vector.load %v_cache[%kv_head, %st_dim] : view<{KVH}x{D}xf16> -> vector<{DPT}xf16>
        scf.yield %kh, %vh : vector<{DPT}xf16>, vector<{DPT}xf16>
      }} else {{
        scf.yield %kz, %kz : vector<{DPT}xf16>, vector<{DPT}xf16>
      }}
      vector.store %kvals, %k_view[%st_key, %st_dim] : vector<{DPT}xf16>, view<{KT}x{DP8}xf16>
      scf.for %jj = [%c0 to %cdpt step %c1] unroll {{
        %vj = vector.extract %vvals[%jj] : vector<{DPT}xf16> -> f16
        %vt_row_value = index.add %st_dim, %jj : index
        %vt_row = index.assume %vt_row_value [range(%vt_row_value, 0, {D - 1})] : index
        view.store %vj, %vt_view[%vt_row, %st_key] : f16, view<{D}x{VTC}xf16>
      }}
      kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
      %tf32 = index.cast %tile_first : index to i32
      %tf_end = scalar.addi %tf32, %c32_i32 : i32
      %t_lo = scalar.cmpi slt, %tf32, %hi_c : i32
      %t_hi = scalar.cmpi sgt, %tf_end, %lo_c : i32
      %t_in = scalar.andi %t_lo, %t_hi : i1
      %act_t = scalar.andi %t_in, %act_c : i1
      %cs_m, %cs_l, %cs_o0, %cs_o1, %cs_o2, %cs_o3 = scf.if %is_compute -> (f32, f32, {vt8}) {{""".replace("{DPT}", str(DPT)).replace("{KVH}", str(KVH)).replace("{D}", str(D)).replace("{KT}", str(KT)).replace("{DP8}", str(DP8)).replace("{VTC}", str(VTC)).replace("{D - 1}", str(D - 1)).replace("{vt8}", vt8))
    # compute (inside the if): per 16-key block QK^T, masks, online softmax, PV (one block at a time: fewer live temporaries)
    mc, lc, ac = "%m", "%l", [f"%a{d}" for d in range(NO)]
    for kb in range(KB):
        acc = "%init"
        for c in range(NF):
            A(f"""        %kl{kb}_{c} = vector.fragment.load<lhs> %k_view[%kb{kb}_row, %dc{c}] shape [%c16, %c16] : view<{KT}x{DP8}xf16> -> vector<16xf16>
        %s{kb}_{c} = vector.mma %kl{kb}_{c}, %qf{c}, {acc} : vector<16xf16>, vector<16xf16>, vector<8xf32>""")
            acc = f"%s{kb}_{c}"
        A(f"""        %sc{kb} = vector.mulf %s{kb}_{NF - 1}, %scale_v : vector<8xf32>
        %base{kb} = scalar.addi %tf32, %kbo{kb} : i32
        %base{kb}_v = vector.splat %base{kb} : vector<8xi32>
        %kpos{kb} = vector.addi %base{kb}_v, %row_off : vector<8xi32>
        %inlo{kb} = vector.cmpi sge, %kpos{kb}, %lo_v : vector<8xi32> -> vector<8xi1>
        %inhi{kb} = vector.cmpi slt, %kpos{kb}, %hi_v : vector<8xi32> -> vector<8xi1>
        %valid{kb} = vector.andi %inlo{kb}, %inhi{kb} : vector<8xi1>
        %masked{kb} = vector.select %valid{kb}, %sc{kb}, %neg_v : vector<8xf32>
        %lmax{kb} = vector.reduce<maxnumf> %masked{kb}, %neg_big : vector<8xf32>, f32
        %omax{kb}, %omax_u{kb} = kernel.subgroup.shuffle<xor> %lmax{kb}, %exchange, %width : f32, i32, i32
        %tmax{kb} = scalar.maxnumf %lmax{kb}, %omax{kb} : f32
        %m_new{kb} = scalar.maxnumf {mc}, %tmax{kb} : f32
        %m_shift{kb} = scalar.subf {mc}, %m_new{kb} : f32
        %alpha{kb} = scalar.expf<afn> %m_shift{kb} : f32
        %m_new_v{kb} = vector.splat %m_new{kb} : vector<8xf32>
        %sh{kb} = vector.subf %masked{kb}, %m_new_v{kb} : vector<8xf32>
        %pe{kb} = vector.expf<afn> %sh{kb} : vector<8xf32>
        %p{kb} = vector.select %valid{kb}, %pe{kb}, %zero_v8 : vector<8xf32>
        %ls{kb} = vector.reduce<addf> %p{kb}, %zero_f : vector<8xf32>, f32""")
        for i in range(8):
            A(f"""        %pi{kb}_{i} = vector.extract %p{kb}[%di{i}] : vector<8xf32> -> f32
        %po{kb}_{i}, %pu{kb}_{i} = kernel.subgroup.shuffle<xor> %pi{kb}_{i}, %exchange, %width : f32, i32, i32""")
        A(f"""        %partner{kb} = vector.from_elements {", ".join(f"%po{kb}_{i}" for i in range(8))} : vector<8xf32>
        %even{kb} = scf.select %hi_half, %partner{kb}, %p{kb} : vector<8xf32>
        %odd{kb} = scf.select %hi_half, %p{kb}, %partner{kb} : vector<8xf32>
        %pt{kb}_f = vector.interleave<0> %even{kb}, %odd{kb} : vector<8xf32>, vector<8xf32> -> vector<16xf32>
        %pt{kb}_h = vector.fptrunc %pt{kb}_f : vector<16xf32> to vector<16xf16>
        %pt{kb} = vector.fragment<rhs> %pt{kb}_h shape [%c16, %c16] : vector<16xf16>
        %osum{kb}, %osum_u{kb} = kernel.subgroup.shuffle<xor> %ls{kb}, %exchange, %width : f32, i32, i32
        %tsum{kb} = scalar.addf %ls{kb}, %osum{kb} : f32
        %l_scaled{kb} = scalar.mulf {lc}, %alpha{kb} : f32
        %l_new{kb} = scalar.addf %l_scaled{kb}, %tsum{kb} : f32
        %alpha_v{kb} = vector.splat %alpha{kb} : vector<8xf32>
        %nb_last{kb} = scalar.addi %base{kb}, %c15_i32 : i32
        %need{kb} = scalar.cmpi sgt, %nb_last{kb}, %pos0_i32 : i32""")
        outs = [f"%ao{d}_{kb}" for d in range(NO)]
        A(f"        {', '.join(outs)} = scf.if %need{kb} -> ({vt8}) {{")
        # slow path: V zeroed beyond each token's own position for that token's columns
        for j in range(T):
            A(f"""          %thr{j}_{kb}_a = scalar.addi %pos0_i32, %ck{j}_i32 : i32
          %thr{j}_{kb} = scalar.subi %thr{j}_{kb}_a, %base{kb} : i32
          %thr_v{j}_{kb} = vector.splat %thr{j}_{kb} : vector<16xi32>
          %mk{j}_{kb} = vector.cmpi sle, %iota16, %thr_v{j}_{kb} : vector<16xi32> -> vector<16xi1>""")
        res = []
        for d in range(NO):
            A(f"          %os{d}_{kb} = vector.mulf {ac[d]}, %alpha_v{kb} : vector<8xf32>")
            prev = f"%os{d}_{kb}"
            for j in range(T):
                A(f"""          %vl{d}_{j}_{kb} = vector.fragment.load<lhs> %vt_view[%drow{d}, %kb{kb}_row] shape [%c16, %c16] : view<{D}x{VTC}xf16> -> vector<16xf16>
          %vs{d}_{j}_{kb} = vector.select %mk{j}_{kb}, %vl{d}_{j}_{kb}, %zero16h : vector<16xf16>
          %vm{d}_{j}_{kb} = vector.fragment<lhs> %vs{d}_{j}_{kb} shape [%c16, %c16] : vector<16xf16>
          %rm{d}_{j}_{kb} = vector.mma %vm{d}_{j}_{kb}, %pt{kb}, %os{d}_{kb} : vector<16xf16>, vector<16xf16>, vector<8xf32>
          %rc{d}_{j}_{kb} = scf.select %isj{j}, %rm{d}_{j}_{kb}, {prev} : vector<8xf32>""")
                prev = f"%rc{d}_{j}_{kb}"
            res.append(prev)
        A(f"          scf.yield {', '.join(res)} : {vt8}\n        }} else {{")
        res = []
        for d in range(NO):
            A(f"""          %fos{d}_{kb} = vector.mulf {ac[d]}, %alpha_v{kb} : vector<8xf32>
          %fv{d}_{kb} = vector.fragment.load<lhs> %vt_view[%drow{d}, %kb{kb}_row] shape [%c16, %c16] : view<{D}x{VTC}xf16> -> vector<16xf16>
          %fr{d}_{kb} = vector.mma %fv{d}_{kb}, %pt{kb}, %fos{d}_{kb} : vector<16xf16>, vector<16xf16>, vector<8xf32>""")
            res.append(f"%fr{d}_{kb}")
        A(f"          scf.yield {', '.join(res)} : {vt8}\n        }}")
        mc, lc, ac = f"%m_new{kb}", f"%l_new{kb}", outs
    A(f"        %m_keep = scf.select %act_t, {mc}, %m : f32\n        %l_keep = scf.select %act_t, {lc}, %l : f32")
    for d in range(NO):
        A(f"        %ok{d} = scf.select %act_t, {ac[d]}, %a{d} : vector<8xf32>")
    A(f"        scf.yield %m_keep, %l_keep, {', '.join(f'%ok{d}' for d in range(NO))} : f32, f32, {vt8}\n      }} else {{")
    A(f"        scf.yield %m, %l, {', '.join(f'%a{d}' for d in range(NO))} : f32, f32, {vt8}\n      }}")
    A("      kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    A(f"      scf.yield %cs_m, %cs_l, {', '.join(f'%cs_o{d}' for d in range(NO))} : f32, f32, {vt8}")
    A("    }")
    # store
    A(f"""    %store_ok = scalar.andi %act_c, %is_compute : i1
    scf.if %store_ok {{
      %acc_flat = buffer.view %part_acc[%zero] : buffer -> view<[%tokens]x{QH}x{S}x{D}xf32>
      %ml_view = buffer.view %part_ml[%zero] : buffer -> view<[%tokens]x{QH}x{S}x2xf32>""")
    for d in range(NO):
        for i in range(8):
            A(f"""      %ov{d}_{i} = vector.extract %o{d}[%di{i}] : vector<8xf32> -> f32
      %dx{d}_{i}_c = index.constant {d * 16 + 2 * i} : index
      %dx{d}_{i}_b = index.add %lane_half, %dh64 : index
      %dx{d}_{i}_a = index.add %dx{d}_{i}_b, %dx{d}_{i}_c : index
      %dx{d}_{i} = index.assume %dx{d}_{i}_a [range(%dx{d}_{i}_a, 0, {D - 1})] : index
      view.store %ov{d}_{i}, %acc_flat[%tq, %hd, %split, %dx{d}_{i}] : f32, view<[%tokens]x{QH}x{S}x{D}xf32>""")
    A(f"""      scf.if %hi_half_not {{
        view.store %m_final, %ml_view[%tq, %hd, %split, 0] : f32, view<[%tokens]x{QH}x{S}x2xf32>
        view.store %l_final, %ml_view[%tq, %hd, %split, 1] : f32, view<[%tokens]x{QH}x{S}x2xf32>
      }}
    }}
  }}
  kernel.return
}}
""")
    text = "\n".join(o)
    abl = os.environ.get("ATTN_MMA_ABL", "")
    if abl == "nostage":   # timing only: no K/V loads
        text = text.replace("%key_valid = index.cmp ule, %key_pos, %pos_max : index", "%key_valid = index.cmp ult, %key_pos, %c0 : index", 1)
    if abl == "nocompute":
        text = text.replace("%is_compute = index.cmp ult, %wave, %cWC : index", "%is_compute = index.cmp ult, %wave, %c0 : index", 1)
    text = text.replace("    %store_ok = scalar.andi", "    %hi_half_not = index.cmp eq, %lane_half, %c0 : index\n    %store_ok = scalar.andi", 1)
    return text
