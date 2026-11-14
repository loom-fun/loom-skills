"""MoE FFN kernels (LFM2-MoE style): sigmoid router with selection bias, normalized top-k
weights, per-assignment expert matvecs (assignment a = token * KU + slot), weighted combine.

g_moe_route        h[t] -> ids[t, KU] (i32), wts[t, KU] (f32)
g_moe_gu_<fmt>     act[a, FE] = silu(Wg[e] h[t]) * (Wu[e] h[t])     (e = ids[a], t = a / KU)
g_moe_dn_<fmt>     ys[a, H]  = Wd[e] act[a]
g_moe_combine      y[t, H]   = sum_k wts[t, k] ys[t*KU + k, :]
"""
FMT = {"q4_k": 144, "q5_k": 176, "q6_k": 210, "q8_0": 34}
SB = {"q4_k": 256, "q5_k": 256, "q6_k": 256, "q8_0": 32}


def row_bytes(fmt, k):
    return k // SB[fmt] * FMT[fmt]


def route(H, E, KU, gating, norm, bias):
    HG = H // 32
    assert E % 8 == 0 and E <= 256 and KU <= 8
    import os as _os0
    NWV = int(_os0.environ.get("MOE_ROUTE_WAVES", "32"))
    EPW = E // NWV
    L = [f"""
// Router: wave w scores experts w*{EPW}..; lane 0 then picks the top {KU} of sigmoid(logit) + bias.
kernel.def export("g_moe_route") @g_moe_route(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant {NWV * 32} : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %x: buffer, %w: buffer, %b: buffer, %ids: buffer, %wts: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, {NWV - 1})] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  %cHG = index.constant {HG} : index
  %cE = index.constant {E} : index
  %zero_f = scalar.constant 0.0 : f32
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
  %w_view = buffer.view %w[%zero] : buffer -> view<{E}x{HG}x32xf32>
  %b_view = buffer.view %b[%zero] : buffer -> view<{E}xf32>
  %lg_bytes = index.constant {E * 4} : offset
  %lg = buffer.alloca<workgroup> align(16) %lg_bytes : buffer
  %lg_view = buffer.view %lg[%zero] : buffer -> view<{E}xf32>
  %first = index.cmp eq, %lane, %c0 : index
  %cEPW = index.constant {EPW} : index
  %e0 = index.mul %wave, %cEPW : index
  scf.for %j = [%c0 to %cEPW step %c1] {{
    %e_raw = index.add %e0, %j : index
    %e = index.assume %e_raw [range(%e_raw, 0, {E - 1})] : index
    %part = scf.for %g = [%lane to %cHG step %c32](%acc = %zero_f : f32) -> (f32) {{
      %bg = index.assume %g [range(%g, 0, {HG - 1})] : index
      %xv = vector.load %x_view[%token, %bg, %c0] : view<[%tokens]x{HG}x32xf32> -> vector<32xf32>
      %wv = vector.load %w_view[%e, %bg, %c0] : view<{E}x{HG}x32xf32> -> vector<32xf32>
      %d = vector.dotf<reassoc|contract> %wv, %xv, %zero_f : vector<32xf32>, vector<32xf32>, f32
      %nx = scalar.addf %acc, %d : f32
      scf.yield %nx : f32
    }}
    %logit = kernel.subgroup.reduce<addf> %part : f32
    scf.if %first {{
      view.store %logit, %lg_view[%e] : f32, view<{E}xf32>
    }}
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %t0 = kernel.workitem.id<x> : index
  %only = index.cmp eq, %t0, %c0 : index
  scf.if %only {{
    %ids_view = buffer.view %ids[%zero] : buffer -> view<[%tokens]x{KU}xi32>
    %wts_view = buffer.view %wts[%zero] : buffer -> view<[%tokens]x{KU}xf32>
    %neg_big = scalar.constant -3.0e38 : f32
    %minus1 = scalar.constant -1 : i32"""]
    prev_ids = []
    for k in range(KU):
        excl = "\n".join(f"""        %ex{k}_{p} = scalar.cmpi eq, %ei32, %id{p} : i32""" for p in range(k))
        any_ex = "%false"
        L.append(f"""    %best{k}, %id{k}, %pr{k} = scf.for %e = [%c0 to %cE step %c1](%bv = %neg_big : f32, %bi = %minus1 : i32, %bp = %zero_f : f32) -> (f32, i32, f32) {{
      %be = index.assume %e [range(%e, 0, {E - 1})] : index
      %lgt = view.load %lg_view[%be] : view<{E}xf32> -> f32
      %prob = {"scalar.logisticf<afn> %lgt : f32" if gating == 2 else "scalar.addf %lgt, %zero_f : f32"}
      %bias_v = view.load %b_view[%be] : view<{E}xf32> -> f32
      %score = {"scalar.addf %prob, %bias_v : f32" if bias else "scalar.addf %prob, %zero_f : f32"}
      %ei32 = index.cast %be : index to i32""")
        if k:
            L.append(excl)
            acc = f"%ex{k}_0"
            for p in range(1, k):
                L.append(f"      %exo{k}_{p} = scalar.ori {acc}, %ex{k}_{p} : i1")
                acc = f"%exo{k}_{p}"
            L.append(f"""      %better_raw = scalar.cmpf ogt, %score, %bv : f32
      %true_c = scalar.constant true : i1
      %not_ex = scalar.xori {acc}, %true_c : i1
      %better = scalar.andi %better_raw, %not_ex : i1""")
        else:
            L.append("      %better = scalar.cmpf ogt, %score, %bv : f32")
        L.append(f"""      %nv = scf.select %better, %score, %bv : f32
      %ni = scf.select %better, %ei32, %bi : i32
      %np = scf.select %better, %prob, %bp : f32
      scf.yield %nv, %ni, %np : f32, i32, f32
    }}""")
    s = "%pr0"
    for k in range(1, KU):
        L.append(f"    %ws{k} = scalar.addf {s}, %pr{k} : f32"); s = f"%ws{k}"
    if norm:
        L.append(f"""    %min_sum = scalar.constant 6.103515625e-05 : f32
    %wsum = scalar.maxnumf {s}, %min_sum : f32""")
    for k in range(KU):
        if norm:
            L.append(f"    %wn{k} = scalar.divf %pr{k}, %wsum : f32")
        else:
            L.append(f"    %wn{k} = scalar.addf %pr{k}, %zero_f : f32")
        L.append(f"""    view.store %id{k}, %ids_view[%token, {k}] : i32, view<[%tokens]x{KU}xi32>
    view.store %wn{k}, %wts_view[%token, {k}] : f32, view<[%tokens]x{KU}xf32>""")
    L.append("""  }
  kernel.return
}
""")
    import os as _os
    if _os.environ.get("MOE_ROUTE_PAR", "1") not in ("", "0"):
        first = L[0]
        cut = first.index("  %t0 = kernel.workitem.id<x> : index")
        return first[:cut] + _route_par_select(E, KU, gating, norm, bias)
    return "\n".join(L)




def _route_par_select(E, KU, gating, norm, bias):
    """Wave 0 selects the top-KU experts in parallel (lane l holds experts l and l+32): per pick, a max
    reduce over the scores and a min reduce over the indices of the lanes at the max (lowest index
    wins ties, as the serial loop's strict '>' does). Same arithmetic as the serial loop per expert."""
    assert E == 64
    L = ["""  %is_w0 = index.cmp eq, %wave, %c0 : index
  scf.if %is_w0 {
    %ids_view = buffer.view %ids[%zero] : buffer -> view<[%tokens]x""" + str(KU) + """xi32>
    %wts_view = buffer.view %wts[%zero] : buffer -> view<[%tokens]x""" + str(KU) + """xf32>
    %neg_big = scalar.constant -3.0e38 : f32
    %big_idx = scalar.constant 1.0e9 : f32
    %ea = index.add %lane, %c0 : index
    %eb_raw = index.add %lane, %c32 : index
    %ea_b = index.assume %ea [range(%ea, 0, 31)] : index
    %eb_b = index.assume %eb_raw [range(%eb_raw, 32, 63)] : index"""]
    for s_, e in (("a", "%ea_b"), ("b", "%eb_b")):
        L.append(f"""    %lg_{s_} = view.load %lg_view[{e}] : view<{E}xf32> -> f32
    %p_{s_} = {"scalar.logisticf<afn> %lg_" + s_ + " : f32" if gating == 2 else "scalar.addf %lg_" + s_ + ", %zero_f : f32"}
    %bias_{s_} = view.load %b_view[{e}] : view<{E}xf32> -> f32
    %sc_{s_}_0 = {"scalar.addf %p_" + s_ + ", %bias_" + s_ + " : f32" if bias else "scalar.addf %p_" + s_ + ", %zero_f : f32"}
    %ei_{s_} = index.cast {e} : index to i32
    %ef_{s_} = scalar.sitofp %ei_{s_} : i32 to f32""")
    for k in range(KU):
        L.append(f"""    %lb{k} = scalar.maxnumf %sc_a_{k}, %sc_b_{k} : f32
    %a_ge{k} = scalar.cmpf oge, %sc_a_{k}, %sc_b_{k} : f32
    %li{k} = scf.select %a_ge{k}, %ef_a, %ef_b : f32
    %m{k} = kernel.subgroup.reduce<maxnumf> %lb{k} : f32
    %at{k} = scalar.cmpf oeq, %lb{k}, %m{k} : f32
    %cand{k} = scf.select %at{k}, %li{k}, %big_idx : f32
    %ncand{k} = scalar.negf %cand{k} : f32
    %nmin{k} = kernel.subgroup.reduce<maxnumf> %ncand{k} : f32
    %idf{k} = scalar.negf %nmin{k} : f32
    %hit_a{k} = scalar.cmpf oeq, %ef_a, %idf{k} : f32
    %hit_b{k} = scalar.cmpf oeq, %ef_b, %idf{k} : f32
    %pa{k} = scf.select %hit_a{k}, %p_a, %zero_f : f32
    %pb{k} = scf.select %hit_b{k}, %p_b, %zero_f : f32
    %pab{k} = scalar.addf %pa{k}, %pb{k} : f32
    %pr{k} = kernel.subgroup.reduce<addf> %pab{k} : f32
    %id{k} = scalar.fptosi %idf{k} : f32 to i32
    %sc_a_{k + 1} = scf.select %hit_a{k}, %neg_big, %sc_a_{k} : f32
    %sc_b_{k + 1} = scf.select %hit_b{k}, %neg_big, %sc_b_{k} : f32""")
    s_ = "%pr0"
    for k in range(1, KU):
        L.append(f"    %ws{k} = scalar.addf {s_}, %pr{k} : f32"); s_ = f"%ws{k}"
    if norm:
        L.append(f"""    %min_sum = scalar.constant 6.103515625e-05 : f32
    %wsum = scalar.maxnumf {s_}, %min_sum : f32""")
    L.append("""    %lane0 = index.cmp eq, %lane, %c0 : index
    scf.if %lane0 {""")
    for k in range(KU):
        L.append(f"      %wn{k} = scalar.divf %pr{k}, %wsum : f32" if norm else f"      %wn{k} = scalar.addf %pr{k}, %zero_f : f32")
        L.append(f"""      view.store %id{k}, %ids_view[%token, {k}] : i32, view<[%tokens]x{KU}xi32>
      view.store %wn{k}, %wts_view[%token, {k}] : f32, view<[%tokens]x{KU}xf32>""")
    L.append("""    }
  }
  kernel.return
}
""")
    return "\n".join(L)


def expert_mv(fmt, K, N, E, KU, R, mode):
    """mode 'gu': act[a] = silu(Wg[e] x[a/KU]) * Wu[e] x[a/KU]; mode 'dn': ys[a] = Wd[e] x[a]."""
    groups = K // 32
    lanes = (groups + 31) // 32 * 32
    waves = lanes // 32
    rb = row_bytes(fmt, K)
    eb = rb * N
    nw = 2 if mode == "gu" else 1
    name = f"g_moe_{mode}_{fmt}_k{K}_r{R}"
    wargs = "%w: buffer, %w2: buffer" if nw == 2 else "%w: buffer"
    xrows = "%xrows" if mode == "gu" else "%tokens"
    L = [f"""
kernel.def export("{name}") @{name}(%n: index, %tokens: index) {{
  %c1 = index.constant 1 : index
  %lanes = index.constant {lanes} : index
  %rows_per = index.constant {R} : index
  %row_groups = index.div %n, %rows_per : index
  kernel.launch.config workgroups(%tokens, %row_groups, %c1) workgroup_size(%lanes, %c1, %c1) : index
}} launch(%n_arg: index, %tokens_arg: index, %ids: buffer, %x: buffer, {wargs}, %y: buffer) {{
  %n = index.assume %n_arg [range(%n_arg, {R}, 262144), multiple_of(%n_arg, {R})] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 65536)] : index
  %group_id = kernel.workgroup.id<y> : index
  %a_id = kernel.workgroup.id<x> : index
  %a = index.assume %a_id [lt(%a_id, %tokens)] : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, {lanes - 1})] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, {waves - 1})] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %cKU = index.constant {KU} : index
  %rows_per = index.constant {R} : index
  %cW = index.constant {waves} : index
  %row_bytes = index.constant {rb} : offset
  %expert_bytes = index.constant {eb} : offset
  %block_bytes = index.constant {FMT[fmt]} : offset
  %zero_f = scalar.constant 0.0 : f32
  %zero32 = vector.constant 0.0 : vector<32xf32>
  %c0_i32 = scalar.constant 0 : i32
  %cE1_i32 = scalar.constant {E - 1} : i32
  %ids_view = buffer.view %ids[%zero] : buffer -> view<[%tokens]xi32>
  %eid = view.load %ids_view[%a] : view<[%tokens]xi32> -> i32
  %elo = scalar.maxsi %eid, %c0_i32 : i32
  %ehi = scalar.minsi %elo, %cE1_i32 : i32
  %eb32 = scalar.assume %ehi [range(%ehi, 0, {E - 1})] : i32
  %e = index.cast %eb32 : i32 to index
  %e_off = index.scale %e, %expert_bytes : index, offset -> offset"""]
    if mode == "gu":
        L.append(f"""  %xrows = index.div %tokens, %cKU : index
  %xr_v = index.div %a, %cKU : index
  %xr = index.assume %xr_v [lt(%xr_v, %xrows)] : index""")
    else:
        L.append("  %xr = index.assume %a [lt(%a, %tokens)] : index")
    L.append(f"""  %x_view = buffer.view %x[%zero] : buffer -> view<[{xrows}]x{groups}x32xf32>
  %row_base = index.mul %group_id, %rows_per : index
  %ok = index.cmp ult, %workitem, %groups_c : index""".replace("%groups_c", f"%cgroups"))
    L.insert(-1, f"  %cgroups = index.constant {groups} : index")
    L.append(f"""  %g_last = index.constant {groups - 1} : index
  %g_c = index.min %workitem, %g_last : index
  %g = index.assume %g_c [range(%g_c, 0, {groups - 1})] : index
  %cgpb = index.constant {SB[fmt] // 32} : index
  %blk = index.div %g, %cgpb : index
  %within = index.rem %g, %cgpb : index
  %ba = index.scale %blk, %block_bytes : index, offset -> offset
  %xg = scf.if %ok -> (vector<32xf32>) {{
    %v = vector.load %x_view[%xr, %g, %c0] : view<[{xrows}]x{groups}x32xf32> -> vector<32xf32>
    scf.yield %v : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}
  %xs = vector.reduce<addf> %xg, %zero_f : vector<32xf32>, f32
  %part_bytes = index.constant {waves * R * nw * 4} : offset
  %parts = buffer.alloca<workgroup> align(16) %part_bytes : buffer
  %pv = buffer.view %parts[%zero] : buffer -> view<{nw}x{waves}x{R}xf32>
  %first = index.cmp eq, %lane, %c0 : index
  scf.for %r = [%c0 to %rows_per step %c1] {{
    %row_value = index.add %row_base, %r : index
    %row = index.assume %row_value [lt(%row_value, %n)] : index
    %ro = index.scale %row, %row_bytes : index, offset -> offset
    %rb = index.assume %r [range(%r, 0, {R - 1})] : index
    %o0 = index.add %ro, %ba : offset
    %o = index.add %o0, %e_off : offset""")
    import os as _os
    fold = _os.environ.get("MOE_FOLD", "1") != "0"
    for wi, w in enumerate(["w", "w2"][:nw]):
        if fold and fmt == "q4_k":
            L.append(f"""    %wv{wi}, %sc{wi}, %mn{wi} = scf.if %ok -> (vector<32xf32>, f32, f32) {{
      %v, %sa, %sb = func.call @g_q4_k_codes(%{w}, %o, %within) : (buffer, offset, index) -> (vector<32xf32>, f32, f32)
      scf.yield %v, %sa, %sb : vector<32xf32>, f32, f32
    }} else {{
      scf.yield %zero32, %zero_f, %zero_f : vector<32xf32>, f32, f32
    }}
    %qd{wi} = vector.dotf<reassoc|contract> %wv{wi}, %xg, %zero_f : vector<32xf32>, vector<32xf32>, f32
    %qm{wi} = scalar.mulf %mn{wi}, %xs : f32
    %qs{wi} = scalar.mulf %qd{wi}, %sc{wi} : f32
    %d{wi} = scalar.subf %qs{wi}, %qm{wi} : f32""")
        elif fold:
            L.append(f"""    %wv{wi} = scf.if %ok -> (vector<32xf32>) {{
      %v = func.call @g_{fmt}_values(%{w}, %o, %within) : (buffer, offset, index) -> (vector<32xf32>)
      scf.yield %v : vector<32xf32>
    }} else {{
      scf.yield %zero32 : vector<32xf32>
    }}
    %d{wi} = vector.dotf<reassoc|contract> %wv{wi}, %xg, %zero_f : vector<32xf32>, vector<32xf32>, f32""")
        else:
          L.append(f"""    %d{wi} = scf.if %ok -> (f32) {{
      %dv = func.call @g_{fmt}_dot(%{w}, %o, %within, %xg) : (buffer, offset, index, vector<32xf32>) -> (f32)
      scf.yield %dv : f32
    }} else {{
      scf.yield %zero_f : f32
    }}""")
        L.append(f"""    %s{wi} = kernel.subgroup.reduce<addf> %d{wi} : f32
    scf.if %first {{
      view.store %s{wi}, %pv[{wi}, %wave, %rb] : f32, view<{nw}x{waves}x{R}xf32>
    }}""")
    L.append(f"""  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %outputs = index.constant {R} : index
  %lanes_step = index.constant {lanes} : index
  scf.for %wr0 = [%workitem to %outputs step %lanes_step] {{
    %rr = index.assume %wr0 [range(%wr0, 0, {R - 1})] : index""")
    for wi in range(nw):
        L.append(f"""    %t{wi} = scf.for %ww = [%c0 to %cW step %c1](%acc = %zero_f : f32) -> (f32) {{
      %bw = index.assume %ww [range(%ww, 0, {waves - 1})] : index
      %p = view.load %pv[{wi}, %bw, %rr] : view<{nw}x{waves}x{R}xf32> -> f32
      %nx = scalar.addf %acc, %p : f32
      scf.yield %nx : f32
    }}""")
    if nw == 2:
        L.append("""    %silu = scalar.siluf<afn> %t0 : f32
    %value = scalar.mulf %silu, %t1 : f32""")
    else:
        L.append("    %value = scalar.addf %t0, %zero_f : f32")
    L.append("""    %out_row = index.add %row_base, %rr : index
    func.call @g_store_row(%y, %tokens, %n, %a, %out_row, %value, %c0_i32) : (buffer, index, index, index, index, f32, i32)
  }
  kernel.return
}
""")
    return "\n".join(L)


def combine(H, KU):
    return f"""
kernel.def export("g_moe_combine") @g_moe_combine(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cb = index.constant {(H + 255) // 256} : index
  kernel.launch.config workgroups(%tokens, %cb, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %ys: buffer, %wts: buffer, %y: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %thread_id = kernel.workitem.id<x> : index
  %blk_id = kernel.workgroup.id<y> : index
  %c256 = index.constant 256 : index
  %boff = index.mul %blk_id, %c256 : index
  %c_raw = index.add %thread_id, %boff : index
  %cH = index.constant {H} : index
  %in = index.cmp ult, %c_raw, %cH : index
  scf.if %in {{
    %ch = index.assume %c_raw [range(%c_raw, 0, {H - 1})] : index
    %zero = index.constant 0 : offset
    %zero_f = scalar.constant 0.0 : f32
    %ys_view = buffer.view %ys[%zero] : buffer -> view<[%tokens]x{KU}x{H}xf32>
    %w_view = buffer.view %wts[%zero] : buffer -> view<[%tokens]x{KU}xf32>
    %y_view = buffer.view %y[%zero] : buffer -> view<[%tokens]x{H}xf32>
""" + "\n".join(f"""    %w{k} = view.load %w_view[%token, {k}] : view<[%tokens]x{KU}xf32> -> f32
    %v{k} = view.load %ys_view[%token, {k}, %ch] : view<[%tokens]x{KU}x{H}xf32> -> f32
    %a{k} = scalar.fmaf %w{k}, %v{k}, {"%zero_f" if k == 0 else f"%a{k - 1}"} : f32""" for k in range(KU)) + f"""
    view.store %a{KU - 1}, %y_view[%token, %ch] : f32, view<[%tokens]x{H}xf32>
  }}
  kernel.return
}}
"""


# ---------------------------------------------------------------- prefill: grouped expert GEMMs
TILE = 64


def bucket(E, KU):
    """Assignments a = t*KU + k bucketed by expert into 64-row tiles (flint_moe_bucket pattern)."""
    assert E <= 256
    return f"""
kernel.def export("g_moe_bucket") @g_moe_bucket(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %max_tiles_arg: index, %info: buffer, %ids: buffer, %tile_info: buffer, %rowmap: buffer, %pos: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %max_tiles = index.assume %max_tiles_arg [range(%max_tiles_arg, 1, 65536)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %cKU = index.constant {KU} : index
  %cT = index.constant {TILE} : index
  %c256 = index.constant 256 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %cshift = scalar.constant 6 : i32
  %cTm1 = scalar.constant {TILE - 1} : i32
  %minus_one = scalar.constant -1 : i32
  %cEm1 = scalar.constant {E - 1} : i32
  %cE = index.constant {E} : index
  %lane32 = index.cast %lane : index to i32
  %assignments = index.mul %tokens, %cKU : index
  %ids_view = buffer.view %ids[%zero] : buffer -> view<[%tokens]x{KU}xi32>
  %token_info = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %counter_bytes = index.constant 2048 : offset
  %counters = buffer.alloca<workgroup> align(16) %counter_bytes : buffer
  %count_view = buffer.view %counters[%zero] : buffer -> view<256xi32>
  %cursor_origin = index.constant 1024 : offset
  %cursor_view = buffer.view %counters[%cursor_origin] : buffer -> view<256xi32>
  view.store %c0_i32, %count_view[%lane] : i32, view<256xi32>
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  scf.for %a = [%lane to %assignments step %c256] {{
    %t_value = index.div %a, %cKU : index
    %t = index.assume %t_value [lt(%t_value, %tokens)] : index
    %s_value = index.rem %a, %cKU : index
    %s = index.assume %s_value [range(%s_value, 0, {KU - 1})] : index
    %id = view.load %ids_view[%t, %s] : view<[%tokens]x{KU}xi32> -> i32
    %low = scalar.maxsi %id, %c0_i32 : i32
    %clamped = scalar.minsi %low, %cEm1 : i32
    %bounded = scalar.assume %clamped [range(%clamped, 0, {E - 1})] : i32
    %e = index.cast %bounded : i32 to index
    %live32 = view.load %token_info[%t, 2] : view<[%tokens]x4xi32> -> i32
    %live = scalar.cmpi ne, %live32, %c0_i32 : i32
    scf.if %live {{
      %old = view.atomic.rmw<addi> %c1_i32, %count_view[%e] {{ordering = relaxed, scope = workgroup}} : i32, view<256xi32> -> i32
    }}
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %count32 = view.load %count_view[%lane] : view<256xi32> -> i32
  %padded = scalar.addi %count32, %cTm1 : i32
  %tiles32 = scalar.shrui %padded, %cshift : i32
  %base32 = kernel.workgroup.scan<addi> %tiles32 {{direction = forward, mode = exclusive}} : i32
  %total32 = kernel.workgroup.reduce<addi> %tiles32 : i32
  %info_view = buffer.view %tile_info[%zero] : buffer -> view<[%max_tiles]x2xi32>
  %rowmap_rows = index.mul %max_tiles, %cT : index
  %rowmap_view = buffer.view %rowmap[%zero] : buffer -> view<[%rowmap_rows]xi32>
  %pos_view = buffer.view %pos[%zero] : buffer -> view<[%assignments]xi32>
  %base_bound = scalar.assume %base32 [range(%base32, 0, 65535)] : i32
  %base = index.cast %base_bound : i32 to index
  %tiles_bound = scalar.assume %tiles32 [range(%tiles32, 0, 65535)] : i32
  %tiles = index.cast %tiles_bound : i32 to index
  %count_bound = scalar.assume %count32 [range(%count32, 0, 2097152)] : i32
  %count = index.cast %count_bound : i32 to index
  scf.for %j = [%c0 to %tiles step %c1] {{
    %tile_value = index.add %base, %j : index
    %tile = index.assume %tile_value [lt(%tile_value, %max_tiles)] : index
    %first = index.mul %j, %cT : index
    %remaining = index.sub %count, %first : index
    %full = index.cmp uge, %remaining, %cT : index
    %rows = scf.select %full, %cT, %remaining : index
    %rows32 = index.cast %rows : index to i32
    view.store %lane32, %info_view[%tile, 0] : i32, view<[%max_tiles]x2xi32>
    view.store %rows32, %info_view[%tile, 1] : i32, view<[%max_tiles]x2xi32>
  }}
  %total_bound = scalar.assume %total32 [range(%total32, 0, 65535)] : i32
  %total = index.cast %total_bound : i32 to index
  %unused_start = index.add %total, %lane : index
  scf.for %u = [%unused_start to %max_tiles step %c256] {{
    %bu = index.assume %u [lt(%u, %max_tiles)] : index
    view.store %minus_one, %info_view[%bu, 0] : i32, view<[%max_tiles]x2xi32>
    view.store %c0_i32, %info_view[%bu, 1] : i32, view<[%max_tiles]x2xi32>
  }}
  %first_row = index.mul %base, %cT : index
  %first_row32 = index.cast %first_row : index to i32
  view.store %first_row32, %cursor_view[%lane] : i32, view<256xi32>
  %used_end = index.add %first_row, %count : index
  %tile_rows = index.mul %tiles, %cT : index
  %end_row = index.add %first_row, %tile_rows : index
  scf.for %r = [%used_end to %end_row step %c1] {{
    %row = index.assume %r [range(%r, 0, 2097151), lt(%r, %rowmap_rows)] : index
    view.store %minus_one, %rowmap_view[%row] : i32, view<[%rowmap_rows]xi32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  scf.for %a = [%lane to %assignments step %c256] {{
    %t_value = index.div %a, %cKU : index
    %t = index.assume %t_value [lt(%t_value, %tokens)] : index
    %s_value = index.rem %a, %cKU : index
    %s = index.assume %s_value [range(%s_value, 0, {KU - 1})] : index
    %id = view.load %ids_view[%t, %s] : view<[%tokens]x{KU}xi32> -> i32
    %low = scalar.maxsi %id, %c0_i32 : i32
    %clamped = scalar.minsi %low, %cEm1 : i32
    %bounded = scalar.assume %clamped [range(%clamped, 0, {E - 1})] : i32
    %e = index.cast %bounded : i32 to index
    %live32p = view.load %token_info[%t, 2] : view<[%tokens]x4xi32> -> i32
    %livep = scalar.cmpi ne, %live32p, %c0_i32 : i32
    %ba = index.assume %a [lt(%a, %assignments)] : index
    scf.if %livep {{
      %row32 = view.atomic.rmw<addi> %c1_i32, %cursor_view[%e] {{ordering = relaxed, scope = workgroup}} : i32, view<256xi32> -> i32
      %row_bound = scalar.assume %row32 [range(%row32, 0, 2097151)] : i32
      %row_value = index.cast %row_bound : i32 to index
      %row = index.assume %row_value [lt(%row_value, %rowmap_rows)] : index
      %a32 = index.cast %a : index to i32
      view.store %a32, %rowmap_view[%row] : i32, view<[%rowmap_rows]xi32>
      view.store %row32, %pos_view[%ba] : i32, view<[%assignments]xi32>
    }} else {{
      view.store %minus_one, %pos_view[%ba] : i32, view<[%assignments]xi32>
    }}
  }}
  kernel.return
}}
"""


def grouped_gemm(fmt, mode, N, K, E, KU, gemm_fn):
    """64x64-tile WMMA GEMM over the bucketed rows of one expert per row tile (rows = max_tiles*64).
    mode 'swiglu' (gate/up): A rows gathered from x16[token = rowmap/KU]; 'store' (down): A = sorted rows."""
    import gen_gemm
    rb = N and (K // SB[fmt] * FMT[fmt])
    eb = rb * N
    text = gemm_fn(fmt, mode)
    base = f"g_gemm_{fmt}_{mode}"
    name = f"g_moe_g{'gu' if mode == 'swiglu' else 'dn'}_{fmt}_k{K}"
    def rep(a, b, cnt=1):
        nonlocal text
        assert text.count(a) == cnt, (name, a, text.count(a))
        text = text.replace(a, b)
    rep(f'export("{base}") @{base}(', f'export("{name}") @{name}(')
    rep("launch(%rows_arg: index, %k_arg: index, %n_arg: index, %x: buffer, %scales: buffer,",
        "launch(%rows_arg: index, %k_arg: index, %n_arg: index, %xrows_arg: index, %tile_info: buffer, %rowmap: buffer, %x: buffer, %scales: buffer,")
    # rows = max_tiles*64 <= 8192 tokens * KU + 64 * E; n is this kernel's N. Tight bounds keep the output tile's dynamic-stride
    # extent (rows * n * 4 bytes) inside what the AMDGPU fragment store accepts (rows 2^20 x n 2^18 is rejected, gfx11 and gfx12).
    rep("[range(%rows_arg, 64, 8192), multiple_of(%rows_arg, 64)]", f"[range(%rows_arg, 64, {8192 * KU + 64 * E}), multiple_of(%rows_arg, 64)]")
    rep("[range(%n_arg, 64, 262144), multiple_of(%n_arg, 64)]", f"[range(%n_arg, 64, {N}), multiple_of(%n_arg, 64)]")
    rep("  %origin = index.constant 0 : offset\n", "  %origin = index.constant 0 : offset\n  %xrows = index.assume %xrows_arg [range(%xrows_arg, 1, 1048576)] : index\n")
    # skip tiles of no expert; expert weight offset
    rep("  %workitem_id = kernel.workitem.id<x> : index\n", f"""  %ti_view = buffer.view %tile_info[%origin] : buffer -> view<[%row_tile_count]x2xi32>
  %tile_expert = view.load %ti_view[%row_group, 0] : view<[%row_tile_count]x2xi32> -> i32
  %c0_i32g = scalar.constant 0 : i32
  %tile_used = scalar.cmpi sge, %tile_expert, %c0_i32g : i32
  scf.if %tile_used {{
  %cEm1g = scalar.constant {E - 1} : i32
  %e_lo = scalar.minsi %tile_expert, %cEm1g : i32
  %e_b = scalar.assume %e_lo [range(%e_lo, 0, {E - 1})] : i32
  %expert = index.cast %e_b : i32 to index
  %expert_bytes = index.constant {eb} : offset
  %expert_off = index.scale %expert, %expert_bytes : index, offset -> offset
  %workitem_id = kernel.workitem.id<x> : index
""")
    rep("  %w_row_origin = index.scale %bounded_w_out_row, %row_bytes : index, offset -> offset\n",
        "  %w_row_origin0 = index.scale %bounded_w_out_row, %row_bytes : index, offset -> offset\n  %w_row_origin = index.add %w_row_origin0, %expert_off : offset\n")
    rep("  %x_view = buffer.view %x[%origin] : buffer -> view<[%rows]x[%k]xf16>", "  %x_view = buffer.view %x[%origin] : buffer -> view<[%xrows]x[%k]xf16>")
    rep("view<[%rows]x[%k]xf16>", "view<[%xrows]x[%k]xf16>", text.count("view<[%rows]x[%k]xf16>"))
    rep("view<[%rows]x2xf32>", "view<[%xrows]x2xf32>", text.count("view<[%rows]x2xf32>"))
    if mode == "swiglu":
        gather = lambda pre, row_expr, out: f"""  %{pre}_prow_v = {row_expr}
  %{pre}_rm_rows = index.mul %row_tile_count, %c64 : index
  %{pre}_prow = index.assume %{pre}_prow_v [lt(%{pre}_prow_v, %{pre}_rm_rows)] : index
  %{pre}_rm_view = buffer.view %rowmap[%origin] : buffer -> view<[%{pre}_rm_rows]xi32>
  %{pre}_a32 = view.load %{pre}_rm_view[%{pre}_prow] : view<[%{pre}_rm_rows]xi32> -> i32
  %{pre}_valid = scalar.cmpi sge, %{pre}_a32, %c0_i32g : i32
  %{pre}_ac = scalar.maxsi %{pre}_a32, %c0_i32g : i32
  %{pre}_ab = scalar.assume %{pre}_ac [range(%{pre}_ac, 0, 1048575)] : i32
  %{pre}_a = index.cast %{pre}_ab : i32 to index
  %{pre}_cKU = index.constant {KU} : index
  %{pre}_t = index.div %{pre}_a, %{pre}_cKU : index
  %{out} = index.assume %{pre}_t [lt(%{pre}_t, %xrows)] : index
"""
        rep("  %x_row = index.add %row_origin, %stage_row : index\n  %bounded_x_row = index.assume %x_row [lt(%x_row, %rows)] : index\n",
            gather("gx", "index.add %row_origin, %stage_row : index", "bounded_x_row"))
        rep("    %x_half = vector.load %x_view[%bounded_x_row, %x_col] : view<[%xrows]x[%k]xf16> -> vector<16xf16>\n",
            """    %x_half_raw = vector.load %x_view[%bounded_x_row, %x_col] : view<[%xrows]x[%k]xf16> -> vector<16xf16>
    %zero16h = vector.constant 0.0 : vector<16xf16>
    %x_half = scf.select %gx_valid, %x_half_raw, %zero16h : vector<16xf16>
""")
        rep("    %inv_row = index.assume %inv_row_value [lt(%inv_row_value, %rows)] : index\n",
            gather("gi", "index.add %inv_row_value, %c0 : index", "inv_row"))
    else:
        rep("  %bounded_x_row = index.assume %x_row [lt(%x_row, %rows)] : index", "  %bounded_x_row = index.assume %x_row [lt(%x_row, %xrows)] : index")
        rep("    %inv_row = index.assume %inv_row_value [lt(%inv_row_value, %rows)] : index", "    %inv_row = index.assume %inv_row_value [lt(%inv_row_value, %xrows)] : index")
    rep("  kernel.return\n}\n", "  }\n  kernel.return\n}\n")
    import gen_gemm_rdna4
    return text + "\n" + gen_gemm_rdna4.twin(text) if gen_gemm_rdna4.enabled() else text   # the gfx12 twin K_rdna4 beside K


def combine_sorted(H, KU):
    return f"""
kernel.def export("g_moe_combine_sorted") @g_moe_combine_sorted(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %cb = index.constant {(H + 255) // 256} : index
  kernel.launch.config workgroups(%tokens, %cb, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %rows_arg: index, %ys: buffer, %pos: buffer, %wts: buffer, %y: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 1048576)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %thread_id = kernel.workitem.id<x> : index
  %blk_id = kernel.workgroup.id<y> : index
  %c256 = index.constant 256 : index
  %boff = index.mul %blk_id, %c256 : index
  %c_raw = index.add %thread_id, %boff : index
  %cH = index.constant {H} : index
  %in = index.cmp ult, %c_raw, %cH : index
  scf.if %in {{
    %ch = index.assume %c_raw [range(%c_raw, 0, {H - 1})] : index
    %zero = index.constant 0 : offset
    %zero_f = scalar.constant 0.0 : f32
    %c0_i32 = scalar.constant 0 : i32
    %ys_view = buffer.view %ys[%zero] : buffer -> view<[%rows]x{H}xf32>
    %pos_view = buffer.view %pos[%zero] : buffer -> view<[%tokens]x{KU}xi32>
    %w_view = buffer.view %wts[%zero] : buffer -> view<[%tokens]x{KU}xf32>
    %y_view = buffer.view %y[%zero] : buffer -> view<[%tokens]x{H}xf32>
""" + "\n".join(f"""    %w{k} = view.load %w_view[%token, {k}] : view<[%tokens]x{KU}xf32> -> f32
    %p{k} = view.load %pos_view[%token, {k}] : view<[%tokens]x{KU}xi32> -> i32
    %ok{k} = scalar.cmpi sge, %p{k}, %c0_i32 : i32
    %v{k} = scf.if %ok{k} -> (f32) {{
      %pb{k} = scalar.assume %p{k} [range(%p{k}, 0, 1048575)] : i32
      %pi{k} = index.cast %pb{k} : i32 to index
      %pr{k} = index.assume %pi{k} [lt(%pi{k}, %rows)] : index
      %vv{k} = view.load %ys_view[%pr{k}, %ch] : view<[%rows]x{H}xf32> -> f32
      scf.yield %vv{k} : f32
    }} else {{
      scf.yield %zero_f : f32
    }}
    %a{k} = scalar.fmaf %w{k}, %v{k}, {"%zero_f" if k == 0 else f"%a{k - 1}"} : f32""" for k in range(KU)) + f"""
    view.store %a{KU - 1}, %y_view[%token, %ch] : f32, view<[%tokens]x{H}xf32>
  }}
  kernel.return
}}
"""
