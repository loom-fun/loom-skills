"""gfx12 fused front of the gated delta-net chain: `g_gdn_front_rdna4` = g_gdn_conv + g_gdn_conv_commit + g_gdn_qk_norm in one launch.

A decode / verify / replay step of a hybrid runs the chain per gated delta-net layer: conv1d + SiLU over the span's inputs and the
stored history (one launch), the history commit (one), the q / k inverse L2 norms (one), the recurrence, the gated norm. The first three
are tiny (10240 channels x a handful of tokens) but cost 1.6 ms of a 35 ms drafted step on the R9700 (doubling: conv_commit alone 1.08 ms,
each launch is a dependent chain of metadata, origins and history loads): 48 layers x 2 passes (verify and replay) x 3 launches.

One wave (32 lanes) owns 128 consecutive channels of one span (blocks 0 .. 2 HK - 1 are the q and k heads, so a block is exactly one head
of the L2 norm), 4 channels per lane (16-byte loads and stores). It walks the span's tokens in tiles of TILE steps: each lane keeps its
channels' K - 1 history values in registers, so the conv of every step reads the inputs once and the history commit needs no other launch
(and no ordering against the readers of the old history: a lane reads and writes only its own channels, so the in-place reader == writer
slot is safe). Every load of a tile is issued before any store and none sits under a branch (a dead step of the last tile reads a clamped
row and is masked). The q / k blocks reduce the tile's squared SiLU outputs over the wave (subgroup reductions, no barrier) and store the
inverse norms. Same arithmetic per output as the three kernels (the conv taps accumulate in the same order); only the L2 sums are
ordered differently. The span's tokens are all live (g_token_info marks every token of a span), so the kernel needs no info buffer.
"""
import os

BLOCK = 128    # channels per wave (= DK: one q or k head)
LANE_CH = 4    # channels per lane


def tile_steps():
    """Steps per tile (env GDN_FRONT_TILE, default 4: a drafted step verifies at most 4 rows at depth 3)."""
    return int(os.environ.get("GDN_FRONT_TILE", "4"))


def front_kernel(G, NG, LB, eps, guard_open="", guard_close="", target=True, tile=None):
    HK, HV, DK, DV, K, C = G["HK"], G["HV"], G["DK"], G["DV"], G["K"], G["C"]
    assert DK == BLOCK and C % BLOCK == 0, "g_gdn_front_rdna4 is built for 128-wide heads"
    S = tile or tile_steps()
    SB = G["state_bytes"]
    NB = C // BLOCK
    H = K - 1
    V4 = "vector<4xf32>"
    L = []
    P = L.append
    sym = "g_gdn_front_rdna4"
    P(f"""
kernel.def{" target(@rdna4)" if target else ""} export("{sym}") @{sym}(%spans: index) {{
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  %nblocks = index.constant {NB} : index
  kernel.launch.config workgroups(%spans, %nblocks, %c1) workgroup_size(%c32, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %layer_arg: index, %metadata: buffer, %origins: buffer, %arena: buffer, %qkv: buffer, %weight: buffer, %out: buffer, %inverse: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {NG - 1})] : index
  %span_id = kernel.workgroup.id<x> : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
  %blk_id = kernel.workgroup.id<y> : index
  %blk = index.assume %blk_id [range(%blk_id, 0, {NB - 1})] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1i = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %zero_f = scalar.constant 0.0 : f32
  %eps = scalar.constant {eps!r} : f32
  %zero4 = vector.splat %zero_f : {V4}
  %active, %length, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
  scf.if %active {{
    %layer_stride = index.constant {LB} : offset
    %layer_bytes = index.scale %layer, %layer_stride : index, offset -> offset
    %reader, %writer, %fresh = func.call @g_gdn_slots(%origins, %row, %start32, %layer_bytes) : (buffer, index, i32, offset) -> (offset, offset, i1)
    %history_skip = index.constant {SB} : offset
    %old_at = index.add %reader, %history_skip : offset
    %new_at = index.add %writer, %history_skip : offset
    %old_view = buffer.view %arena[%old_at] : buffer -> view<{H}x{C}xf32>
    %new_view = buffer.view %arena[%new_at] : buffer -> view<{H}x{C}xf32>
    %qkv_view = buffer.view %qkv[%zero] : buffer -> view<[%tokens]x{C}xf32>
    %out_view = buffer.view %out[%zero] : buffer -> view<[%tokens]x{C}xf32>
    %w_view = buffer.view %weight[%zero] : buffer -> view<{C}x{K}xf32>
    %inv_view = buffer.view %inverse[%zero] : buffer -> view<[%tokens]x{2 * HK}xf32>
    %cblockb = index.constant {BLOCK} : index
    %cblk = index.mul %blk, %cblockb : index
    %lane4 = index.mul %lane, %c4 : index
    %ch0_value = index.add %cblk, %lane4 : index
    %ch0 = index.assume %ch0_value [range(%ch0_value, 0, {C - LANE_CH})] : index""")
    for c in range(LANE_CH):
        P(f"""    %cc{c} = index.constant {c} : index
    %cch{c}_value = index.add %ch0, %cc{c} : index
    %cch{c} = index.assume %cch{c}_value [range(%cch{c}_value, 0, {C - 1})] : index
    %wv{c} = vector.load %w_view[%cch{c}, %c0] : view<{C}x{K}xf32> -> vector<{K}xf32>""")
        for j in range(K):
            P(f"    %w{c}_{j} = vector.extract %wv{c}[{j}] : vector<{K}xf32> -> f32")
    for j in range(H):   # history of this lane's channels; a fresh span starts from zeros (the load is unconditional: the reader slot is always inside the arena)
        P(f"""    %hl{j} = vector.load %old_view[{j}, %ch0] : view<{H}x{C}xf32> -> {V4}
    %h{j} = scf.select %fresh, %zero4, %hl{j} : {V4}""")
    # q / k block bookkeeping: block b < HK is q head b (slot 2 b), HK <= b < 2 HK is k head b - HK (slot 2 (b - HK) + 1)
    P(f"""    %cqk = index.constant {2 * HK} : index
    %chk = index.constant {HK} : index
    %is_qk = index.cmp ult, %blk, %cqk : index
    %is_q = index.cmp ult, %blk, %chk : index
    %kh_k = index.sub %blk, %chk : index
    %kh_v = scf.select %is_q, %blk, %kh_k : index
    %slot_a = index.mul %kh_v, %c2 : index
    %slot_k = index.add %slot_a, %c1i : index
    %slot_v = scf.select %is_q, %slot_a, %slot_k : index
    %slot = index.assume %slot_v [range(%slot_v, 0, {2 * HK - 1})] : index
    %leader = index.cmp eq, %lane, %c0 : index
    %cS = index.constant {S} : index
    %tokm1 = index.sub %tokens, %c1i : index""")
    iters = ", ".join(f"%p{j} = %h{j} : {V4}" for j in range(H))
    fts = ", ".join([V4] * H)
    body = []
    B = body.append
    B("    %tile_left = index.sub %length, %base : index")
    B("    %n_live = index.min %tile_left, %cS : index")
    for s in range(S):
        B(f"""    %s{s}_k = index.constant {s} : index
    %s{s}_step = index.add %base, %s{s}_k : index
    %live{s} = index.cmp ult, %s{s}_step, %length : index
    %tok{s}_a = index.add %first, %s{s}_step : index
    %tok{s}_m = index.min %tok{s}_a, %tokm1 : index
    %tok{s} = index.assume %tok{s}_m [lt(%tok{s}_m, %tokens)] : index
    %xr{s} = vector.load %qkv_view[%tok{s}, %ch0] : view<[%tokens]x{C}xf32> -> {V4}
    %x{s} = scf.select %live{s}, %xr{s}, %zero4 : {V4}""")
    seq = [f"%p{j}" for j in range(H)] + [f"%x{s}" for s in range(S)]
    for s in range(S):
        for c in range(LANE_CH):
            for j in range(K):
                B(f"    %e{s}_{c}_{j} = vector.extract {seq[s + j]}[{c}] : {V4} -> f32")
                prev = "%zero_f" if j == 0 else f"%a{s}_{c}_{j - 1}"
                B(f"    %a{s}_{c}_{j} = scalar.fmaf %w{c}_{j}, %e{s}_{c}_{j}, {prev} : f32")
            B(f"""    %g{s}_{c} = scalar.logisticf<afn> %a{s}_{c}_{K - 1} : f32
    %y{s}_{c} = scalar.mulf %a{s}_{c}_{K - 1}, %g{s}_{c} : f32""")
        B(f"    %y{s} = vector.from_elements {', '.join(f'%y{s}_{c}' for c in range(LANE_CH))} : {V4}")
    for s in range(S):
        B(f"""    scf.if %live{s} {{
      vector.store %y{s}, %out_view[%tok{s}, %ch0] : {V4}, view<[%tokens]x{C}xf32>
    }}""")
    # L2 norms of the q / k head of this block: this lane's 4 squares, then one wave reduction per step
    for s in range(S):
        B(f"""    %q{s}_d = vector.dotf<reassoc|contract> %y{s}, %y{s}, %zero_f : {V4}, {V4}, f32
    %q{s} = scf.select %live{s}, %q{s}_d, %zero_f : f32""")
    B("    scf.if %is_qk {")
    for s in range(S):
        B(f"""      %t{s} = kernel.subgroup.reduce<addf> %q{s} : f32
      %t{s}_e = scalar.addf %t{s}, %eps : f32
      %iv{s} = scalar.rsqrtf %t{s}_e : f32
      %st{s} = scalar.andi %live{s}, %leader : i1
      scf.if %st{s} {{
        view.store %iv{s}, %inv_view[%tok{s}, %slot] : f32, view<[%tokens]x{2 * HK}xf32>
      }}""")
    B("    }")
    # the history after this tile: the K - 1 values before the first step not yet consumed, seq[n_live + j]
    nh = []
    for j in range(H):
        cur = seq[min(S + j, len(seq) - 1)]
        for n in range(S - 1, 0, -1):
            B(f"""    %hn{j}_{n}_k = index.constant {n} : index
    %hn{j}_{n}_c = index.cmp eq, %n_live, %hn{j}_{n}_k : index
    %hn{j}_{n} = scf.select %hn{j}_{n}_c, {seq[n + j]}, {cur} : {V4}""")
            cur = f"%hn{j}_{n}"
        nh.append(cur)
    P(f"    %f_{', %f_'.join(str(j) for j in range(H))} = scf.for %base = [%c0 to %length step %cS]({iters}) -> ({fts}) {{")
    P("\n".join(body))
    P(f"    scf.yield {', '.join(nh)} : {fts}")
    P("    }")
    stores = "\n".join(f"      vector.store %f_{j}, %new_view[{j}, %ch0] : {V4}, view<{H}x{C}xf32>" for j in range(H))
    P(f"{guard_open}{stores}{guard_close}")
    P("  }\n  kernel.return\n}\n")
    return "\n".join(L)
