"""gfx12 fused front of the gated delta-net chain: `g_gdn_front_rdna4` = g_gdn_conv + g_gdn_conv_commit + g_gdn_qk_norm in one launch.

A decode / verify / replay step of a hybrid runs the chain per gated delta-net layer: conv1d + SiLU over the span's inputs and the
stored history (one launch), the history commit (one), the q / k inverse L2 norms (one), the recurrence, the gated norm. The first three
are tiny (10240 channels x a handful of tokens) and each launch costs ~3 us on gfx1201 (R9700, doubling: conv +0.32 ms, qk_norm +0.28 ms
of a 40 ms drafted step over 96 launches each): 48 layers x 2 passes (verify and replay) x 3 launches.

One workgroup of 128 threads owns 128 consecutive channels of one span (blocks 0 .. 2 HK - 1 are the q and k heads, so a block is exactly one
head of the L2 norm) and walks the span's tokens in tiles of S = 8: each thread keeps its channel's K - 1 history values in registers, so
the conv of every step reads the inputs once and the history commit needs no other launch (and no ordering against the readers of the old
history: a thread reads and writes only its own channel, so the in-place reader == writer slot is safe). The q / k blocks reduce the
tile's squared SiLU outputs over the workgroup (one vector<8xf32> workgroup reduction per tile) and store the inverse norms.
Same arithmetic per output as the three kernels (the conv taps accumulate in the same order); only the L2 sums are ordered differently.
The span's tokens are all live (g_token_info marks every token of a span), so the kernel needs no info buffer.
"""

S = 8          # steps per tile
BLOCK = 128    # channels per workgroup (= DK: one q or k head)


def front_kernel(G, NG, LB, eps, guard_open="", guard_close="", target=True):
    HK, HV, DK, DV, K, C = G["HK"], G["HV"], G["DK"], G["DV"], G["K"], G["C"]
    assert DK == BLOCK and C % BLOCK == 0, "g_gdn_front_rdna4 is built for 128-wide heads"
    SB = G["state_bytes"]
    NB = C // BLOCK
    H = K - 1
    L = []
    P = L.append
    sym = "g_gdn_front_rdna4"
    P(f"""
kernel.def{" target(@rdna4)" if target else ""} export("{sym}") @{sym}(%spans: index) {{
  %c1 = index.constant 1 : index
  %cblock = index.constant {BLOCK} : index
  %nblocks = index.constant {NB} : index
  kernel.launch.config workgroups(%spans, %nblocks, %c1) workgroup_size(%cblock, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %layer_arg: index, %metadata: buffer, %origins: buffer, %arena: buffer, %qkv: buffer, %weight: buffer, %out: buffer, %inverse: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {NG - 1})] : index
  %span_id = kernel.workgroup.id<x> : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
  %blk_id = kernel.workgroup.id<y> : index
  %blk = index.assume %blk_id [range(%blk_id, 0, {NB - 1})] : index
  %thread_id = kernel.workitem.id<x> : index
  %thread = index.assume %thread_id [range(%thread_id, 0, {BLOCK - 1})] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1i = index.constant 1 : index
  %c2 = index.constant 2 : index
  %zero_f = scalar.constant 0.0 : f32
  %eps = scalar.constant {eps!r} : f32
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
    %ch_value = index.add %cblk, %thread : index
    %ch = index.assume %ch_value [range(%ch_value, 0, {C - 1})] : index""")
    for j in range(K):
        P(f"    %w{j} = view.load %w_view[%ch, {j}] : view<{C}x{K}xf32> -> f32")
    for j in range(H):
        P(f"""    %h{j} = scf.if %fresh -> (f32) {{
      scf.yield %zero_f : f32
    }} else {{
      %hv{j} = view.load %old_view[{j}, %ch] : view<{H}x{C}xf32> -> f32
      scf.yield %hv{j} : f32
    }}""")
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
    %leader = index.cmp eq, %thread, %c0 : index
    %cS = index.constant {S} : index
    %tokm1 = index.sub %tokens, %c1i : index""")
    iters = ", ".join(f"%p{j} = %h{j} : f32" for j in range(H))
    fts = ", ".join(["f32"] * H)
    body = []
    B = body.append
    B(f"    %tile_left = index.sub %length, %base : index")
    B(f"    %n_live = index.min %tile_left, %cS : index")
    for s in range(S):
        B(f"""    %s{s}_k = index.constant {s} : index
    %s{s}_step = index.add %base, %s{s}_k : index
    %live{s} = index.cmp ult, %s{s}_step, %length : index
    %tok{s}_a = index.add %first, %s{s}_step : index
    %tok{s}_m = index.min %tok{s}_a, %tokm1 : index
    %tok{s} = index.assume %tok{s}_m [lt(%tok{s}_m, %tokens)] : index
    %x{s} = scf.if %live{s} -> (f32) {{
      %xl{s} = view.load %qkv_view[%tok{s}, %ch] : view<[%tokens]x{C}xf32> -> f32
      scf.yield %xl{s} : f32
    }} else {{
      scf.yield %zero_f : f32
    }}""")
    seq = [f"%p{j}" for j in range(H)] + [f"%x{s}" for s in range(S)]
    for s in range(S):
        for j in range(K):
            prev = "%zero_f" if j == 0 else f"%a{s}_{j - 1}"
            B(f"    %a{s}_{j} = scalar.fmaf %w{j}, {seq[s + j]}, {prev} : f32")
        B(f"""    %g{s} = scalar.logisticf<afn> %a{s}_{K - 1} : f32
    %y{s} = scalar.mulf %a{s}_{K - 1}, %g{s} : f32""")
    for s in range(S):
        B(f"""    scf.if %live{s} {{
      view.store %y{s}, %out_view[%tok{s}, %ch] : f32, view<[%tokens]x{C}xf32>
    }}""")
    # L2 norms of the q / k head of this block: one vector<S> workgroup reduction per tile
    sq = "\n".join(f"""    %q{s}_sq = scalar.mulf %y{s}, %y{s} : f32
    %q{s} = scf.select %live{s}, %q{s}_sq, %zero_f : f32""" for s in range(S))
    B(sq)
    B(f"    %sqv = vector.from_elements {', '.join(f'%q{s}' for s in range(S))} : vector<{S}xf32>")
    B(f"    scf.if %is_qk {{\n      %tot = kernel.workgroup.reduce<addf> %sqv : vector<{S}xf32>")
    for s in range(S):
        B(f"""      %t{s} = vector.extract %tot[{s}] : vector<{S}xf32> -> f32
      %t{s}_e = scalar.addf %t{s}, %eps : f32
      %iv{s} = scalar.rsqrtf %t{s}_e : f32
      scf.if %live{s} {{
        scf.if %leader {{
          view.store %iv{s}, %inv_view[%tok{s}, %slot] : f32, view<[%tokens]x{2 * HK}xf32>
        }}
      }}""")
    B("    }")
    # the history after this tile: the K - 1 values before the first step not yet consumed, seq[n_live + j]
    nh = []
    for j in range(H):
        cur = seq[S + j]
        for n in range(S - 1, 0, -1):
            B(f"""    %hn{j}_{n}_k = index.constant {n} : index
    %hn{j}_{n}_c = index.cmp eq, %n_live, %hn{j}_{n}_k : index
    %hn{j}_{n} = scf.select %hn{j}_{n}_c, {seq[n + j]}, {cur} : f32""")
            cur = f"%hn{j}_{n}"
        nh.append(cur)
    P(f"    %f_{', %f_'.join(str(j) for j in range(H))} = scf.for %base = [%c0 to %length step %cS]({iters}) -> ({fts}) {{")
    P("\n".join(body))
    P(f"    scf.yield {', '.join(nh)} : {fts}")
    P("    }")
    stores = "\n".join(f"      view.store %f_{j}, %new_view[{j}, %ch] : f32, view<{H}x{C}xf32>" for j in range(H))
    P(f"{guard_open}{stores}{guard_close}")
    P("  }\n  kernel.return\n}\n")
    return "\n".join(L)
