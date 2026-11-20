"""Two listed 16-token query tiles per workgroup for the gfx11 flash2s whole-tile kernel (gen_flash2s.py, FLASH2_MT256=2; head_dim 256,
16-key tiles, no sliding window).

The plain kernel stages every K/V tile once per (16-token query tile, KV head): at a 32k context the K/V stream of a 1024-token chunk is
re-read by 64 workgroups per KV head, and kbench ablations put the global K/V loads at ~60 % of the 16-key kernel's time. Here one
workgroup holds both query tiles 2x and 2x + 1 (24 waves: sub-tile x head x 128-dim half), so each staged K/V tile feeds 32 query tokens:
half the global K/V traffic and half the staging work per token, at the same 208-VGPR wave shape (one 24-wave workgroup per WGP holds
as many waves as two 12-wave ones).

Both sub-tiles' keys come from one paged-KV row: when their rows differ (a batched prefill whose tiles 2x and 2x + 1 belong to different
sequences) the workgroup makes two passes over the key loop, pass p staging row p's keys; the other sub-tile's waves run along (barriers
are workgroup-wide) and discard their result. An unlisted second tile (odd count) mirrors the first and stores nothing. In one pass the
key range is the later tile's: the earlier tile's waves mask those keys in S (exact zero weights on finite, written V), so results are
bit-identical to the plain kernel's except that a diagonal block of the earlier tile takes the plain (hot) P V path.

Applied by gen_flash2s.py before flash2_peel.py; gen_flash2s.py sets WAVES (24) and the staging segments (32 of 8 dims) itself.
"""
import re


def two_tiles(text, waves_per_tile):
    W = waves_per_tile
    L = text.split("\n")

    def find(pred, start=0):
        return next(i for i in range(start, len(L)) if pred(L[i]))

    # grid: ceil(tiles / 2) workgroups along x
    i = find(lambda s: s.strip().startswith("kernel.launch.config workgroups(%tiles,"))
    L[i:i + 1] = ["  %mt_one = index.constant 1 : index", "  %mt_two = index.constant 2 : index",
                  "  %tiles_p1 = index.add %tiles, %mt_one : index", "  %pairs = index.div %tiles_p1, %mt_two : index",
                  L[i].replace("workgroups(%tiles,", "workgroups(%pairs,")]
    # the listed-tile header: both tiles' entries, rows, positions
    a = find(lambda s: s.strip().startswith("%tile_id = kernel.workgroup.id<x>"))
    b = find(lambda s: s.strip() == "scf.if %listed {", a)
    kv = [s for s in L[a:b] if s.strip().startswith(("%kv_id", "%kv ", "%kv=", "%cng", "%kv_value", "%hgroup"))]
    assert any("%kv = " in s for s in kv), kv
    wave_lines = [L[j] for j in range(b, b + 8) if L[j].strip().startswith(("%wave_id", "%wave ="))]
    assert len(wave_lines) == 2, wave_lines
    waves = int(re.search(r"range\(%wave_id, 0, (\d+)\)", wave_lines[1]).group(1)) + 1
    assert waves == 2 * W, (waves, W)
    hdr = ["  %pair_id = kernel.workgroup.id<x> : index", *kv, *wave_lines,
           f"  %cwpt = index.constant {W} : index",
           "  %sub = index.div %wave, %cwpt : index",
           "  %wave_l = index.rem %wave, %cwpt : index",
           "  %is_b = index.cmp ne, %sub, %c0 : index",
           "  %tiles_view = buffer.view %tiles[%origin] : buffer -> view<[%tokens]xi32>",
           "  %listed_count32 = view.load %tiles_view[%c0] : view<[%tokens]xi32> -> i32",
           "  %listed_count = index.cast %listed_count32 : i32 to index",
           "  %tile_a = index.mul %pair_id, %c2 : index",
           "  %tile_b = index.add %tile_a, %c1 : index",
           "  %listed = index.cmp ult, %tile_a, %listed_count : index",
           "  %listed_b = index.cmp ult, %tile_b, %listed_count : index",
           "  %slot_a_v = index.add %tile_a, %c1 : index",
           "  %slot_a = index.assume %slot_a_v [lt(%slot_a_v, %tokens)] : index",
           "  %slot_b_v = index.add %tile_b, %c1 : index",
           "  %slot_last = index.sub %tokens, %c1 : index",
           "  %slot_b_c = index.min %slot_b_v, %slot_last : index",
           "  %slot_b = index.assume %slot_b_c [lt(%slot_b_c, %tokens)] : index",
           "  %ea32 = view.load %tiles_view[%slot_a] : view<[%tokens]xi32> -> i32",
           "  %eb32r = view.load %tiles_view[%slot_b] : view<[%tokens]xi32> -> i32",
           "  // an unlisted second tile mirrors the first (its entry is not initialized) and stores nothing",
           "  %eb32 = scf.select %listed_b, %eb32r, %ea32 : i32",
           "  %ea = index.cast %ea32 : i32 to index",
           "  %entry_a = index.assume %ea [range(%ea, 0, 8191), lt(%ea, %tokens)] : index",
           "  %eb = index.cast %eb32 : i32 to index",
           "  %entry_b = index.assume %eb [range(%eb, 0, 8191), lt(%eb, %tokens)] : index",
           "  %info_view = buffer.view %info[%origin] : buffer -> view<[%tokens]x4xi32>",
           "  %row_a32 = view.load %info_view[%entry_a, %c0] : view<[%tokens]x4xi32> -> i32",
           "  %pos_a32 = view.load %info_view[%entry_a, %c1] : view<[%tokens]x4xi32> -> i32",
           "  %row_b32 = view.load %info_view[%entry_b, %c0] : view<[%tokens]x4xi32> -> i32",
           "  %pos_b32 = view.load %info_view[%entry_b, %c1] : view<[%tokens]x4xi32> -> i32",
           "  scf.if %listed {",
           "  %ft = scf.select %is_b, %entry_b, %entry_a : index",
           "  %first_token = index.assume %ft [range(%ft, 0, 8191), lt(%ft, %tokens)] : index",
           "  %pos0_32 = scf.select %is_b, %pos_b32, %pos_a32 : i32",
           "  %true_mt = scalar.constant true : i1",
           "  %listed_me = scf.select %is_b, %listed_b, %true_mt : i1",
           "  %same_row = scalar.cmpi eq, %row_a32, %row_b32 : i32",
           "  %npass = scf.select %same_row, %c1, %c2 : index",
           "  %pa_bound = scalar.assume %pos_a32 [range(%pos_a32, 0, 262128)] : i32",
           "  %pa = index.cast %pa_bound : i32 to index",
           "  %pb_bound = scalar.assume %pos_b32 [range(%pos_b32, 0, 262128)] : i32",
           "  %pb = index.cast %pb_bound : i32 to index",
           "  %pos_last_a = index.add %pa, %c15 : index",
           "  %pos_last_b = index.add %pb, %c15 : index",
           "  %pos_last_both = index.max %pos_last_a, %pos_last_b : index"]
    L[a:b + 1] = hdr
    # per-wave head / dim half from the wave's index within its sub-tile; drop the old wave lines and the old row / pos_last
    body_start = a + len(hdr)
    for j in range(body_start, body_start + 40):
        s = L[j].strip()
        if s.startswith(("%wave_id = ", "%wave = ")):
            L[j] = None
        elif s.startswith("%head_in_group = index.div %wave,"):
            L[j] = L[j].replace("index.div %wave,", "index.div %wave_l,")
        elif s.startswith("%dim_half = index.rem %wave,"):
            L[j] = L[j].replace("index.rem %wave,", "index.rem %wave_l,")
        elif s.startswith(("%row_bound = ", "%row = ", "%pos_last = ")):
            L[j] = None
    L = [s for s in L if s is not None]
    # the key range moves into the pass loop
    k0 = find(lambda s: s.strip().startswith("%key_span = index.sub %pos_last, %key_low"))
    k1 = find(lambda s: s.strip().startswith("%key_tiles = index.div"), k0)
    krange = L[k0:k1 + 1]
    del L[k0:k1 + 1]
    # partner wave: wave ^ 1 (head_in_group is per sub-tile now)
    p = find(lambda s: s.strip().startswith("%partner_id_value = index.madd %head_in_group, %c2, %partner_add"))
    L[p:p + 1] = ["  %partner_base = index.sub %wave, %dim_half : index",
                  "  %partner_id_value = index.add %partner_base, %partner_add : index"]
    # pass loop around the key loop and the output
    h = find(lambda s: re.match(r"^ *%m_final, %l_final, .* = scf\.for %kt = ", s) is not None)
    pre = ["  scf.for %pass = [%c0 to %npass step %c1] {",
           "  %pass_b = index.cmp ne, %pass, %c0 : index",
           "  %row32p = scf.select %pass_b, %row_b32, %row_a32 : i32",
           "  %row_bound = scalar.assume %row32p [range(%row32p, 0, 2147483646)] : i32",
           "  %row = index.cast %row_bound : i32 to index",
           "  %pl_one = scf.select %pass_b, %pos_last_b, %pos_last_a : index",
           "  %pos_last = scf.select %same_row, %pos_last_both, %pl_one : index",
           *krange,
           "  %pass_me = index.cmp eq, %pass, %sub : index",
           "  %active = scalar.ori %same_row, %pass_me : i1",
           "  %store_ok = scalar.andi %active, %listed_me : i1"]
    L[h:h] = pre
    o = find(lambda s: s.strip().startswith("// O^T / l, gate, store through"), h)
    L[o:o] = ["  scf.if %store_ok {"]
    r = max(i for i, s in enumerate(L) if s.strip() == "kernel.return")
    assert L[r - 1].strip() == "}", L[r - 1]
    L[r - 1:r - 1] = ["  }", "  }"]
    return "\n".join(L)
