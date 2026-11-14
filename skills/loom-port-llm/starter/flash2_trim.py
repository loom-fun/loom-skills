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
