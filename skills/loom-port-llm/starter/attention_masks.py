"""Opt-in transformations of the ordinary paged decode split kernel."""
import os


def full_blocks(text, name, unroll, twin=False):
    """Bypass per-key predicates on wholly visible blocks; preserve edge arithmetic.

    ceil(lo/U)*U <= j < floor(hi/U)*U is equivalent to all U keys being live
    for an aligned block origin j. Clamp reversed full-block bounds to empty.
    The caller restricts this form to the measured ordinary hd64 decode body.
    """
    from flash2_rdna4 import twin_env
    env = twin_env(os.environ) if twin else os.environ
    if env.get("ATTN_FULL_BLOCKS", "0") != "1":
        return text
    a = text.index(f'kernel.def export("{name}")')
    end = "  kernel.return\n}"
    b = text.index(end, a) + len(end)
    kernel = text[a:b]
    masks = []
    for i in range(unroll):
        legacy = (f"      %okl{i} = index.cmp uge, %key{i}_raw, %lo_abs : index\n"
                  f"      %okh{i} = index.cmp ult, %key{i}_raw, %hi_abs : index\n"
                  f"      %ok{i} = scalar.andi %okl{i}, %okh{i} : i1\n")
        assert kernel.count(legacy) == 1, "full_blocks: per-key predicate anchor"
        kernel = kernel.replace(legacy, "", 1)
        masks.append(f"      %ok{i}_width = index.sub %hi_abs, %lo_abs : index\n"
                     f"      %ok{i}_delta = index.sub %key{i}_raw, %lo_abs : index\n"
                     f"      %ok{i} = index.cmp ult, %ok{i}_delta, %ok{i}_width : index\n")
    loop = "    %m_final, %l_final, %acc_final = scf.for %j"
    assert kernel.count(loop) == 1, "full_blocks: ordinary single-chain loop"
    bounds = f"""    %full_less = index.constant {unroll - 1} : index
    %full_pad = index.add %lo_abs, %full_less : index
    %full_lq = index.div %full_pad, %cU : index
    %full_lo = index.mul %full_lq, %cU : index
    %full_hq = index.div %hi_abs, %cU : index
    %full_hi0 = index.mul %full_hq, %cU : index
    %full_hi = index.max %full_hi0, %full_lo : index
    %full_width = index.sub %full_hi, %full_lo : index
"""
    kernel = kernel.replace(loop, bounds + loop)
    page = "      %kb_off = func.call @g_kv_offset"
    assert kernel.count(page) == 1, "full_blocks: ordinary paged body"
    kernel = kernel.replace(page, """      %full_delta = index.sub %j, %full_lo : index
      %full = index.cmp ult, %full_delta, %full_width : index
""" + page, 1)
    scores = [f"      %s{i} = scf.select %ok{i}, %sr{i}, %neg_big : f32\n" for i in range(unroll)]
    assert kernel.count("".join(scores)) == 1, "full_blocks: consecutive score anchors"
    outs = ", ".join(f"%s{i}" for i in range(unroll))
    types = ", ".join(["f32"] * unroll)
    branch = f"      {outs} = scf.if %full -> ({types}) {{\n"
    branch += "        scf.yield " + ", ".join(f"%sr{i}" for i in range(unroll)) + f" : {types}\n"
    branch += "      } else {\n" + "".join("  " + line for line in masks)
    branch += "".join("  " + line.replace(f"%s{i} =", f"%edge_s{i} =") for i, line in enumerate(scores))
    branch += "        scf.yield " + ", ".join(f"%edge_s{i}" for i in range(unroll)) + f" : {types}\n      }}\n"
    kernel = kernel.replace("".join(scores), branch)
    return text[:a] + kernel + text[b:]
