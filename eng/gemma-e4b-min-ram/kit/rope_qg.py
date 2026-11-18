"""ROPE_QG (choice prefill.attn.rope_qg): the q|gate split folded into the norm + RoPE kernel (gated attention with a fused q|gate
projection, the qwen35 family).

g_split_qgate copied [query | gate] rows of the projection into q and gate; g_qk_rope_store then read q again, normalised and rotated it.
g_qk_rope_store_qg reads the query heads straight from the projection (qg[token, head, 0, dim]), stores the rotated query to q and copies
the gate half (qg[token, head, 1, dim]) to gate: the same values, one kernel and one pass over q less. The kernel text is the plain
kernel's with four edits, so every other behaviour (norm, partial RoPE, MRoPE, paged KV store) is the plain kernel's. The trunk layers'
launches pair a g_split_qgate with the following g_qk_rope_store; the MTP layer's launches (%mtp_layer) keep the plain pair.
"""
import re

SPLIT = "kernel.launch @g_split_qgate[%t](%t, %qg, %q, %gate) : [index](index, buffer, buffer, buffer)"


def _kernel(text):
    a = text.index('kernel.def export("g_qk_rope_store") @g_qk_rope_store(')
    b = text.index("kernel.return\n}\n", a) + len("kernel.return\n}\n")
    return a, b


def fuse(text):
    a, b = _kernel(text)
    k = text[a:b]
    k = k.replace('export("g_qk_rope_store") @g_qk_rope_store(', 'export("g_qk_rope_store_qg") @g_qk_rope_store_qg(', 1)
    assert k.count("%pool_arg: index, %q: buffer, %k: buffer") == 1
    k = k.replace("%pool_arg: index, %q: buffer, %k: buffer", "%pool_arg: index, %qg: buffer, %q: buffer, %gate: buffer, %k: buffer", 1)
    m = re.search(r"^  %q_view = buffer\.view %q\[%zero\] : buffer -> view<\[%tokens\]x(\d+)x(\d+)xf32>\n", k, flags=re.M)
    QH, D = m.group(1), m.group(2)
    k = k.replace(m.group(0), f"  %qg_view = buffer.view %qg[%zero] : buffer -> view<[%tokens]x{QH}x2x{D}xf32>\n"
                              f"  %gate_view = buffer.view %gate[%zero] : buffer -> view<[%tokens]x{QH}x{D}xf32>\n" + m.group(0), 1)
    k, n_ld = re.subn(r"view\.load %q_view\[%token, (%\w+), (%\w+)\] : view<\[%tokens\]x" + QH + "x" + D + r"xf32> -> f32",
                      rf"view.load %qg_view[%token, \1, 0, \2] : view<[%tokens]x{QH}x2x{D}xf32> -> f32", k)
    k, n_st = re.subn(r"^( *)view\.store (%r\d+), %q_view\[%token, (%\w+), (%\w+)\] : f32, view<\[%tokens\]x" + QH + "x" + D + r"xf32>\n",
                      lambda s: (s.group(0) + f"{s.group(1)}{s.group(2)}_gate = view.load %qg_view[%token, {s.group(3)}, 1, {s.group(4)}] : view<[%tokens]x{QH}x2x{D}xf32> -> f32\n"
                                 f"{s.group(1)}view.store {s.group(2)}_gate, %gate_view[%token, {s.group(3)}, {s.group(4)}] : f32, view<[%tokens]x{QH}x{D}xf32>\n"), k, flags=re.M)
    assert n_ld > 0 and n_ld == n_st, (n_ld, n_st)
    text = text[:b] + "\n" + k + text[b:]
    lines, out, held, n = text.split("\n"), [], None, 0
    for ln in lines:
        if ln.strip() == SPLIT:
            held = ln
            continue
        if held is not None and "kernel.launch @g_qk_rope_store[" in ln:
            if re.search(r"@g_qk_rope_store\[%t\]\(%t, %layer_\d+, %rope_\d+, %context, %pool, %q, ", ln):
                ln = ln.replace("@g_qk_rope_store[", "@g_qk_rope_store_qg[", 1).replace("%context, %pool, %q, ", "%context, %pool, %qg, %q, %gate, ", 1)
                ln = ln.replace("index, index, i32, index, index, buffer, buffer, buffer,", "index, index, i32, index, index, buffer, buffer, buffer, buffer, buffer,", 1)
                n += 1
                held = None
            else:
                out.append(held)
                held = None
        out.append(ln)
    assert held is None
    return "\n".join(out), n
