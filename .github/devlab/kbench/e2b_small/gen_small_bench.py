#!/usr/bin/env python3
"""The e2b_small kbench suite: Gemma-4-E2B's per-token prefill kernels (norm, row scale) at 2048 rows under different workgroup sizes.

usage: gen_small_bench.py PROGRAM.loom OUT.loom      PROGRAM: a generated gemma4 llm.loom (the kernel texts come from it)
Each benchmark k_<kernel>_wg<threads>: the kernel text with its thread count replaced (timing only for a size the kernel's own lane
mapping does not cover)."""
import re, sys
t = open(sys.argv[1]).read()
ROWS = 2048
H = 1536


def kern(name):
    m = re.search(r'^kernel\.def[^\n]*export\("%s"\)' % name, t, re.M)
    s = m.start()
    e = t.index("\n  kernel.return\n}\n", s) + len("\n  kernel.return\n}\n")
    k = t[s:e]
    return re.sub(r"^kernel\.def (target\(@\w+\) )?", "kernel.def target(@rdna4) ", k, count=1)


out = ["amdgpu.target<gfx12-generic> @rdna4 {subgroup_size = 32}\n"]
F = f"tensor<{ROWS}x{H}xf32>"


def add(name, text, case):
    text = text.replace(f'export("{name}") @{name}(', f'export("k_{name}") @k_{name}(') if False else text
    out.append(text)
    out.append(case)


def norm(wg):
    k = kern("g_post_add_norm")
    k = k.replace('export("g_post_add_norm") @g_post_add_norm(', f'export("k_norm_wg{wg}") @k_norm_wg{wg}(')
    k = k.replace("%c256 = index.constant 256 : index", f"%c256 = index.constant {wg} : index").replace("[range(%lane_id, 0, 255)]", f"[range(%lane_id, 0, {wg - 1})]")
    return k, f"""check.case public @k_norm_wg{wg}_case {{
  %rows = check.literal value({ROWS}) : index
  %y = check.generate.fill value(0.5) : {F}
  %wp = check.generate.fill value(1.0) : tensor<{H}xf32>
  %x = check.generate.fill value(0.25) : {F}
  %wn = check.generate.fill value(1.0) : tensor<{H}xf32>
  %h = check.generate.fill value(0.0) : {F}
  kernel.launch @k_norm_wg{wg}[%rows](%rows, %y, %wp, %x, %wn, %h) : [index](index, {F}, tensor<{H}xf32>, {F}, tensor<{H}xf32>, {F})
  check.return
}}
check.benchmark<@k_norm_wg{wg}_case> @k_norm_wg{wg}_bm
"""


def rs(wg):
    k = kern("g_row_scale_f16")
    k = k.replace('export("g_row_scale_f16") @g_row_scale_f16(', f'export("k_rs_wg{wg}") @k_rs_wg{wg}(')
    k = k.replace("%c256 = index.constant 256 : index", f"%c256 = index.constant {wg} : index")
    return k, f"""check.case public @k_rs_wg{wg}_case {{
  %rows = check.literal value({ROWS}) : index
  %k = check.literal value({H}) : index
  %x = check.generate.random.uniform seed(%seed0) range(-1.0 to 1.0) : {F}
  %s = check.generate.fill value(0.0) : tensor<{ROWS}x2xf32>
  %x16 = check.generate.fill value(0.0) : tensor<{ROWS}x{H}xf16>
  kernel.launch @k_rs_wg{wg}[%rows](%rows, %k, %x, %s, %x16) : [index](index, index, {F}, tensor<{ROWS}x2xf32>, tensor<{ROWS}x{H}xf16>)
  check.return
}}
check.benchmark<@k_rs_wg{wg}_case> @k_rs_wg{wg}_bm
""".replace("  %x = check.generate.random.uniform seed(%seed0) range(-1.0 to 1.0)", "  %seed0 = check.literal value(7) : i64\n  %x = check.generate.random.uniform seed(%seed0) range(-1.0 to 1.0)")


for wg in (256, 128, 64):
    add("norm", *norm(wg)) if wg in (256, 64) else None
    add("rs", *rs(wg))
open(sys.argv[2], "w").write("\n".join(out))
