"""Test reuse of activation fragments across independent experts without reordering the router's F32 sums."""
from pathlib import Path
import re
import sys

import numpy as np

root, out = map(Path, sys.argv[1:3])
sys.path.insert(0, str(root/"skills/loom-port-llm/starter"))
import gen_moe as moe

target = "amdgpu.target<gfx11-generic> @w32 {subgroup_size = 32}\n"
rng = np.random.default_rng(19)
weights = rng.normal(0,.04,(256,2048)).astype("<f4")
weights[2] = weights[1]
weights[3] = np.nextafter(weights[1], np.float32(1))
np.save(out/"weights.npy", weights)
def kernel(tokens, experts):
    moe.ROUTE_TR = tokens
    source = moe.route_prefill_live(2048,256,8,1,True,False)
    if experts == 1:
        return source
    # EXP independent expert scores are accumulated in the original K order. One LDS load feeds all EXP.
    lines = [f"  %ru = index.constant {experts} : index", "  scf.for %j = [%c0 to %cEPW step %ru] {"]
    for e in range(experts):
        lines += [f"    %sj{e} = index.constant {e} : index", f"    %raw{e}_v = index.add %e0, %j : index",
                  f"    %raw{e} = index.add %raw{e}_v, %sj{e} : index",
                  f"    %expert{e} = index.assume %raw{e} [range(%raw{e}, 0, 255)] : index"]
    indices = [f"{e}_{r}" for e in range(experts) for r in range(tokens)]
    types = ", ".join(["f32"]*len(indices))
    init = ", ".join(f"%acc{i} = %zero_f : f32" for i in indices)
    lines += ["    "+", ".join(f"%sum{i}" for i in indices)+f" = scf.for %bq = [%c0 to %cHB step %c1]({init}) -> ({types}) {{",
              "      %bb = index.assume %bq [range(%bq, 0, 15)] : index"]
    for r in range(tokens):
        lines += [f"      %frag{r} = vector.load %xs4_view[{r}, %bb, %lane, %c0] : view<{tokens}x16x32x4xf32> -> vector<4xf32>"]
    for e in range(experts):
        lines += [f"      %weights{e} = vector.load %w4_view[%expert{e}, %bb, %lane, %c0] : view<256x16x32x4xf32> -> vector<4xf32>"]
        for r in range(tokens):
            lines += [f"      %dot{e}_{r} = vector.dotf<reassoc|contract> %weights{e}, %frag{r}, %zero_f : vector<4xf32>, vector<4xf32>, f32",
                      f"      %v{e}_{r} = scalar.addf %acc{e}_{r}, %dot{e}_{r} : f32"]
    lines += ["      scf.yield "+", ".join(f"%v{i}" for i in indices)+f" : {types}","    }"]
    for e in range(experts):
        for r in range(tokens):
            lines += [f"    %result{e}_{r} = kernel.subgroup.reduce<addf> %sum{e}_{r} : f32",
                      "    scf.if %first {",f"      view.store %result{e}_{r}, %lg_view[{r}, %expert{e}] : f32, view<{tokens}x256xf32>","    }"]
    lines += ["  }"]
    begin = source.index("  scf.for %j = [%c0 to %cEPW")
    end = source.index("  kernel.barrier<workgroup>",begin)
    return source[:begin]+"\n".join(lines)+"\n"+source[end:]

def case(t, verify):
    x = rng_template(t)
    meta = np.ones((t,4),dtype="<i4")
    if verify:
        x[0] = 0 # exactly equal scores
        x[1] = weights[1]*8 # equal and one-ULP neighboring router rows
        if t > 16: meta[((t-8)//4)*4:,2]=0
    np.save(out/f"x{t}.npy",x)
    np.save(out/f"meta{t}.npy",meta)
    ids, vals = f"tensor<{t}x8xi32>",f"tensor<{t}x8xf32>"
    types=f"index, tensor<{t}x2048xf32>, tensor<256x2048xf32>, tensor<256xf32>, {ids}, {vals}, tensor<{t}x4xi32>"
    text = f"""check.case public @t{t} {{
  %t = check.literal value({t}) : index
  %x = check.file.read.npy path("x{t}.npy") : tensor<{t}x2048xf32>
  %w = check.file.read.npy path("weights.npy") : tensor<256x2048xf32>
  %b = check.generate.fill value(0.0) : tensor<256xf32>
  %info = check.file.read.npy path("meta{t}.npy") : tensor<{t}x4xi32>
"""
    for name in (("baseline","candidate") if verify else ("candidate",)):
        text+=f"  %{name}_id = check.generate.fill value(0) : {ids}\n  %{name}_wt = check.generate.fill value(0.0) : {vals}\n"
        text+=f"  kernel.launch @{name}[%t](%t, %x, %w, %b, %{name}_id, %{name}_wt, %info) : [index]({types})\n"
    if verify:
        for label,ty in [("id",ids),("wt",vals)]:
            text+=f"  check.expect.equal actual(%candidate_{label}) expected(%baseline_{label}) : {ty}\n"
    text+="  check.return\n}\n"
    if not verify: text+=f"check.benchmark<@t{t}> @t{t}_bm\n"
    return text
def rng_template(t):
    return np.random.default_rng(t).normal(0,1,(t,2048)).astype("<f4")
def rename(text,name):
    return text.replace("g_moe_route_tr_live",name)
base=kernel(4,1)
for name,tr,experts in [("base",4,1),("pair",4,2),("quad",4,4),("tr2q",2,4)]:
    candidate=rename(kernel(tr,experts),"candidate")
    (out/("bench.loom" if name=="base" else f"bench_{name}.loom")).write_text(target+candidate+case(1024,False)+case(4096,False))
    if name!="base":
        (out/f"test_{name}.loom").write_text(target+rename(base,"baseline")+candidate+case(65,True)+case(257,True))
