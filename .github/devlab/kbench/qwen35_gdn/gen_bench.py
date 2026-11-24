"""Time the sequential and chunked Strix prefill recurrence at the live Qwen 35B shape."""
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

root, out = map(Path, sys.argv[1:3])
cfg = json.loads((root / "fleet/unsloth-Qwen3.5-35B-A3B-loom/llm_config.json").read_text())
sys.path.insert(0, str(root / "skills/loom-port-llm/starter"))
import gen_gdn as gdn
from gen_gdn_chunk import chunk_kernel

os.environ.update(LOOM_RDNA4="0", LLM_DRAFT=json.dumps({"kind":"dflash"}))
geom = gdn.gdn_geometry(cfg)
hk, hv, dk, dv, width, layer_bytes = (geom[key] for key in ("HK","HV","DK","DV","C","layer_bytes"))
layers = len(cfg["recurrent_layers"])
assert (hk,hv,dk,dv) == (16,32,128,128)
head = "amdgpu.target<gfx11-generic> @w32 {subgroup_size = 32}\n"
all_kernels = gdn.gdn_kernels(cfg, cfg["rms_eps"])
base = re.search(r'^kernel\.def export\("g_gdn_recurrence_pf"\).*?^\}\s*$', all_kernels, re.M|re.S).group().strip()
guard = "    %publish = func.call @g_gdn_publish(%metadata, %spans, %span) : (buffer, index, index) -> (i1)\n    scf.if %publish {\n"
modes = {"base":base}
for cn in (16,32):
    modes[f"cn{cn}"] = chunk_kernel(geom,layers,layer_bytes,guard,"\n    }",arch="rdna3",cn=cn).replace("g_gdn_chunk","g_gdn_recurrence_pf")

def case(tokens, continued, verify):
    rng = np.random.default_rng(tokens + continued)
    prefix = f"t{tokens}c{int(continued)}"
    a = -rng.uniform(0.1,1,hv).astype("f")
    dt = rng.normal(size=hv).astype("f")
    alpha = rng.normal(size=(tokens,hv)).astype("f")
    beta = rng.normal(size=(tokens,hv)).astype("f")
    x = rng.normal(size=(tokens,width)).astype("f")
    q,k,v = np.split(x, [hk*dk, 2*hk*dk], axis=1)
    q,k,v = q.reshape(tokens,hk,dk),k.reshape(tokens,hk,dk),v.reshape(tokens,hv,dv)
    qinv = 1/np.sqrt((q.astype("d")**2).sum(axis=2)+cfg["rms_eps"])
    kinv = 1/np.sqrt((k.astype("d")**2).sum(axis=2)+cfg["rms_eps"])
    inv = np.stack([qinv,kinv],axis=-1).reshape(tokens,2*hk).astype("f")
    state = rng.normal(0,0.2,(hv,dv,dk)) if continued else np.zeros((hv,dv,dk))
    arena = np.zeros(2*layer_bytes//4,'f')
    arena[:state.size] = state.ravel()
    pos = 128 if continued else 0
    orig = np.array([0,layer_bytes,0,pos,0],'<i8').view('<i4')
    meta = np.array([1,0,0,0,tokens,pos,0,0,0,0,0,0,0],'<i4')
    data = dict(meta=meta,orig=orig,arena=arena,x=x,inv=inv,alpha=alpha,beta=beta,a=a,dt=dt)
    p = [f"check.case public @{prefix} {{",f"  %t = check.literal value({tokens}) : index","  %spans = check.literal value(1) : index","  %layer = check.literal value(0) : index"]
    def tensor(v):
        return "tensor<"+"x".join(map(str,v.shape))+("xi32>" if v.dtype.kind=="i" else "xf32>")
    for key,val in data.items():
        fname=f"{prefix}_{key}.npy"
        np.save(out/fname,val)
        p.append(f'  %{key} = check.file.read.npy path("{fname}") : {tensor(val)}')
    raw = f"tensor<{tokens}x{hv*dv}xf32>"
    p.append(f"  %raw = check.generate.fill value(0.0) : {raw}")
    p.append("  kernel.launch @g_gdn_recurrence_pf[%spans](%t, %spans, %layer, "+", ".join("%"+key for key in data)+", %raw) : [index](index, index, index, "+", ".join(tensor(v) for v in data.values())+", "+raw+")")
    if verify:
        inds = np.repeat(np.arange(hk),hv//hk) if cfg["gdn"].get("v_order")=="grouped" else np.arange(hv)%hk
        ref = np.empty((tokens,hv,dv))
        for i in range(tokens):
            kk = k[i,inds,:]*kinv[i,inds,None]
            qq = q[i,inds,:]*qinv[i,inds,None]/np.sqrt(dk)
            state *= np.exp(np.logaddexp(0,alpha[i].astype("d")+dt)*a)[:,None,None]
            delta = (v[i]-np.einsum("hvk,hk->hv",state,kk))/(1+np.exp(-beta[i].astype("d")))[:,None]
            state += np.einsum("hv,hk->hvk",delta,kk)
            ref[i] = np.einsum("hvk,hk->hv",state,qq)
        expected = arena.copy()
        expected[layer_bytes//4:layer_bytes//4+state.size] = state.ravel()
        for key,actual,reference in [("outputs","raw",ref.reshape(tokens,hv*dv)),("state","arena",expected)]:
            reference = reference.astype("f")
            np.save(out/f"{prefix}_{key}.npy",reference)
            ty=tensor(reference)
            p.append(f'  %expected_{key} = check.file.read.npy path("{prefix}_{key}.npy") : {ty}')
            p.append(f"  check.expect.close actual(%{actual}) expected(%expected_{key}) atol({.003*np.abs(reference).max():.6e}) rtol(1.0e-2) nan(different) : {ty}")
    p.extend(["  check.return","}"])
    if not verify:
        p.append(f"check.benchmark<@{prefix}> @{prefix}_bm")
    return "\n".join(p)

benchmark = case(4096,False,False)+"\n"+case(2048,True,False)
checks = case(97,True,True)+"\n"+case(128,False,True)
for name, kernel in modes.items():
    module = head+gdn.SLOT+"\n"+kernel+"\n"
    (out/("bench.loom" if name=="base" else f"bench_{name}.loom")).write_text(module+benchmark)
    (out/f"test_{name}.loom").write_text(module+checks)
