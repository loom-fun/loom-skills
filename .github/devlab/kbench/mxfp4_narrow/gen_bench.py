#!/usr/bin/env python3
"""Half-width N tiles: exact-K whole-output proof and all seven model shapes."""
import importlib.util
import os
from pathlib import Path
import sys
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
os.environ['GEMM_MXFP4S_A_TILED']='0'
os.environ.pop('GEMM_MXFP4S_A_TILED__RDNA4',None)
import numpy as np
ROOT,OUT=map(lambda p:Path(p).resolve(),sys.argv[1:3])
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'skills/loom-port-llm/starter'))
import gen_mxfp4s as G
spec=importlib.util.spec_from_file_location('reference',ROOT/'.github/devlab/kbench/gemm_mxfp4s/gen_bench.py')
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
HEAD='amdgpu.target<gfx12-generic> @rdna4 {subgroup_size = 32}\n'


def kernel(mode,name,variant):
    kw={} if variant=='base' else dict(G.F8_KERNELS['narrow'][mode=='swiglu'])
    if variant=='narrow_swp':kw['swp']=True
    text=G.gemm_mxfp4s_f8_m256(mode,**kw)
    old=f'g_gemm_mxfp4s_{mode}_f8_m256_rdna4'
    return text.replace(f'export("{old}") @{old}(',f'export("{name}") @{name}(')


def inputs(tag,rows,width,n,mode,reference):
    rng=np.random.default_rng(11801+rows+width+n+(mode=='swiglu'))
    xf=B.e4m3_round(np.clip(rng.standard_normal((rows,width))*3,-400,400))
    np.save(OUT/(tag+'_x.npy'),B.e4m3_bytes(xf).view(np.int8))
    scales=np.ones((rows,2),np.float32);scales[:,1]=2.0**-8
    np.save(OUT/(tag+'_s.npy'),scales)
    values=[]
    for i in range(2 if mode=='swiglu' else 1):
        w=rng.integers(0,256,(n,width//2),dtype=np.uint8)
        e=rng.integers(116,122,(n,width//32),dtype=np.uint8)
        np.save(OUT/(tag+f'_w{i}.npy'),w.reshape(-1).view(np.int8))
        np.save(OUT/(tag+f'_e{i}.npy'),e.reshape(-1).view(np.int8))
        if reference:values.append(xf.astype(np.float64)@B.dequant(w,e,'pairs').T)
    if reference:
        y=values[0]
        if mode=='swiglu':y=y/(1+np.exp(np.clip(-y,-80,80)))*values[1]
        assert np.isfinite(y).all() and np.max(np.abs(y))>.01
        np.save(OUT/(tag+'_ref.npy'),y.astype(np.float32))


def case(tag,rows,width,n,mode,names,timing=False,negative=False):
    xt=f'tensor<{rows}x{width}xi8>';st=f'tensor<{rows}x2xf32>';yt=f'tensor<{rows}x{n}xf32>'
    cname=tag+('_zero_output' if negative else '')
    text=f'''check.case public @{cname} {{
  %rows = check.literal value({rows}) : index
  %k = check.literal value({width}) : index
  %n = check.literal value({n}) : index
  %x = check.file.read.npy path("{tag}_x.npy") : {xt}
  %s = check.file.read.npy path("{tag}_s.npy") : {st}
'''
    args=[];types=[]
    for i in range(2 if mode=='swiglu' else 1):
        for prefix,size in [('w',n*width//2),('e',n*width//32)]:
            ty=f'tensor<{size}xi8>';types.append(ty);args.append(f'%{prefix}{i}')
            text+=f'  %{prefix}{i} = check.file.read.npy path("{tag}_{prefix}{i}.npy") : {ty}\n'
    for i,name in enumerate(names):
        text+=f'  %y{i} = check.generate.fill value(0.0) : {yt}\n  kernel.launch @{name}[%rows, %n](%rows, %k, %n, %x, %s, '+', '.join(args)+f', %y{i}) : [index, index](index, index, index, {xt}, {st}, '+', '.join(types)+f', {yt})\n'
    if negative:
        text+=f'  %zero = check.generate.fill value(0.0) : {yt}\n  check.expect.approximate actual(%y0) expected(%zero) atol(0.0) rtol(0.0) nan(different) : {yt}, {yt}\n'
    elif not timing:
        text+=f'  %ref = check.file.read.npy path("{tag}_ref.npy") : {yt}\n'
        for i in range(1,len(names)):
            text+=f'  check.expect.equal actual(%y0) expected(%y{i}) : {yt}\n'
        for i in range(len(names)):
            text+=f'  check.expect.approximate actual(%y{i}) expected(%ref) atol(1.0e-3) rtol(2.0e-3) nan(different) : {yt}, {yt}\n'
    text+='  check.return\n}\n'
    if timing:text+=f'check.benchmark<@{cname}> @{cname}_bm\n'
    return text


positive=[];negative=[]
# Every distinct K, both epilogues, two complete M tiles. Whole tensors compared;
# independent FP64 decoding/matmul covers every row and output, no sampled checks.
for rows in [256,512]:
    for mode,width,n in [('store',5120,256),('store',17408,128),('store',6144,128),('swiglu',5120,128)]:
        tag=f'check_{mode}_{width}_{rows}';inputs(tag,rows,width,n,mode,True)
        names=[]
        for variant in ['base','narrow','narrow_swp']:
            name=f'{variant}_{mode}_{width}_{rows}';names.append(name)
            positive.append(kernel(mode,name,variant))
        positive.append(case(tag,rows,width,n,mode,names))
        if rows==256 and width==5120:
            name=f'negative_{mode}'
            negative += [kernel(mode,name,'narrow'),case(tag,rows,width,n,mode,[name],negative=True)]
(OUT/'test.loom').write_text(HEAD+'\n'.join(positive))
(OUT/'negative.loom').write_text(HEAD+'\n'.join(negative))
# Include formerly unpriced small-N full-attention K/V (32 occurrences/token).
shapes=B.SHAPES+[('store',5120,1024)]
for variant in ['base','narrow','narrow_swp']:
    sections=[]
    for rows in [1024,2048]:
        for mode,width,n in shapes:
            tag=f'time_{mode}_{width}_{n}_{rows}';name=f'k_{mode}_{width}_{n}_{rows}'
            if variant=='base':inputs(tag,rows,width,n,mode,False)
            sections += [kernel(mode,name,variant),case(tag,rows,width,n,mode,[name],timing=True)]
    (OUT/('bench.loom' if variant=='base' else f'bench_{variant}.loom')).write_text(HEAD+'\n'.join(sections))
