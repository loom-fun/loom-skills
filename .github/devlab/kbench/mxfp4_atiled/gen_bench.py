#!/usr/bin/env python3
"""Bounded exact A-layout correctness and real-shape interleaved GEMM timing."""
import importlib.util
import os
from pathlib import Path
import sys
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np

ROOT, OUT = map(lambda p: Path(p).resolve(), sys.argv[1:3])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT/'skills/loom-port-llm/starter'))
import gen_mxfp4s as G
spec = importlib.util.spec_from_file_location('oldbench', ROOT/'.github/devlab/kbench/gemm_mxfp4s/gen_bench.py')
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
HEAD = 'amdgpu.target<gfx12-generic> @rdna4 {subgroup_size = 32}\n'


def kernel(mode, name, tiled):
    os.environ['GEMM_MXFP4S_A_TILED'] = '1' if tiled else '0'
    text = G.gemm_mxfp4s_f8_m256(mode)
    old = f'g_gemm_mxfp4s_{mode}_f8_m256_rdna4'
    return text.replace(f'export("{old}") @{old}(', f'export("{name}") @{name}(')


def pack(x):
    # Independent inverse of the wave lane layout; no call to product offset().
    rows, width = x.shape
    return x.reshape(rows//16, 16, width//16, 2, 8).transpose(0, 2, 3, 1, 4).reshape(x.shape)


def save(name, value):
    np.save(OUT/(name+'.npy'), value)


def inputs(tag, rows, width, n, mode, reference=True):
    rng = np.random.default_rng(515 + rows + width + n + (mode == 'swiglu'))
    xf = B.e4m3_round(np.clip(rng.standard_normal((rows,width))*3, -400,400))
    x = B.e4m3_bytes(xf).view(np.int8)
    save(tag+'_x', x); save(tag+'_xt', pack(x))
    scales = np.ones((rows,2),np.float32); scales[:,1] = 2.0**-8
    save(tag+'_s',scales)
    values=[]
    for wi in range(2 if mode == 'swiglu' else 1):
        nib = rng.integers(0,256,(n,width//2),dtype=np.uint8)
        sc = rng.integers(116,122,(n,width//32),dtype=np.uint8)
        save(tag+f'_w{wi}',nib.reshape(-1).view(np.int8));save(tag+f'_e{wi}',sc.reshape(-1).view(np.int8))
        if reference: values.append(xf.astype(np.float64) @ B.dequant(nib,sc,'pairs').T)
    if not reference: return
    y = values[0]
    if mode=='swiglu': y = y/(1+np.exp(np.clip(-y,-80,80)))*values[1]
    assert np.isfinite(y).all() and np.max(np.abs(y))>0.01
    save(tag+'_ref',y.astype(np.float32))
    return y


def case(tag, rows, width, n, mode, names, negative=None, timing=False):
    # names list of (kernel symbol, tiled input). Same physical values per arm.
    types=[f'tensor<{n*width//2}xi8>',f'tensor<{n*width//32}xi8>']*(2 if mode=='swiglu' else 1)
    decl='';args=[]
    for wi in range(2 if mode=='swiglu' else 1):
        decl+=f'  %w{wi} = check.file.read.npy path("{tag}_w{wi}.npy") : {types[2*wi]}\n  %e{wi} = check.file.read.npy path("{tag}_e{wi}.npy") : {types[2*wi+1]}\n'
        args += [f'%w{wi}',f'%e{wi}']
    xt=f'tensor<{rows}x{width}xi8>';st=f'tensor<{rows}x2xf32>';yt=f'tensor<{rows}x{n}xf32>'
    cname=tag+('_'+negative if negative else '')
    text=f'''check.case public @{cname} {{
  %rows = check.literal value({rows}) : index
  %k = check.literal value({width}) : index
  %n = check.literal value({n}) : index
  %x = check.file.read.npy path("{tag}_x.npy") : {xt}
  %xt = check.file.read.npy path("{tag}_xt.npy") : {xt}
  %s = check.file.read.npy path("{tag}_s.npy") : {st}
{decl}'''
    for i,(kn,tiled) in enumerate(names):
        xarg='%xt' if tiled and negative!='wrong_layout' else '%x'
        text+=f'  %y{i} = check.generate.fill value(0.0) : {yt}\n  kernel.launch @{kn}[%rows, %n](%rows, %k, %n, {xarg}, %s, '+', '.join(args)+f', %y{i}) : [index, index](index, index, index, {xt}, {st}, '+', '.join(types)+f', {yt})\n'
    if not timing:
        if negative=='zero_output':
            text+=f'  %z = check.generate.fill value(0.0) : {yt}\n  check.expect.approximate actual(%y0) expected(%z) atol(0.0) rtol(0.0) nan(different) : {yt}, {yt}\n'
        elif negative=='wrong_layout':
            text+=f'  %ref = check.file.read.npy path("{tag}_ref.npy") : {yt}\n  check.expect.approximate actual(%y0) expected(%ref) atol(1.0e-3) rtol(2.0e-3) nan(different) : {yt}, {yt}\n'
        else:
            text+=f'  %ref = check.file.read.npy path("{tag}_ref.npy") : {yt}\n  check.expect.equal actual(%y0) expected(%y1) : {yt}\n  check.expect.approximate actual(%y1) expected(%ref) atol(1.0e-3) rtol(2.0e-3) nan(different) : {yt}, {yt}\n'
    text+='  check.return\n}\n'
    if timing:text+=f'check.benchmark<@{cname}> @{cname}_bm\n'
    return text


positive=[];negative=[]
# Real model widths with bounded output rows; nonzero independent fp64 references.
for mode,width,n in [('store',5120,256),('store',17408,128),('swiglu',5120,128)]:
    tag=f'check_{mode}_{width}';rows=256
    inputs(tag,rows,width,n,mode)
    old,new=f'old_{mode}_{width}',f'new_{mode}_{width}'
    defs=kernel(mode,old,False)+kernel(mode,new,True)
    positive+=[defs,case(tag,rows,width,n,mode,[(old,False),(new,True)])]
    if mode=='store' and width==5120:
        negative += [kernel(mode,'negative_store',True),case(tag,rows,width,n,mode,[('negative_store',True)],'zero_output'),case(tag,rows,width,n,mode,[('negative_store',True)],'wrong_layout')]
(OUT/'test.loom').write_text(HEAD+'\n'.join(positive))
(OUT/'negative.loom').write_text(HEAD+'\n'.join(negative))
# Timing includes original 1024-row chunk and 2048-row rung, every relevant shape.
for tiled,label in [(False,'base'),(True,'atiled')]:
    sections=[]
    for rows in [1024,2048]:
        for mode,width,n in B.SHAPES:
            tag=f'time_{mode}_{width}_{n}_{rows}';kn=f'k_{mode}_{width}_{n}_{rows}'
            if not tiled: inputs(tag,rows,width,n,mode,reference=False)
            sections += [kernel(mode,kn,tiled),case(tag,rows,width,n,mode,[(kn,tiled)],timing=True)]
    (OUT/('bench.loom' if not tiled else 'bench_atiled.loom')).write_text(HEAD+'\n'.join(sections))

# Producer test: extract only the existing literal, never import/run gen_llm.
# Independent max exponent and IEEE FP8 reference; small tails must not write a
# tiled payload into less than 16 rows of backing storage.
import ast
module = ast.parse((ROOT/'skills/loom-port-llm/starter/gen_llm.py').read_text())
row_scale = next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ROW_SCALE' for t in n.targets))
os.environ['GEMM_MXFP4S_A_TILED']='1'
producer=G.f8_producer(row_scale)
producer=producer.replace('g_row_scale_f16_f8_rdna4','k_producer')
prod=[producer]
for rows in [1,8,16,64,256]:
    width=5120;tag=f'producer_{rows}'
    rng=np.random.default_rng(1400+rows)
    xf=(rng.integers(-128,128,(rows,width)).astype(np.float32)/16)
    xf[:,0]=8.0  # exact maximum exponent3 -> FP8 scale16 and inverse2^-12.
    save(tag+'_x',xf)
    expected=pack(B.e4m3_bytes(B.e4m3_round(xf*16)).view(np.int8)) if rows%16==0 else np.full((rows,width),31,np.int8)
    save(tag+'_ref',expected)
    ps=np.tile(np.array([16.0,2.0**-12],np.float32),(rows,1));save(tag+'_sref',ps)
    xt=f'tensor<{rows}x{width}xf32>';ht=f'tensor<{rows}x{width}xf16>';bt=f'tensor<{rows}x{width}xi8>';st=f'tensor<{rows}x2xf32>'
    prod.append(f'''check.case public @{tag} {{
  %rows = check.literal value({rows}) : index
  %k = check.literal value({width}) : index
  %w16 = check.literal value(1) : index
  %x = check.file.read.npy path("{tag}_x.npy") : {xt}
  %ref = check.file.read.npy path("{tag}_ref.npy") : {bt}
  %sref = check.file.read.npy path("{tag}_sref.npy") : {st}
  %s16 = check.generate.fill value(0.0) : {st}
  %s8 = check.generate.fill value(0.0) : {st}
  %h16 = check.generate.fill value(0.0) : {ht}
  %x8 = check.generate.fill value(31) : {bt}
  kernel.launch @k_producer[%rows](%rows, %k, %x, %s16, %h16, %x8, %s8, %w16) : [index](index, index, {xt}, {st}, {ht}, {bt}, {st}, index)
  check.expect.equal actual(%x8) expected(%ref) : {bt}
  check.expect.equal actual(%s8) expected(%sref) : {st}
  check.return
}}
''')
(OUT/'producer.loom').write_text(HEAD+'\n'.join(prod))

# Price producer cost alongside consumer: the layout is not a free prepack.
for tiled,label in [(False,'base'),(True,'atiled')]:
    os.environ['GEMM_MXFP4S_A_TILED']='1' if tiled else '0'
    text=G.f8_producer(row_scale).replace('g_row_scale_f16_f8_rdna4','k_pack_producer')
    for width in [5120,6144,17408]:
        rows=1024;tag=f'pack_{width}'
        if not tiled:
            rng=np.random.default_rng(900+width)
            save(tag+'_x',rng.standard_normal((rows,width)).astype(np.float32))
        xt=f'tensor<{rows}x{width}xf32>';ht=f'tensor<{rows}x{width}xf16>';bt=f'tensor<{rows}x{width}xi8>';st=f'tensor<{rows}x2xf32>'
        text+=f'''check.case public @{tag} {{
  %rows = check.literal value({rows}) : index
  %k = check.literal value({width}) : index
  %w16 = check.literal value(0) : index
  %x = check.file.read.npy path("{tag}_x.npy") : {xt}
  %s16 = check.generate.fill value(0.0) : {st}
  %s8 = check.generate.fill value(0.0) : {st}
  %h16 = check.generate.fill value(0.0) : {ht}
  %x8 = check.generate.fill value(31) : {bt}
  kernel.launch @k_pack_producer[%rows](%rows, %k, %x, %s16, %h16, %x8, %s8, %w16) : [index](index, index, {xt}, {st}, {ht}, {bt}, {st}, index)
  check.return
}}
check.benchmark<@{tag}> @{tag}_bm
'''
    path=OUT/('bench.loom' if not tiled else 'bench_atiled.loom')
    with path.open('a') as f:f.write(text)
