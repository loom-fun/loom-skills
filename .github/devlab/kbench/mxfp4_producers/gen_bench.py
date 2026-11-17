#!/usr/bin/env python3
"""Exact H5120 operand producer pilot: real generated kernels, independent refs.
Usage ROOT OUTDIR [base fixed guarded held h32]. No package/default is changed.
"""
import ast
import json
from pathlib import Path
import re
import sys
import numpy as np
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'skills/loom-port-llm/starter'))
import gen_mxfp4s as G
import mxfp4_producers as P
HEAD='amdgpu.target<gfx12-generic> @rdna4 {subgroup_size = 32}\n'
VARIANTS=('base','fixed','guarded','held','h32')

def baseline_producers(root):
    """Evaluate only actual producer templates/functions, without a model build.

    The H5120/no-postnorm constants are the fixed accepted model contract.
    Unrelated embedding template fields are excluded, not mocked or imported.
    """
    names={'NORMS','ROW_SCALE','ROW_SCALE_LIVE','_TOKEN_ROW','live_skip_kernel','post_add_norm_rs'}
    source=(root/'skills/loom-port-llm/starter/gen_llm.py').read_text()
    tree=ast.parse(source)
    a=source.index('kernel.def export("g_post_add_norm")');b=source.index('\n"""',a)
    post_node=ast.parse('NORMS = f'+repr(source[a:b]+'\n')).body[0]
    tree.body=[post_node if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NORMS' for t in node.targets) else node for node in tree.body]
    selected=[node for node in tree.body if (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in node.targets)) or isinstance(node,ast.FunctionDef) and node.name in names]
    ns=dict(H=5120,HG=160,EPS=1e-6,POST_EPS=1e-8,POST_NORM=False,re=re)
    exec(compile(ast.Module(body=selected,type_ignores=[]),'gen_llm selected source','exec'),ns)
    result=[]
    for text,name in [(ns['ROW_SCALE_LIVE'],'g_row_scale_f16_live'),(ns['post_add_norm_rs'](True),'g_post_add_norm_rs_live')]:
        text=text[text.index('kernel.def'):]
        live,_=ns['live_skip_kernel'](text,name)
        result.append(G.f8_producer(live))
    return result


def kernels(root):
    row,post=baseline_producers(root)
    result={}
    for arm in VARIANTS:
        result[arm]={}
        for width in (5120,6144,17408):
            t=row
            if arm in ('guarded','held','h32'):t=P.guard_half_arithmetic(t)
            if arm in ('held','h32'):t=P.hold_row_inputs(t,width)
            elif arm!='base':t=P.fixed_width(t,width)
            result[arm][f'row_{width}']=P.rename(t,f'row_{width}_{arm}')
        t=post
        if arm in ('guarded','held','h32'):t=P.guard_half_arithmetic(t)
        if arm=='held':t=P.postnorm_wg160(t)
        if arm=='h32':t=P.postnorm_dead_h32(t,paired_ab_fp8=True)
        result[arm]['post']=P.rename(t,f'post_{arm}')
    return result


def fp8_bits(x):
    """Independent RNE against positive E4M3 finite codebook, including ties."""
    codes=np.arange(127,dtype=np.uint8)
    e=(codes.astype(np.int32)>>3)&15;m=codes&7
    values=np.where(e==0,m.astype(np.float64)*2**-9,(1+m.astype(np.float64)/8)*np.exp2(e-7))
    a=np.abs(x.astype(np.float64));hi=np.searchsorted(values,a).clip(0,126);lo=(hi-1).clip(0,126)
    dl=a-values[lo];dh=values[hi]-a
    take=(dh<dl)|((dh==dl)&((hi&1)==0))
    q=np.where(take,hi,lo).astype(np.uint8)
    q|=np.where(np.signbit(x),128,0).astype(np.uint8)
    return q.view(np.int8)


def scales(x,live):
    maximum=np.max(np.abs(x),axis=1).astype(np.float32)
    biased=((maximum.view(np.uint32)>>23)&255).astype(np.int32)
    exponent=np.clip(biased-127-13,0,100)
    s=np.stack((np.exp2(-exponent),np.exp2(exponent)),1).astype(np.float32)
    sf=np.exp2(np.clip(261-biased,1,254)-127).astype(np.float32)
    inv=np.exp2(np.clip(biased-(7+G.MXFP4_F8_SHIFT),1,254)-127).astype(np.float32)
    s8=np.stack((sf,inv),1);s[~live]=0;s8[~live]=0
    return s,s8


def save(out,tag,arrays):
    for name,array in arrays.items():np.save(out/f'{tag}_{name}.npy',array)


def read(tag,name,typ):
    return f'  %{name} = check.file.read.npy path("{tag}_{name}.npy") : {typ}\n'


def eq(actual,expected,typ,approx=False):
    if approx:return f'  check.expect.approximate actual(%{actual}) expected(%{expected}) atol(3.0e-6) rtol(3.0e-6) nan(different) : {typ}, {typ}\n'
    return f'  check.expect.equal actual(%{actual}) expected(%{expected}) : {typ}\n'


def row_case(out,width,rows,w16,arms,check):
    tag=f'row_{width}_{rows}_{w16}';rng=np.random.default_rng(width+rows+w16)
    x=rng.uniform(-7.9,7.9,(rows,width)).astype(np.float32);x[:,0]=8
    # Diverse signs, zero, subnormal, a scale above one and a wide exponent.
    x[1::11]=0;x[2::11]*=2**-20;x[3::11]*=2**20
    info=np.ones((rows,4),np.int32);info[::17,2]=0;live=info[:,2]!=0
    s,s8=scales(x,live);q=fp8_bits((x*s8[:,0,None]).astype(np.float32));q[~live]=31
    h=np.full_like(x,37,dtype=np.float16)
    if w16:h[live]=(x[live]*s[live,0,None]).astype(np.float16)
    T=f'tensor<{rows}x{width}xf32>';I=f'tensor<{rows}x4xi32>';Q=f'tensor<{rows}x{width}xi8>';H=f'tensor<{rows}x{width}xf16>';S=f'tensor<{rows}x2xf32>'
    save(out,tag,dict(x=x,info=info,eq=q,eh=h,es=s,es8=s8))
    lines=f'check.case public @{tag} {{\n  %r = check.literal value({rows}) : index\n  %k = check.literal value({width}) : index\n  %flag = check.literal value({w16}) : index\n'+read(tag,'x',T)+read(tag,'info',I)
    for arm in arms:
        for name,typ,value in [('s',S,'42.0'),('h',H,'37.0'),('q',Q,'31'),('s8',S,'42.0')]:lines+=f'  %{name}_{arm} = check.generate.fill value({value}) : {typ}\n'
        lines+=f'  kernel.launch @row_{width}_{arm}[%r](%r, %info, %k, %x, %s_{arm}, %h_{arm}, %q_{arm}, %s8_{arm}, %flag) : [index](index, {I}, index, {T}, {S}, {H}, {Q}, {S}, index)\n'
    if check:
        for name,typ in [('s',S),('h',H),('q',Q),('s8',S)]:
            ref='e'+name;lines+=read(tag,ref,typ)
            for arm in arms:lines+=eq(name+'_'+arm,ref,typ)
    lines+='  check.return\n}\n'
    if not check:lines+=f'check.benchmark<@{tag}> @{tag}_bm\n'
    return lines


def post_case(out,rows,w16,random,arms,check):
    tag=f'post_{rows}_{w16}_{int(random)}';width=5120
    rng=np.random.default_rng(101+rows+w16)
    x=rng.uniform(-.8,.8,(rows,width)).astype(np.float32) if random else np.full((rows,width),.5,np.float32)
    x[:,1::2]*=-1
    y=x.copy();info=np.ones((rows,4),np.int32);info[::17,2]=0;live=info[:,2]!=0
    xn=x.copy();xn[live]=(x[live]+y[live]).astype(np.float32)
    expected_h=(xn.astype(np.float64)/np.sqrt(np.mean(xn.astype(np.float64)**2,axis=1,keepdims=True)+1e-6)*1.5).astype(np.float32)
    expected_h[~live]=37
    s,s8=scales(expected_h,live)
    q=fp8_bits((expected_h*s8[:,0,None]).astype(np.float32));q[~live]=31
    half=np.full_like(x,37,dtype=np.float16)
    if w16:half[live]=(expected_h[live]*s[live,0,None]).astype(np.float16)
    T=f'tensor<{rows}x5120xf32>';I=f'tensor<{rows}x4xi32>';Q=f'tensor<{rows}x5120xi8>';H=f'tensor<{rows}x5120xf16>';S=f'tensor<{rows}x2xf32>';W='tensor<5120xf32>'
    arrays=dict(y=y,info=info,wp=np.ones(width,np.float32),wn=np.full(width,1.5,np.float32),ex=xn,eh=expected_h,es=s,es8=s8,eq=q,ehalf=half)
    for arm in arms:arrays['x_'+arm]=x.copy()
    save(out,tag,arrays)
    lines=f'check.case public @{tag} {{\n  %r = check.literal value({rows}) : index\n  %flag = check.literal value({w16}) : index\n'+read(tag,'y',T)+read(tag,'info',I)+read(tag,'wp',W)+read(tag,'wn',W)
    for arm in arms:
        lines+=read(tag,'x_'+arm,T)
        for name,typ,value in [('h',T,'37.0'),('s',S,'42.0'),('half',H,'37.0'),('q',Q,'31'),('s8',S,'42.0')]:lines+=f'  %{name}_{arm} = check.generate.fill value({value}) : {typ}\n'
        lines+=f'  kernel.launch @post_{arm}[%r](%r, %info, %y, %wp, %x_{arm}, %wn, %h_{arm}, %s_{arm}, %half_{arm}, %q_{arm}, %s8_{arm}, %flag) : [index](index, {I}, {T}, {W}, {T}, {W}, {T}, {S}, {H}, {Q}, {S}, index)\n'
    if check:
        for name,typ in [('x',T),('h',T),('s',S),('s8',S),('half',H),('q',Q)]:
            ref='e'+name;lines+=read(tag,ref,typ)
            for arm in arms:
                if name=='h' and arm=='h32' and not w16 and rows%32==0:
                    lines+=f'  %dead_h = check.generate.fill value(37.0) : {T}\n'+eq('h_'+arm,'dead_h',T)
                elif not random or name in ('x','h'):lines+=eq(name+'_'+arm,ref,typ,approx=name=='h')
                if arm!='base' and not (name=='h' and arm=='h32' and not w16 and rows%32==0):lines+=eq(name+'_'+arm,name+'_base',typ)
    lines+='  check.return\n}\n'
    if not check:lines+=f'check.benchmark<@{tag}> @{tag}_bm\n'
    return lines


def negatives(out,defs):
    cases=[];kernels=[]
    def finish(text,old,new,assertion):
        text=text[:text.index('check.benchmark')]
        text=text.replace('@'+old,'@'+new)
        return text.replace('  check.return',assertion+'  check.return')
    text=row_case(out,5120,8,1,['held'],False)
    text='\n'.join(line for line in text.splitlines() if 'kernel.launch' not in line)+'\n'
    cases.append(finish(text,'row_5120_8_1','negative_row_no_half',read('row_5120_8_1','eh','tensor<8x5120xf16>')+eq('h_held','eh','tensor<8x5120xf16>',True)))
    bad=defs['held']['row_5120'].replace('row_5120_held','row_5120_scale_bad')
    bad=P._once(bad,'view.store %f8_scale,','view.store %zero_f,')
    kernels.append(bad)
    text=row_case(out,5120,8,0,['scale_bad'],False)
    cases.append(finish(text,'row_5120_8_0','negative_row_scale',read('row_5120_8_0','es8','tensor<8x2xf32>')+eq('s8_scale_bad','es8','tensor<8x2xf32>',True)))
    # A deliberately broken sidecar predicate omits required F16/tail output.
    bad=P._once(defs['h32']['post'],'scalar.ori %h32_tail, %h32_f16','scalar.andi %h32_tail, %h32_f16').replace('post_h32','post_h32_bad')
    kernels.append(bad)
    for rows,flag,label in [(256,1,'f16'),(257,0,'tail')]:
        tag=f'post_{rows}_{flag}_0';typ=f'tensor<{rows}x5120xf32>'
        text=post_case(out,rows,flag,False,['h32_bad'],False)
        cases.append(finish(text,tag,'negative_post_h32_'+label,read(tag,'eh',typ)+eq('h_h32_bad','eh',typ,True)))
    return HEAD+'\n'.join(kernels)+'\n'.join(cases)


def generate(root,out,variants):
    out.mkdir(parents=True,exist_ok=True);defs=kernels(root)
    tests=[]
    for width in (5120,6144,17408):
        for rows in (8,256):
            for flag in (0,1):tests.append(row_case(out,width,rows,flag,VARIANTS,True))
    for rows in (8,256,257):
        for flag in (0,1):
            for random in (False,True):tests.append(post_case(out,rows,flag,random,VARIANTS,True))
    all_defs='\n'.join(text for arm in VARIANTS for text in defs[arm].values())
    (out/'test.loom').write_text(HEAD+all_defs+'\n'.join(tests))
    for arm in variants:
        cases=[]
        for rows in (1024,2048):
            for width in (5120,6144,17408):cases.append(row_case(out,width,rows,0,[arm],False))
            cases.append(post_case(out,rows,0,True,[arm],False))
        filename='bench.loom' if arm=='base' else f'bench_{arm}.loom'
        (out/filename).write_text(HEAD+'\n'.join(defs[arm].values())+'\n'.join(cases))
    (out/'negative.loom').write_text(negatives(out,defs))
    (out/'manifest.json').write_text(json.dumps({'variants':list(VARIANTS),'positive_cases':len(tests),'strict_negative_cases':4,'timing_cases_per_arm':8,'model_contract':{'hidden':5120,'eps':1e-6,'post_norm':False},'kernel_source':'current gen_llm templates + gen_mxfp4s.f8_producer + mxfp4_producers helpers','limits':'No product wiring. Dead h32 only for explicitly proven paired-AB-FP8 epoch; WG160 requires GPU bit equality. Random postnorm quantized payloads compare old/new exactly; h uses independent f64 reference, constant cases additionally check every payload independently.'},indent=2)+'\n')


if __name__=='__main__':
    generate(ROOT,Path(sys.argv[2]),tuple(sys.argv[3:] or VARIANTS))
