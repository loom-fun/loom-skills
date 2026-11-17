"""Generate a whole-model-sized split-attention bench and exact A/B cases from CPU artifacts."""
from pathlib import Path
import re

import sys
base,interval,output=sys.argv[1:4]
OUT=Path(output);OUT.mkdir(parents=True,exist_ok=True)
def fullblock(src):
    a=src.index('kernel.def export("g_attention_split")')
    b=src.index('  kernel.return\n}',a)+len('  kernel.return\n}')
    k=src[a:b]
    mask_pattern=r'(?m)^      %ok\d+(?:_(?:width|delta))? = (?:index.sub|index.cmp ult)[^\n]*\n'
    masks=re.findall(mask_pattern,k)
    assert len(masks)==48,len(masks)
    k=re.sub(mask_pattern,'',k)
    needle='    %m_final, %l_final, %acc_final = scf.for %j'
    pre='''    %full_pad = index.add %lo_abs, %c15 : index
        %full_lq = index.div %full_pad, %cU : index
        %full_lo = index.mul %full_lq, %cU : index
        %full_hq = index.div %hi_abs, %cU : index
        %full_hi0 = index.mul %full_hq, %cU : index
        %full_hi = index.max %full_hi0, %full_lo : index
        %full_width = index.sub %full_hi, %full_lo : index
    '''
    assert k.count(needle)==1;k=k.replace(needle,pre+needle)
    needle='      %kb_off = func.call @g_kv_offset'
    k=k.replace(needle,'''      %full_delta = index.sub %j, %full_lo : index
          %full = index.cmp ult, %full_delta, %full_width : index
    '''+needle,1)
    pattern=r'(?m)^      %s\d+ = scf.select %ok\d+, %sr\d+, %neg_big : f32\n'
    scores=re.findall(pattern,k);assert len(scores)==16
    out=', '.join(f'%s{i}' for i in range(16));types=', '.join(['f32']*16)
    branch=f'      {out} = scf.if %full -> ({types}) {{\n'
    branch+='        scf.yield '+', '.join(f'%sr{i}' for i in range(16))+f' : {types}\n'
    branch+='      } else {\n'+''.join('  '+line for line in masks)
    branch+=''.join('  '+line.replace(f'%s{i} =',f'%edge_s{i} =') for i,line in enumerate(scores))
    branch+='        scf.yield '+', '.join(f'%edge_s{i}' for i in range(16))+f' : {types}\n      }}\n'
    k=k.replace(''.join(scores),branch)
    assert k.count('scf.if %full')==1
    return src[:a]+k+src[b:]

sources={'base':Path(base).read_text(),'interval':Path(interval).read_text()}
sources['fullblock']=fullblock(sources['interval'])

def extract(src, start, end):
    i=src.index(start);j=src.index(end,i)+len(end);return src[i:j]+'\n'

helpers='\n'.join(extract(sources['base'],f'func.def inline @{name}', '\n}') for name in ('g_kv_offset','g_split_count'))
kernels={}
for name,src in sources.items():
    k=extract(src,'kernel.def export("g_attention_split")','  kernel.return\n}')
    kernels[name]=k.replace('g_attention_split', f'k_{name}')
AR=f'tensor<{24*2*32768*2*64}xf16>'
PA='tensor<57344xf32>';PM='tensor<1792xf32>';OT='tensor<1x896xf32>';IT='tensor<1x4xi32>';RT='tensor<2058xi32>'
TYPES=f'[index](index, index, i32, index, index, {OT}, {IT}, {RT}, {AR}, {PA}, {PM})'

def setup(pos):
    return f'''  %t = check.literal value(1) : index
  %sliding = check.literal value(0) : i32
  %ctx = check.literal value(32768) : index
  %pool = check.literal value(32768) : index
  %q = check.generate.fill value(0.02) : {OT}
  %info = check.file.read.npy path("info_{pos}.npy") : {IT}
  %orig = check.file.read.npy path("orig.npy") : {RT}
  %arena = check.generate.fill value(0.01) : {AR}
'''

def launch(name, layer, acc, ml):
    return f'  kernel.launch @k_{name}[%t](%t, %layer{layer}, %sliding, %ctx, %pool, %q, %info, %orig, %arena, {acc}, {ml}) : {TYPES}\n'

for name in kernels:
    s=helpers+'\n'+kernels[name]+'\ncheck.case public @model24_long {\n'+setup(32000)
    s+=f'  %acc = check.generate.fill value(0.0) : {PA}\n  %ml = check.generate.fill value(0.0) : {PM}\n'
    for layer in range(24):
        s+=f'  %layer{layer} = check.literal value({layer}) : index\n'+launch(name,layer,'%acc','%ml')
    s+='  check.return\n}\ncheck.benchmark<@model24_long> @model24_long_bm\n'
    (OUT/('bench.loom' if name=='base' else f'bench_{name}.loom')).write_text(s)

tests=helpers+'\n'+'\n'.join(kernels.values())
positions=(0,1,14,15,16,31,32,63,64,1024,31999,32000,32766)
for pos in positions:
    tests+=f'\ncheck.case public @mask_{pos} {{\n'+setup(pos)+'  %layer0 = check.literal value(0) : index\n'
    tests+=f'  %expected = check.file.read.npy path("expected_{pos}.npy") : {PA}\n'
    for name in kernels:
        acc,ml=f'%acc_{name}',f'%ml_{name}'
        tests+=f'  {acc} = check.generate.fill value(0.0) : {PA}\n  {ml} = check.generate.fill value(0.0) : {PM}\n'
        tests+=launch(name,0,acc,ml)
        tests+=f'  check.expect.approximate actual({acc}) expected(%expected) atol(1.0e-5) rtol(1.0e-4) nan(different) : {PA}, {PA}\n'
        if name!='base':
            for value,typ in (('acc',PA),('ml',PM)):
                tests+=f'  check.expect.approximate actual(%{value}_{name}) expected(%{value}_base) atol(0.0) rtol(0.0) nan(different) : {typ}, {typ}\n'
    tests+='  check.return\n}\n'
(OUT/'test.loom').write_text(tests)
# This small generator is run by the existing DevLab kbench generator hook.
(OUT/'generate_data.py').write_text('''from pathlib import Path
import sys
import numpy as np
out=Path(sys.argv[2])
orig=np.zeros(2058,np.int32);orig[8]=40;orig[10:]=np.arange(2048)
np.save(out/'orig.npy',orig)
for pos in '''+repr(positions)+''':
 info=np.array([[0,pos,1,0]],np.int32);np.save(out/f'info_{pos}.npy',info)
 visible=pos+1;s=min((visible+31)//32,64);per=(visible+s-1)//s
 expected=np.zeros((1,14,64,32,2),np.float32)
 for split in range(s):
  count=max(0,min((split+1)*per,visible)-min(split*per,visible))
  value=np.float32(0)
  for _ in range(count):value=np.float32(value+np.float32(np.float16(0.01)))
  expected[:,:,split,:,:]=value
 np.save(out/f'expected_{pos}.npy',expected.reshape(-1))
''')
print(OUT)
