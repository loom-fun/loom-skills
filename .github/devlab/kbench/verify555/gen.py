import ast,importlib.util,re,sys,subprocess
import numpy as np
from pathlib import Path
base=Path(sys.argv[1]);out=Path(sys.argv[2]);here=Path(__file__).parent
spec=importlib.util.spec_from_file_location('mma',here/'source_mma.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
t=ast.parse((base/'skills/loom-port-llm/starter/gen_llm.py').read_text()); common=next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='COMMON' for x in n.targets))
common=common[common.index('func.def inline @g_kv_offset'):];common=common[:common.index('\n}')+2].replace('index.constant 512 : offset','index.constant 2048 : offset')
split='''func.def inline @g_split_count(%visible: index) -> (index) {
 %c1 = index.constant 1 : index
 %c31 = index.constant 31 : index
 %c32 = index.constant 32 : index
 %cmax = index.constant 128 : index
 %p = index.add %visible, %c31 : index
 %v = index.div %p, %c32 : index
 %n = index.min %v, %cmax : index
 %m = index.max %n, %c1 : index
 func.return %m : index
}
'''
(out/'prior_full.loom').write_text(common+'\n'+split+mod.kernel(128,8,4,4,1024,2048,0.08838834764831845).replace('vector.constant ', 'vector.splatf '))

# Keep the archived baseline fixed; only the candidate changes its LDS padding.
variant=sys.argv[3] if len(sys.argv)>3 else 'original'
if variant != 'original':
 (out/'baseline_fixed.loom').write_text((out/'prior_full.loom').read_text())
 mod.DP8,mod.VTC={'v32':(136,32),'k128v36':(128,36)}[variant]
 (out/'prior_full.loom').write_text(common+'\n'+split+mod.kernel(128,8,4,4,1024,2048,0.08838834764831845).replace('vector.constant ','vector.splatf '))
subprocess.run([sys.executable,str(here/'stage_q.py'),str(out)],check=True)
if variant != 'original':
 (out/'prior_full.loom').write_text((out/'baseline_fixed.loom').read_text())
rng=np.random.default_rng(555)
# Deterministic inputs; exact partial comparison detects every arithmetic difference.
ctx=32832
q=rng.standard_normal((8,32,128)).astype(np.float32)
arena=rng.standard_normal((2,ctx,8,128)).astype(np.float16)
orig=np.zeros(10+ctx//16,np.int32);orig[8]=40;orig[10:]=np.arange(ctx//16)
for name,a in [('q',q),('arena',arena),('orig',orig)]:np.save(out/f'{name}.npy',a)
cases=[]
for t,p0 in [(1,32768),(3,32767),(4,32765),(5,32764),(8,32760)]:
 info=np.zeros((t,4),np.int32);info[:,1]=p0+np.arange(t);info[:,2]=1
 np.save(out/f'info{t}.npy',info)
 def case(which,compare=False):
  s=f"""check.case public @case_{t} {{
 %t = check.literal value({t}) : index
 %l = check.literal value(0) : index
 %sl = check.literal value(0) : i32
 %ctx = check.literal value({ctx}) : index
 %q = check.file.read.npy path("q.npy") : tensor<8x32x128xf32>
 %i = check.file.read.npy path("info{t}.npy") : tensor<{t}x4xi32>
 %orig = check.file.read.npy path("orig.npy") : tensor<{10+ctx//16}xi32>
 %arena = check.file.read.npy path("arena.npy") : tensor<2x{ctx}x8x128xf16>
"""
  for suffix in (['a','b'] if compare else ['a']):
   kernel='k_baseline' if suffix=='a' and compare else 'k_'+which
   sentinel=42.0 if suffix=='b' else 0.0
   s+=f""" %pa{suffix} = check.generate.fill value({sentinel}) : tensor<{t*32*128*128}xf32>
 %pm{suffix} = check.generate.fill value({sentinel}) : tensor<{t*32*128*2}xf32>
 kernel.launch @{kernel}[%t](%t, %l, %sl, %ctx, %ctx, %q, %i, %orig, %arena, %pa{suffix}, %pm{suffix}) : [index](index, index, i32, index, index, tensor<8x32x128xf32>, tensor<{t}x4xi32>, tensor<{10+ctx//16}xi32>, tensor<2x{ctx}x8x128xf16>, tensor<{t*32*128*128}xf32>, tensor<{t*32*128*2}xf32>)
"""
  if compare:
   for n,size in [('pa',t*32*128*128),('pm',t*32*128*2)]:s+=f' check.expect.equal actual(%{n}a) expected(%{n}b) : tensor<{size}xf32>\n'
  return s+' check.return\n}\n'
 cases.append(case('candidate',True))
 for variant,src in [('baseline','prior_full'),('candidate','stage_q')]:
  if t==4:
   text=(out/f'{src}.loom').read_text().replace('g_attention_split_mma','k_'+variant)
   (out/f'bench_{variant}.loom').write_text(text+case(variant)+f'check.benchmark<@case_{t}> @bench\n')
a=(out/'prior_full.loom').read_text().replace('g_attention_split_mma','k_baseline')
b=(out/'stage_q.loom').read_text();b=b[b.index('kernel.def export'):].replace('g_attention_split_mma','k_candidate')
(out/'test.loom').write_text(a+b+''.join(cases))
