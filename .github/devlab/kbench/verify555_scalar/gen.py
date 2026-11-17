"""Offline-only2head x2token query tile using shipped scalar partial arithmetic."""
import ast,importlib.util,os,re,sys,json,subprocess
from pathlib import Path
root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=True)
cfg=json.loads((root/'fleet/ggml-org-Qwen3-4B-loom/llm_config.json').read_text());cfg['tensors']={};(out/'cfg.json').write_text(json.dumps(cfg));os.environ['LLM_CONFIG']=str(out/'cfg.json');os.environ['LOOM_RDNA4']='0'
import types
sys.path.insert(0,str(root/'skills/loom-port-llm/starter'))
import attn_rules
ns=dict(re=re,attn_rules=attn_rules,UNROLL=16,ATTN_MT=4,ATTN_MT_FROM=1024,POS_BYTES=2048,QH=32,KVH=8,G=4,L=36,D=128,DL=4,GATE=False,GATE_ACT='sigmoid',SINKS=False,WINDOW=0,gen_draft=types.SimpleNamespace(enabled=lambda:True),seg_tokens=lambda:0,attention_twin=lambda *args:'')
tree=ast.parse((root/'skills/loom-port-llm/starter/gen_llm.py').read_text())
for n in tree.body:
 if isinstance(n,ast.FunctionDef) and n.name in ['listed','allreduce','multi_token_split','gqa_split','attention_text'] or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['COMMON','ALLRED_SETUP'] for t in n.targets):
  exec(compile(ast.Module(body=[n],type_ignores=[]),'current_scalar','exec'),ns)
g=types.SimpleNamespace(**ns)
# Shipped scalar reference, including GQA8keys/block handover at position4095.
base=g.attention_text(64,'g')[0]
common=g.COMMON;common=common[common.index('func.def inline @g_kv_offset'):];common=common[:common.index('\n}')+2].replace('index.constant 512 : offset','index.constant 2048 : offset')
a=base.index('kernel.def export("g_attention_combine")');ref=base[:a]
(out/'baseline.loom').write_text(common+'\n'+ref)
# Start from existing GQA baseline without the old span wrapper, to retain fallback.
g.attn_rules.ATTN_MT=1
raw=g.attention_text(64,'g')[0];raw=raw[:raw.index('kernel.def export("g_attention_combine")')]
# multi_token_split needs a combine-kernel marker? It slices split-end only; raw suffices.
tree=ast.parse((root/'skills/loom-port-llm/starter/gen_llm.py').read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='multi_token_split');source=ast.get_source_segment((root/'skills/loom-port-llm/starter/gen_llm.py').read_text(),f)
source=source.replace('index.constant {T} : index','index.constant {T // 2} : index').replace('%c{j}u : index','%c{j // 2}u : index').replace('scalar.constant {j} : i32','scalar.constant {j // 2} : i32')
# Gate leaders to the even head of each pair; odd waves suppress their duplicate work.
source=source.replace('%mt_go = scalar.andi %sg_uniform, %sg_leader : i1','%mt_token_go = scalar.andi %sg_uniform, %sg_leader : i1\n  %hp_rem = index.rem %wave, %c2 : index\n  %hp_leader = index.cmp eq, %hp_rem, %c0 : index\n  %mt_go = scalar.andi %mt_token_go, %hp_leader : i1')
source=source.replace('%mt_qr{j} = vector.load', '%mt_hc{j} = index.constant {j % 2} : index\n    %mt_hr{j} = index.add %bounded_head, %mt_hc{j} : index\n    %mt_h{j} = index.assume %mt_hr{j} [range(%mt_hr{j}, 0, {QH - 1})] : index\n    %mt_qr{j} = vector.load')
source=source.replace('[%mg_t{j}, %bounded_head,','[%mg_t{j}, %mt_h{j},')
# GQA's8key block is required for exact partial equality at32k.
source=source.replace('%mt_lo_q = index.div','%mt_cU = index.constant {U} : index\n    %mt_lo_q = index.div').replace('%cU','%mt_cU')
ns['ATTN_MT_FROM']=4095  # exact8-key partials only after shipped GQA handover
exec(compile(source,'scalar2x2','exec'),ns)
g.multi_token_split=ns['multi_token_split']
cand=g.multi_token_split(raw,'g',8,4,None)
(out/'candidate.loom').write_text(common+'\n'+cand)

# Reuse deterministic serving-shaped fixtures and explicit check.literal cases from the existing experiment.
# Baseline kernels below are regenerated from current shipped scalar sources, never the archived WMMA control.
fixture_dir=out/'fixtures';fixture_dir.mkdir(exist_ok=True)
subprocess.run([sys.executable,str(root/'.github/devlab/kbench/verify555/gen.py'),str(root),str(fixture_dir)],check=True)
for f in fixture_dir.glob('*.npy'):
 f.replace(out/f.name)
def partial_sizes(text):
 # Sibling fixture uses128splits; shipped decode uses64. Only flat partial buffers change.
 for t in [1,3,4,5,8]:
  for width in [128,2]:
   text=text.replace(f'tensor<{t*32*128*width}xf32>',f'tensor<{t*32*64*width}xf32>')
 return text
cases=(fixture_dir/'test.loom').read_text();cases=partial_sizes(cases[cases.index('check.case'):])
kernels={}
for v in ['baseline','candidate']:
 src=(out/f'{v}.loom').read_text().replace('g_attention_split','k_'+v)
 kernels[v]=src
 bench=(fixture_dir/f'bench_{v}.loom').read_text();bench=partial_sizes(bench[bench.index('check.case'):])
 (out/f'bench_{v}.loom').write_text(src+bench)
# Add a handover-straddling span. At4093 all64partial slots are active, retaining the42sentinel guard.
import numpy as np
info=np.zeros((4,4),np.int32);info[:,1]=4093+np.arange(4);info[:,2]=1
np.save(out/'handover_info.npy',info)
a=cases.index('check.case public @case_4');b=cases.index('check.case public @case_5',a)
handover=cases[a:b].replace('@case_4','@handover').replace('info4.npy','handover_info.npy')
c=kernels['candidate'];c=c[c.index('kernel.def export'):]
(out/'test.loom').write_text(kernels['baseline']+c+cases+handover)
