#!/usr/bin/env python3
"""Bounded exact LDS lifetime experiment, full state and independent references."""
import importlib.util
import os
from pathlib import Path
import re
import sys
import shutil
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np

ROOT, OUT = map(lambda x: Path(x).resolve(), sys.argv[1:3])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT/'skills/loom-port-llm/starter'))
spec = importlib.util.spec_from_file_location('existing_gdn', ROOT/'.github/devlab/kbench/gdn_fused/gen_bench.py')
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
import gdn_fa_liveness as L
os.environ['GDN_FA_SHARE_LDS'] = '0'
os.environ.pop('GDN_FA_SHARE_LDS__RDNA4', None)
base = B.GG.gdn_kernels(B.CFG, B.EPS)
original = B.GF.phase_a(B.G, len(B.CFG['recurrent_layers']), B.LB, B.EPS, True, B.RATIO)
control = L.rewrite(original, overlap=False).replace('g_gdn_fa_rdna4', 'g_gdn_fa_rdna4_shared')
new = L.rewrite(original).replace('g_gdn_fa_rdna4', 'g_gdn_fa_rdna4_overlap')
head = 'amdgpu.target<gfx12-generic> @rdna4 {subgroup_size = 32}\n'


def names(text, suffix):
    return re.sub(r'%([A-Za-z_][A-Za-z_0-9]*)', lambda m: m.group(0)+suffix, text)


# Reuse the established independent f64 recurrence, convolution, output/state
# and history reference at fresh/continuing/short/partial mixed spans.
old_tests = '\n'.join(B.tests(str(OUT)))
# Distinct file paths make mutable old/new arenas unambiguously independent.
for path in OUT.glob('tf_*.npy'):
    if path.name.startswith(('tf_shared_', 'tf_overlap_')):
        continue
    for prefix in ['tf_shared_', 'tf_overlap_']:
        shutil.copyfile(path, OUT/path.name.replace('tf_', prefix, 1))
old_tests = re.sub(r'check\.expect\.close (actual\(.*\)) : (.*)$', r'check.expect.approximate \1 : \2, \2', old_tests, flags=re.M)
control_tests = old_tests.replace('@gdn_fused_spans', '@gdn_shared_spans').replace('@g_gdn_fa_rdna4[', '@g_gdn_fa_rdna4_shared[').replace('path("tf_', 'path("tf_shared_')
new_tests = old_tests.replace('@gdn_fused_spans', '@gdn_overlap_spans').replace('@g_gdn_fa_rdna4[', '@g_gdn_fa_rdna4_overlap[').replace('path("tf_', 'path("tf_overlap_')
t, ns = 293, 4
AT, MT, OT, WT = B.types(t, ns)
args = B.operands('tf', t, ns, AT, MT, OT, WT)
launch = B.launches(t, ns, AT, MT, OT, WT, 'fused')
exact = 'check.case public @exact_mixed_spans {\n'
for suffix in ['_base','_shared','_overlap']:
    call = launch if suffix=='_base' else launch.replace('@g_gdn_fa_rdna4[', '@g_gdn_fa_rdna4'+suffix+'[')
    arm_args = args if suffix=='_base' else args.replace('path("tf_', 'path("tf'+suffix+'_')
    exact += names(arm_args+call, suffix)
for var,typ in [('o',f'tensor<{t}x{B.VD}xf32>'),('arena',AT),('work',WT)]:
    for suffix in ['_shared','_overlap']:
        exact += f'  check.expect.equal actual(%{var}{suffix}) expected(%{var}_base) : {typ}\n'
exact += '  check.return\n}\n'
(OUT/'test.loom').write_text(head+base+control+new+old_tests+control_tests+new_tests+exact)
# Strict raw comparator failures in their own module, never real gates xfailed.
negative=[]
for var,typ in [('o',f'tensor<{t}x{B.VD}xf32>'),('arena',AT)]:
    negative.append(f'''check.case public @zero_{var}_control {{
{args}{launch.replace('@g_gdn_fa_rdna4[','@g_gdn_fa_rdna4_overlap[')}  %zero = check.generate.fill value(0.0) : {typ}
  check.expect.approximate actual(%{var}) expected(%zero) atol(0.0) rtol(0.0) nan(different) : {typ}, {typ}
  check.return
}}
''')
(OUT/'negative.loom').write_text(head+base+control+new+'\n'.join(negative))
# Phase A alone AND full A+B, so faster parallel work cannot hide serial cost.
for label in ['base','shared','overlap']:
    cases=[]
    for rows in [1024,2048]:
        # No f64 reference is computed for timing, and inputs match each arm.
        for case in B.bench(str(OUT), rows):
            if re.search(r'check.case public @(?:fa|fused)_'+str(rows)+r'_case',case):
                if label!='base':case=case.replace('@g_gdn_fa_rdna4[','@g_gdn_fa_rdna4_'+label+'[')
                cases.append(case)
    (OUT/('bench.loom' if label=='base' else 'bench_'+label+'.loom')).write_text(head+base+control+new+'\n'.join(cases))

