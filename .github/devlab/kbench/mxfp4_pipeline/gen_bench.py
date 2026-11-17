#!/usr/bin/env python3
"""Current-SDK exact-K pipeline, selectively using fragment-order A storage."""
import os
from pathlib import Path
import re
import subprocess
import sys
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
os.environ['GEMM_MXFP4S_A_TILED']='0'
os.environ.pop('GEMM_MXFP4S_A_TILED__RDNA4',None)
ROOT,OUT=map(lambda x:Path(x).resolve(),sys.argv[1:3])
# Existing independent f64/nonzero inputs, exact comparisons, producer payload
# and strict wrong-layout/zero-output controls. It creates real1024/2048 shapes.
subprocess.run([sys.executable,str(ROOT/'.github/devlab/kbench/mxfp4_atiled/gen_bench.py'),str(ROOT),str(OUT)],check=True)
sys.path.insert(0,str(ROOT/'skills/loom-port-llm/starter'))
import gen_mxfp4s as G
import mxfp4s_atiled as A


def replace_kernels(text,which,tiled):
    targets=re.findall(r'kernel.def[^\n]*export\("([^\"]+)"\)',text)
    changed=0
    for name in targets:
        if not which(name):continue
        mode='swiglu' if 'swiglu' in name else 'store'
        # Original SWP is tested on both modes. The packed compound only spends
        # extra registers on store; packed SWP SwiGLU drops a residency tier.
        swp=mode=='store' or not tiled
        generated=G.gemm_mxfp4s_f8_m256(mode,swp=swp)
        if tiled:generated=A.gemm(generated)
        old=f'g_gemm_mxfp4s_{mode}_f8_m256_rdna4'
        generated=generated.replace(old,name)
        at=text.index(f'export("{name}")')
        begin=text.rfind('kernel.def',0,at)
        end=text.index('  kernel.return\n}\n',begin)+len('  kernel.return\n}\n')
        text=text[:begin]+generated+text[end:];changed+=1
    assert changed, 'no matching matrix kernels'
    return text


base=(OUT/'bench.loom').read_text()
packed=(OUT/'bench_atiled.loom').read_text()
(OUT/'bench_swp.loom').write_text(replace_kernels(base,lambda n:n.startswith(('k_store','k_swiglu')),False))
(OUT/'bench_packed_swp.loom').write_text(replace_kernels(packed,lambda n:n.startswith(('k_store','k_swiglu')),True))
original_tests=(OUT/'test.loom').read_text()
# Plain pipeline reads contiguous X for both exact-comparison arms. Different
# file paths remain unnecessary for immutable inputs; outputs are independent.
plain=replace_kernels(original_tests,lambda n:n.startswith('new_'),False)
plain=re.sub(r'path\("([^\"]+)_xt.npy"\)',r'path("\1_x.npy")',plain)
(OUT/'test_swp.loom').write_text(plain)
(OUT/'test_packed_swp.loom').write_text(replace_kernels(original_tests,lambda n:n.startswith('new_'),True))
negative=(OUT/'negative.loom').read_text()
(OUT/'negative.loom').write_text(replace_kernels(negative,lambda n:n=='negative_store',True))
# The standalone losing A1 consumer is intentionally not timed again. Packed
# producer costs stay in the compound timing file and are priced per width.
