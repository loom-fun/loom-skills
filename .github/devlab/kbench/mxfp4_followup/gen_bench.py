#!/usr/bin/env python3
"""One bounded runner job: major exact-K matrix candidates plus GDN overlap."""
from pathlib import Path
import os
import subprocess
import sys
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
root,out=map(lambda x:Path(x).resolve(),sys.argv[1:3])
subprocess.run([sys.executable,str(root/'.github/devlab/kbench/gdn_liveness/gen_bench.py'),str(root),str(out)],check=True)
# Keep all data in the run directory. GDN's tf_/bNNN_ names and the matrix
# suite's check_/time_/pack_/producer_ names are disjoint.
for name in ['test','negative','bench','bench_shared','bench_overlap']:
    (out/(name+'.loom')).replace(out/('gdn_'+name+'.loom'))
subprocess.run([sys.executable,str(root/'.github/devlab/kbench/mxfp4_pipeline/gen_bench.py'),str(root),str(out)],check=True)
