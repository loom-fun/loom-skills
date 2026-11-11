#!/usr/bin/env python3
# gen.py OUT.loom : bench cases for flash_next_q8_matrix_256 at pp2048 (cap 2048, live 1838)
import sys
CAP=2048; LIVE=1838
cases=[('zproj',2560,6144),('ssmout',6144,2560),('qsaq',2560,12288),('shdown',640,2560)]
s=open('/home/jfowers/work/fn/dense/hdr.loom').read()
for name,K,N in cases:
    B=K//32
    # geometry like flash_next_q8_geometry for cap 2048
    import os
    if N<=512: rows=cols=64; wr,wc=4,2
    elif N<=2560: rows=cols=128; wr,wc=4,2
    else: rows=cols=128; wr,wc=4,2
    import os
    if os.environ.get('COLS'): cols=int(os.environ['COLS'])
    s+=f'''
check.case public @{name} {{
  %seed = check.literal value(5) : i64
  %enabled = check.literal value(true) : i1
  %span = check.generate.fill value({LIVE}) : tensor<3xi32>
  %input = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{CAP}x{K}xbf16>
  %weight = check.generate.random.uniform seed(%seed) range(-100 to 100) : tensor<{N}x{B}x34xi8>
  %output = check.generate.fill value(0.0) : tensor<{CAP}x{N}xbf16>
  %cap = check.literal value({CAP}) : index
  %n = check.literal value({N}) : index
  %k = check.literal value({K}) : index
  %rows = check.literal value({rows}) : index
  %cols = check.literal value({cols}) : index
  %wr = check.literal value({wr}) : index
  %wc = check.literal value({wc}) : index
  %depth = check.literal value(2) : index
  kernel.launch @flash_next_q8_matrix_256[%enabled, %cap, %n, %rows, %cols](%k, %n, %rows, %cols, %wr, %wc, %depth, %span, %input, %weight, %output) : [i1, index, index, index, index](index, index, index, index, index, index, index, tensor<3xi32>, tensor<{CAP}x{K}xbf16>, tensor<{N}x{B}x34xi8>, tensor<{CAP}x{N}xbf16>)
  check.return
}}
check.benchmark<@{name}> @{name}_bm
'''
open(sys.argv[1],'w').write(s)
