#!/usr/bin/env python3
# gen.py OUT.loom [COLS] : bench cases wrapping the q8 schedule at pp2048 (cap 2048, live 1838)
import sys
CAP=2048; LIVE=1838
COLS=int(sys.argv[2]) if len(sys.argv)>2 else 128
cases=[('zproj',2560,6144),('ssmout',6144,2560),('qsaq',2560,12288)]
if COLS==128: cases.append(('shdown',640,2560))
s=open('/home/jfowers/work/fn/dense/hdr.loom').read()
for name,K,N in cases:
    B=K//32
    ct=N//COLS
    s+=f'''
kernel.def target(@flash_next_q8_wave32) @k_{name}() {{
  %c1 = index.constant 1 : index
  %groups = index.constant {ct} : index
  %rowt = index.constant {CAP//128} : index
  %threads = index.constant 256 : index
  kernel.launch.config workgroups(%groups, %rowt, %c1) workgroup_size(%threads, %c1, %c1) : index
}} launch(%span: buffer, %input: buffer, %weight: buffer, %output: buffer) {{
  %cap = config.get @flash_next.token_count : index
  %k = index.constant {K} : index
  %n = index.constant {N} : index
  %rows = index.constant 128 : index
  %cols = index.constant {COLS} : index
  %wr = index.constant 4 : index
  %wc = index.constant 2 : index
  %depth = index.constant 2 : index
  template.apply<@flash_next.q8_schedule>(%cap, %k, %n, %rows, %cols, %wr, %wc, %depth, %span, %input, %weight, %output) : (index, index, index, index, index, index, index, index, buffer, buffer, buffer, buffer)
  kernel.return
}}
check.case public @{name} {{
  %seed = check.literal value(5) : i64
  %span = check.generate.fill value({LIVE}) : tensor<3xi32>
  %input = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{CAP}x{K}xbf16>
  %weight = check.generate.random.uniform seed(%seed) range(-100 to 100) : tensor<{N}x{B}x34xi8>
  %output = check.generate.fill value(0.0) : tensor<{CAP}x{N}xbf16>
  kernel.launch @k_{name}(%span, %input, %weight, %output) : (tensor<3xi32>, tensor<{CAP}x{K}xbf16>, tensor<{N}x{B}x34xi8>, tensor<{CAP}x{N}xbf16>)
  check.return
}}
check.benchmark<@{name}> @{name}_bm
'''
open(sys.argv[1],'w').write(s)
