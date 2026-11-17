from pathlib import Path
import re,sys
p=Path(sys.argv[1]);s=(p/'prior_full.loom').read_text()
a=s.index('    %qc0 =');b=s.index('    %zero8 =',a);q=s[a:b]
q=re.sub(r'    %qdep_e =.*?    %qdep =[^\n]*\n','',q,flags=re.S)
q=re.sub(r'    %qcr(\d+) =[^\n]*\n    %qcl\1 =[^\n]*\n','',q);q=re.sub(r'%qcl(\d+)',r'%qc\1',q)
q=re.sub(r'    %qf(\d+) =[^\n]*',lambda m:f'    vector.store %qh{m[1]}, %qs[%lane_col, %qc{m[1]}, %c0] : vector<16xf16>, view<16x8x16xf16>',q)
q='''    %qs_bytes = index.constant 4096 : offset
    %qs_b = buffer.alloca<workgroup> align(16) %qs_bytes : buffer
    %qs = buffer.view %qs_b[%zero] : buffer -> view<16x8x16xf16>
    %qs_wave = index.cmp eq, %wave, %c0 : index
    %qs_lane = index.cmp ult, %lane, %c16 : index
    %qs_do = scalar.andi %qs_wave, %qs_lane : i1
    scf.if %qs_do {
'''+q+'''    }
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
'''
s=s[:a]+q+s[b:]
# Constants formerly defined in scoped producer must be visible for consumers.
for c in range(8):
 s=s.replace(f'    %qc{c} = index.constant {c} : index\n','')
for kb in range(2):
 for c in range(8):
  pat=f'        %s{kb}_{c} = vector.mma '
  setup=f'''        %qsc{kb}_{c} = index.constant {c} : index
        %qsl{kb}_{c} = vector.load %qs[%lane_col, %qsc{kb}_{c}, %c0] : view<16x8x16xf16> -> vector<16xf16>
        %qsf{kb}_{c} = vector.fragment<rhs> %qsl{kb}_{c} shape [%c16, %c16] : vector<16xf16>
'''
  s=s.replace(pat,setup+pat).replace(f'%kl{kb}_{c}, %qf{c},',f'%kl{kb}_{c}, %qsf{kb}_{c},')
# producer chunk constants are scoped.
s=s.replace('    scf.if %qs_do {','    scf.if %qs_do {\n'+''.join(f'    %qc{c} = index.constant {c} : index\n' for c in range(8)))
(p/'stage_q.loom').write_text(s)
