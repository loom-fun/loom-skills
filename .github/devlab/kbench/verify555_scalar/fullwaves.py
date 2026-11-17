"""Remap four waves to two head pairs and two split slots, preserving fallback."""
import re

def transform(s):
    s=s.replace('%mt_go = scalar.andi %mt_token_go, %hp_leader : i1','%mt_go = scalar.andi %sg_uniform, %sg_leader : i1')
    a=s.index('  scf.if %mt_go {\n');b=s.index('  kernel.return',a)
    m=s[a:b]
    m=re.sub(r'%split\b','%mt_split',m)
    setup='''    %mt_wave_pair = index.rem %wave, %c2 : index
        %mt_head_pair = index.mul %mt_wave_pair, %c2 : index
        %mt_kv_base = index.mul %kv_head, %cg : index
        %mt_head_base = index.add %mt_kv_base, %mt_head_pair : index
        %mt_split_half = index.div %wave, %c2 : index
        %mt_split_group = index.mul %split, %c2 : index
        %mt_split = index.add %mt_split_group, %mt_split_half : index
    '''
    m=m.replace('  scf.if %mt_go {\n','  scf.if %mt_go {\n'+setup,1)
    m=m.replace('= index.add %bounded_head, %mt_hc','= index.add %mt_head_base, %mt_hc')
    for j in range(4):
     m=m.replace(f'    scf.if %mt_act{j} {{\n',f'    scf.if %mt_act{j} {{\n      %mt_store_split{j} = index.assume %mt_split [range(%mt_split, 0, 63)] : index\n')
     m=m.replace(f'[%mg_t{j}, %mt_h{j}, %mt_split,',f'[%mg_t{j}, %mt_h{j}, %mt_store_split{j},')
    s=s[:a]+m+s[b:]
    return s
