import sys,re
src=open('/home/jfowers/work/fn/pkg/motif/linear/ggml/q8_0_bf16_matrix.loom').read()
i=src.index('template.def<@ggml.q8_0_f16.contract_k64>')
head,body=src[:i],src[i:]
v=sys.argv[1]
old='''        %half_codes = vector.sitofp %codes : vector<32xi8> to vector<32xf16>
        %scale_view'''
assert old in body
if v in('v2','v2n'):
    new='''        %sign = vector.splati -128 : vector<32xi8>
        %positive_codes = vector.xori %codes, %sign : vector<32xi8>
        %integers = vector.extui %positive_codes : vector<32xi8> to vector<32xi16>
        %exponent = vector.splati 25600 : vector<32xi16>
        %biased_bits = vector.ori %integers, %exponent : vector<32xi16>
        %biased = vector.bitcast %biased_bits : vector<32xi16> to vector<32xf16>
        %bias = vector.splatf -1152.0 : vector<32xf16>
        %half_codes = vector.addf %biased, %bias : vector<32xf16>
        %scale_view'''
    body=body.replace(old,new)
if v=='v2n':  # ablation: no input convert (timing only)
    o2='''        %wide = vector.extf %selected : vector<32xbf16> to vector<32xf32>
        %values = vector.fptrunc %wide : vector<32xf32> to vector<32xf16>
        vector.store %values, %input_stage_view[%local_token, %local_k] : vector<32xf16>'''
    assert o2 in body
    body=body.replace(o2,'''        %values = vector.bitcast %selected : vector<32xbf16> to vector<32xf16>
        vector.store %values, %input_stage_view[%local_token, %local_k] : vector<32xf16>''')
if v=='nodec': # ablation: no weight decode (timing only)
    body=body.replace(old,'''        %half_codes = vector.bitcast %codes : vector<32xi8> to vector<16xf16>
        %half_codes = vector.concat<0> %half_codes, %half_codes : vector<16xf16>, vector<16xf16> -> vector<32xf16>
        %scale_view''')
open(f'motif_{v}.loom','w').write(head+body)
