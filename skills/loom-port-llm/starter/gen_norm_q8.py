"""Opt-in Q8 publication from the residual/RMSNorm producer's registers.

The existing Q8 ABI and operation order stay intact. Derive both pieces from
those generators so eps, post norms, and quantizer rounding cannot drift.
"""
import re


def _chunked_publication(hidden, size):
    """Keep fewer conversion intermediates live; integer code sums remain exact."""
    groups = hidden // 32
    n=32//size
    body='''    %q8_a = vector.absf %hv : vector<32xf32>
    %q8_amax = vector.reduce<maxnumf> %q8_a, %q8_zero_f : vector<32xf32>, f32
    %q8_d = scalar.divf %q8_amax, %q8_c127 : f32
    %q8_nz = scalar.cmpf one, %q8_amax, %q8_zero_f : f32
'''
    body+=f'    %q8_dv = vector.splat %q8_d : vector<{size}xf32>\n    %q8_nzv = vector.splat %q8_nz : vector<{size}xi1>\n    %q8_z = vector.splatf 0.0 : vector<{size}xf32>\n    %q8_ones = vector.splati 1 : vector<{size}xi8>\n    %q8_izero = vector.splati 0 : vector<{size//4}xi32>\n'
    for part in range(n):
        for j in range(size):body+=f'    %q8_e{part}_{j} = vector.extract %hv[{part*size+j}] : vector<32xf32> -> f32\n'
        elems=', '.join(f'%q8_e{part}_{j}' for j in range(size))
        body+=f'''    %q8_v{part} = vector.from_elements {elems} : vector<{size}xf32>
    %q8_sc{part} = vector.divf %q8_v{part}, %q8_dv : vector<{size}xf32>
    %q8_rd{part} = vector.roundf %q8_sc{part} : vector<{size}xf32>
    %q8_rs{part} = vector.select %q8_nzv, %q8_rd{part}, %q8_z : vector<{size}xf32>
    %q8_q{part} = vector.fptosi %q8_rs{part} : vector<{size}xf32> to vector<{size}xi8>
    %q8_qw{part} = vector.bitcast %q8_q{part} : vector<{size}xi8> to vector<{size//4}xi32>
    %q8_c{part} = index.constant {part*size//4} : index
    vector.store %q8_qw{part}, %q8_q_view[%token, %bl, %q8_c{part}] : vector<{size//4}xi32>, view<[%tokens]x{groups}x8xi32>
    %q8_qs{part} = vector.dot4i<s8s8> %q8_q{part}, %q8_ones, %q8_izero : vector<{size}xi8>, vector<{size}xi8>, vector<{size//4}xi32>
    %q8_sum{part} = vector.reduce<addi> %q8_qs{part}, %q8_zero_i : vector<{size//4}xi32>, i32
'''
    acc='%q8_sum0'
    for part in range(1,n):
        body+=f'    %q8_add{part} = scalar.addi {acc}, %q8_sum{part} : i32\n';acc=f'%q8_add{part}'
    body+=f'''    %q8_qsf = scalar.sitofp {acc} : i32 to f32
    %q8_dh = scalar.fptrunc %q8_d : f32 to f16
    %q8_da = scalar.extf %q8_dh : f16 to f32
    %q8_meta = vector.from_elements %q8_da, %q8_qsf : vector<2xf32>
    vector.store %q8_meta, %q8_m_view[%token, %bl, %c0] : vector<2xf32>, view<[%tokens]x{groups}x2xf32>
'''
    return body


def kernel(norms, icodes, hidden, publication_chunk=32):
    assert publication_chunk in (8, 16, 32)
    assert hidden % 32 == 0 and 32 <= hidden <= 8192
    norm = re.search(r'kernel.def export\("g_post_add_norm"\).*?\n}\n', norms, re.S).group()
    quant = icodes[icodes.index('kernel.def export("g_quant_q8")'):]
    # Keep the exact quantization sequence, using the norm's live vector.
    body = quant[quant.index('    %a = vector.absf %v'):quant.index('\n  }\n  kernel.return')]
    bound = {'%v': '%hv', '%row': '%token', '%g': '%bl', '%c0': '%c0'}
    body = re.sub(r'%\w+', lambda m: bound.get(m[0], '%q8_' + m[0][1:]), body)
    groups = hidden // 32
    setup = f'''    %q8_zero = index.constant 0 : offset
    %q8_row_bytes = index.constant {hidden} : offset
    %q8_meta_off = index.scale %tokens, %q8_row_bytes : index, offset -> offset
    %q8_q_view = buffer.view %hq[%q8_zero] : buffer -> view<[%tokens]x{groups}x8xi32>
    %q8_m_view = buffer.view %hq[%q8_meta_off] : buffer -> view<[%tokens]x{groups}x2xf32>
    %q8_zero_f = scalar.constant 0.0 : f32
    %q8_c127 = scalar.constant 127.0 : f32
    %q8_zero_i = scalar.constant 0 : i32
    %q8_zero8i = vector.splati 0 : vector<8xi32>
    %q8_ones8 = vector.splati 1 : vector<32xi8>
    %q8_zero32 = vector.splatf 0.0 : vector<32xf32>
'''
    # The original quantizer's dynamic view dimensions become the producer's
    # static groups and token count; this changes no address or stored value.
    body = body.replace('[%q8_rows]', '[%tokens]').replace('[%q8_groups]', str(groups))
    if publication_chunk != 32:
        body = _chunked_publication(hidden, publication_chunk)
    norm = norm.replace('g_post_add_norm', 'g_post_add_norm_q8')
    norm = norm.replace('%h: buffer)', '%h: buffer, %hq: buffer)')
    anchor = f'    vector.store %hv, %h_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x{groups}x32xf32>\n'
    assert norm.count(anchor) == 1
    return norm.replace(anchor, anchor + setup + body + '\n')


def wire(program):
    """Only fuse an adjacent producer/quantizer pair with identical rows and H.

    Run after prefill row-scale fusion: its existing branch keeps prefill
    unchanged, while the decode else branch presents this exact pair.
    """
    pat = re.compile(
        r'(?P<i> *)kernel.launch @g_post_add_norm\[(?P<t>%\w+)\]\((?P=t), (?P<a>[^\n]+), (?P<h>%\w+)\) : \[index\]\(index, (?P<ty>[^\n]+), buffer\)\n'
        r' *(?:\/\/[^\n]*\n *)?kernel.launch @g_quant_q8\[(?P=t), %nH\]\((?P=t), %nH, (?P=h), (?P<q>%\w+)\) : \[index, index\]\(index, index, buffer, buffer\)')
    def replace(m):
        return (f'{m["i"]}kernel.launch @g_post_add_norm_q8[{m["t"]}]({m["t"]}, {m["a"]}, {m["h"]}, {m["q"]})'
                f' : [index](index, {m["ty"]}, buffer, buffer)')
    return pat.subn(replace, program)
