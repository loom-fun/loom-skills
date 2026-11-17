"""Opt-in Q8 publication from the residual/RMSNorm producer's registers.

The existing Q8 ABI and operation order stay intact. Derive both pieces from
those generators so eps, post norms, and quantizer rounding cannot drift.
"""
import re


def kernel(norms, icodes, hidden):
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
