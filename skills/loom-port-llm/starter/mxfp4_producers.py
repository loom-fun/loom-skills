"""Experimental exact MXFP4 operand-producer schedules; defaults unchanged.

Transform the actual generated contiguous FP8 twins, retaining their ABI and
rounding. These helpers are shared by kernel experiments and any later reviewed
product wiring. A helper emits code, not a claim of numeric or speed validation.
"""
import re


def _once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"producer structure changed: expected exactly one {old!r}")
    return text.replace(old, new, 1)


def rename(text, name):
    old = re.search(r'export\("(\w+)"\) @(\w+)\(', text)
    if old is None or old[1] != old[2]:
        raise ValueError('expected one named kernel')
    return text.replace(old[1], name)


def guard_half_arithmetic(text):
    """Move only the existing F16 multiply/cast into its existing store guard."""
    if 'g_row_scale_f16' in text:
        arithmetic = ('    %sv = vector.mulf %v, %scale4 : vector<4xf32>\n'
                      '    %hv = vector.fptrunc %sv : vector<4xf32> to vector<4xf16>\n')
    elif 'g_post_add_norm_rs' in text:
        arithmetic = re.search(r'    %hsv = vector\.mulf %\w+, %scale32 : vector<32xf32>\n'
                               r'    %h16v = vector\.fptrunc %hsv : vector<32xf32> to vector<32xf16>\n', text)
        if arithmetic is None:
            raise ValueError('unsupported postnorm half arithmetic')
        arithmetic = arithmetic[0]
    else:
        raise ValueError('unsupported producer')
    text = _once(text, arithmetic, '')
    return _once(text, '    scf.if %w16_on {\n',
                 '    scf.if %w16_on {\n' + ''.join('  '+line+'\n' for line in arithmetic.splitlines()))


def fixed_width(text, width):
    if width not in (5120, 6144, 17408):
        raise ValueError('unpriced row-scale width')
    return _once(text, 'range(%k_arg, 256, 65536), multiple_of(%k_arg, 256)',
                 f'range(%k_arg, {width}, {width}), multiple_of(%k_arg, 256)')


def _block_end(text, start):
    level = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            level += 1
        elif text[i] == '}':
            level -= 1
            if level == 0:
                return i+1
    raise ValueError('unbalanced producer')


def hold_row_inputs(text, width):
    """One global input read: keep each lane's original values across max reduce.

    Only actual fixed K controls are admitted. All scalar/vector arithmetic and
    rounding after the reduction stays in original order. Padded/dead row guards
    stay outside this transform. Native resource pressure must be priced per K.
    """
    text = fixed_width(text, width)
    first = text.index('  %local = scf.for %c = [%lane to %k4 step %c256]')
    end = text.index('\n  %maximum =', first)
    code = '  %held_zero = vector.splatf 0.0 : vector<4xf32>\n'
    previous = '%zero_f'
    count = (width+1023)//1024
    for j in range(count):
        code += f'''  %held_c{j} = index.constant {256*j} : index
  %held_i{j} = index.add %lane, %held_c{j} : index
  %held_on{j} = index.cmp ult, %held_i{j}, %k4 : index
  %held_v{j} = scf.if %held_on{j} -> (vector<4xf32>) {{
    %held_b{j} = index.assume %held_i{j} [lt(%held_i{j}, %k4)] : index
    %held_x{j} = vector.load %x4_view[%row, %held_b{j}, %c0] : view<[%rows]x[%k4]x4xf32> -> vector<4xf32>
    scf.yield %held_x{j} : vector<4xf32>
  }} else {{
    scf.yield %held_zero : vector<4xf32>
  }}
  %held_a{j} = vector.absf %held_v{j} : vector<4xf32>
  %held_m{j} = vector.reduce<maxnumf> %held_a{j}, {previous} : vector<4xf32>, f32
'''
        previous = f'%held_m{j}'
    text = text[:first]+code+text[end:]
    text = _once(text, 'kernel.workgroup.reduce<maxnumf> %local',
                 'kernel.workgroup.reduce<maxnumf> '+previous)
    second = text.index('  scf.for %c = [%lane to %k4 step %c256]')
    start = text.index('{', second)
    end = _block_end(text, start)
    body = text[start+1:end-1]
    load = ('\n    %bc = index.assume %c [lt(%c, %k4)] : index\n'
            '    %v_raw2 = vector.load %x4_view[%row, %bc, %c0] : view<[%rows]x[%k4]x4xf32> -> vector<4xf32>\n'
            '    %v = scf.select %row_live, %v_raw2, %zero4 : vector<4xf32>\n')
    body = _once(body, load, '\n')
    blocks = []
    for j in range(count):
        chunk = body
        for name in sorted(set(re.findall(r'(%\w+) = ', chunk)), key=len, reverse=True):
            chunk = re.sub(re.escape(name)+r'\b', name+f'_hs{j}', chunk)
        chunk = re.sub(r'%v\b', f'%held_v{j}', chunk)
        chunk = re.sub(r'%bc\b', f'%held_out{j}', chunk)
        blocks.append(f'  scf.if %held_on{j} {{\n    %held_out{j} = index.assume %held_i{j} [lt(%held_i{j}, %k4)] : index'+chunk+'\n  }')
    return text[:second]+'\n'.join(blocks)+text[end:]


def postnorm_wg160(text):
    """Remove three zero-only waves at H5120; GPU reduction equality is gated."""
    if 'view<[%tokens]x160x32xf32>' not in text:
        raise ValueError('WG160 is only defined for H5120')
    text = _once(text, '%c256 = index.constant 256 : index', '%c256 = index.constant 160 : index')
    return _once(text, '%lane_id [range(%lane_id, 0, 255)]', '%lane_id [range(%lane_id, 0, 159)]')


def postnorm_dead_h32(text, *, paired_ab_fp8):
    """Skip only the H5120 F32 sidecar in complete paired-AB-FP8 epochs.

    Caller must independently prove every actual reader uses the FP8 plane in
    this epoch, and that final h32 is overwritten before public/head output.
    No product caller enables this helper by default. Decode/F16/tail epochs
    retain the store. Merely having an h8 wrapper argument is not reader proof.
    """
    if paired_ab_fp8 is not True:
        raise ValueError('requires proven paired-AB-FP8 readers')
    store = '    vector.store %hv, %h_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x160x32xf32>\n'
    return _once(text, store, '''    %h32_zero = index.constant 0 : index
    %h32_c32 = index.constant 32 : index
    %h32_rem = index.rem %tokens, %h32_c32 : index
    %h32_tail = index.cmp ne, %h32_rem, %h32_zero : index
    %h32_f16 = index.cmp ne, %w16, %h32_zero : index
    %h32_needed = scalar.ori %h32_tail, %h32_f16 : i1
    scf.if %h32_needed {
'''+store+'    }\n')
