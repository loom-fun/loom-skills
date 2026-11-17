"""Exact, opt-in LDS lifetime sharing for the gfx12 fused GDN phase A.

Q is dead after the KK/QK products; beta*V is first consumed after the solve.
A is dead after the last triangular-solve read; T is consumed after its barrier.
Keeping the padded layouts and every arithmetic expression permits those pairs
of disjoint lifetimes to share storage. No numerical approximation is introduced.
"""
import os
import re

from flash2_rdna4 import twin_env


def enabled():
    env = dict(os.environ, GDN_FA_SHARE_LDS=os.environ.get("GDN_FA_SHARE_LDS", "0"))
    return twin_env(env).get("GDN_FA_SHARE_LDS", "0").lower() not in ("", "0", "false", "no", "off")


def rewrite(text):
    """Move independent work across existing barriers, then alias dead buffers.

    Exact producer structure is required. A generator change that invalidates
    a lifetime assumption must fail closed instead of producing a racy kernel.
    """
    begin = '  %isv = index.cmp ult, %thread, %k128 : index'
    barrier = '  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)'
    assert text.count(begin) == 1
    a = text.index(begin)
    b = text.index(barrier, a)
    v = text[a:b]
    text = text[:a] + text[b:]
    solve = '  scf.if %tcn {\n    %cc = index.assume %thread [range(%thread, 0, 31)] : index'
    assert text.count(solve) == 1
    pos = text.index(solve)
    assert text[:pos].rstrip().endswith(barrier.strip())
    text = text[:pos] + v + text[pos:]
    # V stores now follow the barrier ending every Q read. The existing
    # solve->W/U barrier publishes V to all waves before their first V load.
    qv_alloc = '  %bv_buf = buffer.alloca<workgroup> align(16) %kh_bytes : buffer\n'
    assert text.count(qv_alloc) == 1
    assert text.count('%bv_view = buffer.view %bv_buf[') == 1
    text = text.replace(qv_alloc, '').replace('%bv_view = buffer.view %bv_buf[', '%bv_view = buffer.view %q_buf[')
    # Never overwrite A during its solve. Delay casts AND stores until all
    # 32 rows have been solved; every T float is already live through row31.
    stores = re.findall(r'    %T\d+h = scalar.fptrunc[^\n]*\n    view.store %T\d+h, %t_view[^\n]*\n', text)
    assert len(stores) == 32
    for block in stores:
        text = text.replace(block, '', 1)
    end = '  }\n' + barrier + '\n  %wcol ='
    assert text.count(end) == 1
    text = text.replace(end, ''.join(stores) + end, 1)
    at_alloc = '  %t_buf = buffer.alloca<workgroup> align(16) %t_bytes : buffer\n'
    assert text.count(at_alloc) == 1
    assert text.count('%t_view = buffer.view %t_buf[') == 1
    return text.replace(at_alloc, '').replace('%t_view = buffer.view %t_buf[', '%t_view = buffer.view %a_buf[')
