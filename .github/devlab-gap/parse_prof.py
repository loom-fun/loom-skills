"""Parse an IREE HAL .ireeprof (IRPF v2) bundle. Run: python3 -I parse_prof.py FILE.ireeprof OUT.pkl
Pure stdlib. Layouts from hrx-system runtime/src/iree/hal/{utils/profile_file.h,profile_schema.h}."""
import struct, sys, pickle, mmap
f = open(sys.argv[1], 'rb'); mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
magic, vmaj, vmin, hlen, flags, flen = struct.unpack_from('<IHHIIQ', mm, 0)
assert magic == 0x46505249, hex(magic)
end = flen or len(mm)
RH = '<QQQQQQQIIIII4xQQIHH'
assert struct.calcsize(RH) == 104
out = dict(devices=[], queues=[], functions={}, dispatch=[], qevent=[], qdev=[], host=[], clock=[], cmdbuf=[], ops={}, other={})
off = hlen
n = 0
while off < end:
    (rl, pl, sess, stream, ev, exe, cb, hl, ctl, nl, pdev, qord, cflags, dropped, status, rtype, rflags) = struct.unpack_from(RH, mm, off)
    ct = bytes(mm[off+hl:off+hl+ctl]).decode()
    po = off + hl + ctl + nl
    pay_end = po + pl
    kind = ct.rsplit('.', 1)[-1]
    n += 1
    if rtype == 2:
        out['other'][kind] = out['other'].get(kind, 0) + 1
        p = po
        while p < pay_end:
            rlen = struct.unpack_from('<I', mm, p)[0]
            if rlen == 0: break
            if kind == 'dispatch-events':
                (_, fl, eid, sub, cbid, exid, cidx, fo, wc0, wc1, wc2, ws0, ws1, ws2, st, en) = struct.unpack_from('<IIQQQQIIIIIIIIQQ', mm, p)
                out['dispatch'].append((pdev, qord, eid, sub, cbid, exid, cidx, fo, (wc0, wc1, wc2), (ws0, ws1, ws2), st, en))
            elif kind == 'queue-device-events':
                (_, typ, fl, _r, eid, sub, cbid, aid, sid, plen, pd, qo, opc, _r1, st, en) = struct.unpack_from('<IIIIQQQQQQIIIIQQ', mm, p)
                out['qdev'].append((typ, eid, sub, cbid, plen, pd, qo, opc, st, en))
            elif kind == 'queue-events':
                (_, typ, fl, strat, eid, ht, sub, cbid, aid, sid, pd, qo, wcnt, scnt, bcnt, opc, plen, rht) = struct.unpack_from('<IIIIQqQQQQIIIIIIQq', mm, p)
                out['qevent'].append((typ, strat, eid, ht, sub, cbid, plen, wcnt, scnt, bcnt, opc, rht, pd, qo))
            elif kind == 'host-execution-events':
                out['host'].append(bytes(mm[p:p+rlen]))
            elif kind == 'executable-functions':
                (_, fl, exid, fo, cbl, bc, pc, w0, w1, w2, nlen) = struct.unpack_from('<IIQIIIIIIII', mm, p)
                name = bytes(mm[p+rlen-nlen:p+rlen]).decode(errors='replace')
                out['functions'][(exid, fo)] = name
            elif kind == 'devices':
                (_, fl, pd, qc) = struct.unpack_from('<IIII', mm, p)
                uuid = bytes(mm[p+16:p+32]); freq = struct.unpack_from('<Q', mm, p+32)[0] if rlen >= 40 else 0
                out['devices'].append((pd, qc, freq))
            elif kind == 'clock-correlations':
                (_, fl, pd, _r, sid, tick, cpu, sysn, sysf, hb, he) = struct.unpack_from('<IIIIQQQQQqq', mm, p)
                out['clock'].append((pd, tick, cpu, hb, he))
            elif kind == 'command-operations':
                (_, typ, fl, ci, cbid, bo, bco, exid, fo, bcnt) = struct.unpack_from('<IIIIQIIQII', mm, p)
                out['ops'][(cbid, ci)] = (typ, exid, fo)
            p += rlen
    off += rl
out['n_records'] = n
pickle.dump(out, open(sys.argv[2], 'wb'))
print('records', n, {k: (len(v) if hasattr(v, '__len__') else v) for k, v in out.items() if k != 'other'}, out['other'])
print('devices', out['devices'])
