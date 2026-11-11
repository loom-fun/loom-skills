"""Read the on-demand part of per_layer_token_embd.weight (heads K..15) through the page cache. usage: prewarm_ple.py K SHARD..."""
import struct, sys
K = int(sys.argv[1]); def rd(f, fmt): n = struct.calcsize(fmt); return struct.unpack(fmt, f.read(n))
def s(f): (n,) = rd(f, "<Q"); return f.read(n).decode()
def val(f, t):
    if t == 8: return s(f)
    if t == 9:
        (et,) = rd(f, "<I"); (n,) = rd(f, "<Q"); return [val(f, et) for _ in range(n)]
    return rd(f, "<" + {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}[t])[0]
for path in sys.argv[2:]:
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF": continue
        ver, nt, nkv = rd(f, "<IQQ"); align = 32
        for _ in range(nkv):
            k = s(f); (t,) = rd(f, "<I"); v = val(f, t)
            if k == "general.alignment": align = v
        ts = []
        for _ in range(nt):
            nm = s(f); (nd,) = rd(f, "<I"); dims = rd(f, "<%dQ" % nd); ty, off = rd(f, "<IQ"); ts.append((nm, dims, ty, off))
        base = (f.tell() + align - 1) // align * align
        ts.sort(key=lambda x: x[3])
        for i, (nm, dims, ty, off) in enumerate(ts):
            if nm != "per_layer_token_embd.weight": continue
            end = ts[i + 1][3] if i + 1 < len(ts) else None
            if end is None:
                import os; end = os.path.getsize(path) - base
            lo = base + off + (end - off) * K // 16; hi = base + end
            print("prewarm", path, nm, dims, "bytes", hi - lo, flush=True)
            f.seek(lo); left = hi - lo
            while left > 0:
                b = f.read(min(left, 64 << 20))
                if not b: break
                left -= len(b)
            sys.exit(0)
print("tensor not found")
