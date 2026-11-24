"""RDNA3 one-query-head-per-wave prototype for the Strix Qwen G8 kernel suite."""
import re

from flash2_g6 import g6_kernel


def wave_kernel(cfg, qlds):
    """Adapt the classic per-page loop to gfx11's duplicated 16-element inputs and interleaved results."""
    kvh, heads, dim = cfg["kv_heads"], cfg["heads"], cfg["head_dim"]
    assert (kvh, heads, dim) == (2, 16, 256)
    body = g6_kernel(kvh, heads//kvh, dim, heads, dim**-.5, True, True,
                     cfg["layers"]-1, kvh*dim*2,
                     dict(qlds=qlds, qscale=0, fast=0, carry=0, tau=-1.0))
    body = body.replace("index.mul %lane_half, %c128", "index.mul %lane_half, %c0")
    body = re.sub(r"(%(?:qo|kco)(\d+) = index.constant )\d+", lambda m: m[1]+str(16*int(m[2])), body)
    body = re.sub(r"(range\(%(?:qn|kcn)\d+_v, 0, )248\)", r"\g<1>240)", body)
    body = body.replace("x32x8xf16>", "x32x16xf16>").replace("vector<8xf16>", "vector<16xf16>")
    body = re.sub(r"(%ql_bytes = index.constant )(\d+)", lambda m: m[1]+str(2*int(m[2])), body)
    body = "\n".join(line.replace("vector<8xf32>", "vector<16xf32>") if re.match(r"\s*%(?:qraw|qh)\d+ =",line) else line for line in body.splitlines())
    body = body.replace("%lane_h8_i32 = index.cast %lane_h8", "%lane_h8_i32 = index.cast %lane_half")
    body = body.replace("%one_row_i32 = scalar.constant 1", "%one_row_i32 = scalar.constant 2")
    constants = "\n".join(f"  %prob_i{i} = index.constant {i} : index" for i in range(8))
    body = body.replace("  %zero8 =", f"  %prob_high = index.cmp ne, %lane_half, %c0 : index\n{constants}\n  %zero8 =")

    def duplicate_probability(match):
        indent, suffix = match.group(1, 2)
        lines = []
        for i in range(8):
            lines += [f"%px{suffix}{i} = vector.extract %p{suffix}[%prob_i{i}] : vector<8xf32> -> f32",
                      f"%ps{suffix}{i}, %pu{suffix}{i} = kernel.subgroup.shuffle<xor> %px{suffix}{i}, %exchange, %width : f32, i32, i32"]
        partner = ", ".join(f"%ps{suffix}{i}" for i in range(8))
        lines += [f"%partner{suffix} = vector.from_elements {partner} : vector<8xf32>",
                  f"%even{suffix} = scf.select %prob_high, %partner{suffix}, %p{suffix} : vector<8xf32>",
                  f"%odd{suffix} = scf.select %prob_high, %p{suffix}, %partner{suffix} : vector<8xf32>",
                  f"%packed{suffix} = vector.interleave<0> %even{suffix}, %odd{suffix} : vector<8xf32>, vector<8xf32> -> vector<16xf32>",
                  f"%ph16{suffix} = vector.fptrunc %packed{suffix} : vector<16xf32> to vector<16xf16>"]
        return "\n".join(indent+line for line in lines)
    body, count = re.subn(r"^([ ]*)%ph16(\w+) = vector.fptrunc %p\2 : vector<8xf32> to vector<16xf16>", duplicate_probability, body, flags=re.M)
    assert count == 2
    return body
