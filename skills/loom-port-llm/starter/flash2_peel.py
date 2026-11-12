"""Peels the last key tile out of a flash2 kernel's tile loop (used by gen_flash2.py and gen_flash2s.py).

The kernel text has one loop `scf.for %kt` over the key tiles whose body may enter the masked path (`%need<kb>`: a 16-key block
that holds masked keys, which runs once per query column with V zeroed at that column's masked keys). Only the tile of the
last key can hold such a block (the diagonal block); every tile before it is fully visible. Hence the loop runs all tiles but
the last with `%need<kb>` false (straight-line code, no masked code for the register allocator to plan around: one register
file serves the loop, which was measured at 10-14 % of Spark-X2.6 prefill when the masked code sat inside it), and a copy of
the body follows the loop for the last tile, with the masked path.
"""
import re

HEADER = re.compile(r"^(?P<ind> *)%m_final, %l_final, (?P<outs>[^=]*) = scf\.for %kt = \[%c0 to %key_tiles step %c1\]\((?P<args>.*)\) -> \((?P<types>.*)\) \{$")


def peel_last_tile(text):
    lines = text.split("\n")
    hi = next((i for i, ln in enumerate(lines) if HEADER.match(ln)), None)
    if hi is None:
        raise SystemExit("flash2_peel: no tile loop")
    m = HEADER.match(lines[hi])
    ind = m.group("ind")
    yi = next(i for i in range(hi + 1, len(lines)) if lines[i].lstrip().startswith("scf.yield %m_new, %l_new,"))
    assert lines[yi + 1].strip() == "}", lines[yi + 1]
    body = lines[hi + 1:yi]
    yield_line = lines[yi]
    outs = [o.strip() for o in m.group("outs").split(",")]          # %o0 ... %o<NF-1>
    args = m.group("args")
    acc_names = re.findall(r"(%a\d+) = ", args)
    assert len(acc_names) == len(outs)
    # hot copy: no block takes the masked path
    hot = [re.sub(r"^(\s*)(%need\d+) = .* : i1$", r"\1\2 = scalar.constant false : i1", ln) for ln in body]
    # peeled copy: every name it defines gets a suffix; the loop-carried values and %kt come from the hot loop
    defined = set()
    for ln in body:
        d = re.match(r"\s*(%\w+(?:, %\w+)*) = ", ln)
        if d:
            defined.update(re.findall(r"%\w+", d.group(1)))
        defined.update(re.findall(r"scf\.for (%\w+) = ", ln))
        if "scf.for" in ln or "scf.if" in ln:
            defined.update(re.findall(r"[(,] ?(%\w+) = ", ln))
    mp = {"%kt": "%kt_hot", "%m": "%m_hot", "%l": "%l_hot"}
    mp.update({a: o + "_hot" for a, o in zip(acc_names, outs)})
    mp.update({n: n + "_pt" for n in defined})
    sub = lambda t: re.sub(r"%\w+", lambda mm: mp.get(mm.group(0), mm.group(0)), t)
    peel = [sub(ln) for ln in body]
    out = lines[:hi]
    out.append(f"{ind}%kt_hot = index.sub %key_tiles, %c1 : index")
    out.append(f"{ind}%m_hot, %l_hot, " + ", ".join(o + "_hot" for o in outs) + f" = scf.for %kt = [%c0 to %kt_hot step %c1]({args}) -> ({m.group('types')}) {{")
    out += hot + [yield_line, lines[yi + 1]] + peel
    # the loop's results are the peeled tile's: %l_final -> %l_new_pt, %o<d> -> %on<d>_pt in what follows
    final = {"%l_final": "%l_new_pt", "%m_final": "%m_new_pt"}
    final.update({o: "%on" + o[2:] + "_pt" for o in outs})
    rest = "\n".join(lines[yi + 2:])
    rest = re.sub(r"%\w+", lambda mm: final.get(mm.group(0), mm.group(0)), rest)
    return "\n".join(out) + "\n" + rest
