"""Minimal reader of generated Loom program text, for the sampled-testing tools: split a program file into top-level blocks
and build a per-GPU-family view (gfx11 | gfx12) in which two trees can be compared kernel by kernel.

This is the stand-in used until fleet/diff_fleet.py (engineer "fleet") lands; sample_plan.py reads its impact JSON
(see impact_adapter.normalize) and does not depend on this module.
"""
import os
import re

HEAD = re.compile(r"^(kernel\.def|func\.def|func\.decl|command\.program\.def|template\.def|template\.decl|config\.def|config\.decl|"
                  r"global\.rodata\.def|target\.decl|amdgpu\.target)")
SYM = re.compile(r"@([A-Za-z_0-9]+)")
DEFSYM = re.compile(r"@([A-Za-z_0-9.]+)\(")


def split_blocks(text):
    """[(kind, symbol, header_line, body_text)] of the top-level blocks. A block runs from a column-0 keyword line to the
    next column-0 keyword line (comments between blocks are dropped)."""
    out, cur = [], None
    for ln in text.split("\n"):
        if ln.startswith("//"):
            continue
        m = HEAD.match(ln)
        if m:
            if cur:
                out.append(cur)
            kind = m.group(1)
            syms = DEFSYM.findall(ln) or SYM.findall(ln)
            sym = syms[-1] if syms else ""
            cur = [kind, sym, ln, [ln]]
        elif cur is not None:
            cur[3].append(ln)
    if cur:
        out.append(cur)
    return [(k, s, h, "\n".join(b).rstrip()) for k, s, h, b in out]


def _target_of(header):
    m = re.search(r"target\(@(rdna[34])\)", header)
    return m.group(1) if m else None


_TAPPLY = re.compile(r"template\.apply<@([A-Za-z_0-9]+)>\(([^)]*)\)[^\n]*")
_LAUNCH = re.compile(r"kernel\.launch @([A-Za-z_0-9]+)\[([^\]]*)\]\(([^)]*)\)[^\n]*")


def launch_canonicalizer(blocks, want="rdna3"):
    """The one implementation of 'a launch is the same launch whether it is written as kernel.launch or as template.apply of the
    selected target': returns canon(body) -> body in which both forms read `LAUNCH kernel [grid] (args)`. `blocks` = [(kind, header,
    body)] of the whole program (template.def blocks resolve the selection for target `want`)."""
    sel = {}
    for kind, hdr, body in blocks:
        if kind == "template.def" and _target_of(hdr) == want:
            tn = re.match(r"template\.def<@([A-Za-z_0-9]+)>", hdr).group(1)
            m = re.search(r"kernel\.launch @([A-Za-z_0-9]+)\[([^\]]*)\]", body)
            if m:
                sel[tn] = (m.group(1), len([x for x in m.group(2).split(",") if x.strip()]))

    def a(m):
        k, ng = sel.get(m.group(1), (m.group(1), 0))
        args = [x.strip() for x in m.group(2).split(",")]
        return f"LAUNCH {k} [{', '.join(args[:ng])}] ({', '.join(args[ng:])})"

    def canon(body):
        body = _TAPPLY.sub(a, body)
        return _LAUNCH.sub(lambda m: f"LAUNCH {m.group(1)} [{m.group(2)}] ({m.group(3)})", body)
    return canon


def view(program_dir, family="gfx11"):
    """{block key: normalized text} of a program directory as seen by one GPU family.
    gfx11: plain kernels and target(@rdna3) ones; template selection resolved to the rdna3 kernel launch.
    gfx12: kernels named *_rdna4 / target(@rdna4) (twins) only; shared non-twin blocks appear in both views."""
    want = "rdna3" if family == "gfx11" else "rdna4"
    other = "rdna4" if family == "gfx11" else "rdna3"
    blocks = []
    for f in sorted(os.listdir(program_dir)):
        if f.endswith(".loom"):
            with open(os.path.join(program_dir, f)) as fh:
                blocks += [(f, *b) for b in split_blocks(fh.read())]
    canon = launch_canonicalizer([(kind, hdr, body) for _, kind, _, hdr, body in blocks], want)

    out = {}
    for f, kind, sym, hdr, body in blocks:
        if kind in ("target.decl", "amdgpu.target", "template.decl", "template.def"):
            continue
        t = _target_of(hdr)
        if t == other or (kind == "kernel.def" and t is None and sym.endswith("_rdna4") and family == "gfx11"):
            continue
        if family == "gfx12" and kind == "kernel.def" and t is None:
            continue    # shared non-twin kernels are in the gfx11 view (compared there); gfx12 view = twins
        body = body.replace(f" target(@{want})", "").replace(" target(@device)", "")
        if kind == "command.program.def":
            body = canon(body)
        out[f"{kind}:{sym}"] = body
    return out


def diff_views(a, b):
    """changed block keys between two views: added, removed, modified."""
    ks = set(a) | set(b)
    return sorted(k for k in ks if a.get(k) != b.get(k))
