"""Compose the gfx11 and gfx12 views when a scoped choice changes the command graph.

Both inputs are raw gen_llm modules, before dual_target.apply. Preserve the base kernels,
take gfx12 twins from the second generation, and create twins for formerly shared kernels
whose bodies now differ. A command whose launches or control flow differ gets two providers.
"""
import re

import dual_target as dt

# A kernel's launch configuration and body form one definition. The brace between
# them shares a line with "launch(...)"; only the last brace is alone.
DEFINITION = re.compile(r"^(kernel\.def |command\.def |func\.def )[^\n]*\n.*?^\}$", re.M | re.S)
NAME = re.compile(r"\s@([\w.]+)\(")


def definitions(text):
    result = {}
    for m in DEFINITION.finditer(text):
        head = m.group().split("\n", 1)[0]
        key = (m[1].strip(), NAME.search(head)[1])
        if key in result:
            raise ValueError(f"duplicate definition {key}")
        result[key] = m.group()
    return result


def compose(base, twin):
    if base == twin:
        return dt.apply(base)
    a, b = definitions(base), definitions(twin)
    alternate = {}
    new = set()
    replacement = {}
    for key in a.keys() | b.keys():
        kind, name = key
        left, right = a.get(key), b.get(key)
        if left == right:
            continue
        if kind == "command.def":
            if not left or not right or left.splitlines()[0] != right.splitlines()[0]:
                raise ValueError(f"target-specific command signature: {name}")
            alternate[left.splitlines()[0]] = right.splitlines()
        elif kind == "func.def":
            if left and right:
                raise ValueError(f"target-specific inline function: {name}")
            if right:
                new.add(key)
        elif name.endswith(dt.SUFFIX):
            if left:
                replacement[key] = right or ""
            else:
                new.add(key)
        elif left and not right:
            if "target(" not in left.split("\n", 1)[0]:
                replacement[key] = left.replace("kernel.def ", "kernel.def target(@rdna3) ", 1)
        elif not left:
            new.add(key)
        elif ("kernel.def", name + dt.SUFFIX) not in b:
            # No explicit WMMA twin: this shared kernel now needs two implementations.
            twin_name = name + dt.SUFFIX
            header, body = right.split("\n", 1)
            header = header.replace("kernel.def ", "kernel.def target(@rdna4) ", 1)
            header = header.replace(f'"{name}"', f'"{twin_name}"').replace(f"@{name}(", f"@{twin_name}(")
            replacement[key] = left + "\n\n" + header + "\n" + body

    def substitute(m):
        key = (m[1].strip(), NAME.search(m.group().split("\n", 1)[0])[1])
        return replacement.get(key, m.group())

    merged = DEFINITION.sub(substitute, base)
    if new:
        additions = []
        for key in b:
            if key in new:
                block = b[key]
                if key[0] == "kernel.def" and "target(" not in block.split("\n", 1)[0]:
                    block = block.replace("kernel.def ", "kernel.def target(@rdna4) ", 1)
                additions.append(block)
        before = re.search(r"^command\.def ", merged, re.M).start()
        merged = merged[:before] + "\n\n".join(additions) + "\n\n" + merged[before:]
    return dt.apply(merged, alternate=alternate)
