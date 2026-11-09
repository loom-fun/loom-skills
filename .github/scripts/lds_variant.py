"""mkvariant.py SRC_PKG DST_PKG KERNEL ALLOCA_NAME NEW_BYTES: copy a Loom package (parameters hard-linked) with one
workgroup alloca of one kernel resized. Only the size constant changes; the kernel's accesses stay the same."""
import os, re, shutil, sys
src, dst, kern, name, new = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
if os.path.exists(dst): shutil.rmtree(dst)
for root, dirs, files in os.walk(src):
    rel = os.path.relpath(root, src); os.makedirs(os.path.join(dst, rel), exist_ok=True)
    for f in files:
        s, d = os.path.join(root, f), os.path.join(dst, rel, f)
        if rel.startswith('parameter'): os.link(s, d)
        else: shutil.copy2(s, d)
p = os.path.join(dst, 'program', 'llm.loom'); t = open(p).read()
i = t.index(f'kernel.def export("{kern}")'); j = t.find('\nkernel.def', i + 10)
body = t[i:j]
pat = re.compile(r'(%' + name + r' = index\.constant )(\d+)')
m = pat.search(body); assert m, 'constant not found'
print(f'{kern} %{name}: {m.group(2)} -> {new}')
body = pat.sub(lambda m: m.group(1) + str(new), body, count=1)
open(p, 'w', newline='\n').write(t[:i] + body + t[j:])
