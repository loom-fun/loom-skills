"""Tiny harness: write npy inputs + a check.case, run loom-test with given kernel source."""
import os, json, subprocess, numpy as np
VENV = os.environ.get("VENV", os.path.expanduser("~/loom/.venv"))
BIN = os.path.join(VENV, "bin") if os.path.isdir(os.path.join(VENV, "bin")) else os.path.join(VENV, "Scripts")
def ty(a):
    d = {np.float32: "f32", np.int32: "i32", np.int8: "i8", np.uint8: "i8", np.float16: "f16"}[a.dtype.type]
    return f"tensor<{'x'.join(str(s) for s in a.shape)}x{d}>"
class Case:
    def __init__(s, name, d):
        s.name, s.d, s.lines, s.n = name, d, [], 0
    def lit(s, v, t="index"):
        s.n += 1; nm = f"%l{s.n}"; s.lines.append(f"  {nm} = check.literal value({v}) : {t}"); return nm, t
    def arr(s, a):
        s.n += 1; nm = f"%a{s.n}"; fn = f"{s.name}_{s.n}.npy"
        np.save(os.path.join(s.d, fn), a if a.dtype != np.uint8 else a.view(np.int8)); s.lines.append(f'  {nm} = check.file.read.npy path("{fn}") : {ty(a)}'); return nm, ty(a)
    def out(s, shape, dt="f32"):
        s.n += 1; nm = f"%o{s.n}"; t = f"tensor<{'x'.join(map(str, shape))}x{dt}>"
        s.lines.append(f"  {nm} = check.generate.fill value({'0.0' if dt.startswith('f') else '0'}) : {t}"); return nm, t
    def launch(s, kern, grid, args):
        g = ", ".join(a[0] for a in grid); gt = ", ".join(a[1] for a in grid)
        s.lines.append(f"  kernel.launch @{kern}[{g}]({', '.join(a[0] for a in args)}) : [{gt}]({', '.join(a[1] for a in args)})")
    def expect(s, actual, ref, atol=1e-3, rtol=1e-3):
        e = s.arr(np.ascontiguousarray(ref.astype(np.float32)))
        s.lines.append(f"  check.expect.approximate actual({actual[0]}) expected({e[0]}) atol({atol:e}) rtol({rtol:e}) nan(different) : {actual[1]}, {e[1]}")
    def text(s): return f"check.case public @{s.name} {{\n" + "\n".join(s.lines) + "\n  check.return\n}\n"
def run(src, cases, d):
    open(os.path.join(d, "combined.loom"), "w").write(src + "\n" + "".join(c.text() for c in cases))
    r = subprocess.run([os.path.join(BIN, "loom-test"), "combined.loom", "--device=amdgpu", "--file-output-root=."], cwd=d, capture_output=True, text=True)
    try: res = json.loads(r.stdout)
    except ValueError:
        print("COMPILE FAIL\n", "\n".join(l for l in (r.stderr + r.stdout).splitlines() if not l.startswith('{"format'))[-4000:]); return False
    ok = True
    for smp in res["samples"]:
        f = smp["expectations"].get("failures", [])
        print(("PASS " if smp["passed"] else "FAIL ") + smp["case"], "" if smp["passed"] else (str(f)[:600] + str(smp.get("issues", ""))[:600])); ok &= smp["passed"]
    return ok
