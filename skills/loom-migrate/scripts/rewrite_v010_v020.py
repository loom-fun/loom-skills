#!/usr/bin/env python3
"""The mechanical source migration hrx-loom v0.1.0 -> v0.2.0 on Loom text, a package or a tree. Idempotent; standard library only.

usage: rewrite_v010_v020.py [--check] [--json OUT] [--no-publish-epoch] [--no-card] [--no-widen-pool] [--py] PATH...

PATH is a .loom file, a package (a directory with model.json) or any tree. Rewrites, in place:
  vector.constant X : vector<NxT>   -> vector.splatf (T float: f16 f32 f64 bf16 f8*) | vector.splati (int, predicate, index)
  command.program.{def,decl,launch} -> command.{def,decl,launch}
  publish_epoch (def, decl and call sites of @publish_epoch, @*_publish_epoch, @*_publish_epoch_v1): prepends the retained
      `%control_state: buffer` argument (v0.2 ABI); skipped by --no-publish-epoch
  weights policy imports (hrx-system 04aafdbe): `weights.parameter` / `weights.input` gain `key_offset, key_length` after the key; their decls and
      call sites (key = the whole buffer: offset 0, length buffer.length) are rewritten in .loom/.patch files (decls only in .py/.md)
  KV pool bounds (section 2.5, the kit's choice; skipped by --no-widen-pool), matched by the op and symbol, never by the bare number:
      `index.assume %pool* [range(%pool*, 16, 1048576)]` and `config.decl @llm.pool ... range(%v, 16, 1048576)` -> 2147483584,
      `scalar.assume %*physical*|%*_ph* [range(.., 0, 1048575)]` (physical page id) -> 134217727,
      flash2 `index.assume [range(v, 0, 1048560), lt(v, %pl_rows_m)]` -> 2147483568 and `[range(v, 0, 1048575), lt(v, %pool)]` -> 2147483583.
      Any other 1048575/1048576 (resident-row bounds, `index.constant 1048576`) is left alone.
  one-token conv guard (section 2.9): the short-conv / GDN helper `%hasN = index.cmp uge, %rel, %backN` followed by `index.sub %token, %backN`
      becomes `%has_relN` AND `%has_tokenN` (`scalar.andi`); a `uge, %rel, %back` line of another shape is reported as manual with file:line
  vision finite-batch ABI (section 2.4): the v0.1 `@text_image_prepare_batch_v1(%state, %tags, %ranges, %packed)` / `@text_image_run_v1(%stage, ..)`
      component gets the `finite` imports, `%batch: ptr<@FiniteBatch>`, `finite_batch_pack` and the new `@text_image_collect_batch_v1`
      (exactly what vision/gen_component.py emits); any other old shape is reported as manual
  model.json: removes presets.*.execution.pool_capacity (v0.2 rejects it); skipped by --no-card
Files: *.loom, *.loom-test, *.patch; with --py also *.py and *.md (generator sources and docs: renames only where the element type is
literal). `vector.constanti` (the dense-list op) is unchanged in v0.2 and is never touched.
Reports what it could NOT decide (a vector.constant whose element type is a placeholder, a vision ABI or conv guard of another shape) as `manual`
with file:line and the section of versions/v0.1.0-to-v0.2.0.md that says what to do. --check changes nothing and exits 1 if any file would change.
"""
import argparse
import json
import os
import re
import sys

LOOM_EXT = (".loom", ".loom-test", ".patch")
PY_EXT = (".py", ".md")
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv"}

VC = re.compile(r"\bvector\.constant(?=\s)([^\n]*?:\s*vector<[^>\n]*?x(\w+)>)")
VC_ANY = re.compile(r"\bvector\.constant(?=\s)")
CP = re.compile(r"\bcommand\.program\.(def|decl|launch)\b")
PE_NAME = r"@(?:\w+_)?publish_epoch(?:_v1)?"
PE_SIG = re.compile(r"^(func\.(?:def|decl)(?: \w+)* " + PE_NAME + r"\()(?=%plan\s*:)", re.M)
PE_CALL = re.compile(r"(func\.call " + PE_NAME + r"\()(?=%plan\b)([^\n]*?\)\s*:\s*\()")
OLD_PREPARE_BATCH = re.compile(r"@text_image_prepare_batch_v1\(%state: buffer, %tags: buffer, %ranges")
V_PREPARE = re.compile(r"^(func\.def public @text_image_prepare_batch_v1\(%state: buffer, %tags: buffer, )%ranges: buffer, %packed: buffer(\) -> \(i32, buffer, buffer\) \{\n)", re.M)
V_RUN = re.compile(r"^func\.def public @text_image_run_v1\(%stage: i32, %input: hal\.buffer, %output: hal\.buffer\) \{\n(?:(?!\}).*\n)*?\}\n", re.M)
V_DECLS = ('type.decl import("finite", "batch") @FiniteBatch\n'
           'func.decl import("finite", "batch_pack") @finite_batch_pack(%batch: ptr<@FiniteBatch>) -> (buffer, buffer)\n'
           'func.decl import("finite", "batch_finish") @finite_batch_finish(%batch: ptr<@FiniteBatch>)\n')
V_COLLECT = ("\n// One run encodes the whole cohort (one image per run): every member is done when the run is collected.\n"
             "func.def public @text_image_collect_batch_v1(%state: buffer, %batch: ptr<@FiniteBatch>) {\n"
             "  func.call @finite_batch_finish(%batch) : (ptr<@FiniteBatch>)\n  func.return\n}\n")
POOL_ARG = re.compile(r"(index\.assume (%[\w{}]*pool[\w{}]*) \[range\(\2, 16, )1048576\)")
POOL_DECL = re.compile(r"(config\.decl @llm\.pool : (%\w+): index where \[range\(\2, 16, )1048576\)")
PHYS_PAGE = re.compile(r"(scalar\.assume (%[\w{}]*(?:physical|_ph)[\w{}]*) \[range\(\2, 0, )1048575\)")
FLASH_ROWS = re.compile(r"(index\.assume (%[\w{}]+) \[range\(\2, 0, )(?:1048560(\), lt\(\2, %pl_rows_m\))|1048575(\), lt\(\2, %pool\)))")
CONV_GUARD = re.compile(r"^([ \t]*)%has(?!_rel)([\w{}]*) = index\.cmp uge, %rel, (%back[\w{}]*) : index$", re.M)
CONV_ANY = re.compile(r"index\.cmp uge, %rel, %back")
WI_DECL = re.compile(r'(func\.decl import\("weights"\) @input\(%source: i32, %key: buffer, )(%offset: i64, %length: i64, %alignment: i64\))')
WP_DECL = re.compile(r'(func\.decl import\("weights"\) @parameter\(%source: i32, %key: buffer)(\) -> \(buffer, i64\))')
WI_CALL = re.compile(r"^([ \t]*)((?:%\w+ = )?)func\.call @input\((%[\w.]+), (%[\w.]+), (%[\w.]+), (%[\w.]+), (%[\w.]+)\) : \(i32, buffer, i64, i64, i64\) -> \(i64\)", re.M)
WP_CALL = re.compile(r"^([ \t]*)((?:%[\w.]+, %[\w.]+|%[\w.]+) = )func\.call @parameter\((%[\w.]+), (%[\w.]+)\) : \(i32, buffer\) -> \(buffer, i64\)", re.M)
FLOAT = re.compile(r"^(?:f\d+\w*|bf16)$")
INT = re.compile(r"^(?:i\d+|index|offset)$")


COUNT_KEYS = ("splati", "splatf", "command", "publish_epoch", "weights_key_range",
              "pool_arg", "pool_decl", "physical_page", "flash2_rows", "conv_guard", "vision_abi")
SEC_VISION = "versions/v0.1.0-to-v0.2.0.md section 2.4"
SEC_CONV = "versions/v0.1.0-to-v0.2.0.md section 2.9"


def widen_pool(text, counts):
    """The v0.2 KV pool bounds (2.5), by context."""
    def n(key, repl):
        def f(m):
            counts[key] += 1
            return repl(m)
        return f
    text = POOL_ARG.sub(n("pool_arg", lambda m: m.group(1) + "2147483584)"), text)
    text = POOL_DECL.sub(n("pool_decl", lambda m: m.group(1) + "2147483584)"), text)
    text = PHYS_PAGE.sub(n("physical_page", lambda m: m.group(1) + "134217727)"), text)
    return FLASH_ROWS.sub(n("flash2_rows", lambda m: m.group(1) + ("2147483568" + m.group(3) if m.group(3) else "2147483583" + m.group(4))), text)


def conv_guard(text, counts, manual):
    """One-token conv guard (2.9): `token >= back` ANDed with `rel >= back`."""
    def f(m):
        back = m.group(3)
        if not re.search(r"index\.sub %token, " + re.escape(back) + r"\b", text[m.end():m.end() + 400]):
            return m.group(0)                      # not the helper (no %token read): the leftover check reports it
        ind, sfx = m.group(1), m.group(2)
        counts["conv_guard"] += 1
        return (f"{ind}%has_rel{sfx} = index.cmp uge, %rel, {back} : index\n{ind}%has_token{sfx} = index.cmp uge, %token, {back} : index\n"
                f"{ind}%has{sfx} = scalar.andi %has_rel{sfx}, %has_token{sfx} : i1")
    new = CONV_GUARD.sub(f, text)
    for m in CONV_ANY.finditer(new):
        line = new[new.rfind("\n", 0, m.start()) + 1:new.find("\n", m.start())]
        if "%has_rel" not in line:
            manual.append((f"conv helper `uge, %rel, %back` of another shape: add `%token >= %back` to the guard, see {SEC_CONV}", new.count("\n", 0, m.start()) + 1))
    return new


def vision_abi(text, counts, manual):
    """The v0.1 vision component finite-batch ABI (2.4), when it is exactly the kit's v0.1 shape."""
    if not OLD_PREPARE_BATCH.search(text):
        return text
    ok = V_PREPARE.search(text) and V_RUN.search(text) and "@FiniteBatch" not in text
    first = [ln for ln in text.split("\n") if ln.strip() and not ln.startswith("//")]
    if not ok or not first:
        manual.append((f"vision component finite ABI of another shape than the kit's v0.1 component: hand edit, see {SEC_VISION}",
                       text.count("\n", 0, OLD_PREPARE_BATCH.search(text).start()) + 1))
        return text
    new = V_PREPARE.sub(lambda m: m.group(1) + "%batch: ptr<@FiniteBatch>" + m.group(2)
                        + "  %ranges, %packed = func.call @finite_batch_pack(%batch) : (ptr<@FiniteBatch>) -> (buffer, buffer)\n", text, count=1)
    run = V_RUN.search(new)
    body = run.group(0).replace("(%stage: i32, %input: hal.buffer, %output: hal.buffer)",
                                "(%state: buffer, %stage: i32, %input: hal.buffer, %output: hal.buffer, %batch: ptr<@FiniteBatch>)", 1)
    new = new[:run.start()] + body + V_COLLECT + new[run.end():]
    lines = new.split("\n")
    at = next(i for i, ln in enumerate(lines) if ln.strip() and not ln.startswith("//"))
    new = "\n".join(lines[:at]) + ("\n" if at else "") + V_DECLS + "\n".join(lines[at:])
    counts["vision_abi"] += 1
    return new


def rewrite_text(text, publish_epoch=True, calls=True, pool=True):
    """Returns (new_text, counts, manual) for one source text; manual is a list of (what, line)."""
    counts = dict.fromkeys(COUNT_KEYS, 0)
    manual = []

    def vc(m):
        t = m.group(2)
        if FLOAT.match(t):
            counts["splatf"] += 1
            return "vector.splatf" + m.group(1)
        if INT.match(t):
            counts["splati"] += 1
            return "vector.splati" + m.group(1)
        manual.append((f"vector.constant of element type {t!r}: write vector.splatf or vector.splati by hand, see versions/v0.1.0-to-v0.2.0.md section 2.1", text.count("\n", 0, m.start()) + 1))
        return m.group(0)

    new = VC.sub(vc, text)
    left = len(VC_ANY.findall(new)) - sum(1 for x, _ in manual if x.startswith("vector.constant"))
    if left > 0:
        manual.append((f"{left} vector.constant with a non-literal element type (a generator placeholder): write the splat by hand, see versions/v0.1.0-to-v0.2.0.md section 2.1", new.count("\n", 0, VC_ANY.search(new).start()) + 1))
    new, n = CP.subn(r"command.\1", new)
    counts["command"] = n
    if publish_epoch:
        new, n1 = PE_SIG.subn(r"\1%control_state: buffer, ", new)
        new, n2 = PE_CALL.subn(lambda m: m.group(1) + "%control_state, " + m.group(2) + "buffer, ", new)
        counts["publish_epoch"] = n1 + n2
    new, n1 = WI_DECL.subn(r"\1%key_offset: i64, %key_length: i64, \2", new)
    new, n2 = WP_DECL.subn(r"\1, %key_offset: i64, %key_length: i64\2", new)
    counts["weights_key_range"] = n1 + n2
    if calls:
        serial = [0]

        def key_range(m):
            serial[0] += 1
            ind, key, k = m.group(1), m.group(4), serial[0]
            pre = (f"{ind}%wk{k}_off = scalar.constant 0 : i64\n{ind}%wk{k}_ext = buffer.length {key}\n"
                   f"{ind}%wk{k}_len = index.cast %wk{k}_ext : offset to i64\n")
            return pre, f"{key}, %wk{k}_off, %wk{k}_len"

        def wi(m):
            pre, kr = key_range(m)
            counts["weights_key_range"] += 1
            return f"{pre}{m.group(1)}{m.group(2)}func.call @input({m.group(3)}, {kr}, {m.group(5)}, {m.group(6)}, {m.group(7)}) : (i32, buffer, i64, i64, i64, i64, i64) -> (i64)"

        def wp(m):
            pre, kr = key_range(m)
            counts["weights_key_range"] += 1
            return f"{pre}{m.group(1)}{m.group(2)}func.call @parameter({m.group(3)}, {kr}) : (i32, buffer, i64, i64) -> (buffer, i64)"

        new = WI_CALL.sub(wi, new)
        new = WP_CALL.sub(wp, new)
    if pool:
        new = widen_pool(new, counts)
    if calls:
        new = conv_guard(new, counts, manual)
        new = vision_abi(new, counts, manual)
    manual.sort(key=lambda x: x[1])
    return new, counts, manual


def rewrite_card(text):
    """Remove `"pool_capacity": N` lines from model.json text, keeping the formatting. Returns (text, removed)."""
    lines, out, removed = text.split("\n"), [], 0
    for ln in lines:
        if re.match(r'^\s*"pool_capacity"\s*:\s*[^,\n{}\[\]]+,?\s*$', ln):
            removed += 1
            if not ln.rstrip().endswith(",") and out and out[-1].rstrip().endswith(","):
                out[-1] = out[-1].rstrip()[:-1]   # it was the last key: the previous line loses its comma
            continue
        out.append(ln)
    return "\n".join(out), removed


def files_under(paths, exts):
    for p in paths:
        if os.path.isfile(p):
            yield p
            continue
        for d, dirs, names in os.walk(p):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
            for n in sorted(names):
                if n.endswith(exts) or n == "model.json":
                    yield os.path.join(d, n)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--no-publish-epoch", action="store_true")
    ap.add_argument("--no-card", action="store_true")
    ap.add_argument("--no-widen-pool", action="store_true", help="keep the v0.1 KV pool bounds (1048576 positions)")
    ap.add_argument("--py", action="store_true", help="also rewrite .py/.md generator sources and docs")
    a = ap.parse_args(argv)
    exts = LOOM_EXT + (PY_EXT if a.py else ())
    changed, manual, totals = {}, {}, dict.fromkeys(COUNT_KEYS + ("pool_capacity",), 0)
    for f in files_under(a.paths, exts):
        with open(f, encoding="utf-8", newline="") as fh:
            text = fh.read()
        if os.path.basename(f) == "model.json":
            if a.no_card:
                continue
            new, n = rewrite_card(text)
            counts = {"pool_capacity": n}
            man = []
        else:
            new, counts, man = rewrite_text(text, not a.no_publish_epoch, calls=not f.endswith(PY_EXT), pool=not a.no_widen_pool)
        for k, v in counts.items():
            totals[k] += v
        if man:
            manual[f] = [f"{what} [line {ln}]" for what, ln in man]
        if new != text:
            changed[f] = counts
            if not a.check:
                with open(f, "w", encoding="utf-8", newline="") as fh:
                    fh.write(new)
    verb = "would change" if a.check else "changed"
    print(f"{verb} {len(changed)} file(s): " + ", ".join(f"{k}={v}" for k, v in totals.items()))
    for f, m in sorted(manual.items()):
        for x in m:
            what, _, ln = x.rpartition(" [line ")
            print(f"manual: {f}:{ln.rstrip(']')}: {what}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"changed": changed, "manual": manual, "totals": totals}, fh, indent=1)
    return 1 if (a.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
