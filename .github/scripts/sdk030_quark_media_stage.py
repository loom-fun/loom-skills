"""Select Quark's verified A4/BF16 prefill only for media-containing epochs."""

import re
import sys
from pathlib import Path


MEDIA_ROOT = "quark_media_llm_advance"
HEADER = re.compile(r"^(?:(?:func|kernel|command|template)\.(?:def|decl)[^\n]*? @(\w+)\(|global\.rodata\.def @(\w+)\s*=)", re.M)
SYMBOL = re.compile(r"@(\w+(?:\.\w+)*)")


def definitions(text):
    entries = list(HEADER.finditer(text))
    result = {}
    for index, entry in enumerate(entries):
        name = entry[1] or entry[2]
        tail = entries[index + 1].start() if index + 1 < len(entries) else len(text)
        declaration = text[entry.start():tail]
        if "\n}\n" in declaration and not declaration.startswith(("template.decl", "func.decl")):
            declaration = declaration[:declaration.index("\n}\n") + 3]
        else:
            declaration = declaration.splitlines(keepends=True)[0]
        assert name not in result, name
        result[name] = declaration
    return result


def combine_commands(original, media):
    normal = definitions(original)
    alternate = definitions(media)
    implementations = {}
    for name, text in alternate.items():
        match = re.match(r"template\.def<@(\w+)>", text)
        if match:
            implementations.setdefault(match[1], set()).add(name)
    dependencies = {name: (set(SYMBOL.findall(text)) & alternate.keys()) | implementations.get(name, set())
                    for name, text in alternate.items()}
    root = "llm_advance"
    needed = set()
    pending = [root]
    while pending:
        name = pending.pop()
        if name not in needed:
            needed.add(name)
            pending.extend(dependencies[name] - needed)
    changed = {name for name in needed if alternate[name] != normal.get(name)}
    changed.add(root)
    while True:
        additional = {name for name in needed - changed if dependencies[name] & changed}
        if not additional:
            break
        changed.update(additional)
    names = {name: f"quark_media_{name}" for name in changed}
    assert names[root] == MEDIA_ROOT and not (set(names.values()) & normal.keys())
    program = []
    for name, text in alternate.items():
        if name not in changed:
            continue
        renamed = SYMBOL.sub(lambda match: "@" + names.get(match[1], match[1]), text)
        renamed = re.sub(r'export\("(\w+)"\)', lambda match: f'export("{names[match[1]]}")', renamed)
        program.append(renamed)
    print(f"Quark media stage reused {len(needed - changed)} shared declarations and added {len(changed)} scoped declarations")
    return original + "\n// Quark A4 and BF16 prefill for media epochs.\n" + "\n".join(program)


def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def add_stage_table(prepare):
    assert "2048 : offset" in prepare and "quark_media_root" not in prepare
    prepare = replace_once(prepare, "  %control_bytes = index.constant 2048 : offset",
                           "  %control_bytes = index.constant 2560 : offset")
    anchor = "  %smaller = scalar.cmpi ult, %maximum, %prefill : i64"
    body = """  %quark_media_at = index.constant 2048 : offset
  %quark_media_view = buffer.view %state[%quark_media_at] : buffer -> view<128xi32>
  %quark_media_root = global.load @quark_media_root : buffer
  scf.for %qmi = [%c0 to %count step %c1] {
    %qmt = view.load %shape_view[%qmi, 0] : view<[%count]x2xi64> -> i64
    %qms = view.load %shape_view[%qmi, 1] : view<[%count]x2xi64> -> i64
    %qmtag = index.cast %qmi : index to i64
    %qmstage = func.call @stage(%quark_media_root, %qmtag) : (buffer, i64) -> (i32)
    func.call @config_i64(%qmstage, %context_key, %context) : (i32, buffer, i64)
    func.call @config_i64(%qmstage, %pool_key, %pool) : (i32, buffer, i64)
    func.call @config_i64(%qmstage, %tokens_key, %qmt) : (i32, buffer, i64)
    func.call @config_i64(%qmstage, %spans_key, %qms) : (i32, buffer, i64)
    scf.if %logits_cfg {
      %qmlogits = scalar.minsi %qms, %rows : i64
      func.call @config_i64(%qmstage, %logit_slots_key, %qmlogits) : (i32, buffer, i64)
    }
    func.call @parameter(%qmstage, %c0_i32, %c0_source, %checkpoint, %policy) : (i32, i32, i32, i32, buffer)
    view.store %qmstage, %quark_media_view[%qmi] : i32, view<128xi32>
  }
"""
    prepare = replace_once(prepare, anchor, body + anchor)
    return f'global.rodata.def @quark_media_root = bytes("{MEDIA_ROOT.encode().hex()}")\n\n' + prepare


def wire_media_counts(control):
    assert control.count("index.constant 2048 : offset") == 3
    control = control.replace("index.constant 2048 : offset", "index.constant 2560 : offset")
    for indent, stage, view, index in (("  ", "stage", "stages", "shape_index"),
                                       ("    ", "astage", "advance_view", "eff_index")):
        old = f"{indent}%{stage} = view.load %{view}[%{index}] : view<128xi32> -> i32"
        new = f"{indent}%text_{stage} = view.load %{view}[%{index}] : view<128xi32> -> i32\n"
        new += (f"{indent}%{stage} = func.call @quark_select_media_stage(%control_state, %{index}, "
                f"%first_media_count, %next_media_count, %text_{stage}) : (buffer, index, i32, i32, i32) -> (i32)")
        control = replace_once(control, old, new)
    return control + """
func.def @quark_select_media_stage(%state: buffer, %shape: index, %first: i32, %next: i32, %text_stage: i32) -> (i32) {
  %zero = scalar.constant 0 : i32
  %media_count = scalar.addi %first, %next : i32
  %has_media = scalar.cmpi ne, %media_count, %zero : i32
  %stage = scf.if %has_media -> (i32) {
    %media_at = index.constant 2048 : offset
    %table = buffer.view %state[%media_at] : buffer -> view<128xi32>
    %index = index.assume %shape [range(%shape, 0, 127)] : index
    %media_stage = view.load %table[%index] : view<128xi32> -> i32
    scf.yield %media_stage : i32
  } else {
    scf.yield %text_stage : i32
  }
  func.return %stage : i32
}
"""


def main():
    program_dir, half = map(Path, sys.argv[1:])
    transforms = {"llm": lambda original: combine_commands(original, half.read_text()),
                  "prepare": add_stage_table, "control": wire_media_counts}
    changes = {}
    for name, transform in transforms.items():
        path = program_dir / f"{name}.loom"
        changes[path] = transform(path.read_text())
    for path, text in changes.items():
        path.write_text(text, newline="\n")


if __name__ == "__main__":
    main()
