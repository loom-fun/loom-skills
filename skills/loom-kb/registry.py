#!/usr/bin/env python3
"""Data loaders of the loom-kb registries: devices, device families, quant formats, methodology. Standard library only, path-relative
(this file's directory is the KB root), so any skill, script or CI job can import it:

    sys.path.insert(0, os.path.join(SKILLS, "loom-kb")); import registry

  devices()            {key: device dict}            device/<family>/<device>/device.json, plus "family"
  device(name)         one device by key, alias, gfx id or roofline preset name (KeyError when unknown)
  families()           {name: family dict}           device/<family>/family.json, in "order"
  family_of(name)      the family dict of a device
  dual_target()        the rows dual_target.py needs: base + twin families, header text, twin suffix
  detect(text)         the device key whose card "detect" pattern matches text, else None
  roofline_presets()   {preset: {name, bw, flops, device}} every roofline device preset of every device
  quants()             {format: format dict}         quant/<fmt>/format.json
  file_quants()        the GGUF file quants the factory accepts, in order of preference
  methodology()        method/methodology.json
  band(row)            the A/B band of a metric row (aliases such as c4 resolve; unknown rows: default_band)

`python registry.py` prints a one-screen summary. Each reader of a table that used to be hard-coded in a script (card_lib.GPUS,
roofline.DEVICES, dual_target.HEADER ...) builds it from here, byte-identically.
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
_CACHE = {}


def _load(*parts):
    path = os.path.join(ROOT, *parts)
    if path not in _CACHE:
        with open(path, encoding="utf-8") as f:
            _CACHE[path] = json.load(f)
    return _CACHE[path]


def _subdirs(*parts):
    base = os.path.join(ROOT, *parts)
    return sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and not d.startswith((".", "__")))


def families():
    out = {}
    for fam in _subdirs("device"):
        if os.path.exists(os.path.join(ROOT, "device", fam, "family.json")):
            out[fam] = dict(_load("device", fam, "family.json"))
    return dict(sorted(out.items(), key=lambda kv: (kv[1].get("order", 99), kv[0])))


def devices():
    """{key: device dict}, in family order then directory order; each dict has "family" and "path" (relative to the KB root)."""
    out = {}
    for fam in families():
        for d in _subdirs("device", fam):
            p = os.path.join("device", fam, d, "device.json")
            if os.path.exists(os.path.join(ROOT, p)):
                dev = dict(_load(*p.split(os.sep)))
                dev["path"] = os.path.dirname(p)
                out[dev["device"]] = dev
    return out


def device(name):
    devs = devices()
    if name in devs:
        return devs[name]
    for dev in devs.values():
        if name in dev.get("aliases", []) or name == dev.get("gfx") or name in dev["roofline"]["presets"]:
            return dev
    raise KeyError(f"unknown device {name!r} (known: {', '.join(devs)})")


def family_of(name):
    return families()[device(name)["family"]]


def detect(text):
    """The key of the first device whose card detect pattern matches TEXT (a machine_info.py device block joined), else None."""
    for key, dev in devices().items():
        d = dev["card"]["detect"]
        if re.search(d["pattern"], text, re.I if "i" in d.get("flags", "") else 0):
            return key
    return None


def roofline_presets():
    """{preset: {"name", "bw", "flops", "device"}}: every roofline preset of every device (a device may have a spec and a measured one)."""
    out = {}
    for key, dev in devices().items():
        for name, p in dev["roofline"]["presets"].items():
            out[name] = {"name": p["name"], "bw": p["bw"], "flops": p["flops"], "device": key}
    return out


def dual_target():
    """What the dual-target post-pass needs: the base family (its kernels keep their text), the twin families, the header text and the
    suffix of a twin kernel. Only families with "dual_target": true take part, in "order"."""
    fams = [f for f in families().values() if f.get("dual_target")]
    base = next(f for f in fams if f.get("base"))
    twins = [f for f in fams if not f.get("base")]
    header = "target.decl @device\n" + "".join(f"amdgpu.target<{f['gfx_generic']}> @{f['target_symbol']}\n" for f in fams) + "\n"
    return {"base": base["target_symbol"], "twins": [f["target_symbol"] for f in twins], "suffix": twins[0]["twin_suffix"], "header": header}


def quants():
    return {f: _load("quant", f, "format.json") for f in _subdirs("quant") if os.path.exists(os.path.join(ROOT, "quant", f, "format.json"))}


def file_quants():
    """The GGUF file quants (Q4_K_M ...) the factory accepts, in order of preference."""
    return [q["name"] for q in sorted(_load("quant", "file_quants.json")["file_quants"], key=lambda q: q["rank"])]


def methodology():
    return _load("method", "methodology.json")


def band(row, default=None):
    """The A/B band (+- fraction around ratio 1.0) of a metric row; c4 / c4_steady are streams4."""
    m = methodology()
    row = m["row_aliases"].get(row, row)
    return m["rows"][row]["band"] if row in m["rows"] else (m["default_band"] if default is None else default)


if __name__ == "__main__":
    for fam, f in families().items():
        print(f"family {fam:6} {f['status']:10} {f.get('gfx_generic')}  twin {f.get('twin_suffix')!r}  devices {f['devices']}")
    for key, d in devices().items():
        print(f"device {key:14} {d['gfx']}  runner {d['runner']['key']} {d['runner']['labels']}  roofline {sorted(d['roofline']['presets'])}")
    print("quants", " ".join(quants()))
    print("bands", {r: v["band"] for r, v in methodology()["rows"].items()})
