"""Contract test: loom-dev's summarize_info.py reads the info.json this generator writes (both schemas).

contract/summarize_info.py is a vendored copy of loom-fun/loom-dev summarize_info.py (SOURCE.txt: repo, commit). The
website is built from its output, so a card schema change that it cannot read freezes the site. This runs its
summarize() on the golden info.json files of the card tests (loom-card/3 expected cards, loom-card/2 legacy cards) and
checks what the site relies on:
  - every card summarizes (no exception) and carries the keys the site reads;
  - the retired needle gates do not appear;
  - a loom-card/3 card's top-level performance is its primary GPU's (strix-halo when it has results), the other GPUs
    sit under "devices", and a Strix-only/legacy card has no results lost;
  - a loom-card/2 card and the same card as loom-card/3 summarize to the same top level.

    python3 -I contract/run_contract.py          (exit 1 on a failure)
"""
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.dirname(HERE)
sys.path.insert(0, HERE)
try:
    import huggingface_hub  # noqa: F401
except ImportError:   # summarize() is offline; the module only imports the client at the top
    stub = types.ModuleType("huggingface_hub")
    stub.HfApi = stub.hf_hub_download = None
    sys.modules["huggingface_hub"] = stub
import summarize_info as S  # noqa: E402


def load(p):
    with open(os.path.join(T, p), encoding="utf-8") as f:
        return json.load(f)


# (name, info.json, the package dir whose model.json / model.json.lock it summarizes with)
CARDS = [
    ("d1", "d1/expected/info.json", "d1/repo"),
    ("flint", "flint/expected/info.json", "flint/pkg"),
    ("r9700_exec", "r9700_exec/expected/info.json", "r9700/pkg"),
    ("r9700_none", "r9700_none/expected/info.json", "r9700/pkg"),
    ("r9700_preset", "r9700_preset/expected/info.json", "r9700/pkg"),
    ("r9700_staging", "r9700_staging/expected/info.json", "r9700/pkg"),
    ("r9700_two", "r9700_two/expected/info.json", "r9700/pkg"),
    ("legacy/d1", "legacy/d1/info.json", "d1/repo"),
    ("legacy/flint", "legacy/flint/info.json", "flint/pkg"),
]
KEYS = {"schema", "checkpoint", "name", "repo", "revision", "status", "license", "service", "weights", "serve",
        "features", "performance", "gates"}
PERF = {"date", "llamacpp_build", "drafting", "source", "metrics", "not_applicable", "devlab"}
failures = []


def check(ok, what):
    print(("OK   " if ok else "FAIL ") + what)
    if not ok:
        failures.append(what)


def summary(info, pkg):
    return S.summarize(info, load(pkg + "/model.json"), load(pkg + "/model.json.lock"), "rev", "lic")


def has_results(d):
    return bool(d and not d.get("state") and (d.get("performance") or {}).get("sessions"))


out = {}
for name, path, pkg in CARDS:
    info = load(path)
    try:
        s = summary(info, pkg)
    except Exception as e:   # noqa: BLE001 - any failure is the contract breaking
        check(False, f"{name}: summarize_info.py reads {info.get('schema')} ({type(e).__name__}: {e})")
        continue
    out[name] = s
    ok = KEYS <= set(s) and PERF <= set(s["performance"]) and s["schema"] == S.SCHEMA
    check(ok, f"{name}: {info['schema']} summarizes with the keys the site reads")
    check(not {"needle", "needle_full"} & set(s["gates"]["status"]), f"{name}: no retired needle gates")
    check(any(m["loom"] is not None for m in s["performance"]["metrics"].values()) or
          any(m["loom"] is not None for m in s["performance"]["devlab"]["metrics"].values()),
          f"{name}: the top-level performance has measured values")
    if info["schema"] == "loom-card/3":
        devs = info["devices"]
        want = next(g for g in ("strix-halo", "radeon-r9700") if has_results(devs.get(g)))
        check(s.get("primary_device") == want, f"{name}: primary device is {want}")
        others = [g for g in devs if g != want]
        check(sorted(s.get("devices", {})) == sorted(others), f"{name}: the other GPUs {others} are listed under devices")
        for g in others:
            d = s["devices"][g]
            check((d.get("performance") is not None) == has_results(devs[g]) and (has_results(devs[g]) or d.get("state")),
                  f"{name}: {g} lists its results or its state")
    else:
        check("devices" not in s, f"{name}: a loom-card/2 summary has no devices key")

for legacy, current in (("legacy/d1", "d1"), ("legacy/flint", "flint")):
    if legacy in out and current in out:
        a, b = out[legacy], {k: v for k, v in out[current].items() if k not in ("primary_device", "devices")}
        check(a == b, f"{legacy} (loom-card/2) and {current} (loom-card/3) summarize to the same top level")
if failures:
    sys.exit(f"{len(failures)} contract failure(s)")
