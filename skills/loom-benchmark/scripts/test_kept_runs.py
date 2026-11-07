"""Unit test of the Loom runs consistency.py keeps across deferred-reference rounds (bench_common.kept_runs,
loom_runs_key): kept runs are used only under the key they were made with. python test_kept_runs.py"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import kept_runs, loom_runs_key  # noqa: E402

d = tempfile.mkdtemp(); pkg = os.path.join(d, "pkg"); os.makedirs(os.path.join(pkg, "program"))
for f, t in (("model.json", "{}"), ("llm_config.json", "{}"), ("program/llm.loom", "a")):
    open(os.path.join(pkg, f), "w").write(t)
state = os.path.join(d, "pending_consistency.jsonl.loom.json")
key = loom_runs_key(pkg, "m", ["p"])
json.dump({"key": key, "alone": [1]}, open(state, "w"))
checks = [("same package", kept_runs(state, loom_runs_key(pkg, "m", ["p"])) is not None)]
for f in ("model.json", "llm_config.json", "program/llm.loom"):   # any package file changed: the kept runs are stale
    old = open(os.path.join(pkg, f)).read(); open(os.path.join(pkg, f), "w").write(old + " ")
    checks.append((f"{f} changed", kept_runs(state, loom_runs_key(pkg, "m", ["p"])) is None))
    open(os.path.join(pkg, f), "w").write(old)
checks += [("other prompts", kept_runs(state, loom_runs_key(pkg, "m", ["q"])) is None),
           ("no file", kept_runs(os.path.join(d, "none.json"), key) is None)]
shutil.rmtree(d)
for name, ok in checks:
    print(("ok  " if ok else "FAIL"), name)
sys.exit(0 if all(ok for _, ok in checks) else 1)
