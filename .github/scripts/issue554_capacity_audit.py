"""Experiment-only audit of the intended SDK020 fixed native-context preset."""
import argparse
import hashlib
import json
from pathlib import Path
import requests

ap = argparse.ArgumentParser()
ap.add_argument("pkg")
ap.add_argument("url")
ap.add_argument("phase", choices=("before", "after"))
ap.add_argument("--out", required=True)
a = ap.parse_args()
pkg = Path(a.pkg)
out = Path(a.out)
model = json.loads((pkg / "model.json").read_text())
assert list(model["presets"]) == ["chat"], list(model["presets"])
execution = model["presets"]["chat"]["execution"]
assert execution["context_capacity"] == 131072, execution
assert execution["prefill_capacity"] == 2048, execution
assert execution["rows"] == 4 and execution["draft_depth"] == 3, execution
response = requests.get(a.url + "/v1/models", timeout=30)
response.raise_for_status()
status = response.json()
assert len(status["data"]) == 1
active = status["data"][0]
assert active["context_length"] == 131072, active
pools = status["server"]["resources"]["pools"]
assert len(pools) == 1
memory = pools[0]["memory"]
assert memory["backing"] == "fixed" and memory["retained"] is None, memory
total = memory["storage"]["committed_bytes"]
workspace = memory["workspace"]["committed_bytes"]
assert 0 < workspace <= total <= 32 * 2**30, memory
assert memory["workspace"]["exhausted_count"] == 0, memory
assert memory["workspace"]["over_budget_count"] == 0, memory
metrics = requests.get(a.url + "/metrics", timeout=30)
metrics.raise_for_status()
out.with_suffix(".metrics.txt").write_text(metrics.text)
record = {
    "phase": a.phase,
    "preset": "chat",
    "execution": execution,
    "active_context_length": active["context_length"],
    "pool_positions_derived_from_exact_sdk_fixed_contract": 131072,
    "derivation": "SDK 0cd370f model_registration.c:917-947: fixed backing has no memory_pool, so pool_capacity=context_capacity",
    "fixed_total_native_backing_bytes": total,
    "workspace_subset_bytes": workspace,
    "memory_interpretation": "SDK runtime/device/pool.h:94-108: fixed storage is total; workspace is its subset, never added",
    "observation_is_peak": False,
    "program_sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((pkg / "program").glob("*.loom"))
    },
    "status": status,
}
out.write_text(json.dumps(record, indent=1) + "\n")
print("CAPACITY_AUDIT", a.phase, "PASS", json.dumps({
    "preset": "chat", "active_context": active["context_length"],
    "derived_fixed_pool_positions": 131072,
    "total_native_backing": total, "workspace_subset": workspace,
}), flush=True)
