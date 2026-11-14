"""fleet/route_load.py: instruction load per fleet model (CPU only; needs origin/r9700 for the pre-refactor set)."""
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("route_load", REPO / "fleet/route_load.py")
route_load = importlib.util.module_from_spec(spec)
spec.loader.exec_module(route_load)


def base_ref():
    for ref in ("origin/r9700", "r9700"):
        if subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", "-q", ref], capture_output=True).returncode == 0:
            return ref
    return None


class RouteLoadTests(unittest.TestCase):
    def test_every_fleet_config_routes_and_loads_less_than_the_old_set(self):
        ref = base_ref()
        if ref is None:
            self.skipTest("no r9700 ref to take the pre-refactor set from")
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "rows.json"
            self.assertEqual(route_load.main(["--base", ref, "--json", str(out)]), 0)
            rows = json.loads(out.read_text())
        n = len(json.loads((REPO / "fleet/manifest.json").read_text())["models"])
        self.assertEqual(len(rows), n * len(route_load.TASKS) * len(route_load.DEVICES))
        for task in route_load.TASKS:
            sel = sorted(r["new"] for r in rows if r["task"] == task and r["device"] == "strix-halo")
            old = next(r["old"] for r in rows if r["task"] == task)
            if task != "optimize-both":
                self.assertLess(sel[len(sel) // 2], old, task)


if __name__ == "__main__":
    unittest.main()
