#!/usr/bin/env python3
"""Which of the fleet fits a device's memory, from a fit table's sizing columns: usage: fit_device.py DEVICE [--fit .github/devlab/r9700_fit.json] [--json].
Now `.github/scripts/fit.py --device DEVICE --approx [--from TABLE]` (one fit tool; this entry point and verdicts() stay)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, ".github", "scripts"))
import fit  # noqa: E402

verdicts = fit.verdicts

if __name__ == "__main__":
    dev, rest = sys.argv[1], ["--from" if x == "--fit" else x for x in sys.argv[2:]]
    fit.main(["--device", dev, "--approx", *rest])
