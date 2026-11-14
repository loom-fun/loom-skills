#!/usr/bin/env python3
"""The R9700 per-model measurement job: device_jobs.py --device radeon-r9700 (the original entry point, byte-identical output).

usage: r9700_jobs.py [--fit VERDICT.json (default .github/devlab/r9700_fit.json)] [--card-org loom-fun-staging|loom-fun] [--build] SLUG... > .github/devlab/perf.json
       r9700_jobs.py --all [...]                every model of .github/nightly-models.json that fits the R9700
The generator, its options and what a job holds: device_jobs.py (any registry device with a runner, a pool, a fit table and baselines).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import device_jobs  # noqa: E402

if __name__ == "__main__":
    device_jobs.main(device="radeon-r9700", tool="r9700_jobs.py")
