#!/usr/bin/env python3
"""The Strix Halo A/B of the staging cards (devlab-perf, runner strix): write .github/devlab/perf.json for some models.

usage: strix_ab_jobs.py [--gates dev] [--reps 3] [--old-org loom-fun] [--new-org loom-fun-staging] SLUG... > .github/devlab/perf.json

A thin wrapper over ab_jobs.py --device strix-halo (same arguments, the same perf.json byte for byte; see ab_jobs.py for the job layout,
card revision pins and the other device).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_jobs  # noqa: E402

if __name__ == "__main__":
    ab_jobs.main(device="strix-halo")
