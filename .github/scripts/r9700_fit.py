#!/usr/bin/env python3
"""Regenerate .github/devlab/r9700_fit.json: fit.py --device radeon-r9700 (the original entry point, same table).

usage: r9700_fit.py [--pkgs DIR (published packages, default ~/lt/r9700/pkgs)] [--work DIR] [--fit OLD.json] [-o NEW.json] [SLUG...]
What the table holds and how it is built: fit.py (any registry device with a memory pool).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit  # noqa: E402

if __name__ == "__main__":
    fit.main(device="radeon-r9700")
