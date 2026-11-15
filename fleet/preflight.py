"""One up-front check of the python packages the offline fleet tools (build_fleet.py, scope_lint.py --local) and the kit's
build.py need, so a missing package is ONE clear error instead of 51 failed builds.

usage: preflight.py          (exit 1 and the pip line when something is missing)
"""
import importlib.util
import sys

REQUIRED = {"gguf": "gguf", "numpy": "numpy", "jinja2": "jinja2", "yaml": "pyyaml"}   # import name -> pip name


def missing(required=REQUIRED):
    return sorted(pip for mod, pip in required.items() if importlib.util.find_spec(mod) is None)


def require(required=REQUIRED, python=None):
    """Exit with one message naming every missing package (and the interpreter that lacks them)."""
    miss = missing(required)
    if miss:
        py = python or sys.executable
        raise SystemExit(f"missing python packages for {py}: {', '.join(miss)}\n"
                         f"  python3 -m venv W/venv && W/venv/bin/pip install {' '.join(sorted(required.values()))}\n"
                         f"  then run the tool with W/venv/bin/python (see skills/loom-fleet/reference/targeted-fix.md)")


if __name__ == "__main__":
    require()
    print("ok: " + ", ".join(sorted(REQUIRED.values())))
