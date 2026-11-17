#!/usr/bin/env python3
"""Run explicit kernel negative controls; strict comparator failures are required.

This is separate from the positive test path. Compile/setup errors, diagnostic
mismatches, skipped samples and unexpected passes all fail the experiment.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def validate(report, specs):
    pairs = [s.split('=', 1) for s in specs]
    expected = dict(pairs)
    if len(expected) != len(pairs):
        raise ValueError('duplicate expected-failure specs')
    if not expected or any(d != 'EXPECT/003' for d in expected.values()):
        raise ValueError('negative controls require explicit EXPECT/003 comparator diagnostics')
    records = report.get('xfails', [])
    outcomes = {r['record']: r for r in records}
    if len(outcomes) != len(records):
        raise ValueError('duplicate expected-failure records')
    if set(outcomes) != set(expected):
        raise ValueError('missing or extra expected-failure records')
    for name, diagnostic in expected.items():
        r = outcomes[name]
        if r.get('outcome') != 'xfail' or r.get('diagnostic') != diagnostic or r.get('policy') != 'strict':
            raise ValueError(f'{name}: expected strict comparator failure, got {r}')
        samples = [s for s in report.get('samples', []) if s['case'].lstrip('@') == name.lstrip('@')]
        if len(samples) != 1 or any(s.get('passed') is not False or not s.get('expectations', {}).get('failures') for s in samples):
            raise ValueError(f'{name}: missing raw failing comparator sample (compile failure is insufficient)')
    if {s['case'].lstrip('@') for s in report.get('samples', [])} != {n.lstrip('@') for n in expected}:
        raise ValueError('unexpected or skipped negative-control sample')
    return len(report['samples'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('tool'); ap.add_argument('source'); ap.add_argument('output')
    ap.add_argument('xfail', nargs='+')
    a = ap.parse_args()
    cmd = [a.tool, a.source, '--device=amdgpu'] + ['--xfail=' + s for s in a.xfail]
    r = subprocess.run(cmd, text=True, capture_output=True)
    Path(a.output).write_text(r.stdout)
    Path(a.output + '.err').write_text(r.stderr)
    if r.returncode:
        print(f'FAIL negative controls: loom-test exit {r.returncode}', file=sys.stderr)
        return 1
    try:
        n = validate(json.loads(r.stdout), a.xfail)
    except (ValueError, KeyError, TypeError) as e:
        print(f'FAIL negative controls: {e}', file=sys.stderr)
        return 1
    print(f'PASS {n} explicit negative controls: strict EXPECT/003 with raw comparator failures')
    return 0


if __name__ == '__main__':
    sys.exit(main())
