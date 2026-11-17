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


def plan(path, suite_name):
    document = json.loads(Path(path).read_text())
    suite = next(s for s in document['suites'] if s['suite'] == suite_name)
    cases = suite.get('negative_tests', [])
    if not isinstance(cases, list):
        raise ValueError('negative_tests must be a list when declared')
    for case in cases:
        if not isinstance(case, dict) or set(case) != {'file', 'xfail'}:
            raise ValueError('each negative test needs exactly file and xfail')
        if not isinstance(case['file'], str) or not case['file'] or not case['file'].endswith('.loom'):
            raise ValueError('negative test file must be a nonempty .loom path')
        specs = case['xfail']
        if not isinstance(specs, list) or not specs or any(not isinstance(x, str) for x in specs):
            raise ValueError('negative test xfail must be a nonempty string list')
        names = []
        for item in specs:
            parts = item.split('=')
            if len(parts) != 2 or not parts[0].startswith('@') or len(parts[0]) < 2 or parts[1] != 'EXPECT/003':
                raise ValueError('negative test specs require @record=EXPECT/003')
            names.append(parts[0])
        if len(set(names)) != len(names):
            raise ValueError('duplicate negative test specs')
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', nargs=2, metavar=('CONFIG', 'SUITE'))
    ap.add_argument('tool', nargs='?'); ap.add_argument('source', nargs='?'); ap.add_argument('output', nargs='?')
    ap.add_argument('xfail', nargs='*')
    a = ap.parse_args()
    if a.plan:
        try:
            for case in plan(*a.plan):
                print(json.dumps(case))
            return 0
        except (ValueError, KeyError, TypeError, StopIteration, OSError) as e:
            print(f'FAIL negative-control configuration: {e}', file=sys.stderr)
            return 1
    if not (a.tool and a.source and a.output and a.xfail):
        ap.error('tool, source, output and explicit xfail specs are required')
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
