#!/usr/bin/env python3
"""Strict bounded GPU gate; invoked only by the reviewed DevLab job."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('negative', ROOT / '.github/scripts/kbench_negative.py')
negative = importlib.util.module_from_spec(spec); spec.loader.exec_module(negative)


def validate_common(report, count):
    if report.get('format') != 'loom.test.v0':
        raise ValueError('unexpected test report format')
    for key in ('case_count', 'sample_count'):
        if report.get(key) != count:
            raise ValueError(f'{key}: expected {count}')
    for key in ('scenario_count', 'trial_count', 'failed_trial_count', 'skipped_case_count',
                'skipped_scenario_count', 'planning_issue_count', 'xpass_count', 'xfail_mismatch_count'):
        if report.get(key) != 0:
            raise ValueError(f'{key}: must be explicit zero')
    for key in ('trials', 'skipped_cases', 'skipped_scenarios', 'planning_issues'):
        if report.get(key) != []:
            raise ValueError(f'{key}: must be empty')


def validate_positive(report, names, expectations=None):
    validate_common(report, len(names))
    for key in ('failed_sample_count', 'xfail_count'):
        if report.get(key) != 0:
            raise ValueError(f'{key}: positive report must have no failures')
    if report.get('xfails') != []:
        raise ValueError('positive checks cannot be expected failures')
    samples = report.get('samples', [])
    actual = [s.get('case', '').lstrip('@') for s in samples]
    if len(actual) != len(set(actual)) or set(actual) != set(names):
        raise ValueError('missing, extra or repeated positive case')
    for s in samples:
        e = s.get('expectations', {})
        count = e.get('expectation_count')
        if s.get('passed') is not True or type(count) is not int or count < 1:
            raise ValueError('missing positive comparator sample')
        if e.get('passed_count') != count or e.get('failure_count') != 0 or e.get('failures') != []:
            raise ValueError('incomplete/failed positive expectations')
        if expectations is not None and expectations[s['case'].lstrip('@')] != count:
            raise ValueError('positive comparator count changed')


def run(tool, fixtures, out, budget):
    if os.environ.get('GEMMA_E4B_DEVLAB_ACK') != '1':
        raise ValueError('GPU execution is restricted to the reviewed DevLab job')
    manifest = json.loads((fixtures / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if Path(name).name != name or hashlib.sha256((fixtures/name).read_bytes()).hexdigest() != expected:
            raise ValueError('fixture/source hash mismatch: '+name)
    out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + budget
    receipt = {'passed': False, 'budget_seconds': budget,
               'fixture_manifest_sha256':hashlib.sha256((fixtures/'manifest.json').read_bytes()).hexdigest(),
               'target':'amdgpu:gfx1151','phases':[]}
    try:
        for phase in ('positive', 'negative'):
            cmd = [str(tool), str(fixtures/(phase+'.loom')), '--device=amdgpu', '--target=amdgpu:gfx1151']
            if phase == 'negative':
                cmd += ['--xfail='+s for s in manifest['negative_specs']]
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError('kernel gate budget exhausted')
            started = time.monotonic()
            with (out/(phase+'.json')).open('w') as stdout, (out/(phase+'.stderr')).open('w') as stderr:
                p = subprocess.run(cmd, cwd=fixtures, stdout=stdout, stderr=stderr, timeout=left,
                                   env=dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2'))
            if p.returncode:
                raise ValueError(f'{phase}: loom-test exit {p.returncode}; no model arm may run')
            report = json.loads((out/(phase+'.json')).read_text())
            if phase == 'positive':
                validate_positive(report, manifest['positive_cases'], manifest['positive_expectations'])
            else:
                validate_common(report, len(manifest['negative_specs']))
                negative.validate(report, manifest['negative_specs'])
            receipt['phases'].append({'phase':phase,'seconds':time.monotonic()-started,'passed':True})
        receipt['passed'] = True
    except Exception as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (out/'gate.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tool',type=Path,required=True);p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--budget',type=float,default=420)
    a=p.parse_args()
    if not 0 < a.budget <= 420: p.error('budget must be in (0,420] seconds')
    run(a.tool.resolve(),a.fixtures.resolve(),a.out.resolve(),a.budget)
