"""Experiment-only proof of the actual ungated E2B program and live draft counters."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'skills/loom-benchmark/scripts'))
from bench_common import server_draft_counters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pkg')
    ap.add_argument('url')
    ap.add_argument('phase', choices=('before', 'after'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    pkg = Path(a.pkg)
    output = Path(a.out)
    counters = server_draft_counters(a.url)
    if a.phase == 'before':
        config = json.loads((pkg / 'llm_config.json').read_text())
        model = json.loads((pkg / 'model.json').read_text())
        control = (pkg / 'program/control.loom').read_text()
        assert 'occupancy / context gate:' not in control and '%occ_below' not in control
        draft = config.get('draft') or {}
        assert draft.get('kind') == 'assist' and draft.get('depth', 0) > 0
        files = {str(p.relative_to(pkg)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((pkg / 'program').glob('*.loom'))}
        record = {'SPEC_OCC_MIN': 0, 'occupancy_gate_absent_in_emitted_control': True, 'draft': draft,
                  'execution': next(iter(model['presets'].values()))['execution'],
                  'program_sha256': files, 'before': counters}
        (output.parent / 'ungated_control.loom').write_text(control)
    else:
        record = json.loads(output.read_text())
        record['after'] = counters
        record['delta'] = {key: counters.get(key, 0) - record['before'].get(key, 0) for key in counters}
        assert record['delta'].get('draft_proposed_tokens', 0) > 0, record['delta']
    output.write_text(json.dumps(record, indent=1) + '\n')
    print('OCC_AUDIT', a.phase, 'PASS', json.dumps(counters), flush=True)


if __name__ == '__main__':
    main()
