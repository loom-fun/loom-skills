#!/usr/bin/env python3
"""Compare the frozen matched-arm probe summaries without combining counters."""
import argparse
import json
from pathlib import Path


def load(path):
    d = json.loads(Path(path).read_text())
    if not d.get('passed'):
        raise SystemExit(f"probe failed: {path}: {d.get('error')}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('baseline')
    ap.add_argument('candidate')
    a = ap.parse_args()
    b, c = load(a.baseline), load(a.candidate)
    if b.get('target_prompt_tokens') != 65535 or c.get('target_prompt_tokens') != 65535:
        raise SystemExit('both arms must admit exactly 65535 prompt tokens')
    if b.get('retrieval_prompt_sha256') != c.get('retrieval_prompt_sha256'):
        raise SystemExit('long prompts differ between arms')
    for key in ('short_text', 'retrieval_text'):
        if b.get(key) != c.get(key):
            raise SystemExit(f'matched arm output differs: {key}')
    if 'paris' not in b['short_text'].lower():
        raise SystemExit(f"output failed short oracle: {b['short_text']!r}")
    if 'blue' not in b['retrieval_text'].lower():
        raise SystemExit(f"output failed long-retrieval oracle: {b['retrieval_text']!r}")
    result = {
        'matched_prompt_sha256': b['retrieval_prompt_sha256'],
        'matched_outputs_exactly_equal': True,
        'baseline': {'preset': b['preset'], 'execution': b['execution'], 'active': b['active']},
        'candidate': {'preset': c['preset'], 'execution': c['execution'], 'active': c['active']},
        'memory_comparison_policy': 'channels overlap; report independently, never sum RSS/GTT/page-cache/cgroup',
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
