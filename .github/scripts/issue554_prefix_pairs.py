"""Experiment-only exact miss/hit checks with prompts longer than two prefill chunks."""
import argparse
import json
from pathlib import Path
import sys
import time

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'skills/loom-benchmark/scripts'))
from bench_common import Package, doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pkg')
    ap.add_argument('url')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    pkg = Package(a.pkg)
    model = json.loads((Path(a.pkg) / 'model.json').read_text())
    execution = next(iter(model['presets'].values()))['execution']
    chunk = execution['prefill_capacity']
    context = execution['context_capacity']
    ids = pkg.ids(doc(32000))
    while len(ids) < 3 * chunk + 257:
        ids = ids + ids
    rows = []
    for rep in range(3):
        for length in (2 * chunk + 137, 3 * chunk + 257):
            assert length + 256 < context
            prompt = f'Fresh prefix pair {time.time_ns()}-{rep}-{length}:\n' + pkg.tok.decode(ids[:length])
            replies = []
            for side in ('miss', 'hit'):
                response = requests.post(a.url + '/v1/chat/completions', timeout=600, json={
                    'model': pkg.model_id, 'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 32, 'temperature': 0})
                response.raise_for_status()
                reply = response.json()
                message = reply['choices'][0]['message']
                replies.append({'side': side, 'text': (message.get('reasoning_content') or '') + '|' + (message.get('content') or ''),
                                'cached': reply['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0),
                                'prompt_tokens': reply['usage']['prompt_tokens']})
            miss, hit = replies
            ok = miss['cached'] <= 64 and hit['cached'] >= chunk and hit['cached'] > miss['cached'] and miss['text'] == hit['text']
            rows.append({'rep': rep, 'chunk': chunk, 'ok': ok, 'replies': replies})
            print('PREFIX_LONG_PAIR', 'PASS' if ok else 'FAIL', 'chunk', chunk, 'prompt_tokens', miss['prompt_tokens'],
                  'cached', [miss['cached'], hit['cached']], 'exact', miss['text'] == hit['text'], flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1) + '\n')
    raise SystemExit(0 if all(r['ok'] for r in rows) else 1)


if __name__ == '__main__':
    main()
