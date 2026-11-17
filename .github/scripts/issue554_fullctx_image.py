"""Experiment-only completed near-native-context image request, with measured image-token overhead."""
import argparse
import base64
import json
from pathlib import Path
import struct
import sys
import time
import zlib

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'skills/loom-benchmark/scripts'))
from bench_common import Package, doc


def image_url(rgb):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    data = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 224, 224, 8, 2, 0, 0, 0))
    data += chunk(b'IDAT', zlib.compress((b'\0' + bytes(rgb) * 224) * 224)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(data).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pkg')
    ap.add_argument('url')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    pkg = Package(a.pkg)
    model = json.loads((Path(a.pkg) / 'model.json').read_text())
    context = next(iter(model['presets'].values()))['execution']['context_capacity']
    thinking = False if pkg.chat.get('thinking') == 'toggle' else None
    extra = {'chat_template_kwargs': {'enable_thinking': False}} if thinking is False else {}
    calibration_image = image_url((220, 20, 20))
    full_image = image_url((20, 220, 20))

    def request(text, img):
        response = requests.post(a.url + '/v1/chat/completions', timeout=1800, json={
            'model': pkg.model_id,
            'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': img}},
                                                     {'type': 'text', 'text': text}]}],
            'max_tokens': 1, 'temperature': 0, **extra})
        response.raise_for_status()
        return response.json()

    start = time.monotonic()
    calibration = request('Describe.', calibration_image)
    calibration_tokens = calibration['usage']['prompt_tokens']
    plain_calibration = len(pkg.ids(pkg.render_messages([{'role': 'user', 'content': 'Describe.'}], None, thinking)))
    image_overhead = calibration_tokens - plain_calibration
    plain_overhead = len(pkg.ids(pkg.render_messages([{'role': 'user', 'content': 'x'}], None, thinking))) - len(pkg.ids('x'))
    budget = context - 65 - plain_overhead - image_overhead
    assert budget > 0, (context, plain_overhead, image_overhead)
    ids = pkg.ids(doc(32000))
    while len(ids) < budget:
        ids = ids + ids
    text = pkg.tok.decode(ids[:budget])
    result = request(text, full_image)
    tokens = result['usage']['prompt_tokens']
    completed = bool(result.get('choices')) and result['choices'][0].get('finish_reason') is not None
    ok = completed and context - 512 <= tokens < context
    status = requests.get(a.url + '/v1/status', timeout=15).json()
    record = {'completed': completed, 'ok': ok, 'context': context, 'prompt_tokens': tokens,
              'image_token_overhead': image_overhead, 'calibration_prompt_tokens': calibration_tokens,
              'elapsed_seconds': round(time.monotonic() - start, 3), 'usage': result['usage'],
              'finish_reason': result.get('choices', [{}])[0].get('finish_reason'), 'status': status}
    Path(a.out).write_text(json.dumps(record, indent=1) + '\n')
    print('FULLCTX_IMAGE', 'PASS' if ok else 'FAIL', json.dumps({k: record[k] for k in
          ('context', 'prompt_tokens', 'image_token_overhead', 'completed', 'elapsed_seconds')}), flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
