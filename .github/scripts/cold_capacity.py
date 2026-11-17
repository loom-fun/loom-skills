#!/usr/bin/env python3
"""Opt-in DevLab experiment: sampled cold-load memory and exact context limits.

This does not change gates, presets, or serving policy. `monitor` starts BEFORE
serve.sh, writes a baseline and ready marker, and stops at a marker/deadline.
`probe` checks a loaded SDK020 server against its emitted preset. Raw requests,
responses, status, process identity, and samples are retained beside summary.json.
GPU counters are sampled driver totals, not instantaneous allocation peaks.
RSS, unified GPU memory and workspace are overlapping quantities: never add them.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def read(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def kib_fields(text):
    return {k: int(v) * 1024 for k, v in re.findall(r"^(\w+):\s+(\d+) kB$", text or "", re.M)}


def process(pid, proc=Path('/proc')):
    root = proc / str(pid)
    stat = read(root / 'stat')
    if not stat:
        return None
    fields = stat.rsplit(')', 1)[1].split()
    command = (root / 'cmdline').read_bytes().decode('utf-8', 'replace').split('\0')[:-1]
    return {'pid': pid, 'ppid': int(fields[1]), 'pgrp': int(fields[2]),
            'start_ticks': int(fields[19]), 'argv': command,
            'memory': kib_fields(read(root / 'status'))}


def amd_cards(drm=Path('/sys/class/drm')):
    cards = []
    for card in sorted(drm.glob('card[0-9]*')):
        if not re.fullmatch(r'card\d+', card.name) or read(card / 'device/vendor') != '0x1002':
            continue
        dev = card / 'device'
        cards.append({'card': card.name, 'pci_path': str(dev.resolve()),
                      'device_id': read(dev / 'device'), 'product_name': read(dev / 'product_name'),
                      'uevent': read(dev / 'uevent'), 'sysfs': str(dev)})
    return cards


def sample(pid_file, model, phase_file, cards):
    result = {'wall_time_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(),
              'phase': read(phase_file) or 'cold', 'host': kib_fields(read('/proc/meminfo')),
              'devices': {}, 'process': None}
    pid = read(pid_file)
    if pid and pid.isdigit():
        try:
            p = process(int(pid))
            if p:
                p['identity_matches'] = (any(Path(a).name == 'loom-serve' for a in p['argv'])
                                         and '--model=' + str(model) in p['argv'])
                result['process'] = p
        except (OSError, ValueError):
            result['process_disappeared'] = True
    for card in cards:
        counters = {}
        for key in ('mem_info_vram_used', 'mem_info_vram_total', 'mem_info_vis_vram_used',
                    'mem_info_vis_vram_total', 'mem_info_gtt_used', 'mem_info_gtt_total'):
            value = read(Path(card['sysfs']) / key)
            counters[key] = int(value) if value is not None and value.isdigit() else None
        result['devices'][card['card']] = counters
    return result


def monitor(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cards = amd_cards()
    engines = []
    for entry in Path('/proc').glob('[0-9]*'):
        comm = read(entry / 'comm') or ''
        if comm in ('loom-serve', 'llama-server') or comm.startswith('VLLM::'):
            engines.append({'pid': int(entry.name), 'comm': comm})
    write(out / 'engines_before_launch.json', engines)
    if engines:
        raise RuntimeError('cold-start precondition failed: inference engine remains alive')
    write(out / 'monitor_metadata.json', {'interval_seconds': a.interval, 'cards': cards,
          'pid_file': a.pid_file, 'model': a.model,
          'limitations': ['100ms sampled driver totals may miss transient GPU peaks.',
                         'All AMD cards recorded; selected device is identified separately by server status.',
                         'VmHWM is process RSS high-water, not GPU/host combined usage.',
                         'RSS, VRAM/GTT and unified host RAM overlap; do not sum them.']})
    deadline = time.monotonic() + a.timeout
    maxima = {}; minima = {}; identities = {}; count = 0
    with (out / 'memory.jsonl').open('w', buffering=1) as f:
        while True:
            s = sample(a.pid_file, a.model, out / 'phase', cards)
            f.write(json.dumps(s) + '\n'); count += 1
            phase = s['phase']; maxima.setdefault(phase, {}); minima.setdefault(phase, {})
            channels = {'host.MemAvailable': s['host'].get('MemAvailable')}
            for card, values in s['devices'].items():
                channels.update({card + '.' + k: v for k, v in values.items()})
            p = s['process']
            if p:
                key = str(p['pid']) + ':' + str(p['start_ticks'])
                identities[key] = {k: v for k, v in p.items() if k != 'memory'}
                if p['identity_matches']:
                    channels.update({key + '.' + k: v for k, v in p['memory'].items() if k in ('VmRSS', 'VmHWM', 'RssAnon', 'RssFile', 'RssShmem')})
            for key, value in channels.items():
                if value is not None:
                    maxima[phase][key] = max(value, maxima[phase].get(key, value))
                    minima[phase][key] = min(value, minima[phase].get(key, value))
            if count == 1:
                write(out / 'baseline.json', s)
                (out / 'monitor.ready').touch()
            if (out / 'monitor.stop').exists() or time.monotonic() >= deadline:
                break
            time.sleep(a.interval)
    write(out / 'memory_summary.json', {'samples': count, 'maxima_by_phase': maxima,
          'minima_by_phase': minima, 'process_identities': identities,
          'stopped_by_marker': (out / 'monitor.stop').exists(),
          'identity_ok': len(identities) == 1 and all(p['identity_matches'] for p in identities.values()),
          'amd_cards_observed': bool(cards),
          'driver_used_counters_observed': any(k.endswith(('.mem_info_vram_used', '.mem_info_gtt_used'))
                                               for channels in maxima.values() for k in channels),
          'true_instantaneous_gpu_peak': False})


def status_contract(body, model_id, execution):
    models = [m for m in body['data'] if m['id'] == model_id]
    if len(models) != 1:
        raise ValueError('status must contain exactly the intended model')
    model = models[0]; loom = model['loom']
    if loom['registration'] != 'ready' or model['context_length'] != execution['context_capacity']:
        raise ValueError('active context/registration differs from emitted preset')
    pools = [p for p in body['server']['resources']['pools'] if p['key'] == loom['pool']]
    if len(pools) != 1 or pools[0]['memory']['backing'] != 'fixed':
        raise ValueError('expected exactly one selected fixed-backed pool')
    memory = pools[0]['memory']
    if memory['storage']['committed_bytes'] <= 0:
        raise ValueError('fixed backing is not committed')
    return {'context': model['context_length'], 'device': loom['device'], 'pool': loom['pool'],
            'fixed_storage_total_bytes': memory['storage']['committed_bytes'],
            'workspace_subset_bytes': memory['workspace']['committed_bytes'],
            'pool_positions': execution['context_capacity'],
            'pool_positions_basis': 'SDK020 fixed backing sets pool_capacity=context_capacity; not a separate observed token counter',
            'memory_is_peak': False}


def completed(status, body, prompt_tokens):
    if status != 200 or not isinstance(body, dict) or 'error' in body:
        raise ValueError('completion returned an error/non-200')
    choices = body.get('choices', [])
    usage = body.get('usage', {})
    if (len(choices) != 1 or choices[0].get('finish_reason') not in ('stop', 'length')
            or usage.get('prompt_tokens') != prompt_tokens or usage.get('completion_tokens') != 1):
        raise ValueError('missing completed one-token answer or actual token counts differ')
    return usage


def rejected(status, body):
    error = body.get('error', {}) if isinstance(body, dict) else {}
    if (status != 400 or error.get('type') != 'loom_request_error'
            or error.get('code') != 'OUT_OF_RANGE'
            or 'request exceeds the available context size' not in error.get('message', '')):
        raise ValueError('not the structured SDK020 context-overflow rejection')


def calibrate(target, count):
    """Bounded calibration uses server-rendered counts, never assumed tokenizer math."""
    for unit in (' x', ' a', ' hello'):
        base = count(unit)
        if count(unit * 9) - base != 8:
            continue
        n = target - base + 1
        for _ in range(4):
            if n < 1:
                break
            prompt = unit * n; actual = count(prompt)
            if actual == target:
                return prompt
            n += target - actual
    raise ValueError('could not calibrate exact server-counted prompt within bounded attempts')


class Requests:
    def __init__(self, url, out, timeout):
        self.url, self.out, self.timeout, self.sequence = url, out, timeout, 0

    def call(self, name, path, payload=None):
        self.sequence += 1; stem = f'{self.sequence:02d}_{name}'
        if payload is not None:
            write(self.out / (stem + '.request.json'), payload)
        req = urllib.request.Request(self.url + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'})
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                code, raw = response.status, response.read()
        except urllib.error.HTTPError as error:
            code, raw = error.code, error.read()
        (self.out / (stem + '.response')).write_bytes(raw)
        write(self.out / (stem + '.meta.json'), {'http_status': code, 'path': path,
              'elapsed_seconds': time.monotonic() - start, 'request_name': stem})
        return code, json.loads(raw)


def probe(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    package = Path(a.package); model = json.loads((package / 'model.json').read_text())
    preset = a.preset or next(iter(model['presets']))
    execution = model['presets'][preset]['execution']; model_id = model['model']['id']
    request = Requests(a.url, out, a.timeout)
    result = {'passed': False, 'preset': preset, 'execution': execution, 'model_id': model_id,
              'program_sha256': {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(package.rglob('*.loom')) if 'gen' not in p.relative_to(package).parts},
              'package_metadata_sha256': {n: hashlib.sha256((package / n).read_bytes()).hexdigest()
                                         for n in ('model.json', 'model.json.lock', 'llm_config.json') if (package / n).exists()},
              'sdk_context_contract': 'input_tokens + max_tokens - 1 <= context_capacity (SDK020 input.c:63)',
              'phase': 'status'}
    try:
        code, body = request.call('status_before', '/v1/status')
        if code != 200:
            raise ValueError('status is not HTTP200')
        result['active'] = status_contract(body, model_id, execution)
        ctx = execution['context_capacity']
        def payload(prompt, maximum=1):
            return {'model': model_id, 'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': maximum, 'stream': False, 'temperature': 0,
                    'chat_template_kwargs': {'enable_thinking': False}}
        def count(prompt):
            code, body = request.call('count', '/v1/chat/completions/input_tokens', payload(prompt))
            if code != 200 or type(body.get('input_tokens')) is not int:
                raise ValueError('token-count endpoint failed')
            return body['input_tokens']
        (out / 'phase').write_text('full_context')
        result['phase'] = 'logical_full_limit'
        prompt = calibrate(ctx - 1, count)
        code, body = request.call('logical_full_limit', '/v1/chat/completions', payload(prompt))
        result['logical_full_limit_usage'] = completed(code, body, ctx - 1)
        result['phase'] = 'sdk_stored_full_limit'
        prompt = calibrate(ctx, count)
        code, body = request.call('sdk_stored_full_limit', '/v1/chat/completions', payload(prompt))
        result['sdk_stored_full_limit_usage'] = completed(code, body, ctx)
        result['phase'] = 'over_limit'
        code, body = request.call('over_limit', '/v1/chat/completions', payload(prompt, 2))
        rejected(code, body); result['over_limit'] = body
        result['phase'] = 'health_after'
        short = 'Reply with one word.'; n = count(short)
        code, body = request.call('health_after', '/v1/chat/completions', payload(short))
        result['health_after_usage'] = completed(code, body, n)
        code, body = request.call('status_after', '/v1/status')
        if code != 200:
            raise ValueError('final status is not HTTP200')
        result['active_after'] = status_contract(body, model_id, execution)
        result['passed'] = True; result['phase'] = 'complete'
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        write(out / 'summary.json', result)
    print(json.dumps({'passed': result['passed'], 'phase': result['phase'], 'error': result.get('error')}), flush=True)
    return 0 if result['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    m = sub.add_parser('monitor'); m.add_argument('--pid-file', required=True); m.add_argument('--model', required=True)
    m.add_argument('--interval', type=float, default=0.1); m.add_argument('--timeout', type=float, default=10800)
    p = sub.add_parser('probe'); p.add_argument('package'); p.add_argument('url'); p.add_argument('--preset')
    p.add_argument('--timeout', type=float, default=3600)
    for command in (m, p):
        command.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.timeout <= 0 or (args.mode == 'monitor' and args.interval < 0.05):
        parser.error('positive timeout and sampling interval >=50ms required')
    return monitor(args) if args.mode == 'monitor' else probe(args)


if __name__ == '__main__':
    raise SystemExit(main())
