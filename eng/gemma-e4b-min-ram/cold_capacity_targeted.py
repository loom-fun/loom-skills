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
import statistics
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
                      'vendor_id': read(dev / 'vendor'), 'device_id': read(dev / 'device'),
                      'product_name': read(dev / 'product_name'),
                      'uevent': read(dev / 'uevent'), 'sysfs': str(dev)})
    return cards


def pci_bdf(card):
    match = re.search(r'^PCI_SLOT_NAME=([0-9a-fA-F]{4}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])$',
                      card.get('uevent') or '', re.M)
    if not match:
        raise ValueError('AMD DRM card lacks a parseable PCI_SLOT_NAME')
    domain, bus, device, function = (int(value, 16) for value in match.groups())
    return {'domain': domain, 'bus': bus, 'device': device, 'function': function,
            'bdf': f'{domain:04x}:{bus:02x}:{device:02x}.{function}',
            'kfd_location_id_low16': (bus << 8) | (device << 3) | function}


def parse_kfd_properties(text):
    properties = {}
    for line in (text or '').splitlines():
        fields = line.split(None, 1)
        if len(fields) == 2:
            properties[fields[0]] = fields[1]
    return properties


def raw_hardware_inventory(drm, kfd):
    cards = []
    for card in amd_cards(Path(drm)):
        cards.append({key: value for key, value in card.items() if key != 'sysfs'})
    nodes = []
    for properties_path in sorted((Path(kfd) / 'nodes').glob('*/properties')):
        nodes.append({'node': properties_path.parent.name,
                      'properties_path': str(properties_path),
                      'properties_raw': read(properties_path)})
    return {'drm_cards': cards, 'kfd_nodes': nodes}


def property_int(properties, key):
    raw = properties.get(key)
    if raw is None:
        return None
    try:
        return int(raw, 0) if raw.lower().startswith('0x') else int(raw, 10)
    except ValueError as error:
        raise ValueError(f'KFD {key} is not an integer: {raw!r}') from error


def gfx1151_hardware_identity(drm=Path('/sys/class/drm'),
                              kfd=Path('/sys/class/kfd/kfd/topology')):
    """Match the sole AMD DRM card to one KFD GPU node and require gfx_target_version 110501."""
    cards = amd_cards(drm)
    if len(cards) != 1:
        raise ValueError(f'expected exactly one AMD DRM card, observed {len(cards)}')
    card = cards[0]
    bdf = pci_bdf(card)
    vendor_raw = read(Path(card['sysfs']) / 'vendor')
    device_raw = card.get('device_id')
    if not vendor_raw or not device_raw:
        raise ValueError('AMD DRM card is missing vendor/device PCI identifiers')
    try:
        vendor_id, device_id = int(vendor_raw, 16), int(device_raw, 16)
    except ValueError as error:
        raise ValueError('AMD DRM vendor/device PCI identifiers are malformed') from error

    gpu_nodes = []
    for properties_path in sorted((Path(kfd) / 'nodes').glob('*/properties')):
        raw = read(properties_path)
        props = parse_kfd_properties(raw)
        simd = property_int(props, 'simd_count') if 'simd_count' in props else None
        target = property_int(props, 'gfx_target_version') if 'gfx_target_version' in props else None
        if (simd or 0) > 0 or (target or 0) > 0:
            gpu_nodes.append({'node': properties_path.parent.name,
                              'properties_path': str(properties_path),
                              'properties_raw': raw,
                              'properties': props})
    if not gpu_nodes:
        raise ValueError(f'no GPU nodes found under {Path(kfd) / "nodes"}')

    # KFD location_id stores PCI B:D.F in bits [15:0]; domain is a separate property.
    # AMD SMI documents the same KFD-to-PCI field mapping.
    target_nodes = []
    for node in gpu_nodes:
        props = node['properties']
        node_vendor = property_int(props, 'vendor_id') if 'vendor_id' in props else None
        node_device = property_int(props, 'device_id') if 'device_id' in props else None
        node_target = property_int(props, 'gfx_target_version') if 'gfx_target_version' in props else None
        if node_vendor is None or node_device is None or node_target is None:
            raise ValueError(f'KFD GPU node {node["node"]} is missing vendor/device/gfx_target_version')
        node_domain = property_int(props, 'domain') if 'domain' in props else None
        node_location = property_int(props, 'location_id') if 'location_id' in props else None
        if node_vendor == vendor_id and node_device == device_id:
            if (node_domain is None) != (node_location is None):
                raise ValueError(f'KFD GPU node {node["node"]} has partial domain/location identity')
            if node_domain is None:
                if len(gpu_nodes) != 1:
                    raise ValueError(f'KFD GPU node {node["node"]} lacks PCI identity among multiple GPU nodes')
                target_nodes.append((node, 'sole_gpu_node_vendor_device'))
            elif node_domain == bdf['domain'] and (node_location & 0xffff) == bdf['kfd_location_id_low16']:
                target_nodes.append((node, 'domain_location_id'))
        elif (node_domain == bdf['domain'] and node_location is not None
              and (node_location & 0xffff) == bdf['kfd_location_id_low16']):
            raise ValueError(f'KFD GPU node {node["node"]} conflicts with DRM vendor/device at the same BDF')

    if len(target_nodes) != 1:
        raise ValueError(f'expected one KFD GPU node matching {bdf["bdf"]} vendor={vendor_raw} device={device_raw}; '
                         f'observed {len(target_nodes)} matches among {len(gpu_nodes)} GPU nodes')
    node, match_method = target_nodes[0]
    target_version = property_int(node['properties'], 'gfx_target_version')
    if target_version != 110501:
        raise ValueError(f'matched KFD node reports gfx_target_version={target_version}, expected 110501 (gfx1151)')
    return {'passed': True, 'source': 'read-only KFD topology sysfs',
            'architecture': 'gfx1151', 'gfx_target_version': target_version,
            'match_method': match_method,
            'drm_card': {'card': card['card'], 'pci_path': card['pci_path'], 'pci_bdf': bdf['bdf'],
                         'vendor_id': vendor_raw, 'device_id': device_raw,
                         'uevent_raw': card.get('uevent')},
            'kfd_node': node, 'gpu_nodes_seen': gpu_nodes}


def sample(pid_file, model, phase_file, cards, cgroup_root=None):
    result = {'wall_time_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(),
              'phase': read(phase_file) or 'cold', 'host': kib_fields(read('/proc/meminfo')),
              'devices': {}, 'process': None, 'cgroup': None}
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
    if cgroup_root:
        root = Path(cgroup_root)
        try:
            stat = {}
            for line in (root / 'memory.stat').read_text().splitlines():
                key, value = line.split()
                stat[key] = int(value)
            current = (root / 'memory.current').read_text().strip()
            result['cgroup'] = {'root': str(root), 'memory_current_bytes': int(current),
                                'memory_stat_bytes': stat}
        except (OSError, ValueError):
            result['cgroup'] = {'root': str(root), 'read_error': True}
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
    write(out / 'monitor_metadata.json', {'interval_seconds': a.interval, 'cards': cards,
          'pid_file': a.pid_file, 'model': a.model, 'cgroup_root': a.cgroup_root,
          'limitations': ['100ms sampled driver totals may miss transient GPU peaks.',
                         'All AMD cards recorded; selected architecture is independently matched through KFD topology.',
                         'VmHWM is process RSS high-water, not GPU/host combined usage.',
                         'RSS, VRAM/GTT and unified host RAM overlap; do not sum them.']})
    deadline = time.monotonic() + a.timeout
    maxima = {}; minima = {}; identities = {}; count = 0; stop_by_marker = False; deadline_reached = False
    with (out / 'memory.jsonl').open('w', buffering=1) as f:
        while True:
            s = sample(a.pid_file, a.model, out / 'phase', cards, a.cgroup_root)
            f.write(json.dumps(s) + '\n'); count += 1
            phase = s['phase']; maxima.setdefault(phase, {}); minima.setdefault(phase, {})
            channels = {'host.' + k: s['host'].get(k) for k in
                        ('MemTotal', 'MemFree', 'MemAvailable', 'Cached', 'SReclaimable', 'Shmem')}
            for card, values in s['devices'].items():
                channels.update({card + '.' + k: v for k, v in values.items()})
            cg = s.get('cgroup')
            if cg and 'memory_current_bytes' in cg:
                channels['cgroup.memory_current_bytes'] = cg['memory_current_bytes']
                channels.update({'cgroup.' + k: v for k, v in cg['memory_stat_bytes'].items()})
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
            if (out / 'monitor.stop').exists():
                stop_by_marker = True
                break
            if time.monotonic() >= deadline:
                deadline_reached = True
                break
            time.sleep(a.interval)
    write(out / 'memory_summary.json', {'samples': count, 'maxima_by_phase': maxima,
          'minima_by_phase': minima, 'process_identities': identities,
          'stopped_by_marker': stop_by_marker, 'deadline_reached': deadline_reached,
          'identity_ok': len(identities) == 1 and all(p['identity_matches'] for p in identities.values()),
          'amd_cards_observed': bool(cards), 'true_instantaneous_gpu_peak': False})


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
            'context_capacity_tokens': execution['context_capacity'],
            'pool_positions': None,
            'pool_positions_basis': 'not directly reported by status; infer only if storage committed bytes are confirmed to be KV-only',
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


def calibrate_retrieval(target, count, answer='blue'):
    """Calibrate a long retrieval prompt using the server's own chat-template counter."""
    prefix = f"At the start of this note the secret code is {answer}. "
    suffix = f"\n\nAfter reading the note, what is the secret code? Reply with only that one word."
    unit = ' x'
    base = count(prefix + suffix)
    if count(prefix + unit * 9 + suffix) - base != 9:
        raise ValueError('repeated filler is not one stable token per unit in this chat template')
    n = target - base
    for _ in range(8):
        if n < 0:
            break
        prompt = prefix + unit * n + suffix
        actual = count(prompt)
        if actual == target:
            return prompt
        n += target - actual
    raise ValueError('could not calibrate exact retrieval prompt within bounded attempts')


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
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            body = {'_raw_text': raw.decode('utf-8', 'replace')}
        return code, body


def short_with_readiness_retry(request, payload, out, budget=120.0, interval=1.0):
    """The first real short oracle may trigger lazy weight loading. Retry only HTTP 503."""
    started = time.monotonic(); deadline = started + budget
    attempts = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'first short oracle readiness exceeded {budget:.1f}s')
        previous_timeout = request.timeout
        request.timeout = min(previous_timeout, remaining)
        code, body = request.call('short_reference_attempt_%02d' % (len(attempts) + 1),
                                  '/v1/chat/completions', payload)
        request.timeout = previous_timeout
        meta_path = out / ('%02d_short_reference_attempt_%02d.meta.json' %
                           (request.sequence, len(attempts) + 1))
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        attempts.append({'attempt': len(attempts) + 1, 'http_status': code,
                         'elapsed_seconds': meta.get('elapsed_seconds'),
                         'elapsed_since_first_attempt_seconds': time.monotonic() - started,
                         'body': body})
        write(out / 'readiness_attempts.json', {'budget_seconds': budget,
              'attempts': attempts, 'loaded_after_seconds': time.monotonic() - started if code == 200 else None})
        if code != 503:
            return code, body, attempts
        if time.monotonic() >= deadline:
            raise TimeoutError(f'first short oracle remained HTTP 503 for {budget:.1f}s')
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))


def validate_monitor(summary, metadata, probe_summary, hardware_identity):
    """Fail closed on monitor/process/device integrity; never label overlapping metrics TOTAL RAM."""
    if not summary or not metadata or not probe_summary:
        raise ValueError('missing monitor metadata/summary or probe summary')
    if not summary.get('stopped_by_marker') or summary.get('deadline_reached') or summary.get('samples', 0) < 1:
        raise ValueError('monitor did not produce samples and stop by marker')
    identities = summary.get('process_identities') or {}
    if not summary.get('identity_ok') or len(identities) != 1:
        raise ValueError('monitor lacks one stable matching server PID/start_ticks identity')
    identity = next(iter(identities.values()))
    if not identity.get('identity_matches') or not identity.get('start_ticks'):
        raise ValueError('server process identity is not verified')
    cards = metadata.get('cards') or []
    if not summary.get('amd_cards_observed') or len(cards) != 1:
        raise ValueError(f'expected one AMD sysfs device for selected-card attribution; observed {len(cards)}')
    active = (probe_summary.get('active') or {}).get('device', '')
    if not isinstance(active, str) or not active.strip():
        raise ValueError('probe status has no device label')
    explicit_arch = re.search(r'gfx\d+', active.lower())
    if explicit_arch and explicit_arch.group(0) != 'gfx1151':
        raise ValueError(f'probe status conflicts with required gfx1151: {active!r}')
    if not hardware_identity or not hardware_identity.get('passed'):
        raise ValueError('independent KFD hardware identity proof is missing or failed')
    if hardware_identity.get('architecture') != 'gfx1151' or hardware_identity.get('gfx_target_version') != 110501:
        raise ValueError('independent KFD hardware identity does not prove gfx1151')
    proof_card = hardware_identity.get('drm_card') or {}
    card = cards[0]
    current_bdf = pci_bdf(card)['bdf']
    pci_id = re.search(r'^PCI_ID=([0-9a-fA-F]{4}):([0-9a-fA-F]{4})$', card.get('uevent') or '', re.M)
    if not pci_id:
        raise ValueError('monitor metadata lacks a parseable PCI_ID for the selected AMD card')
    if (proof_card.get('pci_bdf') != current_bdf
            or proof_card.get('vendor_id', '').lower() != ('0x' + pci_id.group(1).lower())
            or proof_card.get('device_id', '').lower() != ('0x' + pci_id.group(2).lower())):
        raise ValueError('KFD gfx1151 proof does not match the monitored DRM card BDF/vendor/device')
    maxima = summary.get('maxima_by_phase') or {}
    observed = {key: value for phase in maxima.values() for key, value in phase.items()}
    gpu_keys = [key for key, value in observed.items()
                if key.endswith(('.mem_info_gtt_used', '.mem_info_vram_used', '.mem_info_vis_vram_used'))
                and value is not None]
    if not gpu_keys:
        raise ValueError('AMD device observed but all GPU memory counters are unknown')
    required_host = ('host.MemTotal', 'host.MemAvailable', 'host.Cached')
    coverage = {key: ('observed' if any(key in phase for phase in maxima.values()) else 'unknown')
                for key in required_host}
    return {'host_channel_coverage': coverage, 'amd_card': cards[0], 'amd_counter_keys': gpu_keys,
            'selected_device': active, 'hardware_identity': hardware_identity,
            'process_identity': identity,
            'total_ram_verified': False,
            'total_ram_reason': 'RSS, UMA/GTT, file pages and cgroup may overlap; no union attribution is available.'}


def card_identity(card):
    return {key: card.get(key) for key in ('card', 'pci_path', 'device_id')}


def device_counters(card):
    sysfs = Path(card['sysfs'])
    values = {}
    for key in ('mem_info_vram_used', 'mem_info_gtt_used'):
        raw = read(sysfs / key)
        values[key] = int(raw) if raw is not None and raw.isdigit() else None
    mem = kib_fields(read('/proc/meminfo'))
    return {'wall_time_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(),
            'identity': card_identity(card), 'vram_used_bytes': values['mem_info_vram_used'],
            'gtt_used_bytes': values['mem_info_gtt_used'],
            'host_memavailable_bytes': mem.get('MemAvailable')}


def idle_baseline(samples, stable_range_bytes):
    if not samples:
        raise ValueError('no pre-arm device samples')
    identity = samples[0]['identity']
    if any(sample['identity'] != identity for sample in samples):
        raise ValueError('pre-arm GPU PCI identity changed during baseline capture')
    medians = {}
    for key in ('vram_used_bytes', 'gtt_used_bytes'):
        values = [sample[key] for sample in samples]
        if any(value is None for value in values):
            raise ValueError(f'pre-arm {key} counter is unknown')
        if max(values) - min(values) > stable_range_bytes:
            raise ValueError(f'pre-arm {key} counter is not stable within {stable_range_bytes} bytes')
        medians[key] = int(statistics.median(values))
    return {'identity': identity, 'baseline': medians,
            'baseline_stability_range_bytes': stable_range_bytes,
            'host_memavailable_bytes_range': host_available_range(samples)}


def host_available_range(samples):
    values = [sample.get('host_memavailable_bytes') for sample in samples
              if sample.get('host_memavailable_bytes') is not None]
    return {'minimum': min(values), 'maximum': max(values)} if values else None


def settled_window(idle, samples, tolerance_bytes, stable_range_bytes, stable_samples):
    if len(samples) < stable_samples:
        return False
    window = samples[-stable_samples:]
    if any(sample['identity'] != idle['identity'] for sample in window):
        raise ValueError('AMD target PCI identity changed during inter-arm settle')
    for key in ('vram_used_bytes', 'gtt_used_bytes'):
        values = [sample[key] for sample in window]
        if any(value is None for value in values):
            raise ValueError(f'inter-arm {key} counter became unknown')
        if any(abs(value - idle['baseline'][key]) > tolerance_bytes for value in values):
            return False
        if max(values) - min(values) > stable_range_bytes:
            return False
    return True


def wait_for_device_settle(idle, read_sample, timeout_seconds=60.0, interval_seconds=1.0,
                           tolerance_bytes=64*1024*1024, stable_range_bytes=16*1024*1024,
                           stable_samples=5, now=time.monotonic, sleep=time.sleep):
    start = now(); deadline = start + timeout_seconds; samples = []
    result = {'passed': False, 'timeout_seconds': timeout_seconds,
              'tolerance_bytes': tolerance_bytes, 'stable_range_bytes': stable_range_bytes,
              'stable_samples_required': stable_samples, 'samples': samples}
    while now() < deadline:
        sample = read_sample(); samples.append(sample)
        if now() > deadline:
            break
        if sample['identity'] != idle['identity']:
            result['error'] = 'AMD target PCI identity changed during inter-arm settle'
            break
        try:
            if settled_window(idle, samples, tolerance_bytes, stable_range_bytes, stable_samples):
                result['passed'] = True
                result['settled_after_seconds'] = now() - start
                break
        except ValueError as error:
            result['error'] = str(error); break
        remaining = deadline - now()
        if remaining <= 0:
            break
        sleep(min(interval_seconds, remaining))
    result['host_memavailable_bytes_range'] = host_available_range(samples)
    result['host_memory_used_as_settle_gate'] = False
    result['host_memory_interpretation'] = 'recorded for context only; no global cache eviction or total-RAM claim'
    if not result['passed']:
        result.setdefault('error', 'VRAM/GTT did not return within idle tolerance for the required stable sample window')
        result['elapsed_seconds'] = min(now() - start, timeout_seconds)
    return result


def capture_idle(a):
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    cards = amd_cards(); samples = []; result = {'passed': False, 'samples': samples}
    try:
        if len(cards) != 1:
            raise ValueError(f'expected one AMD target card before arm1; observed {len(cards)}')
        card = cards[0]
        for index in range(a.samples):
            sample = device_counters(card); samples.append(sample)
            if index + 1 < a.samples:
                time.sleep(a.interval)
        result.update(idle_baseline(samples, a.stable_range_bytes))
        result['card'] = {k:v for k,v in card.items() if k != 'sysfs'}
        result['passed'] = True
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    write(out, result)
    print(json.dumps({'passed':result['passed'], 'error':result.get('error')}), flush=True)
    return 0 if result['passed'] else 1


def settle(a):
    idle_path = Path(a.idle); out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    try:
        idle = json.loads(idle_path.read_text())
        if not idle.get('passed'):
            raise ValueError('pre-arm idle counter baseline did not pass')
        cards = amd_cards()
        if len(cards) != 1 or card_identity(cards[0]) != idle['identity']:
            raise ValueError('AMD target card/PCI identity differs from pre-arm idle baseline')
        result = wait_for_device_settle(idle, lambda: device_counters(cards[0]),
              timeout_seconds=a.timeout, interval_seconds=a.interval,
              tolerance_bytes=a.tolerance_bytes, stable_range_bytes=a.stable_range_bytes,
              stable_samples=a.stable_samples)
    except Exception as error:
        result = {'passed':False, 'error':f'{type(error).__name__}: {error}', 'samples':[],
                  'host_memory_used_as_settle_gate':False,
                  'host_memory_interpretation':'not a settle gate; no cache eviction or total-RAM claim'}
    write(out,result)
    print(json.dumps({'passed':result['passed'], 'error':result.get('error'),
                      'elapsed_seconds':result.get('elapsed_seconds',result.get('settled_after_seconds'))}), flush=True)
    return 0 if result['passed'] else 1


def probe(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    package = Path(a.package); model = json.loads((package / 'model.json').read_text())
    preset = a.preset or next(iter(model['presets']))
    execution = model['presets'][preset]['execution']; model_id = model['model']['id']
    request = Requests(a.url, out, a.timeout)
    result = {'passed': False, 'preset': preset, 'execution': execution, 'model_id': model_id,
              'program_sha256': {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(package.rglob('*.loom'))},
              'sdk_context_contract': 'input_tokens + max_tokens - 1 <= context_capacity (SDK020 input.c:63)',
              'phase': 'status'}
    try:
        code, body = request.call('status_before', '/v1/status')
        if code != 200:
            raise ValueError('status is not HTTP200')
        result['active'] = status_contract(body, model_id, execution)
        ctx = execution['context_capacity']
        target = a.target_prompt_tokens if a.target_prompt_tokens is not None else ctx
        if target < 1 or target > ctx:
            raise ValueError('target prompt token count must fit the emitted preset context')
        result['target_prompt_tokens'] = target
        def payload(prompt, maximum=1):
            return {'model': model_id, 'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': maximum, 'stream': False, 'temperature': 0,
                    'chat_template_kwargs': {'enable_thinking': False}}
        def count(prompt):
            code, body = request.call('count', '/v1/chat/completions/input_tokens', payload(prompt))
            if code != 200 or type(body.get('input_tokens')) is not int:
                raise ValueError('token-count endpoint failed')
            return body['input_tokens']
        (out / 'phase').write_text('short_reference')
        result['phase'] = 'short_reference'
        short = 'What is the capital of France? Answer with one word.'
        short_tokens = count(short)
        original_timeout = request.timeout
        request.timeout = min(original_timeout, 30.0)
        try:
            code, body, readiness = short_with_readiness_retry(request, payload(short),
                                                                out, budget=120.0, interval=1.0)
        finally:
            request.timeout = original_timeout
        result['readiness_attempts'] = readiness
        result['short_usage'] = completed(code, body, short_tokens)
        result['short_text'] = body['choices'][0]['message']['content']
        result['short_elapsed_seconds'] = sum(a['elapsed_seconds'] or 0 for a in readiness)
        if 'paris' not in result['short_text'].strip().lower():
            raise ValueError('short deterministic reference did not answer Paris')
        (out / 'phase').write_text('full_context_retrieval')
        result['phase'] = 'target_context_retrieval'
        prompt = calibrate_retrieval(target, count)
        result['retrieval_prompt_sha256'] = hashlib.sha256(prompt.encode()).hexdigest()
        code, body = request.call('target_context_retrieval', '/v1/chat/completions', payload(prompt))
        result['target_context_usage'] = completed(code, body, target)
        result['retrieval_text'] = body['choices'][0]['message']['content']
        result['target_context_elapsed_seconds'] = json.loads((out / f'{request.sequence:02d}_target_context_retrieval.meta.json').read_text())['elapsed_seconds']
        result['prompt_tokens_per_second'] = target / result['target_context_elapsed_seconds']
        if 'blue' not in result['retrieval_text'].strip().lower():
            raise ValueError('64k retrieval did not return the planted code word')
        # Boundary check is only meaningful when the requested target equals the preset capacity.
        if target == ctx:
            result['phase'] = 'over_limit'
            code, body = request.call('over_limit', '/v1/chat/completions', payload(prompt, 2))
            rejected(code, body); result['over_limit'] = body
        result['phase'] = 'health_after'
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
    m.add_argument('--interval', type=float, default=0.1); m.add_argument('--timeout', type=float, default=10800); m.add_argument('--cgroup-root', default=None, help='isolated cgroup-v2 directory; records memory.current/stat including file cache')
    p = sub.add_parser('probe'); p.add_argument('package'); p.add_argument('url'); p.add_argument('--preset'); p.add_argument('--target-prompt-tokens', type=int, default=None, help='server-counted prompt length; defaults to full preset context')
    p.add_argument('--timeout', type=float, default=3600)
    v = sub.add_parser('validate-monitor'); v.add_argument('summary'); v.add_argument('metadata'); v.add_argument('probe_summary')
    v.add_argument('hardware_identity')
    i = sub.add_parser('capture-idle'); i.add_argument('--samples', type=int, default=5); i.add_argument('--interval', type=float, default=1.0); i.add_argument('--stable-range-bytes', type=int, default=16*1024*1024); i.add_argument('--out', required=True)
    s = sub.add_parser('settle'); s.add_argument('--idle', required=True); s.add_argument('--timeout', type=float, default=60.0); s.add_argument('--interval', type=float, default=1.0); s.add_argument('--tolerance-bytes', type=int, default=64*1024*1024); s.add_argument('--stable-range-bytes', type=int, default=16*1024*1024); s.add_argument('--stable-samples', type=int, default=5); s.add_argument('--out', required=True)
    g = sub.add_parser('identify-gpu'); g.add_argument('--out', required=True); g.add_argument('--drm-root', default='/sys/class/drm'); g.add_argument('--kfd-root', default='/sys/class/kfd/kfd/topology')
    for command in (m, p):
        command.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.mode == 'validate-monitor':
        try:
            result = validate_monitor(json.loads(Path(args.summary).read_text()),
                                      json.loads(Path(args.metadata).read_text()),
                                      json.loads(Path(args.probe_summary).read_text()),
                                      json.loads(Path(args.hardware_identity).read_text()))
            print(json.dumps(result, indent=2)); return 0
        except Exception as error:
            print(f'{type(error).__name__}: {error}', flush=True); return 1
    if args.mode == 'identify-gpu':
        try:
            result = gfx1151_hardware_identity(Path(args.drm_root), Path(args.kfd_root))
        except Exception as error:
            result = {'passed': False, 'source': 'read-only KFD topology sysfs',
                      'error': f'{type(error).__name__}: {error}',
                      'raw_inventory': raw_hardware_inventory(args.drm_root, args.kfd_root)}
        write(args.out, result)
        print(json.dumps({'passed': result.get('passed', False), 'architecture': result.get('architecture'),
                          'error': result.get('error')}), flush=True)
        return 0 if result.get('passed') else 1
    if args.mode == 'capture-idle':
        if args.samples < 2 or args.interval <= 0 or args.stable_range_bytes < 0:
            parser.error('idle baseline needs at least two samples and nonnegative stable range')
        return capture_idle(args)
    if args.mode == 'settle':
        if args.timeout <= 0 or args.interval <= 0 or args.stable_samples < 2 or min(args.tolerance_bytes,args.stable_range_bytes) < 0:
            parser.error('settle requires positive timeout/interval and at least two stable samples')
        return settle(args)
    if args.timeout <= 0 or (args.mode == 'monitor' and args.interval < 0.05):
        parser.error('positive timeout and sampling interval >=50ms required')
    return monitor(args) if args.mode == 'monitor' else probe(args)


if __name__ == '__main__':
    raise SystemExit(main())
