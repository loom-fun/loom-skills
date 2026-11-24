from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile


root, evidence, docs = (Path(path).resolve() for path in sys.argv[1:4])
shipping = len(sys.argv) == 5 and sys.argv[4] == 'shipping'
# Keep the hub cache in the job scope and reuse exact checkpoint downloads across packages.
os.environ['HF_HOME'] = str(root / 'hf-home')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
release_hashes = {
    'loom-serve-0.2.0-win_amd64.zip': '7864cbd21bb054296245b5e78623767a5a20f460a1e402529c14e38c5c321976',
    'loom-serve-0.3.0-win_amd64.zip': '2f0bb6b4ab6053941041659c1ae8a8bb8386b6d49d4c7731b61adacffd3e521a',
    'hrx_loom-0.2.0-py3-none-win_amd64.whl': '14c133b032eb3b9c9b59b2092b7f564699fd2ec69fc8c1d04874e74be30664fa',
    'hrx_loom-0.3.0-py3-none-win_amd64.whl': '06afbc88370fde741037ebd38dba91c2184cb4d2fbd5cee337720cd494a06752',
}
packages = {
    'qwen3-0.6b': ('loom-fun/unsloth-Qwen3-0.6B-loom', '564d578f4e487c6fbabd84915439f873651b5953', 'chat'),
    'flash-next': ('loom-fun/unsloth-Qwen3.8-Flash-Next-loom', 'd847c21bc93c0c6bc45752a9edaf91d80ac74490', 'vision-mtp'),
}
staged_flash = ('loom-fun-staging/unsloth-Qwen3.8-Flash-Next-loom',
                '473e5815cedcf3117fcac905c27ed3434f9fe836', 'vision-mtp')
flash_lock = '53a173df16767be788e32dcf4b219d07a1726b3384b3acb28660c8d0ca0b6c07'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download_sdk(version, name):
    target = root / name
    subprocess.run(['curl.exe', '--fail', '--location', '--silent', '--show-error', '--retry', '3',
                    f'https://github.com/ROCm/hrx-system/releases/download/v{version}/{name}',
                    '--output', str(target)], check=True)
    if sha_file(target) != release_hashes[name]:
        raise ValueError(f'SDK asset digest did not match: {name}')
    return target


def setup():
    if os.name != 'nt':
        raise RuntimeError('This workflow requires native Windows')
    write_json(evidence / 'pins.json', {'releases': release_hashes, 'packages': packages,
                                       'staged_flash': staged_flash if shipping else None})
    executables = {}
    for version in ('0.2.0', '0.3.0'):
        archive = download_sdk(version, f'loom-serve-{version}-win_amd64.zip')
        install = root / version
        install.mkdir()
        with zipfile.ZipFile(archive) as files, (install / 'loom-serve.exe').open('wb') as dest:
            assert files.namelist() == ['loom-serve.exe'], files.namelist()
            dest.write(files.read('loom-serve.exe'))
        executable = install / 'loom-serve.exe'
        executables[version] = executable
        version_print = subprocess.check_output([executable, '--version'], timeout=30, stderr=subprocess.STDOUT)
        (evidence / f'sdk-{version}.txt').write_bytes(version_print)
    fetchers = {}
    for version in (('0.2.0', '0.3.0') if shipping else ('0.2.0',)):
        wheel = download_sdk(version, f'hrx_loom-{version}-py3-none-win_amd64.whl')
        venv = root / f'fetch-venv-{version}'
        subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
        subprocess.run([venv / 'Scripts/python.exe', '-m', 'pip', 'install', '--disable-pip-version-check',
                        f'{wheel}[hf]'], check=True, timeout=240)
        fetchers[version] = venv / 'Scripts/loom-fetch.exe'
    return executables, fetchers


def fetch(fetch_exe, label, repo, revision, preset, expected_abi=1):
    target = root / label
    url = f'https://huggingface.co/{repo}/resolve/{revision}/model.json'
    print(f'{label}: fetching {repo}@{revision} preset {preset}', flush=True)
    if shipping:
        with urlopen(f'{url}.lock', timeout=60) as lock:
            if hashlib.sha256(lock.read()).hexdigest() != flash_lock:
                raise ValueError(f'The locked Flash checkpoint differs for {repo}')
    with (evidence / f'{label}-fetch.log').open('wb') as log:
        args = [fetch_exe, url, f'--preset={preset}', f'--output={target}', '--no-input']
        if shipping:
            args.append('--materialize=share')
        subprocess.run(args, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=2400)
    card = target / 'model.json'
    config = json.loads(card.read_text(encoding='utf-8'))
    service = config['presets'][preset]['service']
    if config['services'][service]['version'] != expected_abi:
        raise ValueError(f'Expected controller ABI {expected_abi} for {label}')
    if expected_abi == 2:
        sys.path.insert(0, str(docs.parent / 'scripts/card'))
        from card_lib import programs_sha256
        if programs_sha256(target) != 'c0cf0958949af15430bb8791eb167e8108a5a374f3f5aa3828eb55f0d88ac920':
            raise ValueError('The staged Flash program digest differs')
    write_json(evidence / f'{label}-package.json', {
        'model_sha256': sha_file(card), 'service': service, 'abi': expected_abi,
        'source_lock_sha256': flash_lock if shipping else None,
        'program_file_sha256': {
            file.relative_to(target).as_posix(): sha_file(file)
            for file in sorted((target / 'program').rglob('*')) if file.is_file()
        },
    })
    return card


def get_json(url, timeout=4):
    with urlopen(url, timeout=timeout) as response:
        return json.load(response)


def wait_ready(base_url, server):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f'loom-serve exited 0x{server.returncode & 0xffffffff:08x}')
        try:
            status = get_json(base_url + '/v1/status')
        except (OSError, ValueError):
            time.sleep(1)
            continue
        models = status.get('data', [])
        registrations = [model.get('loom', {}).get('registration') for model in models]
        if 'failed' in registrations:
            raise RuntimeError(f'Model registration failed: {status}')
        if registrations and all(item == 'ready' for item in registrations):
            return status
        time.sleep(1)
    raise TimeoutError('Timed out waiting for active model registration')


def request(base_url, model, prompt, limit, thinking):
    data = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
            'max_tokens': limit, 'stream': True, 'stream_options': {'include_usage': True},
            'cache_prompt': False}
    if thinking is not None:
        data['chat_template_kwargs'] = {'enable_thinking': thinking}
    message = Request(base_url + '/v1/chat/completions', data=json.dumps(data).encode(),
                      headers={'Content-Type': 'application/json'})
    start = time.perf_counter()
    first, last, usage, finish = None, None, None, None
    text = []
    try:
        with urlopen(message, timeout=260) as response:
            for line in response:
                if not line.startswith(b'data: {'):
                    continue
                payload = json.loads(line[6:])
                usage = payload.get('usage') or usage
                for choice in payload.get('choices', []):
                    delta = choice.get('delta', {})
                    piece = (delta.get('reasoning_content') or '') + (delta.get('content') or '')
                    if piece:
                        last = time.perf_counter()
                        first = last if first is None else first
                        text.append(piece)
                    finish = choice.get('finish_reason') or finish
    except HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode(errors="replace")}') from error
    total = time.perf_counter() - start
    if not usage or first is None or last is None:
        raise ValueError(f'Missing token usage or visible output, finish={finish}, chars={sum(map(len, text))}')
    prompt_tokens = usage['prompt_tokens']
    completion = usage['completion_tokens']
    cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
    if cached or prompt_tokens <= 0 or completion < min(64, limit):
        raise ValueError(f'Cached or shortened measurement: {usage}, finish={finish}')
    return {'prompt_tokens': prompt_tokens, 'completion_tokens': completion,
            'cached_tokens': cached, 'finish': finish, 'ttft_s': first - start, 'wall_s': total,
            'prefill_effective_tps': prompt_tokens / (first - start),
            'decode_tps': (completion - 1) / (last - first) if last > first else None,
            'reply_sha256': hashlib.sha256(''.join(text).encode()).hexdigest()}


def workload(tag, document):
    return (f'[{tag}] Write an extended technical essay based on this documentation. '
            'Discuss details and give many examples. Continue for several paragraphs.\n\n' + document)


def append_sample(row):
    with (evidence / 'samples.jsonl').open('a', encoding='utf-8') as log:
        log.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def measure(label, card, preset, executable, release, session, document_set):
    case = f'{label}-{release}-{session}'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    args = [str(executable), '--device=amdgpu://?completion_spin_ns=4294967295',
            f'--listen=127.0.0.1:{port}']
    if shipping:
        args += [f'--model={card.parent}', '--pool_backing=fixed' if release == '0.2.0' else '--workers=8']
    else:
        args += ['--pool_backing=elastic', '--workers=8', f'--model={card}', f'--preset={preset}']
    write_json(evidence / f'{case}-args.json', args)
    with (evidence / f'{case}-serve.log').open('wb') as log:
        server = subprocess.Popen(args, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        try:
            status = wait_ready(url, server)
            write_json(evidence / f'{case}-status.json', status)
            model = status['data'][0]['id']
            thinking = False if label == 'flash-next' else None
            request(url, model, f'[{case} warmup] Write a detailed essay about wind patterns.', 16, thinking)
            identifiers = {'model': label, 'release': release, 'session': session}
            for sample in range(3):
                for name, document in document_set.items():
                    tag = f'{label}.paired.{session}.{sample}.{name}'
                    result = request(url, model, workload(tag, document), 128, thinking)
                    append_sample({**identifiers, 'workload': name, 'sample': sample, **result})
            with ThreadPoolExecutor(max_workers=4) as pool:
                start = time.perf_counter()
                futures = [pool.submit(request, url, model,
                                       workload(f'{label}.paired.{session}.c4.{index}', document_set['2k']),
                                       128, thinking) for index in range(4)]
                replies = [item.result() for item in futures]
                seconds = time.perf_counter() - start
            append_sample({**identifiers, 'workload': 'c4', 'sample': 0, 'requests': replies,
                           'wall_s': seconds,
                           'aggregate_tps': sum(item['completion_tokens'] for item in replies) / seconds})
        finally:
            if server.poll() is None:
                server.terminate()
                server.wait(timeout=15)


def summarize(label):
    rows = [json.loads(line) for line in (evidence / 'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    old, new = ({(r['workload'], r['session'], r['sample']): r for r in rows
                 if r['release'] == release and r['model'] == label} for release in ('0.2.0', '0.3.0'))
    if not old or old.keys() != new.keys():
        raise ValueError(f'Incomplete paired samples for {label}')
    results = {}
    for name in ('2k', '32k', 'c4'):
        pairs = [(old[key], new[key]) for key in old if key[0] == name]
        for before, after in pairs:
            pre = before.get('requests', [before])
            post = after.get('requests', [after])
            for a, b in zip(pre, post):
                if any(a[key] != b[key] for key in ('prompt_tokens', 'completion_tokens', 'reply_sha256')):
                    raise ValueError(f'Greedy result or token counts differ in matched case for {label}, {name}')
        metrics = ('aggregate_tps',) if name == 'c4' else ('prefill_effective_tps', 'decode_tps', 'ttft_s')
        results[name] = {}
        for metric in metrics:
            a = [old_row[metric] for old_row, _ in pairs]
            b = [new_row[metric] for _, new_row in pairs]
            results[name][metric] = {
                'samples_0.2': a, 'samples_0.3': b,
                'median_0.2': statistics.median(a), 'median_0.3': statistics.median(b),
                'paired_change_percent': statistics.median(100 * (right / left - 1) for left, right in zip(a, b)),
            }
    write_json(evidence / f'{label}-summary.json', results)
    print(json.dumps({label: results}), flush=True)


executables, fetchers = setup()
documents = {name: (docs / f'doc_{count}.txt').read_text(encoding='utf-8')
             for name, count in (('2k', 2000), ('32k', 32000))}
sequence = (('0.2.0', 0), ('0.3.0', 0), ('0.3.0', 1), ('0.2.0', 1))
if shipping:
    # Present both immutable packages on the same machine, using their respective README launch arguments.
    cards = {'0.2.0': fetch(fetchers['0.2.0'], 'flash-production', *packages['flash-next']),
             '0.3.0': fetch(fetchers['0.3.0'], 'flash-staged', *staged_flash, expected_abi=2)}
    for release, session in sequence:
        print(f'Flash published/staged {release} period {session}', flush=True)
        measure('flash-next', cards[release], 'vision-mtp', executables[release], release, session, documents)
    summarize('flash-next')
    sys.exit(0)
failed = []
for label, (repo, revision, preset) in packages.items():
    try:
        card = fetch(fetchers['0.2.0'], label, repo, revision, preset)
        for release, session in sequence:
            print(f'{label} {release} period {session}', flush=True)
            measure(label, card, preset, executables[release], release, session, documents)
        summarize(label)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'{label} failed: {error}', flush=True)
        failed.append(label)
if failed:
    sys.exit(f'Unverified model cases: {", ".join(failed)}')
