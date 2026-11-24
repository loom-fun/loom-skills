import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tarfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile


phase, root_arg, evidence_arg = sys.argv[1:]
root, evidence = (Path(value).resolve() for value in (root_arg, evidence_arg))
windows = sys.platform == 'win32'
suffix = '.exe' if windows else ''
platform = 'win_amd64' if windows else 'manylinux_2_39_x86_64'
wheel = f'hrx_loom-0.3.0-py3-none-{platform}.whl'
standalone = f'loom-serve-0.3.0-{platform}.zip'
release = 'https://github.com/ROCm/hrx-system/releases/download/v0.3.0/'
pins = {
    wheel: ('06afbc88370fde741037ebd38dba91c2184cb4d2fbd5cee337720cd494a06752' if windows
            else '6252bb0df262351175bdb2c4aaca22ffa63d80d791659aa4d2fa5c92a0af426f'),
    standalone: ('2f0bb6b4ab6053941041659c1ae8a8bb8386b6d49d4c7731b61adacffd3e521a' if windows
                 else 'b41d42e9a1a957f2a425fa25c48360cd9d3845ae76a9412d9f552ec1baa5809e'),
}
installed_exe = ('c3071350a0404291a43843524bb9a9b891c2a715a58b84c457d440bcac1f523a' if windows
                 else '1db8c4d0cdf4787af6a685dd945802eb559e15e3948a931b446fb6a6570fa71d')
previous_exe = ('e2d926f1ad37efd6a340ad52ee674516fc2186a268637b53bc9412378fecbb46' if windows
                else 'b95424e54d77dac1a77a81f930aa53ebf78cd2a636c6e331565b59ab733e997a')
cases = (
    ('legacy', 'Qwen3-0.6B-Loom', 'loom-fun/unsloth-Qwen3-0.6B-loom',
     '564d578f4e487c6fbabd84915439f873651b5953', 'chat', 1),
    ('flash', 'user.FlashNextStage', 'loom-fun-staging/unsloth-Qwen3.8-Flash-Next-loom',
     '7d27c03e6450de529a4bf7d76118a8170b2ac1aa', 'vision-mtp', 2),
)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, data):
    (evidence / name).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def download(url, path, timeout=100):
    subprocess.run(['curl.exe' if windows else 'curl', '--fail', '--silent', '--show-error', '--location', '--retry', '3',
                    url, '--output', str(path)], check=True, timeout=timeout)


def snapshot(repo, revision):
    location = root / 'weights-hf/hub' / ('models--' + repo.replace('/', '--'))
    return location / 'snapshots' / revision, location / 'refs/main'


def prepare():
    os.environ['HF_HOME'] = str(root / 'weights-hf')
    os.environ['HF_HUB_CACHE'] = str(root / 'weights-hf/hub')
    for name, expected in pins.items():
        download(release + name, root / name)
        assert sha(root / name) == expected, name
    venv = root / 'fetch-env'
    bin_dir = venv / ('Scripts' if windows else 'bin')
    subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
    subprocess.run([bin_dir / ('python' + suffix), '-m', 'pip', 'install', '--disable-pip-version-check',
                    f'{root / wheel}[hf]'], check=True, timeout=250)
    records = {}
    for label, alias, repo, revision, preset, abi in cases:
        target = root / ('package-' + label)
        with (evidence / f'{label}-fetch.log').open('wb') as log:
            print(f'Preparing {repo}@{revision}', flush=True)
            subprocess.run([bin_dir / ('loom-fetch' + suffix), f'huggingface:{repo}:{preset}@{revision}',
                            f'--output={target}', '--materialize=share', '--no-input'],
                           check=True, stdout=log, stderr=subprocess.STDOUT, timeout=2700)
        base = f'https://huggingface.co/{repo}/resolve/{revision}'
        # Keep upstream metadata paired with its lock. The ready directory also
        # contains every parameter and source file verified by loom-fetch.
        for name in ('model.json', 'model.json.lock', 'info.json'):
            path = target / name
            temporary = target / (name + '.raw')
            download(base + '/' + name, temporary)
            temporary.replace(path)
        card = json.loads((target / 'model.json').read_text(encoding='utf-8'))
        service = card['presets'][preset]['service']
        assert card['services'][service]['version'] == abi
        if abi == 2:
            tools = Path(os.environ['GITHUB_WORKSPACE']) / 'skills/loom-benchmark/scripts/card'
            sys.path.insert(0, str(tools))
            from card_lib import programs_sha256
            assert programs_sha256(target) == 'c0cf0958949af15430bb8791eb167e8108a5a374f3f5aa3828eb55f0d88ac920'
        snap, ref = snapshot(repo, revision)
        snap.mkdir(parents=True, exist_ok=True)
        for source in target.rglob('*'):
            if not source.is_file():
                continue
            dest = snap / source.relative_to(target)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                continue
            os.link(source.resolve(), dest)
        for name in ('model.json', 'model.json.lock', 'info.json'):
            assert sha(snap / name) == sha(target / name), name
        ref.parent.mkdir(exist_ok=True)
        ref.write_text(revision + '\n', encoding='utf-8')
        records[label] = {'alias': alias, 'repo': repo, 'revision': revision, 'controller_abi': abi,
                          'model_sha256': sha(snap / 'model.json')}
    write('packages.json', records)


def health(server, base):
    for _ in range(120):
        if server.poll() is not None:
            raise RuntimeError(f'lemond exited 0x{server.returncode & 0xffffffff:08x}')
        try:
            with urlopen(base + '/api/v1/health', timeout=3) as result:
                return json.load(result)
        except (OSError, ValueError):
            time.sleep(0.5)
    raise TimeoutError('Lemonade did not become healthy')


def chat(exe, cache, config, label, alias, registry_repo=None):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    with (evidence / f'{label}-lemonade.log').open('wb') as log:
        args = [exe, cache, config, '--port', str(port), '--host', '127.0.0.1']
        server = subprocess.Popen(args, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=not windows)
        try:
            write(f'{label}-health.json', health(server, base))
            if registry_repo:
                definition = {'model_name': alias, 'checkpoint': registry_repo,
                              'recipe': 'loom', 'source': 'huggingface'}
                registration = Request(base + '/api/v1/models/register',
                                       data=json.dumps(definition).encode(),
                                       headers={'Content-Type': 'application/json'})
                try:
                    with urlopen(registration, timeout=60) as reply:
                        registered = json.load(reply)
                except HTTPError as error:
                    (evidence / f'{label}-register-error.json').write_bytes(error.read())
                    raise
                write(f'{label}-registration.json', registered)
                assert registered['canonical_model_name'] == alias, registered
            body = {'model': alias, 'messages': [{'role': 'user', 'content':
                     'What is the capital of France? Answer with only the city name. /no_think'}],
                    'temperature': 0, 'max_tokens': 48, 'chat_template_kwargs': {'enable_thinking': False}}
            request = Request(base + '/api/v1/chat/completions', data=json.dumps(body).encode(),
                              headers={'Content-Type': 'application/json'})
            start = time.monotonic()
            try:
                with urlopen(request, timeout=600) as reply:
                    result = json.load(reply)
            except HTTPError as error:
                (evidence / f'{label}-api-error.json').write_bytes(error.read())
                raise
            write(f'{label}-chat.json', result)
            content = result['choices'][0]['message']['content'].strip().rstrip('.')
            assert content == 'Paris', result
            print(f'{label}: {content} through Lemonade in {time.monotonic() - start:.2f}s including model load', flush=True)
        finally:
            if server.poll() is None:
                if windows:
                    subprocess.run(['taskkill.exe', '/PID', str(server.pid), '/T', '/F'],
                                   capture_output=True, timeout=15, check=True)
                else:
                    os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=15)


def run():
    os.environ['HF_HUB_CACHE'] = str(root / 'weights-hf/hub')
    os.environ['HF_HOME'] = str(root / 'weights-hf')
    draft = root / 'draft'
    assert (draft / 'draft-head.txt').read_text().strip() == os.environ['LEMONADE_REVISION']
    archive, = draft.glob('lemonade-embeddable-*-windows-x64.zip' if windows
                          else 'lemonade-embeddable-*-ubuntu-x64.tar.gz')
    assert sha(archive) == (draft / 'draft-build-sha256.txt').read_text().strip()
    write('draft-build.json', {'revision': os.environ['LEMONADE_REVISION'], 'sha256': sha(archive)})
    app = root / 'application'
    if windows:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(app)
    else:
        with tarfile.open(archive) as bundle:
            bundle.extractall(app, filter='data')
    executable, = app.rglob('lemond' + suffix)
    cache, config = root / 'lc', root / 'cfg'
    install = cache / 'bin/loom/hrx'
    install.mkdir(parents=True)
    config.mkdir()
    config_file = config / 'config.json'
    config_file.write_text(json.dumps({'websocket_port': 0}), encoding='utf-8')
    backend = install / ('loom-serve' + suffix)
    with zipfile.ZipFile(root / standalone) as bundle:
        backend.write_bytes(bundle.read(backend.name))
    backend.chmod(0o755)
    (install / 'version.txt').write_text('v0.3.0', encoding='utf-8')
    assert sha(backend) == previous_exe
    before = sha(backend)
    label, alias, *_ = cases[0]
    chat(executable, cache, config, label, alias)
    assert sha(backend) == installed_exe
    runtime_files = ('iree.dll', 'loomc.dll') if windows else ('libiree.so', 'libloomc.so')
    libraries = {name: sha(install / name) for name in runtime_files}
    write('same-version-upgrade.json', {'version': (install / 'version.txt').read_text().strip(),
                                      'before': before, 'after': installed_exe, 'libraries': libraries})
    label, alias, repo, revision, *_ = cases[1]
    snap, ref = snapshot(repo, revision)
    before_card = sha(snap / 'model.json')
    chat(executable, cache, config, label, alias, registry_repo=repo)
    assert ref.read_text(encoding='utf-8').strip() == revision
    assert sha(snap / 'model.json') == before_card
    assert not (root / 'weights-hf/hub/models--loom-fun--unsloth-Qwen3.8-Flash-Next-loom').exists()
    write('flash-loaded-revision.json', {'repo': repo, 'revision': revision, 'model_sha256': before_card})
    assert sha(backend) == installed_exe


assert windows or sys.platform == 'linux'
{'prepare': prepare, 'run': run}[phase]()
