import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen
import zipfile


root, out = (Path(value).resolve() for value in sys.argv[1:])
source_commit = 'e2c41a720363911c04606ce0f4417e2d55630352'
windows = sys.platform == 'win32'
platform = 'win_amd64' if windows else 'manylinux_2_39_x86_64'
wheel_name = f'hrx_loom-0.3.0-py3-none-{platform}.whl'
archive_name = f'loom-serve-0.3.0-{platform}.zip'
assets = {
    wheel_name: ('06afbc88370fde741037ebd38dba91c2184cb4d2fbd5cee337720cd494a06752' if windows
                 else '6252bb0df262351175bdb2c4aaca22ffa63d80d791659aa4d2fa5c92a0af426f'),
    archive_name: ('2f0bb6b4ab6053941041659c1ae8a8bb8386b6d49d4c7731b61adacffd3e521a' if windows
                    else 'b41d42e9a1a957f2a425fa25c48360cd9d3845ae76a9412d9f552ec1baa5809e'),
}


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['curl.exe' if windows else 'curl', '--fail', '--location', '--silent', '--show-error', '--retry', '3',
                    url, '--output', str(path)], check=True, timeout=120)
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def launch(exe, card, label, cxx, working_dir=None):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    args = [exe, '--device=amdgpu', f'--model={card}', f'--listen=127.0.0.1:{port}']
    with (out / f'{label}.log').open('wb') as log:
        server = subprocess.Popen(args, cwd=working_dir or root, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f'{label}: exited 0x{server.returncode & 0xffffffff:08x}')
                try:
                    with urlopen(base + '/v1/status', timeout=3) as response:
                        status = json.load(response)
                except (OSError, ValueError):
                    time.sleep(0.2)
                    continue
                model = status.get('data', [{}])[0] if status.get('data') else {}
                registration = model.get('loom', {}).get('registration')
                if registration not in ('ready', 'failed'):
                    time.sleep(0.2)
                    continue
                record(f'{label}-status.json', status)
                if cxx and registration == 'failed':
                    reason = model.get('loom', {}).get('error', {})
                    return {'registration': registration, 'error': reason, 'diagnostic': reason.get('message', '')}
                assert registration == 'ready', status
                body = {'model': model['id'], 'messages': [{'role': 'user', 'content': 'abxa'}],
                        'temperature': 0, 'max_tokens': 2}
                req = Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                              headers={'Content-Type': 'application/json'})
                with urlopen(req, timeout=60) as response:
                    answer = json.load(response)
                record(f'{label}-answer.json', answer)
                assert answer['choices'][0]['message']['content'] == 'bx', answer
                return {'registration': registration, 'control_answer': 'bx'}
            raise TimeoutError(label)
        finally:
            if server.poll() is None:
                server.terminate()
                server.wait(timeout=10)


assert windows or sys.platform == 'linux'
for name, digest in assets.items():
    path = download('https://github.com/ROCm/hrx-system/releases/download/v0.3.0/' + name, root / name)
    assert sha(path) == digest, name
venv = root / 'venv'
bin_dir = venv / ('Scripts' if windows else 'bin')
suffix = '.exe' if windows else ''
subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
subprocess.run([bin_dir / ('python' + suffix), '-m', 'pip', 'install', '--disable-pip-version-check',
                f'{root / wheel_name}[hf]'], check=True, timeout=240)
exe = root / 'standalone' / ('loom-serve' + suffix)
exe.parent.mkdir()
with zipfile.ZipFile(root / archive_name) as bundle:
    assert bundle.namelist() == [exe.name], bundle.namelist()
    exe.write_bytes(bundle.read(exe.name))
exe.chmod(0o755)

fixture = root / 'fixture'
fixture_url = f'https://raw.githubusercontent.com/ROCm/hrx-system/{source_commit}/loom/docs/examples/serving/model-publication/model/tiny-text/'
for relative in ('model.json', 'model.json.lock', 'tokenizer.json', 'program/advance.loom', 'program/control.loom',
                 'program/prepare.loom', 'program/storage.loom', 'program/weights.loom'):
    download(fixture_url + relative, fixture / relative)
control = root / 'control'
subprocess.run([bin_dir / ('loom-fetch' + suffix), fixture / 'model.json', '--materialize=copy',
                f'--output={control}', '--no-input'], check=True, timeout=120)
cxx = root / 'cxx'
shutil.copytree(control, cxx)
card = cxx / 'model.json'
model = json.loads(card.read_text(encoding='utf-8'))
model['programs']['source']['inputs'].append({'format': 'cxx', 'path': 'program/probe.cxx'})
card.write_text(json.dumps(model), encoding='utf-8')
(cxx / 'program/probe.cxx').write_text('int loom_import_probe = ;\n', encoding='utf-8')
valid = root / 'valid-cxx'
shutil.copytree(cxx, valid)
download(f'https://raw.githubusercontent.com/ROCm/hrx-system/{source_commit}/loom/docs/examples/getting-started/first-kernel/saxpy.cxx',
         valid / 'program/probe.cxx')

saxpy = (valid / 'program/probe.cxx').read_text(encoding='utf-8')
source_rel = Path('program/shared/program/text/qwen3_8/publish.cxx')
header_rel = Path('program/shared/program/text/publication.h')
lexical_suffix = 'program/shared/program/text/qwen3_8/../../../program/text/publication.h'
exe = bin_dir / ('loom-serve' + suffix)
results = {}
for length in (230, 251, 259, 260, 261, 271):
    padding_size = length - len(str(root / lexical_suffix)) - 1
    assert 1 <= padding_size <= 240
    case = root / ('x' * padding_size)
    shutil.copytree(control, case)
    card = case / 'model.json'
    model = json.loads(card.read_text(encoding='utf-8'))
    model['programs']['source']['inputs'].append({
        'format': 'cxx', 'path': source_rel.as_posix(),
        'options': ["cxx:'I=../../..'"], 'data': [header_rel.as_posix()],
    })
    card.write_text(json.dumps(model), encoding='utf-8')
    (case / source_rel).parent.mkdir(parents=True)
    (case / source_rel).write_text('#include "program/text/publication.h"\n' + saxpy, encoding='utf-8')
    (case / header_rel).write_text('#define SDK030_PACKAGE_HEADER 1\n', encoding='utf-8')
    candidate = case / lexical_suffix
    assert len(str(candidate)) == length
    results[str(length)] = {
        'canonical_length': len(str(case / header_rel)),
        'header_readable_by_python': candidate.is_file(),
        'backend': launch(exe, card, f'lexical-{length}', True),
    }
results['relative_271'] = launch(exe, Path('model.json'), 'relative-271', True, working_dir=case)
record('path-depth-matrix.json', results)
print(json.dumps(results, indent=2))
assert results['230']['backend']['registration'] == 'ready', results
