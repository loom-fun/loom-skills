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
wheel_name = 'hrx_loom-0.3.0-py3-none-win_amd64.whl'
archive_name = 'loom-serve-0.3.0-win_amd64.zip'
assets = {
    wheel_name: '06afbc88370fde741037ebd38dba91c2184cb4d2fbd5cee337720cd494a06752',
    archive_name: '2f0bb6b4ab6053941041659c1ae8a8bb8386b6d49d4c7731b61adacffd3e521a',
}


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['curl.exe', '--fail', '--location', '--silent', '--show-error', '--retry', '3',
                    url, '--output', str(path)], check=True, timeout=120)
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def launch(exe, card, label, cxx):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    args = [exe, '--device=amdgpu', f'--model={card}', f'--listen=127.0.0.1:{port}']
    with (out / f'{label}.log').open('wb') as log:
        server = subprocess.Popen(args, cwd=root, stdout=log, stderr=subprocess.STDOUT)
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
                if cxx:
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


assert sys.platform == 'win32'
for name, digest in assets.items():
    path = download('https://github.com/ROCm/hrx-system/releases/download/v0.3.0/' + name, root / name)
    assert sha(path) == digest, name
venv = root / 'venv'
subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
subprocess.run([venv / 'Scripts/python.exe', '-m', 'pip', 'install', '--disable-pip-version-check',
                f'{root / wheel_name}[hf]'], check=True, timeout=240)
exe = root / 'standalone/loom-serve.exe'
exe.parent.mkdir()
with zipfile.ZipFile(root / archive_name) as bundle:
    assert bundle.namelist() == ['loom-serve.exe'], bundle.namelist()
    exe.write_bytes(bundle.read(exe.name))

fixture = root / 'fixture'
fixture_url = f'https://raw.githubusercontent.com/ROCm/hrx-system/{source_commit}/loom/docs/examples/serving/model-publication/model/tiny-text/'
for relative in ('model.json', 'model.json.lock', 'tokenizer.json', 'program/advance.loom', 'program/control.loom',
                 'program/prepare.loom', 'program/storage.loom', 'program/weights.loom'):
    download(fixture_url + relative, fixture / relative)
control = root / 'control'
subprocess.run([venv / 'Scripts/loom-fetch.exe', fixture / 'model.json', '--materialize=copy',
                f'--output={control}', '--no-input'], check=True, timeout=120)
cxx = root / 'cxx'
shutil.copytree(control, cxx)
card = cxx / 'model.json'
model = json.loads(card.read_text(encoding='utf-8'))
model['programs']['source']['inputs'].append({'format': 'cxx', 'path': 'program/probe.cxx'})
card.write_text(json.dumps(model), encoding='utf-8')
(cxx / 'program/probe.cxx').write_text('int loom_import_probe = ;\n', encoding='utf-8')

results = {}
for name, executable in (('wheel', venv / 'Scripts/loom-serve.exe'), ('standalone', exe)):
    results[name] = {'executable_sha256': sha(executable),
                     'version': subprocess.check_output([executable, '--version'], timeout=30).decode().strip(),
                     'control': launch(executable, control / 'model.json', f'{name}-control', False),
                     'cxx': launch(executable, card, f'{name}-cxx', True)}
record('result.json', results)
print(json.dumps(results, indent=2))
