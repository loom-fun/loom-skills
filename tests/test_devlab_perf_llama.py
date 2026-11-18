"""Execute real baseline argument resolution with only engine/network stubs."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / '.github/scripts/devlab_perf.sh'


def line(prefix):
    return next(s for s in SCRIPT.read_text().splitlines() if s.startswith(prefix))


def function(name):
    text = SCRIPT.read_text().split(name + '() {', 1)[1]
    return name + '() {' + text.split('\n}', 1)[0] + '\n}\n'


def baseline(job, backend='vulkan', mods='.np1', policy='auto'):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root/'state').mkdir()
        (root/'plan.json').write_text(json.dumps({'jobs': [dict(name='j', llama_spec='', **job)]}))
        server = root/'llamacpp_server.sh'
        server.write_text('''#!/bin/bash
[ "$1" = --stop ] && exit 0
printf '%s\\n' "$@" > "$WORK/argv"
printf '%s\\n' "$LLAMA_SPEC" > "$WORK/spec"
printf '%s\\n' "$@" > "$LLAMACPP_ARGS_FILE"
''')
        server.chmod(0o755)
        fetch = root/'fetch_llamacpp.sh'
        fetch.write_text('#!/bin/bash\nprintf "%s\\n" "$WORK"\n')
        fetch.chmod(0o755)
        (root/'release.sha256').write_text('test-digest\n')
        values = dict(WORK=tmp, out=tmp, card=tmp+'/card', B=tmp, exp=str(root/'plan.json'),
                      job='j', T='python3', BK=backend, MODS=mods, vkv=policy, vtag='test', tag='test',
                      bkt='-'+backend+mods, LOOM_SERVE_STATE=str(root/'state'),
                      LLAMACPP_STATE=str(root/'state'), LLAMACPP_DIR=tmp,
                      LLAMA_PORT='8131', lurl='http://unused', gguf='/exact/pinned-target.gguf', LSPEC='')
        shell='\n'.join(f'export {k}={shlex.quote(v)}' for k,v in values.items())+'\n'
        shell += line('q()')+'\n'+line('lkv=')+'\n'+line('lvkv=')+'\n'
        shell += 'step() { :; }\nhf_sub() { printf "%s\\n" "$1"; }\n'
        shell += function('llama_up')+'\nllama_up\n'
        result = subprocess.run(['bash','-c',shell],capture_output=True,text=True)
        argv=(root/'argv').read_text().splitlines() if (root/'argv').exists() else None
        spec=(root/'spec').read_text().strip() if (root/'spec').exists() else None
        return result,argv,spec


@unittest.skipIf(os.name == 'nt', 'DevLab shell runs only on Linux')
class LlamaCapacity(unittest.TestCase):
    def check(self, job, mods, expected, backend='vulkan', policy='auto'):
        result,args,spec=baseline(job,backend,mods,policy)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(args,['/exact/pinned-target.gguf','8131',*expected])
        self.assertEqual(spec,'')

    def test_legacy_vulkan_slot_defaults_unchanged(self):
        for mods,expected,policy in [('.np1',['1','36864'],'auto'),
                                    ('.perslot.nodraft',['4','36864'],'auto'),
                                    ('',['4','36864'],'auto'),
                                    ('',['4','65536','unified'],'unified')]:
            with self.subTest(mods=mods,policy=policy):self.check({},mods,expected,policy=policy)

    def test_vulkan_override_retains_single_and_four_slot_policies(self):
        for mods,expected,policy in [('.np1',['1','10240'],'auto'),
                                    ('.perslot.nodraft',['4','10240'],'auto'),
                                    ('',['4','10240','unified'],'unified')]:
            with self.subTest(mods=mods,policy=policy):self.check({'llama_kv_vulkan':10240},mods,expected,policy=policy)

    def test_backend_capacities_are_independent(self):
        config={'llama_kv_rocm':8192,'llama_kv_vulkan':10240}
        self.check(config,'.np1',['1','8192'],backend='rocm')
        self.check(config,'.nodraft',['4','8192','unified'],backend='rocm')
        self.check(config,'.np1',['1','10240'])
        self.check({'llama_kv_vulkan':10240},'.np1',['1','65536'],backend='rocm')

    def test_invalid_override_fails_before_engine_start(self):
        for value in (0,-1,True,'8192',8192.5,[],{}):
            with self.subTest(value=value):
                result,args,_=baseline({'llama_kv_vulkan':value})
                self.assertNotEqual(result.returncode,0)
                self.assertIsNone(args)
                self.assertIn('llama_kv_vulkan must be a positive integer',result.stderr)


if __name__ == '__main__':
    unittest.main()
