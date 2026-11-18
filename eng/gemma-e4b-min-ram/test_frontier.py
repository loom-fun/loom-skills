import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('sampler', HERE / 'cold_capacity_targeted.py')
sampler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sampler)
HF_SPEC = importlib.util.spec_from_file_location('hf_download', HERE / 'runner' / 'hf_download.py')
hf_download = importlib.util.module_from_spec(HF_SPEC)
HF_SPEC.loader.exec_module(hf_download)


def kfd_identity_fixture(root, target=110501, node_count=1, include_bdf=True, location_id=49920):
    drm = Path(root) / 'drm'
    device = drm / 'card0' / 'device'
    device.mkdir(parents=True)
    (device / 'vendor').write_text('0x1002\n')
    (device / 'device').write_text('0x1586\n')
    (device / 'product_name').write_text('')
    (device / 'uevent').write_text('DRIVER=amdgpu\nPCI_ID=1002:1586\nPCI_SLOT_NAME=0000:c3:00.0\n')
    kfd = Path(root) / 'kfd' / 'topology'
    for index in range(node_count):
        node = kfd / 'nodes' / f'{index + 1}'
        node.mkdir(parents=True)
        props = [f'simd_count 128', 'vendor_id 4098', 'device_id 5510',
                 f'gfx_target_version {target}']
        if include_bdf:
            props += ['domain 0', f'location_id {location_id}']
        (node / 'properties').write_text('\n'.join(props) + '\n')
    return sampler.gfx1151_hardware_identity(drm, kfd)


class FrontierTests(unittest.TestCase):
    def test_hf_fetch_invocation_normalizes_empty_unset_and_preserves_present_token(self):
        from unittest.mock import patch

        for env, expected in ((None, None), ('', None), ('hf_test_token', 'hf_test_token')):
            with self.subTest(env=env):
                received = {}
                def mocked_hub_download(**kwargs):
                    received.update(kwargs)
                    return '/mock/checkpoint.gguf'
                with patch.dict('os.environ', {}, clear=True):
                    if env is not None:
                        import os
                        os.environ['HF_TOKEN'] = env
                    path = hf_download.download_with_optional_token(
                        mocked_hub_download, repo_id='repo', filename='file.gguf')
                self.assertEqual(path, '/mock/checkpoint.gguf')
                self.assertEqual(received['token'], expected)
                self.assertEqual(received['repo_id'], 'repo')
                self.assertEqual(received['filename'], 'file.gguf')

    def test_readiness_retries_same_short_oracle_only_for_503_and_records_attempts(self):
        class Fake:
            def __init__(self, out): self.out=out; self.sequence=0; self.codes=[503,200]; self.timeout=30
            def call(self, name, path, payload):
                self.sequence += 1
                code=self.codes.pop(0); stem=f'{self.sequence:02d}_{name}'
                sampler.write(self.out / (stem+'.meta.json'), {'elapsed_seconds':0.02,'http_status':code})
                (self.out / (stem+'.response')).write_text('loading' if code==503 else 'ready')
                return code, {'loading':True} if code==503 else {'choices':[{'message':{'content':'Paris'},'finish_reason':'stop'}],
                               'usage':{'prompt_tokens':9,'completion_tokens':1}}
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); fake=Fake(out)
            code,body,attempts=sampler.short_with_readiness_retry(fake, {'prompt':'same'}, out,
                                                                  budget=1, interval=0)
            self.assertEqual(code,200); self.assertEqual(len(attempts),2)
            self.assertEqual([a['http_status'] for a in attempts],[503,200])
            self.assertTrue((out/'readiness_attempts.json').exists())

    def test_readiness_stops_on_non_503_and_times_out(self):
        class Fake:
            def __init__(self,out,codes): self.out=out; self.sequence=0; self.codes=list(codes); self.timeout=30
            def call(self,name,path,payload):
                self.sequence+=1; code=self.codes.pop(0); stem=f'{self.sequence:02d}_{name}'
                sampler.write(self.out/(stem+'.meta.json'),{'elapsed_seconds':0,'http_status':code})
                return code, {'error':'failed'}
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); fake=Fake(out,[500,200])
            code,_,attempts=sampler.short_with_readiness_retry(fake,{},out,budget=1,interval=0)
            self.assertEqual(code,500); self.assertEqual(len(attempts),1)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); fake=Fake(out,[503,503])
            with self.assertRaises(TimeoutError):
                sampler.short_with_readiness_retry(fake,{},out,budget=0,interval=0)

    def test_monitor_contract_requires_stable_process_gfx1151_and_counters(self):
        identity={'pid':42,'start_ticks':10,'identity_matches':True,'argv':['loom-serve']}
        summary={'samples':2,'stopped_by_marker':True,'deadline_reached':False,
                 'identity_ok':True,'amd_cards_observed':True,
                 'process_identities':{'42:10':identity},
                 'maxima_by_phase':{'load':{'host.MemTotal':10,'host.MemAvailable':8,'host.Cached':2,
                                            'card0.mem_info_gtt_used':4}}}
        card={'card':'card0','product_name':None,'device_id':'0x1586',
              'pci_path':'/sys/devices/pci0000:00/0000:c3:00.0',
              'uevent':'DRIVER=amdgpu\nPCI_ID=1002:1586\nPCI_SLOT_NAME=0000:c3:00.0\n'}
        metadata={'cards':[card]}
        probe={'active':{'device':'device'}}
        with tempfile.TemporaryDirectory() as d:
            hardware=kfd_identity_fixture(d)
        report=sampler.validate_monitor(summary,metadata,probe,hardware)
        self.assertFalse(report['total_ram_verified'])
        self.assertEqual(report['host_channel_coverage']['host.Cached'],'observed')
        self.assertEqual(report['selected_device'],'device')
        self.assertEqual(report['hardware_identity']['match_method'],'domain_location_id')
        incomplete=dict(summary, maxima_by_phase={'load':{'card0.mem_info_gtt_used':4}})
        unknown=sampler.validate_monitor(incomplete,metadata,probe,hardware)
        self.assertEqual(unknown['host_channel_coverage']['host.MemTotal'],'unknown')
        self.assertFalse(unknown['total_ram_verified'])
        for changed in (dict(summary,identity_ok=False), dict(summary,maxima_by_phase={'p':{}})):
            with self.assertRaises(ValueError): sampler.validate_monitor(changed,metadata,probe,hardware)
        with self.assertRaisesRegex(ValueError,'independent KFD'):
            sampler.validate_monitor(summary,metadata,probe,None)
        with self.assertRaises(ValueError): sampler.validate_monitor(summary,[],probe,hardware)
        with self.assertRaisesRegex(ValueError,'conflicts'):
            sampler.validate_monitor(summary,metadata,{'active':{'device':'amdgpu://gfx1100'}},hardware)

    def test_kfd_hardware_identity_positive_missing_wrong_arch_and_ambiguous(self):
        import subprocess, sys
        with tempfile.TemporaryDirectory() as d:
            proof=kfd_identity_fixture(d)
            self.assertEqual(proof['architecture'],'gfx1151')
            self.assertEqual(proof['gfx_target_version'],110501)
            self.assertEqual(proof['drm_card']['pci_bdf'],'0000:c3:00.0')
            self.assertIn('gfx_target_version 110501',proof['kfd_node']['properties_raw'])
            output=Path(d)/'hardware_identity.json'
            proc=subprocess.run([sys.executable,str(HERE/'cold_capacity_targeted.py'),
                'identify-gpu','--out',str(output),'--drm-root',str(Path(d)/'drm'),
                '--kfd-root',str(Path(d)/'kfd'/'topology')],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
            self.assertTrue(json.loads(output.read_text())['passed'])
        with tempfile.TemporaryDirectory() as d:
            drm=Path(d)/'drm'; device=drm/'card0'/'device'; device.mkdir(parents=True)
            (device/'vendor').write_text('0x1002\n'); (device/'device').write_text('0x1586\n')
            (device/'uevent').write_text('PCI_ID=1002:1586\nPCI_SLOT_NAME=0000:c3:00.0\n')
            with self.assertRaisesRegex(ValueError,'no GPU nodes'):
                sampler.gfx1151_hardware_identity(drm,Path(d)/'kfd'/'topology')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'expected 110501'):
                kfd_identity_fixture(d,target=110000)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'observed 0 matches'):
                kfd_identity_fixture(d,location_id=49921)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'observed 2 matches'):
                kfd_identity_fixture(d,node_count=2)
        with tempfile.TemporaryDirectory() as d:
            proof=kfd_identity_fixture(d,include_bdf=False)
            self.assertEqual(proof['match_method'],'sole_gpu_node_vendor_device')

    def test_mocked_interarm_settle_success_and_timeout(self):
        identity={'card':'card0','pci_path':'/sys/devices/pci0000:00/0000:00:01.0','device_id':'0x164e'}
        idle={'identity':identity,'baseline':{'vram_used_bytes':1000,'gtt_used_bytes':2000}}
        class Clock:
            value=0.0
            def now(self): return self.value
            def sleep(self,seconds): self.value+=seconds
        clock=Clock(); good=[{'identity':identity,'vram_used_bytes':1010+i,'gtt_used_bytes':2010+i,
                              'host_memavailable_bytes':500-i} for i in range(5)]
        result=sampler.wait_for_device_settle(idle,lambda:good.pop(0),timeout_seconds=10,
                interval_seconds=1,tolerance_bytes=64,stable_range_bytes=16,stable_samples=5,
                now=clock.now,sleep=clock.sleep)
        self.assertTrue(result['passed']); self.assertEqual(len(result['samples']),5)
        self.assertFalse(result['host_memory_used_as_settle_gate'])
        self.assertEqual(result['host_memavailable_bytes_range'],{'minimum':496,'maximum':500})

        clock=Clock(); stuck={'identity':identity,'vram_used_bytes':5000,'gtt_used_bytes':9000,
                              'host_memavailable_bytes':400}
        result=sampler.wait_for_device_settle(idle,lambda:stuck,timeout_seconds=3,
                interval_seconds=1,tolerance_bytes=64,stable_range_bytes=16,stable_samples=3,
                now=clock.now,sleep=clock.sleep)
        self.assertFalse(result['passed']); self.assertEqual(result['elapsed_seconds'],3)
        self.assertIn('did not return',result['error'])

    def test_server_counted_retrieval_calibrates_exact_count_and_preserves_secret(self):
        # Simulates the SDK chat counter with fixed template overhead and one token per ` x`.
        count = lambda s: 17 + s.count(' x')
        prompt = sampler.calibrate_retrieval(65535, count)
        self.assertEqual(count(prompt), 65535)
        self.assertIn('secret code is blue', prompt)
        self.assertIn('what is the secret code?', prompt)

    def test_calibration_fails_fast_for_non_linear_filler(self):
        with self.assertRaisesRegex(ValueError, 'not one stable token'):
            sampler.calibrate_retrieval(65535, lambda s: len(s.split()) + s.count(' x') // 2)

    def test_completion_requires_exact_prompt_and_one_output_token(self):
        body = {'choices': [{'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 65535, 'completion_tokens': 1}}
        self.assertEqual(sampler.completed(200, body, 65535)['completion_tokens'], 1)
        with self.assertRaisesRegex(ValueError, 'token counts differ'):
            sampler.completed(200, body, 65534)

    def test_context_overflow_requires_structured_sdk_error(self):
        sampler.rejected(400, {'error': {'type': 'loom_request_error', 'code': 'OUT_OF_RANGE',
                                         'message': 'request exceeds the available context size'}})
        with self.assertRaisesRegex(ValueError, 'not the structured'):
            sampler.rejected(400, {'error': {'message': 'something else'}})

    def test_runtime_pool_reporting_does_not_assume_pool_equals_context(self):
        body = {
            'data': [{'id': 'gemma', 'context_length': 65536,
                      'loom': {'registration': 'ready', 'device': 'amdgpu://gfx1151', 'pool': 'chat'}}],
            'server': {'resources': {'pools': [{'key': 'chat', 'memory': {
                'backing': 'fixed', 'storage': {'committed_bytes': 123456789},
                'workspace': {'committed_bytes': 987654321}}}]}}
        }
        result = sampler.status_contract(body, 'gemma', {'context_capacity': 65536})
        self.assertEqual(result['fixed_storage_total_bytes'], 123456789)
        self.assertIsNone(result['pool_positions'])
        self.assertIn('not directly reported', result['pool_positions_basis'])

    def test_storage_constants_and_64k_reservation(self):
        text = (HERE / 'storage.loom').read_text()
        self.assertIn('%kv_position_bytes = scalar.constant 2048', text)
        self.assertIn('%plane_count = scalar.constant 48', text)
        self.assertEqual(65536 * 2048 * 48, 6442450944)
        self.assertEqual(131072 * 2048 * 48, 12884901888)

    def test_compare_rejects_mismatched_prompts_and_outputs(self):
        baseline = {'passed': True, 'target_prompt_tokens': 65535, 'retrieval_prompt_sha256': 'a',
                    'short_text': 'Paris', 'retrieval_text': 'blue', 'preset': 'b', 'execution': {}, 'active': {}}
        candidate = dict(baseline, retrieval_prompt_sha256='b')
        with tempfile.TemporaryDirectory() as d:
            a, c = Path(d) / 'a.json', Path(d) / 'c.json'
            a.write_text(json.dumps(baseline)); c.write_text(json.dumps(candidate))
            import subprocess, sys
            p = subprocess.run([sys.executable, str(HERE / 'compare_arms.py'), str(a), str(c)], capture_output=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn(b'prompts differ', p.stderr)


if __name__ == '__main__':
    unittest.main()
