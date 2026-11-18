"""CPU tests for the synthetic kernel gate; never instantiate a GPU or weights."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
try:
    import numpy as np
except ModuleNotFoundError as exc:
    if exc.name != 'numpy':
        raise
    np = None

ROOT=Path(__file__).resolve().parents[1]
ADAPTER=ROOT/'.github/devlab/compact-kv'

def module(name):
    spec=importlib.util.spec_from_file_location(name,ADAPTER/(name+'.py'))
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

g=module('generate_checks') if np is not None else None
gate=module('run_checks');runner=module('materialize_runner')


class AdapterTests(unittest.TestCase):
    def report(self):
        d={k:0 for k in ('scenario_count','trial_count','failed_trial_count','skipped_case_count','skipped_scenario_count','planning_issue_count','xpass_count','xfail_mismatch_count','failed_sample_count','xfail_count')}
        d.update(format='loom.test.v0',case_count=1,sample_count=1,
                 samples=[dict(case='good',passed=True,expectations=dict(expectation_count=3,passed_count=3,failure_count=0,failures=[]))])
        d.update({k:[] for k in ('trials','skipped_cases','skipped_scenarios','planning_issues','xfails')})
        return d

    def test_strict_positive_report(self):
        gate.validate_positive(self.report(),['good'],{'good':3})
        for field,value in [('passed',None),('passed',False),('case','wrong')]:
            d=self.report();d['samples'][0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):gate.validate_positive(d,['good'])
        for field,value in [('sample_count',0),('xpass_count',1),('planning_issues',['bad']),('samples',[]),('xfails',[{}])]:
            d=self.report();d[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):gate.validate_positive(d,['good'])
        with self.assertRaises(ValueError):gate.validate_positive(self.report(),['good'],{'good':4})

    @unittest.skipIf(np is None, 'numpy not installed (numeric qualification uses the SDK tools environment)')
    def test_float64_oracle_matches_scalar_full_history(self):
        c=g.Case('tiny',2,2);rng=np.random.default_rng(7)
        q=rng.normal(0,.1,(2,8,256));h=rng.normal(0,.1,(4,2,2,256)).astype(np.float16)
        got=g.reference(c,q,h,np.arange(4)).reshape(2,8,256)
        for token in range(2):
            for head in range(8):
                n=3+token;kv=head//4
                scores=np.array([sum(float(q[token,head,d])*float(h[p,0,kv,d]) for d in range(256)) for p in range(n)])
                weights=np.exp(scores-max(scores));weights/=sum(weights)
                expected=np.array([sum(float(weights[p])*float(h[p,1,kv,d]) for p in range(n)) for d in range(256)])
                np.testing.assert_allclose(got[token,head],expected,atol=1e-8,rtol=1e-6)

    @unittest.skipIf(np is None, 'numpy not installed (numeric qualification uses the SDK tools environment)')
    def test_safe_negative_controls_have_comparator_sensitivity(self):
        for c in g.NEGATIVES:
            with self.subTest(case=c.name):
                a,_,limit=g.fixture(c)
                positions=np.arange(max(0,c.start-526),c.start+c.tokens)
                pages=a['new_origin'][10:].astype(np.int64)
                writer=int(a['new_origin'][:10].view(np.int64)[1])
                capacity=int(c.bad[4:]) if c.bad.startswith('ring') else g.RING
                reads=g.addresses(c,positions,pages,True,capacity,writer)
                if c.bad=='v_stride':reads[:,1]=reads[:,0]+g.CONTEXT*2048
                self.assertGreaterEqual(int(reads.min()),0)
                self.assertLessEqual(int(reads.max())+1024,limit)
                packed={int(addr):values[kv] for row,values in zip(a['new_addresses'],a['new_values']) for kv,addr in enumerate(row)}
                self.assertEqual(len(packed),a['new_addresses'].size)
                history=np.zeros((len(positions),2,2,256),np.float16)
                for n,row in enumerate(reads):
                    for kv,addr in enumerate(row):
                        history[n,kv]=packed.get(int(addr),0)
                wrong=g.reference(c,a['q'].reshape(c.tokens,8,256),history,positions)
                ratio=np.abs(wrong-a['expected'])/(.003+.003*np.abs(a['expected']))
                self.assertGreater(float(ratio.max()),2,'negative must fail declared tolerance with margin')
                self.assertTrue(all(np.isfinite(v).all() for v in a.values()))
                control=g.replace(c,name='control_'+c.name,bad='')
                correct,_,_=g.fixture(control)
                np.testing.assert_array_equal(correct['expected'],a['expected'])
                text=g.check_case(c,a,2*g.CONTEXT*2048,limit,c.bad+'__')
                self.assertNotIn('kernel.launch @legacy__',text)
                self.assertIn('g16d256_attention_combine[',text)
                self.assertNotIn('check.expect.equal',text)

    def test_real_runner_contracts_preserved(self):
        # Compatibility CI has only its shallow PR checkout. The DevLab job
        # explicitly fetches this frozen cross-branch source before materializing.
        if subprocess.run(['git','-C',str(ROOT),'cat-file','-e',runner.PIN+'^{commit}'],capture_output=True).returncode:
            self.skipTest('frozen ordinary-runner commit not in this checkout (shallow clone)')
        source={name:subprocess.check_output(['git','-C',str(ROOT),'show',runner.PIN+':'+runner.PREFIX+name]).decode() for name in runner.FILES}
        changed=runner.patch_sources(source,'a'*40,'b'*64,'c'*64)
        def functions(text):
            return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        old,new=functions(source['cold_capacity_targeted.py']),functions(changed['cold_capacity_targeted.py'])
        self.assertEqual({k:v for k,v in old.items() if k!='probe'},{k:v for k,v in new.items() if k!='probe'})
        start=source['run_frontier.sh'].index('run_arm() {');end=source['run_frontier.sh'].index('# Sample a stable pre-arm')
        original=source['run_frontier.sh'][start:end]
        start2=changed['run_frontier.sh'].index('run_arm() {');end2=changed['run_frontier.sh'].index('# Sample a stable pre-arm')
        self.assertEqual(original.replace('65535','65536'),changed['run_frontier.sh'][start2:end2])
        self.assertEqual(source['runner/sdk_pin.py'],changed['runner/sdk_pin.py'])
        self.assertIn('context=65536; rows=1',changed['prepare_packages.sh'])
        self.assertIn('--pool 65536',changed['prepare_packages.sh'])
        self.assertIn('"$SDK_ROOT/.tools/bin/python" "$ADAPTER/generate_checks.py"',changed['run_frontier.sh'])
        with self.assertRaises(ValueError):runner.replace_once('ambiguous ambiguous','ambiguous','x')

    @unittest.skipIf(np is None, 'numpy not installed (numeric qualification uses the SDK tools environment)')
    def test_flash_launches_respect_native_multiple_and_pair_split_counts(self):
        for c in g.CASES:
            if 'flash' in g.variants(c):self.assertEqual(c.tokens%16,0)
        c=g.Case('small',511,2);a,ob,nb=g.fixture(c);s=g.check_case(c,a,ob,nb)
        self.assertIn('@compact__g16d256_attention_combine[%t]',s)
        self.assertIn('@compact__gd256_attention_combine[%t]',s)
        self.assertEqual(s.count('check.expect.'),6)


    @unittest.skipIf(np is None, 'numpy not installed (numeric qualification uses the SDK tools environment)')
    def test_listed_scalar_binding_preserves_count_buffer_and_list(self):
        for count in (1, 32, 8192):
            self.assertEqual(g.listed_scalar_count(count, count), count)
        for count, tokens in ((31,32), (33,32), (0,0), (8193,8193), (-1,1), (True,1), (1.0,1)):
            with self.subTest(count=count,tokens=tokens),self.assertRaises(ValueError):
                g.listed_scalar_count(count,tokens)
        c=g.Case('edge512',512,1);a,ob,nb=g.fixture(c);s=g.check_case(c,a,ob,nb)
        for arm in ('old','new'):
            self.assertIn(f'%{arm}_listed_count = check.generate.fill value(1) : tensor<1xi32>',s)
            self.assertIn(f'%{arm}_listed_list = check.generate.iota offset(0) step(-1) : tensor<1xi32>',s)
            self.assertEqual(s.count(f'[%{arm}_listed_count, %{arm}_listed_scalar]'),2)
        self.assertEqual(s.count('check.expect.'),9)

    @unittest.skipIf(np is None, 'numpy not installed (numeric qualification uses the SDK tools environment)')
    def test_listed_projection_preserves_device_and_rejects_source_drift(self):
        # Use the maintained listed-kernel constructor, without importing its
        # model-building module or reading a checkpoint.
        source=(ROOT/'skills/loom-port-llm/starter/gen_llm.py').read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='listed')
        scope={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<listed-constructor>','exec'),scope)
        blocks=[]
        for dim in (256,512):
            for part in ('split','combine'):
                name=f'g16d{dim}_attention_{part}'
                plain=f'''kernel.def export("{name}") @{name}(%tokens: index) {{
}} launch(%tokens_arg: index, %out: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  kernel.return
}}
'''
                blocks.append(scope['listed'](plain,name,'  %c1 = index.constant 1 : index\n  kernel.launch.config workgroups(%count, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index'))
        text,records=g.extract('\n'.join(blocks),[f'g16d{d}_attention_{p}_listed' for d in (256,512) for p in ('split','combine')],'legacy__')
        emitted,proof=g.project_listed_launch(text,'legacy__',copy.deepcopy(records))
        before={n:b.partition('} launch(')[2] for _,n,_,b in g.split_blocks(text)}
        after={n:b.partition('} launch(')[2] for _,n,_,b in g.split_blocks(emitted)}
        self.assertEqual(before,after);self.assertEqual(len(proof),4)
        with self.assertRaises(ValueError):
            g.project_listed_launch(text.replace('view<1xi32> -> i32','view<2xi32> -> i32',1),'legacy__',copy.deepcopy(records))
        bad=copy.deepcopy(records);bad['g16d256_attention_split_listed']['emitted_sha256']='0'*64
        with self.assertRaises(ValueError):g.project_listed_launch(text,'legacy__',bad)


if __name__=='__main__':unittest.main()
