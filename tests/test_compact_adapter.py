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


if __name__=='__main__':unittest.main()
