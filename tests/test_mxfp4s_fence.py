"""The real KB/factory fence choice changes only compiler scheduling."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'skills/loom-port-llm/starter'))
import gen_mxfp4s as G


class CompilerFences(unittest.TestCase):
    def test_kb_selected_factory_keeps_all_numeric_and_memory_operations(self):
        sys.path.insert(0,str(ROOT/'skills/loom-kb'))
        import resolve
        for choice in ['fenced','fenced_swp']:
            with tempfile.TemporaryDirectory() as tmp:
                node=Path(tmp)/'model'/'fence-test';node.mkdir(parents=True)
                (node/'scope.json').write_text(json.dumps({
                    'scope':'model/fence-test','match':{'slug':'fence-test'},'owner':'test',
                    'choices':[{'when':{'devices':['rdna4']},
                                'set':{'prefill.gemm.mxfp4s_f8_kernel':choice},
                                'evidence':'test: real resolved factory'}]}))
                cfg={'name':'fence-test','arch':'qwen3','hidden':5120,'layers':64,
                     'heads':24,'kv_heads':4,'head_dim':256,'ffn':17408,'tensors':{}}
                env=resolve.resolve_full(cfg,('rdna3','rdna4'),tmp,draft=None,slug='fence-test')['env']
            self.assertEqual(env,{'GEMM_MXFP4S_F8_KERNEL':choice})
            with patch.dict(os.environ,env,clear=True):
                for mode in ['store','swiglu']:
                    text=G.gemm_mxfp4s_f8_m256(mode,**G.f8_kernel_kwargs(mode))
                    # Removing only the scheduling annotations gives EXACTLY the
                    # original direct or SWP kernel, including all memory effects,
                    # constants, barrier placement and each ordered accumulator.
                    fences='    buffer.fence scope(subgroup) ordering(acq_rel)\n'
                    self.assertEqual(text.count(fences),2)
                    self.assertEqual(text.count('vector.mma '),128)
                    baseline=G.gemm_mxfp4s_f8_m256(mode,swp=choice=='fenced_swp' and mode=='store')
                    self.assertEqual(text.replace(fences,''),baseline)
                    # An actual fence must precede each group's loads, rather
                    # than living in an empty Low helper that is ineffective.
                    for k in [0,4]:self.assertIn(fences+f'    %fk{k} = index.constant {16*k} : index\n',text)

    def test_incompatible_fragment_paths_fail_closed(self):
        for kw in [{'lv':True},{'lvs':True},{'ws':4},{'a_pf':True},{'a_direct':False}]:
            with self.subTest(kw=kw),self.assertRaisesRegex(AssertionError,'original-K'):
                G.gemm_mxfp4s_f8_m256('store',fence_k=4,**kw)


if __name__=='__main__':unittest.main()
