"""Narrow tiles cover every output once and preserve each K accumulation."""
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'skills/loom-port-llm/starter'))
import gen_mxfp4s as G


class NarrowTiles(unittest.TestCase):
    def test_real_kb_rule_selects_factory(self):
        sys.path.insert(0,str(ROOT/'skills/loom-kb'))
        import resolve
        with tempfile.TemporaryDirectory() as tmp:
            node=Path(tmp)/'model'/'narrow-test';node.mkdir(parents=True)
            (node/'scope.json').write_text(json.dumps({
                'scope':'model/narrow-test','match':{'slug':'narrow-test'},'owner':'test',
                'choices':[{'when':{'devices':['rdna4']},
                            'set':{'prefill.gemm.mxfp4s_f8_kernel':'narrow'},
                            'evidence':'test: actual factory resolution'}]}))
            cfg={'name':'narrow-test','arch':'qwen3','hidden':5120,'layers':64,
                 'heads':24,'kv_heads':4,'head_dim':256,'ffn':17408,'tensors':{}}
            r=resolve.resolve_full(cfg,('rdna3','rdna4'),tmp,draft=None,slug='narrow-test')
        self.assertEqual(r['env'],{'GEMM_MXFP4S_F8_KERNEL':'narrow'})
        with patch.dict(os.environ,r['env'],clear=True):
            for mode in ['store','swiglu']:
                text=G.gemm_mxfp4s_f8_m256(mode,**G.f8_kernel_kwargs(mode))
                self.assertEqual(text.count('vector.mma '),64)

    def test_native_result_mapping_is_bijective_and_k_order_unchanged(self):
        with patch.dict(os.environ,{},clear=True):
            for mode in ['store','swiglu']:
                base=G.gemm_mxfp4s_f8_m256(mode)
                text=G.gemm_mxfp4s_f8_m256(mode,**G.F8_KERNELS['narrow'][mode=='swiglu'])
                def constant(name):
                    return int(re.search(r'%'+name+r' = index.constant (\d+) : index',text).group(1))
                nt,tn,wc,wr,cols=map(constant,['cnt','tn','cwc','cwr','wave_cols'])
                seen=set()
                for wave in range(nt//32):
                    for rowfrag in range(wr//16):
                        for colfrag in range(cols//16):
                            for lane in range(32):
                                for element in range(8):
                                    row=(wave//wc)*wr+16*rowfrag+(lane//16)*8+element
                                    col=(wave%wc)*cols+16*colfrag+lane%16
                                    self.assertNotIn((row,col),seen)
                                    seen.add((row,col))
                self.assertEqual(seen,{(r,c) for r in range(256) for c in range(tn)})
                # This output's original8 ascendingK16 MMA updates are retained;
                # other outputs use the same codegen over a bijective mapping.
                stream=lambda s: [l.strip() for l in s.splitlines() if re.match(r'\s*%m0r0c0_\d+ = vector.mma',l)]
                self.assertEqual(len(stream(text)),8)
                self.assertEqual(stream(base),stream(text))


if __name__=='__main__':unittest.main()
