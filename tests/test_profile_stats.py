import contextlib, importlib.util, io, json, os, sys, tempfile, unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("profile_stats", os.path.join(HERE, "..", "skills", "loom-benchmark", "scripts", "profile_stats.py"))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)

LOG = """IREE HAL device statistics:
  aggregate_rows=3
  dispatch_function_total=2.5 ms
  session=1 dispatch g_gemm_mxfp4s_store_f8_m256_rdna4  count=4 total=1.000 ms avg=250.000 us
  session=1 dispatch g_gdn_recurrence  count=2 total=500.000 us avg=250.000 us
  session=1 dispatch g_row_scale_f16_live  count=10 total=1000 ns avg=100 ns
"""


class ProfileStats(unittest.TestCase):
    def test_parse_units(self):
        rows = P.parse(LOG)
        self.assertEqual(rows["g_gemm_mxfp4s_store_f8_m256_rdna4"], [4, 1.0])
        self.assertAlmostEqual(rows["g_gdn_recurrence"][1], 0.5)
        self.assertAlmostEqual(rows["g_row_scale_f16_live"][1], 0.001)

    def test_full_perf_warmup_is_counted_in_profile_totals(self):
        with tempfile.TemporaryDirectory() as work:
            log, req, result = (os.path.join(work, name) for name in ('serve.log', 'req.json', 'out.json'))
            with open(log, 'w') as stream:
                stream.write(LOG)
            data = {'scenario': 'dec', 'warmup_mode': 'perf-essay',
                    'warmup': {'completion_tokens': 300, 'total_s': 7, 'ttft_s': .1},
                    'requests': [{'completion_tokens': 300, 'total_s': 8, 'ttft_s': .2}]}
            with open(req, 'w') as stream:
                json.dump(data, stream)
            with patch.object(sys, 'argv', ['profile_stats.py', log, req, '--json', result]), \
                    contextlib.redirect_stdout(io.StringIO()):
                P.main()
            with open(result) as stream:
                out = json.load(stream)
            self.assertEqual(out['tokens'], 600)
            self.assertEqual(out['wall_ms'], 15000)
            self.assertTrue(out['includes_warmup'])

    def test_families_shipped(self):
        fams = json.load(open(os.path.join(HERE, "..", "skills", "loom-benchmark", "scripts", "profile_families.json")))
        self.assertEqual(P.family("g_gemm_mxfp4s_store_f8_m256_rdna4", fams), "GEMM store (fp8)")
        self.assertEqual(P.family("g_gemm_mxfp4s_swiglu_f8_m256_rdna4", fams), "GEMM swiglu (fp8)")
        self.assertEqual(P.family("g16_attention_split_listed", fams), "attention (flash + rope + qgate + split/combine)")
        self.assertEqual(P.family("g_mv_mxfp4s_k5120_t4", fams), "alpha/beta matvecs (in_proj_a/b)")
        self.assertEqual(P.family("g_mv_mxfp4s_k17408_t4_q8_r8", fams), "verify matvecs, MXFP4 linears (t4)")
        self.assertEqual(P.family("g_head_mv_q8_p40", fams), "MTP proposals: block + head copy (q8_0/q4_0)")
        self.assertEqual(P.family("g_head_mv_t4", fams), "verify head (bf16 lm_head)")


if __name__ == "__main__":
    unittest.main()
