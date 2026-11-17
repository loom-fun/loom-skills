"""CPU mocked HTTP/sysfs checks of the capacity-proof harness, not GPU evidence."""
from pathlib import Path
import contextlib,io,importlib.util,json,sys,tempfile,types,unittest
from unittest.mock import patch
P=Path(__file__).resolve().parents[1]/".github/scripts"
sp=importlib.util.spec_from_file_location('contextprobe',P/'context_completion_probe.py');C=importlib.util.module_from_spec(sp);sp.loader.exec_module(C)

class Response:
 def __init__(self,status,body):self.status_code=status;self.body=body;self.text=json.dumps(body)
 def json(self):return self.body
 def raise_for_status(self):
  if self.status_code>=400:raise RuntimeError(self.text)

class Package:
 chat={'thinking':'toggle'};model_id='fixture-model'
 def __init__(self,*_):self.tok=types.SimpleNamespace(decode=lambda ids:'FILLER '*len(ids))
 def render_messages(self,*_):return 'fixture chat header'
 def ids(self,*_):return list(range(32))

class ContextHarness(unittest.TestCase):
 def run_fixture(self,finish='stop',cache=3072,status=200,cold=True,active=4):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);dev=root/'fixture-gpu';dev.mkdir()
   (dev/'vendor').write_text('0x1002');(dev/'mem_info_vram_total').write_text(str(32*2**30));used=dev/'mem_info_vram_used';used.write_text(str(29*2**30))
   out=root/'out';posts=[]
   def post(url,json,**_):
    posts.append(json)
    if cold and len(posts)==1:return Response(503,{'error':'fixture asynchronous cold load'})
    answer='Plants use sunlight. Chlorophyll captures light. Carbon dioxide supplies carbon. Plants release oxygen. Glucose stores energy. Water supplies hydrogen.\nCAPACITY-CEDAR-731'
    return Response(200,{'choices':[{'finish_reason':finish,'message':{'content':answer}}],'usage':{'prompt_tokens':3584,'completion_tokens':64,'prompt_tokens_details':{'cached_tokens':cache}}})
   def get(*_,**__):return Response(status,{'data':[{'id':'fixture-model','loom':{'status':{'active_requests':active}}}]})
   args=['probe','fakepkg','http://fixture',str(out),'--scripts','/tmp/fake','--context','4096','--concurrent']+(['--cold-first'] if cold else [])
   mock=types.SimpleNamespace(Package=Package,doc=lambda _:'fixture document')
   with patch.object(sys,'argv',args),patch.dict(sys.modules,{'bench_common':mock}),patch.object(Path,'glob',return_value=iter([used])),patch.object(C,'requests',types.SimpleNamespace(post=post,get=get)),patch.object(C.time,'sleep',return_value=None):
    with contextlib.redirect_stdout(io.StringIO()):rc=C.main()
   d=json.loads((out/'result.json').read_text());attempts=json.loads((out/'near_first-attempts.json').read_text());memory=(out/'memory.jsonl').read_text()
   return rc,d,posts,attempts,memory

 def test_sdk_cli_requires_actual_pinned_wheel_hash(self):
  import subprocess
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'sdk';pkg=Path(tmp)/'pkg';root.mkdir();pkg.mkdir()
   pin={'hrx_loom':'0.2.0','source':'pypi','wheel_sha256':'a'*64}
   (pkg/'llm_config.json').write_text(json.dumps({'sdk':pin}))
   for have,rc in [(pin,0),({**pin,'hrx_loom':'0.1.0'},1),({k:v for k,v in pin.items() if k!='wheel_sha256'},1)]:
    (root/'loom-install.json').write_text(json.dumps(have))
    r=subprocess.run([sys.executable,str(P/'context_completion_probe.py'),'--check-sdk',str(root),str(pkg)],capture_output=True,text=True)
    self.assertEqual(r.returncode,rc,r.stderr);self.assertEqual(json.loads(r.stdout)['passed'],rc==0)

 def test_cold_same_workload_then_cache_and_observed_overlap(self):
  rc,d,posts,attempts,memory=self.run_fixture()
  self.assertEqual(rc,0);self.assertTrue(d['passed']);self.assertTrue(d['cold_first'])
  self.assertEqual([x['status'] for x in attempts],[503,200])
  self.assertEqual(posts[0],posts[1]);self.assertEqual(posts[1],posts[2])
  self.assertGreater(len(posts[0]['messages'][0]['content']),15000)
  self.assertEqual(len(posts),7)  # retry + nearfirst/repeat + four realrequests
  self.assertTrue(d['near_repeat_exact']);self.assertTrue(d['c4_overlap_proved']);self.assertIn('fixture-gpu',memory)

 def test_length_or_uncached_or_absent_status_cannot_pass(self):
  for kwargs in [{'finish':'length'},{'cache':0},{'status':503},{'active':3}]:
   with self.subTest(kwargs=kwargs):
    rc,d,*_=self.run_fixture(**kwargs);self.assertEqual(rc,1);self.assertFalse(d['passed'])

if __name__=='__main__':unittest.main()
