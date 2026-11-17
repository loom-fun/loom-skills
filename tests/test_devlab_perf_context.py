"""Only the explicit capacity-proof caller may bypass short warmup generation."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'.github/scripts/devlab_perf.sh'


def function(name):
 lines=SCRIPT.read_text().splitlines();start=next(i for i,s in enumerate(lines) if s.startswith(name+'()'))
 end=next(i for i in range(start+1,len(lines)) if lines[i]=='}')
 return '\n'.join(lines[start:end+1])


@unittest.skipIf(os.name=='nt','The actual DevLab Bash harness runs only on Linux')
class ColdCapacityCaller(unittest.TestCase):
 def invoke(self,cold,warm_rc):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);B=p/'B';SV=p/'SV';B.mkdir();SV.mkdir();state=p/'state';state.mkdir();pkg=p/'D/v/pkg';pkg.mkdir(parents=True)
   (pkg/'model.json').write_text('{"presets":{"chat":{"execution":{}}}}')
   for name in ['llamacpp_server.sh','vllm_server.sh','wait_vram.sh']:(B/name).write_text('#!/bin/bash\nexit 0\n');(B/name).chmod(0o755)
   for name in ['serve.sh','stop.sh']:(SV/name).write_text('#!/bin/bash\nexit 0\n');(SV/name).chmod(0o755)
   (B/'wait_model.py').write_text('import os,sys\nwith open(os.environ["REC"],"a") as f:f.write("warm\\n")\nsys.exit(int(os.environ["WARM_RC"]))\n')
   values={'here':ROOT,'T':sys.executable,'D':p/'D','out':p,'B':B,'SV':SV,'LOOM_SERVE_STATE':state,'REC':p/'calls','WARM_RC':warm_rc,'LLAMA_PORT':1,'PORT':1,'url':'http://fixture','RDEV':'','RSUF':'','RUNNER':'strix','mem':17000000000}
   code='set -uo pipefail\n'+'\n'.join('export '+k+'='+shlex.quote(str(v)) for k,v in values.items())+'\n'
   code+='''vllm_down() { :; }
sdk_root() { SDKR=/fixture/sdk; }
step() { :; }
curl() { printf 'models\\n' >> "$REC"; return 0; }
qv() { case "$2" in
  *active_wait*) echo 1 ;;
  *serve_args*) echo '' ;;
  *memory_bytes*) echo 17000000000 ;;
  *elastic*) echo False ;;
  *) echo '' ;;
esac; }
'''
   code+=function('serve_alone')+'\n'+('CONTEXT_PROOF_COLD=1 ' if cold else '')+'serve_alone v\n'
   code+='result=$?\nprintf "RESULT %s\\n" "$result"\n'
   r=subprocess.run(['bash','-c',code],capture_output=True,text=True)
   calls=(p/'calls').read_text().splitlines() if (p/'calls').exists() else []
   return r.stdout,r.stderr,calls

 def test_ordinary_generation_wait_and_failure_are_preserved(self):
  for rc,result in [(0,0),(7,1)]:
   with self.subTest(warm_rc=rc):
    out,err,calls=self.invoke(False,rc)
    self.assertIn(f'RESULT {result}',out,err);self.assertEqual(calls,['models','warm'])

 def test_explicit_context_caller_skips_only_short_generation(self):
  out,err,calls=self.invoke(True,7)
  self.assertIn('RESULT 0',out,err);self.assertEqual(calls,['models'])


if __name__=='__main__':unittest.main()
