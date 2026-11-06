import json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import cpu_gate,worker,launch
class Tests(unittest.TestCase):
 def tearDown(self):
  for n in ('_timing192_cpu_gate','_timing192_ref_cpu_preflight','_timing192_launcher_gpu_gate'):sys.modules.pop(n,None)
 def test_cpu_missing_independent_proof_rejected(self):
  with tempfile.TemporaryDirectory() as temp,patch.object(cpu_gate,'I',Path(temp)),patch.object(cpu_gate,'plan',return_value={'source_sha256':{}}),patch.object(cpu_gate,'terminal'),patch.object(cpu_gate,'sha',return_value='x'),patch.object(cpu_gate,'read',return_value={'status':'passed','test_returncode':0,'plan_sha256':'x','source_sha256':{},'GT_opened':False,'target_predictions_opened':False,'GPU_authorized':False}):
   with self.assertRaisesRegex(ValueError,'independent source review missing'):cpu_gate.admit()
 def test_gpu_launcher_stops_before_creating_output(self):
  with tempfile.TemporaryDirectory() as temp,patch.object(sys,'argv',['launch.py','--run-id','001']):
   with self.assertRaises((ValueError,FileNotFoundError)):launch.main()
   self.assertEqual(list(Path(temp).iterdir()),[])
 def test_actual_cpu_preflight_main_with_fake_imports(self):
  cid='a'*64
  with tempfile.TemporaryDirectory() as temp:
   out=Path(temp);workerhash=__import__('hashlib').sha256(Path(worker.__file__).read_bytes()).hexdigest();approval={'status':'passed','plan_sha256':'p'}
   (out/'container_identity.json').write_text(json.dumps({'container_id':cid,'host_cgroup':'/'+cid,'host_init':{'start_ticks':1}}));(out/'launch.json').write_text(json.dumps({'approval_bundle':approval,'run_out':str(out),'worker_sha256':workerhash}))
   cg=types.SimpleNamespace(admit=lambda:approval,plan=lambda:{})
   def prepare(plan,path,require_no_gpu):
    self.assertTrue(require_no_gpu);v={'status':'passed','inputs_checked':222,'references_checked':888,'cuda_initialized':False,'model_loaded':False,'GT_opened':False};Path(path).write_text(json.dumps(v));return v
   pre=types.SimpleNamespace(prepare=prepare);actual=Path.read_text
   def fake_read(path,*args,**kwargs):
    if str(path)=='/proc/self/cgroup':return '/'+cid
    return actual(path,*args,**kwargs)
   def load(name,path):return cg if name=='_timing192_cpu_gate' else pre
   with patch.object(worker,'load',side_effect=load),patch.object(Path,'read_text',fake_read),patch.object(sys,'argv',['worker.py','--out',str(out),'--preflight-only']):worker.main()
   result=json.loads((out/'worker_preflight.json').read_text());self.assertEqual((result['inputs'],result['jobs']),(222,7920));self.assertFalse(result['cuda_initialized']);self.assertFalse(result['fixed32_new_references_opened'])
if __name__=='__main__':unittest.main()
