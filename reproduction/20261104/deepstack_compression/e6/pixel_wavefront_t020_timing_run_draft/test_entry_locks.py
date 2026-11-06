import subprocess,sys,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_cli_is_locked(self):
  for name in ('gpu_worker.py','gpu_launch.py'):
   p=subprocess.run([sys.executable,str(D/name)],capture_output=True,text=True,timeout=10)
   self.assertNotEqual(p.returncode,0);self.assertIn('GPU locked',p.stderr)
 def test_import_rejects_same_name_contamination(self):
  code="import types,sys;f=types.ModuleType('_pixel_t020_gpu_bridge');f.__file__='/tmp/foreign.py';sys.modules[f.__name__]=f;import gpu_worker\ntry:gpu_worker.load(f.__name__,gpu_worker.R/'bridge.py')\nexcept ValueError:pass\nelse:raise AssertionError('foreign source accepted')"
  p=subprocess.run([sys.executable,'-c',code],cwd=D,capture_output=True,text=True,timeout=10)
  self.assertEqual(p.returncode,0,p.stderr)
if __name__=='__main__':unittest.main()

class IntegrationTests(unittest.TestCase):
 def test_launcher_real_24_job_control_flow_with_fake_docker(self):
  import gpu_launch,json,types,tempfile,importlib.util,hashlib
  from unittest.mock import patch
  design=json.loads((gpu_launch.C/'design001.json').read_text());jobs=design['jobs'];refs={(j['task'],str(j['qid']),j['arm']):{} for j in jobs};calls=[]
  spec=importlib.util.spec_from_file_location('synthetic_timing_controller',gpu_launch.R/'controller.py');controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
  class Host:
   @staticmethod
   def bounded_one(job,jobdir,command,name,ctrl,plan,worker,wsha,bounded,pids,identity,supervisor,timeout):
    jobdir.mkdir();calls.append(job['order']);(jobdir/'raw.json').write_text('{}');(jobdir/'record.json').write_text(json.dumps({'job':job,'record':{},'metrics':{'ok':True}}))
    return {'container_identity':{'container_id':format(job['order']+1,'064x'),'host_init':{'pid':100+job['order'],'start_ticks':1,'boot_id':'boot'}}}
  def loader(name,path):
   if name.endswith('controller'):return controller
   if name.endswith('host'):return Host
   if name.endswith('job_check'):return types.SimpleNamespace(check=lambda *args:{'ok':True})
   if name.endswith('reference'):return types.SimpleNamespace(load=lambda:(refs,{}))
   if name.endswith('bounded'):return types.SimpleNamespace(run_bounded=None)
   if name.endswith('worker'):return types.SimpleNamespace()
   raise AssertionError(name)
  runtime=types.SimpleNamespace(container_command=lambda author,args,gpu,name:['docker','--rm']+args,identity=lambda pid:{'pid':pid},gpu_pids=lambda:[])
  with tempfile.TemporaryDirectory() as t,patch.object(gpu_launch,'load',side_effect=loader),patch.object(gpu_launch.time,'time',return_value=1):
   out=Path(t)/'run';r=gpu_launch.execute_after_admission('plan','sha',hashlib.sha256((gpu_launch.C/'design001.json').read_bytes()).hexdigest(),jobs,out,100000,runtime,lambda:None,inner_seconds=1,cleanup_seconds=1)
   self.assertEqual(calls,list(range(24)));self.assertEqual(r['measured'],18);self.assertEqual(json.loads((out/'controller.jsonl').read_text().splitlines()[-1])['event'],'all_completed')
