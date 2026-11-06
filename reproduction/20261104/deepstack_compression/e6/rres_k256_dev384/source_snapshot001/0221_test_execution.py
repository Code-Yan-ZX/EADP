import json,tempfile,unittest,sys
from pathlib import Path
from types import SimpleNamespace
import gpu_dispatch,worker,gpu_gate,runtime_admit
from unittest.mock import patch
class Tests(unittest.TestCase):
 def test_gpu_hard_locked_without_reviews(self):
  with self.assertRaises(RuntimeError):gpu_gate.admit_gpu()
  with self.assertRaises(RuntimeError):runtime_admit.admit_gpu()
 def test_actual_worker_refuses_before_gpu(self):
  with tempfile.TemporaryDirectory() as t:
   f=Path(t);out=f/'run001/observations';out.mkdir(parents=True)
   with patch.object(worker,'F',f),patch.object(sys,'argv',['worker.py','--out',str(out)]),self.assertRaises(RuntimeError):worker.main()
 def test_real_gpu_callback_forwarded_and_no_preexisting_gpu(self):
  calls=[]
  class Runtime:
   approval={'test':True}
   def gpu_pids(self):calls.append('idle');return []
   def identity(self,pid):return {'pid':pid}
   def container_command(self,author,args,gpu,name):return ['docker','run','--rm',*args]
  def bounded(cmd,name,timeout,gpu_pids,identity,run,require_gpu_idle):
   self.assertTrue(callable(gpu_pids));self.assertEqual(gpu_pids(),[]);self.assertTrue(require_gpu_idle);self.assertEqual(timeout,21000);return 0
  with tempfile.TemporaryDirectory() as t:
   run=Path(t)/'run';self.assertEqual(gpu_dispatch.dispatch(Runtime(),SimpleNamespace(run_bounded=bounded),{},run,Path('/worker')),0);self.assertEqual(calls,['idle','idle']);self.assertTrue((run/'launch.json').exists())
 def test_busy_gpu_prevents_container_creation(self):
  runtime=SimpleNamespace(gpu_pids=lambda:[123])
  with tempfile.TemporaryDirectory() as t:
   run=Path(t)/'run'
   with self.assertRaises(ValueError):gpu_dispatch.dispatch(runtime,None,{},run,Path('/worker'))
   self.assertFalse(run.exists())
 def test_e1_control_complete_and_unique(self):
  with tempfile.TemporaryDirectory() as t:
   path=Path(t)/'journal.jsonl';rows=[{'task':'t','qid':str(i),'arm':'P3-D1','forced':False,'generation_scalars':{'x':1},'eos_ids':[2]} for i in range(384)]
   path.write_text(''.join(json.dumps(x)+'\n' for x in rows));completion={'files':{'journal':{'path':str(path),'sha256':worker.sha(path)}}}
   self.assertEqual(len(worker.controls_from_e1(completion)),384)
   path.write_text(path.read_text()+json.dumps(rows[0])+'\n');completion['files']['journal']['sha256']=worker.sha(path)
   with self.assertRaises(ValueError):worker.controls_from_e1(completion)
if __name__=='__main__':unittest.main()
