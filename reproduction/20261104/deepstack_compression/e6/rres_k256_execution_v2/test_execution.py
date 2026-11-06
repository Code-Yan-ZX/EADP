"""CPU checks of actual generation-control wrapper and GPU lock."""
import unittest,tempfile,json
from pathlib import Path
from types import SimpleNamespace
import gpu_gate,worker
import gpu_dispatch
class Fake:
 def __init__(self):self.called=0;self.generation_config=SimpleNamespace(eos_token_id=[151645,151643])
 def generate(self,*a,**kw):self.called+=1;return 'ok'
class Tests(unittest.TestCase):
 def setUp(self):self.control={'generation_scalars':{'do_sample':False,'max_new_tokens':2048,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0},'eos_ids':[151645,151643]}
 def test_exact_recipe_calls_real_target(self):
  m=Fake();g=worker.checked_generation(m,self.control);self.assertEqual(g(**self.control['generation_scalars']),'ok');self.assertEqual(m.called,1)
 def test_wrong_recipe_or_eos_never_calls_target(self):
  for field in self.control['generation_scalars']:
   m=Fake();g=worker.checked_generation(m,self.control);kw=dict(self.control['generation_scalars']);kw[field]='wrong'
   with self.assertRaises(ValueError):g(**kw)
   self.assertEqual(m.called,0)
  m=Fake();m.generation_config.eos_token_id=[9];g=worker.checked_generation(m,self.control)
  with self.assertRaises(ValueError):g(**self.control['generation_scalars'])
  self.assertEqual(m.called,0)
 def test_no_gpu_plan_is_locked(self):
  with self.assertRaises(FileNotFoundError):gpu_gate.admit_gpu()
 def test_actual_dispatch_uses_real_callback_and_blocks_busy_gpu(self):
  class Runtime:
   approval={'tested':True}
   def __init__(self,pids):self.pids=pids;self.calls=0
   def gpu_pids(self):self.calls+=1;return self.pids
   def identity(self,pid):return {'pid':pid}
   def container_command(self,_a,args,gpu,name):assert gpu;return ['docker','run','--rm',name,*args]
  class Timing:
   def __init__(self):self.called=False
   def run_bounded(self,command,name,timeout,callback,identity,out,require_gpu_idle):
    self.called=True;assert timeout==3600 and callable(callback) and callback()==[] and require_gpu_idle;return 0
  with tempfile.TemporaryDirectory() as tmp:
   runtime=Runtime([]);timing=Timing();out=Path(tmp)/'ok';self.assertEqual(gpu_dispatch.dispatch(runtime,timing,{},out,'worker.py'),0);self.assertTrue(timing.called);self.assertEqual(runtime.calls,2);self.assertEqual(json.loads((out/'launch.json').read_text())['gpu_pids_before'],[])
   runtime=Runtime([999]);timing=Timing();out=Path(tmp)/'busy'
   with self.assertRaises(ValueError):gpu_dispatch.dispatch(runtime,timing,{},out,'worker.py')
   self.assertFalse(timing.called);self.assertFalse(out.exists())
if __name__=='__main__':unittest.main()
