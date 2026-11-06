import unittest
from observe import run
class Cuda:
 def __init__(self,initialized=False):self.init=initialized;self.ops=[]
 def is_initialized(self):return self.init
 def synchronize(self):self.ops.append('sync');self.init=True
 def reset_peak_memory_stats(self):self.ops.append('reset')
 def max_memory_allocated(self):return 12
 def max_memory_reserved(self):return 20
class Torch:
 def __init__(self,initialized=False):self.cuda=Cuda(initialized)
class Tests(unittest.TestCase):
 def test_fresh_context_two_allocator_windows(self):
  t=Torch();events=[];counter=iter(range(1,20));model,result=run(t,lambda:'m',lambda m:('answer',m),events.append,lambda:next(counter))
  self.assertEqual((model,result),('m',('answer','m')));self.assertEqual([e['event'] for e in events],['model_load_start','model_load_end','request_start','request_end']);self.assertEqual(t.cuda.ops,['sync','reset','sync']);self.assertEqual(events[1]['allocated_peak_bytes'],12);self.assertEqual(events[-1]['reserved_peak_bytes'],20)
 def test_worker_is_locked_without_loading_model(self):
  import os,subprocess,sys
  from pathlib import Path
  worker=Path(__file__).resolve().parent/'worker.py'
  code="import sys,runpy;sys.path.insert(0,sys.argv[1]);import gate;gate.review=lambda **kw:(_ for _ in ()).throw(ValueError('synthetic admission denied'));sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name='__main__')"
  for flag in ([],['--preflight-only']):
   r=subprocess.run([sys.executable,'-c',code,str(worker.parent),str(worker),'--out','unused',*flag],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=10)
   self.assertNotEqual(r.returncode,0);self.assertIn('synthetic admission denied',r.stderr)
 def test_existing_cuda_and_failure(self):
  with self.assertRaises(ValueError):run(Torch(True),lambda:None,lambda m:None,lambda e:None,lambda:1)
  t=Torch();events=[]
  def failed():raise MemoryError('synthetic')
  with self.assertRaises(MemoryError):run(t,failed,lambda m:None,events.append,lambda:1)
  self.assertEqual([e['event'] for e in events],['model_load_start','failure']);self.assertEqual(events[-1]['error_type'],'MemoryError')
if __name__=='__main__':unittest.main(verbosity=2)
