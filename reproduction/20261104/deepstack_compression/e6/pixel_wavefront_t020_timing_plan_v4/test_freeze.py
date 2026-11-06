import unittest
import freeze
class Tests(unittest.TestCase):
 def test_schedule_count_and_duplicate(self):
  jobs=[{'order':i,'phase':'warmup' if i<6 else 'measurement','task':str(i),'qid':str(i),'arm':'a','repeat':0} for i in range(24)]
  self.assertTrue(freeze.validate_schedule(jobs))
  with self.assertRaises(ValueError):freeze.validate_schedule(jobs[:-1])
  jobs[4]=dict(jobs[3],order=4)
  with self.assertRaises(ValueError):freeze.validate_schedule(jobs)
if __name__=='__main__':unittest.main()

class AdmissionTest(unittest.TestCase):
 def test_missing_independent_result_does_not_create_plan(self):
  from pathlib import Path
  import tempfile
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as t:
   d=Path(t);r=d/'r';i=d/'i';r.mkdir();i.mkdir()
   with patch.object(freeze,'R',r),patch.object(freeze,'I',i):
    with self.assertRaises(ValueError):freeze.freeze()
   self.assertFalse((r/'gpu_plan001.json').exists())

class SmokeTest(unittest.TestCase):
 def test_real_entry_required(self):
  from pathlib import Path
  from unittest.mock import patch
  root=Path('/tmp/test_cwd');run=root/'launch.py'
  start={'cwd':str(root),'command':['python',str(run),'--no-gpu-dependency-smoke'],'worker_pid':10}
  term={'reason':'completed','worker_returncode':0,'remaining_process_group':[],'worker_pid':10}
  result={'status':'passed','gpu_queried':False,'model_loaded':False,'paths':['x']*7}
  with patch.object(freeze,'B',root),patch.object(freeze,'R',root):
   self.assertTrue(freeze.validate_host_smoke(start,term,result))
   for bad in [dict(term,reason='worker_failed'),dict(term,worker_pid=11)]:
    with self.assertRaises(ValueError):freeze.validate_host_smoke(start,bad,result)
   with self.assertRaises(ValueError):freeze.validate_host_smoke(start,term,dict(result,gpu_queried=True))

class PreflightSmokeTest(unittest.TestCase):
 def test_real_preflight_entry_required(self):
  from pathlib import Path
  from unittest.mock import patch
  root=Path('/tmp/test_cwd');start={'cwd':str(root),'command':['python',str(root/'preflight_launch.py'),'--no-gpu-dependency-smoke'],'worker_pid':10};term={'reason':'completed','worker_returncode':0,'remaining_process_group':[],'worker_pid':10};result={'status':'passed','GPU_queried':False,'model_loaded':False}
  with patch.object(freeze,'B',root),patch.object(freeze,'R',root):
   self.assertTrue(freeze.validate_preflight_smoke(start,term,result))
   with self.assertRaises(ValueError):freeze.validate_preflight_smoke(start,term,dict(result,GPU_queried=True))
   with self.assertRaises(ValueError):freeze.validate_preflight_smoke(dict(start,command=['python',str(root/'launch.py'),'--no-gpu-dependency-smoke']),term,result)
