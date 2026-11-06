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
