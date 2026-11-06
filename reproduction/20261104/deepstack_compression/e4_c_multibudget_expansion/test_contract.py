import unittest
from unittest.mock import patch
from pilot_contract import TASKS,build
import gate
class Tests(unittest.TestCase):
 def fixture(self):
  cohort={};identity={}
  for task in TASKS:
   m=[f'{task}-m{i:02}' for i in range(64)];w=[f'{task}-w{i:02}' for i in range(10)];cohort[task]={'measured':m,'warmup':w}
   for g in m+w:identity[(task,g)]={'task':task,'qid':g,'group_id':g,'image_path':'x','image_sha256':'a'*64,'message_sha256':'b'*64}
  return cohort,identity
 def test_complete_deterministic_blocks(self):
  c,i=self.fixture();groups,jobs=build(c,i);self.assertEqual(build(c,i),(groups,jobs));self.assertEqual(len(jobs),1520);self.assertEqual(sum(j['phase']=='measure' for j in jobs),1280);self.assertEqual(sum(j['phase']=='warmup' for j in jobs),240)
  for j in jobs[::8]:
   block=jobs[j['order']:j['order']+8];self.assertEqual(len({(x['budget'],x['protocol']) for x in block}),8);self.assertTrue(all(x['group_id']==j['group_id'] for x in block))
 def test_overlap_and_missing_identity_fail(self):
  c,i=self.fixture();c[TASKS[0]]['warmup'][0]=c[TASKS[0]]['measured'][0]
  with self.assertRaises(ValueError):build(c,i)
  c,i=self.fixture();i.clear()
  with self.assertRaises(KeyError):build(c,i)
 def test_gpu_locked(self):
  with self.assertRaisesRegex(RuntimeError,'GPU LOCKED'):gate.require_gpu()
 def test_deadline_and_source(self):
  plan={'budget_source_sha256':'x','source_sha256':{},'GPU_authorized':False,'measured_jobs':1280,'warmup_jobs':240}
  with patch('gate.sha',return_value='x'),patch('gate.Path.read_text',return_value='{"budget_end_unix":100}'):
   self.assertTrue(gate.check_plan(plan,50,30,10))
   with self.assertRaisesRegex(ValueError,'insufficient'):gate.check_plan(plan,70,30,1)
  with patch('gate.sha',return_value='different'):
   with self.assertRaisesRegex(ValueError,'changed'):gate.check_plan(plan,50,30,10)
if __name__=='__main__':unittest.main(verbosity=2)
