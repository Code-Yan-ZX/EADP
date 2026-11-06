import unittest
from unittest.mock import patch
import torch,stage_core
from test_contracts import Event,schedule
from contracts import COMPONENTS
class Tests(unittest.TestCase):
 def setUp(self):
  self.row={'task':'D','qid':'1','group_id':'g','message_sha256':'m','image_start':4,'split':'DEV'};self.ref={k:self.row[k] for k in ('task','qid','group_id','message_sha256','image_start')};self.events=[];self.archives=[]
 def run_stage(self,preflight):
  s=schedule();ops={k:(lambda k=k:k) for k in COMPONENTS};refs={k:k for k in COMPONENTS}
  with patch.object(stage_core,'capture_one',return_value=(ops,refs,{'dummy':1},lambda:True)) as capture,patch.object(stage_core,'validate',return_value={'status':'passed'}):
   result=stage_core.execute(torch,None,None,self.row,self.ref,s,s,preflight,None,None,None,None,None,lambda a,b:a==b,repr,Event,lambda:None,lambda:True,self.events.append,lambda a,r:self.archives.append((a,r)))
   return result,capture
 def test_preflight_precedes_capture_and_archive_precedes_profile(self):
  result,capture=self.run_stage(lambda *args:None)
  self.assertEqual(result['measurements'],60);self.assertEqual(len(self.archives),1);self.assertEqual(self.events[0]['status'],'capture_identity_passed');self.assertEqual(sum(x['status']=='measured' for x in self.events),60);self.assertEqual(capture.call_count,1)
 def test_bad_processor_or_identity_prevents_capture(self):
  with patch.object(stage_core,'capture_one',side_effect=AssertionError('must not capture')):
   def fail(*args):raise ValueError('processor')
   with self.assertRaisesRegex(ValueError,'processor'):self.run_stage(fail)
  self.assertEqual(self.events,[]);self.assertEqual(self.archives,[])
if __name__=='__main__':unittest.main(verbosity=2)
