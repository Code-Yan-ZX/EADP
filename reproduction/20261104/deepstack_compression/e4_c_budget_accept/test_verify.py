import copy,importlib.util,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_budget_accept_test',HERE/'verify.py');verify=importlib.util.module_from_spec(spec);spec.loader.exec_module(verify)
class Tests(unittest.TestCase):
 def test_reference_uses_accepted_records_inventory(self):
  job={'task':'D','qid':'1','budget':128};refs={'rows':[],'records':{('D','1',128):{'answer':'ok'}}}
  self.assertEqual(verify.reference_record(refs,job),{'answer':'ok'})
  with self.assertRaises(KeyError):verify.reference_record(refs,dict(job,budget=512))
 def test_order_raw_and_output_integrity(self):
  job={'task':'T','qid':'1','group_id':'g','budget':256,'forced':False};raw=dict(job,ds_on=True,output_ids=[1,2],eos_ids=[2],generation_scalars={'max_new_tokens':2048})
  seg=[{'event':'arm_started','side':'fresh','job':job},{'event':'raw_generation','side':'fresh','job':job,'raw':raw},
       {'event':'arm_completed','side':'fresh','job':job,'record':copy.deepcopy(raw)}]
  self.assertEqual(verify.event(seg,job,'fresh')['output_ids'],[1,2])
  bad=copy.deepcopy(seg);bad[2]['record']['output_ids']=[1,3]
  with self.assertRaisesRegex(ValueError,'raw/result'):verify.event(bad,job,'fresh')
  bad=copy.deepcopy(seg);bad[0]['side']='light'
  with self.assertRaisesRegex(ValueError,'unordered'):verify.event(bad,job,'fresh')
  with self.assertRaisesRegex(ValueError,'unordered'):verify.event(seg[:2],job,'fresh')
if __name__=='__main__':unittest.main()
