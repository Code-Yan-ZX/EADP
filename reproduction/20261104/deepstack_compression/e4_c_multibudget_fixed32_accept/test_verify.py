import copy,unittest
import verify
class Tests(unittest.TestCase):
 def test_actual_arm_event_and_missing_or_mismatched_raw(self):
  job={'task':'D','qid':'1','group_id':'g','budget':128,'side':'fresh','protocol':'fixed32'}
  rec={'task':'D','qid':'1','group_id':'g','budget':128,'ds_on':True,'forced':False,'protocol':'fixed32','output_ids':[1]*32,'eos_ids':[9],'generation_scalars':{'max_new_tokens':32},'min_new_tokens':32}
  segment=[{'event':'arm_started','job':job},{'event':'raw_generation','job':job,'raw':dict(rec)},{'event':'arm_completed','job':job,'record':dict(rec)}]
  self.assertEqual(verify.arm(segment,job),rec)
  broken=copy.deepcopy(segment);broken[1]['raw']['output_ids']=[3]*32
  with self.assertRaisesRegex(ValueError,'raw/result'):verify.arm(broken,job)
  with self.assertRaisesRegex(ValueError,'unordered'):verify.arm(segment[1:],job)
  broken=copy.deepcopy(segment);broken[1]['job']['side']='light'
  with self.assertRaisesRegex(ValueError,'unordered'):verify.arm(broken,job)
if __name__=='__main__':unittest.main()
