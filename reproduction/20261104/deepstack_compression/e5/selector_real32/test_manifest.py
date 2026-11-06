import copy,unittest
from freeze_manifest import build
class Tests(unittest.TestCase):
 def test_frozen_128_includes_all_pilot_and_refuses_mismatch(self):
  tasks=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST');rows=[]
  for t in tasks:
   for i in range(128):rows.append({'task':t,'qid':str(i),'group_id':t+str(i),'message_sha256':'m','image_sha256':'i'})
  pilot=rows[:11]+rows[128:139]+rows[256:266];selected=build(rows,pilot)
  self.assertEqual(len(selected),128);self.assertEqual(len({x['group_id'] for x in selected}),128);self.assertTrue({x['group_id'] for x in pilot}<={x['group_id'] for x in selected})
  bad=copy.deepcopy(pilot);bad[0]['message_sha256']='tampered'
  with self.assertRaises(ValueError):build(rows,bad)
if __name__=='__main__':unittest.main()
