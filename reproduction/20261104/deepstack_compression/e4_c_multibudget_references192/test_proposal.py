import copy,unittest
import proposal
class Tests(unittest.TestCase):
 def test_full_222_and_duplicate_or_nond_ev_rejected(self):
  rows=[{'task':'D','qid':str(i),'group_id':str(i),'split':'DEV'} for i in range(222)]
  p={'measurement_identities':rows[:192],'warmup_identities':rows[192:]}
  self.assertEqual(len(proposal.inputs(p)),222)
  for kind in ('group','split','missing'):
   q=copy.deepcopy(p)
   if kind=='group':q['warmup_identities'][0]['group_id']=q['measurement_identities'][0]['group_id']
   if kind=='split':q['measurement_identities'][0]['split']='FINAL'
   if kind=='missing':q['measurement_identities'].pop()
   with self.assertRaises(ValueError):proposal.inputs(q)
if __name__=='__main__':unittest.main()
