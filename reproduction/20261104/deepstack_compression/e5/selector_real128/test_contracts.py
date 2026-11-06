import copy,unittest
from expansion_contract import partition,selected,performance_eligible
class Tests(unittest.TestCase):
 def sample(self):
  rows=[dict(task='t',qid=str(i),group_id='g'+str(i),message_sha256='m'+str(i),image_sha256='i'+str(i)) for i in range(128)];return {'status':'frozen','budgets':[128,256,512],'full_128':rows,'pilot':copy.deepcopy(rows[::4])}
 def test_partition(self):
  s=self.sample();r=partition(s);self.assertEqual(len(r),96);self.assertEqual([x['qid'] for x in r],[str(i) for i in range(128) if i%4])
 def test_changed_pilot_or_duplicate(self):
  s=self.sample();s['pilot'][0]['message_sha256']='bad'
  with self.assertRaises(ValueError):partition(s)
  s=self.sample();s['full_128'][-1]['group_id']=s['full_128'][0]['group_id']
  with self.assertRaises(ValueError):partition(s)
 def test_same_set_is_not_same_order(self):
  a=list(range(128));b=a.copy();b[20],b[21]=b[21],b[20];r=selected(a,b,128)
  self.assertTrue(r['spatial_set_equal']);self.assertFalse(r['ordered_equal']);self.assertEqual(r['ordered_positions_differing'],2);self.assertEqual(r['first_difference_zero_based'],20)
  self.assertFalse(performance_eligible(True,True,True,r))
 def test_unavailable_is_not_success(self):
  a=list(range(128));r=selected(a,a,128)
  for flags in [(False,True,True),(True,False,True),(True,True,False)]:self.assertFalse(performance_eligible(*flags,r))
  self.assertTrue(performance_eligible(True,True,True,r))
 def test_invalid_inputs(self):
  a=list(range(128));b=a.copy();b[-1]=b[0]
  with self.assertRaises(ValueError):selected(a,b,128)
  with self.assertRaises(ValueError):selected(a,a,256)
if __name__=='__main__':unittest.main(verbosity=2)
