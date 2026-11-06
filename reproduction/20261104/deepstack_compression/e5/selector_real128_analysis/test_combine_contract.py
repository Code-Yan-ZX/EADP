import unittest
from combine_contract import summarize
class Tests(unittest.TestCase):
 def test_separate_strict_order_and_set_denominators(self):
  rows=[{'task':'t','qid':str(i),'group_id':'g'+str(i)} for i in range(128)];pilot=rows[:32];rest=rows[32:];cases=[]
  for row in rows:
   for k in (128,256,512):
    candidate={'ordered_equal':True,'spatial_set_equal':True,'all_repeats_ordered_equal':True}
    if row['qid']=='7' and k==512:candidate={'ordered_equal':False,'spatial_set_equal':True,'all_repeats_ordered_equal':False}
    if row['qid']=='40' and k==512:candidate=None
    cases.append({**row,'budget':k,'fused':candidate,'graph':candidate})
  out=summarize(pilot,rest,cases[:96],cases[96:]);x=out['combined_128']['512']['fused']
  self.assertEqual(x,{'planned':128,'strict_order_pass':126,'strict_order_fail':1,'unavailable':1,'initial_spatial_equal_posthoc':127,'initial_spatial_different_posthoc':0})
  self.assertEqual(out['observed_pilot_32']['512']['fused']['strict_order_fail'],1)
  with self.assertRaises(ValueError):summarize(pilot,rest,cases[:95],cases[96:])
if __name__=='__main__':unittest.main(verbosity=2)
