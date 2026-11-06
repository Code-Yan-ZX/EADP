import copy
import unittest
from identity_contract import validate_e1_identity

class IdentityTests(unittest.TestCase):
 def setUp(self):
  self.sample={'task':'DocVQA_VAL','qid':'1','group_id':'g','message_sha256':'m','image_sha256':'i'}
  self.ref={'task':'DocVQA_VAL','qid':'1','group_id':'g','message_sha256':'m','arm':'P3-D1','forced':False,'processor':{'input_ids':{'sha256':'1'}},'visual_sha256':{'sha256':'2'},'deepstack_full_sha256':[{'sha256':str(i)} for i in range(3)],'original_position_ids':[[[1]],[[1]],[[1]]],'full_rope_delta':[[-2]]}
  self.actual={**self.sample,'processor':copy.deepcopy(self.ref['processor']),'main_visual':copy.deepcopy(self.ref['visual_sha256']),'full_deepstack':copy.deepcopy(self.ref['deepstack_full_sha256']),'full_position_ids':copy.deepcopy(self.ref['original_position_ids']),'full_delta':copy.deepcopy(self.ref['full_rope_delta'])}
 def test_matching(self):self.assertTrue(validate_e1_identity(self.actual,self.sample,self.ref))
 def test_each_observed_component_rejected(self):
  for key in ('processor','main_visual','full_deepstack','full_position_ids','full_delta','image_sha256','qid','group_id','message_sha256'):
   a=copy.deepcopy(self.actual);a[key]=None
   with self.subTest(key=key),self.assertRaises(ValueError):validate_e1_identity(a,self.sample,self.ref)
 def test_ref_must_be_correct_natural_arm(self):
  for key,value in [('arm','P1-D0'),('forced',True)]:
   r=copy.deepcopy(self.ref);r[key]=value
   with self.assertRaises(ValueError):validate_e1_identity(self.actual,self.sample,r)
if __name__=='__main__':unittest.main()
