import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import component_gate as g

class TestGate(unittest.TestCase):
 def setUp(self):
  self.formal={'status':'passed','complete_96_identity':True,'GT_opened':False,'scoring_performed':False,'cuda_initialized':False,'errors':[],
    'budgets':[{'k':k,'outcome':'completed','compared_images':32,'identity_mismatch_images':[]} for k in (128,256,512)]}
  self.review={'status':'passed','independent_review':True,'complete_96_identity':True,'GT_opened':False,'scoring_performed':False,'errors':[],
    'formal_sha256':'a'*64,'plan_sha256':'b'*64}
  self.bindings={'formal_sha256':'a'*64,'plan_sha256':'b'*64}
 def test_valid_summary(self):g.validate_summary(self.formal,self.review,self.bindings)
 def test_incomplete_or_changed_proof_rejected(self):
  for target,key,value in [('formal','complete_96_identity',False),('review','complete_96_identity',False),
                           ('review','formal_sha256','c'*64),('review','GT_opened',True)]:
   a=copy.deepcopy(self.formal);b=copy.deepcopy(self.review);(a if target=='formal' else b)[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):g.validate_summary(a,b,self.bindings)
 def test_incomplete_or_mismatched_budget_rejected(self):
  for mutate in ('missing','short','mismatch'):
   a=copy.deepcopy(self.formal)
   if mutate=='missing':a['budgets'].pop()
   elif mutate=='short':a['budgets'][2]['compared_images']=31
   else:a['budgets'][2]['identity_mismatch_images']=[17]
   with self.subTest(mutate=mutate),self.assertRaises(ValueError):g.validate_summary(a,self.review,self.bindings)
 def test_missing_actual_proof_fails_closed(self):
  with tempfile.TemporaryDirectory() as d,patch.object(g,'FORMAL',Path(d)/'missing.json'):
   with self.assertRaises(FileNotFoundError):g.require()

if __name__=='__main__':unittest.main(verbosity=2)
