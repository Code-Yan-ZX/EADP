import copy,unittest
from unittest.mock import patch
from pathlib import Path
import smoke_gate as g
class SmokeGateTests(unittest.TestCase):
 def fixture(self):
  anchors=[{'task':t,'qid':str(i),'cpu_K':k} for i,(t,k) in enumerate((('DocVQA_VAL',5),('TextVQA_VAL',6),('AI2D_TEST',7),('DocVQA_VAL',1)))];files={str((g.S/x).resolve()):'hash:'+str((g.S/x).resolve()) for x in ('plan001.json','monitor_smoke001/started.json','monitor_smoke001/terminal.json','run_smoke001/launch.json','run_smoke001/exit.json','independent_prelaunch_review.json')};files.update({str(g.S/('f'+str(i))):'hash:'+str(g.S/('f'+str(i))) for i in range(10)});r={'status':'passed','independent_review':True,'images':4,'GT_opened':False,'scoring_performed':False,'errors':[],'reviewer_sha256':'hash:'+str(g.R/'review_pixel_t020_result.py'),'observed':[{'task':x['task'],'qid':x['qid'],'visual_tokens':x['cpu_K'],'cpu_gpu_indices_equal':True,'eos':True,'generated_tokens':1} for x in anchors],'files_sha256':files};plan={'smoke_review_sha256':'hash:'+str(g.R/'pixel_t020_result_review.json'),'smoke_plan_sha256':'hash:'+str(g.S/'plan001.json')};return plan,r,{'anchors':anchors}
 def check(self,plan,r,sp):
  def read(p):return sp if p==g.S/'plan001.json' else r
  with patch.object(g,'supervised'),patch.object(g,'read',side_effect=read),patch.object(g,'sha',side_effect=lambda p:'hash:'+str(p)):
   return g.validate(plan)
 def test_positive_and_identity_mismatch(self):
  p,r,s=self.fixture();self.assertTrue(self.check(p,r,s)['hashed_prior_smoke_predictions']);r['observed'][0]['cpu_gpu_indices_equal']=False
  with self.assertRaises(ValueError):self.check(p,r,s)
 def test_failure_missing_or_changed_file(self):
  p,r,s=self.fixture()
  for kind in ('failed','missing','changed'):
   q=copy.deepcopy(r)
   if kind=='failed':q['status']='failed'
   if kind=='missing':q['files_sha256'].pop(next(iter(q['files_sha256'])))
   if kind=='changed':q['files_sha256'][next(iter(q['files_sha256']))]='wrong'
   with self.assertRaises(ValueError):self.check(p,q,s)
 def test_missing_extreme_and_noncompletion(self):
  p,r,s=self.fixture();r['observed'][-1]['eos']=False
  with self.assertRaises(ValueError):self.check(p,r,s)
  p,r,s=self.fixture();r['observed'][-1]['visual_tokens']=2
  with self.assertRaises(ValueError):self.check(p,r,s)
if __name__=='__main__':unittest.main()
