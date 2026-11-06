import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import gpu_admission as a
class Tests(unittest.TestCase):
 def fixture(self,d):
  D=Path(d)/'D';I=Path(d)/'I';D.mkdir();I.mkdir();plan=D/'execution_plan001.json';proof=D/'independent_gpu_prelaunch_review.json';plan.touch();proof.touch()
  cpu=D/'cpu_check001/verification.json';reviewer=I/'review_pixel_wavefront_t020_run_v2_gpu_prelaunch.py';pre=I/'pixel_wavefront_t020_run_v2_preflight_result_review.json';precode=I/'review_pixel_wavefront_t020_run_v2_preflight_result.py'
  h=lambda p:'h:'+str(p);sources={'/fixed':'digest'}
  plan_data={'stage':'t020 three-anchor six-container identity run','threshold':.2,'jobs':[{}]*6,'source_sha256':sources,'GT_opened':False,'scoring_performed':False,'preflight_result_sha256':h(pre)}
  cpu_data={'status':'passed','source_sha256_before':sources,'source_sha256_after':sources,'gpu_used':False,'model_loaded':False,'existing_DEV_predictions_opened':True};preplan_data={'source_sha256':sources}
  pre_data={'status':'passed','independent_review':True,'GT_opened':False,'errors':[],'gpu_authorized':False,'reviewer_sha256':h(precode),**{k:h(D/path) for k,path in {'plan_sha256':'preflight_plan001.json','worker_preflight_sha256':'run_preflight001/worker_preflight.json','launch_sha256':'run_preflight001/launch.json','exit_sha256':'run_preflight001/exit.json','started_sha256':'monitor_preflight001/started.json','terminal_sha256':'monitor_preflight001/terminal.json'}.items()}}
  proof_data={'status':'passed','independent_review':True,'scope':'t020_identity_gpu_prelaunch','GT_opened':False,'scoring_performed':False,'errors':[],'reviewer_sha256':h(reviewer),'source_sha256':sources,'plan_sha256':h(plan),'preflight_result_sha256':h(pre),'cpu_sha256':h(cpu),'cpu_started_sha256':h(D/'monitor_cpu001/started.json'),'cpu_terminal_sha256':h(D/'monitor_cpu001/terminal.json')}
  return D,I,plan,proof,pre,cpu,plan_data,pre_data,proof_data,cpu_data,preplan_data,h,sources
 def test_fake_full_contract_and_tampered_proof(self):
  with tempfile.TemporaryDirectory() as d:
   D,I,plan,proof,pre,cpu,p,pr,r,cp,pp,h,src=self.fixture(d)
   def read(path):return {plan:p,proof:r,pre:pr,cpu:cp,D/'preflight_plan001.json':pp}[path]
   with patch.object(a,'D',D),patch.object(a,'I',I),patch.object(a,'sha',side_effect=h),patch.object(a,'hashes',return_value=src),patch.object(a,'terminal'),patch.object(a,'read',side_effect=read):
    self.assertEqual(a.require_gpu()['source_sha256'],src)
    pr['exit_sha256']='tampered'
    with self.assertRaises(ValueError):a.require_gpu()
    pr['exit_sha256']=h(D/'run_preflight001/exit.json')
    r['independent_review']=False
    with self.assertRaises(ValueError):a.require_gpu()
    r['independent_review']=True;r['source_sha256']={'/fixed':'tampered'}
    with self.assertRaises(ValueError):a.require_gpu()
 def test_missing_plan_rejects_before_source_hashing(self):
  with tempfile.TemporaryDirectory() as d,patch.object(a,'D',Path(d)),patch.object(a,'hashes',side_effect=AssertionError('hash called')):
   with self.assertRaisesRegex(RuntimeError,'GPU locked'):a.require_gpu()
if __name__=='__main__':unittest.main(verbosity=2)
