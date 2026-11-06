import json,sys,tempfile,unittest
import numpy as np
import norm_contract,pilot_contract
from pathlib import Path
import pilot_engine as p
class Cuda:
 def synchronize(self):pass
class T:cuda=Cuda();float64='float64'
class Tensor:
 def __init__(self,v):self.v=np.asarray(v,dtype=np.float64)
 def detach(self):return self
 def to(self,**kw):return self
 def numpy(self):return self.v
class EngineTests(unittest.TestCase):
 def test_missing_input_preserves_failure_and_remainder(self):
  chosen=[{'task':'T','qid':str(i)} for i in range(9)];rows=[{'task':'T','qid':str(i)} for i in range(384)];events=[]
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):p.run(chosen,rows,{}, {},out=Path(d)/'out',torch=T(),wrapper=None,native=None,author_generate=None,Adapter=None,Capture=None,clone=None,capture_one=None,preflight_one=None,validate_e1=None,tensor_meta=None,identity_check=None,norm_mod=None,feature_mod=None,journal=events.append)
   result=json.loads((Path(d)/'out/status.json').read_text());self.assertEqual(result['status'],'failed');self.assertEqual([x['status'] for x in result['rows']],['failed']+['not_evaluated']*8);self.assertEqual(sum(x['status']=='image_not_evaluated' for x in events),8)
 def test_nine_image_success_and_norms_reconstruction(self):
  feat=sys.modules.get('_multistream_prior_features_352') or norm_contract.original();arr=Tensor(np.ones((1024,2)));proc={k:arr for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
  chosen=[dict(task='T',qid=str(i),group_id='g'+str(i),image_sha256='i',message_sha256='m') for i in range(9)]
  rows=[dict(task='T',qid=str(i),group_id='g'+str(i),image_sha256='i',message_sha256='m',split='DEV') for i in range(384)]
  refs={(r['task'],r['qid']):{} for r in chosen};meta=lambda _: {'hash':'fixed'}
  records={(r['task'],r['qid']):dict(r,budget=256,ds_on=True,selection_mode='author_greedy',actual_injection_verified=True,processor={k:meta(arr) for k in proc},main_visual=meta(arr),full_deepstack=[meta(arr)]*3) for r in chosen}
  actual={'processor':proc,'visual':(arr,[arr,arr,arr])};events=[]
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'out';r=p.run(chosen,rows,refs,records,out=out,torch=T(),wrapper=None,native=None,author_generate=None,Adapter=None,Capture=None,clone=None,capture_one=lambda *args:({}, {},actual,lambda:True),preflight_one=lambda *args:None,validate_e1=lambda *args:{'status':'passed'},tensor_meta=meta,identity_check=pilot_contract.check_identity,norm_mod=norm_contract,feature_mod=feat,journal=events.append)
   self.assertEqual(len(r),9);self.assertEqual(len(list(out.glob('*_norms.npy'))),9);self.assertEqual(json.loads((out/'status.json').read_text())['images_passed'],9)
   stored=np.load(out/'image_000_norms.npy',allow_pickle=False);entry=json.loads((out/'image_000.json').read_text());self.assertEqual(entry['features'],norm_contract.features_from_norms(stored,feat))
 def test_invalid_cohort_never_invokes_callbacks(self):
  with self.assertRaises(ValueError):p.run([],[],{}, {},out='not_created',torch=T(),wrapper=None,native=None,author_generate=None,Adapter=None,Capture=None,clone=None,capture_one=None,preflight_one=None,validate_e1=None,tensor_meta=None,identity_check=None,norm_mod=None,feature_mod=None,journal=lambda _:self.fail())
if __name__=='__main__':unittest.main()
