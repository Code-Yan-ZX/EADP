import copy,json,sys,tempfile,unittest
import numpy as np
from pathlib import Path
sys.path.insert(1,str(Path(__file__).resolve().parent.parent/'e4_multistream_expansion352'))
import norm_contract,pilot_contract
import pilot_engine as p
class Cuda:
 def synchronize(self):pass
class T:cuda=Cuda();float64='float64'
class Tensor:
 def __init__(self,v):self.v=np.asarray(v,dtype=np.float64)
 def detach(self):return self
 def to(self,**kw):return self
 def cpu(self):return self
 def reshape(self,*args):return self
 def tolist(self):return self.v.tolist()
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
  meta=lambda _: {'hash':'fixed'}
  val=arr.tolist();base_ref={'processor':{k:meta(arr) for k in proc},'visual_sha256':meta(arr),'deepstack_full_sha256':[meta(arr)]*3,'ordered_ids':val,'spatial_ids':sorted(val),'prefill_embed_sha256':meta(arr),'attention_mask_sha256':meta(arr),'prefill_position_ids':val,'rope_delta':val,'injected_mask':val,'injected_deepstack':[meta(arr)]*3};refs={(r['task'],r['qid']):base_ref for r in chosen}
  records={(r['task'],r['qid']):dict(r,budget=256,ds_on=True,selection_mode='author_greedy',actual_injection_verified=True,processor={k:meta(arr) for k in proc},main_visual=meta(arr),full_deepstack=[meta(arr)]*3) for r in chosen}
  actual={'processor':proc,'visual':(arr,[arr,arr,arr]),'ordered':arr,'prefill':{'inputs_embeds':arr,'attention_mask':arr,'position_ids':arr,'visual_pos_masks':arr,'deepstack_visual_embeds':[arr,arr,arr]},'rope_delta':arr};events=[]
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'out';r=p.run(chosen,rows,refs,records,out=out,torch=T(),wrapper=None,native=None,author_generate=None,Adapter=None,Capture=None,clone=None,capture_one=lambda *args:({}, {},actual,lambda:True),preflight_one=lambda *args:None,validate_e1=lambda *args:{'status':'passed'},tensor_meta=meta,identity_check=pilot_contract.check_identity,norm_mod=norm_contract,feature_mod=feat,journal=events.append)
   self.assertEqual(len(r),9);self.assertEqual(len(list(out.glob('*_norms.npy'))),9);self.assertEqual(json.loads((out/'status.json').read_text())['images_passed'],9)
   stored=np.load(out/'image_000_norms.npy',allow_pickle=False);entry=json.loads((out/'image_000.json').read_text());self.assertEqual(entry['features'],norm_contract.features_from_norms(stored,feat))
   self.assertIn('prefill_embed_sha256',entry['actual_identity']);bad=copy.deepcopy(refs);bad[('T','0')]['prefill_embed_sha256']='mismatch'
   with self.assertRaises(ValueError):p.run(chosen,rows,bad,records,out=Path(d)/'bad',torch=T(),wrapper=None,native=None,author_generate=None,Adapter=None,Capture=None,clone=None,capture_one=lambda *args:({}, {},actual,lambda:True),preflight_one=lambda *args:None,validate_e1=lambda *args:{'status':'passed'},tensor_meta=meta,identity_check=pilot_contract.check_identity,norm_mod=norm_contract,feature_mod=feat,journal=events.append)
   self.assertEqual(json.loads((Path(d)/'bad/status.json').read_text())['images_passed'],0)
 def test_invalid_cohort_never_invokes_callbacks(self):
  with self.assertRaises(ValueError):p.run([],[],{}, {},out='not_created',torch=T(),wrapper=None,native=None,author_generate=None,Adapter=None,Capture=None,clone=None,capture_one=None,preflight_one=None,validate_e1=None,tensor_meta=None,identity_check=None,norm_mod=None,feature_mod=None,journal=lambda _:self.fail())
if __name__=='__main__':unittest.main()
