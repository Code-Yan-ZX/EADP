import copy,importlib.util,sys,unittest
from pathlib import Path
p=Path(__file__).parent/'review_e4c_multibudget_references192_result.py';spec=importlib.util.spec_from_file_location('_review_888_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tok:
 def decode(self,ids,**kw):return 'ok'
class Test(unittest.TestCase):
 def sample(self):
  job={'task':'Doc','qid':'1','group_id':'g','budget':128,'protocol':'fixed32'};r={k:None for k in m.FIELDS};r.update(task='Doc',qid='1',group_id='g',budget=128,protocol='fixed32',ds_on=True,forced=False,min_new_tokens=32,max_new_tokens=32,generation_scalars={'max_new_tokens':32},eos_ids=[9],output_ids=[1]*32,answer='ok',original_length=1026,adjusted_delta=[[0]],prefill_position_ids=[[[0]]]*3,arrays={'kind':'summary_only'})
  seq=130;r['decode_trace']=[]
  for i in range(32):r['decode_trace'].append({'position_ids':r['prefill_position_ids'] if i==0 else [[[seq+i-1]]]*3,'cache_position':list(range(seq)) if i==0 else [seq+i-1],'rope_delta':[[0]],'layer_cache_lengths':[] if i==0 else [seq+i-1]*36,'layer_cache_lengths_after':[seq+i]*36})
  value={'dtype':'torch.bfloat16','shape':[1,8,seq,128],'sha256':'a'*64};r['cache_fingerprints']=[[{'key':copy.deepcopy(value),'value':copy.deepcopy(value)} for _ in range(36)]]
  panel=copy.deepcopy(r);panel['actual_injection_verified']=True;panel['generation_scalars']={'max_new_tokens':2048}
  raw={k:r[k] for k in ('task','qid','group_id','budget','ds_on','forced','protocol','output_ids','eos_ids','generation_scalars','min_new_tokens')}
  return job,r,panel,raw
 def test_positive_and_mutations(self):
  job,r,panel,raw=self.sample();m.validate_arm(job,r,panel,raw,Tok())
  for mode in ('identity','cache','early_eos','fingerprint','prefill'):
   a=copy.deepcopy(r);b=copy.deepcopy(raw)
   if mode=='identity':a['qid']=b['qid']='2'
   if mode=='cache':a['decode_trace'][17]['layer_cache_lengths_after'][2]+=1
   if mode=='early_eos':a['output_ids'][0]=b['output_ids'][0]=9
   if mode=='fingerprint':a['cache_fingerprints'][0][3]['key']['sha256']=''
   if mode=='prefill':a['ordered_ids']=[4]
   with self.assertRaises(ValueError,msg=mode):m.validate_arm(job,a,panel,b,Tok())
if __name__=='__main__':unittest.main()
