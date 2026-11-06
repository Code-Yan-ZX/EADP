import copy,hashlib,tempfile,unittest
from pathlib import Path
import numpy as np
from budget_compare import COMMON,compare_metadata,compare_archives
class Tests(unittest.TestCase):
 def sample(self,k):
  x={name:None for name in COMMON};n=1028;start=2;spatial=list(range(k));retained=list(range(start))+[start+i for i in spatial]+list(range(start+1024,n));full=[[[*range(n)]] for _ in range(3)]
  x.update(task='t',qid='1',group_id='g',image_sha256='i',message_sha256='m',budget=k,ds_on=True,selection_mode='keep_all' if k==1024 else 'author_greedy',greedy_calls=0 if k==1024 else 1,
   ordered_ids=spatial,spatial_ids=spatial,image_start=start,original_length=n,retained_ids=retained,full_position_ids=full,full_delta=[[0]],prefill_position_ids=[[[axis[0][j] for j in retained]] for axis in full],adjusted_delta=[[1024-k]],injected_mask=[[False]*start+[True]*k+[False]*2],injected_deepstack=[1,2,3],prefill_logits={'shape':[1]},cache_fingerprints=[[1]],forced=False,ds_none=[False]*3)
  x.update(output_ids=[9],eos_ids=[9],max_new_tokens=2048,forced_token=None,eos=True,truncated=False)
  x['decode_trace']=[{'position_ids':x['prefill_position_ids'],'cache_position':list(range(len(retained))),'rope_delta':x['adjusted_delta'],'layer_cache_lengths':[0]*36,'layer_cache_lengths_after':[len(retained)]*36}]
  meta={'shape':[1,8,len(retained),128],'sha256':'a'*64};x['cache_fingerprints']=[[{'key':copy.deepcopy(meta),'value':copy.deepcopy(meta)} for _ in range(36)]]
  y=copy.deepcopy(x);y['actual_injection_verified']=True
  return x,y
 def test_all_four_budgets_and_correlated_wrong_delta(self):
  for k in (128,256,512,1024):
   x,y=self.sample(k);self.assertEqual(compare_metadata(x,y)['status'],'passed')
   x['adjusted_delta']=y['adjusted_delta']=[[99999]]
   with self.assertRaisesRegex(ValueError,'delta'):compare_metadata(x,y)
 def test_forced_32_decode_detects_matching_wrong_cache(self):
  x,y=self.sample(128);seq=len(x['retained_ids']);delta=x['adjusted_delta'][0][0]
  x.update(forced=True,forced_token=7,output_ids=[7]*33,eos_ids=[],max_new_tokens=33,eos=False,truncated=False)
  trace=[];fps=[]
  for i in range(33):
   prev=0 if i==0 else seq+i-1;after=seq+i
   trace.append({'position_ids':x['prefill_position_ids'] if i==0 else [[[seq+i-1+delta]]]*3,'cache_position':list(range(seq)) if i==0 else [seq+i-1],'rope_delta':[[delta]],'layer_cache_lengths':[prev]*36,'layer_cache_lengths_after':[after]*36})
   meta={'shape':[1,8,after,128],'sha256':'a'*64};fps.append([{'key':copy.deepcopy(meta),'value':copy.deepcopy(meta)} for _ in range(36)])
  x['decode_trace']=trace;x['cache_fingerprints']=fps;y=copy.deepcopy(x);y['actual_injection_verified']=True
  self.assertEqual(compare_metadata(x,y)['status'],'passed')
  x['decode_trace'][30]['layer_cache_lengths_after'][35]+=1;y['decode_trace']=copy.deepcopy(x['decode_trace'])
  with self.assertRaisesRegex(ValueError,'trajectory'):compare_metadata(x,y)
 def test_mask_and_keepall_rejected(self):
  x,y=self.sample(128);x['injected_mask']=y['injected_mask']=[[False]*132]
  with self.assertRaisesRegex(ValueError,'DS mask'):compare_metadata(x,y)
  x,y=self.sample(1024);x['greedy_calls']=y['greedy_calls']=1
  with self.assertRaisesRegex(ValueError,'method path'):compare_metadata(x,y)
 def test_actual_selection_and_ds_arrays(self):
  x,y=self.sample(128);k=128;n=1024;v=np.arange(n*4,dtype=np.uint8).reshape(n,4);pre=np.zeros((1,k+4,4),dtype=np.uint8);pre[0,2:2+k]=v[:k]
  arr={'visual':v,'pruned':v[:k],'prefill':pre,'prefill_logits':np.array([1])}
  for i in range(3):arr['ds'+str(i)]=v.copy();arr['injected'+str(i)]=v[:k].copy()
  with tempfile.TemporaryDirectory() as d:
   a=Path(d)/'a.npz';b=Path(d)/'b.npz'
   def write():
    np.savez(a,**arr);np.savez(b,**{key:value for key,value in arr.items() if key!='pruned'})
    x['arrays']={'kind':'full_arrays','path':str(a),'sha256':hashlib.sha256(a.read_bytes()).hexdigest()};y['arrays']={'kind':'full_arrays','path':str(b),'sha256':hashlib.sha256(b.read_bytes()).hexdigest()}
   write();self.assertGreater(compare_archives(x,y)['n_common_arrays'],5)
   arr['injected1']=arr['injected1'].copy();arr['injected1'][0,0]^=1;write()
   with self.assertRaisesRegex(ValueError,'DS stream'):compare_archives(x,y)
if __name__=='__main__':unittest.main(verbosity=2)
