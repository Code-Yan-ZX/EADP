import copy,unittest
from compare_fixed import COMMON,compare_metadata
from fixed_common import verify_kwargs

class Tests(unittest.TestCase):
 def sample(self,k):
  x={name:None for name in COMMON};n=1028;start=2;spatial=list(range(k));retained=list(range(start))+[start+i for i in spatial]+list(range(start+1024,n));full=[[[*range(n)]] for _ in range(3)]
  x.update(task='t',qid='1',group_id='g',image_sha256='i',message_sha256='m',budget=k,ds_on=True,selection_mode='keep_all' if k==1024 else 'author_greedy',greedy_calls=0 if k==1024 else 1,ordered_ids=spatial,spatial_ids=spatial,image_start=start,original_length=n,retained_ids=retained,full_position_ids=full,full_delta=[[0]],prefill_position_ids=[[[axis[0][j] for j in retained]] for axis in full],adjusted_delta=[[1024-k]],injected_mask=[[False]*start+[True]*k+[False]*2],injected_deepstack=[1,2,3],prefill_logits={'shape':[1]},forced=False,ds_none=[False]*3,protocol='fixed32',min_new_tokens=32,output_ids=list(range(100,132)),eos_ids=[9],max_new_tokens=32,forced_token=None,eos=False,truncated=False)
  seq=len(retained);delta=1024-k;trace=[];fps=[]
  for i in range(32):
   prev=0 if i==0 else seq+i-1;after=seq+i
   trace.append({'position_ids':x['prefill_position_ids'] if i==0 else [[[seq+i-1+delta]]]*3,'cache_position':list(range(seq)) if i==0 else [seq+i-1],'rope_delta':[[delta]],'layer_cache_lengths':[prev]*36,'layer_cache_lengths_after':[after]*36})
   meta={'dtype':'torch.bfloat16','shape':[1,8,after,128],'sha256':'a'*64};fps.append([{'key':copy.deepcopy(meta),'value':copy.deepcopy(meta)} for _ in range(36)])
  x['decode_trace']=trace;x['cache_fingerprints']=fps;y=copy.deepcopy(x);y['actual_injection_verified']=True
  return x,y
 def test_all_budgets_and_full31_decode(self):
  for k in (128,256,512,1024):
   x,y=self.sample(k);self.assertEqual(compare_metadata(x,y)['status'],'passed')
 def test_joint_wrong_position_and_cache_are_rejected(self):
  x,y=self.sample(128);x['adjusted_delta']=y['adjusted_delta']=[[999]]
  with self.assertRaisesRegex(ValueError,'delta'):compare_metadata(x,y)
  x,y=self.sample(128);x['decode_trace'][31]['layer_cache_lengths_after'][35]+=1;y['decode_trace']=copy.deepcopy(x['decode_trace'])
  with self.assertRaisesRegex(ValueError,'trajectory'):compare_metadata(x,y)
 def test_wrong_32_output_and_three_ds(self):
  for field,value in [('output_ids',list(range(100,110))+[9]+list(range(111,132))),('min_new_tokens',31),('injected_deepstack',[1,3,2])]:
   x,y=self.sample(128)
   if field=='injected_deepstack':x[field]=value
   else:x[field]=y[field]=value
   with self.assertRaises(ValueError):compare_metadata(x,y)
 def test_actual_recipe_checked_at_boundary(self):
  original={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0};panel={'generation_scalars':original,'eos_ids':[9]};kw=dict(original,max_new_tokens=32,min_new_tokens=32)
  self.assertTrue(verify_kwargs(kw,[9],panel))
  for k in kw:
   bad=dict(kw);bad[k]='bad'
   with self.assertRaises(ValueError):verify_kwargs(bad,[9],panel)
  with self.assertRaises(ValueError):verify_kwargs(kw,[10],panel)
  with self.assertRaises(ValueError):verify_kwargs(dict(kw,logits_processor=[]),[9],panel)
 def test_matching_fake_kv_is_rejected(self):
  for field,value in [('dtype','torch.float32'),('shape',[1,9,133,128]),('sha256','z'*64)]:
   x,y=self.sample(128);x['cache_fingerprints'][0][0]['key'][field]=value;y['cache_fingerprints']=copy.deepcopy(x['cache_fingerprints'])
   with self.assertRaisesRegex(ValueError,'fingerprint'):compare_metadata(x,y)
if __name__=='__main__':unittest.main()
