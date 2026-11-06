import unittest
from copy import deepcopy
from verify import tensor_hash_int64,validate_item
class T:
 def decode(self,ids,**kwargs):return 'ok'
def example():
 prefix=list(range(1028));n=3;d=[[-992]];initial=7
 anchor={'task':'D','qid':'1','group_id':'g','message_sha256':'m','image_sha256':'i','image_start':4,'processor':{'input_ids':tensor_hash_int64(prefix)}}
 trace=[{'rope_delta':d,'model_rope_delta':d,'layers':[{'key':[1,8,initial+i,128],'value':[1,8,initial+i,128]} for _ in range(36)],'position_ids':None if i==0 else [[[61]]]*3,'cache_position':list(range(initial)) if i==0 else [1028]} for i in range(2)]
 raw={'full_sequence_ids':prefix+[42,99],'generated_token_ids':[42,99],'prompt_length':1028}
 row={'task':'D','qid':'1','group_id':'g','message_sha256':'m','image_sha256':'i','status':'completed','selected_count':n,'selected_indices':[0,2,4],'visual_rows':n,'ds_rows':[n]*3,'prefill_ds_rows':[n]*3,'prefill_visual_count':n,'prefill_ds_match':[True]*3,'prefill_main_match':True,'prefill_text_match':True,'prefill_positions_match':True,'hook_cleanup_valid':True,'generation_kwargs':{'do_sample':False,'max_new_tokens':2048,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0,'num_beams':1,'use_cache':True},'processor_identity':{'processor':anchor['processor'],'image_start':4,'visual_tokens':1024},'sequence_length':1028,'generated_count':2,'ends_eos':True,'eos_token_ids':[99],'decoded':'ok','trace':trace,'original_rope_delta':d,'original_max_position':60}
 return row,raw,anchor
class Tests(unittest.TestCase):
 def test_valid(self):
  r,w,a=example();self.assertEqual(validate_item(r,w,a,T())['selected_count'],3)
 def test_invalid_dynamic_k_and_ds(self):
  for mode in ('index','ds'):
   r,w,a=example()
   if mode=='index':r['selected_indices']=[0,2,2]
   else:r['prefill_ds_rows'][2]=2
   with self.assertRaises(ValueError):validate_item(r,w,a,T())
 def test_reject_changed_prefix_or_short_generation(self):
  r,w,a=example();w['full_sequence_ids'][0]=-1
  with self.assertRaisesRegex(ValueError,'prompt'):validate_item(r,w,a,T())
  r,w,a=example();r['ends_eos']=False;r['eos_token_ids']=[88]
  with self.assertRaisesRegex(ValueError,'termination'):validate_item(r,w,a,T())
 def test_reject_cache_position_and_decode(self):
  r,w,a=example();r['trace'][1]['layers'][4]['key'][-2]=3
  with self.assertRaisesRegex(ValueError,'KV'):validate_item(r,w,a,T())
  r,w,a=example();r['decoded']='wrong'
  with self.assertRaisesRegex(ValueError,'decode'):validate_item(r,w,a,T())
 def test_reject_generation_contract_and_forward_count(self):
  r,w,a=example();r['generation_kwargs']['top_k']=20
  with self.assertRaisesRegex(ValueError,'scalars'):validate_item(r,w,a,T())
  r,w,a=example();r['trace'].pop()
  with self.assertRaisesRegex(ValueError,'forward'):validate_item(r,w,a,T())
if __name__=='__main__':unittest.main(verbosity=2)
