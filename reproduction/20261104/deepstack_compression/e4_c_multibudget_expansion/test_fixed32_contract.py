import copy,unittest
from fixed32_contract import compare
class Tests(unittest.TestCase):
 def sample(self):
  k=128;n=50+k;fp={'dtype':'torch.bfloat16','shape':[1,2],'sha256':'a'*64};tensors={s:copy.deepcopy(fp) for s in ('input_ids','attention_mask','pixel_values','image_grid_thw','prefill_embeds','prefill_mask','prefill_position_ids','ds0','ds1','ds2')}
  return {'task':'DocVQA_VAL','qid':'1','group_id':'g','image_sha256':'i','message_sha256':'m','budget':k,'ordered_ids':list(range(k)),'spatial_ids':list(range(k)),'greedy_calls':1,'original_prefill_length':1074,'compact_prefill_length':n,'protocol':'fixed32','generation_scalars':{'do_sample':False,'min_new_tokens':32,'max_new_tokens':32,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0},'eos_ids':[999],'output_ids':[3]*31+[999],'tensors':tensors,'adjusted_delta':[[718]],'decode_trace':[{'step':i,'cache_before':n+i,'cache_after':n+i+1,'layer_cache_after':[n+i+1]*36,'cache_position':1074+i,'position_ids':[i,i,i],'rope_delta':[[718]]} for i in range(31)]}
 def test_identity(self):
  x=self.sample();self.assertTrue(compare(x,copy.deepcopy(x)))
 def test_candidate_mutations_rejected(self):
  x=self.sample()
  for mutate in ('ds','token','position','kv','selection','length','early_eos','sha'):
   y=copy.deepcopy(x)
   if mutate=='ds':y['tensors']['ds1']['sha256']='b'*64
   if mutate=='token':y['output_ids'][3]=4
   if mutate=='position':y['decode_trace'][4]['position_ids'][1]+=1
   if mutate=='kv':y['decode_trace'][8]['layer_cache_after'][4]+=1
   if mutate=='selection':y['ordered_ids'][2]=y['ordered_ids'][1]
   if mutate=='length':y['output_ids'].pop()
   if mutate=='early_eos':y['output_ids'][2]=999
   if mutate=='sha':y['tensors']['ds0']['sha256']='z'*64
   with self.subTest(mutate=mutate),self.assertRaises(ValueError):compare(x,y)
 def test_shared_invalid_kv_cannot_pass(self):
  x=self.sample();x['decode_trace'][3]['cache_after']-=1
  with self.assertRaisesRegex(ValueError,'KV'):compare(x,copy.deepcopy(x))
if __name__=='__main__':unittest.main(verbosity=2)
