import unittest
from arm import validate_trace,compare_keepall,validate_recipe,validate_fingerprints
class Tests(unittest.TestCase):
 def test_fingerprint_contract_rejects_invalid_shapes_and_hashes(self):
  meta={'dtype':'torch.bfloat16','shape':[1,8,258,128],'sha256':'a'*64};layers=[{'key':dict(meta),'value':dict(meta)} for _ in range(36)];self.assertTrue(validate_fingerprints([layers],False,258));layers[17]['key']['shape']=[1,8,257,128]
  with self.assertRaises(ValueError):validate_fingerprints([layers],False,258)
  layers[17]['key']['shape']=[1,8,258,128];layers[4]['value']['sha256']='no'
  with self.assertRaises(ValueError):validate_fingerprints([layers],False,258)
 def test_recipe_and_eos_are_anchored(self):
  recipe={'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0};actual={'generation_scalars':{**recipe,'do_sample':False,'max_new_tokens':2048},'eos_ids':[151645]}
  self.assertTrue(validate_recipe(actual,recipe,False,[151645]));actual['generation_scalars']['top_k']=2
  with self.assertRaises(ValueError):validate_recipe(actual,recipe,False,[151645])
  actual['generation_scalars']['top_k']=1;actual['eos_ids']=[1]
  with self.assertRaises(ValueError):validate_recipe(actual,recipe,False,[151645])
 def test_original_nominal_vs_compact_physical_decode(self):
  l=1100;k=256;c=l-1024+k;delta=3
  trace=[{'cache_position':list(range(c)),'rope_delta':[[delta]],'layer_cache_lengths':[],'layer_cache_lengths_after':[c]*36}]
  for i in (1,2):trace.append({'cache_position':[c+i-1],'position_ids':[[[l+i-1+delta]]]*3,'rope_delta':[[delta]],'layer_cache_lengths':[c+i-1]*36,'layer_cache_lengths_after':[c+i]*36})
  self.assertTrue(validate_trace(trace,[1,2,3],l,k,delta));trace[1]['position_ids']=[[[c+delta]]]*3
  with self.assertRaises(ValueError):validate_trace(trace,[1,2,3],l,k,delta)
 def test_keepall_compares_actual_fields(self):
  fields=('task','qid','group_id','image_sha256','message_sha256','forced','processor','main_visual','full_deepstack','prefill_embeds','prefill_mask','prefill_position_ids','rope_delta','injected_mask','injected_deepstack','prefill_logits','output_ids','answer','eos_ids','eos','truncated','max_new_tokens','generation_scalars','decode_trace','cache_fingerprints');native={k:'x' for k in fields};native['path']='stock_native';local={**native,'path':'local_keepall','budget':1024,'spatial_ids':list(range(1024))};self.assertTrue(compare_keepall(native,local));local['prefill_logits']='tamper'
  with self.assertRaises(ValueError):compare_keepall(native,local)
if __name__=='__main__':unittest.main(verbosity=2)
