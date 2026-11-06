import unittest
import torch
from verify import validate_record
from identity_records import tensor_meta
class Tokenizer:
 def __call__(self,*a,**kw):return {'input_ids':torch.tensor([[1,2]])}
 def decode(self,*a,**kw):return 'hello'
class Tests(unittest.TestCase):
 def test_full_record_and_identity_tampering(self):
  k=256;length=1026;keep=torch.tensor([0]+list(range(1,257))+[1025]);pos=torch.arange(length).view(1,1,-1).expand(3,1,-1).tolist();selected=[[[axis[0][i] for i in keep.tolist()]] for axis in pos];mask=[[False]+[True]*k+[False]];meta=tensor_meta(torch.ones(2));qids=tensor_meta(torch.tensor([[1,2]]));retained=tensor_meta(keep);proc={x:meta for x in ('input_ids','attention_mask','pixel_values','image_grid_thw')};scalars={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0};recipe={k:scalars[k] for k in ('top_p','top_k','temperature','repetition_penalty')};row={'task':'x','qid':'1','group_id':'g','image_sha256':'image','message_sha256':'message','message':[{'type':'text','value':'q'}]};ref={'processor':proc,'main_visual':meta,'full_deepstack':[meta]*3,'full_position_ids':pos,'full_delta':[[3]],'original_length':length,'image_start':1,'generation_scalars':scalars,'eos_ids':[3]}
  tensors={'premerger':meta,'averaged_premerger':meta,'visual':meta,'prefill':meta,'prefill_logits':meta,'question_ids':qids,'question_embeddings':meta,'prefill_mask':meta,'prefill_positions':meta,'visual_mask':meta,'retained_indices':retained,**{'ds'+str(i):meta for i in range(3)},**{'injected'+str(i):meta for i in range(3)},**{'processor_'+x:meta for x in proc}}
  r={**row,'path':'local_stacktok','budget':256,'selector_calls':1,'forced':False,'processor':proc,'main_visual':meta,'full_deepstack':[meta]*3,'rope_delta':[[3]],'generation_scalars':scalars,'eos_ids':[3],'spatial_ids':list(range(256)),'greedy_ids':list(range(256)),'prefill_position_ids':selected,'decode_trace':[{'position_ids':selected,'cache_position':list(range(258)),'rope_delta':[[3]],'layer_cache_lengths':[],'layer_cache_lengths_after':[258]*36}],'injected_mask':mask,'injected_deepstack':[meta]*3,'output_ids':[3],'eos':True,'truncated':False,'answer':'hello','question_text':'Question: q','question_ids':qids,'question_embeddings':meta,'retained_indices':retained,'prefill_embeds':meta,'prefill_mask':meta,'prefill_positions':meta,'visual_mask':meta,'prefill_logits':meta,'premerger':meta,'averaged_premerger':meta,'arrays':{'kind':'summary_only','tensors':tensors}}
  event={key:r[key] for key in ('task','qid','group_id','path','forced','output_ids','eos_ids','generation_scalars')};self.assertTrue(validate_record(r,event,row,ref,recipe,Tokenizer()));r['eos']=False
  with self.assertRaises(ValueError):validate_record(r,event,row,ref,recipe,Tokenizer())
if __name__=='__main__':unittest.main(verbosity=2)
