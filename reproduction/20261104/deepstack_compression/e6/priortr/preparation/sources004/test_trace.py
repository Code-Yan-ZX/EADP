import unittest,torch
from transformers.generation.logits_process import LogitsProcessor,LogitsProcessorList
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from visual_token_pruning.config import VTRConfig
from visual_token_pruning.model.vtr_qwen3_vl import VTRQwen3VLForConditionalGeneration
from trace import Trace
class Forced(LogitsProcessor):
 def __call__(self,input_ids,scores):scores.fill_(-float('inf'));scores[:,7]=0;return scores
class TraceTests(unittest.TestCase):
 def test_tiny_multimodal_real_selection_and_unmodified_logits(self):
  torch.manual_seed(4)
  text={'vocab_size':64,'hidden_size':32,'intermediate_size':64,'num_hidden_layers':4,'num_attention_heads':4,'num_key_value_heads':4,'head_dim':8,'use_cache':True,'rope_parameters':{'rope_type':'default','rope_theta':5000000.0,'mrope_section':[2,1,1]}}
  vision={'depth':3,'hidden_size':32,'intermediate_size':64,'num_heads':4,'in_channels':3,'patch_size':2,'spatial_merge_size':2,'temporal_patch_size':2,'out_hidden_size':32,'num_position_embeddings':64,'deepstack_visual_indexes':[0,1,2]}
  cfg=Qwen3VLConfig(text_config=text,vision_config=vision,image_token_id=60,video_token_id=59,vision_start_token_id=58,vision_end_token_id=57,pad_token_id=0,eos_token_id=63);cfg._attn_implementation='eager';cfg.text_config._attn_implementation='eager';cfg.vision_config._attn_implementation='eager'
  model=VTRQwen3VLForConditionalGeneration(cfg,vtr_config=VTRConfig(enabled=True,strategy='priortr',prune_layer=3,keep_tokens=4)).eval();grid=torch.tensor([[1,4,8]]);pixels=torch.randn((32,24));ids=torch.tensor([[2,58]+[60]*8+[57,3,4,5]])
  def run():return model.generate(input_ids=ids,pixel_values=pixels,image_grid_thw=grid,attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=33,min_new_tokens=33,eos_token_id=63,pad_token_id=0,logits_processor=LogitsProcessorList([Forced()]),use_cache=True)
  reference=[];h=model.lm_head.register_forward_hook(lambda _m,_i,o:reference.append(o.detach().clone()))
  with torch.inference_mode():out1=run()
  h.remove()
  with torch.inference_mode(),Trace(model,forced=True) as tr:out2=run()
  self.assertTrue(torch.equal(out1,out2));self.assertEqual(len(reference),33)
  for i,x in enumerate(reference):self.assertTrue(torch.equal(x,tr.arrays['logits_'+str(i)]))
  self.assertEqual([x['tokens'] for x in tr.layers if x['step']==0],[14,14,14,10]);self.assertEqual([x['rows'] for x in tr.injections],[8,8,4]);self.assertEqual(tr.prunes[0]['range_after'],[2,6]);self.assertEqual(tr.arrays['selected_ids_0'].numel(),4)
  import hashlib,copy
  import smoke_verify
  raw={'input_ids':ids,'attention_mask':torch.ones_like(ids),'pixel_values':pixels,'image_grid_thw':grid}
  for k,v in raw.items():tr.save('processor_'+k,v)
  def meta(t):
   t=t.detach().contiguous().cpu();return {'dtype':str(t.dtype),'shape':list(t.shape),'sha256':hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()}
  tensors={k:meta(v) for k,v in tr.arrays.items()};arrays={k:v.view(torch.uint8).numpy() for k,v in tr.arrays.items()};ref={k:meta(v) for k,v in raw.items()}
  rec={'case':'k256_forced','ids':out2[0,ids.shape[1]:].tolist(),'forced':True,'force_token':7,'eos_ids':[63],'eos':False,'truncated':False,'calls':tr.calls,'layers':tr.layers,'injections':tr.injections,'prunes':tr.prunes,'arrays':{'tensors':tensors},'rope_delta':model.model.rope_deltas.tolist()}
  self.assertEqual(smoke_verify.validate_record(rec,arrays,ref,4,visual_tokens=8,test_k=4),4)
  bad=copy.deepcopy(rec);bad['calls'][1]['cache_before'][0]+=1
  with self.assertRaisesRegex(ValueError,'KV trajectory'):smoke_verify.validate_record(bad,arrays,ref,4,visual_tokens=8,test_k=4)
  bad=copy.deepcopy(rec);bad['calls'][1]['position_ids'][-1][0][0]+=1
  with self.assertRaisesRegex(ValueError,'mRoPE'):smoke_verify.validate_record(bad,arrays,ref,4,visual_tokens=8,test_k=4)
  altered=dict(arrays); altered['prune_after_position_0']=altered['prune_after_position_0'].copy();altered['prune_after_position_0'].flat[0]^=1
  with self.assertRaisesRegex(ValueError,'tensor hash'):smoke_verify.validate_record(rec,altered,ref,4,visual_tokens=8,test_k=4)
if __name__=='__main__':unittest.main()
