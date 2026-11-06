import types,unittest
import torch
from model_adapter import LocalStackTokAdapter
class Merger(torch.nn.Module):
 def forward(self,x):return x.reshape(-1,4,x.shape[-1]).mean(1)
class Vision(torch.nn.Module):
 def __init__(self):super().__init__();self.merger=Merger();self.spatial_merge_size=2
 def forward(self,x):
  y=self.merger(x);return y,[y+1,y+2,y+3]
class Language(torch.nn.Module):
 def forward(self,**kw):return kw
class Tokenizer:
 bos_token_id=1
 def __call__(self,words,**kw):return {'input_ids':torch.tensor([[1,2]])}
class Selector:
 target_vision_tokens=256
 def stacktok_single(self,text,projected,premerge,**kw):
  ids=list(range(256));return ids,projected[ids],{'selected_before_sort':ids}
class Cache:
 def __init__(self,length,bad_layer=None):self.length=length;self.bad_layer=bad_layer
 def get_seq_length(self,i):return self.length+(i==self.bad_layer)
class Wrapper:
 def __init__(self):
  self.vision=Vision();self.lm=Language();self.table=torch.nn.Embedding(8,2);self.core=types.SimpleNamespace(visual=self.vision,language_model=self.lm,rope_deltas=torch.tensor([[3]]));self.model=types.SimpleNamespace(model=self.core,config=types.SimpleNamespace(image_token_id=99),get_input_embeddings=lambda:self.table);self.processor=types.SimpleNamespace(tokenizer=Tokenizer())
 def _processor_inputs(self,*args,**kw):
  return {'input_ids':torch.tensor([[7]+[99]*1024+[6]]),'attention_mask':torch.ones((1,1026),dtype=torch.long),'pixel_values':torch.zeros(4096,2),'image_grid_thw':torch.tensor([[1,64,64]])}
class Tests(unittest.TestCase):
 def run_path(self,k):
  w=Wrapper();sel=Selector();old=(tuple(w.vision._forward_hooks),tuple(w.vision.merger._forward_pre_hooks),tuple(w.lm._forward_pre_hooks));row={'message':[{'type':'text','value':'question'}]}
  with LocalStackTokAdapter(w,row,sel,k) as adapter:
   inp=w._processor_inputs(None);main,ds=w.vision(torch.arange(8192,dtype=torch.float32).reshape(4096,2));emb=torch.zeros((1,1026,2));emb[:,1:1025]=main;pos=torch.arange(1026).view(1,1,-1).expand(3,1,-1);kwargs={'inputs_embeds':emb,'attention_mask':inp['attention_mask'],'position_ids':pos,'visual_pos_masks':inp['input_ids']==99,'deepstack_visual_embeds':ds,'cache_position':torch.arange(1026)}
   result=w.lm(**kwargs);keep=1026-(1024-k);self.assertEqual(result['inputs_embeds'].shape[1],keep);self.assertEqual(result['cache_position'].tolist(),list(range(keep)));self.assertEqual([len(x) for x in result['deepstack_visual_embeds']],[k]*3)
   if k==1024:self.assertIs(result['inputs_embeds'],emb);self.assertIs(result['position_ids'],pos)
   dec=w.lm(inputs_embeds=torch.zeros((1,1,2)),attention_mask=torch.ones((1,1027),dtype=torch.long),position_ids=torch.tensor([[[1029]]]*3),cache_position=torch.tensor([1026]),past_key_values=Cache(keep))
   self.assertEqual(dec['attention_mask'].shape[1],keep+1);self.assertEqual(dec['cache_position'].item(),keep)
  self.assertEqual(old,(tuple(w.vision._forward_hooks),tuple(w.vision.merger._forward_pre_hooks),tuple(w.lm._forward_pre_hooks)));self.assertNotIn('_processor_inputs',w.__dict__)
 def test_keepall_native_identity(self):self.run_path(1024)
 def test_k256_all_streams_and_decode_cache(self):self.run_path(256)
 def test_decode_rejects_skips_positions_mask_and_kv(self):
  for corruption in ('skip','position','prefix','suffix','kv'):
   with self.subTest(corruption=corruption):
    w=Wrapper()
    with LocalStackTokAdapter(w,{'message':[{'type':'text','value':'q'}]},Selector(),256):
     inp=w._processor_inputs(None);main,ds=w.vision(torch.arange(8192,dtype=torch.float32).reshape(4096,2));emb=torch.zeros((1,1026,2));emb[:,1:1025]=main
     w.lm(inputs_embeds=emb,attention_mask=inp['attention_mask'],position_ids=torch.arange(1026).view(1,1,-1).expand(3,1,-1),visual_pos_masks=inp['input_ids']==99,deepstack_visual_embeds=ds,cache_position=torch.arange(1026))
     dec=dict(inputs_embeds=torch.zeros((1,1,2)),attention_mask=torch.ones((1,1027),dtype=torch.long),position_ids=torch.full((3,1,1),1029),cache_position=torch.tensor([1026]),past_key_values=Cache(258))
     if corruption=='skip':dec['cache_position']+=1
     if corruption=='position':dec['position_ids'][2,0,0]+=1
     if corruption=='prefix':dec['attention_mask'][0,3]=0
     if corruption=='suffix':dec['attention_mask'][0,-1]=0
     if corruption=='kv':dec['past_key_values']=Cache(258,17)
     with self.assertRaises(ValueError):w.lm(**dec)
 def test_exception_unwinds_hooks(self):
  w=Wrapper();old=tuple(w.lm._forward_pre_hooks)
  with self.assertRaisesRegex(ValueError,'missing actual'):
   with LocalStackTokAdapter(w,{'message':[{'type':'text','value':'q'}]},Selector(),256):w.lm(inputs_embeds=None)
  self.assertEqual(old,tuple(w.lm._forward_pre_hooks));self.assertNotIn('_processor_inputs',w.__dict__)
if __name__=='__main__':unittest.main(verbosity=2)
