import sys,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path[:0]=[str(HERE),str(ROOT/'e4')]
from test_off_adapter import FakeWrapper,FakePruner
from stock_capture import run_stock
class Cache:
 def __init__(self,n):self.n=n
 def get_seq_length(self,i):return self.n
class Language(torch.nn.Module):
 def __init__(self):super().__init__();self.layers=[None]*36
 def forward(self,inputs_embeds,attention_mask,position_ids,cache_position,visual_pos_masks=None,deepstack_visual_embeds=None,past_key_values=None):
  return SimpleNamespace(last_hidden_state=inputs_embeds,past_key_values=Cache(int(cache_position[-1])+1))
class StockTests(unittest.TestCase):
 def make(self):
  w=FakeWrapper();w.visual_token_num=256;w.pruner=type('DivPruner',(FakePruner,),{})(w.namespace);w.pruner.visual_token_num=256
  w.model.model.language_model=Language();w.model.lm_head=torch.nn.Linear(4,16);w.model.generation_config=SimpleNamespace(eos_token_id=[2])
  def processor(_message):
   ids=torch.tensor([[1,2]+[999]*1024+[3,4,5]])
   return {'input_ids':ids,'attention_mask':torch.ones_like(ids),'pixel_values':torch.arange(4096,dtype=torch.float32).reshape(1024,4),'image_grid_thw':torch.tensor([[1,64,64]])}
  w._processor_inputs=processor
  def generate(**kwargs):
   inp=w._processor_inputs([]);visual,_=w.model.visual(inp['pixel_values']);pruned,_=w.pruner(visual)
   embeds=w.model.get_input_embeddings()(inp['input_ids']);embeds=torch.cat((embeds[:,:2],pruned.unsqueeze(0),embeds[:,1026:]),dim=1)
   n=embeds.shape[1];w.model.model.rope_deltas=torch.tensor([[0]])
   result=w.model.model.language_model(inputs_embeds=embeds,attention_mask=torch.ones((1,n),dtype=torch.long),
       position_ids=torch.arange(n).view(1,1,n).expand(3,1,n),cache_position=torch.arange(n))
   w.model.lm_head(result.last_hidden_state);return torch.tensor([[2]])
  w.model.generate=generate
  return w
 def test_real_stock_hooks_preserve_original_selection_and_eos(self):
  w=self.make();events=[];cfg={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.}
  row={'task':'fake','qid':'1','group_id':'g','split':'DEV','image_sha256':'x','message_sha256':'y'}
  def call(model,_row):model.model.generate(**cfg);return ''
  rec=run_stock(torch,w,'div',row,call,events.append,lambda _: {'kind':'full_arrays','path':'fake','sha256':'a'*64},cfg)
  self.assertEqual(rec['spatial_ids'],list(range(256)));self.assertEqual(rec['output_ids'],[2]);self.assertTrue(rec['eos'])
  self.assertEqual(events[0]['path'],'author_stock');self.assertEqual(len(rec['full_deepstack']),3)
  self.assertFalse(w.model.visual._forward_hooks);self.assertFalse(w.model.model.language_model._forward_pre_hooks)
 def test_wrong_recipe_rejected_before_generation(self):
  w=self.make();events=[];cfg={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.}
  row={'task':'fake','qid':'1','group_id':'g','split':'DEV','image_sha256':'x','message_sha256':'y'}
  def call(model,_row):model.model.generate(max_new_tokens=1);return ''
  with self.assertRaisesRegex(ValueError,'recipe'):run_stock(torch,w,'div',row,call,events.append,lambda _:{},cfg)
  self.assertEqual(events,[])
 def test_forced_33_tokens_capture_all_actual_decode_calls(self):
  w=self.make();events=[];cfg={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.}
  row={'task':'fake','qid':'1','group_id':'g','split':'DEV','image_sha256':'x','message_sha256':'y'}
  def forced_generate(**kw):
   self.assertEqual(kw['max_new_tokens'],33);self.assertEqual(kw['min_new_tokens'],33);self.assertEqual(kw['eos_token_id'],[])
   scores=kw['logits_processor'](torch.tensor([[1]]),torch.ones((1,16)))
   self.assertEqual(int(scores.argmax()),7)
   inp=w._processor_inputs([]);visual,_=w.model.visual(inp['pixel_values']);pruned,_=w.pruner(visual)
   embeds=w.model.get_input_embeddings()(inp['input_ids']);embeds=torch.cat((embeds[:,:2],pruned.unsqueeze(0),embeds[:,1026:]),dim=1)
   n=embeds.shape[1];w.model.model.rope_deltas=torch.tensor([[0]])
   for step in range(33):
    cp=torch.arange(n) if step==0 else torch.tensor([n+step-1]);pos=cp.view(1,1,-1).expand(3,1,-1)
    result=w.model.model.language_model(inputs_embeds=embeds if step==0 else embeds[:,:1],attention_mask=torch.ones((1,n),dtype=torch.long),
      position_ids=pos,cache_position=cp,past_key_values=None if step==0 else Cache(n+step-1))
    w.model.lm_head(result.last_hidden_state)
   return torch.tensor([[7]*33])
  w.model.generate=forced_generate
  def call(model,_row):model.model.generate(**cfg);return ''
  rec=run_stock(torch,w,'div',row,call,events.append,lambda _:{'kind':'full_arrays','path':'fake','sha256':'a'*64},cfg,7)
  self.assertTrue(rec['forced']);self.assertEqual(rec['output_ids'],[7]*33);self.assertEqual(len(rec['decode_trace']),33)
  self.assertEqual(len(rec['decode_trace'][-1]['layer_cache_lengths_after']),36)
if __name__=='__main__':unittest.main()
