"""Synthetic tensor test; no model, GT or GPU required."""
import os, unittest
import torch
from contracts import mask_span,compare_selection
from intervention import propose
from selection_observer import SelectionObserver
from types import SimpleNamespace
class Tests(unittest.TestCase):
 def setUp(self):
  assert not torch.cuda.is_initialized()
  ids=torch.tensor([[8,9]+[99]*1024+[10,11]])
  ordered=torch.arange(255,-1,-1,dtype=torch.int64);spatial=ordered.sort().values
  visual=torch.arange(1024*8,dtype=torch.float32).reshape(1024,8)
  token=torch.arange(ids.shape[1]*8,dtype=torch.float32).reshape(1,ids.shape[1],8)
  attn=torch.ones_like(ids);pruned=visual.index_select(0,spatial)
  embed=torch.cat([token[:,:2],pruned.unsqueeze(0),token[:,1026:]],dim=1)
  compact_attn=torch.ones((1,260),dtype=torch.int64)
  pre=dict(inputs_embeds=embed,position_ids=torch.zeros(3,1,260,dtype=torch.int64),attention_mask=compact_attn,cache_position=torch.arange(260),visual_pos_masks=None,deepstack_visual_embeds=None)
  self.rec=dict(ordered_ids=ordered,spatial_ids=spatial,split_sizes=[256],visual_features=visual,pruner_input=visual,pruner_output=pruned,deepstack=[visual+1,visual+2,visual+3],processor_input_ids=ids,processor_mask=attn,processor_pixel_values=torch.ones(1,8),processor_image_grid_thw=torch.tensor([[1,64,64]]),original_token_embeddings=token,generation_kwargs=dict(inputs_embeds=embed,attention_mask=compact_attn,do_sample=False,max_new_tokens=2048,top_p=.001,top_k=1,temperature=.01,repetition_penalty=1.),original_prefill=pre,rope_deltas=torch.tensor([[0]]),image_token_id=99)
  self.pre=pre
 def test_success_changes_only_two_fields_and_spatial_rows(self):
  out,applied,error=propose(torch,self.rec,self.rec,self.pre);self.assertTrue(applied,error)
  self.assertEqual({k for k in out if out[k] is not self.pre[k]},{'visual_pos_masks','deepstack_visual_embeds'})
  self.assertEqual(int(out['visual_pos_masks'].sum()),256)
  for j in range(3):self.assertTrue(torch.equal(out['deepstack_visual_embeds'][j],self.rec['deepstack'][j][:256]))
 def test_selection_and_nonvisual_perturbations_rejected(self):
  for name in ('ordered_ids','processor_input_ids','processor_pixel_values','processor_image_grid_thw','original_token_embeddings','pruner_output'):
   c=dict(self.rec);c[name]=self.rec[name].clone();c[name].reshape(-1)[0]+=1
   out,applied,error=propose(torch,self.rec,c,self.pre);self.assertIs(out,self.pre);self.assertFalse(applied);self.assertTrue(error)
 def test_prefill_and_rope_perturbations_rejected(self):
  changed=dict(self.pre);changed['position_ids']=self.pre['position_ids']+1
  out,applied,error=propose(torch,self.rec,self.rec,changed);self.assertIs(out,changed);self.assertFalse(applied)
  changed=dict(self.rec);changed['rope_deltas']=self.rec['rope_deltas']+1
  out,applied,error=propose(torch,self.rec,changed,self.pre);self.assertIs(out,self.pre);self.assertFalse(applied)
 def test_selection_observer_returns_original_object_and_restores(self):
  out=(torch.tensor([[2,1]]),torch.tensor([2])); calls=[]
  def original(*a,**kw):calls.append((a,kw));return out
  native=SimpleNamespace(_greed_select_impl=original)
  with SelectionObserver(native) as capture:
   actual=native._greed_select_impl('input',token_num=2)
   self.assertIs(actual,out);self.assertEqual(capture.calls,1);self.assertEqual(len(capture.results),1)
  self.assertIs(native._greed_select_impl,original);self.assertEqual(len(calls),1)
 def test_observer_error_does_not_change_original_result(self):
  sentinel=object();original=lambda:sentinel;native=SimpleNamespace(_greed_select_impl=original)
  with SelectionObserver(native) as capture:self.assertIs(native._greed_select_impl(),sentinel)
  self.assertEqual(capture.calls,1);self.assertTrue(capture.errors);self.assertIs(native._greed_select_impl,original)
 def test_signed_zero_bytes_rejected_despite_equal_values(self):
  self.rec['original_token_embeddings'][0,2,0]=0.
  changed=dict(self.rec);changed['original_token_embeddings']=self.rec['original_token_embeddings'].clone()
  changed['original_token_embeddings'][0,2,0]=-0.
  self.assertTrue(torch.equal(changed['original_token_embeddings'],self.rec['original_token_embeddings']))
  out,applied,error=propose(torch,self.rec,changed,self.pre);self.assertIs(out,self.pre);self.assertFalse(applied);self.assertTrue(error)
 def test_invalid_mask_and_ids_rejected(self):
  with self.assertRaises(ValueError):mask_span([99]*1023+[0,99],99)
  with self.assertRaises(ValueError):compare_selection([0]*256,list(range(256)))
if __name__=='__main__':
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 assert not torch.cuda.is_initialized()
 raise SystemExit(0 if result.wasSuccessful() else 1)
