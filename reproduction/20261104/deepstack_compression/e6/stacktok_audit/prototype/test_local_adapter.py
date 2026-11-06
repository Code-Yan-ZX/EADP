import importlib.util,sys,unittest
from pathlib import Path
import torch
from local_adapter import group_premerger,user_question,author_selection,p3_ds_prefill
SOURCE=Path(__file__).resolve().parents[2]/'external_sources/stacktok_571bdc6/stacktok/core/selector.py'
def author_module():
 spec=importlib.util.spec_from_file_location('_stacktok_author_selector_tests',SOURCE);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
class TestAdapter(unittest.TestCase):
 def test_group_order_numbered_grid(self):
  rowmajor=torch.arange(16,dtype=torch.float32).reshape(1,4,4,1);packed=rowmajor.reshape(1,2,2,2,2,1).permute(0,1,3,2,4,5).reshape(16,1)
  x=group_premerger(packed,(1,4,4));self.assertTrue(torch.equal(x[:,0],torch.tensor([2.5,4.5,10.5,12.5])))
  with self.assertRaises(ValueError):group_premerger(packed,(1,3,4))
 def test_question_full_text_no_state(self):
  self.assertEqual(user_question([{'type':'image','value':'path'},{'type':'text','value':'<image> What?'},{'type':'text','value':'A. one B. two'}]),'Question: What? A. one B. two')
  self.assertEqual(user_question([{'type':'text','value':'Next'}]),'Question: Next')
  with self.assertRaises(ValueError):user_question([{'type':'image','value':'x'}])
 def test_author_core_identity_ties_and_nontrivial(self):
  mod=author_module();torch.manual_seed(41);v=torch.randn(12,8);pre=torch.randn(12,5);text=torch.randn(3,8)
  for k in (1,4,8):
   a=mod.StackTokSelector(target_vision_tokens=k,swap_mode='auto');b=mod.StackTokSelector(target_vision_tokens=k,swap_mode='auto');actual=author_selection(a,text,v,pre,k);ids,tokens,info=b.stacktok_single(text,v,pre,tv_temp=.01,vv_temp=.2,padding_patch_indices=None)
   self.assertEqual(actual['spatial_ids'],ids);self.assertEqual(actual['greedy_ids'],info['selected_before_sort']);self.assertTrue(torch.equal(actual['tokens'],tokens))
  zero=mod.StackTokSelector(target_vision_tokens=4,swap_mode='auto');a=author_selection(zero,torch.empty(0,8),torch.zeros(12,8),torch.zeros(12,5),4);self.assertEqual(len(a['spatial_ids']),4)
 def test_keepall_and_p3_three_streams(self):
  ids=torch.tensor([[7,99,99,99,99,8]]);text=torch.arange(24,dtype=torch.float32).reshape(1,6,4);visual=torch.arange(16,dtype=torch.float32).reshape(4,4)+100;ds=[visual+100*i for i in range(1,4)];pos=torch.arange(18).reshape(3,1,6);delta=torch.tensor([[3]])
  allout=p3_ds_prefill(ids,torch.ones_like(ids),text,pos,delta,1,visual,ds,[0,1,2,3],99,decode_cache_basis='compact');expected=text.clone();expected[:,1:5]=visual
  self.assertTrue(torch.equal(allout['inputs_embeds'],expected));self.assertTrue(torch.equal(allout['position_ids'],pos));self.assertTrue(torch.equal(allout['rope_delta'],delta));self.assertTrue(all(torch.equal(x,y) for x,y in zip(allout['deepstack_visual_embeds'],ds)))
  out=p3_ds_prefill(ids,torch.ones_like(ids),text,pos,delta,1,visual,ds,[1,3],99,decode_cache_basis='compact');keep=torch.tensor([0,2,4,5]);self.assertTrue(torch.equal(out['retained_indices'],keep));self.assertTrue(torch.equal(out['position_ids'],pos.index_select(2,keep)));self.assertEqual(out['rope_delta'].item(),5);native=p3_ds_prefill(ids,torch.ones_like(ids),text,pos,delta,1,visual,ds,[1,3],99,decode_cache_basis='original');self.assertEqual(native['rope_delta'].item(),3);self.assertTrue(all(torch.equal(x,y[[1,3]]) for x,y in zip(out['deepstack_visual_embeds'],ds)));self.assertEqual(out['visual_pos_masks'].tolist(),[[False,True,True,False]])
  with self.assertRaises(ValueError):p3_ds_prefill(ids,torch.ones_like(ids),text,pos,delta,1,visual,ds,[3,1],99,decode_cache_basis='compact')
  with self.assertRaises(ValueError):p3_ds_prefill(ids,torch.ones_like(ids),text,pos,delta,1,visual,ds[:2],[1,3],99,decode_cache_basis='compact')
 def test_keepall_shortcut_author_core_semantics(self):
  mod=author_module();v=torch.randn(4,8);r=author_selection(mod.StackTokSelector(target_vision_tokens=4),torch.randn(1,8),v,torch.randn(4,5),4)
  self.assertEqual(r['spatial_ids'],[0,1,2,3]);self.assertIs(r['tokens'],v);self.assertTrue(r['keep_all_shortcut']);self.assertIsNone(r['greedy_ids'])
if __name__=='__main__':unittest.main(verbosity=2)
