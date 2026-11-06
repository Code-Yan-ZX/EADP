import copy, importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from test_stock_capture import StockTests
from stock_capture import run_stock
from pathlib import Path
spec=importlib.util.spec_from_file_location('selected_smoke_contract',Path(__file__).with_name('smoke_contract.py'))
contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
validate_stock,compare_pair,compare_forced=contract.validate_stock,contract.compare_pair,contract.compare_forced


class ContractTests(unittest.TestCase):
 def captured(self):
  wrapper=StockTests().make();arrays={};events=[]
  cfg={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.}
  row={'task':'fake','qid':'1','group_id':'g','split':'DEV','image_sha256':'x','message_sha256':'y'}
  def save(tensors):
   arrays.update({k:v.detach().contiguous().cpu().view(torch.uint8).numpy().copy() for k,v in tensors.items()})
   return {'kind':'full_arrays','path':'fake','sha256':'a'*64}
  rec=run_stock(torch,wrapper,'div',row,lambda w,r:(w.model.generate(**cfg) and ''),events.append,save,cfg)
  return rec,arrays
 def test_actual_arrays_roundtrip_and_corruption(self):
  rec,arrays=self.captured();tok=SimpleNamespace(decode=lambda *a,**k:'')
  self.assertTrue(validate_stock(rec,arrays,tok))
  bad={k:v.copy() for k,v in arrays.items()};bad['visual'][0,0]^=1
  with self.assertRaises(ValueError):validate_stock(rec,bad,tok)
  bad=copy.deepcopy(rec);bad['retained_ids'][1]+=1
  with self.assertRaisesRegex(ValueError,'retained'):validate_stock(bad,arrays,tok)
  bad=copy.deepcopy(rec);bad['main_visual']['shape'][1]+=1
  with self.assertRaisesRegex(ValueError,'shape'):validate_stock(bad,arrays,tok)
 def test_stock_pair_checks_selector_and_not_output(self):
  stock,_=self.captured();adapt=dict(stock,selection_mode='author_div',selector_calls=1,greedy_calls=0,ds_on=True,actual_injection_verified=True,output_ids=[99])
  self.assertTrue(compare_pair(stock,adapt))
  adapt['spatial_ids']=list(range(1,257))
  with self.assertRaisesRegex(ValueError,'spatial_ids'):compare_pair(stock,adapt)
 def test_forced_anchor_requires_same_prefill(self):
  stock,_=self.captured();natural=dict(stock,ordered_ids=stock['spatial_ids'],adjusted_delta=[[768]],injected_mask=[True],injected_deepstack=['a'],selected_deepstack=['a'])
  forced=dict(natural,forced=True)
  self.assertTrue(compare_forced(natural,forced))
  forced['prefill_logits']={'sha256':'wrong'}
  with self.assertRaisesRegex(ValueError,'prefill_logits'):compare_forced(natural,forced)
 def test_adapted_incomplete_natural_generation_is_refused(self):
  record={'method':'div','selection_mode':'author_div','selector_calls':1,'greedy_calls':0,'ordered_ids':[1],
          'spatial_ids':[1],'output_ids':[7],'eos_ids':[2],'forced':False,'eos':False,'truncated':False}
  with patch.object(contract,'validate_budget_record',return_value=True):
   with self.assertRaisesRegex(ValueError,'endpoint'):contract.validate_adapted(record,{}, {},None)

if __name__=='__main__':unittest.main()
