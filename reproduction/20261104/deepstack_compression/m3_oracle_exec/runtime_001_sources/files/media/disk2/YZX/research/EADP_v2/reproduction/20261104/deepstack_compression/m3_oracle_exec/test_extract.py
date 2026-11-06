import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
spec=importlib.util.spec_from_file_location('m3_extract_test',Path(__file__).resolve().parent/'extract.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=SimpleNamespace();self.config=SimpleNamespace(image_token_id=511)
        self.model.get_image_features=lambda pix,grid:([pix],[pix+1,pix+2,pix+3])
        self.model.get_rope_index=lambda ids,grid,v,attention_mask:(torch.arange(ids.shape[1]).reshape(1,1,-1).expand(3,1,-1),torch.tensor([[-2]]))
class Pruner:
    visual_token_num=256
    def __init__(self,ns):self.ns=ns
    def __call__(self,image,text,seq,grid):
        ids,_=self.ns._greed_select_impl();return image[ids[0].sort().values],[256]
class Wrapper:
    visual_token_num=256
    def __init__(self):
        self.model=Model();self.ns=SimpleNamespace(_greed_select_impl=lambda:(torch.arange(255,-1,-1).view(1,-1),torch.tensor([256])))
        self.pruner=Pruner(self.ns)
    def _build_messages(self,message,dataset=None):return message
    def _processor_inputs(self,_):
        ids=torch.tensor([[1]+[511]*1024+[2]])
        return {'input_ids':ids,'attention_mask':torch.ones_like(ids),'pixel_values':torch.randn(1024,8),'image_grid_thw':torch.tensor([[1,64,64]])}
    def _get_instruction_sequence_embedding(self,_message,dataset=None):return torch.ones(1,4,8)
class ExtractTests(unittest.TestCase):
    def test_actual_original_pruner_and_restoration(self):
        w=Wrapper();original=w.ns._greed_select_impl;row={'split':'DEV','message':[],'task':'fake'};validated=[]
        result=e.extract(torch,w,w.ns,row,lambda r,x:validated.append((r['task'],x['input_ids'].shape)))
        self.assertEqual(result['greedy_order'],list(range(255,-1,-1)));self.assertEqual(result['start'],1)
        self.assertEqual(len(result['streams']),4);self.assertEqual(result['delta'],-2)
        self.assertEqual(validated,[('fake',torch.Size([1,1026]))]);self.assertIs(w.ns._greed_select_impl,original)
        with self.assertRaisesRegex(ValueError,'DEV'):e.extract(torch,w,w.ns,dict(row,split='FINAL'),lambda *_:None)
if __name__=='__main__':unittest.main()
