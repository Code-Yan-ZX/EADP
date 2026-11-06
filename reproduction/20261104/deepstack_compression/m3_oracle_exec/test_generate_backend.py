import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

spec=importlib.util.spec_from_file_location('m3_generate_backend_test',Path(__file__).resolve().parent/'generate_backend.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
class Adapter:
    def __init__(self,*args):self.state={'actual_injection_verified':True}
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def verify_calls(self):return True
class Observer:
    def __init__(self,*args):self.trace=[]
    def __enter__(self):return self
    def __exit__(self,*args):pass
class GenerateTests(unittest.TestCase):
    def test_actual_ids_saved_before_postprocessing_failure_and_recipe_checked(self):
        class Model:
            generation_config=SimpleNamespace(eos_token_id=[2])
            def generate(self,*args,**kwargs):return torch.tensor([[5,2]])
        wrapper=SimpleNamespace(visual_token_num=256,pruner=SimpleNamespace(visual_token_num=256),model=Model())
        names=('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty')
        expected=dict(zip(names,[False,2048,None,None,None,None]));events=[]
        def author(w,_row):
            w.model.generate(**expected);raise RuntimeError('postprocessing failed')
        with patch.object(g,'SelectedAdapter',Adapter),patch.object(g,'DecodeObserver',Observer):
            with self.assertRaisesRegex(RuntimeError,'postprocessing'):
                g.generate(wrapper,None,{'split':'DEV'},list(range(256)),list(range(256)),events.append,author,torch,expected,[2],{})
        self.assertEqual(events[0]['token_ids'],[5,2]);self.assertEqual(wrapper.model.generate.__self__,wrapper.model)
        def wrong(w,_row):w.model.generate(max_new_tokens=100,do_sample=False)
        with patch.object(g,'SelectedAdapter',Adapter),patch.object(g,'DecodeObserver',Observer):
            with self.assertRaisesRegex(ValueError,'recipe'):
                g.generate(wrapper,None,{'split':'DEV'},list(range(256)),list(range(256)),events.append,wrong,torch,expected,[2],{})
if __name__=='__main__':unittest.main()
