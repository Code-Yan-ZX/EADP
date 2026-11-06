from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import torch
import arm

class FakeModule:
    def register_forward_hook(self,_fn):return SimpleNamespace(remove=lambda:None)

class ArmTests(unittest.TestCase):
    def test_completed_generation_is_journaled_before_later_failure(self):
        original=lambda *a,**kw:torch.tensor([[23,9]])
        model=SimpleNamespace(visual=FakeModule(),lm_head=FakeModule(),generate=original,
                              generation_config=SimpleNamespace(eos_token_id=9))
        w=SimpleNamespace(model=model,pruner=FakeModule(),_processor_inputs=lambda:None)
        class Adapter:
            def __init__(self,*args):self.state={}
            def __enter__(self):return self
            def __exit__(self,*args):return False
        def generate(wrapper,_sample):
            wrapper.model.generate(max_new_tokens=2048,eos_token_id=9)
            raise RuntimeError('after generation')
        captured=[];row={'task':'x','qid':'q','group_id':'g'}
        with patch.object(arm,'ObservedOffAdapter',Adapter),patch.object(arm,'DecodeObserver',lambda _:nullcontext(None)), \
             patch.object(arm,'CacheFingerprints',lambda *_:nullcontext(None)):
            with self.assertRaisesRegex(RuntimeError,'after generation'):
                arm.run_arm(torch,w,None,row,'C-SKIP',generate,captured.append,lambda _:None)
        self.assertEqual(captured[0]['output_ids'],[23,9]);self.assertIs(w.model.generate,original)

if __name__=='__main__':unittest.main()
