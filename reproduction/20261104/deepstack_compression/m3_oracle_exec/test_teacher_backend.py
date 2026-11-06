import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'m3_oracle'))
import teacher_contract as contract
import oracle_teacher as teacher
import saliency
spec=importlib.util.spec_from_file_location('backend_under_test',HERE/'teacher_backend.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

class LM(torch.nn.Module):
    def forward(self,inputs_embeds,position_ids,attention_mask,visual_pos_masks,deepstack_visual_embeds,use_cache):
        self.positions=position_ids.detach().clone();self.ds=[x.detach().clone() for x in deepstack_visual_embeds]
        # The streams alter the vocabulary distribution rather than a common offset.
        x=inputs_embeds+inputs_embeds.mean(dim=1,keepdim=True)+sum(y.mean(dim=0).view(1,1,-1) for y in deepstack_visual_embeds)
        return SimpleNamespace(last_hidden_state=x)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(512,8);self.model=torch.nn.Module();self.model.language_model=LM();self.lm_head=torch.nn.Linear(8,512,bias=False)
    def get_input_embeddings(self):return self.embedding

class BackendTests(unittest.TestCase):
    def test_real_autograd_and_shared_spatial_gather(self):
        torch.manual_seed(1);model=Model().eval().requires_grad_(False)
        ids=torch.tensor([[1]+[511]*1024+[2]]);positions=torch.arange(1026).view(1,1,-1).expand(3,1,-1)
        streams=[torch.randn(1024,8) for _ in range(4)];target=torch.tensor([[100,200,300]])
        loss,util,evidence=b.utilities(teacher,saliency,model,ids,1,streams,positions,-2,target)
        self.assertTrue(loss>0);self.assertEqual([len(x) for x in util],[1024]*4);self.assertEqual(len(evidence['gradients']),4)
        spatial=list(range(0,512,2));score=b.scorer(teacher,contract,model,ids,1,streams,positions,-2,target)
        self.assertTrue(score(spatial)>0)
        self.assertTrue(torch.equal(model.model.language_model.ds[0],streams[1][spatial]))
        self.assertEqual(model.model.language_model.positions[0,0,1:257].tolist(),[x+1 for x in spatial])
        self.assertEqual(model.model.language_model.positions[0,0,-2:].tolist(),[1024,1025])
        with torch.inference_mode():
            with self.assertRaisesRegex(ValueError,'autograd'):b.utilities(teacher,saliency,model,ids,1,streams,positions,-2,target)
if __name__=='__main__':unittest.main()
