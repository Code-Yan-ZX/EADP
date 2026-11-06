from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from local_imports import local,teacher

C=local('contract');core=local('core')

class FakeLM(torch.nn.Module):
    def forward(self,inputs_embeds,position_ids,attention_mask,visual_pos_masks,deepstack_visual_embeds,use_cache):
        # A differentiable global dependency on all 1024 rows of all streams.
        effect=sum(x.mean() for x in deepstack_visual_embeds)+inputs_embeds.mean()
        return SimpleNamespace(last_hidden_state=inputs_embeds+effect)

class FakeInner(torch.nn.Module):
    def __init__(self,features):
        super().__init__();self.language_model=FakeLM();self.features=features
    def get_image_features(self,_pixels,_grid):return (self.features[:1],self.features[1:])
    def get_rope_index(self,ids,_grid,_videos,attention_mask):
        p=torch.arange(ids.shape[1]).view(1,1,-1).expand(3,1,-1);return p,torch.tensor([[-2]])

class FakeModel(torch.nn.Module):
    def __init__(self,features):
        super().__init__();self.model=FakeInner(features);self.embedding=torch.nn.Embedding(512,4096);self.lm_head=torch.nn.Linear(4096,512,bias=False)
        self.config=SimpleNamespace(image_token_id=511)
        with torch.no_grad():self.embedding.weight.mul_(0.001);self.lm_head.weight.mul_(0.001)
    def get_input_embeddings(self):return self.embedding

class FakeWrapper:
    def __init__(self,model,inputs):self.model=model;self.inputs=inputs
    def _build_messages(self,message,dataset=None):return message
    def _processor_inputs(self,_messages):return self.inputs

class CoreTests(unittest.TestCase):
    def test_real_four_leaf_autograd_and_no_inference_mode(self):
        torch.manual_seed(7);features=[torch.randn(1024,4096)*0.001 for _ in range(4)]
        model=FakeModel(features);ids=torch.tensor([[1]+[511]*1024+[2]])
        inputs={'input_ids':ids,'attention_mask':torch.ones_like(ids),'pixel_values':torch.zeros(1),
                'image_grid_thw':torch.tensor([[1,64,64]])}
        wrapper=FakeWrapper(model,inputs);position,delta=model.model.get_rope_index(ids,inputs['image_grid_thw'],None,inputs['attention_mask'])
        with torch.no_grad():
            prompt=model.embedding(ids).clone();prompt[:,1:1025]=features[0]
            visual_mask=torch.zeros_like(ids,dtype=torch.bool);visual_mask[:,1:1025]=True
            result=model.model.language_model(inputs_embeds=prompt,position_ids=position,attention_mask=inputs['attention_mask'],
                visual_pos_masks=visual_mask,deepstack_visual_embeds=features[1:],use_cache=False)
            native=model.lm_head(result.last_hidden_state[:,-1,:]).to(torch.bfloat16)
        ref={'processor':{k:C.tensor_meta(v) for k,v in inputs.items()},'main_visual':C.tensor_meta(features[0]),
             'full_deepstack':[C.tensor_meta(x) for x in features[1:]],'prefill_position_ids':position.tolist(),
             'rope_delta':delta.tolist(),'prefill_embeds':C.tensor_meta(prompt),'prefill_mask':C.tensor_meta(inputs['attention_mask']),
             'prefill_logits':C.tensor_meta(native),'injected_mask':visual_mask.tolist()}
        length=ids.shape[1];first=length-2
        forced={'decode_trace':[{}, {'position_ids':[[[first]],[[first]],[[first]]]},
                                  {'position_ids':[[[first+1]],[[first+1]],[[first+1]]]}]}
        row={'task':'fake','qid':'1','group_id':'g','message':[]};plan={'target_token_ids':[100,200,300],'native_first_logits_max_abs_tolerance':.125}
        teacher_mod,saliency=teacher();events=[]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ref.npz';np.savez(path,prefill_logits=native.contiguous().view(torch.uint8).numpy());ref['arrays']={'path':str(path)}
            archive_path=Path(tmp)/'output.npz'
            def archive(arrays):
                np.savez(archive_path,**{k:v.detach().contiguous().cpu().view(torch.uint8).numpy() for k,v in arrays.items()})
                return {'path':str(archive_path),'sha256':C.sha(archive_path),'keys':sorted(arrays)}
            with patch.object(core,'compare_logits',wraps=core.compare_logits) as compare:
                result=core.run(torch,wrapper,row,ref,forced,teacher_mod,saliency,plan,events.append,archive)
                self.assertEqual(compare.call_count,2)
            self.assertEqual(len(result['gradients']),4);self.assertTrue(all(x>0 for x in result['gradient_nonzero_counts']))
            self.assertEqual(events,['processing_started','native_prefill_started','backward_started','gradients_saved'])
            from verify import verify_arrays
            verify_arrays(archive_path,result)
            with patch.object(core,'compare_logits',side_effect=[{},ValueError('teacher mismatch')]):
                with self.assertRaisesRegex(ValueError,'teacher mismatch'):
                    core.run(torch,wrapper,row,ref,forced,teacher_mod,saliency,plan,lambda _:None,lambda _:None)
            with torch.inference_mode():
                with self.assertRaisesRegex(ValueError,'inference_mode'):core.run(torch,wrapper,row,ref,forced,teacher_mod,saliency,plan,events.append,lambda _:None)

if __name__=='__main__':unittest.main()
