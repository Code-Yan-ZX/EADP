"""CPU sentinel audit of the real NeXT spatial merge and variable-quota selector."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace, MethodType

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'LLaVA'))
import torch
from llava.model.llava_arch import LlavaMetaForCausalLM as Arch
from llava.constants import IMAGE_TOKEN_INDEX


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261009)
    feats=torch.arange(2880,dtype=torch.float32).reshape(5,576,1).expand(-1,-1,2).clone()
    cfg=SimpleNamespace(mm_patch_merge_type='spatial_unpad',image_aspect_ratio='anyres',
                        image_grid_pinpoints=[[672,672]],tokenizer_model_max_length=4096)
    fake=SimpleNamespace(config=cfg,device=torch.device('cpu'))
    fake.get_vision_tower=lambda:SimpleNamespace(num_patches_per_side=24,config=SimpleNamespace(image_size=336))
    fake.get_model=lambda:SimpleNamespace(embed_tokens=lambda ids:-10000*ids.float()[:,None].expand(-1,2))
    cases=[]
    masks={'all':torch.ones(5,576,dtype=torch.bool),
           'random_sparse':torch.rand(5,576)<.055,
           'heterogeneous':torch.zeros(5,576,dtype=torch.bool),
           'corner_only':torch.zeros(5,576,dtype=torch.bool)}
    for i,k in enumerate([1,24,40,90,2]):masks['heterogeneous'][i,torch.randperm(576)[:k]]=True
    masks['corner_only'][:,[0,23,552,575]]=True
    ids=torch.tensor([[1,IMAGE_TOKEN_INDEX,2]])
    for name,mask in masks.items():
        def encode(images,texts=None,split_sizes=None):
            assert split_sizes==[5] and images.shape[0]==5
            return feats,mask
        fake.encode_images=encode
        # Independent oracle: global tokens, then the full 48x48 raster.
        expected=[i for i in range(576) if mask[0,i]]
        for y in range(48):
            for x in range(48):
                crop=1+(y//24)*2+x//24
                local=(y%24)*24+x%24
                if mask[crop,local]:expected.append(crop*576+local)
        for size in [(640,480),(480,640),(672,672)]:
            ret=Arch.prepare_inputs_labels_for_multimodal(fake,ids,torch.arange(3)[None,:],
                torch.ones_like(ids),None,None,torch.zeros(1,5,3,336,336),image_sizes=[size],texts='sentinel')
            actual=ret[4][0,1:-1,0].long().tolist()
            assert actual==expected and ret[-1]==int(mask.sum())
            assert ret[1][0].tolist()==list(range(len(expected)+2))
            cases.append(dict(mask=name,image_size=size,selected=len(actual),aligned=True))
    selection=[]
    for seed in range(8):
        torch.manual_seed(seed)
        importance=torch.rand(5,12)
        visual=torch.rand(5,12,8);visual=visual/visual.norm(dim=-1,keepdim=True)
        similarities=(visual@visual.transpose(1,2)+1)/2
        selector=SimpleNamespace(visual_token_num=2,is_joint=False)
        selector.strategy_importance_allocation=MethodType(Arch.strategy_importance_allocation,selector)
        indices,quotas=Arch.greed_select(selector,importance,similarities,K=5,strategy='importance_based')
        for crop in range(5):
            selected=[]
            for step in range(int(quotas[crop])):
                gains=[]
                for candidate in range(12):
                    if candidate in selected:gains.append(float('-inf'));continue
                    gain=0.
                    for target in range(12):
                        old=max((float(similarities[crop,i,target]) for i in selected),default=0.)
                        gain+=max(0.,float(similarities[crop,candidate,target])-old)*float(importance[crop,target])
                    gains.append(gain)
                selected.append(max(range(12),key=lambda i:gains[i]))
            assert indices[crop,:len(selected)].tolist()==selected
        selection.append(dict(seed=seed,quotas=quotas.tolist(),matches_independent_greedy=True))
    report=dict(cpu_only=True,cuda_initialized=torch.cuda.is_initialized(),spatial_merge_cases=cases,
                selector_cases=selection,source_sha256={str(ROOT/'LLaVA/llava/model/llava_arch.py'):
                hashlib.sha256((ROOT/'LLaVA/llava/model/llava_arch.py').read_bytes()).hexdigest()},
                conclusion='No mask-feature misalignment in the real five-crop merge; variable-quota selections match the independent facility-gain oracle in these cases.')
    out=ROOT/'Qwen_vl/outputs/audit_followup_20261008/next_runtime_repair_20261009/crop_alignment.json'
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(spatial_merge_cases=len(cases),selector_cases=len(selection),passed=True,cuda_initialized=torch.cuda.is_initialized())))


if __name__=='__main__':main()
