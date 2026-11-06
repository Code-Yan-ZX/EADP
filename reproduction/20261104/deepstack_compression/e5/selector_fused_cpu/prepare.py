"""Reconstruct one accepted selector's numeric inputs on CPU, no model/GT.

The E1-verified pilot reference contains the true main visual features and
actual prefill text embeddings.  Match the wrapper's separately tokenized user
instruction against the original processor IDs and map it into compressed
prefill. CPU arithmetic may differ from the prior GPU; do not assume identity.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
ROOT=Path('/media/disk2/YZX/research/EADP_v2');BASE=ROOT/'reproduction/20261104/deepstack_compression';HERE=BASE/'e5/selector_fused_cpu'
AUTHOR=ROOT/'_reproduction/author_eadp/Qwen_vl'
MODEL=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
sys.path.insert(0,str(AUTHOR))


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()


def main():
    plan=json.loads((HERE/'plan001.json').read_text())
    for p,h in plan['sha256'].items():
        if sha(p)!=h:raise ValueError('source mismatch: '+p)
    import torch
    from transformers import AutoTokenizer,AutoConfig
    from model.pruner import VisualTokenPruner,_sim_visual_impl,_greed_select_impl
    torch.set_num_threads(4)
    data=torch.load(BASE/'e4_c_microbench_pilot/pilot_001/reference.pt',map_location='cpu',weights_only=True)
    actual=data['actual']; rows=[json.loads(x) for x in (BASE/'e1/input_preparation_002/inputs.jsonl').read_text().splitlines()]
    rows=[x for x in rows if x['task']=='DocVQA_VAL' and str(x['qid'])=='16539']
    if len(rows)!=1:raise ValueError('anchor uniqueness')
    row=rows[0]
    if row['message_sha256']!=plan['message_sha256']:raise ValueError('message mismatch')
    user_text=' '.join(x['value'] for x in row['message'] if x['type']=='text').strip()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    config=AutoConfig.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    instruction=tokenizer(user_text,return_tensors='pt',add_special_tokens=True,padding=True,truncation=True,
                          max_length=tokenizer.model_max_length).input_ids[0].tolist()
    ids=actual['processor']['input_ids'][0].tolist()
    matches=[i for i in range(len(ids)-len(instruction)+1) if ids[i:i+len(instruction)]==instruction]
    if len(matches)!=1:raise ValueError('instruction IDs not uniquely present in original prompt')
    visual=[i for i,x in enumerate(ids) if x==config.image_token_id]
    if len(visual)!=1024 or matches[0] <= visual[-1]:raise ValueError('unexpected image/text layout')
    start=matches[0]-768;end=start+len(instruction)
    prefill=actual['prefill']; mask=prefill['visual_pos_masks'][0]
    if start<0 or end>len(mask) or bool(mask[start:end].any()):raise ValueError('text remapping crosses visual span')
    seq=prefill['inputs_embeds'][0,start:end,:].clone().unsqueeze(0)
    image=actual['visual'][0].clone();grid=actual['processor']['image_grid_thw']
    if image.shape!=(1024,4096) or seq.shape[1]!=len(instruction) or grid.tolist()!=[[1,64,64]]:
        raise ValueError('unexpected dimensions')
    pruner=VisualTokenPruner(visual_token_num=256,alpha=.5,beta=2.,visual_dim=4096,spatial_merge_size=2,device='cpu')
    with torch.inference_mode():
        features=image.unsqueeze(0); sim=_sim_visual_impl(features)
        importance=pruner.compute_importance(features,seq.mean(dim=1),seq,32,32)
        selected,quota=_greed_select_impl(importance,sim,256)
    if not torch.isfinite(sim).all() or not torch.isfinite(importance).all():raise ValueError('nonfinite numeric inputs')
    previous=actual['ordered'].to(torch.long)
    values={'similarity':sim,'importance':importance,'cpu_ordered':selected[0], 'gpu_accepted_ordered':previous}
    artifact=HERE/'numeric_inputs001.pt'
    with artifact.open('xb') as f:torch.save(values,f);f.flush();os.fsync(f.fileno())
    cpu=selected[0].tolist();gpu=previous.tolist()
    summary={'status':'completed','scope':'one accepted DEV image, CPU numeric reconstruction; no model weights, labels, scoring or GPU',
             'plan_sha256':sha(HERE/'plan001.json'),'artifact_sha256':sha(artifact),'instruction_tokens':len(instruction),
             'full_sequence_sha256':hashlib.sha256(json.dumps(instruction).encode()).hexdigest(),
             'cpu_vs_gpu_same_order':cpu==gpu,'cpu_vs_gpu_same_set':set(cpu)==set(gpu),
             'first_mismatch':next((i for i,(a,b) in enumerate(zip(cpu,gpu)) if a!=b),None),
             'quota':quota.tolist(),'similarity_shape':list(sim.shape),'importance_shape':list(importance.shape),
             'cuda_initialized':torch.cuda.is_initialized()}
    with (HERE/'result001.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


if __name__=='__main__':main()
