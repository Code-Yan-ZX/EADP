"""Validate user-text selector against all DEV templates and three real processors."""
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
ROOT=Path('/media/disk2/YZX/research/EADP_v2');BASE=ROOT/'reproduction/20261104/deepstack_compression';HERE=BASE/'e6/ziprerank_qtext'
AUTHOR=ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit';MODEL=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
sys.path.insert(0,str(AUTHOR))
from selector import locate_user_text


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    plan=json.loads((HERE/'plan001.json').read_text())
    for p,h in plan['sha256'].items():
        if sha(p)!=h:raise ValueError('source mismatch')
    import torch
    from transformers import AutoProcessor,AutoConfig
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    config=AutoConfig.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    wrapper=object.__new__(Qwen3VLChatFixedRes);wrapper.system_prompt=None;wrapper.min_pixels=wrapper.max_pixels=1048576
    wrapper.total_pixels=None;wrapper.processor=processor;wrapper.model=SimpleNamespace(device='cpu',dtype=torch.bfloat16)
    rows=[json.loads(x) for x in (BASE/'e1/input_preparation_002/inputs.jsonl').read_text().splitlines()]
    scope=json.loads((BASE/'e6/ziprerank_question_scope/result001.json').read_text())
    expected={(x['task'],x['qid']):x for x in scope['records']}; seen=set();records=[]
    for row in rows:
        key=(row['task'],str(row['qid'])); old=expected[key]
        if key in seen or old['message_sha256']!=row['message_sha256']:raise ValueError('identity mismatch')
        seen.add(key);messages=wrapper._build_messages(row['message'],dataset=row['task'])
        rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        raw=processor.tokenizer(rendered,add_special_tokens=False)['input_ids'];i=raw.index(config.image_token_id)
        synthetic=torch.tensor([raw[:i]+[config.image_token_id]*1024+raw[i+1:]])
        selected=locate_user_text(processor,messages,row['message'][1]['value'],synthetic,config.image_token_id)
        values=synthetic[0,selected].tolist();digest=hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()
        if digest!=old['question_ids_sha256'] or len(values)!=old['question_token_count']:raise ValueError('selector diverges from prior scope')
        if old['real_processor_verified']:
            if sha(row['image_path'])!=row['image_sha256']:raise ValueError('image mismatch')
            actual=wrapper._processor_inputs(messages)['input_ids']
            real=locate_user_text(processor,messages,row['message'][1]['value'],actual,config.image_token_id)
            if not torch.equal(real,selected) or not torch.equal(actual,synthetic):raise ValueError('real processor mismatch')
        records.append({'task':key[0],'qid':key[1],'user_token_count':len(values),'user_token_sha256':digest,
                        'real_processor_verified':old['real_processor_verified']})
    if len(seen)!=384 or sum(x['real_processor_verified'] for x in records)!=3 or torch.cuda.is_initialized():
        raise ValueError('coverage/CUDA mismatch')
    result={'status':'passed','scope':'384 tokenizer templates; 3 actual image processors; original full user text includes instructions/options; no weights, labels, scoring or GPU',
            'plan_sha256':sha(HERE/'plan001.json'),'records':records}
    with (HERE/'result001.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


if __name__=='__main__':main()
