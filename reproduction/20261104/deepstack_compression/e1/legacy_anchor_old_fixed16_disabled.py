"""Run fixed-16 E0 anchors through author legacy vs P1-D0, no answers/scores.

This is a bounded model worker: the caller must supervise the real container,
record its actual exit, and run an independent verifier before using results.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from contract import sha, validate_decode_trace
from runner import ROOT, PROJECT, E0, PLAN, OLD, validate_gate, meta, append

PANEL=ROOT/'reproduction/20261028/qwen/full1024_two_path_panel/plan.json'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--execute',action='store_true'); p.add_argument('--e0-proof',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    if not a.execute: raise ValueError('requires explicit --execute')
    raise ValueError('legacy anchor locked: old 16-anchor panel overlaps CONFIRM/FINAL; replace with DEV-only frozen inputs before use')
    e0=validate_gate(a.e0_proof)
    plan=json.loads(PLAN.read_text()); panel=json.loads(PANEL.read_text())['panel']
    if len(panel)!=16: raise ValueError('fixed panel must have exactly 16 anchors')
    for group in ('source_sha256','implementation_sha256'):
        for path,digest in plan[group].items():
            if sha(path)!=digest: raise ValueError('source changed: '+path)
    for row in panel:
        if sha(row['image_path'])!=row['image_sha256']: raise ValueError('anchor image changed')
    weights_file=ROOT/'reproduction/20261028/qwen/qwen_weight_identity.json'
    weights=json.loads(weights_file.read_text())['files']
    if len(weights)!=4: raise ValueError('weight shard count')
    weight_stats={}
    for entry in weights:
        path=Path(entry['path']); stat=path.stat()
        if stat.st_size!=entry['bytes'] or sha(path)!=entry['expected_lfs_sha256']: raise ValueError('weight SHA mismatch')
        weight_stats[str(path)]=[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]
    a.out.mkdir(parents=True,exist_ok=False)
    frozen={'e0_proof_sha256':sha(a.e0_proof),'e0_adapter_sha256':e0['adapter_sha256'],
            'position_adapter_sha256':sha(Path(__file__).with_name('position_adapter.py')),
            'worker_sha256':sha(__file__),'runner_sha256':sha(Path(__file__).with_name('runner.py')),
            'contract_sha256':sha(Path(__file__).with_name('contract.py')),
            'panel_sha256':sha(PANEL),'recipe_sha256':sha(PLAN),'weight_manifest_sha256':sha(weights_file),
            'weight_stats_before':weight_stats,'n':16,'gt_read':False,'scoring':False}
    (a.out/'frozen.json').write_text(json.dumps(frozen,indent=2)+'\n')
    sys.path[:0]=[str(E0),str(OLD),plan['author_root'],str(Path(plan['author_root'])/'VLMEvalKit')]
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList
    from position_adapter import FourArm
    from selection_observer import SelectionObserver
    from model import pruner as eadp_pruner
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP, Qwen3VLChatCDPruner
    if {n:importlib.metadata.version(n) for n in plan['expected_versions']}!=plan['expected_versions']: raise ValueError('environment mismatch')
    if torch.cuda.get_device_name(0)!='NVIDIA A40': raise ValueError('unexpected GPU')
    recipe=dict(plan['recipe']); recipe['visual_token_num']=256
    w=Qwen3VLChatEADP(**recipe)
    if w.model.dtype!=torch.bfloat16 or w.model.config._attn_implementation!='flash_attention_2': raise ValueError('dtype/attention mismatch')
    token=w.processor.tokenizer.encode(' the',add_special_tokens=False)[0]
    class Fixed(LogitsProcessor):
        def __call__(self, input_ids, scores):
            scores.fill_(-float('inf')); scores[:,token]=0
            return scores
    def run(row, adapted, forced):
        state={'llm':[],'logits':[]}; handles=[]
        proc=w._processor_inputs; generate=w.model.generate
        def checked(*args,**kwargs):
            x=proc(*args,**kwargs)
            state['processor']={k:meta(x[k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
            if state['processor']!=row['processor_expected']: raise ValueError('anchor processor mismatch')
            return x
        def head(_module,_args,out):
            if forced or not state['logits']: state['logits'].append(out[:,-1,:].detach().cpu())
        def llm(_module,_args,kwargs):
            cache=kwargs.get('past_key_values')
            state['llm'].append({'position_ids':kwargs['position_ids'].detach().cpu().tolist(),
                'cache_position':kwargs['cache_position'].detach().cpu().tolist(),
                'layer_cache_lengths':[] if cache is None else [cache.get_seq_length(i) for i in range(len(w.model.model.language_model.layers))],
                'rope_delta':w.model.model.rope_deltas.detach().cpu().tolist(),
                'inputs_embeds':meta(kwargs.get('inputs_embeds')),'attention_mask':meta(kwargs.get('attention_mask')),
                'visual_pos_masks':meta(kwargs.get('visual_pos_masks')),'deepstack_visual_embeds':kwargs.get('deepstack_visual_embeds') is not None})
        def capture(*args,**kwargs):
            if forced: kwargs.update(max_new_tokens=33,min_new_tokens=33,eos_token_id=[],logits_processor=LogitsProcessorList([Fixed()]))
            out=generate(*args,**kwargs); state['output_ids']=out.detach().cpu().tolist(); return out
        w._processor_inputs=checked; w.model.generate=capture
        observer=SelectionObserver(eadp_pruner)
        intervention=FourArm(torch,w,'P1-D0',observer) if adapted else contextlib.nullcontext()
        try:
            with observer,intervention:
                handles.append(w.model.model.language_model.register_forward_pre_hook(llm,with_kwargs=True))
                handles.append(w.model.lm_head.register_forward_hook(head))
                with torch.inference_mode():
                    answer=Qwen3VLChatCDPruner.generate_inner_transformers_fixed(w,row['message'],dataset='DocVQA_VAL')
            if observer.errors or observer.calls!=1 or len(observer.results)!=1: raise ValueError('missing actual greedy observer')
            state['ordered_ids']=observer.results[0]['ordered_ids'].detach().cpu().reshape(-1).tolist()
            state['answer']=answer
            return state
        finally:
            for handle in handles: handle.remove()
            w._processor_inputs=proc; w.model.generate=generate
    for row in panel:
        for forced in (False,True):
            t=time.monotonic(); old=run(row,False,forced); new=run(row,True,forced)
            if old['processor']!=new['processor'] or old['ordered_ids']!=new['ordered_ids']:
                raise ValueError('legacy/P1 input or selection differs')
            if old['llm'][0]!=new['llm'][0] or old['output_ids']!=new['output_ids'] or old['answer']!=new['answer']:
                raise ValueError('legacy/P1 prefill, decode output or answer differs')
            if len(old['logits'])!=len(new['logits']) or any(not torch.equal(x,y) for x,y in zip(old['logits'],new['logits'])):
                raise ValueError('legacy/P1 logits differ')
            if forced:
                if len(old['llm'])!=33 or old['llm']!=new['llm']: raise ValueError('legacy/P1 forced decode trace differs')
                length=len(old['llm'][0]['position_ids'][0][0])
                validate_decode_trace(old['llm'],length,old['llm'][0]['rope_delta'][0][0])
            def summary(value):
                return {'processor':value['processor'],'ordered_ids':value['ordered_ids'],
                        'llm':value['llm'],'output_ids':value['output_ids'],'answer':value['answer'],
                        'logits':[meta(z) for z in value['logits']]}
            append(a.out/'comparisons.jsonl',{'index':str(row['index']),'forced':forced,
                'legacy':summary(old),'p1':summary(new),'elapsed_seconds':time.monotonic()-t})
    for path,before in weight_stats.items():
        st=Path(path).stat()
        if [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]!=before: raise ValueError('weight stat changed')
    (a.out/'worker_complete.json').write_text(json.dumps({'n':16,'comparisons':32,'forced_decode_steps':32,'end_unix':time.time()})+'\n')


if __name__=='__main__': main()
