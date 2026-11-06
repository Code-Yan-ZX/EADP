"""Run 16 DEV-only images through author legacy vs P1-D0, no answers/scores.

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

from contract import sha, read_jsonl, validate_decode_trace, choose_legacy_dev, validate_anchor_exclusion
from runner import ROOT, PROJECT, E0, PLAN, OLD, validate_gate, verified_inputs, meta, append

PANEL=ROOT/'reproduction/20261028/qwen/full1024_two_path_panel/plan.json'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--execute',action='store_true'); p.add_argument('--e0-proof',type=Path,required=True)
    p.add_argument('--manifest-dir',type=Path,required=True); p.add_argument('--inputs',type=Path,required=True); p.add_argument('--input-proof',type=Path,required=True)
    p.add_argument('--limit',type=int,choices=(1,16),required=True)
    p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    if not a.execute: raise ValueError('requires explicit --execute')
    e0=validate_gate(a.e0_proof)
    records=verified_inputs(a.manifest_dir,a.inputs,a.input_proof)
    old_anchors=json.loads(PANEL.read_text())['panel']
    if len(old_anchors)!=16: raise ValueError('old anchor inventory incomplete')
    full=read_jsonl(a.manifest_dir/'full_manifest.jsonl')
    panel,old_groups=choose_legacy_dev(records,full,old_anchors)
    panel=panel[:a.limit]
    plan=json.loads(PLAN.read_text())
    for group in ('source_sha256','implementation_sha256'):
        for path,digest in plan[group].items():
            if sha(path)!=digest: raise ValueError('source changed: '+path)
    for row in panel:
        if sha(row['image_path'])!=row['image_sha256']: raise ValueError('DEV image changed')
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
            'e0_verifier_terminal_sha256':sha(E0/'monitor_verify_array16/terminal.json'),
            'position_adapter_sha256':sha(Path(__file__).with_name('position_adapter.py')),
            'worker_sha256':sha(__file__),'runner_sha256':sha(Path(__file__).with_name('runner.py')),
            'contract_sha256':sha(Path(__file__).with_name('contract.py')),
            'old_anchor_panel_sha256':sha(PANEL),'panel_sha256':sha(a.manifest_dir/'panel_manifest.jsonl'),
            'anchor_exclusion':validate_anchor_exclusion(a.manifest_dir,PANEL),
            'full_manifest_sha256':sha(a.manifest_dir/'full_manifest.jsonl'), 'input_sha256':sha(a.inputs),
            'input_proof_sha256':sha(a.input_proof), 'excluded_old_anchor_groups':sorted(old_groups),
            'selected_dev':[{k:r[k] for k in ('task','qid','group_id')} for r in panel],
            'recipe_sha256':sha(PLAN),'weight_manifest_sha256':sha(weights_file),
            'weight_stats_before':weight_stats,'n':a.limit,'gt_read':False,'scoring':False}
    (a.out/'frozen.json').write_text(json.dumps(frozen,indent=2)+'\n')
    sys.path[:0]=[str(E0),str(OLD),plan['author_root'],str(Path(plan['author_root'])/'VLMEvalKit')]
    import torch
    import numpy as np
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
        state={'llm':[],'logits':[],'arrays':{}}; handles=[]
        proc=w._processor_inputs; generate=w.model.generate
        def save_array(name, tensor):
            if tensor is not None:
                state['arrays'][name]=tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
        def checked(*args,**kwargs):
            x=proc(*args,**kwargs)
            state['processor']={k:meta(x[k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
            for key in ('input_ids','attention_mask','pixel_values','image_grid_thw'): save_array('processor_'+key,x[key])
            return x
        def head(_module,_args,out):
            if forced or not state['logits']:
                z=out[:,-1,:].detach().cpu(); save_array('logits_'+str(len(state['logits'])),z); state['logits'].append(z)
        def visual(_module,_args,out):
            main,deep=out
            save_array('visual_main',main)
            if len(deep)!=3: raise ValueError('expected three DS visual streams')
            for idx,z in enumerate(deep): save_array('visual_ds_'+str(idx),z)
        def pruned(_module,_args,out):
            values,_splits=out
            save_array('pruned_main',values)
        def llm(_module,_args,kwargs):
            cache=kwargs.get('past_key_values')
            if not state['llm']:
                for key in ('inputs_embeds','attention_mask','position_ids','cache_position','visual_pos_masks'):
                    save_array('prefill_'+key,kwargs.get(key))
                for idx,z in enumerate(kwargs.get('deepstack_visual_embeds') or []): save_array('prefill_ds_'+str(idx),z)
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
                handles.append(w.model.visual.register_forward_hook(visual))
                handles.append(w.pruner.register_forward_hook(pruned))
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
            t=time.monotonic()
            def summary(value, label):
                path=a.out/f'{str(row["qid"])}_{"forced" if forced else "natural"}_{label}.npz'
                with path.open('xb') as file:
                    np.savez(file,**value['arrays']); file.flush(); os.fsync(file.fileno())
                return {'processor':value['processor'],'ordered_ids':value['ordered_ids'],
                        'llm':value['llm'],'output_ids':value['output_ids'],'answer':value['answer'],
                        'logits':[meta(z) for z in value['logits']], 'array_path':str(path),
                        'array_sha256':sha(path),'array_keys':sorted(value['arrays'])}
            old=run(row,False,forced)
            old_summary=summary(old,'legacy')
            append(a.out/'arms.jsonl',{'qid':str(row['qid']),'group_id':row['group_id'],'forced':forced,'arm':'legacy','observation':old_summary})
            new=run(row,True,forced)
            new_summary=summary(new,'p1')
            append(a.out/'arms.jsonl',{'qid':str(row['qid']),'group_id':row['group_id'],'forced':forced,'arm':'p1','observation':new_summary})
            if old['processor']!=new['processor'] or old['ordered_ids']!=new['ordered_ids']:
                raise ValueError('legacy/P1 input or selection differs')
            if old['llm'][0]!=new['llm'][0] or old['output_ids']!=new['output_ids'] or old['answer']!=new['answer']:
                raise ValueError('legacy/P1 prefill, decode output or answer differs')
            if len(old['logits'])!=len(new['logits']) or any(not torch.equal(x,y) for x,y in zip(old['logits'],new['logits'])):
                raise ValueError('legacy/P1 logits differ')
            if forced:
                if len(old['llm'])!=33 or old['llm']!=new['llm']: raise ValueError('legacy/P1 forced decode trace differs')
                if old['output_ids']!=[[token]*33]: raise ValueError('forced prefix mismatch')
                length=len(old['llm'][0]['position_ids'][0][0])
                validate_decode_trace(old['llm'],length,old['llm'][0]['rope_delta'][0][0])
            append(a.out/'comparisons.jsonl',{'qid':str(row['qid']),'group_id':row['group_id'],'forced':forced,
                'legacy':old_summary,'p1':new_summary,'elapsed_seconds':time.monotonic()-t})
    for path,before in weight_stats.items():
        st=Path(path).stat()
        if [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]!=before: raise ValueError('weight stat changed')
    (a.out/'worker_complete.json').write_text(json.dumps({'n':a.limit,'comparisons':2*a.limit,'forced_decode_steps':32,'end_unix':time.time()})+'\n')


if __name__=='__main__': main()
