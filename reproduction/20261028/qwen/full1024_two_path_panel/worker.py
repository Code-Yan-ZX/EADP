"""Candidate native two-path worker. No launcher; execute only via reviewed controller."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import common as c


def execute(note_path):
    plan=c.read(c.PLAN); c.plan_checks(plan,'container')
    note=c.read(note_path)
    assert note['authorization']=='explicit_root_gpu_allocation' and note['run_id']==c.RUN_ID
    assert note['plan_sha256']==c.sha(c.PLAN) and note['GPU_device']==0 and note['single_attempt']
    assert Path(sys.executable).resolve()==Path(c.read(c.PREV/'run_plan.json')['python']).resolve()
    out=c.HERE/'execution';out.mkdir(exist_ok=False)
    import numpy as np
    import torch
    torch.set_num_threads(4)
    old=c.read(c.PREV/'run_plan.json')
    sys.path.insert(0,old['author_root']);sys.path.insert(0,str(Path(old['author_root'])/'VLMEvalKit'))
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes,Qwen3VLChatCDPruner,Qwen3VLChatEADP
    assert {name:importlib.metadata.version(name) for name in old['expected_versions']}==old['expected_versions']
    recipe=dict(plan['base_recipe']);recipe['visual_token_num']=1024
    wrapper=Qwen3VLChatEADP(**recipe)
    assert wrapper.visual_token_num==wrapper.pruner.visual_token_num==1024
    assert wrapper.model.config._attn_implementation=='flash_attention_2'
    assert wrapper.model.visual.config._attn_implementation=='flash_attention_2'
    assert wrapper.model.dtype==wrapper.model.visual.dtype==torch.bfloat16
    assert {str(p.device) for p in wrapper.model.parameters()}=={'cuda:0'}
    assert torch.cuda.get_device_name(0)=='NVIDIA A40'
    source_names=['vlmeval.vlm.qwen3_vl.model_fixed_res','vlmeval.vlm.qwen3_vl.model','model.pruner','transformers.models.qwen3_vl.modeling_qwen3_vl']
    bound_sources={name:str(Path(sys.modules[name].__file__).resolve()) for name in source_names}
    for path in bound_sources.values(): assert c.sha(path)==old['source_sha256'][path]
    c.save(out/'runtime.json',dict(source_modules=bound_sources,model_dtype=str(wrapper.model.dtype),
        visual_dtype=str(wrapper.model.visual.dtype),backend=wrapper.model.config._attn_implementation,
        GPU=torch.cuda.get_device_name(0),actual_recipe=recipe,model_instances=1,plan_sha256=c.sha(c.PLAN)))
    records=[json.loads(s) for s in Path(plan['source_panel_records']).read_text().splitlines()]
    def arrays_and_meta(values):
        arrays={};metadata={}
        for name,t in values.items():
            if t is None: continue
            x=t.detach().cpu().contiguous()
            a=x.view(torch.uint16).numpy() if x.dtype==torch.bfloat16 else x.numpy()
            arrays[name]=a
            metadata[name]=dict(dtype=str(x.dtype),shape=list(x.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
        return arrays,metadata
    def generate(item,arm,index):
        from observer import Recorder,passthrough
        rec=Recorder();handles=[]
        def snapshot(value):
            if isinstance(value,torch.Tensor): return value.detach().clone()
            if isinstance(value,(list,tuple)): return type(value)(snapshot(x) for x in value)
            if isinstance(value,dict): return {k:snapshot(v) for k,v in value.items()}
            return value
        def vision_hook(module,args,output):rec.record('vision',lambda:snapshot(output))
        def pruner_hook(module,args,output):rec.record('pruner',lambda:(snapshot(args),snapshot(output)))
        def llm_prehook(module,args,kwargs):
            # Only prefill is retained to bound GPU memory; count every forward.
            rec.record('llm_forward',lambda:None)
            if rec.count('llm_forward')==1:
                rec.record('prefill',lambda:{k:snapshot(kwargs.get(k)) for k in ['inputs_embeds','position_ids','attention_mask','cache_position','visual_pos_masks','deepstack_visual_embeds']})
                rec.record('rope_deltas',lambda:snapshot(wrapper.model.model.rope_deltas))
        handles.append(wrapper.model.visual.register_forward_hook(vision_hook))
        handles.append(wrapper.pruner.register_forward_hook(pruner_hook))
        handles.append(wrapper.model.model.language_model.register_forward_pre_hook(llm_prehook,with_kwargs=True))
        old_generate=wrapper.model.generate;old_processor=wrapper._processor_inputs
        def observed_generate(*args,**kwargs):
            # Snapshot actual call inputs before the native call; return its result unchanged.
            rec.record('generate_kwargs',lambda:snapshot(kwargs))
            result=old_generate(*args,**kwargs)
            rec.record('generate_result',lambda:result)
            return result
        def observed_processor(*args,**kwargs):
            result=old_processor(*args,**kwargs)
            rec.record('processor_inputs',lambda:{k:snapshot(v) for k,v in result.items()})
            return result
        wrapper.model.generate=observed_generate;wrapper._processor_inputs=observed_processor
        response=None;failure=None
        try:
            method=Qwen3VLChatFixedRes.generate_inner_transformers_fixed if arm=='plain' else Qwen3VLChatCDPruner.generate_inner_transformers_fixed
            response=method(wrapper,item['message'],dataset='DocVQA_VAL')
        except BaseException as error:
            failure=repr(error)
        finally:
            wrapper.model.generate=old_generate;wrapper._processor_inputs=old_processor
            for handle in handles:handle.remove()
        # Preserve returned response and failure before interpreting observations.
        target=out/f'{index:02d}_{arm}.json'
        saved=dict(position=item['position'],index=item['index'],arm=arm,response=response,native_failure=failure,
            observation_errors=rec.errors,counts={k:rec.count(k) for k in rec.events},status='raw_capture')
        c.save(target,saved)
        values={}; inputs=rec.first('processor_inputs',{});prefill=rec.first('prefill',{})
        visual=rec.first('vision');pruner=rec.first('pruner');ids=rec.first('generate_result');kwargs=rec.first('generate_kwargs',{})
        def collect(name,func):
            try:
                value=func()
                if isinstance(value,torch.Tensor):values[name]=value
            except Exception as error:rec.errors.append(dict(event='collect:'+name,error=repr(error)))
        if isinstance(visual,(tuple,list)) and len(visual)==2:
            collect('visual_features',lambda:visual[0])
            if isinstance(visual[1],(tuple,list)):
                for j,v in enumerate(visual[1]):collect('visual_deepstack_'+str(j),lambda v=v:v)
        for key in ('inputs_embeds','position_ids','attention_mask','cache_position','visual_pos_masks'):
            collect('prefill_'+key,lambda key=key:prefill.get(key))
        for j,v in enumerate(prefill.get('deepstack_visual_embeds') or []):collect('prefill_deepstack_'+str(j),lambda v=v:v)
        collect('rope_deltas',lambda:rec.first('rope_deltas'))
        for key,v in inputs.items():collect('processor_'+key,lambda v=v:v)
        collect('output_ids',lambda:ids)
        collect('output_suffix_ids',lambda:ids[:,inputs['input_ids'].shape[1]:] if arm=='plain' else ids)
        if pruner is not None:
            collect('pruner_input',lambda:pruner[0][0]);collect('pruner_output',lambda:pruner[1][0])
        for key,value in kwargs.items():collect('generate_'+key,lambda value=value:value)
        arrays={};metadata={}
        for name,value in values.items():
            try:
                part,meta=arrays_and_meta({name:value});arrays.update(part);metadata.update(meta)
            except Exception as error:rec.errors.append(dict(event='serialize:'+name,error=repr(error)))
        path=out/f'{index:02d}_{arm}.npz';np.savez_compressed(path,**arrays)
        splits=pruner[1][1] if pruner is not None else None
        scalar_kwargs={key:value for key,value in kwargs.items() if value is None or type(value) in (str,int,float,bool)}
        saved.update(archive=str(path),archive_sha256=c.sha(path),arrays=metadata,observation_errors=rec.errors,
            generate_kwargs_keys=sorted(kwargs),generate_non_tensor_kwargs=scalar_kwargs,
            pruner_splits=splits,generated_id_semantics='Full input IDs plus generated suffix' if arm=='plain' else 'Generated IDs returned for inputs_embeds call',
            prefill_deepstack_present=prefill.get('deepstack_visual_embeds') is not None,
            prefill_visual_mask_present=prefill.get('visual_pos_masks') is not None,
            hook_snapshots_made_at_actual_forward=True,CPU_serialization_after_generation=True,GT_or_score_used=False)
        saved['status']='raw_preserved';c.save(target,saved)
        from observer import validate_capture
        validate_capture(arm,saved['counts'],rec.errors,failure,response,metadata,splits,kwargs)
        return saved
    summary=[]
    for i,(item,oldrec) in enumerate(zip(plan['panel'],records)):
        assert c.sha(item['image_path'])==item['image_sha256']
        a=generate(item,'plain',i);b=generate(item,'eadp1024',i)
        expected=item['processor_expected']
        for r in (a,b):
            assert {k:r['arrays']['processor_'+k] for k in expected}==expected
        same={key:a['arrays'][key]==b['arrays'][key] for key in ('visual_features','prefill_inputs_embeds','prefill_position_ids','prefill_attention_mask')}
        same['old_feature_anchor']=a['arrays']['visual_features']==oldrec['tensors']['visual_features_BF16_bits']
        same['eadp_no_prune_feature_identity']=b['arrays']['pruner_input']==b['arrays']['pruner_output']==b['arrays']['visual_features']
        same['output_suffix_ids_equal']=a['arrays']['output_suffix_ids']==b['arrays']['output_suffix_ids']
        same['response_equal']=a['response']==b['response']
        summary.append(dict(position=item['position'],index=item['index'],comparisons=same,
            plain_deepstack=a['prefill_deepstack_present'],eadp_deepstack=b['prefill_deepstack_present'],
            plain_visual_mask=a['prefill_visual_mask_present'],eadp_visual_mask=b['prefill_visual_mask_present']))
        c.save(out/'partial_summary.json',summary)
    # Recheck frozen source and payload identities after all native generations.
    for group in ('source_sha256','implementation_sha256','reference_sha256'):
        for path,digest in old[group].items(): assert c.sha(path)==digest,path
    from namespace_guards import check_model_stats
    from diagnose import stat
    check_model_stats(old,stat,'container')
    for path,digest in plan['files'].items(): assert c.sha(path)==digest,path
    for item in plan['panel']: assert c.sha(item['image_path'])==item['image_sha256']
    # Only structural metadata and hashes; keep each actual array and response.
    c.save(out/'summary.json',dict(success=True,n_questions=16,n_native_generations=32,results=summary,
        run_plan_sha256=c.sha(c.PLAN),model_instances=1,benchmark_result=False,GT_read=False,scoring_started=False,
        source_inputs_checked=True,source_feature_anchor_matches=sum(x['comparisons']['old_feature_anchor'] for x in summary),
        limits=plan['limits']))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--allocation-note',type=Path)
    args=ap.parse_args();assert args.execute and args.allocation_note
    try:execute(args.allocation_note)
    except BaseException as error:
        c.save(c.HERE/'worker_failure.json',dict(error=repr(error),status='terminal_failure_no_retry',partial_artifacts_retained=True))
        raise
