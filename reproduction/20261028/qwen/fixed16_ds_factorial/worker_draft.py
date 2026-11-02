"""Research worker DRAFT: not a launcher; root review/allocation still required."""
import argparse,hashlib,importlib.metadata,json,sys
from pathlib import Path
import common as c
from intervention import change


def execute(note_path):
    plan=c.read(c.PLAN);c.check(plan,'container');note=c.read(note_path)
    assert note['authorization']=='explicit_root_gpu_allocation' and note['run_id']==c.RUN_ID
    assert note['plan_sha256']==c.sha(c.PLAN) and note['GPU_device']==0 and note['single_attempt']
    old=c.read(c.OLD/'run_plan.json');prev=c.read(c.PREV/'plan.json')
    assert Path(sys.executable).resolve()==Path(old['python']).resolve()
    out=c.HERE/'execution';out.mkdir(exist_ok=False)
    import numpy as np
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0,old['author_root']);sys.path.insert(0,str(Path(old['author_root'])/'VLMEvalKit'))
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes,Qwen3VLChatCDPruner,Qwen3VLChatEADP
    assert {n:importlib.metadata.version(n) for n in old['expected_versions']}==old['expected_versions']
    recipe=dict(old['recipe']);recipe['visual_token_num']=1024
    wrapper=Qwen3VLChatEADP(**recipe)
    assert wrapper.pruner.visual_token_num==1024 and wrapper.model.dtype==torch.bfloat16 and wrapper.post_process is False
    assert wrapper.model.config._attn_implementation==wrapper.model.visual.config._attn_implementation=='flash_attention_2'
    assert {str(x.device) for x in wrapper.model.parameters()}=={'cuda:0'} and torch.cuda.get_device_name(0)=='NVIDIA A40'
    source_modules=['vlmeval.vlm.qwen3_vl.model_fixed_res','vlmeval.vlm.qwen3_vl.model','model.pruner','transformers.models.qwen3_vl.modeling_qwen3_vl']
    paths={n:str(Path(sys.modules[n].__file__).resolve()) for n in source_modules}
    for path in paths.values():assert c.sha(path)==old['source_sha256'][path]
    c.save(out/'runtime.json',dict(recipe=recipe,model_instances=1,backend='flash_attention_2',model_dtype=str(wrapper.model.dtype),
        source_modules=paths,post_process=wrapper.post_process,plan_sha256=c.sha(c.PLAN),GT_read=False,scoring_started=False))
    def snapshot(v):
        if isinstance(v,torch.Tensor):return v.detach().clone()
        if isinstance(v,(list,tuple)):return type(v)(snapshot(x) for x in v)
        if isinstance(v,dict):return {k:snapshot(x) for k,x in v.items()}
        return v
    def safe(rec,key,fn):
        try:rec[key]=fn()
        except Exception as error:rec['errors'].append(dict(event=key,error=repr(error)))
    def generate(index,item,arm):
        rec=dict(errors=[],vision_calls=0,llm_calls=0,processor_calls=0,generate_calls=0,
                 intervention_applied=False,intervention_error=None,intervention_attempts=0,continuation=[])
        def vision_hook(module,args,result):
            rec['vision_calls']+=1
            # Retain the actual tensors for potential injection and snapshot.
            rec['raw_vision']=result
            safe(rec,'vision',lambda:snapshot(result))
        def llm_injection_hook(module,args,kwargs):
            rec['llm_calls']+=1
            if rec['llm_calls']!=1:return None
            effective=kwargs
            if arm in ('plain_ds_off','eadp1024_ds_on'):
                raw=rec.get('raw_vision')
                deep=raw[1] if isinstance(raw,(tuple,list)) and len(raw)==2 else None
                rec['intervention_attempts']+=1
                effective,applied,error=change(arm,kwargs,rec.get('image_mask'),deep)
                rec['intervention_applied']=applied;rec['intervention_error']=error
            fields=('inputs_embeds','position_ids','attention_mask','cache_position','visual_pos_masks','deepstack_visual_embeds')
            safe(rec,'prefill_original',lambda:{k:snapshot(kwargs.get(k)) for k in fields})
            safe(rec,'prefill_proposed',lambda:{k:snapshot(effective.get(k)) for k in fields})
            if effective is not kwargs:return args,effective
        def llm_final_observer(module,args,kwargs):
            # Runs after the injection hook and returns None, without changes.
            # These are last registered PyTorch pre-hook values, not internal attention tensors.
            if rec['llm_calls']==1:
                fields=('inputs_embeds','position_ids','attention_mask','cache_position','visual_pos_masks','deepstack_visual_embeds')
                safe(rec,'prefill',lambda:{k:snapshot(kwargs.get(k)) for k in fields})
                safe(rec,'rope_deltas',lambda:snapshot(wrapper.model.model.rope_deltas))
            else:
                try:
                    rec['continuation'].append(dict(position_ids=snapshot(kwargs.get('position_ids')),
                        cache_position=snapshot(kwargs.get('cache_position')),
                        rope_deltas=snapshot(wrapper.model.model.rope_deltas),
                        visual_mask_present=kwargs.get('visual_pos_masks') is not None,
                        deepstack_present=kwargs.get('deepstack_visual_embeds') is not None))
                except Exception as error:rec['errors'].append(dict(event='continuation',error=repr(error)))
        # No pre-existing per-module/global forward hooks: the observer below
        # is the final registered PyTorch pre-hook for each call in this run.
        assert not wrapper.model.model.language_model._forward_pre_hooks
        assert not torch.nn.modules.module._global_forward_pre_hooks
        hooks=[wrapper.model.visual.register_forward_hook(vision_hook),
               wrapper.model.model.language_model.register_forward_pre_hook(llm_injection_hook,with_kwargs=True),
               wrapper.model.model.language_model.register_forward_pre_hook(llm_final_observer,with_kwargs=True)]
        old_generate=wrapper.model.generate;old_processor=wrapper._processor_inputs
        def processor(*args,**kwargs):
            result=old_processor(*args,**kwargs);rec['processor_calls']+=1
            safe(rec,'inputs',lambda:{k:snapshot(v) for k,v in result.items()})
            # A fresh actual mask on the GPU; no change to native inputs.
            safe(rec,'image_mask',lambda:result['input_ids']==wrapper.model.config.image_token_id)
            return result
        def model_generate(*args,**kwargs):
            rec['generate_calls']+=1;safe(rec,'generate_kwargs',lambda:snapshot(kwargs))
            result=old_generate(*args,**kwargs);safe(rec,'output_ids',lambda:snapshot(result));return result
        wrapper._processor_inputs=processor;wrapper.model.generate=model_generate
        response=None;failure=None
        try:
            method=Qwen3VLChatFixedRes.generate_inner_transformers_fixed if arm.startswith('plain') else Qwen3VLChatCDPruner.generate_inner_transformers_fixed
            response=method(wrapper,item['message'],dataset='DocVQA_VAL')
        except BaseException as error:failure=repr(error)
        finally:
            wrapper._processor_inputs=old_processor;wrapper.model.generate=old_generate
            for hook in hooks:hook.remove()
        record_path=out/f'{index:02d}_{arm}.json'
        record=dict(index=item['index'],position=item['position'],arm=arm,response=response,native_failure=failure,
            observation_errors=rec['errors'],intervention_applied=rec['intervention_applied'],intervention_error=rec['intervention_error'],
            vision_calls=rec['vision_calls'],processor_calls=rec['processor_calls'],generate_calls=rec['generate_calls'],llm_calls=rec['llm_calls'],
            intervention_attempts=rec['intervention_attempts'],status='raw_capture')
        c.save(record_path,record)
        values={}
        def collect(name,fn):
            try:
                value=fn()
                if isinstance(value,torch.Tensor):values[name]=value
            except Exception as error:rec['errors'].append(dict(event='collect:'+name,error=repr(error)))
        vis=rec.get('vision')
        if isinstance(vis,(tuple,list)) and len(vis)==2:
            collect('visual_features',lambda:vis[0])
            for j,x in enumerate(vis[1]):collect('visual_deepstack_'+str(j),lambda x=x:x)
        for label in ('prefill_original','prefill_proposed','prefill'):
            pf=rec.get(label,{})
            for key in ('inputs_embeds','position_ids','attention_mask','cache_position','visual_pos_masks'):
                collect(label+'_'+key,lambda key=key,pf=pf:pf.get(key))
            for j,x in enumerate(pf.get('deepstack_visual_embeds') or []):collect(label+'_deepstack_'+str(j),lambda x=x:x)
        collect('rope_deltas',lambda:rec.get('rope_deltas'))
        for key,x in rec.get('inputs',{}).items():collect('processor_'+key,lambda x=x:x)
        for key,x in rec.get('generate_kwargs',{}).items():collect('generate_'+key,lambda x=x:x)
        collect('output_ids',lambda:rec.get('output_ids'))
        collect('output_suffix_ids',lambda:rec['output_ids'][:,rec['inputs']['input_ids'].shape[1]:] if arm.startswith('plain') else rec['output_ids'])
        collect('injection_image_mask',lambda:rec.get('image_mask'))
        for j,step in enumerate(rec['continuation']):
            for key,x in step.items():collect('continuation_%04d_%s'%(j,key),lambda x=x:x)
        arrays={};metadata={}
        for name,value in values.items():
            try:
                x=value.detach().cpu().contiguous();a=x.view(torch.uint16).numpy() if x.dtype==torch.bfloat16 else x.numpy()
                arrays[name]=a;metadata[name]=dict(dtype=str(x.dtype),shape=list(x.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
            except Exception as error:rec['errors'].append(dict(event='serialize:'+name,error=repr(error)))
        path=out/f'{index:02d}_{arm}.npz';np.savez_compressed(path,**arrays)
        g=rec.get('generate_kwargs',{});scalars={k:v for k,v in g.items() if v is None or type(v) in (str,int,float,bool)}
        continuation_presence=[{k:step[k] for k in ('visual_mask_present','deepstack_present')} for step in rec['continuation']]
        record.update(status='raw_preserved',archive=str(path),archive_sha256=c.sha(path),arrays=metadata,
            generate_kwargs_keys=sorted(g),generate_non_tensor_kwargs=scalars,continuation_presence=continuation_presence,
            observation_errors=rec['errors'],continuation_steps=len(rec['continuation']),GT_or_score_used=False)
        c.save(record_path,record)
        assert failure is None and isinstance(response,str) and not rec['errors']
        assert rec['vision_calls']==rec['processor_calls']==rec['generate_calls']==1 and rec['llm_calls']>=1
        assert len(rec['continuation'])==rec['llm_calls']-1
        proposed={k:v for k,v in metadata.items() if k.startswith('prefill_proposed_')}
        effective={k.replace('prefill_','prefill_proposed_',1):v for k,v in metadata.items() if k.startswith('prefill_') and not k.startswith(('prefill_original_','prefill_proposed_'))}
        assert proposed==effective
        if arm in ('plain_ds_off','eadp1024_ds_on'):assert rec['intervention_attempts']==1 and rec['intervention_applied'] and rec['intervention_error'] is None
        else:assert rec['intervention_attempts']==0 and not rec['intervention_applied']
        return record
    common=['visual_features','visual_deepstack_0','visual_deepstack_1','visual_deepstack_2',
            'processor_input_ids','processor_attention_mask','processor_pixel_values','processor_image_grid_thw',
            'prefill_inputs_embeds','prefill_position_ids','prefill_attention_mask','prefill_cache_position','rope_deltas']
    # All32 endpoints must close against the prior accepted run before either toggle.
    for i,item in enumerate(plan['panel']):
        for arm in ('plain','eadp1024'):
            result=generate(i,item,arm);prior=c.read(c.PREV/'execution'/f'{i:02d}_{arm}.json')
            keys=common+['output_ids','output_suffix_ids']
            if arm=='plain':keys+=['prefill_visual_pos_masks','prefill_deepstack_0','prefill_deepstack_1','prefill_deepstack_2']
            for key in keys:assert result['arrays'][key]==prior['arrays'][key],(i,arm,key)
            assert result['response']==prior['response']
            assert result['generate_kwargs_keys']==prior['generate_kwargs_keys'] and result['generate_non_tensor_kwargs']==prior['generate_non_tensor_kwargs']
    c.save(out/'endpoints_verified.json',dict(n_endpoints=32,all_byte_anchors_passed=True,plan_sha256=c.sha(c.PLAN)))
    for i,item in enumerate(plan['panel']):
        for arm,base in [('plain_ds_off','plain'),('eadp1024_ds_on','eadp1024')]:
            result=generate(i,item,arm);prior=c.read(c.PREV/'execution'/f'{i:02d}_{base}.json')
            for key in common:assert result['arrays'][key]==prior['arrays'][key],(i,arm,key)
            assert result['generate_kwargs_keys']==prior['generate_kwargs_keys'] and result['generate_non_tensor_kwargs']==prior['generate_non_tensor_kwargs']
            if arm=='plain_ds_off':
                assert 'prefill_visual_pos_masks' not in result['arrays'] and all('prefill_deepstack_'+str(j) not in result['arrays'] for j in range(3))
            else:
                reference=c.read(c.PREV/'execution'/f'{i:02d}_plain.json')
                assert result['arrays']['prefill_visual_pos_masks']==reference['arrays']['prefill_visual_pos_masks']
                for j in range(3):assert result['arrays']['prefill_deepstack_'+str(j)]==result['arrays']['visual_deepstack_'+str(j)]
    for path,digest in plan['files'].items():assert c.sha(path)==digest,path
    for group in ('source_sha256','implementation_sha256','reference_sha256'):
        for path,digest in old[group].items():assert c.sha(path)==digest,path
    from namespace_guards import check_model_stats
    from diagnose import stat
    check_model_stats(old,stat,'container')
    for item in plan['panel']:assert c.sha(item['image_path'])==item['image_sha256']
    c.save(out/'summary.json',dict(success=True,n_questions=16,n_native_generations=64,endpoint_anchors=32,
        toggles=32,plan_sha256=c.sha(c.PLAN),GT_read=False,scoring_started=False,benchmark_result=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--allocation-note',type=Path,required=True);args=parser.parse_args()
    try:execute(args.allocation_note.resolve())
    except BaseException as error:
        c.save(c.HERE/'worker_failure.json',dict(status='terminal_failure_no_retry',error=repr(error),artifacts_preserved=True));raise
