"""DRAFT only: native 5349-row prefill toggles. Never launches itself."""
import argparse, hashlib, importlib.metadata, json, os, sys
from pathlib import Path
import common as c
sys.path.insert(0, str(c.PANEL))
from intervention import change


def worker(note_path):
    p = c.read(c.PLAN); c.check(p, 'container'); note = c.read(note_path)
    assert note['authorization'] == 'explicit_root_gpu_allocation' and note['run_id'] == c.RUN_ID
    assert note['plan_sha256'] == c.sha(c.PLAN) and note['GPU_device'] == 0
    assert note['single_attempt'] and not note['automatic_retry']
    old = c.read(c.OLD/'run_plan.json'); assert Path(sys.executable).resolve() == Path(old['python']).resolve()
    out = c.HERE/'execution'; out.mkdir(exist_ok=False)
    c.save(out/'identity.json', dict(**c.identity(os.getpid()), plan_sha256=c.sha(c.PLAN), allocation_sha256=c.sha(note_path)))
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0, old['author_root']); sys.path.insert(0, str(Path(old['author_root'])/'VLMEvalKit'))
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes, Qwen3VLChatCDPruner, Qwen3VLChatEADP
    assert {n: importlib.metadata.version(n) for n in old['expected_versions']} == old['expected_versions']
    recipe = dict(old['recipe']); recipe['visual_token_num'] = 1024
    wrapper = Qwen3VLChatEADP(**recipe)
    assert wrapper.visual_token_num == wrapper.pruner.visual_token_num == 1024 and wrapper.post_process is False
    assert wrapper.model.config._attn_implementation == wrapper.model.visual.config._attn_implementation == 'flash_attention_2'
    assert wrapper.model.dtype == wrapper.model.visual.dtype == torch.bfloat16
    assert {str(x.device) for x in wrapper.model.parameters()} == {'cuda:0'} and torch.cuda.get_device_name(0) == 'NVIDIA A40'
    sources = {n: str(Path(sys.modules[n].__file__).resolve()) for n in ['vlmeval.vlm.qwen3_vl.model_fixed_res', 'vlmeval.vlm.qwen3_vl.model', 'model.pruner', 'transformers.models.qwen3_vl.modeling_qwen3_vl']}
    for path in sources.values(): assert c.sha(path) == old['source_sha256'][path]
    c.save(out/'runtime.json', dict(actual_recipe=recipe, actual_sources=sources, model_dtype=str(wrapper.model.dtype), backend='flash_attention_2', post_process=False, GPU=torch.cuda.get_device_name(0), model_instances=1, plan_sha256=c.sha(c.PLAN), cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, cudnn_deterministic=torch.backends.cudnn.deterministic, GT_read=False, scoring_started=False))
    wrapper.set_dump_image(lambda row: [row['image_path']])
    items = c.rows(c.BASE/'input_manifest.jsonl'); panel = c.read(c.PANEL/'plan.json')['panel']
    panel_index = {x['position']: i for i,x in enumerate(panel)}
    old_obs = {a:c.rows(c.BASE/'execution'/f'{a}.observations.jsonl') for a in ('plain','eadp1024')}
    old_pred = {a:c.rows(c.BASE/'execution'/f'{a}.predictions.jsonl') for a in ('plain','eadp1024')}
    def snapshot(v):
        if isinstance(v, torch.Tensor): return v.detach().clone()
        if isinstance(v, (tuple,list)): return type(v)(snapshot(x) for x in v)
        if isinstance(v, dict): return {k:snapshot(x) for k,x in v.items()}
        return v
    def meta(t):
        if t is None: return None
        a = t.detach().cpu().contiguous(); b = a.view(torch.uint16).numpy() if a.dtype == torch.bfloat16 else a.numpy()
        return dict(dtype=str(a.dtype), shape=list(a.shape), sha256=hashlib.sha256(b.tobytes()).hexdigest())
    fields = ('inputs_embeds','position_ids','attention_mask','cache_position','visual_pos_masks')
    def fields_snapshot(kw): return {**{k:snapshot(kw.get(k)) for k in fields}, 'deepstack_visual_embeds':snapshot(kw.get('deepstack_visual_embeds'))}
    def fields_meta(v):
        deep = v.get('deepstack_visual_embeds') or []
        assert len(deep) in (0,3)
        return {**{k:meta(v.get(k)) for k in fields}, **{'deepstack_'+str(j):meta(deep[j]) if deep else None for j in range(3)}}
    def write(stream, value): stream.write(json.dumps(value, ensure_ascii=False)+'\n'); stream.flush()
    def generate(item, arm, ps, os_):
        rec = dict(errors=[], vision_calls=0, processor_calls=0, generate_calls=0, llm_calls=0, pruner_calls=0, intervention_attempts=0, intervention_applied=False, intervention_error=None, continuation=[])
        def safe(key, fn):
            try: rec[key] = fn()
            except Exception as error: rec['errors'].append(dict(event=key,error=repr(error)))
        def vision_hook(module,args,result):
            rec['vision_calls'] += 1; rec['raw_vision'] = result; safe('vision',lambda:snapshot(result))
        def pruner_hook(module,args,result):
            rec['pruner_calls'] += 1
            safe('pruner_input_features',lambda:snapshot(args[0]))
            safe('pruner_output_features',lambda:snapshot(result[0]))
            safe('pruner_split_sizes',lambda:list(result[1]))
        def inject(module,args,kwargs):
            rec['llm_calls'] += 1
            if rec['llm_calls'] != 1: return None
            effective = kwargs
            if arm in ('plain_ds_off','eadp1024_ds_on'):
                raw = rec.get('raw_vision'); deep = raw[1] if isinstance(raw,(tuple,list)) and len(raw)==2 else None
                rec['intervention_attempts'] += 1
                effective, rec['intervention_applied'], rec['intervention_error'] = change(arm,kwargs,rec.get('image_mask'),deep)
            safe('prefill_original',lambda:fields_snapshot(kwargs)); safe('prefill_proposed',lambda:fields_snapshot(effective))
            if effective is not kwargs: return args,effective
        def observe(module,args,kwargs):
            if rec['llm_calls']==1:
                safe('prefill_final',lambda:fields_snapshot(kwargs)); safe('rope_deltas',lambda:snapshot(wrapper.model.model.rope_deltas))
            else:
                try: rec['continuation'].append(dict(position_ids=snapshot(kwargs.get('position_ids')),cache_position=snapshot(kwargs.get('cache_position')),rope_deltas=snapshot(wrapper.model.model.rope_deltas),visual_pos_masks_present=kwargs.get('visual_pos_masks') is not None,deepstack_visual_embeds_present=kwargs.get('deepstack_visual_embeds') is not None))
                except Exception as error: rec['errors'].append(dict(event='continuation',error=repr(error)))
        assert not wrapper.model.model.language_model._forward_pre_hooks and not torch.nn.modules.module._global_forward_pre_hooks
        hooks=[wrapper.model.visual.register_forward_hook(vision_hook), wrapper.pruner.register_forward_hook(pruner_hook), wrapper.model.model.language_model.register_forward_pre_hook(inject,with_kwargs=True), wrapper.model.model.language_model.register_forward_pre_hook(observe,with_kwargs=True)]
        old_generate, old_processor = wrapper.model.generate, wrapper._processor_inputs
        def processor(*args,**kwargs):
            result = old_processor(*args,**kwargs); rec['processor_calls'] += 1
            safe('inputs',lambda:snapshot(result)); safe('image_mask',lambda:result['input_ids']==wrapper.model.config.image_token_id)
            return result
        def model_generate(*args,**kwargs):
            rec['generate_calls'] += 1; safe('generate_kwargs',lambda:snapshot(kwargs))
            result=old_generate(*args,**kwargs); safe('output_ids',lambda:snapshot(result)); return result
        wrapper._processor_inputs=processor; wrapper.model.generate=model_generate
        response=None; failure=None
        try:
            msg=wrapper._build_vqa_prompt(item,'DocVQA_VAL'); messages=wrapper._build_messages(msg,dataset='DocVQA_VAL')
            assert hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest()==item['message_sha256']
            method=Qwen3VLChatFixedRes.generate_inner_transformers_fixed if arm.startswith('plain') else Qwen3VLChatCDPruner.generate_inner_transformers_fixed
            response=method(wrapper,msg,dataset='DocVQA_VAL')
        except BaseException as error: failure=repr(error)
        finally:
            wrapper._processor_inputs=old_processor; wrapper.model.generate=old_generate
            for h in hooks: h.remove()
        # The native response is durably retained before any completeness or observation test.
        pred=dict(position=item['position'],index=item['index'],prediction=response,native_failure=failure); write(ps,pred)
        obs=dict(position=item['position'],index=item['index'],arm=arm,native_failure=failure,observer_errors=rec['errors'])
        try:
            inp=rec['inputs']; output=rec['output_ids']; suffix=output[:,inp['input_ids'].shape[1]:] if arm.startswith('plain') else output
            ids=suffix.detach().cpu().tolist()[0]; mask=rec['image_mask'].detach().cpu().tolist()[0]; vision=rec['vision']
            g=rec['generate_kwargs']; scalar={k:v for k,v in g.items() if v is None or type(v) in (str,int,float,bool)}
            obs.update(actual_processor={k:meta(v) for k,v in inp.items()}, input_ids=inp['input_ids'].detach().cpu().tolist()[0], output_ids=output.detach().cpu().tolist()[0], output_suffix_ids=ids,
                native_redecode=wrapper.processor.tokenizer.batch_decode([ids],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0], image_mask=mask,
                image_mask_sha256=meta(rec['image_mask'])['sha256'],image_mask_shape=list(rec['image_mask'].shape),image_mask_true_count=sum(mask),
                vision_features=meta(vision[0]),visual_deepstack=[meta(x) for x in vision[1]],rope_deltas=meta(rec['rope_deltas']),
                pruner_calls=rec['pruner_calls'],pruner_input_features=meta(rec.get('pruner_input_features')),pruner_output_features=meta(rec.get('pruner_output_features')),pruner_split_sizes=rec.get('pruner_split_sizes'),
                generate_kwargs_keys=sorted(g),generate_non_tensor_kwargs=scalar,generate_tensor_kwargs={k:meta(v) for k,v in g.items() if isinstance(v,torch.Tensor)},
                continuation=[{**{k:meta(step[k]) for k in ('position_ids','cache_position','rope_deltas')}, **{k:step[k] for k in ('visual_pos_masks_present','deepstack_visual_embeds_present')}} for step in rec['continuation']])
            for label in ('prefill_original','prefill_proposed','prefill_final'): obs[label]=fields_meta(rec[label])
            for key in ('vision_calls','processor_calls','generate_calls','llm_calls','intervention_attempts','intervention_applied','intervention_error'): obs[key]=rec[key]
            obs['continuation_steps']=len(rec['continuation'])
        except Exception as error: obs['observer_errors'].append(dict(event='serialize',error=repr(error)))
        write(os_,obs)
        assert failure is None and isinstance(response,str) and not obs['observer_errors']
        assert rec['vision_calls']==rec['processor_calls']==rec['generate_calls']==1 and rec['llm_calls']>=1
        assert obs['continuation_steps']==rec['llm_calls']-1 and obs['prefill_proposed']==obs['prefill_final']
        assert obs['actual_processor']==item['processor_expected'] and obs['native_redecode']==response
        if arm.startswith('plain'): assert obs['pruner_calls']==0 and obs['pruner_input_features'] is None and obs['pruner_output_features'] is None and obs['pruner_split_sizes'] is None
        else: assert obs['pruner_calls']==1 and obs['pruner_split_sizes']==[1024] and obs['pruner_input_features']==obs['pruner_output_features']==obs['vision_features']
        if arm in ('plain_ds_off','eadp1024_ds_on'): assert rec['intervention_attempts']==1 and rec['intervention_applied'] and rec['intervention_error'] is None
        return pred,obs
    def match_prior(index,arm,pred,obs):
        prior=c.read(c.PANEL/'execution'/f'{index:02d}_{arm}.json'); a=prior['arrays']
        assert prior['response']==pred['prediction'] and prior['generate_kwargs_keys']==obs['generate_kwargs_keys'] and prior['generate_non_tensor_kwargs']==obs['generate_non_tensor_kwargs']
        assert a['output_suffix_ids']['sha256']==hashlib.sha256(__import__('struct').pack('<'+'q'*len(obs['output_suffix_ids']),*obs['output_suffix_ids'])).hexdigest()
        assert a['output_ids']['sha256']==hashlib.sha256(__import__('struct').pack('<'+'q'*len(obs['output_ids']),*obs['output_ids'])).hexdigest()
        for k,v in obs['actual_processor'].items(): assert a['processor_'+k]==v
        assert a['visual_features']==obs['vision_features']
        for j in range(3): assert a['visual_deepstack_'+str(j)]==obs['visual_deepstack'][j]
        for label, prefix in [('prefill_original','prefill_original_'),('prefill_proposed','prefill_proposed_'),('prefill_final','prefill_')]:
            for k,v in obs[label].items(): assert a.get(prefix+k)==v,(index,arm,prefix+k)
        assert a.get('rope_deltas')==obs['rope_deltas']
    streams={}
    try:
        for arm in ('plain','eadp1024'):
            streams[arm]=((out/f'endpoint_{arm}.predictions.jsonl').open('x'),(out/f'endpoint_{arm}.observations.jsonl').open('x'))
        for i,panel_item in enumerate(panel):
            item=items[panel_item['position']]; assert item['index']==panel_item['index']
            for arm in ('plain','eadp1024'):
                pred,obs=generate(item,arm,*streams[arm]); match_prior(i,arm,pred,obs)
                assert obs['actual_processor']==old_obs[arm][item['position']]['actual_processor']
                assert obs['output_suffix_ids']==old_obs[arm][item['position']]['output_suffix_ids'] and pred['prediction']==old_pred[arm][item['position']]['prediction']
        for arm in ('plain','eadp1024'):
            for f in streams[arm]:f.close()
        c.save(out/'endpoints_verified.json',dict(success=True,n_endpoints=32,plan_sha256=c.sha(c.PLAN),prediction_sha256={a:c.sha(out/f'endpoint_{a}.predictions.jsonl') for a in ('plain','eadp1024')},observation_sha256={a:c.sha(out/f'endpoint_{a}.observations.jsonl') for a in ('plain','eadp1024')},old_full_suffix_matches=32,old_processor_matches=32,old_prefill_metadata_matches=32))
        for arm in ('plain_ds_off','eadp1024_ds_on'):
            streams[arm]=((out/f'{arm}.predictions.jsonl').open('x'),(out/f'{arm}.observations.jsonl').open('x'))
        for item in items:
            for arm in ('plain_ds_off','eadp1024_ds_on'):
                pred,obs=generate(item,arm,*streams[arm])
                if item['position'] in panel_index: match_prior(panel_index[item['position']],arm,pred,obs)
            if (item['position']+1)%50==0: print(json.dumps(dict(completed_pairs=item['position']+1,total=5349)),flush=True)
    finally:
        for pair in streams.values():
            for f in pair: f.close()
    for group in ('source_sha256','implementation_sha256','reference_sha256'):
        for path,digest in old[group].items(): assert c.sha(path)==digest,path
    from namespace_guards import check_model_stats
    from diagnose import stat
    check_model_stats(old,stat,'container')
    for path,digest in p['file_guards'].items(): assert c.sha(path)==digest,path
    for path,digest in {r['image_path']:r['image_sha256'] for r in items}.items(): assert c.sha(path)==digest,path
    arms=('plain_ds_off','eadp1024_ds_on')
    c.save(out/'native_complete.json',dict(success=True,n_questions=5349,n_generations=10698,endpoint_anchors=32,plan_sha256=c.sha(c.PLAN),prediction_sha256={a:c.sha(out/f'{a}.predictions.jsonl') for a in arms},observation_sha256={a:c.sha(out/f'{a}.observations.jsonl') for a in arms},endpoints_verified_sha256=c.sha(out/'endpoints_verified.json'),all_generation_before_scoring=True,GT_read=False,scoring_started=False))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--allocation-note',type=Path,required=True);args=ap.parse_args()
    try:worker(args.allocation_note.resolve())
    except BaseException as error:
        c.save(c.HERE/'worker_failure.json',dict(status='terminal_failure_no_retry',error=repr(error),partial_predictions_preserved=True));raise
