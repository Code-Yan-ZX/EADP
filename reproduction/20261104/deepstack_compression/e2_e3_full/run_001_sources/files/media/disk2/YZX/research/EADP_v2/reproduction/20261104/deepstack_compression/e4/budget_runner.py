"""E4-A instrumented single-arm core; no CLI and no admission authority.

The caller must independently admit a completed E1 run, freeze DEV inventory,
supervise process/container exit, and validate real-model smoke before a full
panel. This core does not open labels, score, load a model or launch itself.
"""
from budget_adapter import BudgetAdapter
from identity_records import tensor_meta
from decode_observer import DecodeObserver,validate_trace


def run_arm(torch, wrapper, native_pruner, sample, budget, ds_on, generate, on_generated, archive_arrays,
            forced_token=None,capture_prefill_logits=False):
    if wrapper.visual_token_num != budget or wrapper.pruner.visual_token_num != budget:
        raise ValueError('wrapper and actual pruner must both have requested budget')
    if sample.get('split') != 'DEV':
        raise ValueError('only frozen DEV inputs may be used')
    with BudgetAdapter(torch, wrapper, native_pruner, budget, ds_on) as adapter:
        # Capture the actual model.generate invocation, not a caller assertion.
        seen={}; original=wrapper.model.generate
        forced=forced_token is not None
        if forced and (type(forced_token) is not int or forced_token<0):raise ValueError('invalid forced token')
        logits_handle=None
        if forced or capture_prefill_logits:
            def capture_logits(_module,_args,output):
                if 'prefill_logits' not in seen:seen['prefill_logits']=output[:,-1,:].detach().clone()
            logits_handle=wrapper.model.lm_head.register_forward_hook(capture_logits)
        def capture(*args,**kw):
            if seen:raise ValueError('multiple generate calls in one arm')
            if forced:
                from transformers import LogitsProcessor,LogitsProcessorList
                class ForceToken(LogitsProcessor):
                    def __call__(self,_input_ids,scores):
                        scores.fill_(-float('inf'));scores[:,forced_token]=0
                        return scores
                kw.update(max_new_tokens=33,min_new_tokens=33,eos_token_id=[],logits_processor=LogitsProcessorList([ForceToken()]))
            names=('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty')
            seen['generation_scalars']={key:kw.get(key) for key in names}
            eos=kw.get('eos_token_id',wrapper.model.generation_config.eos_token_id)
            seen['eos_ids']=[eos] if isinstance(eos,int) else list(eos or [])
            result=original(*args,**kw);seen['output_ids']=result.detach().cpu().tolist()[0]
            # Record exposure before any downstream hashing, observation, or
            # validation can fail. The caller must fsync this separate journal.
            on_generated({'task':sample['task'],'qid':str(sample['qid']),'group_id':sample['group_id'],
                          'budget':budget,'ds_on':ds_on,'output_ids':seen['output_ids'],
                          'eos_ids':seen['eos_ids'],'generation_scalars':seen['generation_scalars'],'forced':forced})
            return result
        wrapper.model.generate=capture
        try:
            with DecodeObserver(wrapper) as observer:answer=generate(wrapper, sample)
        finally:
            wrapper.model.generate=original
            if logits_handle is not None:logits_handle.remove()
        adapter.verify_calls()
        if not seen or 'output_ids' not in seen:raise ValueError('no actual completed model.generate')
        ids,eos_ids,maximum=seen['output_ids'],seen['eos_ids'],seen['generation_scalars']['max_new_tokens']
        if (not isinstance(ids, list) or not ids or any(type(x) is not int for x in ids)
                or not isinstance(answer, str) or not isinstance(eos_ids, list)
                or any(type(x) is not int for x in eos_ids) or maximum != (33 if forced else 2048) or len(ids) > maximum):
            raise ValueError('invalid natural generation record')
        s = adapter.state; inp = s['inputs']; eos = ids[-1] in eos_ids
        validate_trace(observer.trace,ids,len(s['retained']),int(s['adjusted_delta'][0,0]))
        if forced and (ids!=[forced_token]*33 or eos_ids!=[] or 'prefill_logits' not in seen):
            raise ValueError('fixed-prefix diagnostic incomplete')
        if capture_prefill_logits and 'prefill_logits' not in seen:raise ValueError('anchor prefill logits missing')
        tensors={'visual':s['visual'],'prefill':s['prefill_embeds'],**{'ds'+str(i):x for i,x in enumerate(s['deepstack'])}}
        if ds_on:tensors.update({'injected'+str(i):x for i,x in enumerate(s['actual_deepstack'])})
        if forced:
            tensors.update({'processor_'+k:inp[k] for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')})
        if forced or capture_prefill_logits:
            tensors['prefill_logits']=seen['prefill_logits']
        arrays=archive_arrays(tensors)
        if not isinstance(arrays,dict) or not (
            (arrays.get('kind')=='full_arrays' and set(arrays)=={'kind','path','sha256'}) or
            (arrays.get('kind')=='summary_only' and set(arrays)=={'kind','tensors'})):
            raise ValueError('explicit array archive or summary evidence required')
        return {'task': sample['task'], 'qid': str(sample['qid']), 'group_id': sample['group_id'],
                'image_sha256': sample['image_sha256'], 'message_sha256': sample['message_sha256'],
                'budget': budget, 'ds_on': ds_on, 'selection_mode': s['selection_mode'],
                'processor': {k: tensor_meta(inp[k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},
                'main_visual': tensor_meta(s['visual']), 'pruned_visual': tensor_meta(s['pruned']),
                'image_start':s['start'],'original_length':inp['input_ids'].shape[1], 'arrays':arrays,
                'full_deepstack': [tensor_meta(x) for x in s['deepstack']],
                'selected_deepstack': [tensor_meta(x) for x in s['selected_deepstack']],
                'injected_mask': None if s['actual_ds_mask'] is None else s['actual_ds_mask'].detach().cpu().tolist(),
                'injected_deepstack': [tensor_meta(x) for x in s['actual_deepstack']] if ds_on else None,
                'ordered_ids': s['ordered'].detach().cpu().tolist(),
                'spatial_ids': s['spatial'].detach().cpu().tolist(),
                'retained_ids': s['retained'].detach().cpu().tolist(),
                'full_position_ids': s['full_pos'].detach().cpu().tolist(),
                'full_delta': s['full_delta'].detach().cpu().tolist(),
                'prefill_position_ids': s['prefill_position_ids'].detach().cpu().tolist(),
                'adjusted_delta': s['adjusted_delta'].detach().cpu().tolist(),
                'prefill_embeds': tensor_meta(s['prefill_embeds']), 'prefill_mask': tensor_meta(s['prefill_mask']),
                'prefill_text_before':tensor_meta(s['prefill_embeds'][:,:s['start']]),
                'prefill_text_after':tensor_meta(s['prefill_embeds'][:,s['start']+budget:]),
                'greedy_calls': s['greedy_calls'], 'visual_calls': s['visual_calls'],
                'main_merger_calls': s['main_merger_calls'], 'block_calls': s['block_calls'], 'ds_calls': s['ds_calls'],
                'actual_injection_verified': s['actual_injection_verified'], 'output_ids': ids, 'answer': answer,
                'eos_ids': eos_ids, 'eos': eos, 'truncated': not forced and not eos and len(ids) >= maximum,
                'max_new_tokens': maximum, 'generation_scalars': seen['generation_scalars'],
                'forced':forced,'forced_token':forced_token,'decode_trace':observer.trace,
                'prefill_logits':tensor_meta(seen.get('prefill_logits'))}
