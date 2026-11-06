"""Record/compare E4 natural diagnostic evidence; never scores answers."""
import hashlib

def tensor_meta(tensor):
    import torch
    if tensor is None:return None
    value=tensor.detach().contiguous().cpu()
    return {'dtype':str(value.dtype),'shape':list(value.shape),
            'sha256':hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}

def make_record(adapter, sample, output_ids, answer, first_logits, eos_ids, max_new_tokens=2048):
    adapter.verify_calls();s=adapter.state
    if not isinstance(answer,str) or not output_ids or not all(isinstance(x,int) for x in output_ids):
        raise ValueError('natural output IDs and answer required')
    if first_logits is None:raise ValueError('first-step logits evidence required')
    if not isinstance(eos_ids,list) or not all(isinstance(x,int) for x in eos_ids):raise ValueError('EOS IDs required')
    if max_new_tokens<=0 or len(output_ids)>max_new_tokens:raise ValueError('invalid natural output length')
    inp=s['inputs'];eos=output_ids[-1] in eos_ids
    return {'task':sample['task'],'qid':str(sample['qid']),'group_id':sample['group_id'],
        'image_sha256':sample['image_sha256'],'message_sha256':sample['message_sha256'],'mode':adapter.mode,
        'processor':{k:tensor_meta(inp[k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},
        'main_visual':tensor_meta(s['visual']),'pruned_visual':tensor_meta(s['pruned']),
        'ordered_ids':s['ordered'].detach().cpu().tolist(),'spatial_ids':s['spatial'].detach().cpu().tolist(),
        'retained_ids':s['retained'].detach().cpu().tolist(),'full_position_ids':s['full_pos'].detach().cpu().tolist(),
        'full_delta':s['full_delta'].detach().cpu().tolist(),'prefill_position_ids':s['prefill_position_ids'].detach().cpu().tolist(),
        'adjusted_delta':s['adjusted_delta'].detach().cpu().tolist(),'prefill_embeds':tensor_meta(s['prefill_embeds']),
        'prefill_mask':tensor_meta(s['prefill_mask']),'actual_ds_off':s['actual_ds_off'],
        'visual_calls':s['visual_calls'],'greedy_calls':s['greedy_calls'],'block_calls':s['block_calls'],
        'main_merger_calls':s['main_merger_calls'],'ds_calls':s['ds_calls'],'ds_results':s['ds_results'],
        'first_logits':tensor_meta(first_logits),'first_argmax':int(first_logits.detach().reshape(-1).argmax().item()),
        'output_ids':list(output_ids),'answer':answer,'eos_ids':list(eos_ids),'eos':eos,
        'truncated':bool(not eos and len(output_ids)>=max_new_tokens),'max_new_tokens':max_new_tokens}

def compare_pair(drop,skip):
    if (drop['mode'],skip['mode'])!=('C-DROP','C-SKIP'):raise ValueError('expected C-DROP then C-SKIP')
    common=('task','qid','group_id','image_sha256','message_sha256','processor','main_visual','pruned_visual',
            'ordered_ids','spatial_ids','retained_ids','full_position_ids','full_delta','prefill_position_ids',
            'adjusted_delta','prefill_embeds','prefill_mask','first_logits','first_argmax','output_ids','answer',
            'eos_ids','eos','truncated','max_new_tokens','visual_calls','greedy_calls','block_calls','main_merger_calls','ds_calls')
    for key in common:
        if drop[key]!=skip[key]:raise ValueError('C-DROP/C-SKIP identity mismatch: '+key)
    if not drop['actual_ds_off'] or not skip['actual_ds_off']:raise ValueError('DS-off prefill not observed')
    if sorted(tuple(x) for x in drop['ds_results'])!=[(i,False) for i in range(3)]:raise ValueError('C-DROP did not compute three DS streams')
    if sorted(tuple(x) for x in skip['ds_results'])!=[(i,True) for i in range(3)]:raise ValueError('C-SKIP did not skip three DS streams')
    if drop['ds_calls']!=[1,1,1] or any(n!=1 for n in drop['block_calls']) or drop['main_merger_calls']!=1:
        raise ValueError('visual modules not called exactly once')
    return {'status':'identity_passed_bitwise','task':drop['task'],'qid':drop['qid'],'group_id':drop['group_id'],
            'scope':'recorded natural diagnostic pair; no accuracy or timing assertion'}
