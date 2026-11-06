"""Preliminary post-run validation logic; cannot read GT or launch anything."""
from cohort import validate_ids

NAMES=('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty')

def case(item,arm,reference,record,raw,validate_trace):
    ids=item['ids'][arm];validate_ids(ids)
    for key in ('task','qid','group_id','image_sha256','message_sha256'):
        if str(record.get(key))!=str(item.get(key)):
            raise ValueError('record input identity '+key)
    if (record.get('candidate_arm')!=arm or record.get('budget')!=256 or record.get('ds_on') is not True or
        record.get('selection_mode')!='oracle_fixed_indices' or record.get('ordered_ids')!=ids or
        record.get('spatial_ids')!=ids or record.get('actual_injection_verified') is not True or
        record.get('cleanup_passed') is not True):
        raise ValueError('selection/cleanup mismatch')
    for a,b in (('processor','processor'),('main_visual','visual_sha256'),('full_deepstack','deepstack_full_sha256'),
                ('full_position_ids','original_position_ids'),('full_delta','full_rope_delta')):
        if record.get(a)!=reference.get(b):raise ValueError('reference mismatch '+a)
    if (len(record['full_deepstack'])!=3 or len(record.get('selected_deepstack',[]))!=3 or
        record.get('selected_deepstack')!=record.get('injected_deepstack')):
        raise ValueError('DS metadata mismatch')
    controls=reference['generation_scalars']
    if (not isinstance(controls,dict) or set(controls)!=set(NAMES) or
        record.get('generation_scalars')!=controls or controls['do_sample'] is not False or
        controls['max_new_tokens']!=2048 or record.get('eos_ids')!=reference['eos_ids']):
        raise ValueError('generation controls')
    length=record['original_length'];start=record['image_start']
    if type(length) is not int or type(start) is not int or not 0<=start<=length-1024:
        raise ValueError('image start/length')
    retained=list(range(start))+[start+i for i in ids]+list(range(start+1024,length))
    if record.get('retained_ids')!=retained or len(retained)!=length-768:
        raise ValueError('retained positions')
    full=record['full_position_ids'];pre=record['prefill_position_ids']
    if not isinstance(full,list) or len(full)!=3 or any(len(axis)!=1 or len(axis[0])!=length for axis in full):
        raise ValueError('full P3 shape')
    if pre!=[[[axis[0][i] for i in retained]] for axis in full]:raise ValueError('P3 gather')
    mask=record['injected_mask']
    if mask!=[[False]*start+[True]*256+[False]*(len(retained)-start-256)]:raise ValueError('DS mask')
    delta=record['full_delta'];adjusted=record['adjusted_delta']
    if (not isinstance(delta,list) or len(delta)!=1 or len(delta[0])!=1 or
        adjusted!=[[delta[0][0]+768]]):raise ValueError('RoPE delta')
    if (raw.get('task')!=item['task'] or str(raw.get('qid'))!=str(item['qid']) or
        raw.get('group_id')!=item['group_id'] or raw.get('arm')!=arm or
        raw.get('processor_input_length')!=length or raw.get('generate_input_ids_shape') is not None or
        not isinstance(raw.get('generate_inputs_embeds_shape'),list) or
        raw['generate_inputs_embeds_shape'][:2]!=[1,len(retained)]):
        raise ValueError('raw generate input identity')
    output=record['output_ids'];eos=reference['eos_ids']
    if raw.get('returned_batch')!=[output] or not output or any(type(x) is not int for x in output) or len(output)>2048:
        raise ValueError('raw output mismatch')
    reached=output[-1] in eos;truncated=not reached and len(output)==2048
    if (not reached and not truncated) or record.get('eos')!=reached or record.get('truncated')!=truncated:
        raise ValueError('natural completion')
    validate_trace(record['decode_trace'],output,len(retained),int(adjusted[0][0]))
    return True
