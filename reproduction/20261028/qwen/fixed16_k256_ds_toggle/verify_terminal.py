"""Independent CPU audit of the recorded K256 DS pair; never run model or score.

Old full K256 generation has literal responses but no saved token IDs/prefill;
only its text is anchored. Historical selection and tower arrays are separate.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import common as c

ARMS=('ds_off','ds_on')
FIELDS=('inputs_embeds','position_ids','attention_mask','cache_position',
        'visual_pos_masks','deepstack_0','deepstack_1','deepstack_2')
SHARED=('ordered_ids','spatial_ids','quotas','visual_features','visual_deepstack_0',
        'visual_deepstack_1','visual_deepstack_2','pruner_input','pruner_output',
        'processor_input_ids','processor_attention_mask','processor_pixel_values',
        'processor_image_grid_thw','original_token_embeddings','generate_inputs_embeds',
        'generate_attention_mask','rope_deltas')
DTYPES={'torch.bfloat16':np.uint16,'torch.bool':np.bool_,'torch.int64':np.int64,
        'torch.int32':np.int32,'torch.float16':np.float16,'torch.float32':np.float32,
        'torch.float64':np.float64}


def byte_equal(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def check_metadata(array,meta):
    assert meta['dtype'] in DTYPES and array.dtype==DTYPES[meta['dtype']]
    assert list(array.shape)==meta['shape'] and hashlib.sha256(array.tobytes()).hexdigest()==meta['sha256']


def archive(record,path):
    assert c.sha(path)==record['archive_sha256']
    with np.load(path,allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    assert set(data)==set(record['arrays'])
    for key,a in data.items():
        m=record['arrays'][key]
        assert m['dtype'] in DTYPES and a.dtype==DTYPES[m['dtype']] and list(a.shape)==m['shape'],key
        assert hashlib.sha256(a.tobytes()).hexdigest()==m['sha256'],key
    return data


def check_arm(d,r,arm,image_token_id,vocab_size):
    assert r['status']==('historical_anchor_passed' if arm=='ds_off' else 'observations_preserved')
    assert r['native_failure'] is None and not r['observation_errors'] and not r['selection_errors']
    assert isinstance(r['response'],str) and r['response']==r['native_redecode']
    assert not r['GT_read'] and not r['scoring_started']
    assert all(r[k]==1 for k in ('vision_calls','pruner_calls','processor_calls','generate_calls','selection_calls'))
    assert r['intervention_attempts']==(arm=='ds_on') and r['intervention_applied']==(arm=='ds_on') and r['intervention_error'] is None
    assert r['generate_kwargs_keys']==sorted(['inputs_embeds','attention_mask','do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty'])
    assert r['generate_non_tensor_kwargs']==dict(do_sample=False,max_new_tokens=2048,top_p=.001,top_k=1,temperature=.01,repetition_penalty=1.)
    ordered=d['ordered_ids'];spatial=d['spatial_ids'];quota=d['quotas']
    assert ordered.dtype==np.int64 and ordered.shape==(1,256) and spatial.dtype==np.int64 and spatial.shape==(256,)
    assert quota.dtype==np.int64 and quota.shape==(1,) and quota.tolist()==[256]
    assert np.all((ordered>=0)&(ordered<1024)) and len(np.unique(ordered))==256
    assert np.array_equal(spatial,np.sort(ordered[0]))
    vision=d['visual_features'];assert vision.ndim==2 and vision.shape[0]==1024
    assert byte_equal(d['pruner_input'],vision) and byte_equal(d['pruner_output'],vision[spatial])
    for j in range(3):
        layer=d['visual_deepstack_'+str(j)]
        assert layer.shape==vision.shape and layer.dtype==vision.dtype
    ids=d['processor_input_ids'];attention=d['processor_attention_mask'];orig=d['original_token_embeddings']
    assert ids.dtype==np.int64 and ids.ndim==2 and ids.shape[0]==1 and attention.shape==ids.shape
    slots=np.flatnonzero(ids[0]==image_token_id)
    assert len(slots)==1024 and np.array_equal(slots,np.arange(slots[0],slots[0]+1024))
    start=int(slots[0]);length=ids.shape[1]-768
    assert orig.shape[:2]==ids.shape and orig.ndim==3
    embed=d['generate_inputs_embeds'];mask=d['generate_attention_mask']
    assert embed.shape==(1,length,vision.shape[1]) and mask.shape==(1,length)
    assert byte_equal(embed[0,:start],orig[0,:start]) and byte_equal(embed[0,start:start+256],d['pruner_output'])
    assert byte_equal(embed[0,start+256:],orig[0,start+1024:])
    want_attention=np.concatenate((attention[:,:start],np.ones((1,256),dtype=attention.dtype),attention[:,start+1024:]),axis=1)
    assert byte_equal(mask,want_attention)
    for field in FIELDS:
        orig_key='original_prefill_'+field;proposed='proposed_prefill_'+field;final='final_prefill_'+field
        assert (proposed in d)==(final in d)
        if proposed in d:assert byte_equal(d[proposed],d[final]),field
        changed=arm=='ds_on' and field in FIELDS[4:]
        if changed:assert orig_key not in d and final in d,field
        else:
            assert (orig_key in d)==(final in d),field
            if orig_key in d:assert byte_equal(d[orig_key],d[final]),field
    assert byte_equal(d['final_prefill_inputs_embeds'],embed) and byte_equal(d['final_prefill_attention_mask'],mask)
    assert d['final_prefill_position_ids'].shape==(3,1,length)
    assert d['final_prefill_cache_position'].shape==(length,)
    assert d['rope_deltas'].shape==(1,1)
    if arm=='ds_on':
        target=np.zeros((1,length),dtype=np.bool_);target[0,start:start+256]=True
        assert byte_equal(d['final_prefill_visual_pos_masks'],target)
        for j in range(3):assert byte_equal(d['final_prefill_deepstack_'+str(j)],d['visual_deepstack_'+str(j)][spatial])
    output=d['output_ids'];assert output.dtype==np.int64 and output.ndim==2 and output.shape[0]==1 and 0<output.shape[1]<=2048
    assert np.all((output>=0)&(output<vocab_size))
    n=r['continuation_steps'];assert n==r['llm_calls']-1==len(r['continuation_presence'])==output.shape[1]-1
    expected=set()
    for i,presence in enumerate(r['continuation_presence']):
        assert presence==dict(visual_mask_present=False,deepstack_present=False)
        for field,shape in (('position_ids',(3,1,1)),('cache_position',(1,)),('rope_deltas',(1,1))):
            k='continuation_%04d_%s'%(i,field);expected.add(k)
            assert k in d and d[k].dtype==np.int64 and d[k].shape==shape,k
    assert {k for k in d if k.startswith('continuation_')}==expected


def ending(tokens,eos_ids):
    assert len(tokens)>0
    return dict(output_tokens=len(tokens),EOS_present=bool(np.isin(tokens,eos_ids).any()),
                EOS_at_end=bool(int(tokens[-1]) in eos_ids),at_max_tokens=len(tokens)==2048)


def check_pair(off,on):
    for key in SHARED:assert byte_equal(off[key],on[key]),key
    for field in FIELDS:
        a='final_prefill_'+field;b='original_prefill_'+field
        assert (a in off)==(b in on),field
        if a in off:assert byte_equal(off[a],on[b]),field
    x=off['output_ids'][0];y=on['output_ids'][0];prefix=0
    for a,b in zip(x,y):
        if a!=b:break
        prefix+=1
    n=min(prefix,len(x)-1,len(y)-1)
    differences={f:[] for f in ('position_ids','cache_position','rope_deltas')}
    for i in range(n):
        for field in differences:
            key='continuation_%04d_%s'%(i,field)
            if not byte_equal(off[key],on[key]):differences[field].append(i)
    return dict(output_ids_equal=byte_equal(off['output_ids'],on['output_ids']),
                common_prefix_tokens=prefix,common_prefix_continuation_steps=n,
                same_prefix_continuation_differing_steps=differences)


def check_historical(i,item,record,data,records,predictions):
    old_record=records[i];assert old_record['position']==item['position'] and old_record['index']==item['index']
    cache=c.OLD/'execution'/('question_%02d.npz'%i);assert c.sha(cache)==old_record['cache']['sha256']
    prior=c.FACTOR/'execution'/('%02d_eadp1024.npz'%i)
    prior_record=c.read(c.FACTOR/'execution'/('%02d_eadp1024.json'%i));assert c.sha(prior)==prior_record['archive_sha256']
    with np.load(cache,allow_pickle=False) as old,np.load(prior,allow_pickle=False) as full:
        for a,b in [('ordered_ids','native_ordered_ids'),('spatial_ids','native_spatial_ids'),('pruner_input','visual_features_BF16_bits'),('pruner_output','native_pruned_BF16_bits'),('processor_input_ids','full_prompt_input_ids'),('processor_image_grid_thw','image_grid_thw')]:
            assert byte_equal(data[a],old[b]),a
        for key in ('visual_features','visual_deepstack_0','visual_deepstack_1','visual_deepstack_2','processor_input_ids','processor_attention_mask','processor_pixel_values','processor_image_grid_thw'):
            assert byte_equal(data[key],full[key]),key
            check_metadata(full[key],prior_record['arrays'][key])
    row=predictions[item['position']]
    assert row['index']==item['index'] and not row['prediction_changed_by_native_na_loader']
    assert row['raw_prediction']==record['response']
    assert record['historical_anchor']==dict(historical_literal_answer_equal=True,old_native_selection_and_feature_equal=True,old_full_visual_and_three_deepstack_equal=True,old_generation_token_ids_unavailable=True,old_generation_prefill_unavailable=True)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert all(not Path('/dev/'+name).exists() for name in ('nvidia0','nvidiactl','nvidia-uvm'))
    plan=c.read(c.PLAN);c.check(plan,'container');plan_sha=c.sha(c.PLAN)
    state=c.read(c.HERE/'state.json');out=c.HERE/'execution'
    assert state['status']=='native_exited_validation_pending' and state['native_exit_code']==0
    assert state['original_container_exited'] and state['GPU_pid_absent'] and state['plan_sha256']==plan_sha
    complete=c.read(out/'native_complete.json');anchors=c.read(out/'native_anchors.json')
    assert complete['success'] and complete['n_questions']==16 and complete['n_generations']==32 and complete['native_anchors']==16
    assert anchors['success'] and anchors['n']==16 and anchors['all_native_anchors_before_any_DS_on']
    assert complete['plan_sha256']==anchors['plan_sha256']==plan_sha and complete['native_anchors_sha256']==c.sha(out/'native_anchors.json')
    assert not complete['GT_read'] and not complete['scoring_started'] and not anchors['GT_read'] and not anchors['scoring_started']
    expected={f'{i:02d}_{a}' for i in range(16) for a in ARMS}
    assert set(complete['records'])==set(complete['archives'])==expected
    assert set(anchors['records'])==set(anchors['archives'])=={f'{i:02d}' for i in range(16)}
    old_plan=c.read(c.OLD/'run_plan.json');model=Path(old_plan['recipe']['model_path'])
    runtime_path=out/'runtime.json';runtime=c.read(runtime_path)
    assert runtime['plan_sha256']==plan_sha and runtime['actual_recipe']==old_plan['recipe']
    assert runtime['actual_recipe']['visual_token_num']==256 and runtime['actual_recipe']['alpha']==.5 and runtime['actual_recipe']['beta']==2.
    assert runtime['model_dtype']=='torch.bfloat16' and runtime['backend']=='flash_attention_2'
    assert runtime['post_process'] is False and runtime['GPU']=='NVIDIA A40' and runtime['model_instances']==1
    assert not runtime['GT_read'] and not runtime['scoring_started']
    expected_modules={'vlmeval.vlm.qwen3_vl.model_fixed_res','vlmeval.vlm.qwen3_vl.model','model.pruner','transformers.models.qwen3_vl.modeling_qwen3_vl'}
    assert set(runtime['actual_sources'])==expected_modules
    for module,path in runtime['actual_sources'].items():
        assert Path(path).name==module.rsplit('.',1)[-1]+'.py'
        assert path in old_plan['source_sha256'] and c.sha(path)==old_plan['source_sha256'][path]
    config=c.read(model/'config.json');image_token_id=config['image_token_id'];vocab_size=config['text_config']['vocab_size']
    eos_ids=c.read(model/'generation_config.json')['eos_token_id']
    eos_ids=[eos_ids] if isinstance(eos_ids,int) else eos_ids
    assert eos_ids and all(type(x) is int and 0<=x<vocab_size for x in eos_ids)
    old_summary=c.read(c.OLD/'execution/summary.json');old_records_path=c.OLD/'execution/per_question.jsonl'
    assert old_summary['per_question_sha256']==c.sha(old_records_path)
    old_records=[json.loads(s) for s in old_records_path.open()]
    predictions=[json.loads(s) for s in (c.AUDIT/'public_fa2_full_run02_DocVQA_VAL.per_question.jsonl').open()]
    assert len(old_records)==16 and len(predictions)==5349
    import torch
    from transformers import AutoProcessor
    assert not torch.cuda.is_initialized()
    tokenizer=AutoProcessor.from_pretrained(str(model),local_files_only=True).tokenizer
    results=[];count=0;ending_counts={arm:dict(EOS_present=0,EOS_missing=0,at_max_tokens=0) for arm in ARMS}
    for i,item in enumerate(plan['panel']):
        pair={};response={};endings={}
        for arm in ARMS:
            key=f'{i:02d}_{arm}';path=out/(key+'.json');path_npz=out/(key+'.npz');raw_path=out/(key+'.raw.json')
            assert c.sha(path)==complete['records'][key] and c.sha(path_npz)==complete['archives'][key]
            record=c.read(path);raw=c.read(raw_path)
            assert c.sha(raw_path)==record['raw_sha256']
            assert raw==dict(index=item['index'],position=item['position'],arm=arm,response=record['response'],native_failure=record['native_failure'])
            assert record['index']==item['index'] and record['position']==item['position'] and record['arm']==arm
            data=archive(record,path_npz);check_arm(data,record,arm,image_token_id,vocab_size)
            for field,meta in item['processor_expected'].items():assert record['arrays']['processor_'+field]==meta
            decoded=tokenizer.batch_decode(data['output_ids'].tolist(),skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
            assert decoded==record['response']==record['native_redecode']
            if arm=='ds_off':
                assert c.sha(path)==anchors['records'][f'{i:02d}'] and c.sha(path_npz)==anchors['archives'][f'{i:02d}']
                check_historical(i,item,record,data,old_records,predictions)
            tokens=data['output_ids'][0];endings[arm]=ending(tokens,eos_ids)
            ending_counts[arm]['EOS_present' if endings[arm]['EOS_present'] else 'EOS_missing']+=1
            ending_counts[arm]['at_max_tokens']+=int(endings[arm]['at_max_tokens'])
            pair[arm]=data;response[arm]=record['response'];count+=1
        change=check_pair(pair['ds_off'],pair['ds_on']);change['response_equal']=response['ds_off']==response['ds_on']
        results.append(dict(position=item['position'],index=item['index'],comparison=change,endings=endings))
    assert not torch.cuda.is_initialized() and count==32
    target=out/'independent_verification.json';assert not target.exists()
    c.save(target,dict(success=True,n_questions=16,n_generations=32,native_anchors=16,plan_sha256=plan_sha,
        native_complete_sha256=c.sha(out/'native_complete.json'),runtime_sha256=c.sha(runtime_path),
        independent_token_redecodes=32,results=results,ending_counts=ending_counts,CPU_only=True,model_loaded=False,CUDA_initialized=False,
        GT_read=False,scoring_started=False,benchmark_result=False,root_controller_actual_exit_proof_required=True,
        historical_full_generation_anchor='literal answer only; token IDs and prefill not preserved',
        continuation_comparison='only steps on equal generated prefixes; no KV equality claimed'))
    print(json.dumps(dict(success=True,n_questions=16,n_generations=32,independent_token_redecodes=32)))

if __name__=='__main__':main()
