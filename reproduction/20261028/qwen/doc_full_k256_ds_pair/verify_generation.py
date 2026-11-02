"""Independent CPU audit of all paired K256 generations; no model or labels.

Full-run high-dimensional arrays are not saved; their actual runtime summaries
cannot be independently reconstructed here. Fixed16 anchors retain full arrays.
"""
import hashlib
import json
import os
import struct
from pathlib import Path

ARMS=('ds_off','ds_on')
FIELDS=('inputs_embeds','position_ids','attention_mask','cache_position',
        'visual_pos_masks','deepstack_0','deepstack_1','deepstack_2')


def ids_sha(values):
    assert all(type(v) is int for v in values)
    return hashlib.sha256(struct.pack('<'+'q'*len(values),*values)).hexdigest()


def meta_check(m,dtype=None,shape=None):
    assert isinstance(m,dict) and set(m)=={'dtype','shape','sha256'}
    assert isinstance(m['dtype'],str) and isinstance(m['shape'],list)
    assert all(type(x) is int and x>=0 for x in m['shape'])
    assert isinstance(m['sha256'],str) and len(m['sha256'])==64
    int(m['sha256'],16)
    if dtype is not None:assert m['dtype']==dtype
    if shape is not None:assert m['shape']==shape


def bool_meta(mask):
    assert all(type(x) is bool for x in mask)
    return dict(dtype='torch.bool',shape=[1,len(mask)],sha256=hashlib.sha256(bytes(mask)).hexdigest())


def check_observation(item,prediction,obs,arm,image_token_id,vocab_size):
    assert arm in ARMS and obs['arm']==prediction['arm']==arm
    assert obs['position']==prediction['position']==item['position'] and obs['index']==prediction['index']==item['index']
    assert prediction['native_failure'] is None and isinstance(prediction['response'],str)
    assert obs['native_failure'] is None and not obs['observer_errors'] and not obs['selection_errors']
    assert obs['actual_processor']==item['processor_expected']
    ids=obs['input_ids'];assert ids and obs['actual_processor']['input_ids']==dict(dtype='torch.int64',shape=[1,len(ids)],sha256=ids_sha(ids))
    places=[i for i,x in enumerate(ids) if x==image_token_id]
    assert len(places)==1024 and places==list(range(places[0],places[0]+1024))
    start=places[0];n=len(ids)-768
    expected_mask=[start<=i<start+256 for i in range(n)]
    mask=obs['image_mask'];assert mask==expected_mask and all(type(x) is bool for x in mask)
    ordered=obs['ordered_ids'];spatial=obs['spatial_ids']
    assert len(ordered)==256 and len(spatial)==256 and all(type(x) is int and 0<=x<1024 for x in ordered)
    assert len(set(ordered))==256 and spatial==sorted(ordered) and obs['quotas']==[256] and obs['pruner_split_sizes']==[256]
    for key in ('vision_calls','pruner_calls','processor_calls','generate_calls','selection_calls'):assert obs[key]==1
    visual=obs['vision_features'];meta_check(visual,'torch.bfloat16',[1024,4096])
    assert obs['pruner_input']==visual
    selected=obs['selected_vision_features'];meta_check(selected,'torch.bfloat16',[256,4096])
    assert obs['pruner_output']==selected
    layers=obs['visual_deepstack'];selected_layers=obs['selected_visual_deepstack']
    assert len(layers)==len(selected_layers)==3
    for layer,subset in zip(layers,selected_layers):
        meta_check(layer,'torch.bfloat16',[1024,4096]);meta_check(subset,'torch.bfloat16',[256,4096])
    meta_check(obs['original_token_embeddings'],'torch.bfloat16',[1,len(ids),4096])
    meta_check(obs['reconstructed_inputs_embeds'],'torch.bfloat16',[1,n,4096])
    meta_check(obs['reconstructed_attention_mask'],'torch.int64',[1,n])
    originals,proposed,final=(obs[k] for k in ('prefill_original','prefill_proposed','prefill_final'))
    for p in (originals,proposed,final):
        assert set(p)==set(FIELDS)
        for value in p.values():
            if value is not None:meta_check(value)
    assert proposed==final
    for field in FIELDS[:4]:assert originals[field] is not None and originals[field]==final[field],field
    for key,dtype,shape in (('inputs_embeds','torch.bfloat16',[1,n,4096]),('position_ids','torch.int64',[3,1,n]),('attention_mask','torch.int64',[1,n]),('cache_position','torch.int64',[n])):
        meta_check(originals[key],dtype,shape)
    for key in FIELDS[4:]:assert originals[key] is None
    if arm=='ds_on':
        assert final['visual_pos_masks']==bool_meta(mask)
        for i in range(3):assert final['deepstack_'+str(i)]==selected_layers[i]
    else:
        for key in FIELDS[4:]:assert final[key] is None
    assert obs['intervention_attempts']==int(arm=='ds_on') and obs['intervention_applied'] is (arm=='ds_on')
    assert obs['intervention_error'] is None
    scalars=dict(do_sample=False,max_new_tokens=2048,top_p=.001,top_k=1,temperature=.01,repetition_penalty=1.)
    assert obs['generate_non_tensor_kwargs']==scalars
    assert set(obs['generate_kwargs_keys'])==set(scalars)|{'inputs_embeds','attention_mask'}
    tensors=obs['generate_tensor_kwargs'];assert set(tensors)=={'inputs_embeds','attention_mask'}
    assert tensors['inputs_embeds']==originals['inputs_embeds']==obs['reconstructed_inputs_embeds']
    assert tensors['attention_mask']==originals['attention_mask']==obs['reconstructed_attention_mask']
    meta_check(obs['rope_deltas'],'torch.int64',[1,1])
    out=obs['output_ids'];assert 0<len(out)<=2048 and all(type(x) is int and 0<=x<vocab_size for x in out)
    assert obs['native_redecode']==prediction['response']
    assert obs['continuation_steps']==obs['llm_calls']-1==len(obs['continuation'])==len(out)-1
    for step in obs['continuation']:
        assert set(step)=={'position_ids','cache_position','rope_deltas','visual_mask_present','deepstack_present'}
        assert step['visual_mask_present'] is False and step['deepstack_present'] is False
        for key,shape in (('position_ids',[3,1,1]),('cache_position',[1]),('rope_deltas',[1,1])):meta_check(step[key],'torch.int64',shape)


def check_pair(off,on):
    for key in ('actual_processor','input_ids','image_mask','ordered_ids','spatial_ids','quotas','pruner_split_sizes',
                'vision_features','pruner_input','pruner_output','selected_vision_features','visual_deepstack',
                'selected_visual_deepstack','original_token_embeddings','reconstructed_inputs_embeds',
                'reconstructed_attention_mask','generate_kwargs_keys','generate_non_tensor_kwargs','generate_tensor_kwargs','rope_deltas'):
        assert off[key]==on[key],key
    assert off['prefill_final']==on['prefill_original']
    x,y=off['output_ids'],on['output_ids'];prefix=0
    for a,b in zip(x,y):
        if a!=b:break
        prefix+=1
    n=min(prefix,len(off['continuation']),len(on['continuation']))
    differing={k:[] for k in ('position_ids','cache_position','rope_deltas')}
    for i in range(n):
        for key in differing:
            if off['continuation'][i][key]!=on['continuation'][i][key]:differing[key].append(i)
    return dict(output_ids_equal=x==y,common_prefix_tokens=prefix,common_prefix_continuation_steps=n,
                same_prefix_continuation_differing_steps=differing)


def check_ordered(items,original,ordered):
    assert len(original)==len(ordered)==len(items)
    assert [x['position'] for x in ordered]==list(range(len(items)))
    assert all(row['index']==items[pos]['index'] for pos,row in enumerate(ordered))
    assert ordered==sorted(original,key=lambda x:x['position'])


def ending(tokens,eos_ids):
    assert len(tokens)>0
    return dict(output_tokens=len(tokens),EOS_present=any(x in eos_ids for x in tokens),
                EOS_at_end=tokens[-1] in eos_ids,at_max_tokens=len(tokens)==2048)


def check_anchor(c,index,arm,prediction,obs,anchors):
    """Reopen actual two fixed16 archives and bind new summaries to arrays."""
    import numpy as np
    key=f'{index:02d}_{arm}';old_dir=c.FIXED/'execution';new_dir=c.HERE/'execution'/'anchors'
    record=c.read(new_dir/(key+'.json'));prior=c.read(old_dir/(key+'.json'))
    new_path=new_dir/(key+'.npz');old_path=old_dir/(key+'.npz')
    assert c.sha(new_dir/(key+'.json'))==anchors['records'][key]
    assert c.sha(new_path)==anchors['archives'][key]==record['archive_sha256']
    assert c.sha(old_path)==prior['archive_sha256']
    assert record['position']==prior['position']==obs['position']
    assert record['index']==prior['index']==obs['index'] and record['arm']==prior['arm']==arm
    assert record['response']==prior['response']==prediction['response']
    assert record['arrays']==prior['arrays']
    assert record['generate_kwargs_keys']==prior['generate_kwargs_keys']==obs['generate_kwargs_keys']
    assert record['generate_non_tensor_kwargs']==prior['generate_non_tensor_kwargs']==obs['generate_non_tensor_kwargs']
    dtypes={'torch.bfloat16':np.uint16,'torch.bool':np.bool_,'torch.int64':np.int64,'torch.int32':np.int32,
            'torch.float16':np.float16,'torch.float32':np.float32,'torch.float64':np.float64}
    with np.load(new_path,allow_pickle=False) as new,np.load(old_path,allow_pickle=False) as old:
        assert set(new.files)==set(old.files)==set(record['arrays'])
        for k in new.files:
            a,b=new[k],old[k];m=record['arrays'][k]
            assert m['dtype'] in dtypes and a.dtype==b.dtype==dtypes[m['dtype']] and list(a.shape)==list(b.shape)==m['shape'],k
            assert a.tobytes()==b.tobytes() and hashlib.sha256(a.tobytes()).hexdigest()==m['sha256'],k
        maps={'vision_features':'visual_features','pruner_input':'pruner_input','pruner_output':'pruner_output',
              'original_token_embeddings':'original_token_embeddings','rope_deltas':'rope_deltas'}
        for key,field in maps.items():assert obs[key]==record['arrays'][field],key
        for key,val in obs['actual_processor'].items():assert val==record['arrays']['processor_'+key]
        for i in range(3):assert obs['visual_deepstack'][i]==record['arrays']['visual_deepstack_'+str(i)]
        for key in ('inputs_embeds','attention_mask'):assert obs['generate_tensor_kwargs'][key]==record['arrays']['generate_'+key]
        for name,prefix in (('prefill_original','original_prefill_'),('prefill_proposed','proposed_prefill_'),('prefill_final','final_prefill_')):
            for key,value in obs[name].items():assert value==record['arrays'].get(prefix+key),(name,key)
        assert obs['ordered_ids']==new['ordered_ids'][0].tolist() and obs['spatial_ids']==new['spatial_ids'].tolist()
        assert obs['quotas']==new['quotas'].tolist() and obs['output_ids']==new['output_ids'][0].tolist()
        assert record['continuation_steps']==len(obs['continuation'])==len(record['continuation_presence'])
        for i,(step,presence) in enumerate(zip(obs['continuation'],record['continuation_presence'])):
            assert step['visual_mask_present']==presence['visual_mask_present'] and step['deepstack_present']==presence['deepstack_present']
            for key in ('position_ids','cache_position','rope_deltas'):
                assert step[key]==record['arrays']['continuation_%04d_%s'%(i,key)]
        spatial=np.array(obs['spatial_ids'],dtype=np.int64)
        def bfmeta(a):return dict(dtype='torch.bfloat16',shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
        assert obs['selected_vision_features']==bfmeta(new['visual_features'][spatial])
        for j in range(3):assert obs['selected_visual_deepstack'][j]==bfmeta(new['visual_deepstack_'+str(j)][spatial])
        start=obs['image_mask'].index(True)
        rebuilt=np.concatenate((new['original_token_embeddings'][:,:start],new['pruner_output'][None,:,:],new['original_token_embeddings'][:,start+1024:]),axis=1)
        assert rebuilt.tobytes()==new['generate_inputs_embeds'].tobytes() and obs['reconstructed_inputs_embeds']==bfmeta(rebuilt)
        original_mask=new['processor_attention_mask']
        rebuilt_mask=np.concatenate((original_mask[:,:start],np.ones((1,256),dtype=original_mask.dtype),original_mask[:,start+1024:]),axis=1)
        assert rebuilt_mask.dtype==new['generate_attention_mask'].dtype and rebuilt_mask.tobytes()==new['generate_attention_mask'].tobytes()
        assert obs['reconstructed_attention_mask']==dict(dtype='torch.int64',shape=list(rebuilt_mask.shape),sha256=hashlib.sha256(rebuilt_mask.tobytes()).hexdigest())


def check_runtime(c,runtime,plan_sha):
    old=c.read(c.OLD/'run_plan.json')
    assert runtime['actual_recipe']==old['recipe'] and runtime['actual_recipe']['visual_token_num']==256
    assert runtime['actual_recipe']['alpha']==.5 and runtime['actual_recipe']['beta']==2.
    assert runtime['model_dtype']=='torch.bfloat16' and runtime['backend']=='flash_attention_2'
    assert runtime['post_process'] is False and runtime['GPU']=='NVIDIA A40' and runtime['model_instances']==1
    assert runtime['plan_sha256']==plan_sha and runtime['GT_read'] is False and runtime['scoring_started'] is False
    assert set(runtime['actual_sources'])=={'vlmeval.vlm.qwen3_vl.model_fixed_res','vlmeval.vlm.qwen3_vl.model','model.pruner','transformers.models.qwen3_vl.modeling_qwen3_vl'}
    for module,path in runtime['actual_sources'].items():
        assert Path(path).name==module.rsplit('.',1)[-1]+'.py'
        assert path in old['source_sha256'] and c.sha(path)==old['source_sha256'][path]


def main():
    import common as c
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not any(Path('/dev',name).exists() for name in ('nvidia0','nvidiactl','nvidia-uvm'))
    plan=c.read(c.PLAN);c.check(plan,'container');plan_sha=c.sha(c.PLAN)
    state=c.read(c.HERE/'state.json');out=c.HERE/'execution';target=out/'independent_verification.json'
    assert not target.exists()
    assert state['status']=='native_exited_validation_pending' and state['native_exit_code']==0
    assert state['original_container_exited'] and state['GPU_pid_absent'] and state['plan_sha256']==plan_sha
    runtime_path=out/'runtime.json';runtime=c.read(runtime_path);check_runtime(c,runtime,plan_sha)
    done=c.read(out/'native_complete.json');anchors=c.read(out/'native_anchors.json')
    assert done['success'] and done['n_questions']==5349 and done['n_generations']==10698 and done['native_anchors']==32
    assert anchors['success'] and anchors['n']==32 and anchors['all_fixed16_before_remaining']
    assert done['plan_sha256']==anchors['plan_sha256']==plan_sha and done['native_anchors_sha256']==c.sha(out/'native_anchors.json')
    assert not done['GT_read'] and not done['scoring_started'] and not anchors['GT_read'] and not anchors['scoring_started']
    expected={f'{i:02d}_{a}' for i in range(16) for a in ARMS}
    assert set(anchors['records'])==set(anchors['archives'])==expected
    assert set(done['prediction_sha256'])==set(done['observation_sha256'])==set(ARMS)
    assert set(done['ordered_prediction_sha256'])==set(done['ordered_observation_sha256'])==set(ARMS)
    items=c.rows(c.INPUT);panel=plan['panel']
    assert len(items)==5349 and [x['position'] for x in items]==list(range(5349))
    assert len({x['index'] for x in items})==5349 and len(panel)==16 and len({r['position'] for r in panel})==16
    for p in panel:assert items[p['position']]['index']==p['index']
    fixed_positions={r['position']:i for i,r in enumerate(panel)}
    order=[r['position'] for r in panel]+[i for i in range(5349) if i not in fixed_positions]
    assert len(order)==len(set(order))==5349
    old=c.read(c.OLD/'run_plan.json');model=Path(old['recipe']['model_path']);config=c.read(model/'config.json')
    token_id=config['image_token_id'];vocab=config['text_config']['vocab_size']
    eos=c.read(model/'generation_config.json')['eos_token_id'];eos=[eos] if isinstance(eos,int) else eos
    assert eos and all(type(x) is int and 0<=x<vocab for x in eos) and runtime['eos_token_ids']==eos
    all_rows={};all_predictions={};summaries={a:dict(EOS_present=0,EOS_missing=0,at_max_tokens=0) for a in ARMS}
    for arm in ARMS:
        pred_path=out/(arm+'.predictions.jsonl');obs_path=out/(arm+'.observations.jsonl')
        assert c.sha(pred_path)==done['prediction_sha256'][arm] and c.sha(obs_path)==done['observation_sha256'][arm]
        predictions=c.rows(pred_path);observations=c.rows(obs_path)
        sorted_pred_path=out/(arm+'.ordered.predictions.jsonl');sorted_obs_path=out/(arm+'.ordered.observations.jsonl')
        assert c.sha(sorted_pred_path)==done['ordered_prediction_sha256'][arm] and c.sha(sorted_obs_path)==done['ordered_observation_sha256'][arm]
        check_ordered(items,predictions,c.rows(sorted_pred_path));check_ordered(items,observations,c.rows(sorted_obs_path))
        assert len(predictions)==len(observations)==5349
        assert [x['position'] for x in predictions]==[x['position'] for x in observations]==order
        for pos,pred,obs in zip(order,predictions,observations):
            check_observation(items[pos],pred,obs,arm,token_id,vocab)
            end=ending(obs['output_ids'],eos)
            assert obs['eos']==dict(ids=eos,present=end['EOS_present'],at_end=end['EOS_at_end'],at_max_tokens=end['at_max_tokens'])
            summaries[arm]['EOS_present' if end['EOS_present'] else 'EOS_missing']+=1
            summaries[arm]['at_max_tokens']+=int(end['at_max_tokens'])
            if pos in fixed_positions:check_anchor(c,fixed_positions[pos],arm,pred,obs,anchors)
        all_rows[arm]=observations;all_predictions[arm]=predictions
    results=[]
    for pos,off,on,p0,p1 in zip(order,all_rows['ds_off'],all_rows['ds_on'],all_predictions['ds_off'],all_predictions['ds_on']):
        comparison=check_pair(off,on);comparison['response_equal']=p0['response']==p1['response']
        results.append(dict(position=pos,index=items[pos]['index'],comparison=comparison,
            endings={a:ending(o['output_ids'],eos) for a,o in (('ds_off',off),('ds_on',on))}))
    # Load the author's processor/tokenizer only after all raw and archive gates.
    # It does not load model weights or invoke a scorer.
    import torch
    from transformers import AutoProcessor
    assert not torch.cuda.is_initialized()
    tokenizer=AutoProcessor.from_pretrained(str(model),local_files_only=True).tokenizer
    for arm in ARMS:
        decoded=tokenizer.batch_decode([x['output_ids'] for x in all_rows[arm]],skip_special_tokens=True,clean_up_tokenization_spaces=False)
        assert decoded==[x['response'] for x in all_predictions[arm]],arm
    assert not torch.cuda.is_initialized()
    for path,digest in plan['files'].items():assert c.sha(path)==digest,path
    c.save(target,dict(success=True,n_questions=5349,n_generations=10698,native_anchors=32,processor_identity_checks=10698,
        selection_checks=10698,prefill_intervention_checks=10698,independent_token_redecodes=10698,
        plan_sha256=plan_sha,native_complete_sha256=c.sha(out/'native_complete.json'),native_anchors_sha256=c.sha(out/'native_anchors.json'),
        runtime_sha256=c.sha(runtime_path),prediction_sha256=done['prediction_sha256'],observation_sha256=done['observation_sha256'],
        ordered_prediction_sha256=done['ordered_prediction_sha256'],ordered_observation_sha256=done['ordered_observation_sha256'],
        results=results,ending_counts=summaries,CPU_only=True,model_loaded=False,CUDA_initialized=False,GT_read=False,scoring_started=False,
        native_container_exit_verified_by_controller=True,root_controller_actual_exit_proof_required=True,
        high_dimensional_arrays_independently_reconstructed_for_nonanchors=False,
        limits=['Outside the 32 fixed16 anchors, high-dimensional tensors were not saved; dtype/shape/SHA and selected-row or reconstruction relationships were captured during generation and cannot be independently rebuilt from hashes.',
                'The earlier full K256 generation did not preserve internal tensors or token IDs. Its previously verified literal-response anchor covers only the 16 fixed questions.',
                'Changing DeepStack can change later KV and outputs; continuation is compared only over equal generated prefixes.',
                'No ground truth or scoring is used by this generation audit.']))
    print(json.dumps(dict(success=True,n_questions=5349,n_generations=10698,independent_token_redecodes=10698)))

if __name__=='__main__':main()
