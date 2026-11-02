"""Independent CPU audit of actual MMB paired guidance panel artifacts.

Earlier completed MMB data preserved native answers and geometry, not native
internal arrays or output IDs. Reference input hashes were computed on CPU.
This module never loads an inference model, reads GT, or computes a score.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ARMS=('source','question_only')

def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def bits(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and np.ascontiguousarray(a).tobytes()==np.ascontiguousarray(b).tobytes()

def check_selection(mask,ordered,quota):
    assert mask.dtype==np.bool_ and mask.shape==(1,576)
    assert ordered.dtype==quota.dtype==np.int64 and ordered.shape==(1,128) and quota.shape==(1,) and quota.tolist()==[128]
    values=ordered[0].tolist();active=np.flatnonzero(mask[0]).tolist()
    assert len(set(values))==128 and all(0<=x<576 for x in values) and sorted(values)==active
    return dict(greedy_quotas=[128],greedy_effective_prefixes=[values],effective_mask_indices=[active],returned_retained_tokens=128)

def check_embeddings(embedded,projected,mask,input_ids):
    assert input_ids.dtype==np.int64 and input_ids.ndim==2 and input_ids.shape[0]==1
    marker=np.flatnonzero(input_ids[0]==-200);assert len(marker)==1;marker=int(marker[0])
    assert projected.dtype==embedded.dtype==np.float16 and projected.shape==(1,576,4096)
    assert mask.dtype==np.bool_ and mask.shape==(1,576) and int(mask.sum())==128
    assert embedded.shape==(1,input_ids.shape[1]-1+128,4096)
    visual=projected[0,np.flatnonzero(mask[0])]
    assert bits(embedded[0,marker:marker+128],visual)
    nonvisual=np.concatenate((embedded[0,:marker],embedded[0,marker+128:]),axis=0)
    assert nonvisual.shape==(input_ids.shape[1]-1,4096)
    return nonvisual,dict(marker_index=marker,n_visual=128,visual_fp16_sha256=array_sha(visual),
                          nonvisual_fp16_sha256=array_sha(nonvisual),prepared_embeddings_shape=list(embedded.shape))

def check_clip_inputs(ids,mask,fingerprint):
    segments=fingerprint['n_segments'];assert type(segments) is int and segments>=1
    assert ids.dtype==mask.dtype==np.int64 and ids.shape==mask.shape==(segments,77)
    assert array_sha(ids)==fingerprint['input_ids_sha256'] and array_sha(mask)==fingerprint['attention_mask_sha256']
    n=fingerprint['n_tokens'];assert (n-1)//77+1==segments
    assert np.all((mask==0)|(mask==1)) and np.all(mask.reshape(-1)[n:]==0) and np.all(ids.reshape(-1)[n:]==0)

def check_shapes(arrays,segments):
    expected={
        'actual_input_fp16':(np.float16,(1,3,336,336)),
        'projected_fp16':(np.float16,(1,576,4096)),
        'final_mask':(np.bool_,(1,576)),
        'text_global':(np.float32,(segments,768)),
        'text_dense':(np.float32,(segments,77,768)),
        'image_embeds':(np.float32,(1,576,768)),
        'global_sim':(np.float32,(1,576,segments)),
        'dense_sim':(np.float32,(1,576,segments,77)),
        'importance':(np.float32,(1,576)),
        'visual_sim':(np.float32,(1,576,576)),
        'internal_ids':(np.int64,(1,128)),'quotas':(np.int64,(1,)),
        'clip_input_ids':(np.int64,(segments,77)),'clip_attention_mask':(np.int64,(segments,77)),
    }
    assert set(arrays)==set(expected)|{'actual_generate_input_ids','actual_input_ids','prepared_embeddings_fp16','output_ids'}
    for key,(dtype,shape) in expected.items():
        assert arrays[key].dtype==dtype and arrays[key].shape==shape,key
    for key,value in arrays.items():
        assert np.isfinite(value).all(),key
    assert bits(arrays['actual_input_ids'],arrays['actual_generate_input_ids'])


def check_pair(source,variant):
    for key in ('actual_input_ids','actual_input_fp16','projected_fp16','image_embeds','visual_sim'):
        assert bits(source[key],variant[key]),key
    nonvisual0,_=check_embeddings(source['prepared_embeddings_fp16'],source['projected_fp16'],source['final_mask'],source['actual_input_ids'])
    nonvisual1,_=check_embeddings(variant['prepared_embeddings_fp16'],variant['projected_fp16'],variant['final_mask'],variant['actual_input_ids'])
    assert bits(nonvisual0,nonvisual1)
    compare=('text_global','text_dense','global_sim','dense_sim','importance','internal_ids','final_mask','prepared_embeddings_fp16','output_ids')
    return {k+'_changed':not bits(source[k],variant[k]) for k in compare}

def check_output(output,vocab,bos,eos):
    assert output.dtype==np.int64 and output.ndim==2 and output.shape[0]==1 and 1<output.shape[1]<=1025
    assert int(output[0,0])==bos and np.all((output>=0)&(output<vocab))
    values=output[0].tolist();new=values[1:];return dict(returned_tokens=len(values),new_tokens=len(new),
        EOS_present=any(v in eos for v in new),EOS_at_end=new[-1] in eos,at_max_tokens=len(new)==1024)


def text_sha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()


def check_manifest(c,manifest):
    """Verify source text, CPU input reference and deterministic panel separately."""
    from PIL import Image
    mmb=c.HERE.parent;refdir=mmb/'question_only_guidance_cpu_audit_20261030'
    assert manifest['schema_version']==1 and manifest['n_rows']==60 and manifest['n_groups']==16
    assert manifest['source_TSV_contains_GT_bytes'] and manifest['source_TSV_bytes_read_including_GT']
    assert manifest['GT_columns_parsed'] is False and manifest['all_rounds'] is False and manifest['round_id']==0
    assert manifest['contains_historical_model_responses'] and not manifest['historical_responses_are_GT']
    assert c.sha(manifest['panel_tsv']['path'])==manifest['panel_tsv']['sha256']
    evidence=[json.loads(s) for s in (refdir/'token_fingerprints.jsonl').read_text().splitlines()]
    groups=sorted({x['group'] for x in evidence});assert len(evidence)==4377 and len(groups)==1176
    chosen=[groups[(2*i+1)*len(groups)//32] for i in range(16)]
    assert chosen==manifest['group_ids']
    selected=[x for x in evidence if x['group'] in set(chosen)]
    rows=manifest['rows'];assert len(selected)==len(rows)==60
    assert [x['position'] for x in rows]==list(range(60)) and len({x['index'] for x in rows})==60
    assert [x['index'] for x in selected]==[x['index'] for x in rows]
    assert sorted(sum(r['group_id']==g for r in rows) for g in chosen)==[2,2]+[4]*14
    prior=manifest['sources']['source_predictions'];runtime_ref=manifest['sources']['source_runtime']
    assert c.sha(prior['path'])==prior['sha256'] and c.sha(runtime_ref['path'])==runtime_ref['sha256']
    prior_lines=Path(prior['path']).read_bytes().splitlines();prior_rows=[json.loads(s) for s in prior_lines]
    runtime=[json.loads(s) for s in Path(runtime_ref['path']).read_text().splitlines()]
    assert len(prior_rows)==len(runtime)==4377
    cpu_inputs=c.read(mmb/'cpu_inputs_current.json');cpu_inputs_pinned=c.read(mmb/'cpu_inputs_author_pins.json')
    llm_rows=c.read(mmb/'cpu_llm_tokens_current.json')['records'];llm_pinned=c.read(mmb/'cpu_llm_tokens_author_pins.json')['records']
    llm_by_id={r['index']:r for r in llm_rows};llm_pin_by_id={r['index']:r for r in llm_pinned}
    for row,reference in zip(rows,selected):
        pos=row['original_position'];previous=prior_rows[pos];old=runtime[pos];img=row['image']
        assert row['round_id']==0 and row['group_id']==row['index']%1000000==reference['group']
        assert row['accepted_source']['source_runtime_call']==old['call']==pos+1
        assert previous['question_id']==row['index'] and previous['round_id']==0 and previous['text']==row['accepted_source']['text']
        assert previous['prompt']==row['full_guidance'] and hashlib.sha256(prior_lines[pos]).hexdigest()==row['accepted_source']['raw_line_sha256']
        assert row['accepted_source']['raw_prediction_sha256']==prior['sha256']
        assert old['guidance_text_sha256']==text_sha(row['full_guidance'])==row['source_clip']['text_sha256']
        assert old['input_shape']==[1,row['llm_n_tokens']] and old['image_shape']==img['pixel_shape']==[1,3,336,336]
        assert old['image_dtype']==img['pixel_dtype']=='torch.float16' and old['image_sizes']==[img['size']]
        assert old['max_new_tokens']==1024 and old['do_sample'] is False and old['returned_visual_token_num']==128
        assert c.sha(img['path'])==img['payload_sha256']==reference['image_payload_sha256']
        with Image.open(img['path']) as image:
            rgb=image.convert('RGB');assert list(rgb.size)==img['size'] and rgb.mode==img['mode']=='RGB'
            assert hashlib.sha256(rgb.tobytes()).hexdigest()==img['decoded_rgb_sha256']
        image_reference=cpu_inputs['images'][img['payload_sha256']]
        assert image_reference==cpu_inputs_pinned['images'][img['payload_sha256']]
        assert image_reference['size']==img['size'] and image_reference['mode']==img['mode']
        assert image_reference['decoded_rgb_sha256']==img['decoded_rgb_sha256'] and image_reference['shape']==img['pixel_shape']
        assert image_reference['float16_sha256']==img['pixel_float16_sha256'] and image_reference['float32_sha256']==img['pixel_float32_sha256']
        llm_reference=llm_by_id[row['index']];assert llm_reference==llm_pin_by_id[row['index']]
        assert llm_reference['prompt_sha256']==text_sha(row['llm_prompt']) and llm_reference['n_tokens']==row['llm_n_tokens']
        assert llm_reference['token_ids_sha256']==row['llm_input_ids_sha256']
        llm=np.asarray([row['llm_input_ids']],dtype=np.int64)
        assert llm.shape==(1,row['llm_n_tokens']) and array_sha(llm)==row['llm_input_ids_sha256']
        for arm,old_name in (('source','full_guidance'),('question_only','question_only_guidance')):
            expected=row[arm+'_clip'];stored=reference[old_name]
            assert text_sha(row[old_name])==expected['text_sha256']==stored['text_sha256']
            for key in ('n_tokens','n_segments','input_ids_sha256','attention_mask_sha256'):assert expected[key]==stored[key]
            check_clip_inputs(np.asarray(expected['input_ids'],dtype=np.int64),np.asarray(expected['attention_mask'],dtype=np.int64),expected)
    for group in chosen:
        subset=[r for r in rows if r['group_id']==group]
        assert len({r['question_only_guidance'] for r in subset})==1
        assert len({r['image']['payload_sha256'] for r in subset})==1
    return rows


PARAMETERS=dict(do_sample=False,temperature=0.0,top_p=None,num_beams=1,max_new_tokens=1024,use_cache=True)


def check_case(c,arm,index,item,description,author,runtime):
    label='%s_%02d'%(arm,index);record_path=c.OUT/(label+'.json')
    assert Path(description['path'])==record_path and c.sha(record_path)==description['sha256']
    record=c.read(record_path)
    assert record['status']=='captured_unvalidated' and record['arm']==arm and record['position']==index
    assert record['index']==item['index'] and record['group_id']==item['group_id']
    assert record['calls']==dict(encode=1,cross=1,greedy=1,prepare=1,clip=1) and not record['observer_errors']
    assert record['generate_scalar_kwargs']==PARAMETERS
    assert record['generate_keys']==sorted(['images','image_sizes','texts',*PARAMETERS]) and record['positional_args_count']==1
    guidance=item['full_guidance'] if arm=='source' else item['question_only_guidance'];clip=item[arm+'_clip']
    assert record['guidance']==guidance and record['encode_texts']==[guidance] and text_sha(guidance)==clip['text_sha256']
    assert record['image_sizes']==[item['image']['size']] and record['returned_visual_token_num']==128
    paths={key:c.OUT/(label+suffix) for key,suffix in (('raw','.raw.json'),('output_ids_file','.output_ids.json'),('archive','.npz'))}
    for key,path in paths.items():
        assert Path(record[key]['path'])==path and c.sha(path)==record[key]['sha256'],key
    raw=c.read(paths['raw']);ids_record=c.read(paths['output_ids_file'])
    assert raw['arm']==ids_record['arm']==arm and raw['position']==ids_record['position']==index
    assert raw['index']==ids_record['index']==item['index'] and raw['group_id']==item['group_id']
    assert raw['native_returned'] and raw['native_failure'] is None and ids_record['native_returned']
    assert raw['response']==record['response']==author['text'] and raw['output_ids']==ids_record['output_ids']
    assert raw['returned_visual_token_num']==128
    assert author['question_id']==item['index'] and author['round_id']==0 and author['prompt']==item['full_guidance']
    assert isinstance(record['response'],str)
    with np.load(paths['archive'],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    assert set(arrays)==set(record['arrays'])
    for key,array in arrays.items():
        meta=record['arrays'][key]
        assert meta==dict(dtype=str(array.dtype),shape=list(array.shape),sha256=array_sha(array)),key
    check_shapes(arrays,clip['n_segments'])
    assert arrays['actual_input_ids'].tolist()==[item['llm_input_ids']]
    assert arrays['actual_input_ids'].shape==tuple([1,item['llm_n_tokens']])
    assert array_sha(arrays['actual_input_ids'])==item['llm_input_ids_sha256']
    assert array_sha(arrays['actual_input_fp16'])==item['image']['pixel_float16_sha256']
    check_clip_inputs(arrays['clip_input_ids'],arrays['clip_attention_mask'],clip)
    assert arrays['clip_input_ids'].tolist()==clip['input_ids'] and arrays['clip_attention_mask'].tolist()==clip['attention_mask']
    check_selection(arrays['final_mask'],arrays['internal_ids'],arrays['quotas'])
    check_embeddings(arrays['prepared_embeddings_fp16'],arrays['projected_fp16'],arrays['final_mask'],arrays['actual_input_ids'])
    assert arrays['output_ids'].tolist()==raw['output_ids']
    eos=runtime['eos_token_id'];eos=[eos] if isinstance(eos,int) else eos
    ending=check_output(arrays['output_ids'],runtime['vocab_size'],runtime['bos_token_id'],eos)
    if arm=='source':assert record['response']==item['accepted_source']['text']
    return record,arrays,ending


def terminal():
    import os,sys
    import common as c
    assert sys.flags.optimize==0 and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not any(Path('/dev',x).exists() for x in ('nvidia0','nvidiactl','nvidia-uvm'))
    plan=c.read(c.PLAN);c.check(plan);plan_sha=c.sha(c.PLAN)
    release=c.read(c.OUT/'native_release.json')
    assert release['status']=='native_exited_validation_pending' and release['native_exit_code']==0
    assert release['original_worker_exited'] and release['GPU_pid_absent'] and release['plan_sha256']==plan_sha
    identity=c.read(c.OUT/'worker_identity.json');claim=c.read(c.HERE/'launch.claim.json')
    assert identity['identity']==release['worker'] and identity['parent']==claim['controller']==release['controller']
    assert identity['plan_sha256']==claim['plan_sha256']==plan_sha
    assert identity['allocation_sha256']==claim['allocation_sha256']==release['allocation_sha256']
    manifest=c.read(c.HERE/'inputs/manifest.json');rows=check_manifest(c,manifest)
    complete=c.read(c.OUT/'native_complete.json');assert complete['status']=='complete'
    assert complete['plan_sha256']==plan_sha and complete['n_source']==complete['n_question_only']==60 and not complete['scoring']
    def described(entry,path):
        assert Path(entry['path'])==path and c.sha(path)==entry['sha256']
    described(complete['runtime'],c.OUT/'runtime.json');runtime=c.read(c.OUT/'runtime.json')
    assert runtime['plan_sha256']==plan_sha and runtime['one_loaded_model']
    assert runtime['model_path']==str(c.MODEL) and runtime['model_dtype']=='torch.float16' and runtime['device']=='cuda:0'
    assert runtime['attention_implementation']=='sdpa' and runtime['K']==128 and runtime['alpha']==.5 and runtime['beta']==1.
    assert runtime['torch']=='2.1.2+cu118' and runtime['transformers']=='4.37.2' and runtime['source_sha256']==c.sha(c.SOURCE)
    import sys as _sys
    _sys.path.insert(0,str(c.HERE.parent/'question_only_guidance_panel_design'))
    import source_boundary
    assert runtime['source_boundary_proof']==source_boundary.proof()
    old_wait=c.read(c.HERE.parent/'streamwait_control/runtime_patch_validation.json')
    for key in ('source_sha256','original_forward_ast_sha256','patched_forward_ast_sha256','exact_added_statement_count','added_statements','restored_ast_identical'):
        assert runtime['wait_patch'][key]==old_wait[key],key
    assert runtime['wait_patch']['exact_added_statement_count']==3 and runtime['wait_patch']['restored_ast_identical']
    described(complete['source_anchor_gate'],c.OUT/'source_anchor_gate.json');gate=c.read(c.OUT/'source_anchor_gate.json')
    assert gate['passed'] and gate['n_sources']==60 and gate['all_sources_before_variant'] and gate['plan_sha256']==plan_sha
    assert gate['records']==complete['source_records'] and gate['author_output']==complete['source_author']
    assert len(complete['source_records'])==len(complete['question_only_records'])==60
    author={}
    for arm in ARMS:
        path=c.OUT/(arm+'_author.jsonl');described(complete[arm+'_author'],path)
        author[arm]=[json.loads(s) for s in path.read_text().splitlines()];assert len(author[arm])==60
    # The author tokenizer is loaded only on CPU. No model weights or scorer.
    import torch
    from transformers import AutoTokenizer
    assert not torch.cuda.is_initialized()
    tokenizer=AutoTokenizer.from_pretrained(str(c.MODEL),use_fast=False,local_files_only=True)
    results=[];groups={};endings={a:dict(EOS_present=0,EOS_missing=0,at_max_tokens=0) for a in ARMS}
    for i,item in enumerate(rows):
        pair={};records={};end={}
        for arm in ARMS:
            record,arrays,status=check_case(c,arm,i,item,complete[arm+'_records'][i],author[arm][i],runtime)
            decoded=tokenizer.batch_decode(arrays['output_ids'].tolist(),skip_special_tokens=True)[0].strip()
            assert decoded==record['response']
            pair[arm]=arrays;records[arm]=record;end[arm]=status
            endings[arm]['EOS_present' if status['EOS_present'] else 'EOS_missing']+=1
            endings[arm]['at_max_tokens']+=int(status['at_max_tokens'])
        comparison=check_pair(pair['source'],pair['question_only'])
        comparison['response_changed']=records['source']['response']!=records['question_only']['response']
        comparison['visual_input_changed']=not bits(pair['source']['projected_fp16'][0,pair['source']['final_mask'][0]],
                                                     pair['question_only']['projected_fp16'][0,pair['question_only']['final_mask'][0]])
        results.append(dict(position=i,index=item['index'],group_id=item['group_id'],comparison=comparison,endings=end))
        group=groups.setdefault(item['group_id'],{a:dict(mask_sha256=[],responses=[]) for a in ARMS})
        for arm in ARMS:
            group[arm]['mask_sha256'].append(array_sha(pair[arm]['final_mask']))
            group[arm]['responses'].append(records[arm]['response'])
        del pair
    assert not torch.cuda.is_initialized()
    for group in groups.values():
        for arm in ARMS:
            x=group[arm];x['unique_mask_count']=len(set(x['mask_sha256']));x['unique_response_count']=len(set(x['responses']))
    assert len(groups)==16
    for item in plan['files']:assert c.sha(item['path'])==item['sha256']
    target=c.OUT/'independent_verification.json';assert not target.exists()
    c.save_new(target,dict(success=True,n_groups=16,n_rows=60,n_generations=120,source_anchors_verified=60,
        actual_clip_segment_checks=120,cpu_reference_LLM_image_checks=120,independent_token_redecodes=120,
        plan_sha256=plan_sha,native_complete_sha256=c.sha(c.OUT/'native_complete.json'),
        native_release_sha256=c.sha(c.OUT/'native_release.json'),runtime_sha256=c.sha(c.OUT/'runtime.json'),
        source_anchor_gate_sha256=c.sha(c.OUT/'source_anchor_gate.json'),results=results,groups=groups,ending_counts=endings,
        CPU_only=True,model_loaded=False,CUDA_initialized=False,GT_read=False,scoring_started=False,benchmark_result=False,
        native_process_exit_recorded_by_controller=True,root_actual_controller_exit_required=True,
        limits=['During earlier panel preparation, the original TSV bytes containing GT were read; only input columns were parsed. This verifier does not read that TSV or GT.',
                'Historical native source records save literal output, guidance hash and input/image geometry; the historical GPU run did not preserve token IDs, internal features, masks or output IDs.',
                'New actual LLM IDs and FP16 pixels are compared to independent CPU fingerprints, not to old saved GPU bytes.',
                'Question-only changes options, hint when present, and CLIP segmentation together; no accuracy or single-cause attribution follows.',
                'Responses within a circular group are not required to share a letter because option ordering changes.']))
    print(json.dumps(dict(success=True,n_groups=16,n_rows=60,n_generations=120,independent_token_redecodes=120)))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--terminal',action='store_true');args=parser.parse_args()
    if args.terminal:terminal()
