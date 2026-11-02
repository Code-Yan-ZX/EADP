"""CPU terminal primitives for full MMB paired guidance; no model or GT.

Non-anchor high-dimensional tensors are retained only as hashes recorded after
native generation. Equality of these hashes does not reconstruct those tensors.
"""
import hashlib
import struct
from pathlib import Path
import numpy as np

ARMS=('source','question_only')

def bytes_sha(value):return hashlib.sha256(value).hexdigest()
def array_sha(a):return bytes_sha(np.ascontiguousarray(a).tobytes())
def array_equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and np.ascontiguousarray(a).tobytes()==np.ascontiguousarray(b).tobytes()

def int64_sha(values):
    assert all(type(v) is int for v in values)
    return bytes_sha(struct.pack('<'+'q'*len(values),*values))

def check_metadata(meta,dtype=None,shape=None):
    assert isinstance(meta,dict) and set(meta)=={'dtype','shape','sha256'}
    assert isinstance(meta['dtype'],str) and isinstance(meta['shape'],list)
    assert all(type(x) is int and x>=0 for x in meta['shape'])
    assert isinstance(meta['sha256'],str) and len(meta['sha256'])==64
    assert all(x in '0123456789abcdef' for x in meta['sha256'])
    if dtype is not None:assert meta['dtype']==dtype
    if shape is not None:assert meta['shape']==shape

def check_selection(ordered,quota,mask):
    assert isinstance(ordered,list) and len(ordered)==128
    assert all(type(x) is int and 0<=x<576 for x in ordered) and len(set(ordered))==128
    assert quota==[128] and len(mask)==576 and all(type(x) is bool for x in mask)
    spatial=[i for i,x in enumerate(mask) if x]
    assert len(spatial)==128 and spatial==sorted(ordered)
    return spatial

def check_clip(ids,mask,ref):
    n=ref['n_tokens'];segments=ref['n_segments']
    assert type(n) is int and n>0 and type(segments) is int and segments==(n-1)//77+1
    assert len(ids)==len(mask)==segments and all(len(row)==77 for row in ids+mask)
    a=[x for row in ids for x in row];b=[x for row in mask for x in row]
    assert all(type(x) is int and x>=0 for x in a) and all(type(x) is int and x in (0,1) for x in b)
    assert not any(a[n:]) and not any(b[n:])
    assert int64_sha(a)==ref['input_ids_sha256'] and int64_sha(b)==ref['attention_mask_sha256']
    if 'input_ids' in ref:assert ids==ref['input_ids']
    if 'attention_mask' in ref:assert mask==ref['attention_mask']

def check_visual_relationship(selected,prepared_visual,nonvisual,n_input_ids):
    check_metadata(selected,'float16',[128,4096]);check_metadata(prepared_visual,'float16',[128,4096])
    check_metadata(nonvisual,'float16',[n_input_ids-1,4096]);assert selected==prepared_visual

def check_output(ids,vocab_size,bos,eos):
    assert isinstance(ids,list) and 1<len(ids)<=1025 and all(type(v) is int and 0<=v<vocab_size for v in ids)
    assert ids[0]==bos and eos and all(type(x) is int for x in eos)
    return dict(returned_tokens=len(ids),new_tokens=len(ids)-1,EOS_present=any(v in eos for v in ids[1:]),
                EOS_at_end=ids[-1] in eos,at_max_tokens=len(ids)-1==1024)

def check_ordered(items,original,ordered):
    assert len(items)==len(original)==len(ordered)
    assert [r['position'] for r in ordered]==list(range(len(items)))
    assert all(row['index']==items[pos]['index'] for pos,row in enumerate(ordered))
    assert ordered==sorted(original,key=lambda x:x['position'])


def check_saved_small(values,metadata,dtype,shape=None):
    """Rebuild recorded small integer/bool arrays and check byte fingerprint."""
    assert dtype in ('int64','bool')
    def validate(x):
        if isinstance(x,list):
            for y in x:validate(y)
        else:assert type(x) is (int if dtype=='int64' else bool)
    validate(values)
    a=np.asarray(values,dtype=np.int64 if dtype=='int64' else np.bool_)
    check_metadata(metadata,dtype,shape)
    assert list(a.shape)==metadata['shape'] and array_sha(a)==metadata['sha256']
    return a


def check_full_anchor(new_path,metadata,prior_path,prior_metadata):
    """Independent bytewise full archive comparison, no model or inference."""
    with np.load(new_path,allow_pickle=False) as left,np.load(prior_path,allow_pickle=False) as right:
        assert set(left.files)==set(right.files)==set(metadata)==set(prior_metadata)
        for key in left.files:
            a,b=left[key],right[key];m=metadata[key];p=prior_metadata[key]
            assert array_equal(a,b),key
            assert m==p==dict(dtype=str(a.dtype),shape=list(a.shape),sha256=array_sha(a)),key
            assert np.isfinite(a).all(),key


def check_pair(source,variant):
    """Compare actual-capture metadata and small tensors, never reconstruct hashes."""
    a,b=source['arrays'],variant['arrays'];sa,sb=source['small_arrays'],variant['small_arrays']
    assert sa['actual_input_ids']==sb['actual_input_ids']
    for name in ('actual_input_ids','actual_input_fp16','projected_fp16','image_embeds','visual_sim'):
        assert a[name]==b[name],name
    assert source['derived']['prepare_nonvisual']==variant['derived']['prepare_nonvisual']
    for record in (source,variant):
        ids=record['small_arrays']['actual_input_ids']
        assert len(ids)==1 and len(ids[0])>1
        d=record['derived'];check_visual_relationship(d['selected_visual'],d['prepare_visual'],d['prepare_nonvisual'],len(ids[0]))
    return dict(final_mask_changed=sa['final_mask']!=sb['final_mask'],
                ordered_ids_changed=sa['internal_ids']!=sb['internal_ids'],
                output_ids_changed=sa['output_ids']!=sb['output_ids'],
                response_changed=source['response']!=variant['response'],
                importance_changed=a['importance']!=b['importance'],
                visual_input_changed=source['derived']['prepare_visual']!=variant['derived']['prepare_visual'])


PARAMETERS=dict(do_sample=False,temperature=0.0,top_p=None,num_beams=1,max_new_tokens=1024,use_cache=True)
EXPECTED_ARRAYS={'actual_input_fp16','actual_generate_input_ids','clip_input_ids','clip_attention_mask',
                 'text_global','text_dense','image_embeds','global_sim','dense_sim','importance','visual_sim',
                 'internal_ids','quotas','projected_fp16','final_mask','actual_input_ids','prepared_embeddings_fp16','output_ids'}
SMALL={'actual_input_ids','clip_input_ids','clip_attention_mask','internal_ids','quotas','final_mask','output_ids'}


def text_sha(value):return bytes_sha(value.encode('utf-8'))


def check_case(record,raw,ids_record,item,arm,runtime):
    assert record['status']=='captured'
    assert arm in ARMS and record['arm']==raw['arm']==ids_record['arm']==arm
    assert record['position']==raw['position']==ids_record['position']==item['original_position']
    assert record['index']==raw['index']==ids_record['index']==item['index']
    assert record['group_id']==raw['group_id']==ids_record['group_id']==item['group_id']
    assert raw['native_failure'] is None and raw['native_returned'] and ids_record['native_returned']
    assert isinstance(record['response'],str) and record['response']==raw['response']
    assert record['calls']==dict(encode=1,cross=1,greedy=1,prepare=1,clip=1) and not record['observer_errors']
    assert record['generate_scalar_kwargs']==PARAMETERS
    assert record['generate_keys']==sorted(['images','image_sizes','texts',*PARAMETERS]) and record['positional_args_count']==1
    assert record['image_sizes']==[item['image']['size']] and record['returned_visual_token_num']==128
    guidance=item['full_guidance'] if arm=='source' else item['question_only_guidance'];ref=item[arm+'_clip']
    assert record['guidance']==guidance and record['encode_texts']==[guidance] and text_sha(guidance)==ref['text_sha256']
    arrays=record['arrays'];small=record['small_arrays'];assert set(arrays)==EXPECTED_ARRAYS and set(small)==SMALL
    for m in arrays.values():check_metadata(m)
    ids=check_saved_small(small['actual_input_ids'],arrays['actual_input_ids'],'int64')
    assert ids.ndim==2 and ids.shape[0]==1 and ids.tolist()==[item['llm_input_ids']]
    assert array_sha(ids)==item['llm_input_ids_sha256'] and arrays['actual_generate_input_ids']==arrays['actual_input_ids']
    assert np.count_nonzero(ids[0]==-200)==1
    pix=arrays['actual_input_fp16'];check_metadata(pix,'float16',[1,3,336,336])
    assert pix['sha256']==item['image']['pixel_float16_sha256']
    length=ref['n_segments'];assert type(length) is int and length>0
    for k in ('clip_input_ids','clip_attention_mask'):
        check_saved_small(small[k],arrays[k],'int64',[length,77])
    check_clip(small['clip_input_ids'],small['clip_attention_mask'],ref)
    selected=check_saved_small(small['internal_ids'],arrays['internal_ids'],'int64',[1,128])
    quota=check_saved_small(small['quotas'],arrays['quotas'],'int64',[1])
    mask=check_saved_small(small['final_mask'],arrays['final_mask'],'bool',[1,576])
    check_selection(selected[0].tolist(),quota.tolist(),mask[0].tolist())
    output=check_saved_small(small['output_ids'],arrays['output_ids'],'int64')
    assert output.ndim==2 and output.shape[0]==1 and output.tolist()==raw['output_ids']==ids_record['output_ids']
    eos=runtime['eos_token_id'];eos=[eos] if isinstance(eos,int) else eos
    ending=check_output(output[0].tolist(),runtime['vocab_size'],runtime['bos_token_id'],eos)
    assert record['ending']==ending
    for key,shape in (('text_global',[length,768]),('text_dense',[length,77,768]),('image_embeds',[1,576,768]),
        ('global_sim',[1,576,length]),('dense_sim',[1,576,length,77]),('importance',[1,576]),('visual_sim',[1,576,576])):
        check_metadata(arrays[key],'float32',shape)
    check_metadata(arrays['projected_fp16'],'float16',[1,576,4096])
    check_metadata(arrays['prepared_embeddings_fp16'],'float16',[1,ids.shape[1]+127,4096])
    derived=record['derived'];assert set(derived)=={'selected_visual','prepare_visual','prepare_nonvisual','marker_index'}
    assert derived['marker_index']==ids[0].tolist().index(-200)
    check_visual_relationship(derived['selected_visual'],derived['prepare_visual'],derived['prepare_nonvisual'],ids.shape[1])
    return ending


def pair_comparison(source,variant):
    check_pair(source,variant)
    left,right=source['arrays'],variant['arrays']
    fields=('text_global','text_dense','global_sim','dense_sim','importance','internal_ids','final_mask','prepared_embeddings_fp16','output_ids')
    return {key+'_changed':left[key]!=right[key] for key in fields}


def check_manifest(c,manifest,panel):
    """Bind sanitized inputs to the saved no-GT CPU references and prior 60."""
    from PIL import Image
    mmb=c.HERE.parent;refs=mmb/'question_only_guidance_cpu_audit_20261030'
    assert manifest['n_rows']==4377 and manifest['n_groups']==1176 and manifest['all_rounds'] is False
    assert manifest['source_TSV_bytes_read_including_GT'] is True and manifest['GT_columns_parsed'] is False
    rows=manifest['rows'];assert len(rows)==4377 and len(panel)==60
    assert [r['original_position'] for r in rows]==list(range(4377)) and len({r['index'] for r in rows})==4377
    audit=[__import__('json').loads(x) for x in (refs/'token_fingerprints.jsonl').read_text().splitlines()]
    assert len(audit)==4377 and [x['index'] for x in audit]==[x['index'] for x in rows]
    groups={r['group_id'] for r in rows};assert len(groups)==1176
    assert all(item['group_id']==item['index']%1000000==ref['group'] for item,ref in zip(rows,audit))
    cpu=c.read(mmb/'cpu_inputs_current.json');pin=c.read(mmb/'cpu_inputs_author_pins.json')
    llm={r['index']:r for r in c.read(mmb/'cpu_llm_tokens_current.json')['records']}
    llm_pin={r['index']:r for r in c.read(mmb/'cpu_llm_tokens_author_pins.json')['records']}
    checked_images=set()
    for item,ref in zip(rows,audit):
        assert item['round_id']==0
        for arm,key in (('source','full_guidance'),('question_only','question_only_guidance')):
            expected=item[arm+'_clip'];original=ref[key]
            assert text_sha(item[key])==expected['text_sha256']==original['text_sha256']
            for field in ('n_tokens','n_segments','input_ids_sha256','attention_mask_sha256'):assert expected[field]==original[field]
            check_clip(expected['input_ids'],expected['attention_mask'],expected)
        inp=item['llm_input_ids'];assert all(type(x) is int for x in inp)
        assert int64_sha(inp)==item['llm_input_ids_sha256'] and len(inp)==item['llm_n_tokens']
        ref_llm=llm[item['index']];assert ref_llm==llm_pin[item['index']]
        assert ref_llm['token_ids_sha256']==item['llm_input_ids_sha256'] and ref_llm['n_tokens']==len(inp)
        assert ref_llm['prompt_sha256']==text_sha(item['llm_prompt'])
        image=item['image'];payload=image['payload_sha256'];assert payload==ref['image_payload_sha256']
        previous=cpu['images'][payload];assert previous==pin['images'][payload]
        for a,b in (('size','size'),('mode','mode'),('decoded_rgb_sha256','decoded_rgb_sha256'),('shape','pixel_shape'),('float16_sha256','pixel_float16_sha256'),('float32_sha256','pixel_float32_sha256')):
            assert previous[a]==image[b]
        image_key=(payload,image['path'])
        if image_key not in checked_images:
            assert c.sha(image['path'])==payload
            with Image.open(image['path']) as pic:
                rgb=pic.convert('RGB');assert list(rgb.size)==image['size'] and rgb.mode==image['mode']=='RGB'
                assert bytes_sha(rgb.tobytes())==image['decoded_rgb_sha256']
            checked_images.add(image_key)
    for group in groups:
        members=[x for x in rows if x['group_id']==group]
        assert len({x['question_only_guidance'] for x in members})==1
        assert len({x['image']['payload_sha256'] for x in members})==1
    anchors={x['original_position']:i for i,x in enumerate(panel)};assert len(anchors)==60
    for pos,i in anchors.items():
        old=panel[i];new=rows[pos]
        for key in ('index','group_id','full_guidance','question_only_guidance','llm_prompt','llm_input_ids','llm_input_ids_sha256','source_clip','question_only_clip'):
            assert old[key]==new[key],(pos,key)
        for key in ('payload_sha256','decoded_rgb_sha256','size','mode','pixel_shape','pixel_dtype','pixel_float16_sha256','pixel_float32_sha256'):
            assert old['image'][key]==new['image'][key],(pos,key)
    return rows,anchors


def described(c,entry,path):
    assert Path(entry['path'])==path and c.sha(path)==entry['sha256']


def load_case(c,description,arm,pos,item,runtime):
    label='%s_%04d'%(arm,pos);base=c.OUT/'cases';path=base/(label+'.json')
    described(c,description,path);record=c.read(path)
    paths={key:base/(label+suffix) for key,suffix in (('raw','.raw.json'),('output_ids_file','.output_ids.json'),('capture','.capture.json'))}
    for name,file in paths.items():described(c,record[name],file)
    raw=c.read(paths['raw']);ids=c.read(paths['output_ids_file']);capture=c.read(paths['capture'])
    assert capture['arm']==arm and capture['position']==pos and capture['index']==item['index'] and capture['group_id']==item['group_id']
    assert capture['raw']==record['raw'] and capture['output_ids_file']==record['output_ids_file']
    for key in ('arrays','small_arrays','calls','observer_errors'):assert capture[key]==record[key],key
    ending=check_case(record,raw,ids,item,arm,runtime)
    return record,raw,ending


def check_panel_anchor(c,record,arm,panel_index,panel_row):
    panel=c.HERE.parent/'question_only_guidance_panel';a=record['anchor']
    assert a['panel_index']==panel_index and a['all_arrays_bitwise_equal'] and a['output_equal']
    prior_record_path=panel/'execution'/('%s_%02d.json'%(arm,panel_index))
    described(c,a['prior_record'],prior_record_path);prior=c.read(prior_record_path)
    assert prior['index']==panel_row['index']==record['index'] and prior['arm']==arm and prior['position']==panel_index
    prior_path=panel/'execution'/('%s_%02d.npz'%(arm,panel_index))
    assert a['prior_archive']==prior['archive'];described(c,a['prior_archive'],prior_path)
    new_path=c.OUT/'anchors'/('%s_%02d.npz'%(arm,panel_index));described(c,a['archive'],new_path)
    check_full_anchor(new_path,record['arrays'],prior_path,prior['arrays'])
    prior_raw=c.read(prior['raw']['path']);described(c,prior['raw'],panel/'execution'/('%s_%02d.raw.json'%(arm,panel_index)))
    assert record['response']==prior_raw['response'] and record['small_arrays']['output_ids']==prior_raw['output_ids']


def terminal():
    import json,os,sys
    import common as c
    assert sys.flags.optimize==0 and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not any(Path('/dev',x).exists() for x in ('nvidia0','nvidiactl','nvidia-uvm'))
    plan=c.read(c.PLAN);c.check(plan);plan_sha=c.sha(c.PLAN)
    release=c.read(c.OUT/'native_release.json');claim=c.read(c.HERE/'launch.claim.json');worker=c.read(c.OUT/'worker_identity.json')
    assert release['status']=='native_exited_validation_pending' and release['native_exit_code']==0
    assert release['original_worker_exited'] and release['GPU_pid_absent'] and release['plan_sha256']==plan_sha
    assert worker['identity']==release['worker'] and worker['parent']==release['controller']==claim['controller']
    assert worker['plan_sha256']==claim['plan_sha256']==plan_sha
    assert worker['allocation_sha256']==release['allocation_sha256']==claim['allocation_sha256']
    complete=c.read(c.OUT/'native_complete.json')
    assert complete['status']=='complete' and complete['plan_sha256']==plan_sha
    assert complete['n_rows']==4377 and complete['n_generations']==8754 and not complete['scoring']
    described(c,complete['runtime'],c.OUT/'runtime.json');runtime=c.read(c.OUT/'runtime.json')
    assert runtime['plan_sha256']==plan_sha and runtime['one_loaded_model']
    assert runtime['model_path']==str(c.MODEL) and runtime['model_dtype']=='torch.float16' and runtime['device']=='cuda:0'
    assert runtime['attention_implementation']=='sdpa' and runtime['K']==128 and runtime['alpha']==.5 and runtime['beta']==1.
    assert runtime['torch']=='2.1.2+cu118' and runtime['transformers']=='4.37.2' and runtime['source_sha256']==c.sha(c.SOURCE)
    sys.path.insert(0,str(c.HERE.parent/'question_only_guidance_panel_design'))
    import source_boundary
    assert runtime['source_boundary_proof']==source_boundary.proof()
    previous_wait=c.read(c.HERE.parent/'streamwait_control/runtime_patch_validation.json')
    for key in ('source_sha256','original_forward_ast_sha256','patched_forward_ast_sha256','exact_added_statement_count','added_statements','restored_ast_identical'):
        assert runtime['wait_patch'][key]==previous_wait[key]
    assert runtime['wait_patch']['exact_added_statement_count']==3 and runtime['wait_patch']['restored_ast_identical']
    panel_dir=c.HERE.parent/'question_only_guidance_panel';old_proof=c.read(panel_dir/'root_exit_proof.json')
    assert old_proof['root_pty_actual_exit_code']==0 and old_proof['verifier_container_not_running']
    assert all(x['original_process_absent'] for x in old_proof['process_checks'])
    assert old_proof['state_sha256']==c.sha(panel_dir/'execution/state.json')
    assert old_proof['verifier_sha256']==c.sha(panel_dir/'execution/independent_verification.json')
    old_report=c.read(panel_dir/'execution/independent_verification.json');assert old_report['success'] and old_report['n_generations']==120
    panel=c.read(panel_dir/'inputs/manifest.json')['rows'];manifest=c.read(c.HERE/'inputs/manifest.json')
    rows,anchors=check_manifest(c,manifest,panel)
    described(c,complete['anchor_gate'],c.OUT/'anchor_gate.json');gate=c.read(c.OUT/'anchor_gate.json')
    assert gate['passed'] and gate['n_rows']==60 and gate['n_generations']==120 and gate['all_anchors_before_remaining']
    assert gate['plan_sha256']==plan_sha and gate['prior_panel_root_proof_sha256']==c.sha(panel_dir/'root_exit_proof.json')
    assert len(gate['pairs'])==60
    gate_pairs={row['original_position']:gate['pairs'][i] for i,row in enumerate(panel)}
    ordered={};prediction_sha={};observation_sha={}
    for arm in ARMS:
        info=complete['ordered'][arm];pred_path=c.OUT/(arm+'_predictions.jsonl');obs_path=c.OUT/(arm+'_observations.jsonl')
        described(c,info['predictions'],pred_path);described(c,info['observations'],obs_path)
        prediction_sha[arm]=info['predictions']['sha256'];observation_sha[arm]=info['observations']['sha256']
        predictions=[json.loads(x) for x in pred_path.read_text().splitlines()]
        observations=[json.loads(x) for x in obs_path.read_text().splitlines()]
        assert len(predictions)==len(observations)==4377
        assert [r['position'] for r in predictions]==[r['position'] for r in observations]==list(range(4377))
        assert [r['index'] for r in predictions]==[r['index'] for r in observations]==[r['index'] for r in rows]
        ordered[arm]=(predictions,observations)
    import torch
    from transformers import AutoTokenizer
    assert not torch.cuda.is_initialized()
    tokenizer=AutoTokenizer.from_pretrained(str(c.MODEL),use_fast=False,local_files_only=True)
    eos_counts={arm:dict(EOS_present=0,EOS_missing=0,at_max_tokens=0) for arm in ARMS}
    groups={};results=[]
    for pos,item in enumerate(rows):
        source_obs=ordered['source'][1][pos];variant_obs=ordered['question_only'][1][pos]
        assert source_obs['pair']==variant_obs['pair']
        pair_path=c.OUT/'pairs'/('%04d.json'%pos);described(c,source_obs['pair'],pair_path);pair=c.read(pair_path)
        assert pair['position']==pos and pair['index']==item['index'] and pair['direct_actual_bitwise_pair_passed']
        if pos in anchors:assert source_obs['pair']==gate_pairs[pos]
        recs={};ends={}
        for arm,observation in (('source',source_obs),('question_only',variant_obs)):
            assert observation['record']==pair[arm] and observation['position']==pos and observation['index']==item['index']
            record,raw,ending=load_case(c,observation['record'],arm,pos,item,runtime)
            prediction=ordered[arm][0][pos]
            assert prediction==dict(position=pos,index=item['index'],group_id=item['group_id'],text=record['response'],output_ids=raw['output_ids'])
            author_path=c.OUT/'author'/('%s_%04d.jsonl'%(arm,pos));described(c,pair[arm+'_author'],author_path)
            author=[json.loads(x) for x in author_path.read_text().splitlines()];assert len(author)==1
            assert author[0]['question_id']==item['index'] and author[0]['round_id']==0
            assert author[0]['prompt']==item['full_guidance'] and author[0]['text']==record['response']
            if pos in anchors:check_panel_anchor(c,record,arm,anchors[pos],panel[anchors[pos]])
            else:assert 'anchor' not in record
            output=record['small_arrays']['output_ids'];decoded=tokenizer.batch_decode(output,skip_special_tokens=True)[0].strip()
            assert decoded==record['response']
            eos_counts[arm]['EOS_present' if ending['EOS_present'] else 'EOS_missing']+=1
            eos_counts[arm]['at_max_tokens']+=int(ending['at_max_tokens'])
            recs[arm]=record;ends[arm]=ending
        assert pair_comparison(recs['source'],recs['question_only'])==pair['comparison']
        change=check_pair(recs['source'],recs['question_only'])
        results.append(dict(position=pos,index=item['index'],group_id=item['group_id'],comparison=change,endings=ends))
        group=groups.setdefault(str(item['group_id']),{a:[] for a in ARMS})
        for arm in ARMS:group[arm].append(recs[arm]['arrays']['final_mask']['sha256'])
    assert not torch.cuda.is_initialized() and len(results)==4377 and len(groups)==1176
    group_summary={arm:dict(constant=sum(len(set(g[arm]))==1 for g in groups.values()),
                            varied=sum(len(set(g[arm]))>1 for g in groups.values())) for arm in ARMS}
    for item in plan['files']:assert c.sha(item['path'])==item['sha256']
    target=c.OUT/'generation_verification.json';assert not target.exists()
    c.save_new(target,dict(success=True,n_rows=4377,n_generations=8754,n_anchor_rows=60,n_groups=1176,
        processor_selection_and_prefill_checks=8754,independent_token_redecodes=8754,
        plan_sha256=plan_sha,native_complete_sha256=c.sha(c.OUT/'native_complete.json'),native_release_sha256=c.sha(c.OUT/'native_release.json'),
        anchor_gate_sha256=c.sha(c.OUT/'anchor_gate.json'),runtime_sha256=c.sha(c.OUT/'runtime.json'),
        prediction_sha256=prediction_sha,observation_sha256=observation_sha,ending_counts=eos_counts,group_mask_summary=group_summary,results=results,
        CPU_only=True,model_loaded=False,CUDA_initialized=False,GT_read=False,scoring_started=False,benchmark_result=False,
        native_process_exit_recorded_by_controller=True,root_actual_controller_exit_required=True,
        high_dimensional_arrays_independently_reconstructed_for_nonanchors=False,
        limits=['Input preparation read original TSV bytes containing GT but did not parse or use its answer column; generation and this verifier read only sanitized inputs.',
                'Only 60 previously completed rows have actual saved historical feature/mask/output-ID arrays; the older full run did not retain those internals.',
                'Non-anchor high-dimensional tensors were retained only as runtime metadata; their contents cannot be independently reconstructed from SHA values.',
                'Guidance changes options, any hint and segmentation together; no single-cause or paper-argv claim follows.',
                'No ground truth or scoring is used in this generation audit.']))
    print(json.dumps(dict(success=True,n_rows=4377,n_generations=8754,independent_token_redecodes=8754)))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--terminal',action='store_true');options=parser.parse_args()
    if options.terminal:terminal()
