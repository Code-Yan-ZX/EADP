"""Prepared source-comment guidance probe. Default check cannot start a GPU."""
import argparse
from datetime import datetime,timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode=True
from contracts import ARMS,POSITIONS,guidance,sha,read,save,array_sha,equal_bits,selection,embeddings,require_all_sources,stat

HERE=Path(__file__).resolve().parent
NEXT=HERE.parent
PLAN=HERE/'plan.json'
OUT=HERE/'execution'
PYTHON='/home/dell/miniconda3/envs/llava_pruner/bin/python'
FULL=NEXT/'full_default_K64_public'
PANEL=NEXT/'gpu_formula_facility_panel'
UTC=lambda:datetime.now(timezone.utc).isoformat()


def identity(pid):
    try:return dict(pid=pid,start_ticks=int(Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]))
    except FileNotFoundError:return None


def gpu_pids():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return sorted({int(x.strip()) for x in raw.splitlines() if x.strip()})


def describe(path):
    path=Path(path)
    return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)


def check(plan):
    assert plan['run_id']=='NeXT_fixed16_K64_source_vs_question_only_guidance'
    assert plan['parameters']==dict(K=64,alpha=.5,beta=1.,temperature=0.,top_p=None,num_beams=1,max_new_tokens=128,conv_mode='vicuna_v1',use_cache=True)
    assert [r['full_question_position'] for r in plan['panel']]==POSITIONS and len({r['image'] for r in plan['panel']})==16
    for group in ['source_sha256','implementation_sha256','reference_sha256']:
        for path,digest in plan[group].items():assert sha(path)==digest,path
    for path,guard in plan['model_stat_guards'].items():assert stat(path)=={k:v for k,v in guard.items() if k!='prior_verified_sha256'},path
    for row in plan['panel']:
        assert sha(row['image_path'])==row['image_sha256'] and sha(row['cache']['path'])==row['cache']['sha256']
        assert guidance(row['prompt'],'question_only').strip()
    fixture=read(HERE/'source_anchor_fixture.json')
    assert [r['position'] for r in fixture['rows']]==POSITIONS
    for row,panel in zip(fixture['rows'],plan['panel']):
        assert row['question_id']==panel['question_id'] and row['prompt']==panel['prompt']
        obs=row['observation'];assert obs['position']==row['position'] and obs['input_shape']==[1,row['llm_token_reference']['n_tokens']]
        assert obs['input_ids_sha256']==row['llm_token_reference']['token_ids_sha256']
    assert 'torch' not in sys.modules


def prepare():
    assert not PLAN.exists() and not OUT.exists()
    prior=read(PANEL/'run_plan.json')
    assert sha(PANEL/'run_plan.json')=='6ea965865b8211d37c213352b57ff5dc5302d3950f918a0be518fa1bd5b418b7'
    assert [r['full_question_position'] for r in prior['panel']]==POSITIONS
    root=read(FULL/'root_terminal_process_proof.json')
    assert sha(FULL/'root_terminal_process_proof.json')=='100454d855cd853632a389693312cff5de9702fd5124a512a292e3ef3b16eae4'
    assert root['root_owned_exec_actual_exit_code']==0 and root['all_original_processes_exited']
    proof=read(FULL/'execution/independent_verification.json')
    assert proof['success'] and all(proof[k]==5000 for k in ['n_rows','token_decoding_checks','input_budget_observation_checks','per_question_score_checks'])
    assert sha(FULL/'execution/independent_verification.json')==root['independent_verification_sha256']
    assert sha(FULL/'execution/release.json')==root['release_sha256']
    for entry in proof['artifacts'].values():assert sha(entry['path'])==entry['sha256']
    def lines(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]
    answers=lines(FULL/'execution/answers.jsonl');observations=lines(FULL/'execution/input_budget_observations.jsonl');traces=lines(FULL/'execution/runtime.jsonl')
    assert len(answers)==len(observations)==len(traces)==5000
    references=read(NEXT/'llm_tokens_current.json')['rows'];assert references==read(NEXT/'llm_tokens_author_pins.json')['rows']
    caches=lines(PANEL/'execution/per_image.jsonl')
    fixture=[];panels=[]
    for i,panel in enumerate(prior['panel']):
        position=panel['full_question_position'];a,o,t=answers[position],observations[position],traces[position]
        assert a['prompt']==panel['prompt'] and a['question_id']==panel['question_id']==t['question_id']
        assert o['position']==t['position']==references[position]['position']==position
        assert caches[i]['panel']==panel
        panels.append(dict(panel,cache=dict(path=caches[i]['cache_path'],sha256=caches[i]['cache_sha256'])))
        fixture.append(dict(position=position,question_id=a['question_id'],prompt=a['prompt'],expected_source_text=a['text'],
            expected_source_output_token_ids=t['output_token_ids'],observation=o,llm_token_reference=references[position]))
    save(HERE/'source_anchor_fixture.json',dict(purpose='Fixed16 source consistency gate only; no score, GT or sample selection.',rows=fixture,
        accepted_full_K64_root_proof_sha256=sha(FULL/'root_terminal_process_proof.json'),source_raw=describe(FULL/'execution/answers.jsonl'),
        source_runtime=describe(FULL/'execution/runtime.jsonl'),source_observations=describe(FULL/'execution/input_budget_observations.jsonl')))
    reference_paths=[HERE/'source_anchor_fixture.json',PANEL/'run_plan.json',PANEL/'independent_verification.json',PANEL/'root_terminal_process_proof.json',PANEL/'execution/per_image.jsonl',
        FULL/'plan.json',FULL/'root_terminal_process_proof.json',FULL/'execution/independent_verification.json',FULL/'execution/release.json',
        NEXT/'llm_tokens_current.json',NEXT/'llm_tokens_author_pins.json',NEXT/'guidance_question_boundary_audit.json',NEXT/'guidance_question_boundary_exact_tokenization.json',NEXT/'guidance_question_boundary_exact_token_rows.jsonl',
        NEXT/'guidance_question_boundary_cache_feasibility.json',NEXT.parent/'pope/wait_only_patch.py']
    implementation=[HERE/'probe.py',HERE/'contracts.py',HERE/'cpu_acceptance.py',HERE/'README.md']
    plan=dict(run_id='NeXT_fixed16_K64_source_vs_question_only_guidance',prepared_utc=UTC(),status='prepared_not_launched',python=PYTHON,
        parameters=dict(K=64,alpha=.5,beta=1.,temperature=0.,top_p=None,num_beams=1,max_new_tokens=128,conv_mode='vicuna_v1',use_cache=True),
        llava_root=prior['llava_root'],pope_root=prior['pope_root'],model_path=prior['model_path'],
        source_sha256=prior['source_sha256'],implementation_sha256={str(p):sha(p) for p in implementation},reference_sha256={str(p):sha(p) for p in reference_paths},
        model_stat_guards=prior['model_stat_guards'],expected_component_dtypes=prior['expected_dtype'],panel=panels,
        panel_selection='Exactly prior NeXT16 uniform panel positions, unchanged; no answers/GT used to choose cases.',
        arms=list(ARMS),arm_order='All16 source complete and anchored before any16 question_only generation.',
        source_anchors=['Native FP16 pixels/projected and FP32global/dense/visualsim exactly match old native16 cache, with K-independent beta1 importance anchor.',
            'Actual complete LLM input IDs hash, crop quotas/effective-prefix/finalmask and source output token IDs/text exactly match qualified fullK64 corresponding16 records.'],
        sole_intervention='Only texts= string: source removes exact answer suffix; question_only first split at newline Reference OCR token then same suffix removal. Full LLM OCR conversation, images, original native model/selection/generation arithmetic remain.',
        observations='Pass-through tensor-reference observers; no forward tensor clone/copy/read. All tensor content observation and serialization happens after native generate returns.',
        saved_new_arrays=['projected_fp16','image_embeds_fp32','text_global_fp32','text_dense_fp32','global_fp32','dense_fp32','visual_sim_fp32','importance_fp32','internal_ids','quotas','final_mask','actual_input_fp16','actual_LLM_input_ids','prepared_embeddings_fp16'],
        whole_feature_equality='Both arm real native projected,image_embeds,visualsim and pixels must be FP16/FP32 bit-identical. Nonvisual embedding prefix/suffix and complete LLM input IDs must be identical despite different visual counts.',
        source_wait='Declared existing AST-only3caller-stream wait patch; no other source or numerical change.',
        backend='sdpa',dtype='torch.float16',source_comment='Fresh model_vqa_loader.py:113 inactive split; author-supported input hypothesis, not a unique paper recipe or a predeclared bug fix.',
        no_GT_or_accuracy=True,source_accepted_predictions_used_only_for_anchor=True,
        launch_policy=dict(prepared_only=True,root_GPU_allocation_required=True,single_attempt=True,automatic_retry=False,queue_or_watcher=False,max_GPU_models=2),
        limits=['One fixed16 panel, no GT/accuracy/benchmark result; raw answer/ID changes are mechanism observations only.',
            'Source comment is inactive. Paper sameP andM<=77 do not identify this as the table-generating TextVQA boundary.',
            'Source anchors compare independent prior executions; any anchor failure is retained and stops before variant, no automatic retry or relaxed equality.',
            'Native unchanged chunks/padding/q.2/epsilon/global five-crop normalization/quota floor/min1/facility remain, no exact320 fill.',
            'Outputs may reach128new tokens withoutEOS. Always record BOS/EOS/returned length; same outputs within this horizon cannot exclude logit/later differences.',
            'Text is decoded by the worker native tokenizer. The independent CPU gate checks recorded IDs/source equality and horizon metadata but does not independently re-decode text.',
            'New captures are observations and timing is not an efficiency benchmark.'])
    save(PLAN,plan);check(plan)
    save(HERE/'READY.json',dict(status='prepared_not_launched',plan_sha256=sha(PLAN),GPU_started=False,model_loaded=False,queue_started=False))
    print(json.dumps(read(HERE/'READY.json')))


def allocation(note,plan):
    assert note['authorization']=='explicit_root_gpu_allocation' and note['run_id']==plan['run_id']
    assert note['plan_sha256']==sha(PLAN) and note['GPU_slot'] in ['A','B'] and note['GPU_device']==0
    assert note['single_attempt'] is True and note['automatic_retry'] is False
    assert note['CPU_acceptance_sha256']==sha(HERE/'cpu_acceptance.json')
    cpu=read(HERE/'cpu_acceptance.json');assert cpu['success'] and cpu['plan_sha256']==sha(PLAN) and cpu['GPU_started'] is False
    assert note['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    others=note['allowed_other_gpu_processes'];assert len(others)<2
    assert gpu_pids()==sorted(x['pid'] for x in others)
    assert all(identity(x['pid'])==x for x in others)


def worker(note_path):
    plan=read(PLAN);check(plan);note=read(note_path);claim=read(HERE/'launch.claim.json')
    assert sha(PLAN)==claim['plan_sha256']==note['plan_sha256'] and sha(note_path)==claim['allocation_note_sha256']
    assert identity(os.getppid())==claim['controller'];allocation(note,plan)
    OUT.mkdir(exist_ok=False)
    save(OUT/'process_identity.json',dict(**identity(os.getpid()),session_id=os.getsid(0),started_utc=UTC(),plan_sha256=sha(PLAN),allocation_note_sha256=sha(note_path)))
    import numpy as np
    import torch
    from PIL import Image
    assert not torch.cuda.is_initialized()
    assert torch.__version__=='2.1.2+cu118' and __import__('transformers').__version__=='4.37.2'
    sys.path.insert(0,plan['llava_root']);sys.path.insert(0,plan['pope_root'])
    from wait_only_patch import install
    save(OUT/'wait_patch_proof.json',install())
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path,process_images,tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX,DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.utils import disable_torch_init
    disable_torch_init()
    tokenizer,model,processor,_=load_pretrained_model(plan['model_path'],None,get_model_name_from_path(plan['model_path']),visual_token_num=64,alpha=.5,beta=1.)
    assert model.visual_token_num==64 and model.alpha==.5 and model.beta==1. and str(model.dtype)=='torch.float16' and model.config._attn_implementation=='sdpa'
    assert model.config.mm_patch_merge_type=='spatial_unpad' and model.config.image_aspect_ratio=='anyres' and not model.config.mm_use_im_start_end
    assert not torch.is_autocast_enabled() and not torch.backends.cuda.matmul.allow_tf32 and torch.get_default_dtype()==torch.float32
    tower=model.get_vision_tower()
    dtype=lambda m:sorted({str(p.dtype) for p in m.parameters()})
    dtypes={k:dtype(m) for k,m in dict(vision_body=tower.vision_tower.vision_model,visual_projection=tower.vision_tower.visual_projection,text_tower=tower.text_tower,mm_projector=model.get_model().mm_projector).items()}
    assert dtypes==plan['expected_component_dtypes']
    for function in [model.generate,model.prepare_inputs_labels_for_multimodal,model.encode_images,model.sim_cross,model.greed_select]:
        path=inspect.getfile(inspect.unwrap(function.__func__));assert sha(path)==plan['source_sha256'][path]
    save(OUT/'loaded_runtime.json',dict(model_class=type(model).__module__+'.'+type(model).__name__,backend=model.config._attn_implementation,model_dtype=str(model.dtype),component_dtypes=dtypes,
        alpha=model.alpha,beta=model.beta,K=model.visual_token_num,torch=torch.__version__,transformers=__import__('transformers').__version__,
        bos_token_id=model.generation_config.bos_token_id,eos_token_id=model.generation_config.eos_token_id,plan_sha256=sha(PLAN),worker_sha256=sha(__file__),
        observer_policy='Original bound methods called once; only references retained; tensor reads after generate.'))
    native={k:getattr(model,k) for k in ['encode_images','sim_cross','greed_select','prepare_inputs_labels_for_multimodal']}
    capture={};calls={};current_guidance=None
    def encode(*args,**kwargs):
        calls['encode']+=1
        assert kwargs.get('texts')==current_guidance and kwargs.get('split_sizes')==[5]
        value=native['encode_images'](*args,**kwargs);capture['projected_fp16'],capture['final_mask']=value;return value
    def cross(*args,**kwargs):
        calls['cross']+=1;capture['text_global_fp32'],capture['text_dense_fp32'],capture['image_embeds_fp32']=args[:3]
        value=native['sim_cross'](*args,**kwargs);capture['global_fp32'],capture['dense_fp32']=value;return value
    def greedy(*args,**kwargs):
        calls['greedy']+=1;capture['importance_fp32'],capture['visual_sim_fp32']=args[:2]
        value=native['greed_select'](*args,**kwargs);capture['internal_ids'],capture['quotas']=value;return value
    def prepare_inputs(*args,**kwargs):
        result=native['prepare_inputs_labels_for_multimodal'](*args,**kwargs)
        if len(result)==7:
            calls['full_prepare']+=1;capture['actual_LLM_input_ids']=args[0];capture['prepared_embeddings_fp16']=result[4]
        return result
    model.encode_images=encode;model.sim_cross=cross;model.greed_select=greedy;model.prepare_inputs_labels_for_multimodal=prepare_inputs
    fixture=read(HERE/'source_anchor_fixture.json')['rows'];all_records=[]
    source_records=[]
    for arm in ARMS:
        if arm=='question_only':
            require_all_sources(source_records)
            save(OUT/'all16_source_anchor_gate.json',dict(passed=True,n_sources=16,all_sources_before_variant=True,
                source_records=[describe(OUT/('source_%02d.json'%i)) for i in range(16)]))
        for index,(panel,anchor) in enumerate(zip(plan['panel'],fixture)):
            image=Image.open(panel['image_path']).convert('RGB')
            pixels=process_images([image],processor,model.config)
            assert isinstance(pixels,torch.Tensor) and tuple(pixels.shape)==(1,5,3,336,336)
            conv=conv_templates['vicuna_v1'].copy();conv.append_message(conv.roles[0],DEFAULT_IMAGE_TOKEN+'\n'+panel['prompt']);conv.append_message(conv.roles[1],None)
            cpu_ids=tokenizer_image_token(conv.get_prompt(),tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0)
            assert array_sha(cpu_ids.numpy())==anchor['observation']['input_ids_sha256'] and list(cpu_ids.shape)==anchor['observation']['input_shape']
            # Match the published loader's nonblocking H2D inputs, with only the
            # separately disclosed caller-stream dependencies in its CLIP tower.
            input_ids=cpu_ids.to(device='cuda',non_blocking=True)
            image_tensors=pixels.to(dtype=torch.float16,device='cuda',non_blocking=True)
            current_guidance=guidance(panel['prompt'],arm);capture.clear();calls=dict(encode=0,cross=0,greedy=0,full_prepare=0)
            started=time.monotonic()
            with torch.inference_mode():
                output,retained=model.generate(input_ids,images=image_tensors,image_sizes=[image.size],texts=current_guidance,
                    do_sample=False,temperature=0.,top_p=None,num_beams=1,max_new_tokens=128,use_cache=True)
            # No observation of any captured GPU tensor's content before here.
            assert calls==dict(encode=1,cross=1,greedy=1,full_prepare=1)
            arrays={k:v.detach().cpu().numpy() for k,v in capture.items()}
            arrays['actual_input_fp16']=image_tensors.detach().cpu().numpy()
            assert all(np.isfinite(v).all() for v in arrays.values())
            assert arrays['projected_fp16'].shape==(5,576,4096) and arrays['projected_fp16'].dtype==np.float16
            assert arrays['image_embeds_fp32'].shape==(5,576,768) and arrays['image_embeds_fp32'].dtype==np.float32
            assert all(arrays[k].dtype==np.float32 for k in ['text_global_fp32','text_dense_fp32','global_fp32','dense_fp32','visual_sim_fp32','importance_fp32'])
            assert equal_bits(arrays['actual_LLM_input_ids'],cpu_ids.numpy())
            observed=selection(arrays['final_mask'],arrays['internal_ids'],arrays['quotas'])
            assert int(retained)==observed['returned_retained_tokens']
            nonvisual,embedded=embeddings(arrays['prepared_embeddings_fp16'],arrays['projected_fp16'],arrays['final_mask'],arrays['actual_LLM_input_ids'])
            tokens=output.detach().cpu().tolist();text=tokenizer.batch_decode(output,skip_special_tokens=True)[0].strip()
            assert len(tokens)==1 and tokens[0][0]==model.generation_config.bos_token_id and 1<len(tokens[0])<=129
            source_anchor=False
            if arm=='source':
                with np.load(panel['cache']['path'],allow_pickle=False) as old:
                    for name in ['projected_fp16','global_fp32','dense_fp32','visual_sim_fp32','actual_input_fp16']:
                        assert equal_bits(arrays[name],old[name]),('old_native_feature_anchor',index,name)
                    assert equal_bits(arrays['importance_fp32'],old['native_importance'])
                for k in ['greedy_quotas','effective_mask_indices','greedy_effective_prefixes','returned_retained_tokens']:
                    assert observed[k]==anchor['observation'][k],('fullK64_selection_anchor',index,k)
                assert tokens==anchor['expected_source_output_token_ids'] and text==anchor['expected_source_text'],('fullK64_output_anchor',index)
                source_anchor=True
            else:
                source=read(OUT/('source_%02d.json'%index))
                with np.load(source['arrays']['path'],allow_pickle=False) as first:
                    for name in ['projected_fp16','image_embeds_fp32','visual_sim_fp32','actual_input_fp16','actual_LLM_input_ids']:
                        assert equal_bits(arrays[name],first[name]),('only_guidance_may_change',index,name)
                    old_nonvisual,_=embeddings(first['prepared_embeddings_fp16'],first['projected_fp16'],first['final_mask'],first['actual_LLM_input_ids'])
                    assert equal_bits(nonvisual,old_nonvisual),('nonvisual_embeddings',index)
            path=OUT/('%s_%02d.npz'%(arm,index));np.savez_compressed(path,**arrays)
            record=dict(arm=arm,index=index,position=panel['full_question_position'],question_id=panel['question_id'],
                full_LLM_prompt_sha256=hashlib.sha256(panel['prompt'].encode()).hexdigest(),guidance_sha256=hashlib.sha256(current_guidance.encode()).hexdigest(),
                calls=calls,post_generation_tensor_reads_only=True,all_source_anchors_passed=source_anchor,
                selection=observed,embedded_input_contract=embedded,actual_LLM_input_ids_sha256=array_sha(arrays['actual_LLM_input_ids']),
                output_token_ids=tokens,text=text,returned_tokens=len(tokens[0]),initial_BOS_tokens=1,new_tokens=len(tokens[0])-1,
                EOS_present=model.generation_config.eos_token_id in tokens[0],generation_horizon=128,
                arrays=describe(path),tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=array_sha(v)) for k,v in arrays.items()},
                elapsed_seconds=time.monotonic()-started,timing_is_efficiency_measurement=False)
            save(OUT/('%s_%02d.json'%(arm,index)),record);all_records.append(record)
            if arm=='source':source_records.append(record)
            print(json.dumps(dict(arm=arm,completed_index=index,position=record['position'],all_source_anchors_passed=source_anchor)),flush=True)
    assert len(all_records)==32;require_all_sources(source_records)
    # Terminal guards repeat after all observations; do not call check(), whose
    # CPU-only import gate correctly rejects an already-loaded torch module.
    for group in ['source_sha256','implementation_sha256','reference_sha256']:
        assert all(sha(p)==v for p,v in plan[group].items())
    assert all(stat(p)=={k:v for k,v in g.items() if k!='prior_verified_sha256'} for p,g in plan['model_stat_guards'].items())
    assert all(sha(r['image_path'])==r['image_sha256'] and sha(r['cache']['path'])==r['cache']['sha256'] for r in plan['panel'])
    save(OUT/'native_complete.json',dict(success=True,n_generations=32,source_generations=16,variant_generations=16,all16_source_gate_passed=True,
        records=all_records,plan_sha256=sha(PLAN),GT_or_accuracy_used=False,benchmark_result=False))


def verify():
    import numpy as np
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch' not in sys.modules
    state=read(HERE/'state.json');assert state['status']=='native_exited' and state['native_exit_code']==0 and not (HERE/'failure.json').exists()
    assert identity(state['native_worker']['pid'])!=state['native_worker'] and state['native_worker']['pid'] not in gpu_pids()
    plan=read(PLAN);check(plan);complete=read(OUT/'native_complete.json');assert complete['success'] and complete['n_generations']==32
    records=complete['records'];assert [r['arm'] for r in records]==['source']*16+['question_only']*16
    require_all_sources(records[:16]);assert read(OUT/'all16_source_anchor_gate.json')['passed']
    fixture=read(HERE/'source_anchor_fixture.json')['rows'];changes=[]
    generation=read(Path(plan['model_path'])/'generation_config.json');config=read(Path(plan['model_path'])/'config.json')
    assert generation['bos_token_id']==config['bos_token_id']==1 and generation['eos_token_id']==config['eos_token_id']==2
    loaded=read(OUT/'loaded_runtime.json');assert loaded['bos_token_id']==1 and loaded['eos_token_id']==2
    for index,(panel,anchor) in enumerate(zip(plan['panel'],fixture)):
        pair=[]
        for arm in ARMS:
            record=read(OUT/('%s_%02d.json'%(arm,index)));assert record==records[(0 if arm=='source' else 16)+index]
            ids=record['output_token_ids'];assert len(ids)==1 and ids[0][0]==1 and 1<len(ids[0])<=129
            assert all(type(x) is int and 0<=x<config['vocab_size'] for x in ids[0])
            assert record['returned_tokens']==len(ids[0]) and record['initial_BOS_tokens']==1 and record['new_tokens']==len(ids[0])-1<=128
            assert record['generation_horizon']==128 and record['EOS_present']==(2 in ids[0])
            assert sha(record['arrays']['path'])==record['arrays']['sha256']
            with np.load(record['arrays']['path'],allow_pickle=False) as data:
                arrays={k:data[k] for k in data.files}
            assert all(list(v.shape)==record['tensors'][k]['shape'] and str(v.dtype)==record['tensors'][k]['dtype'] and array_sha(v)==record['tensors'][k]['sha256'] and np.isfinite(v).all() for k,v in arrays.items())
            assert selection(arrays['final_mask'],arrays['internal_ids'],arrays['quotas'])==record['selection']
            nv,em=embeddings(arrays['prepared_embeddings_fp16'],arrays['projected_fp16'],arrays['final_mask'],arrays['actual_LLM_input_ids'])
            assert em==record['embedded_input_contract'] and array_sha(arrays['actual_LLM_input_ids'])==anchor['observation']['input_ids_sha256']
            assert record['guidance_sha256']==hashlib.sha256(guidance(panel['prompt'],arm).encode()).hexdigest()
            if arm=='source':
                assert record['output_token_ids']==anchor['expected_source_output_token_ids'] and record['text']==anchor['expected_source_text']
                assert all(record['selection'][k]==anchor['observation'][k] for k in record['selection'])
                with np.load(panel['cache']['path'],allow_pickle=False) as old:
                    assert all(equal_bits(arrays[k],old[k]) for k in ['projected_fp16','global_fp32','dense_fp32','visual_sim_fp32','actual_input_fp16'])
                    assert equal_bits(arrays['importance_fp32'],old['native_importance'])
            pair.append((record,arrays,nv))
        (s,sa,sn),(v,va,vn)=pair
        assert all(equal_bits(sa[k],va[k]) for k in ['projected_fp16','image_embeds_fp32','visual_sim_fp32','actual_input_fp16','actual_LLM_input_ids']) and equal_bits(sn,vn)
        changes.append(dict(index=index,position=panel['full_question_position'],quotas_changed=s['selection']['greedy_quotas']!=v['selection']['greedy_quotas'],
            final_masks_changed=not equal_bits(sa['final_mask'],va['final_mask']),effective_prefixes_changed=s['selection']['greedy_effective_prefixes']!=v['selection']['greedy_effective_prefixes'],
            added_by_crop=[np.flatnonzero(va['final_mask'][c]&~sa['final_mask'][c]).tolist() for c in range(5)],removed_by_crop=[np.flatnonzero(sa['final_mask'][c]&~va['final_mask'][c]).tolist() for c in range(5)],
            output_token_IDs_changed=s['output_token_ids']!=v['output_token_ids'],raw_text_changed=s['text']!=v['text'],source_returned_tokens=s['returned_tokens'],variant_returned_tokens=v['returned_tokens'],source_EOS=s['EOS_present'],variant_EOS=v['EOS_present']))
    check(plan)
    save(OUT/'independent_verification.json',dict(success=True,n_generations=32,source_anchors_verified=16,variant_inputs_verified=16,
        full_LLM_IDs_and_nonvisual_embeddings_unchanged=True,real_projected_image_embeds_visualsim_pixels_equal=True,
        source_native_feature_and_fullK64_selection_output_anchors=True,changes=changes,plan_sha256=sha(PLAN),native_complete_sha256=sha(OUT/'native_complete.json'),
        artifacts=[describe(OUT/('%s_%02d.%s'%(arm,i,ext))) for arm in ARMS for i in range(16) for ext in ['json','npz']],
        model_loaded=False,CPU_only=True,GT_or_accuracy_used=False,new_benchmark_row=False,
        independent_token_redecode_performed=False,text_decoding_scope='Worker native tokenizer only; independent ID/source/horizon checks, no separate tokenizer decode.',
        limits=plan['limits'],root_actual_controller_exit_required=True))
    print(json.dumps(dict(success=True,n_generations=32,source_anchors=16,variant_inputs=16,benchmark_result=False)))


def run(note_path):
    plan=read(PLAN);check(plan);note=read(note_path);allocation(note,plan)
    assert Path(sys.executable).resolve()==Path(PYTHON).resolve() and not OUT.exists()
    controller=identity(os.getpid())
    save(HERE/'launch.claim.json',dict(controller=controller,plan_sha256=sha(PLAN),allocation_note_sha256=sha(note_path),automatic_retry=False))
    state=dict(status='starting',controller=controller,started_utc=UTC(),plan_sha256=sha(PLAN))
    def state_save(): (HERE/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    for key in list(env):
        if key.startswith(('USE_LLAVA_ARCH_','ANCHORZIP_','EADP_','ENTROPY_')):env.pop(key)
    try:
        with (HERE/'generation.log').open('x') as log:
            p=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'worker','--allocation-note',str(note_path.resolve())],env=env,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='generating',native_worker=identity(p.pid));state_save();assert state['native_worker']
            code=p.wait()
        state.update(status='native_exited',native_exit_code=code);state_save();assert code==0
        assert identity(p.pid)!=state['native_worker'] and p.pid not in gpu_pids()
        with (HERE/'verification.log').open('x') as log:
            v=subprocess.run([PYTHON,str(Path(__file__).resolve()),'verify'],env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
        assert v.returncode==0 and read(OUT/'independent_verification.json')['success'];check(plan)
        save(OUT/'release.json',dict(success=True,n_generations=32,native_worker=state['native_worker'],controller=controller,actual_native_exited=True,
            native_exit_code=0,plan_sha256=sha(PLAN),independent_verification_sha256=sha(OUT/'independent_verification.json'),controller_actual_exit_requires_root_proof=True,automatic_retry=False))
        state.update(status='complete',finished_utc=UTC(),release_sha256=sha(OUT/'release.json'));state_save()
        print(json.dumps(dict(status='complete',n_generations=32,GT_or_accuracy_used=False)))
    except BaseException as error:
        state.update(status='terminal_failure_no_retry',error=repr(error));state_save();save(HERE/'failure.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','check','run','worker','verify'],nargs='?',default='check');parser.add_argument('--allocation-note',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='check':check(read(PLAN));print(json.dumps(dict(CPU_guards=True,model_loaded=False,GPU_started=False)))
    elif args.action=='verify':verify()
    elif args.action=='worker':assert args.allocation_note is not None;worker(args.allocation_note)
    else:assert args.allocation_note is not None;run(args.allocation_note)
