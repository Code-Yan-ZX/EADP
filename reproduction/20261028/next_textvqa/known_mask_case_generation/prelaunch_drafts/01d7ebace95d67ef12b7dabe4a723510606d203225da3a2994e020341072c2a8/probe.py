"""Prepare/check on CPU. One known-case pair requires a later root allocation."""
import argparse
from datetime import datetime,timezone
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode=True
from cache_contract import OUT,NEXT,PANEL,ARMS,sha,load,array_sha,verify_embeddings

PLAN=OUT/'plan.json'
EXEC=OUT/'execution'
PYTHON='/home/dell/miniconda3/envs/llava_pruner/bin/python'
K64_PLAN_SHA='9ef9b4c69727d7baf324c5e3b3d93fd8c7d23a899d466f35e8c2389f05405d0d'
K64_NOTE_SHA='77602cb7ef1ffba87aee01bebbd2dbe54c5e02dcf7f3b890f81200a07c5fe204'
read=lambda p:json.loads(Path(p).read_text())
save=lambda p,d:Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
utc=lambda:datetime.now(timezone.utc).isoformat()


def identity(pid):
    try:return dict(pid=pid,start_ticks=int(Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]))
    except FileNotFoundError:return None


def gpu_pids():
    value=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return sorted({int(x.strip()) for x in value.splitlines() if x.strip()})


def stat(path):
    p=Path(path);s=p.stat()
    return dict(resolved=str(p.resolve()),bytes=s.st_size,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def check(plan):
    assert plan['run_id']=='NeXT_known3632_cached_source_vs_Eq6' and plan['arms']==ARMS
    assert plan['parameters']==dict(K=128,alpha=.5,beta=2.,max_new_tokens=128,temperature=0.,top_p=None,num_beams=1,conv_mode='vicuna_v1',use_cache=True)
    for path,digest in plan['source_sha256'].items():assert sha(path)==digest,path
    for path,guard in plan['model_stat_guards'].items():assert stat(path)=={k:v for k,v in guard.items() if k!='prior_verified_sha256'},path
    for entry in plan['frozen_evidence']:assert sha(entry['path'])==entry['sha256'],entry['path']
    _,contract=load();assert contract==plan['cache_contract']
    assert 'torch' not in sys.modules


def prepare():
    assert not PLAN.exists() and not EXEC.exists()
    base=read(PANEL/'run_plan.json');_,contract=load()
    question=contract['panel']
    ref=read(NEXT/'llm_tokens_current.json')['rows'][3632]
    assert str(ref['question_id'])==str(question['question_id']) and ref==read(NEXT/'llm_tokens_author_pins.json')['rows'][3632]
    from PIL import Image
    with Image.open(question['image_path']) as im:size=list(im.size)
    assert sha(question['image_path'])==question['image_sha256']
    paths=[Path(__file__),OUT/'cache_contract.py',OUT/'cpu_acceptance.py',PANEL/'run_plan.json',PANEL/'independent_verification.json',
        PANEL/'root_terminal_process_proof.json',PANEL/'causal_selection_case.json',NEXT/'llm_tokens_current.json',NEXT/'llm_tokens_author_pins.json']
    proof=read(PANEL/'independent_verification.json');assert proof['success'] and proof['source_anchor_images_exact']==16
    plan=dict(run_id='NeXT_known3632_cached_source_vs_Eq6',prepared_utc=utc(),status='prepared_not_launched',
        purpose='Post-hoc known-case mechanism probe only; no GT, historical answers, accuracy or benchmark claim.',
        arms=ARMS,parameters=dict(K=128,alpha=.5,beta=2.,max_new_tokens=128,temperature=0.,top_p=None,num_beams=1,conv_mode='vicuna_v1',use_cache=True),
        python=PYTHON,llava_root=base['llava_root'],model_path=base['model_path'],source_sha256=base['source_sha256'],
        model_stat_guards=base['model_stat_guards'],cache_capture_runtime=read(PANEL/'execution/loaded_runtime.json'),
        cache_contract=contract,input_reference=ref,image_size=size,
        frozen_evidence=[dict(path=str(p.resolve()),sha256=sha(p)) for p in paths],
        intervention='Return the same exact cached native projected_fp16 from encode_images, with only the frozen source/Eq6 mask switching. Native prepare/generate bodies unchanged; no fresh CLIP/projector forward or selection.',
        observations='All GPU->CPU copies and prepared-input observations after native generate returns. Exactly2 generation calls,1encode/fullprepare per arm,0CLIP/projector forwards. Save actual complete inputs_embeds for independent CPU comparison.',
        launch_policy=dict(single_attempt=True,automatic_retry=False,root_allocation_required=True,after_K64_full_terminal=True,max_GPU_models=2,watcher_started=False),
        predecessor_root_proof=dict(path=str(NEXT/'full_default_K64_public/root_terminal_process_proof.json'),
            expected_root_owned_exec_handle=63618,plan_sha256=K64_PLAN_SHA,allocation_note_sha256=K64_NOTE_SHA,
            required_keys=['root_owned_exec_handle','root_owned_exec_actual_exit_code','all_original_processes_exited',
                           'controller_identity','native_identity','plan_sha256','allocation_note_sha256',
                           'release_sha256','independent_verification_sha256']),
        limitations=['The original native end-to-end feature/selection anchor was beta1/K128; beta2 source mask is unchanged-source-helper replay on those same native features, not another beta2 extraction anchor.',
            'Only this already discovered case, not a new candidate search or unbiased sample. Source versus Eq6 masks share637 tokens/quotas, but the native spatial ordering moves83 visual sequence rows.',
            'Text difference tests propagation in this case only. Same text does not rule out logit changes; neither outcome estimates accuracy, paper-gap cause or full5000 behavior.'],
        command=[PYTHON,str(Path(__file__).resolve()),'run','--allocation-note','FUTURE_ROOT_ALLOCATION.json'])
    save(PLAN,plan);check(plan)
    save(OUT/'READY.json',dict(status='prepared_not_launched',plan_sha256=sha(PLAN),GPU_started=False,model_loaded=False,watcher_started=False))
    print(json.dumps(read(OUT/'READY.json')))


def validate_k64_root(root,state,claim,release_sha,report_sha):
    assert root['root_owned_exec_handle']==63618 and root['root_owned_exec_actual_exit_code']==0
    assert root['all_original_processes_exited'] is True
    assert root['controller_identity']==state['controller']==claim['controller']==dict(pid=2734202,start_ticks=1332948026)
    assert root['native_identity']==state['native_worker']==dict(pid=2734206,start_ticks=1332948051)
    assert root['plan_sha256']==claim['plan_sha256']==K64_PLAN_SHA
    assert root['allocation_note_sha256']==claim['allocation_note_sha256']==K64_NOTE_SHA
    assert root['release_sha256']==release_sha and root['independent_verification_sha256']==report_sha


def k64_terminal_gate():
    directory=NEXT/'full_default_K64_public';state=read(directory/'controller_state.json')
    assert not (directory/'failure.json').exists() and state['status']=='complete' and state['native_exit_code']==0
    release=read(directory/'execution/release.json');proof=read(directory/'execution/independent_verification.json')
    assert sha(directory/'plan.json')==release['plan_sha256']==K64_PLAN_SHA
    assert sha(directory/'execution/release.json')==state['release_sha256']
    assert proof['success'] and proof['n_rows']==proof['token_decoding_checks']==proof['input_budget_observation_checks']==proof['per_question_score_checks']==5000
    assert sha(directory/'execution/independent_verification.json')==release['independent_verification_sha256']==state['independent_verification_sha256']
    assert sha(directory/'execution/official.score.json')==release['official_score_sha256']==proof['official_score_sha256']
    root=read(directory/'root_terminal_process_proof.json');claim=read(directory/'launch.claim.json')
    assert read(directory/'allocation.json')['note_sha256']==K64_NOTE_SHA
    validate_k64_root(root,state,claim,sha(directory/'execution/release.json'),sha(directory/'execution/independent_verification.json'))
    for original in [state['controller'],state['native_worker']]:
        assert identity(original['pid'])!=original and original['pid'] not in gpu_pids()
    return dict(controller_state_sha256=sha(directory/'controller_state.json'),release_sha256=sha(directory/'execution/release.json'),
                independent_verification_sha256=sha(directory/'execution/independent_verification.json'),
                root_terminal_process_proof_sha256=sha(directory/'root_terminal_process_proof.json'),actual_controller_and_native_exited=True,root_owned_exec_actual_exit_code=0)


def worker(note_path):
    plan=read(PLAN);check(plan);note=read(note_path);claim=read(OUT/'launch.claim.json')
    assert note['plan_sha256']==sha(PLAN)==claim['plan_sha256'] and sha(note_path)==claim['note_sha256']
    assert os.getppid()==claim['controller']['pid']
    EXEC.mkdir(exist_ok=False)
    import numpy as np
    import torch
    sys.path.insert(0,plan['llava_root'])
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path,tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX,DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.utils import disable_torch_init
    disable_torch_init()
    tokenizer,model,_,_=load_pretrained_model(plan['model_path'],None,get_model_name_from_path(plan['model_path']),visual_token_num=128,alpha=.5,beta=2.)
    assert model.alpha==.5 and model.beta==2. and model.visual_token_num==128 and model.config._attn_implementation=='sdpa'
    assert str(model.dtype)=='torch.float16' and torch.__version__=='2.1.2+cu118'
    assert __import__('transformers').__version__=='4.37.2' and not torch.is_autocast_enabled() and not torch.backends.cuda.matmul.allow_tf32
    assert model.config.mm_patch_merge_type=='spatial_unpad' and model.config.image_aspect_ratio=='anyres'
    assert not model.config.mm_use_im_start_end
    tower=model.get_vision_tower()
    dtypes={name:sorted({str(p.dtype) for p in module.parameters()}) for name,module in {
        'vision_body':tower.vision_tower.vision_model,'visual_projection':tower.vision_tower.visual_projection,
        'text_tower':tower.text_tower,'mm_projector':model.get_model().mm_projector}.items()}
    assert dtypes==plan['cache_capture_runtime']['dtype']
    native_prepare=model.prepare_inputs_labels_for_multimodal
    for function in [native_prepare,model.generate]:
        path=inspect.getfile(inspect.unwrap(function.__func__))
        assert path in plan['source_sha256'] and sha(path)==plan['source_sha256'][path]
    arrays,contract=load()
    features=torch.from_numpy(arrays['projected_fp16']).to('cuda')
    pixels=torch.from_numpy(arrays['actual_input_fp16']).to('cuda')
    masks={arm:torch.from_numpy(arrays[arm+'_final_mask']).to('cuda') for arm in ARMS}
    prompt=contract['panel']['prompt'];conv=conv_templates['vicuna_v1'].copy()
    conv.append_message(conv.roles[0],DEFAULT_IMAGE_TOKEN+'\n'+prompt);conv.append_message(conv.roles[1],None)
    rendered=conv.get_prompt();ids=tokenizer_image_token(rendered,tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0)
    assert array_sha(ids.numpy())==plan['input_reference']['token_ids_sha256'] and ids.shape[1]==plan['input_reference']['n_tokens']
    markers=(ids[0]==IMAGE_TOKEN_INDEX).nonzero().flatten().tolist();assert len(markers)==1
    marker=markers[0];ids=ids.to('cuda');calls={};refs={};current=None
    forbidden_counts={'CLIP':0,'projector':0}
    def forbidden(name):
        def hook(*args,**kwargs):
            forbidden_counts[name]+=1
            raise AssertionError('Cached probe must never call CLIP/projector forward')
        return hook
    hooks=[model.get_vision_tower().register_forward_pre_hook(forbidden('CLIP')),model.get_model().mm_projector.register_forward_pre_hook(forbidden('projector'))]
    def encode(images,texts=None,split_sizes=None):
        assert split_sizes==[5] and list(images.shape)==[5,3,336,336]
        assert texts==prompt.replace('\nAnswer the question using a single word or phrase.','')
        calls['encode']+=1
        return features,masks[current]
    def prepare_inputs(*args,**kwargs):
        result=native_prepare(*args,**kwargs)
        if len(result)==7:
            calls['full_prepare']+=1;refs['prepared']=result
        return result
    model.encode_images=encode;model.prepare_inputs_labels_for_multimodal=prepare_inputs
    save(EXEC/'loaded_runtime.json',dict(model_class=type(model).__module__+'.'+type(model).__name__,backend=model.config._attn_implementation,
        model_dtype=str(model.dtype),alpha=model.alpha,beta=model.beta,K=model.visual_token_num,native_prepare_file=inspect.getfile(native_prepare.__func__),
        torch=torch.__version__,transformers=__import__('transformers').__version__,plan_sha256=sha(PLAN),worker_sha256=sha(__file__),
        same_GPU_cached_features_object=True,vision_or_projector_forward_allowed=False,component_dtypes=dtypes))
    records=[]
    for arm in ARMS:
        current=arm;calls={'encode':0,'full_prepare':0};refs.clear();started=time.monotonic()
        with torch.inference_mode():
            output,retained=model.generate(ids,images=pixels,image_sizes=[plan['image_size']],
                texts=prompt.replace('\nAnswer the question using a single word or phrase.',''),
                do_sample=False,temperature=0.,top_p=None,num_beams=1,max_new_tokens=128,use_cache=True)
        # All observations of live GPU tensor contents occur after generation.
        assert calls=={'encode':1,'full_prepare':1} and int(retained)==637
        embedded=refs['prepared'][4].detach().cpu().numpy()
        np.save(EXEC/(arm+'.inputs_embeds.npy'),embedded,allow_pickle=False)
        assert array_sha(features.detach().cpu().numpy())==contract['projected_fp16_sha256']
        tokens=output.detach().cpu().tolist();text=tokenizer.batch_decode(output,skip_special_tokens=True)[0].strip()
        record=dict(arm=arm,output_token_ids=tokens,text=text,actual_retained_tokens=int(retained),calls=calls,
            CLIP_forward_calls=forbidden_counts['CLIP'],projector_forward_calls=forbidden_counts['projector'],
            marker_index=marker,input_token_count=int(ids.shape[1]),input_ids_sha256=array_sha(ids.detach().cpu().numpy()),
            prepared_embeddings_path=str(EXEC/(arm+'.inputs_embeds.npy')),prepared_embeddings_sha256=sha(EXEC/(arm+'.inputs_embeds.npy')),
            shared_features_after_generation_sha256=contract['projected_fp16_sha256'],elapsed_seconds=time.monotonic()-started,observations_after_generate=True)
        records.append(record);save(EXEC/(arm+'.json'),record)
    save(EXEC/'native_complete.json',dict(success=True,n_generations=2,records=records,GT_or_accuracy_used=False))


def verify():
    import numpy as np
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    state=read(OUT/'state.json')
    assert state['status']=='native_exited' and state['native_exit_code']==0 and not (OUT/'failure.json').exists()
    assert identity(state['native_worker']['pid'])!=state['native_worker'] and state['native_worker']['pid'] not in gpu_pids()
    plan=read(PLAN);check(plan);arrays,contract=load();result=read(EXEC/'native_complete.json')
    assert result['success'] and result['n_generations']==2 and [r['arm'] for r in result['records']]==ARMS
    records=result['records'];embedded={}
    for record in records:
        assert record['calls']==dict(encode=1,full_prepare=1) and record['CLIP_forward_calls']==record['projector_forward_calls']==0
        assert record['actual_retained_tokens']==637 and record['observations_after_generate']
        assert record['input_ids_sha256']==plan['input_reference']['token_ids_sha256'] and record['input_token_count']==plan['input_reference']['n_tokens']
        assert sha(record['prepared_embeddings_path'])==record['prepared_embeddings_sha256']
        embedded[record['arm']]=np.load(record['prepared_embeddings_path'],allow_pickle=False)
    assert records[0]['marker_index']==records[1]['marker_index']
    verify_embeddings(embedded,records[0]['input_token_count'],records[0]['marker_index'],arrays,contract)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(plan['model_path'],use_fast=False,local_files_only=True)
    for record in records:
        decoded=tokenizer.batch_decode(record['output_token_ids'],skip_special_tokens=True)[0].strip()
        assert decoded==record['text']
    proof=dict(success=True,n_generations=2,visual_rows_verified_per_arm=637,nonvisual_embeddings_byte_identical=True,
        visual_sequence_rows_shifted=83,texts_identical=records[0]['text']==records[1]['text'],output_token_ids_identical=records[0]['output_token_ids']==records[1]['output_token_ids'],
        records=records,plan_sha256=sha(PLAN),native_complete_sha256=sha(EXEC/'native_complete.json'),CPU_only=True,model_loaded=False,
        GT_or_accuracy_used=False,benchmark_result=False,output_token_decoding_checks=2,limitations=plan['limitations'])
    import torch
    assert not torch.cuda.is_initialized()
    save(EXEC/'independent_verification.json',proof);return proof


def run(note_path):
    plan=read(PLAN);check(plan);prerequisite=k64_terminal_gate();note=read(note_path)
    assert note['authorization']=='explicit_root_gpu_allocation' and note['run_id']==plan['run_id']
    assert note['plan_sha256']==sha(PLAN) and note['GPU_slot'] in ['A','B']
    assert gpu_pids()==sorted(note['allowed_other_gpu_pids']) and len(gpu_pids())<2
    assert not EXEC.exists() and Path(sys.executable).resolve()==Path(PYTHON).resolve()
    controller=identity(os.getpid())
    with (OUT/'launch.claim.json').open('x') as stream:json.dump(dict(controller=controller,plan_sha256=sha(PLAN),note_sha256=sha(note_path),automatic_retry=False),stream)
    state=dict(status='starting',controller=controller,started_utc=utc(),prerequisite=prerequisite)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1')
    for key in list(env):
        if key.startswith(('USE_LLAVA_ARCH_','ANCHORZIP_','EADP_','ENTROPY_')):env.pop(key)
    try:
        with (OUT/'generation.log').open('x') as log:
            process=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'worker','--allocation-note',str(note_path.resolve())],env=env,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='generating',native_worker=identity(process.pid));save(OUT/'state.json',state)
            code=process.wait()
        state.update(status='native_exited',native_exit_code=code);save(OUT/'state.json',state)
        assert code==0
        assert identity(process.pid)!=state['native_worker'] and process.pid not in gpu_pids()
        with (OUT/'verification.log').open('x') as log:
            cpu=subprocess.run([PYTHON,str(Path(__file__).resolve()),'verify'],env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
        assert cpu.returncode==0
        proof=read(EXEC/'independent_verification.json');assert proof['success'];check(plan)
        save(EXEC/'release.json',dict(success=True,n_generations=2,native_exit_code=code,actual_native_exited=True,
            native_worker=state['native_worker'],controller=controller,controller_exit_must_be_confirmed_after_return=True,
            plan_sha256=sha(PLAN),independent_verification_sha256=sha(EXEC/'independent_verification.json'),score_direction_gate=False))
        state.update(status='complete',native_exit_code=code,finished_utc=utc(),release_sha256=sha(EXEC/'release.json'));save(OUT/'state.json',state)
        print(json.dumps(dict(success=True,n_generations=2,texts_identical=proof['texts_identical'],benchmark_result=False)))
    except BaseException as error:
        state.update(status='terminal_failure_no_retry',error=repr(error));save(OUT/'failure.json',state);save(OUT/'state.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','check','run','worker','verify'],nargs='?',default='check');parser.add_argument('--allocation-note',type=Path);args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='check':check(read(PLAN));print(json.dumps(dict(CPU_validation=True,GPU_started=False,model_loaded=False)))
    elif args.action=='verify':verify()
    elif args.action=='worker':assert args.allocation_note is not None;worker(args.allocation_note)
    else:assert args.allocation_note is not None;run(args.allocation_note)
