"""Fixed16 native source1/1 then beta2; preserves raw attempts before checks."""
import hashlib
import inspect
import os
from pathlib import Path
import sys
import time
import common as c
from contracts import ARMS,guidance,array_sha,equal_bits,selection,embeddings,require_all_sources


def worker(note_path):
    plan=c.read(c.PLAN);c.check(plan);note=c.read(note_path);claim=c.read(c.HERE/'launch.claim.json')
    assert c.sha(c.PLAN)==claim['plan_sha256']==note['plan_sha256'] and c.sha(note_path)==claim['allocation_note_sha256']
    from run import identity,allocation
    assert identity(os.getppid())==claim['controller'];allocation(note,plan)
    c.OUT.mkdir(exist_ok=False)
    c.save(c.OUT/'process_identity.json',dict(**identity(os.getpid()),session_id=os.getsid(0),started_utc=c.UTC(),
        plan_sha256=c.sha(c.PLAN),allocation_note_sha256=c.sha(note_path)))
    import numpy as np
    import torch
    from PIL import Image
    assert not torch.cuda.is_initialized()
    assert torch.__version__=='2.1.2+cu118' and __import__('transformers').__version__=='4.37.2'
    sys.path.insert(0,plan['llava_root']);sys.path.insert(0,plan['pope_root'])
    from wait_only_patch import install
    c.save(c.OUT/'wait_patch_proof.json',install())
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path,process_images,tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX,DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.utils import disable_torch_init
    disable_torch_init()
    tokenizer,model,processor,_=load_pretrained_model(plan['model_path'],None,get_model_name_from_path(plan['model_path']),
        visual_token_num=32,alpha=1.,beta=1.)
    assert model.visual_token_num==32 and model.alpha==model.beta==1. and str(model.dtype)=='torch.float16'
    assert model.config._attn_implementation=='sdpa' and model.config.image_aspect_ratio=='pad' and not model.config.mm_use_im_start_end
    assert not torch.is_autocast_enabled() and not torch.backends.cuda.matmul.allow_tf32 and torch.get_default_dtype()==torch.float32
    tower=model.get_vision_tower();dtype=lambda m:sorted({str(p.dtype) for p in m.parameters()})
    dtypes={k:dtype(m) for k,m in dict(vision_body=tower.vision_tower.vision_model,
        visual_projection=tower.vision_tower.visual_projection,text_tower=tower.text_tower,mm_projector=model.get_model().mm_projector).items()}
    assert dtypes==plan['expected_component_dtypes']
    for function in [model.generate,model.prepare_inputs_labels_for_multimodal,model.encode_images,model.sim_cross,model.greed_select]:
        path=inspect.getfile(inspect.unwrap(function.__func__));assert c.sha(path)==plan['source_sha256'][path]
    native={k:getattr(model,k) for k in ['encode_images','sim_cross','greed_select','prepare_inputs_labels_for_multimodal']}
    gain_refs=[]
    observed_greedy,gain_proof=c.gain_observer.install(native['greed_select'],gain_refs)
    assert {k:gain_proof[k] for k in plan['gain_observer_proof']}==plan['gain_observer_proof']
    c.save(c.OUT/'loaded_runtime.json',dict(model_class=type(model).__module__+'.'+type(model).__name__,backend=model.config._attn_implementation,
        model_dtype=str(model.dtype),component_dtypes=dtypes,initial_alpha=model.alpha,initial_beta=model.beta,K=model.visual_token_num,
        torch=torch.__version__,CUDA=torch.version.cuda,device='cuda:0',GPU=torch.cuda.get_device_name(0),
        transformers=__import__('transformers').__version__,bos_token_id=model.generation_config.bos_token_id,
        eos_token_id=model.generation_config.eos_token_id,plan_sha256=c.sha(c.PLAN),worker_sha256=c.sha(__file__),
        gain_observer_proof=gain_proof,observer_policy='Bound native calls once; refs retained, all tensor content reads/stacks after generate.',
        backend_flags=dict(matmul_TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic)))
    capture={};calls={};current_guidance=None;current_beta=None
    def encode(*args,**kwargs):
        calls['encode']+=1
        assert kwargs.get('texts')==current_guidance and kwargs.get('split_sizes') is None
        assert model.alpha==1. and model.beta==current_beta and model.visual_token_num==32
        capture['actual_alpha_beta_K']=(model.alpha,model.beta,model.visual_token_num)
        value=native['encode_images'](*args,**kwargs);capture['projected_fp16'],capture['final_mask']=value;return value
    def cross(*args,**kwargs):
        calls['cross']+=1;capture['text_global_fp32'],capture['text_dense_fp32'],capture['image_embeds_fp32']=args[:3]
        value=native['sim_cross'](*args,**kwargs);capture['global_fp32'],capture['dense_fp32']=value;return value
    def greedy(*args,**kwargs):
        calls['greedy']+=1;capture['importance_fp32'],capture['visual_sim_fp32']=args[:2]
        value=observed_greedy(*args,**kwargs);capture['internal_ids'],capture['quotas']=value;return value
    def prepare_inputs(*args,**kwargs):
        value=native['prepare_inputs_labels_for_multimodal'](*args,**kwargs)
        if len(value)==7:
            calls['full_prepare']+=1;capture['actual_LLM_input_ids']=args[0];capture['prepared_embeddings_fp16']=value[4]
        return value
    model.encode_images=encode;model.sim_cross=cross;model.greed_select=greedy;model.prepare_inputs_labels_for_multimodal=prepare_inputs
    fixture=c.read(c.HERE/'source_anchor_fixture.json')['rows'];all_records=[];source_records=[]
    for arm in ARMS:
        if arm=='beta2':
            require_all_sources(source_records)
            c.save(c.OUT/'all16_source_anchor_gate.json',dict(passed=True,n_sources=16,all_sources_before_variant=True,
                source_records=[c.describe(c.OUT/('source_%02d.json'%i)) for i in range(16)]))
        current_beta=1. if arm=='source' else 2.
        # encode_images reads this exact native scalar through getattr(self,'beta').
        # No weights, native function bodies or generation arguments are changed.
        model.beta=current_beta
        for index,(panel,anchor) in enumerate(zip(plan['panel'],fixture)):
            image=Image.open(panel['image_path']).convert('RGB');pixels=process_images([image],processor,model.config)
            assert isinstance(pixels,torch.Tensor) and tuple(pixels.shape)==(1,3,336,336)
            conv=conv_templates['vicuna_v1'].copy();conv.append_message(conv.roles[0],DEFAULT_IMAGE_TOKEN+'\n'+panel['prompt']);conv.append_message(conv.roles[1],None)
            cpu_ids=tokenizer_image_token(conv.get_prompt(),tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0)
            input_ids=cpu_ids.to(device='cuda',non_blocking=True);image_tensors=pixels.to(dtype=torch.float16,device='cuda',non_blocking=True)
            current_guidance=guidance(panel['prompt'],arm);capture.clear();gain_refs.clear();calls=dict(encode=0,cross=0,greedy=0,full_prepare=0)
            started=time.monotonic()
            with torch.inference_mode():
                output,retained=model.generate(input_ids,images=image_tensors,image_sizes=[image.size],texts=current_guidance,
                    do_sample=False,temperature=0.,top_p=None,num_beams=1,max_new_tokens=128,use_cache=True)
            # All captured tensor content observation begins after native generate.
            actual_alpha_beta_K=capture.pop('actual_alpha_beta_K',None)
            arrays={k:v.detach().cpu().numpy() for k,v in capture.items()}
            if gain_refs:arrays['gain_trace_fp32']=torch.stack(gain_refs).detach().cpu().numpy()
            arrays['actual_input_fp16']=image_tensors.detach().cpu().numpy()
            tokens=output.detach().cpu().tolist();text=tokenizer.batch_decode(output,skip_special_tokens=True)[0].strip()
            path=c.OUT/('%s_%02d.npz'%(arm,index));assert not path.exists();np.savez_compressed(path,**arrays)
            attempt_path=c.OUT/('%s_%02d.attempt.json'%(arm,index))
            attempt=dict(status='raw_measurement_saved_unvalidated',arm=arm,index=index,position=index,question_id=panel['question_id'],
                arrays=c.describe(path),output_token_ids=tokens,text=text,calls=calls,actual_retained_tokens=int(retained),
                actual_alpha_beta_K=list(actual_alpha_beta_K) if actual_alpha_beta_K is not None else None,
                gain_reference_count=len(gain_refs),post_generation_tensor_reads_only=True,
                all_checks_completed=False,accepted_case_record_required=True)
            c.save(attempt_path,attempt)
            try:
                assert calls==dict(encode=1,cross=1,greedy=1,full_prepare=1) and actual_alpha_beta_K==(1.,current_beta,32)
                assert all(np.isfinite(value).all() for value in arrays.values())
                assert arrays['projected_fp16'].shape==(1,576,4096) and arrays['projected_fp16'].dtype==np.float16
                assert arrays['image_embeds_fp32'].shape==(1,576,768) and arrays['image_embeds_fp32'].dtype==np.float32
                assert all(arrays[k].dtype==np.float32 for k in ['text_global_fp32','text_dense_fp32','global_fp32','dense_fp32','visual_sim_fp32','importance_fp32','gain_trace_fp32'])
                assert arrays['gain_trace_fp32'].shape==(32,1,576) and equal_bits(arrays['actual_LLM_input_ids'],cpu_ids.numpy())
                observed=selection(arrays['final_mask'],arrays['internal_ids'],arrays['quotas']);assert int(retained)==32
                nonvisual,embedded=embeddings(arrays['prepared_embeddings_fp16'],arrays['projected_fp16'],arrays['final_mask'],arrays['actual_LLM_input_ids'])
                assert len(tokens)==1 and tokens[0][0]==model.generation_config.bos_token_id and 1<len(tokens[0])<=129
                source_anchor=False
                if arm=='source':
                    with np.load(panel['cache']['path'],allow_pickle=False) as old:
                        for name,old_name in [('projected_fp16','projected_fp16'),('actual_input_fp16','input_fp16'),('global_fp32','global_fp32'),
                            ('dense_fp32','dense_fp32'),('visual_sim_fp32','visual_sim_fp32'),('importance_fp32','native_importance'),
                            ('internal_ids','native_selected_ids'),('final_mask','native_final_mask'),('gain_trace_fp32','alpha1_beta1__source__gain_trace')]:
                            assert equal_bits(arrays[name],old[old_name]),('old_native_or_helper_anchor',index,name)
                    assert text==anchor['expected_source_text'],('historical_text_only_anchor',index)
                    source_anchor=True
                else:
                    source=c.read(c.OUT/('source_%02d.json'%index))
                    with np.load(source['arrays']['path'],allow_pickle=False) as first:
                        for name in ['projected_fp16','image_embeds_fp32','text_global_fp32','text_dense_fp32','global_fp32','dense_fp32',
                            'visual_sim_fp32','actual_input_fp16','actual_LLM_input_ids']:
                            assert equal_bits(arrays[name],first[name]),('only_beta_may_change',index,name)
                        old_nonvisual,_=embeddings(first['prepared_embeddings_fp16'],first['projected_fp16'],first['final_mask'],first['actual_LLM_input_ids'])
                        assert equal_bits(nonvisual,old_nonvisual),('nonvisual_embeddings',index)
            except BaseException as error:
                c.save(c.OUT/('%s_%02d.failed_checks.json'%(arm,index)),dict(status='terminal_case_validation_failed_no_retry',
                    error=repr(error),raw_attempt=c.describe(attempt_path),arrays=c.describe(path),later_generations_permitted=False));raise
            record=dict(arm=arm,index=index,position=index,question_id=panel['question_id'],image=panel['image'],
                full_LLM_prompt_sha256=hashlib.sha256(panel['prompt'].encode()).hexdigest(),guidance_sha256=hashlib.sha256(current_guidance.encode()).hexdigest(),
                actual_alpha_beta_K=list(actual_alpha_beta_K),calls=calls,post_generation_tensor_reads_only=True,all_source_anchors_passed=source_anchor,
                selection=observed,embedded_input_contract=embedded,actual_LLM_input_ids_sha256=array_sha(arrays['actual_LLM_input_ids']),
                output_token_ids=tokens,text=text,returned_tokens=len(tokens[0]),initial_BOS_tokens=1,new_tokens=len(tokens[0])-1,
                EOS_present=model.generation_config.eos_token_id in tokens[0],generation_horizon=128,
                arrays=c.describe(path),raw_attempt=c.describe(attempt_path),checks_status='all_passed',
                tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=array_sha(v)) for k,v in arrays.items()},
                elapsed_seconds=time.monotonic()-started,timing_is_efficiency_measurement=False)
            c.save(c.OUT/('%s_%02d.json'%(arm,index)),record);all_records.append(record)
            if arm=='source':source_records.append(record)
            print(__import__('json').dumps(dict(arm=arm,completed_index=index,all_source_anchors_passed=source_anchor)),flush=True)
    assert len(all_records)==32;require_all_sources(source_records)
    for group in ['source_sha256','implementation_sha256','reference_sha256']:assert all(c.sha(p)==v for p,v in plan[group].items())
    assert all(c.stat(p)=={k:v for k,v in g.items() if k!='prior_verified_sha256'} for p,g in plan['model_stat_guards'].items())
    assert all(c.sha(r['image_path'])==r['image_sha256'] and c.sha(r['cache']['path'])==r['cache']['sha256'] for r in plan['panel'])
    c.save(c.OUT/'native_complete.json',dict(success=True,n_generations=32,source_generations=16,variant_generations=16,
        all16_source_gate_passed=True,records=all_records,plan_sha256=c.sha(c.PLAN),GT_or_accuracy_used=False,benchmark_result=False))


if __name__=='__main__':
    assert len(sys.argv)==2
    try:worker(Path(sys.argv[1]).resolve())
    except BaseException as error:
        c.save(c.HERE/'worker_failure.json',dict(status='terminal_failure_no_retry',error=repr(error)));raise
