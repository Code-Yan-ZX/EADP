"""Single author-model, 60 source then 60 question-only diagnostic worker.

This entry point requires a controller claim and a root-issued allocation note.
Importing it never loads a model. Nothing here scores or reads GT.
"""
import argparse, ast, hashlib, inspect, json, os, sys, time, traceback
from pathlib import Path
import common as c

ARMS=('source','question_only')
PARAMETERS=dict(do_sample=False,temperature=0.0,top_p=None,num_beams=1,max_new_tokens=1024,use_cache=True)

def text_sha(x):return hashlib.sha256(x.encode('utf-8')).hexdigest()
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def describe(p):return dict(path=str(p),sha256=c.sha(p))

def run(note_path):
    plan=c.read(c.PLAN);c.check(plan)
    note=c.read(note_path);claim=c.read(c.HERE/'launch.claim.json')
    assert note['authorization']=='explicit_root_gpu_allocation' and note['run_id']==plan['run_id']
    assert note['plan_sha256']==claim['plan_sha256']==c.sha(c.PLAN)
    assert claim['allocation_sha256']==c.sha(note_path) and claim['controller']==c.identity(os.getppid())
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(note['GPU_device'])
    assert c.OUT.exists() and not (c.OUT/'worker_identity.json').exists()
    c.save_new(c.OUT/'worker_identity.json',dict(identity=c.identity(os.getpid()),parent=c.identity(os.getppid()),plan_sha256=c.sha(c.PLAN),allocation_sha256=c.sha(note_path)))
    import numpy as np
    import torch
    import transformers
    from observer import Observer
    from verifier import check_selection,check_embeddings,check_clip_inputs,check_pair,check_output,check_shapes,array_sha
    assert not torch.cuda.is_initialized() and sys.flags.optimize==0
    assert torch.__version__=='2.1.2+cu118' and transformers.__version__=='4.37.2'
    root=c.SOURCE.parents[2];sys.path.insert(0,str(root))
    sys.path.insert(0,str(c.HERE.parent/'question_only_guidance_panel_design'))
    sys.path.insert(0,str(c.HERE.parent/'streamwait_control'))
    import source_boundary as boundary
    from patch_only import build_patch
    import llava.model.multimodal_encoder.clip_encoder as clip
    proof=boundary.proof(); assert proof['success']
    patched,validation=build_patch()
    historical=c.read(c.HERE.parent/'streamwait_control/runtime_patch_validation.json')
    for key in ('source_sha256','original_forward_ast_sha256','patched_forward_ast_sha256','exact_added_statement_count','added_statements','restored_ast_identical'):
        assert validation[key]==historical[key],key
    namespace=dict(clip.__dict__)
    patched_name=validation['source_file']+':in_memory_mmb_streamwait'
    code=compile(patched,patched_name,'exec',dont_inherit=True,optimize=0)
    exec(code,namespace)
    clip.CLIPVisionTower.forward=namespace['forward']
    underlying=inspect.unwrap(clip.CLIPVisionTower.forward).__code__
    compiled_forward=[v for v in code.co_consts if getattr(v,'co_name',None)=='forward']
    assert len(compiled_forward)==1 and underlying==compiled_forward[0] and underlying.co_filename==patched_name
    assert c.sha(validation['source_file'])==validation['source_sha256']
    data=c.SOURCE.read_bytes();assert hashlib.sha256(data).hexdigest()==boundary.EXPECTED_SOURCE_SHA
    original=ast.parse(data,filename=str(c.SOURCE));trees=[original,boundary.transform(original)]
    modules=[]
    for arm,tree in zip(ARMS,trees):
        module_code=compile(tree,str(c.SOURCE),'exec',dont_inherit=True,optimize=0)
        namespace={'__name__':'mmb_panel_'+arm,'__file__':str(c.SOURCE)}
        exec(module_code,namespace)
        expected=boundary.codefn(module_code)
        assert namespace['eval_model'].__code__==expected and expected.co_filename==str(c.SOURCE)
        modules.append(namespace)
    assert c.sha(c.SOURCE)==boundary.EXPECTED_SOURCE_SHA
    manifest=c.read(c.HERE/'inputs/manifest.json');rows=manifest['rows'];assert len(rows)==60
    historical_runtime=lines(c.HERE.parent/'streamwait_control/observed/runtime.jsonl')
    from llava.model.builder import load_pretrained_model
    state={'phase':None,'position':0,'loaded':None,'observer':None,'native_generate':None,'load_signature':None}
    records={arm:[] for arm in ARMS}

    def array_meta(a):return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=array_sha(a))
    def capture_generate(*args,**kwargs):
        arm=state['phase'];i=state['position'];assert arm in ARMS and i<60
        item=rows[i];label='%s_%02d'%(arm,i);raw_path=c.OUT/(label+'.raw.json')
        obs=state['observer'];obs.reset();obs.arrays['actual_input_fp16']=kwargs.get('images')
        obs.arrays['actual_generate_input_ids']=args[0] if args else None
        started=time.monotonic()
        try:
            answer=state['native_generate'](*args,**kwargs)
        except BaseException as exc:
            c.save_new(raw_path,dict(arm=arm,position=i,index=item['index'],group_id=item['group_id'],native_failure=repr(exc),traceback=traceback.format_exc(),native_returned=False))
            raise
        # Preserve IDs and native answer before any observation checks or array copying.
        output=answer[0].detach().cpu().numpy();tokens=output.tolist()
        c.save_new(c.OUT/(label+'.output_ids.json'),dict(arm=arm,position=i,index=item['index'],output_ids=tokens,native_returned=True))
        try:
            response=state['loaded'][0].batch_decode(answer[0],skip_special_tokens=True)[0].strip()
        except BaseException as exc:
            c.save_new(raw_path,dict(arm=arm,position=i,index=item['index'],group_id=item['group_id'],native_failure=None,native_returned=True,output_ids=tokens,decoding_failure=repr(exc),traceback=traceback.format_exc()))
            raise
        c.save_new(raw_path,dict(arm=arm,position=i,index=item['index'],group_id=item['group_id'],native_failure=None,native_returned=True,
            output_ids=tokens,response=response,returned_visual_token_num=int(answer[1]),seconds=time.monotonic()-started))
        try:
            obs.arrays['output_ids']=answer[0]
            errors=list(obs.errors);arrays={}
            for key,value in obs.arrays.items():
                try:arrays[key]=value.detach().cpu().numpy()
                except Exception as exc:errors.append('serialize %s: %s: %s'%(key,type(exc).__name__,exc))
            import torch.nn.modules.module as torch_module
            tower=state['loaded'][1].get_vision_tower().text_tower
            if torch_module._global_forward_pre_hooks or list(tower._forward_pre_hooks)!=[obs.handle.id]:errors.append('unexpected CLIP pre-hook state')
            npz_path=c.OUT/(label+'.npz')
            with npz_path.open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
            scalar={k:kwargs.get(k) for k in PARAMETERS}
            record=dict(status='captured_unvalidated',arm=arm,position=i,index=item['index'],group_id=item['group_id'],
                raw=describe(raw_path),output_ids_file=describe(c.OUT/(label+'.output_ids.json')),archive=describe(npz_path),arrays={k:array_meta(v) for k,v in arrays.items()},
                calls=obs.calls,observer_errors=errors,encode_texts=obs.encode_texts,
                guidance=kwargs.get('texts'),image_sizes=kwargs.get('image_sizes'),generate_scalar_kwargs=scalar,
                generate_keys=sorted(kwargs),positional_args_count=len(args),response=response,
                returned_visual_token_num=int(answer[1]))
            c.save_new(c.OUT/(label+'.json'),record)
            state['position']+=1
        except BaseException as exc:
            c.save_new(c.OUT/(label+'.observation_failure.json'),dict(error=repr(exc),traceback=traceback.format_exc(),raw=describe(raw_path)))
            raise
        return answer

    def cached_loader(*args,**kwargs):
        signature=repr((args,kwargs))
        if state['loaded'] is not None:
            assert signature==state['load_signature']
            return state['loaded']
        state['load_signature']=signature
        loaded=load_pretrained_model(*args,**kwargs);tokenizer,model,processor,context=loaded
        assert model.visual_token_num==128 and model.alpha==.5 and model.beta==1. and str(model.dtype)=='torch.float16'
        assert model.config._attn_implementation=='sdpa' and not torch.is_autocast_enabled()
        state['loaded']=loaded;state['native_generate']=model.generate
        state['observer']=Observer(model).install();model.generate=capture_generate
        c.save_new(c.OUT/'runtime.json',dict(plan_sha256=c.sha(c.PLAN),torch=torch.__version__,transformers=transformers.__version__,
            model_class=type(model).__module__+'.'+type(model).__name__,model_path=str(c.MODEL),model_dtype=str(model.dtype),
            device=str(model.device),attention_implementation=model.config._attn_implementation,K=model.visual_token_num,alpha=model.alpha,beta=model.beta,
            vocab_size=model.config.vocab_size,bos_token_id=model.generation_config.bos_token_id,eos_token_id=model.generation_config.eos_token_id,
            source_sha256=c.sha(c.SOURCE),source_boundary_proof=proof,wait_patch=validation,one_loaded_model=True))
        return loaded
    for namespace in modules:namespace['load_pretrained_model']=cached_loader

    def check_arm(arm):
        author=lines(c.OUT/(arm+'_author.jsonl'));assert len(author)==60
        for i,(item,a) in enumerate(zip(rows,author)):
            rec=c.read(c.OUT/('%s_%02d.json'%(arm,i)));raw=c.read(rec['raw']['path'])
            assert rec['calls']==dict(encode=1,cross=1,greedy=1,prepare=1,clip=1) and not rec['observer_errors']
            assert rec['generate_scalar_kwargs']==PARAMETERS and rec['generate_keys']==sorted(['images','image_sizes','texts',*PARAMETERS])
            assert rec['positional_args_count']==1 and rec['image_sizes']==[item['image']['size']]
            guidance=item['full_guidance'] if arm=='source' else item['question_only_guidance']
            assert rec['guidance']==guidance and rec['encode_texts']==[guidance]
            assert a['question_id']==item['index'] and a['round_id']==0 and a['prompt']==item['full_guidance'] and a['text']==raw['response']
            with np.load(rec['archive']['path'],allow_pickle=False) as arr:
                check_shapes(arr,item[arm+'_clip']['n_segments'])
                assert np.array_equal(arr['actual_input_ids'],arr['actual_generate_input_ids'])
                assert array_sha(arr['actual_input_ids'])==item['llm_input_ids_sha256']
                assert arr['actual_input_ids'].tolist()==[item['llm_input_ids']]
                assert list(arr['actual_input_fp16'].shape)==item['image']['pixel_shape']
                assert str(arr['actual_input_fp16'].dtype)=='float16' and array_sha(arr['actual_input_fp16'])==item['image']['pixel_float16_sha256']
                check_clip_inputs(arr['clip_input_ids'],arr['clip_attention_mask'],item[arm+'_clip'])
                assert arr['clip_input_ids'].tolist()==item[arm+'_clip']['input_ids']
                assert arr['clip_attention_mask'].tolist()==item[arm+'_clip']['attention_mask']
                check_selection(arr['final_mask'],arr['internal_ids'],arr['quotas'])
                check_embeddings(arr['prepared_embeddings_fp16'],arr['projected_fp16'],arr['final_mask'],arr['actual_input_ids'])
                assert rec['returned_visual_token_num']==128
                eos=state['loaded'][1].generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                check_output(arr['output_ids'],state['loaded'][1].config.vocab_size,state['loaded'][1].generation_config.bos_token_id,eos)
            if arm=='source':
                assert raw['response']==item['accepted_source']['text']
                call=item['accepted_source']['source_runtime_call'];old=historical_runtime[call-1];assert old['call']==call
                assert old['guidance_text_sha256']==text_sha(guidance)
                assert old['input_shape']==[1,item['llm_n_tokens']] and old['image_shape']==item['image']['pixel_shape']
                assert old['image_dtype']==item['image']['pixel_dtype'] and old['image_sizes']==[item['image']['size']]
            records[arm].append(describe(c.OUT/('%s_%02d.json'%(arm,i))))

    try:
        for arm,namespace in zip(ARMS,modules):
            if arm=='question_only':
                assert len(records['source'])==60
                c.save_new(c.OUT/'source_anchor_gate.json',dict(passed=True,n_sources=60,records=records['source'],author_output=describe(c.OUT/'source_author.jsonl'),plan_sha256=c.sha(c.PLAN),all_sources_before_variant=True))
            state['phase']=arm;state['position']=0
            args=argparse.Namespace(model_path=str(c.MODEL),model_base=None,question_file=str(c.HERE/'inputs/panel.tsv'),answers_file=str(c.OUT/(arm+'_author.jsonl')),
                conv_mode='vicuna_v1',num_chunks=1,chunk_idx=0,all_rounds=False,single_pred_prompt=True,lang='en',temperature=0.,top_p=None,num_beams=1,visual_token_num=128,beta=1.,alpha=.5)
            namespace['eval_model'](args)
            assert state['position']==60
            check_arm(arm)
        for i in range(60):
            with np.load(c.OUT/('source_%02d.npz'%i),allow_pickle=False) as source,np.load(c.OUT/('question_only_%02d.npz'%i),allow_pickle=False) as variant:
                check_pair(source,variant)
        c.save_new(c.OUT/'native_complete.json',dict(status='complete',plan_sha256=c.sha(c.PLAN),runtime=describe(c.OUT/'runtime.json'),
            source_anchor_gate=describe(c.OUT/'source_anchor_gate.json'),source_records=records['source'],question_only_records=records['question_only'],
            source_author=describe(c.OUT/'source_author.jsonl'),question_only_author=describe(c.OUT/'question_only_author.jsonl'),n_source=60,n_question_only=60,scoring=False))
    except BaseException as exc:
        c.save_new(c.OUT/'worker_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),phase=state['phase'],position=state['position']))
        raise
    finally:
        if state['loaded'] is not None:
            state['loaded'][1].generate=state['native_generate']
        if state['observer'] is not None:state['observer'].close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--allocation-note',required=True);opts=parser.parse_args();run(opts.allocation_note)
