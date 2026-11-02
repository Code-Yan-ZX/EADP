"""One author model, fixed60 anchors then paired full MMB; explicit controller only."""
import argparse,ast,hashlib,inspect,json,os,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
import common as c

ARMS=('source','question_only')
PARAMETERS=dict(do_sample=False,temperature=0.0,top_p=None,num_beams=1,max_new_tokens=1024,use_cache=True)
PANEL=c.HERE.parent/'question_only_guidance_panel'

def describe(path):return dict(path=str(path),sha256=c.sha(path))
def textsha(s):return hashlib.sha256(s.encode()).hexdigest()
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]

def run(note_path):
    plan=c.read(c.PLAN);c.check(plan);note=c.read(note_path);claim=c.read(c.HERE/'launch.claim.json')
    assert note['authorization']=='explicit_root_gpu_allocation' and note['run_id']==plan['run_id']
    assert note['plan_sha256']==claim['plan_sha256']==c.sha(c.PLAN)
    assert claim['allocation_sha256']==c.sha(note_path) and claim['controller']==c.identity(os.getppid())
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(note['GPU_device'])
    for name in ('cases','pairs','anchors','author'):(c.OUT/name).mkdir(exist_ok=False)
    c.save_new(c.OUT/'worker_identity.json',dict(identity=c.identity(os.getpid()),parent=c.identity(os.getppid()),plan_sha256=c.sha(c.PLAN),allocation_sha256=c.sha(note_path)))
    import numpy as np
    import pandas as pd
    import torch,transformers
    from observer import Observer
    sys.path.insert(0,str(PANEL))
    from verifier import check_shapes,check_selection,check_embeddings,check_clip_inputs,check_pair,check_output,array_sha,bits
    assert not torch.cuda.is_initialized() and sys.flags.optimize==0
    assert torch.__version__=='2.1.2+cu118' and transformers.__version__=='4.37.2'
    sys.path.insert(0,str(c.SOURCE.parents[2]));sys.path.insert(0,str(c.HERE.parent/'question_only_guidance_panel_design'));sys.path.insert(0,str(c.HERE.parent/'streamwait_control'))
    import source_boundary as boundary
    from patch_only import build_patch
    import llava.model.multimodal_encoder.clip_encoder as clip
    proof=boundary.proof();assert proof['success'];patch,validation=build_patch();oldwait=c.read(c.HERE.parent/'streamwait_control/runtime_patch_validation.json')
    for k in ('source_sha256','original_forward_ast_sha256','patched_forward_ast_sha256','exact_added_statement_count','added_statements','restored_ast_identical'):assert validation[k]==oldwait[k]
    namespace=dict(clip.__dict__);name=validation['source_file']+':in_memory_mmb_streamwait';patched=compile(patch,name,'exec',dont_inherit=True,optimize=0);exec(patched,namespace)
    clip.CLIPVisionTower.forward=namespace['forward'];underlying=inspect.unwrap(clip.CLIPVisionTower.forward).__code__;expected=[v for v in patched.co_consts if getattr(v,'co_name',None)=='forward']
    assert len(expected)==1 and underlying==expected[0] and underlying.co_filename==name and c.sha(validation['source_file'])==validation['source_sha256']
    source=c.SOURCE.read_bytes();assert hashlib.sha256(source).hexdigest()==boundary.EXPECTED_SOURCE_SHA
    original=ast.parse(source,filename=str(c.SOURCE));modules={}
    for arm,tree in zip(ARMS,(original,boundary.transform(original))):
        code=compile(tree,str(c.SOURCE),'exec',dont_inherit=True,optimize=0);ns={'__name__':'mmb_full_'+arm,'__file__':str(c.SOURCE)};exec(code,ns)
        actual=ns['eval_model'].__code__;assert actual==boundary.codefn(code) and actual.co_filename==str(c.SOURCE);modules[arm]=ns
    assert c.sha(c.SOURCE)==boundary.EXPECTED_SOURCE_SHA
    manifest=c.read(c.HERE/'inputs/manifest.json');rows=manifest['rows'];assert len(rows)==4377
    panel=c.read(PANEL/'inputs/manifest.json')['rows'];assert len(panel)==60
    anchor_position={r['original_position']:i for i,r in enumerate(panel)};assert len(anchor_position)==60
    for pos,i in anchor_position.items():assert rows[pos]['index']==panel[i]['index']
    frame=pd.read_table(c.HERE/'inputs/panel.tsv');assert len(frame)==4377 and list(frame.columns)==['index','question','hint','A','B','C','D','image']
    assert [int(x) for x in frame['index']]==[r['index'] for r in rows]
    from llava.model.builder import load_pretrained_model
    state=dict(phase=None,position=None,loaded=None,observer=None,native_generate=None,load_signature=None,arrays=None)
    record_paths={};pair_paths={}
    def meta(a):return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=array_sha(a))
    def gen(*args,**kwargs):
        arm=state['phase'];pos=state['position'];item=rows[pos];label='%s_%04d'%(arm,pos);out=c.OUT/'cases';raw=out/(label+'.raw.json');ids=out/(label+'.output_ids.json')
        obs=state['observer'];obs.reset();obs.arrays['actual_input_fp16']=kwargs.get('images');obs.arrays['actual_generate_input_ids']=args[0] if args else None
        start=time.monotonic()
        try:answer=state['native_generate'](*args,**kwargs)
        except BaseException as exc:
            c.save_new(raw,dict(arm=arm,position=pos,index=item['index'],group_id=item['group_id'],native_returned=False,native_failure=repr(exc),traceback=traceback.format_exc()));raise
        tokens=answer[0].detach().cpu().tolist()
        c.save_new(ids,dict(arm=arm,position=pos,index=item['index'],group_id=item['group_id'],native_returned=True,output_ids=tokens))
        try:response=state['loaded'][0].batch_decode(answer[0],skip_special_tokens=True)[0].strip()
        except BaseException as exc:
            c.save_new(raw,dict(arm=arm,position=pos,index=item['index'],group_id=item['group_id'],native_returned=True,native_failure=None,output_ids=tokens,decoding_failure=repr(exc)));raise
        c.save_new(raw,dict(arm=arm,position=pos,index=item['index'],group_id=item['group_id'],native_returned=True,native_failure=None,output_ids=tokens,response=response,returned_visual_token_num=int(answer[1]),seconds=time.monotonic()-start))
        try:
            obs.arrays['output_ids']=answer[0];arrays={};errors=list(obs.errors)
            for key,v in obs.arrays.items():
                try:arrays[key]=v.detach().cpu().numpy()
                except Exception as exc:errors.append('%s: %s'%(key,repr(exc)))
            import torch.nn.modules.module as tm
            tower=state['loaded'][1].get_vision_tower().text_tower
            if tm._global_forward_pre_hooks or list(tower._forward_pre_hooks)!=[obs.handle.id]:errors.append('unexpected CLIP pre-hook state')
            small=('actual_input_ids','clip_input_ids','clip_attention_mask','internal_ids','quotas','final_mask','output_ids')
            capture=out/(label+'.capture.json')
            array_metadata={};small_values={}
            for key,value in arrays.items():
                try:array_metadata[key]=meta(value)
                except Exception as exc:errors.append('metadata %s: %s'%(key,repr(exc)))
                if key in small:
                    try:small_values[key]=value.tolist()
                    except Exception as exc:errors.append('small %s: %s'%(key,repr(exc)))
            c.save_new(capture,dict(arm=arm,position=pos,index=item['index'],group_id=item['group_id'],
                arrays=array_metadata,small_arrays=small_values,
                calls=obs.calls,observer_errors=errors,raw=describe(raw),output_ids_file=describe(ids)))
            archive=None
            if pos in anchor_position:
                archive=c.OUT/'anchors'/('%s_%02d.npz'%(arm,anchor_position[pos]))
                with archive.open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
            # Raw is already durable: now inspect true captured arrays and compute relations.
            assert not errors and obs.calls==dict(encode=1,cross=1,greedy=1,prepare=1,clip=1)
            check_shapes(arrays,item[arm+'_clip']['n_segments'])
            assert kwargs.get('texts')==(item['full_guidance'] if arm=='source' else item['question_only_guidance'])==obs.encode_texts[0]
            assert {k:kwargs.get(k) for k in PARAMETERS}==PARAMETERS and sorted(kwargs)==sorted(['images','image_sizes','texts',*PARAMETERS]) and len(args)==1
            assert kwargs['image_sizes']==[item['image']['size']]
            assert arrays['actual_input_ids'].tolist()==[item['llm_input_ids']] and array_sha(arrays['actual_input_ids'])==item['llm_input_ids_sha256']
            assert array_sha(arrays['actual_input_fp16'])==item['image']['pixel_float16_sha256']
            check_clip_inputs(arrays['clip_input_ids'],arrays['clip_attention_mask'],item[arm+'_clip'])
            assert arrays['clip_input_ids'].tolist()==item[arm+'_clip']['input_ids'] and arrays['clip_attention_mask'].tolist()==item[arm+'_clip']['attention_mask']
            check_selection(arrays['final_mask'],arrays['internal_ids'],arrays['quotas']);nonvisual,details=check_embeddings(arrays['prepared_embeddings_fp16'],arrays['projected_fp16'],arrays['final_mask'],arrays['actual_input_ids'])
            marker=details['marker_index'];selected=arrays['projected_fp16'][0,arrays['final_mask'][0]];actual_visual=arrays['prepared_embeddings_fp16'][0,marker:marker+128]
            assert bits(selected,actual_visual) and int(answer[1])==128
            eos=state['loaded'][1].generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
            ending=check_output(arrays['output_ids'],state['loaded'][1].config.vocab_size,state['loaded'][1].generation_config.bos_token_id,eos)
            record=dict(status='captured',arm=arm,position=pos,index=item['index'],group_id=item['group_id'],raw=describe(raw),output_ids_file=describe(ids),response=response,
                capture=describe(capture),
                arrays=array_metadata,small_arrays=small_values,
                derived=dict(selected_visual=meta(selected),prepare_visual=meta(actual_visual),prepare_nonvisual=meta(nonvisual),marker_index=marker),
                calls=obs.calls,observer_errors=errors,encode_texts=obs.encode_texts,guidance=kwargs['texts'],image_sizes=kwargs['image_sizes'],
                generate_scalar_kwargs={k:kwargs[k] for k in PARAMETERS},generate_keys=sorted(kwargs),positional_args_count=len(args),returned_visual_token_num=128,ending=ending)
            if pos in anchor_position:
                i=anchor_position[pos];prior=c.read(PANEL/'execution'/('%s_%02d.json'%(arm,i)));oldraw=c.read(prior['raw']['path'])
                with np.load(prior['archive']['path'],allow_pickle=False) as previous:
                    assert set(arrays)==set(previous.files) and all(bits(arrays[k],previous[k]) for k in arrays)
                assert response==oldraw['response'] and tokens==oldraw['output_ids']
                record['anchor']=dict(panel_index=i,prior_record=describe(PANEL/'execution'/('%s_%02d.json'%(arm,i))),prior_archive=prior['archive'],archive=describe(archive),all_arrays_bitwise_equal=True,output_equal=True)
            record_path=out/(label+'.json');c.save_new(record_path,record);record_paths[(arm,pos)]=describe(record_path);state['arrays']=arrays
        except BaseException as exc:
            c.save_new(out/(label+'.observation_failure.json'),dict(error=repr(exc),traceback=traceback.format_exc(),raw=describe(raw),output_ids_file=describe(ids)));raise
        return answer

    def loader(*args,**kwargs):
        signature=repr((args,kwargs))
        if state['loaded'] is not None:
            assert signature==state['load_signature'];return state['loaded']
        state['load_signature']=signature;loaded=load_pretrained_model(*args,**kwargs);tok,model,proc,context=loaded
        assert model.visual_token_num==128 and model.alpha==.5 and model.beta==1. and str(model.dtype)=='torch.float16' and model.config._attn_implementation=='sdpa'
        state['loaded']=loaded;state['native_generate']=model.generate;state['observer']=Observer(model).install();model.generate=gen
        c.save_new(c.OUT/'runtime.json',dict(plan_sha256=c.sha(c.PLAN),torch=torch.__version__,transformers=transformers.__version__,model_path=str(c.MODEL),model_class=type(model).__module__+'.'+type(model).__name__,model_dtype=str(model.dtype),device=str(model.device),attention_implementation=model.config._attn_implementation,K=128,alpha=.5,beta=1.,vocab_size=model.config.vocab_size,bos_token_id=model.generation_config.bos_token_id,eos_token_id=model.generation_config.eos_token_id,source_sha256=c.sha(c.SOURCE),source_boundary_proof=proof,wait_patch=validation,one_loaded_model=True))
        return loaded
    for ns in modules.values():ns['load_pretrained_model']=loader
    def one(arm,pos):
        state['phase']=arm;state['position']=pos;state['arrays']=None
        # Namespace-local reader only: original entry sees the exact selected row.
        modules[arm]['pd']=SimpleNamespace(read_table=lambda path:frame.iloc[[pos]].copy())
        args=argparse.Namespace(model_path=str(c.MODEL),model_base=None,question_file=str(c.HERE/'inputs/panel.tsv'),answers_file=str(c.OUT/'author'/('%s_%04d.jsonl'%(arm,pos))),conv_mode='vicuna_v1',num_chunks=1,chunk_idx=0,all_rounds=False,single_pred_prompt=True,lang='en',temperature=0.,top_p=None,num_beams=1,visual_token_num=128,beta=1.,alpha=.5)
        modules[arm]['eval_model'](args);answer=lines(args.answers_file);assert len(answer)==1
        record=c.read(record_paths[(arm,pos)]['path']);a=answer[0];assert a['question_id']==rows[pos]['index'] and a['round_id']==0 and a['prompt']==rows[pos]['full_guidance'] and a['text']==record['response']
        return state['arrays'],describe(args.answers_file)
    def pair(pos):
        left,left_author=one('source',pos);right,right_author=one('question_only',pos)
        comp=check_pair(left,right);record=dict(position=pos,index=rows[pos]['index'],source=record_paths[('source',pos)],question_only=record_paths[('question_only',pos)],source_author=left_author,question_only_author=right_author,direct_actual_bitwise_pair_passed=True,comparison=comp)
        path=c.OUT/'pairs'/('%04d.json'%pos);c.save_new(path,record);pair_paths[pos]=describe(path)
    try:
        for pos in [r['original_position'] for r in panel]:pair(pos)
        assert len(pair_paths)==60
        c.save_new(c.OUT/'anchor_gate.json',dict(passed=True,n_rows=60,n_generations=120,all_anchors_before_remaining=True,pairs=[pair_paths[r['original_position']] for r in panel],prior_panel_root_proof_sha256=c.sha(PANEL/'root_exit_proof.json'),plan_sha256=c.sha(c.PLAN)))
        for pos in range(4377):
            if pos not in anchor_position:pair(pos)
        assert len(pair_paths)==4377 and len(record_paths)==8754
        ordered={}
        for arm in ARMS:
            pred=c.OUT/(arm+'_predictions.jsonl');obs=c.OUT/(arm+'_observations.jsonl')
            with pred.open('x') as pf,obs.open('x') as of:
                for pos,item in enumerate(rows):
                    record=c.read(record_paths[(arm,pos)]['path']);raw=c.read(record['raw']['path']);predrow=dict(position=pos,index=item['index'],group_id=item['group_id'],text=raw['response'],output_ids=raw['output_ids'])
                    pf.write(json.dumps(predrow,ensure_ascii=False)+'\n');of.write(json.dumps(dict(position=pos,index=item['index'],record=record_paths[(arm,pos)],pair=pair_paths[pos]),ensure_ascii=False)+'\n')
                pf.flush();os.fsync(pf.fileno());of.flush();os.fsync(of.fileno())
            ordered[arm]=dict(predictions=describe(pred),observations=describe(obs))
        c.save_new(c.OUT/'native_complete.json',dict(status='complete',plan_sha256=c.sha(c.PLAN),runtime=describe(c.OUT/'runtime.json'),anchor_gate=describe(c.OUT/'anchor_gate.json'),ordered=ordered,n_rows=4377,n_generations=8754,scoring=False))
    except BaseException as exc:
        c.save_new(c.OUT/'worker_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),phase=state['phase'],position=state['position']));raise
    finally:
        if state['loaded'] is not None:state['loaded'][1].generate=state['native_generate']
        if state['observer'] is not None:state['observer'].close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--allocation-note',required=True);a=p.parse_args();run(a.allocation_note)
