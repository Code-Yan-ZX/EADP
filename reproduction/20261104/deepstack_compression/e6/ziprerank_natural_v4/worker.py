"""Three DEV anchors, natural generation with explicit ZipRerank adapters.

No reference labels or scoring. All enabled arms use local RoPE-state repair;
the text arm additionally substitutes full user-message embeddings.
"""
import argparse, importlib.util, inspect, json, os, sys, traceback
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/media/disk2/YZX/research/EADP_v2')
BASE=ROOT/'reproduction/20261104/deepstack_compression'
HERE=BASE/'e6/ziprerank_natural_v4'
sys.path.insert(0,str(BASE/'e6/ziprerank_qtext_smoke_v2'))
spec=importlib.util.spec_from_file_location('zip_diagnostic', BASE/'e6/ziprerank_qtext_smoke_v2/worker.py')
diagnostic=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnostic)
sha,save,progress=diagnostic.sha,diagnostic.save,diagnostic.progress
tensor_sha,cache_summary,configure_rope_state=diagnostic.tensor_sha,diagnostic.cache_summary,diagnostic.configure_rope_state
ZIP,AUTHOR,MODEL=diagnostic.ZIP,diagnostic.AUTHOR,diagnostic.MODEL
locate_user_text,traced_selector=diagnostic.locate_user_text,diagnostic.traced_selector
sys.path.insert(0,str(BASE/'e6/selected_smoke'))
from generation_contract import controls


def weight_stats(weights):
    result=diagnostic.weight_stats(weights)
    for rec in result:rec['ctime_ns']=Path(rec['path']).stat().st_ctime_ns
    return result


def preflight(plan):
    for path,expected in plan['sha256'].items():
        if sha(path)!=expected:raise ValueError('source mismatch: '+path)
    rows=[json.loads(line) for line in (BASE/'e1/input_preparation_002/inputs.jsonl').read_text().splitlines()]
    selected=[]
    for anchor in plan['anchors']:
        matches=[r for r in rows if r['task']==anchor['task'] and str(r['qid'])==anchor['qid']]
        if len(matches)!=1:raise ValueError('anchor uniqueness')
        row=matches[0]
        if row['group_id']!=anchor['group_id'] or row['message_sha256']!=anchor['message_sha256'] or sha(row['image_path'])!=anchor['image_sha256']:
            raise ValueError('anchor identity')
        if [x['type'] for x in row['message']]!=['image','text']:raise ValueError('message order')
        selected.append(row)
    return selected


def persist_then_decode(generated,seq_len,path,decoder,expected_prefix):
    """Fsync raw generated token IDs before any decode, cache or summary work."""
    raw=generated.sequences.detach().cpu().tolist()
    save(path,{'input_sequence_length':seq_len,'sequences':raw})
    if len(raw)!=1 or len(raw[0])<seq_len:raise ValueError('unexpected generated sequence shape')
    if raw[0][:seq_len]!=expected_prefix:raise ValueError('generated prompt prefix mismatch')
    tokens=raw[0][seq_len:]
    return tokens,decoder(tokens)


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    plan=json.loads((HERE/'plan001.json').read_text());rows=preflight(plan)
    import torch,transformers
    from transformers import AutoProcessor
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
    from models.qwen3vl_with_qi_early import Qwen3VLWithQIEarly
    if Path(inspect.getfile(Qwen3VLWithQIEarly)).resolve()!=ZIP/'models/qwen3vl_with_qi_early.py':raise ValueError('wrong author class')
    import selector,selector_trace,generation_contract,transformers.models.qwen3_vl.modeling_qwen3_vl as hf_model
    expected={selector:BASE/'e6/ziprerank_qtext/selector.py',selector_trace:BASE/'e6/ziprerank_qtext_smoke_v2/selector_trace.py',
              generation_contract:BASE/'e6/selected_smoke/generation_contract.py',
              hf_model:ROOT/'_reproduction/envs/qwen_author/lib/python3.10/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py'}
    if any(Path(inspect.getfile(obj)).resolve()!=path.resolve() for obj,path in expected.items()):raise ValueError('import path mismatch')
    if Path(inspect.getfile(Qwen3VLChatFixedRes)).resolve()!=AUTHOR/'vlmeval/vlm/qwen3_vl/model_fixed_res.py':raise ValueError('wrapper path mismatch')
    if transformers.__version__!='4.57.1':raise ValueError('transformers version mismatch')
    weights=json.loads((ROOT/'reproduction/20261028/qwen/qwen_weight_identity.json').read_text());stats=weight_stats(weights)
    if a.preflight_only:
        if torch.cuda.is_initialized():raise ValueError('CUDA initialized')
        for item in weights['files']:
            if sha(item['path'])!=item['expected_lfs_sha256']:raise ValueError('weight SHA mismatch')
        if weight_stats(weights)!=stats:raise ValueError('weight changed during hash')
        save(a.out,{'status':'passed','cuda_initialized':False,'model_loaded':False,'GT_opened':False,'source_plan_sha256':sha(HERE/'plan001.json'),
                    'torch':torch.__version__,'transformers':transformers.__version__,'weight_sha256_verified':4,'weight_stats':stats});return
    if stats!=json.loads((HERE/'preflight001.json').read_text())['weight_stats']:raise ValueError('weights changed')
    if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('GPU mismatch')
    a.out.mkdir(exist_ok=False);progress(a.out/'progress.jsonl',{'event':'worker_started'})
    save(a.out/'started.json',{'plan_sha256':sha(HERE/'plan001.json'),'GT_opened':False,'scoring':False})
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    model=Qwen3VLWithQIEarly.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to('cuda').eval()
    if model.dtype!=torch.bfloat16 or model.device.type!='cuda' or model.config._attn_implementation!='flash_attention_2':raise ValueError('model dtype, device or backend mismatch')
    wrapper=object.__new__(Qwen3VLChatFixedRes);wrapper.system_prompt=None;wrapper.min_pixels=wrapper.max_pixels=1048576
    wrapper.total_pixels=None;wrapper.processor=processor;wrapper.model=SimpleNamespace(device='cuda',dtype=torch.bfloat16)
    kwargs=dict(do_sample=False,use_cache=True,max_new_tokens=2048,top_p=.001,top_k=1,temperature=.01,repetition_penalty=1.0,return_dict_in_generate=True)
    statuses={}
    for row in rows:
        task=row['task'];qid=str(row['qid']);messages=wrapper._build_messages(row['message'],dataset=task)
        inputs=wrapper._processor_inputs(messages);ids=inputs['input_ids'];seq_len=ids.shape[1]
        positions,delta=model.model.get_rope_index(ids,inputs['image_grid_thw'],None,inputs['attention_mask'])
        image_positions=torch.where(ids[0]==model.config.image_token_id)[0]
        if image_positions.numel()!=1024:raise ValueError('expected 1024 image tokens')
        user_positions=locate_user_text(processor,messages,row['message'][1]['value'],ids,model.config.image_token_id)
        frozen_ids=ids.clone();digest=tensor_sha(ids);expected_prefix=ids[0].detach().cpu().tolist()
        for name,enabled,ratio in [('disabled',False,1.),('keep_all',True,1.),('author_prefix_256',True,.25),('user_text_256',True,.25)]:
            key=task+'_'+qid+'_'+name;progress(a.out/'progress.jsonl',{'event':'arm_started','key':key})
            save(a.out/(key+'_started.json'),{'task':task,'qid':qid,'arm':name,'prompt_sha256':digest})
            selected=[];calls=[];prefill={};trace={'calls':0}
            def pruner(_m,_a,output):selected.extend(x.detach().cpu().tolist() for x in output[1])
            def lm_pre(_m,_a,kw):
                pos=kw.get('position_ids');mask=kw.get('visual_pos_masks');ds=kw.get('deepstack_visual_embeds');emb=kw.get('inputs_embeds')
                if not calls:
                    chosen=selected[0] if enabled and len(selected)==1 else list(range(1024)) if not enabled else []
                    kept=torch.ones(seq_len,dtype=torch.bool,device=ids.device);kept[image_positions]=False
                    if chosen:kept[image_positions[torch.tensor(chosen,device=ids.device)]]=True
                    prefill.update(selected_count=len(chosen),selected_unique=len(set(chosen))==len(chosen),
                        selected_in_range=all(0<=x<1024 for x in chosen),positions_match=pos is not None and torch.equal(pos,positions[:,:,kept]),
                        visual_count=None if mask is None else int(mask.sum().item()),ds_rows=None if ds is None else [int(x.shape[0]) for x in ds])
                calls.append({'input_length':None if emb is None else int(emb.shape[1]),'cache_length_before':None if kw.get('past_key_values') is None else int(kw['past_key_values'].get_seq_length())})
                if len(calls)%25==0:progress(a.out/'progress.jsonl',{'event':'decode_progress','key':key,'calls':len(calls)})
            handles=[]
            before=('_extract_query_text_embeds' in model.__dict__,model.__dict__.get('_extract_query_text_embeds'))
            rec={'task':task,'qid':qid,'group_id':row['group_id'],'arm':name,'local_rope_adapter':enabled,
                 'full_user_text_adapter':name=='user_text_256',
                 'prompt_sha256':digest,'user_text_token_count':int(user_positions.numel())}
            try:
                configure_rope_state(model,enabled,delta);model.set_qi_early_enabled(enabled);model.set_qi_early_keep_ratio(ratio)
                rec['generation_controls']=controls(model.generation_config,kwargs,False)
                handles.append(model.qi_early_pruner.register_forward_hook(pruner))
                handles.append(model.model.language_model.register_forward_pre_hook(lm_pre,with_kwargs=True))
                ctx=traced_selector(model,ids,user_positions) if name=='user_text_256' else nullcontext(trace)
                with ctx as trace,torch.inference_mode():
                    generated=model.generate(**inputs,**kwargs)
                    tokens,answer=persist_then_decode(generated,seq_len,a.out/(key+'_generated.json'),lambda x:processor.tokenizer.decode(x,skip_special_tokens=True),expected_prefix)
                torch.cuda.synchronize();eos=rec['generation_controls']['eos_token_id'];eos=[eos] if isinstance(eos,int) else eos or []
                rec.update(status='completed',output_ids=tokens,answer=answer,
                           eos=bool(tokens and tokens[-1] in eos),max_new_tokens_reached=bool(len(tokens)>=2048),
                           truncated=bool(len(tokens)>=2048 and not (tokens and tokens[-1] in eos)),
                           final_cache=cache_summary(generated.past_key_values))
            except Exception as exc:
                rec.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
            finally:
                for handle in handles:handle.remove()
            rec.update(selection_trace=trace,selected_indices=selected,prefill=prefill,calls=calls,
                       prompt_ids_unchanged=bool(torch.equal(ids,frozen_ids)),
                       selector_restored=before==('_extract_query_text_embeds' in model.__dict__,model.__dict__.get('_extract_query_text_embeds')))
            save(a.out/(key+'.json'),rec);statuses[key]=rec['status'];progress(a.out/'progress.jsonl',{'event':'arm_recorded','key':key,'status':rec['status']})
    if weight_stats(weights)!=stats:raise ValueError('weights changed during inference')
    save(a.out/'worker_complete.json',{'status':'attempts_recorded','statuses':statuses,'no_GT':True,'scoring':False,'limitations':'3 anchors only; natural generations; both enabled adapters are local; no quality comparison'})

if __name__=='__main__':main()
