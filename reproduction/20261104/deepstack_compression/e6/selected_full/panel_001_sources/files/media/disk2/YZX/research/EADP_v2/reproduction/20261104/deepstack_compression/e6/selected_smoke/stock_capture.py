"""Observe unmodified author Div/CD K256 natural generation; no selector changes.

The caller owns all admission, no-GT inventory, lifecycle, recipe and archival
controls. Raw token exposure must be fsynced by the provided callback.
"""
from contextlib import ExitStack
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];E4=ROOT/'e4';BASE=HERE.parent/'selected_baselines'
sys.path[:0]=[str(BASE),str(E4)]
from recover_indices import recover
from identity_records import tensor_meta
from decode_observer import DecodeObserver,validate_trace
from generation_contract import controls

def run_stock(torch,w,method,row,generate,on_generated,archive,expected_scalars,forced_token=None):
    expected={'div':'DivPruner','cd':'CDPruner'}
    if method not in expected or type(w.pruner).__name__!=expected[method] or row.get('split')!='DEV':
        raise ValueError('wrong DEV author selector')
    if w.visual_token_num!=256 or w.pruner.visual_token_num!=256:raise ValueError('wrong budget')
    forced=forced_token is not None
    if forced and (type(forced_token) is not int or forced_token<0):raise ValueError('invalid forced token')
    state={};seen={}
    with ExitStack() as stack:
        prior=w._processor_inputs;had='_processor_inputs' in w.__dict__;raw=w.__dict__.get('_processor_inputs')
        def processor(*args,**kwargs):
            out=prior(*args,**kwargs)
            if 'inputs' in state:raise ValueError('multiple processor invocations')
            state['inputs']={k:out[k].detach().clone() for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
            ids=out['input_ids'];pos=(ids[0]==w.model.config.image_token_id).nonzero(as_tuple=True)[0]
            if ids.shape[0]!=1 or pos.numel()!=1024 or not torch.equal(pos,torch.arange(pos[0],pos[0]+1024,device=pos.device)):
                raise ValueError('not one contiguous 1024-token image')
            state['start']=int(pos[0]);return out
        w._processor_inputs=processor
        def restore():
            if had:w._processor_inputs=raw
            else:delattr(w,'_processor_inputs')
        stack.callback(restore)
        def visual(_m,_a,result):
            if 'visual' in state or not isinstance(result,(tuple,list)) or len(result)!=2 or len(result[1])!=3 or any(x is None for x in result[1]):
                raise ValueError('wrong author visual output')
            state['visual']=result[0].detach().clone();state['ds']=[x.detach().clone() for x in result[1]]
        def pruner(_m,_a,result):
            if 'pruned' in state or not isinstance(result,(tuple,list)) or len(result)!=2 or result[1]!=[256]:
                raise ValueError('wrong author selector output')
            state['pruned']=result[0].detach().clone()
        def prefill(_m,_a,kw):
            if 'prefill' in state:return
            def clone(x):
                if x is None:return None
                if isinstance(x,(tuple,list)):return [clone(y) for y in x]
                return x.detach().clone()
            state['prefill']={k:clone(kw.get(k)) for k in ('inputs_embeds','attention_mask','position_ids','visual_pos_masks','deepstack_visual_embeds')}
            state['delta']=w.model.model.rope_deltas.detach().clone()
        def logits(_m,_a,out):
            if 'logits' not in state:state['logits']=out[:,-1,:].detach().clone()
        for module,fn,pre in ((w.model.visual,visual,False),(w.pruner,pruner,False),
                              (w.model.model.language_model,prefill,True),(w.model.lm_head,logits,False)):
            h=module.register_forward_pre_hook(fn,with_kwargs=True) if pre else module.register_forward_hook(fn)
            stack.callback(h.remove)
        original=w.model.generate
        def capture(*args,**kwargs):
            if seen:raise ValueError('multiple generations')
            if forced:
                from transformers import LogitsProcessor,LogitsProcessorList
                class ForceToken(LogitsProcessor):
                    def __call__(self,_input_ids,scores):
                        scores.fill_(-float('inf'));scores[:,forced_token]=0
                        return scores
                kwargs.update(max_new_tokens=33,min_new_tokens=33,eos_token_id=[],logits_processor=LogitsProcessorList([ForceToken()]))
            names=('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty')
            scalars={k:kwargs.get(k) for k in names}
            wanted=dict(expected_scalars)
            if forced:wanted['max_new_tokens']=33
            if scalars!=wanted:raise ValueError('stock generation recipe differs')
            settings=controls(w.model.generation_config,kwargs,forced)
            eos=kwargs.get('eos_token_id',w.model.generation_config.eos_token_id);eos=[eos] if isinstance(eos,int) else list(eos or [])
            result=original(*args,**kwargs);ids=result.detach().cpu().tolist()[0]
            on_generated({'method':method,'path':'author_stock','task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],
                          'forced':forced,'output_ids':ids,'eos_ids':eos,'generation_scalars':scalars,'generation_controls':settings})
            seen.update(output_ids=ids,eos_ids=eos,generation_scalars=scalars,generation_controls=settings)
            return result
        w.model.generate=capture;stack.callback(setattr,w.model,'generate',original)
        with DecodeObserver(w) as trace:answer=generate(w,row)
    if not all(k in state for k in ('inputs','visual','ds','pruned','prefill','delta','logits')) or not seen:raise ValueError('stock observations incomplete')
    if not isinstance(answer,str) or not seen['output_ids'] or not torch.isfinite(state['logits']).all():raise ValueError('invalid stock output')
    ids=recover(torch,state['visual'],state['pruned']);inp=state['inputs'];pf=state['prefill'];start=state['start']
    if len(ids)!=256 or pf['visual_pos_masks'] is not None or pf['deepstack_visual_embeds'] is not None:raise ValueError('not author stock DS path')
    full,delta=w.model.model.get_rope_index(inp['input_ids'],inp['image_grid_thw'],None,attention_mask=inp['attention_mask'])
    retained=torch.tensor(list(range(start))+[start+i for i in ids]+list(range(start+1024,inp['input_ids'].shape[1])),device=inp['input_ids'].device)
    embeds=w.model.get_input_embeddings()(inp['input_ids']);expected=torch.cat((embeds[:,:start],state['pruned'].unsqueeze(0),embeds[:,start+1024:]),dim=1)
    if not torch.equal(expected,pf['inputs_embeds']) or not torch.equal(inp['attention_mask'].index_select(1,retained),pf['attention_mask']):
        raise ValueError('stock prefill differs from selected main')
    tokens=seen['output_ids'];end=bool(tokens[-1] in seen['eos_ids']);truncated=not end and len(tokens)>=2048
    if (forced and (tokens!=[forced_token]*33 or seen['eos_ids']!=[]) or
        not forced and (not(end or truncated) or len(tokens)>2048)):
        raise ValueError('stock generation endpoint')
    validate_trace(trace.trace,tokens,len(retained),int(state['delta'].item()))
    tensors={'visual':state['visual'],'pruned':state['pruned'],'prefill':pf['inputs_embeds'],'prefill_logits':state['logits'],
             **{'ds'+str(i):x for i,x in enumerate(state['ds'])},**{'processor_'+k:v for k,v in inp.items()}}
    arrays=archive(tensors)
    if (not isinstance(arrays,dict) or set(arrays)!={'kind','path','sha256'} or
        arrays['kind']!='full_arrays' or not isinstance(arrays['path'],str) or
        not isinstance(arrays['sha256'],str) or len(arrays['sha256'])!=64 or
        any(c not in '0123456789abcdef' for c in arrays['sha256'])):
        raise ValueError('full raw array archive evidence required')
    return {'method':method,'path':'author_stock','task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],
            'image_sha256':row['image_sha256'],'message_sha256':row['message_sha256'],'forced':forced,'forced_token':forced_token,
            'processor':{k:tensor_meta(v) for k,v in inp.items()},'main_visual':tensor_meta(state['visual']),
            'full_deepstack':[tensor_meta(x) for x in state['ds']],'pruned_visual':tensor_meta(state['pruned']),
            'image_start':start,'original_length':inp['input_ids'].shape[1],'spatial_ids':ids,'retained_ids':retained.detach().cpu().tolist(),'full_position_ids':full.detach().cpu().tolist(),
            'full_delta':delta.detach().cpu().tolist(),'prefill_position_ids':pf['position_ids'].detach().cpu().tolist(),
            'rope_delta':state['delta'].detach().cpu().tolist(),'prefill_embeds':tensor_meta(pf['inputs_embeds']),
            'prefill_mask':tensor_meta(pf['attention_mask']),'prefill_logits':tensor_meta(state['logits']),
            'injected_mask':None,'injected_deepstack':None,'decode_trace':trace.trace,'output_ids':tokens,'answer':answer,
            'eos_ids':seen['eos_ids'],'eos':end,'truncated':truncated,'generation_scalars':seen['generation_scalars'],
            'generation_controls':seen['generation_controls'],'arrays':arrays}
