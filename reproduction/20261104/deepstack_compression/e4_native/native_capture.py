"""Actual stock path capture; only observation hooks, no visual intervention."""
from contextlib import ExitStack
from native_contract import E4
from decode_observer import DecodeObserver,validate_trace
from identity_records import tensor_meta


class CacheFingerprints:
    def __init__(self,w,all_steps):self.w=w;self.all_steps=all_steps;self.fingerprints=[]
    def __enter__(self):
        def observe(_module,_args,_kwargs,out):
            if self.fingerprints and not self.all_steps:return
            cache=out.past_key_values
            if cache is None or len(cache.layers)!=36:raise ValueError('expected 36 actual KV cache layers')
            self.fingerprints.append([{'key':tensor_meta(layer.keys),'value':tensor_meta(layer.values)} for layer in cache.layers])
        self.handle=self.w.model.model.language_model.register_forward_hook(observe,with_kwargs=True)
        return self
    def __exit__(self,*_args):self.handle.remove()


def run_native(torch,w,sample,generate,on_generated,archive_arrays,forced_token=None):
    forced=forced_token is not None;seen={};snapshot={}
    with ExitStack() as stack:
        original_processor=w._processor_inputs;had='_processor_inputs' in w.__dict__;raw=w.__dict__.get('_processor_inputs')
        def processor(*args,**kwargs):
            inp=original_processor(*args,**kwargs)
            snapshot['inputs']={k:v.detach().clone() for k,v in inp.items() if hasattr(v,'detach')}
            ids=inp['input_ids'];pos=(ids[0]==w.model.config.image_token_id).nonzero(as_tuple=True)[0]
            if ids.shape[0]!=1 or pos.numel()!=1024 or not torch.equal(pos,torch.arange(pos[0],pos[0]+1024,device=pos.device)):
                raise ValueError('native input is not a single 1024-token image')
            return inp
        w._processor_inputs=processor
        def restore():
            if had:w._processor_inputs=raw
            else:delattr(w,'_processor_inputs')
        stack.callback(restore)
        def hook(module,fn,pre=False):
            h=(module.register_forward_pre_hook(fn,with_kwargs=True) if pre else module.register_forward_hook(fn,with_kwargs=True))
            stack.callback(h.remove)
        def visual(_module,_args,_kwargs,out):
            if 'visual' in snapshot:raise ValueError('multiple native vision forwards')
            if not isinstance(out,(tuple,list)) or len(out)!=2 or len(out[1])!=3 or any(x is None for x in out[1]):
                raise ValueError('native visual output lacks three streams')
            snapshot['visual']=out[0].detach().clone();snapshot['ds']=[x.detach().clone() for x in out[1]]
        def prefill(_module,_args,kw):
            if 'prefill' in snapshot:return
            def clone(x):
                if x is None:return None
                if isinstance(x,(tuple,list)):return [clone(y) for y in x]
                return x.detach().clone()
            snapshot['prefill']={k:clone(kw.get(k)) for k in ('inputs_embeds','attention_mask','position_ids','visual_pos_masks','deepstack_visual_embeds')}
            snapshot['delta']=w.model.model.rope_deltas.detach().clone()
        def logits(_module,_args,_kwargs,out):
            if 'logits' not in snapshot:snapshot['logits']=out[:,-1,:].detach().clone()
        hook(w.model.visual,visual);hook(w.model.model.language_model,prefill,pre=True);hook(w.model.lm_head,logits)
        original=w.model.generate
        def capture(*args,**kw):
            if seen:raise ValueError('multiple native generations')
            if 'input_ids' not in kw or kw.get('inputs_embeds') is not None:raise ValueError('not native input-ID generation')
            if forced:
                from transformers import LogitsProcessor,LogitsProcessorList
                class ForceToken(LogitsProcessor):
                    def __call__(self,_ids,scores):scores.fill_(-float('inf'));scores[:,forced_token]=0;return scores
                kw.update(max_new_tokens=33,min_new_tokens=33,eos_token_id=[],logits_processor=LogitsProcessorList([ForceToken()]))
            names=('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty')
            seen['generation_scalars']={key:kw.get(key) for key in names}
            eos=kw.get('eos_token_id',w.model.generation_config.eos_token_id);seen['eos_ids']=[eos] if isinstance(eos,int) else list(eos or [])
            result=original(*args,**kw);ids=result.detach().cpu().tolist()[0];prompt=kw['input_ids'].detach().cpu().tolist()[0]
            if ids[:len(prompt)]!=prompt:raise ValueError('native generation did not retain exact prompt IDs')
            seen['output_ids']=ids[len(prompt):]
            on_generated({'path':'stock_native','task':sample['task'],'qid':str(sample['qid']),'group_id':sample['group_id'],
                          'forced':forced,'output_ids':seen['output_ids'],'eos_ids':seen['eos_ids'],'generation_scalars':seen['generation_scalars']})
            return result
        w.model.generate=capture;stack.callback(setattr,w.model,'generate',original)
        with DecodeObserver(w) as trace,CacheFingerprints(w,forced) as kv:
            answer=generate(w,sample)
        if not all(k in snapshot for k in ('visual','ds','prefill','delta','inputs','logits')) or 'output_ids' not in seen:
            raise ValueError('native observations incomplete')
        ids=seen['output_ids'];eos=seen['eos_ids'];maximum=seen['generation_scalars']['max_new_tokens']
        if not ids or maximum!=(33 if forced else 2048) or len(ids)>maximum or not isinstance(answer,str):raise ValueError('invalid native output')
        if forced and (ids!=[forced_token]*33 or eos!=[]):raise ValueError('native fixed-prefix mismatch')
        if not torch.isfinite(snapshot['logits']).all():raise ValueError('native nonfinite prefill logits')
        inp=snapshot['inputs'];pf=snapshot['prefill'];delta=snapshot['delta'];full_length=inp['input_ids'].shape[1]
        validate_trace(trace.trace,ids,full_length,int(delta[0,0]))
        if pf['visual_pos_masks'] is None or not isinstance(pf['deepstack_visual_embeds'],(tuple,list)) or len(pf['deepstack_visual_embeds'])!=3:
            raise ValueError('native did not inject all DS streams')
        tensors={'visual':snapshot['visual'],'prefill':pf['inputs_embeds'],'prefill_logits':snapshot['logits'],
                 **{'ds'+str(i):x for i,x in enumerate(snapshot['ds'])},
                 **{'injected'+str(i):x for i,x in enumerate(pf['deepstack_visual_embeds'])},
                 **{'processor_'+k:inp[k] for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}}
        arrays=archive_arrays(tensors)
        return {'path':'stock_native','task':sample['task'],'qid':str(sample['qid']),'group_id':sample['group_id'],
            'image_sha256':sample['image_sha256'],'message_sha256':sample['message_sha256'],'forced':forced,
            'processor':{k:tensor_meta(inp[k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},
            'main_visual':tensor_meta(snapshot['visual']),'full_deepstack':[tensor_meta(x) for x in snapshot['ds']],
            'prefill_embeds':tensor_meta(pf['inputs_embeds']),'prefill_mask':tensor_meta(pf['attention_mask']),
            'prefill_position_ids':pf['position_ids'].detach().cpu().tolist(),'rope_delta':delta.detach().cpu().tolist(),
            'injected_mask':pf['visual_pos_masks'].detach().cpu().tolist(),
            'injected_deepstack':[tensor_meta(x) for x in pf['deepstack_visual_embeds']],
            'prefill_logits':tensor_meta(snapshot['logits']),'output_ids':ids,'answer':answer,'eos_ids':eos,
            'eos':ids[-1] in eos,'truncated':not forced and ids[-1] not in eos and len(ids)>=maximum,
            'max_new_tokens':maximum,'generation_scalars':seen['generation_scalars'],'decode_trace':trace.trace,
            'cache_fingerprints':kv.fingerprints,'arrays':arrays}
