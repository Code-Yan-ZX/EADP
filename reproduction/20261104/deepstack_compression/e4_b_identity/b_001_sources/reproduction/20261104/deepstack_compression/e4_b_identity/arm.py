"""Diagnostic C-DROP/C-SKIP natural arm; not a timing implementation."""
from contextlib import ExitStack
from pathlib import Path
import sys
E4=Path(__file__).resolve().parent.parent/'e4';sys.path.insert(1,str(E4))
NATIVE=E4.parent/'e4_native';sys.path.insert(1,str(NATIVE))
from decode_observer import DecodeObserver,validate_trace
from identity_records import make_record,tensor_meta
from native_capture import CacheFingerprints
from b_imports import local
ObservedOffAdapter=local('off_capture').ObservedOffAdapter

def run_arm(torch,w,pruner,sample,mode,generate,on_generated,archive_arrays,forced_token=None):
    seen={}
    with ExitStack() as stack:
        adapter=stack.enter_context(ObservedOffAdapter(torch,w,pruner,mode));s=adapter.state
        previous=w._processor_inputs
        def processor(*a,**kw):
            x=previous(*a,**kw)
            s['inputs']={k:(v.detach().clone() if hasattr(v,'detach') else v) for k,v in x.items()}
            return x
        w._processor_inputs=processor;stack.callback(setattr,w,'_processor_inputs',previous)
        def snapshot_visual(_m,_args,out):
            s['visual']=out[0].detach().clone()
        def snapshot_pruned(_m,_args,out):
            s['pruned']=out[0].detach().clone()
        h=w.model.visual.register_forward_hook(snapshot_visual);stack.callback(h.remove)
        h=w.pruner.register_forward_hook(snapshot_pruned);stack.callback(h.remove)
        def logits(_module,_args,out):
            if 'logits' not in seen:seen['logits']=out[:,-1,:].detach().clone()
        h=w.model.lm_head.register_forward_hook(logits);stack.callback(h.remove)
        original=w.model.generate
        def capture(*args,**kw):
            if 'output_ids' in seen:raise ValueError('multiple generate calls')
            if kw.get('max_new_tokens')!=2048:raise ValueError('wrong natural generation limit')
            if forced_token is not None:
                from transformers import LogitsProcessor,LogitsProcessorList
                class ForceToken(LogitsProcessor):
                    def __call__(self,_ids,scores):scores.fill_(-float('inf'));scores[:,forced_token]=0;return scores
                kw.update(max_new_tokens=33,min_new_tokens=33,eos_token_id=[],logits_processor=LogitsProcessorList([ForceToken()]))
            eos=kw.get('eos_token_id',w.model.generation_config.eos_token_id)
            seen['eos_ids']=[eos] if isinstance(eos,int) else list(eos or [])
            names=('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty')
            seen['generation_scalars']={key:kw.get(key) for key in names}
            seen['min_new_tokens']=kw.get('min_new_tokens')
            result=original(*args,**kw);seen['output_ids']=result.detach().cpu().tolist()[0]
            on_generated({'task':sample['task'],'qid':str(sample['qid']),'group_id':sample['group_id'],
                          'mode':mode,'forced':forced_token is not None,'output_ids':seen['output_ids'],'eos_ids':seen['eos_ids'],
                          'generation_scalars':seen['generation_scalars'],'min_new_tokens':seen['min_new_tokens']})
            return result
        w.model.generate=capture;stack.callback(setattr,w.model,'generate',original)
        with DecodeObserver(w) as trace,CacheFingerprints(w,forced_token is not None) as kv:answer=generate(w,sample)
        if not seen.get('output_ids') or 'logits' not in seen or not torch.isfinite(seen['logits']).all():
            raise ValueError('missing or nonfinite prefill logits or generation')
        record=make_record(adapter,sample,seen['output_ids'],answer,seen['logits'],seen['eos_ids'],
                           max_new_tokens=33 if forced_token is not None else 2048)
        if forced_token is not None:
            if seen['output_ids']!=[forced_token]*33 or seen['eos_ids']!=[]:raise ValueError('forced prefix mismatch')
            record['truncated']=False
        elif not (record['eos'] or record['truncated']):raise ValueError('invalid generation endpoint')
        validate_trace(trace.trace,seen['output_ids'],len(s['retained']),int(s['adjusted_delta'][0,0]))
        inp=s['inputs'];tensors={'visual':s['visual'],'pruned':s['pruned'],'prefill':s['prefill_embeds'],
            'prefill_logits':seen['logits'],**{'processor_'+k:inp[k] for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}}
        if mode=='C-DROP':tensors.update({'ds'+str(i):x for i,x in enumerate(s['deepstack'])})
        record.update(image_start=s['start'],original_length=inp['input_ids'].shape[1],
                      full_deepstack=[tensor_meta(x) for x in s['deepstack']] if mode=='C-DROP' else None,
                      generation_scalars=seen['generation_scalars'],min_new_tokens=seen['min_new_tokens'],
                      forced=forced_token is not None,forced_token=forced_token,decode_trace=trace.trace,
                      cache_fingerprints=kv.fingerprints,arrays=archive_arrays(tensors))
        return record
