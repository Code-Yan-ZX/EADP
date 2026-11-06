"""Diagnostic-only C-DROP/C-SKIP EADP K256, P3 DS-off adapter.

This hook-based adapter is deliberately instrumented. It is for identity
verification, not timing, and must run in an isolated single-threaded process.
"""
from contextlib import ExitStack
from skip_deepstack import SkipDeepStackMergers, require_skip_visual_result


class E4OffAdapter:
    def __init__(self, torch, wrapper, pruner_module, mode):
        if mode not in ('C-DROP','C-SKIP'): raise ValueError('unsupported DS-off mode')
        self.torch,self.w,self.pruner_module,self.mode=torch,wrapper,pruner_module,mode
        self.state={'llm_calls':0,'visual_calls':0,'greedy_calls':0,'block_calls':[],
                    'main_merger_calls':0,'ds_calls':[0,0,0],'ds_results':[]}
        self.stack=None;self.entered_once=False

    def __enter__(self):
        if self.stack is not None or self.entered_once:raise RuntimeError('adapter cannot be re-entered')
        self.entered_once=True
        stack=ExitStack();self.stack=stack
        try:
            v=self.w.model.visual
            self.state['block_calls']=[0]*len(v.blocks)
            if self.mode=='C-SKIP':self.skip=stack.enter_context(SkipDeepStackMergers(v,audit_calls=True))
            original_processor=self.w._processor_inputs
            had_processor='_processor_inputs' in self.w.__dict__
            raw_processor=self.w.__dict__.get('_processor_inputs')
            def processor(*args,**kw):
                inputs=original_processor(*args,**kw);ids=inputs['input_ids']
                if ids.shape[0]!=1 or inputs.get('pixel_values_videos') is not None:raise ValueError('single image only')
                positions=(ids[0]==self.w.model.config.image_token_id).nonzero(as_tuple=True)[0]
                if positions.numel()!=1024 or not self.torch.equal(positions,self.torch.arange(positions[0],positions[0]+1024,device=positions.device)):
                    raise ValueError('expected contiguous 1024 image tokens')
                self.state.update(inputs=inputs,start=int(positions[0]))
                return inputs
            self.w._processor_inputs=processor
            def restore_processor():
                if had_processor:self.w._processor_inputs=raw_processor
                else:delattr(self.w,'_processor_inputs')
            stack.callback(restore_processor)
            original_greedy=self.pruner_module._greed_select_impl
            def observe_greedy(*args,**kw):
                result=original_greedy(*args,**kw)
                self.state['greedy_calls']+=1
                self.state['ordered']=result[0].detach().clone().reshape(-1)
                self.state['quotas']=result[1].detach().clone().reshape(-1)
                return result
            self.pruner_module._greed_select_impl=observe_greedy
            stack.callback(setattr,self.pruner_module,'_greed_select_impl',original_greedy)
            def hook(module,fn):
                handle=module.register_forward_hook(fn);stack.callback(handle.remove)
            def visual_hook(_module,_args,result):
                self.state['visual_calls']+=1
                if self.mode=='C-SKIP':main=require_skip_visual_result(result)
                else:
                    if not isinstance(result,(tuple,list)) or len(result)!=2 or len(result[1])!=3 or any(x is None for x in result[1]):
                        raise ValueError('C-DROP requires three actually computed DS features')
                    main=result[0]
                self.state['visual']=main;self.state['deepstack']=result[1]
            def pruner_hook(_module,_args,result):self.state['pruned'],self.state['splits']=result
            def main_hook(*_args):self.state['main_merger_calls']+=1
            def block_hook(i):
                def inner(*_args):self.state['block_calls'][i]+=1
                return inner
            def ds_hook(i):
                def inner(_module,_args,result):
                    self.state['ds_calls'][i]+=1
                    self.state['ds_results'].append((i,result is None))
                return inner
            hook(v,visual_hook);hook(self.w.pruner,pruner_hook);hook(v.merger,main_hook)
            for i,block in enumerate(v.blocks):hook(block,block_hook(i))
            for i,merger in enumerate(v.deepstack_merger_list):hook(merger,ds_hook(i))
            handle=self.w.model.model.language_model.register_forward_pre_hook(self._prefill,with_kwargs=True)
            stack.callback(handle.remove)
            handle=self.w.model.model.language_model.register_forward_pre_hook(self._observe_prefill,with_kwargs=True)
            stack.callback(handle.remove)
            return self
        except BaseException:
            stack.close();self.stack=None
            raise

    def _prefill(self,_module,args,kw):
        s,t=self.state,self.torch
        s['llm_calls']+=1
        if s['llm_calls']!=1:return None
        if s['greedy_calls']!=1 or s['visual_calls']!=1 or s['splits']!=[256]:raise ValueError('missing actual visual/pruner result')
        order=s['ordered'];ids=order.detach().cpu().tolist()
        if len(ids)!=256 or len(set(ids))!=256 or any(x<0 or x>=1024 for x in ids) or s['quotas'].tolist()!=[256]:
            raise ValueError('invalid greedy selection')
        spatial=order.sort().values
        if not t.equal(s['visual'].index_select(0,spatial),s['pruned']):raise ValueError('selected features mismatch')
        x,start=s['inputs'],s['start'];length=x['input_ids'].shape[1]
        full,delta=self.w.model.model.get_rope_index(x['input_ids'],x['image_grid_thw'],None,attention_mask=x['attention_mask'])
        retained=t.cat((t.arange(start,device=spatial.device),start+spatial,t.arange(start+1024,length,device=spatial.device)))
        embeds=self.w.model.get_input_embeddings()(x['input_ids'])
        expected=t.cat((embeds[:,:start],s['visual'].index_select(0,spatial).unsqueeze(0),embeds[:,start+1024:]),dim=1)
        if not t.equal(expected,kw['inputs_embeds']) or not t.equal(x['attention_mask'].index_select(1,retained),kw['attention_mask']):
            raise ValueError('actual prefill features/mask mismatch')
        kw=dict(kw);kw['position_ids']=full.index_select(-1,retained)
        kw['visual_pos_masks']=None;kw['deepstack_visual_embeds']=None
        self.w.model.model.rope_deltas=delta+768
        s.update(spatial=spatial,retained=retained,full_pos=full,full_delta=delta,
                 prefill_position_ids=kw['position_ids'],prefill_embeds=kw['inputs_embeds'],
                 prefill_mask=kw['attention_mask'],adjusted_delta=self.w.model.model.rope_deltas)
        return args,kw

    def _observe_prefill(self,_module,_args,kw):
        s=self.state
        if 'actual_ds_off' not in s:
            if kw.get('visual_pos_masks') is not None or kw.get('deepstack_visual_embeds') is not None:
                raise ValueError('DeepStack unexpectedly injected')
            if not self.torch.equal(kw['position_ids'],s['prefill_position_ids']):raise ValueError('actual position mismatch')
            s['actual_ds_off']=True

    def verify_calls(self):
        s=self.state
        if s['visual_calls']!=1 or s['greedy_calls']!=1 or not s['llm_calls'] or s['main_merger_calls']!=1 or s.get('actual_ds_off') is not True:
            raise ValueError('missing main visual or selection call')
        if any(n!=1 for n in s['block_calls']) or s['ds_calls']!=[1,1,1]:raise ValueError('unexpected visual block/DS calls')
        if sorted(s['ds_results'])!=[(i,self.mode=='C-SKIP') for i in range(3)]:raise ValueError('unexpected DS merger return')
        if self.mode=='C-SKIP' and self.skip.skipped_calls!=[1,1,1]:raise ValueError('DS skip instrumentation incomplete')
        return True

    def __exit__(self,typ,value,tb):
        try:
            return self.stack.__exit__(typ,value,tb)
        finally:self.stack=None
