"""Capture actual prefill/decode positions and every layer's KV lengths."""

class DecodeObserver:
    def __init__(self, wrapper):
        self.w=wrapper;self.trace=[];self.handles=[]

    def __enter__(self):
        lm=self.w.model.model.language_model
        def before(_module,_args,kw):
            def as_list(x):return None if x is None else x.detach().cpu().tolist()
            cache=kw.get('past_key_values');n=len(lm.layers)
            self.trace.append({'position_ids':as_list(kw.get('position_ids')),
                'cache_position':as_list(kw.get('cache_position')),
                'rope_delta':as_list(self.w.model.model.rope_deltas),
                'layer_cache_lengths':[] if cache is None else [cache.get_seq_length(i) for i in range(n)]})
        def after(_module,_args,_kw,out):
            cache=out.past_key_values;n=len(lm.layers)
            self.trace[-1]['layer_cache_lengths_after']=[] if cache is None else [cache.get_seq_length(i) for i in range(n)]
        self.handles.append(lm.register_forward_pre_hook(before,with_kwargs=True))
        self.handles.append(lm.register_forward_hook(after,with_kwargs=True))
        return self

    def __exit__(self,*_args):
        for h in reversed(self.handles):h.remove()


def validate_trace(trace,output_ids,compact_length,delta,layers=36):
    if not output_ids or len(trace)!=len(output_ids):raise ValueError('actual generation/decode count mismatch')
    first=trace[0]
    if (first['cache_position']!=list(range(compact_length)) or first['rope_delta']!=[[delta]] or
        first['layer_cache_lengths'] not in ([],[0]*layers) or first['layer_cache_lengths_after']!=[compact_length]*layers):
        raise ValueError('actual prefill KV/cache mismatch')
    for step,obs in enumerate(trace[1:],start=1):
        pos=compact_length+step-1
        if (obs['cache_position']!=[pos] or obs['position_ids']!=[[[pos+delta]]]*3 or obs['rope_delta']!=[[delta]] or
            obs['layer_cache_lengths']!=[pos]*layers or obs['layer_cache_lengths_after']!=[pos+1]*layers):
            raise ValueError('actual decode position or KV mismatch')
    return True
