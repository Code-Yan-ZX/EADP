"""Memory-only instrumentation, never attach to interleaved latency worker.

KV is logical live key/value tensor bytes, not allocator overhead or a physical
allocation attribution. Process-level NVML sampling is a separate measurement.
"""
class KVProbe:
 def __init__(self,model):self.model=model;self.handle=None;self.calls=0;self.latest_bytes=None;self.max_bytes=0
 def __enter__(self):
  def observe(_module,_args,output):
   cache=getattr(output,'past_key_values',None)
   if cache is None:raise ValueError('model output did not return KV cache')
   total=0;layers=0
   for pair in cache:
    if len(pair)!=2:raise ValueError('unexpected KV layer')
    for tensor in pair:
     if tensor is None:raise ValueError('missing KV tensor')
     total+=tensor.numel()*tensor.element_size()
    layers+=1
   if layers<1:raise ValueError('empty KV cache')
   self.calls+=1;self.latest_bytes=total;self.max_bytes=max(self.max_bytes,total)
  self.handle=self.model.register_forward_hook(observe)
  return self
 def __exit__(self,typ,value,tb):
  self.handle.remove();self.handle=None

def memory_measure(torch,model,run):
 """Fresh per-arm allocator peaks; no NVML or exclusive-GPU claims here."""
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 with KVProbe(model) as probe:
  result=run()
  torch.cuda.synchronize()
  if probe.calls<1:raise ValueError('no KV observed')
  stats={'allocated_peak_bytes':int(torch.cuda.max_memory_allocated()),
         'reserved_peak_bytes':int(torch.cuda.max_memory_reserved()),
         'kv_cache_bytes_final':probe.latest_bytes,'kv_cache_bytes_peak_logical':probe.max_bytes,
         'kv_observed_forwards':probe.calls}
 return result,stats
