"""Memory-only measurement bridge; no GPU entry point or admission.

The caller must first independently establish a released one-process-per-mode
container, sampled GPU exclusivity, source identity and supervision. These
functions do not provide that authority. No TTFT result is inferred here.
"""

def make_measure(torch,wrapper,native,execute_one,memory_measure,mode):
 if mode not in ('C-ON','C-DROP','C-SKIP'):raise ValueError('invalid isolated mode')
 def measure(row,job,partial):
  if job.get('mode')!=mode or job.get('protocol') not in ('natural','fixed32') or type(row.get('image_start')) is not int:raise ValueError('memory job identity')
  def run():
   captured=[]
   original=wrapper.model.generate
   def capture(*args,**kwargs):
    result=original(*args,**kwargs)
    if captured:raise ValueError('unexpected multiple generate calls')
    tokens=result.detach().cpu().tolist()
    if len(tokens)!=1 or not isinstance(tokens[0],list) or not tokens[0] or any(type(t) is not int for t in tokens[0]):raise ValueError('generated token shape')
    # Persist tokens before returning to the wrapper or tearing down hooks.
    captured.append(tokens[0])
    partial({'phase':'generate_return','mode':mode,'protocol':job['protocol'],'output_ids':tokens[0]})
    return result
   wrapper.model.generate=capture
   try:
    with torch.inference_mode():value=execute_one(wrapper,native,torch,row,mode,job['protocol'],torch.cuda.synchronize)
    if len(captured)!=1 or value.get('output_ids')!=captured[0]:raise ValueError('generate result differs from persisted tokens')
    return value
   finally:wrapper.model.generate=original
  value,stats=memory_measure(torch,wrapper.model,run)
  keys=('allocated_peak_bytes','reserved_peak_bytes','kv_cache_bytes_final','kv_cache_bytes_peak_logical','kv_observed_forwards')
  if value.get('mode')!=mode or value.get('protocol')!=job['protocol'] or any(type(stats.get(k)) is not int or stats[k]<0 for k in keys):raise ValueError('memory measurement metadata')
  if stats['reserved_peak_bytes']<stats['allocated_peak_bytes'] or stats['kv_cache_bytes_peak_logical']<stats['kv_cache_bytes_final'] or stats['kv_observed_forwards']<1:raise ValueError('memory measurements inconsistent')
  return {**value,'memory_only':{k:stats[k] for k in keys},'scope':'memory only; no latency claim'}
 return measure

def main():raise SystemExit('locked: sampler/controller/independent GPU admission not implemented')
if __name__=='__main__':main()
