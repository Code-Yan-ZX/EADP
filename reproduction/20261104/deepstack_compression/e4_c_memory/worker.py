"""CPU-testable orchestration for one future isolated memory process.

No GPU-capable CLI is provided. The eventual launcher must bind a reviewed
model/adapter identity, a fresh container per mode, complete sampler handshake,
source/input identities, exclusive device, budget, and verified exit chain.
"""
import os
from plan import split

def run_memory_jobs(source,expected,rows,stage,mode,measure,journal):
 if source!=expected:raise ValueError('source schedule differs from independently reconstructed schedule')
 plans=split(source,stage)
 if mode not in plans:raise ValueError('unknown memory mode')
 jobs=plans[mode]['jobs'];bykey={(r['task'],str(r['qid'])):r for r in rows}
 if len(bykey)!=len(rows) or any(r.get('split')!='DEV' for r in rows):raise ValueError('DEV inventory')
 for job in jobs:
  row=bykey.get((job['task'],str(job['qid'])))
  if row is None or row.get('group_id')!=job['group_id'] or job['mode']!=mode:raise ValueError('job/input identity mismatch')
 for job in jobs:
  row=bykey[(job['task'],str(job['qid']))]
  def partial(value):journal({'status':'generated_partial','job':job,'worker_pid':os.getpid(),'result':value,'scope':'memory-only'})
  try:
   value=measure(row,job,partial)
   journal({'status':'generated','job':job,'worker_pid':os.getpid(),'result':value,'scope':'memory-only'})
   memory=value.get('memory_only')
   keys=('allocated_peak_bytes','reserved_peak_bytes','kv_cache_bytes_final','kv_cache_bytes_peak_logical','kv_observed_forwards')
   if value.get('mode')!=mode or value.get('protocol')!=job['protocol'] or not isinstance(memory,dict) or any(type(memory.get(k)) is not int or memory[k]<0 for k in keys) or memory['reserved_peak_bytes']<memory['allocated_peak_bytes'] or memory['kv_cache_bytes_peak_logical']<memory['kv_cache_bytes_final'] or memory['kv_observed_forwards']<1:raise ValueError('memory result identity or accounting mismatch')
  except BaseException as exc:
   journal({'status':'failed','job':job,'worker_pid':os.getpid(),'error_type':type(exc).__name__,'error':str(exc),'scope':'memory-only'});raise
  journal({'status':'completed','job':job,'worker_pid':os.getpid(),'scope':'memory-only'})
 return {'status':'completed','jobs':len(jobs),'mode':mode,'scope':'memory-only; no TTFT inference'}

def main():raise SystemExit('memory worker locked until independently reviewed GPU/container/sampler admission')
if __name__=='__main__':main()
