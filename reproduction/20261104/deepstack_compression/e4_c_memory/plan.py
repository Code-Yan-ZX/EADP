"""Derive memory-only process jobs from the fixed latency schedule.

Caller must independently reconstruct the source schedule from accepted DEV
inputs before calling. This CPU helper grants no authority to load a model.
"""
from collections import Counter
MODES=('C-ON','C-DROP','C-SKIP')

def split(schedule,stage):
 expected={'smoke':72,'full':5940}.get(stage);jobs=schedule.get('schedule')
 if expected is None or not isinstance(jobs,list) or len(jobs)!=expected:raise ValueError('invalid source schedule inventory')
 if any(j.get('order')!=i or j.get('mode') not in MODES for i,j in enumerate(jobs)):raise ValueError('invalid source schedule order')
 if Counter(j['mode'] for j in jobs)!={m:expected//3 for m in MODES}:raise ValueError('unequal configuration inventory')
 out={}
 for mode in MODES:
  selected=[dict(job) for job in jobs if job['mode']==mode]
  # Keep the original interleaving index for audit; local order is explicit.
  for index,job in enumerate(selected):job['latency_order']=job['order'];job['order']=index
  if len(selected)!=expected//3 or set(j['protocol'] for j in selected)!={'natural','fixed32'} or set(j['phase'] for j in selected)!={'warmup','measure'}:raise ValueError('memory process coverage')
  out[mode]={'stage':stage,'mode':mode,'role':'memory_isolated','jobs':selected,'scope':'fresh process per configuration; no interleaved latency claim'}
 return out
