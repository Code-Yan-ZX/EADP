"""CPU verification of completed latency job records; no GT or scoring."""
import math
from interference import annotate_events

def validate_events(schedule,events,worker_pid,decode):
 jobs=schedule['schedule']
 if type(worker_pid) is not int or worker_pid<=0 or len(events)!=2*len(jobs):raise ValueError('latency journal inventory')
 metrics=[]
 for job,generated,completed in zip(jobs,events[::2],events[1::2]):
  if (generated.get('status')!='generated' or completed.get('status')!='completed' or generated.get('job')!=job or completed.get('job')!=job or
      generated.get('worker_pid')!=worker_pid or completed.get('worker_pid')!=worker_pid or generated.get('measurement_role')!='latency_interleaved' or completed.get('measurement_role')!='latency_interleaved'):
   raise ValueError('latency event identity/order mismatch')
  r=generated['result'];ids=r.get('output_ids');eos=r.get('eos_ids');fixed=job['protocol']=='fixed32';maximum=32 if fixed else 2048
  recipe={'do_sample':False,'max_new_tokens':maximum,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0}
  if (r.get('mode')!=job['mode'] or r.get('protocol')!=job['protocol'] or r.get('generation_scalars')!=recipe or r['generation_scalars'].get('do_sample') is not False or
      r.get('min_new_tokens')!=(32 if fixed else None) or not isinstance(ids,list) or not ids or any(type(x) is not int for x in ids) or len(ids)>maximum or
      r.get('output_tokens')!=len(ids) or not isinstance(eos,list) or any(type(x) is not int for x in eos) or (fixed and len(ids)!=32)):
   raise ValueError('actual generation recipe/output invalid')
  ended=ids[-1] in eos;truncated=not fixed and not ended and len(ids)==2048
  if r.get('eos') is not ended or r.get('truncated') is not truncated or (not fixed and not ended and not truncated) or r.get('answer')!=decode(ids):raise ValueError('terminal token or decoding mismatch')
  b=r['boundaries'];times=[b.get(k) for k in ('t0_before_image_read_ns','first_token_selected_ns','generation_complete_ns')]
  if (any(type(x) is not int or x<0 for x in times) or times!=sorted(times) or b.get('first_criteria_length')!=1 or b.get('criteria_calls')!=len(ids) or
      b.get('explicit_sync_boundaries')!=['before_t0','first_token','generation_end'] or b.get('logits_ready_ns') is not None or r.get('instrumented_logits_ready') is not False):
   raise ValueError('first-token timing boundary mismatch')
  ttft=(times[1]-times[0])/1e6;total=(times[2]-times[0])/1e6
  if not math.isfinite(ttft) or not math.isfinite(total):raise ValueError('nonfinite duration')
  metrics.append({'job':job,'ttft_ms':ttft,'end_to_end_ms':total,'output_tokens':len(ids),'eos':ended,'truncated':truncated})
 return metrics

def annotate_validated(schedule,events,worker_pid,decode,snapshots,namespace,started,terminal):
 metrics=validate_events(schedule,events,worker_pid,decode)
 annotated=annotate_events(events,snapshots,namespace,started,terminal)
 if annotated['n_generated']!=len(metrics):raise ValueError('monitor annotation inventory mismatch')
 return {'metrics':metrics,**annotated}
