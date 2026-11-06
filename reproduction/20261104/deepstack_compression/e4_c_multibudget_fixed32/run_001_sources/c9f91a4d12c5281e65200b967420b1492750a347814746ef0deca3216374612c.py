"""CPU-only E4-C timing schedule and measurement contracts. No timers/GPU work."""
import hashlib
from collections import Counter,defaultdict
TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')
MODES=('C-ON','C-DROP','C-SKIP')
PROTOCOLS=('natural','fixed32')
SEED='20261104'

def rank(*xs):return hashlib.sha256(('|'.join(map(str,(SEED,*xs)))).encode()).hexdigest()
def make_schedule(rows,excluded=()):
 if len(rows)!=384 or any(r.get('split')!='DEV' for r in rows) or Counter(r['task'] for r in rows)!={t:128 for t in TASKS}:raise ValueError('must use exact DEV input inventory')
 if len({(r['task'],str(r['qid'])) for r in rows})!=384:raise ValueError('duplicate question')
 excluded=set(excluded);cohort={};out=[]
 for task in TASKS:
  options=[r for r in rows if r['task']==task and r['group_id'] not in excluded]
  if len({r['group_id'] for r in options})!=len(options):raise ValueError('one DEV question per image required')
  options.sort(key=lambda r:(rank('cohort',task,r['group_id']),r['group_id']))
  if len(options)<74:raise ValueError('insufficient distinct DEV groups')
  measure=options[:64];warm=options[64:74];cohort[task]={'measured':[r['group_id'] for r in measure],'warmup':[r['group_id'] for r in warm]}
  for protocol in PROTOCOLS:
   # Warmups are never included in the measured denominator.
   for row in warm:
    for mode in sorted(MODES,key=lambda m:rank('warm',task,protocol,row['group_id'],m)):
     out.append({'task':task,'qid':str(row['qid']),'group_id':row['group_id'],'phase':'warmup','protocol':protocol,'replicate':0,'mode':mode})
   for row in measure:
    for rep in range(5):
     for mode in sorted(MODES,key=lambda m:rank('measure',task,protocol,row['group_id'],rep,m)):
      out.append({'task':task,'qid':str(row['qid']),'group_id':row['group_id'],'phase':'measure','protocol':protocol,'replicate':rep,'mode':mode})
 for i,r in enumerate(out):r['order']=i
 return {'seed':SEED,'cohort':cohort,'schedule':out,'scope':'DEV batch1 diagnostic, no labels'}

def validate_schedule(obj):
 rows=obj['schedule'];cohort=obj['cohort']
 if obj.get('seed')!=SEED or [r.get('order') for r in rows]!=list(range(len(rows))):raise ValueError('seed/order invalid')
 if len(rows)!=5940:raise ValueError('wrong total schedule')
 for task in TASKS:
  measured=cohort[task]['measured'];warm=cohort[task]['warmup']
  if len(set(measured))!=64 or len(set(warm))!=10 or set(measured)&set(warm):raise ValueError('cohort inventory')
  for protocol in PROTOCOLS:
   selected=[r for r in rows if r['task']==task and r['protocol']==protocol]
   if sum(r['phase']=='warmup' for r in selected)!=30 or sum(r['phase']=='measure' for r in selected)!=960:raise ValueError('warmup/measurement inventory')
   for row in selected:
    if row['mode'] not in MODES:raise ValueError('unknown config')
    if row['phase']=='warmup' and (row['group_id'] not in warm or row['replicate']!=0):raise ValueError('bad warmup')
    if row['phase']=='measure' and (row['group_id'] not in measured or row['replicate'] not in range(5)):raise ValueError('bad measured item')
   for gid in measured:
    for rep in range(5):
     block=[r for r in selected if r['group_id']==gid and r['phase']=='measure' and r['replicate']==rep]
     if len(block)!=3 or set(x['mode'] for x in block)!=set(MODES) or max(x['order'] for x in block)-min(x['order'] for x in block)!=2:raise ValueError('configuration block broken')
   for gid in warm:
    block=[r for r in selected if r['group_id']==gid and r['phase']=='warmup']
    if len(block)!=3 or set(x['mode'] for x in block)!=set(MODES):raise ValueError('warmup block broken')
 return True

def validate_measurement(r):
 """Validate claims only; instrumentation/worker identity require separate proof."""
 if r.get('mode') not in MODES or r.get('protocol') not in PROTOCOLS or r.get('batch_size')!=1 or r.get('phase') not in ('warmup','measure'):raise ValueError('bad measurement identity')
 if r.get('profiler_enabled') is not False or r.get('tensor_hashing_in_window') is not False or r.get('per_layer_sync_in_window') is not False:raise ValueError('instrumented main timing')
 if r.get('measurement_role') not in ('latency_interleaved','memory_isolated') or r.get('exclusive_gpu') is not True:raise ValueError('measurement role/concurrency condition unproven')
 if r['measurement_role']=='memory_isolated' and r.get('fresh_process_per_config') is not True:raise ValueError('memory isolation not established')
 ticks=[r.get(k) for k in ('t0_before_image_read_ns','first_token_selected_ns','generation_complete_ns')]
 if any(type(x) is not int or x<0 for x in ticks) or ticks!=sorted(ticks):raise ValueError('invalid timing boundaries')
 logits=r.get('logits_ready_ns')
 if logits is not None:raise ValueError('logits profiler is a separate run')
 if r.get('token_selection_materialized') is not True or r.get('boundary_sync_complete') is not True:raise ValueError('first token/sync not observed')
 n=r.get('output_tokens')
 if type(n) is not int or n<1 or n>2048 or (r['protocol']=='fixed32' and n!=32):raise ValueError('output length mismatch')
 if r['protocol']=='fixed32' and r.get('scored') is not False:raise ValueError('forced output cannot be scored')
 if r['measurement_role']=='memory_isolated':
  memory=r.get('memory_only')
  if not isinstance(memory,dict):raise ValueError('missing memory-only observation')
  for key in ('allocated_peak_bytes','reserved_peak_bytes','kv_cache_bytes'):
   if type(memory.get(key)) is not int or memory[key]<0:raise ValueError('invalid memory accounting')
  if memory['reserved_peak_bytes']<memory['allocated_peak_bytes']:raise ValueError('reserved smaller than allocated')
 return {'ttft_ms':(ticks[1]-ticks[0])/1e6,'first_logits_ms':None if logits is None else (logits-ticks[0])/1e6,'end_to_end_ms':(ticks[2]-ticks[0])/1e6}


def validate_process_memory_summary(summary,mode,pid,start_ticks,gpu_uuid):
 """Whole-process sampled maximum, never a per-sample peak or exact maximum."""
 if (mode not in MODES or summary.get('type')!='summary' or summary.get('pid')!=pid or
     summary.get('start_ticks')!=start_ticks or summary.get('gpu_uuid')!=gpu_uuid or
     type(pid) is not int or pid<=0 or type(start_ticks) is not int or start_ticks<=0 or
     not isinstance(gpu_uuid,str) or not gpu_uuid.startswith('GPU-') or
     type(summary.get('samples')) is not int or summary['samples']<1 or summary.get('target_observed') is not True or
     summary.get('other_gpu_process_observed') is not False or
     type(summary.get('nvml_process_sampled_peak_bytes')) is not int or summary['nvml_process_sampled_peak_bytes']<0):
  raise ValueError('invalid isolated process NVML summary')
 return {'mode':mode,'scope':'whole isolated process; MiB-rounded sampled maximum',
         'nvml_process_sampled_peak_bytes':summary['nvml_process_sampled_peak_bytes']}
