"""E4-C separate 32-image/20-repeat microbenchmark schedule; CPU-only."""
from contract import TASKS,rank,validate_schedule
COUNTS={'DocVQA_VAL':11,'TextVQA_VAL':11,'AI2D_TEST':10}
COMPONENTS=('selector','ds_merger_three_streams','ds_injection')

def make_microbench_schedule(timing):
 validate_schedule(timing);selected=[];jobs=[]
 for task in TASKS:
  groups=sorted(timing['cohort'][task]['measured'],key=lambda gid:(rank('microbench',task,gid),gid))[:COUNTS[task]]
  selected += [(task,g) for g in groups]
 for task,gid in selected:
  for repeat in range(20):
   # Deterministic rotation of component ordering; times remain separate.
   order=sorted(COMPONENTS,key=lambda c:rank('microbench_order',task,gid,repeat,c))
   for component in order:jobs.append({'task':task,'group_id':gid,'repeat':repeat,'component':component,'order':len(jobs)})
 if len(selected)!=32 or len(jobs)!=1920:raise ValueError('microbenchmark inventory')
 return {'scope':'separate batch1 DEV profiler only; not end-to-end TTFT', 'budget':256,'seed':'20261104',
         'images':selected,'repeat_per_image_component':20,'components':list(COMPONENTS),'jobs':jobs,
         'timing_policy':'GPU events/profiler and synchronizations only in this separate run',
         'input_policy':'same frozen real inputs and selected token identities; reuse feature tensors outside timed window',
         'limitations':['GPU-event intervals are not a device-wide upper bound','not a quality or end-to-end timing result',
                        'actual endpoint identity and independent launcher gates required']}
