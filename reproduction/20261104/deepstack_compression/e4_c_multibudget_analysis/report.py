"""Run fixed, paired pilot analysis only after both complete acceptances."""
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent
EXPECTED_GATE_SHA='b0f524eef3bae79351feb36a78c7d47cac310b9131b6327f476ebc40bd04c7de'
def load(name,path):
 if name in sys.modules:raise ValueError('preloaded module '+name)
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
 if a.out.resolve()!=D/'analysis004.json' or a.out.exists():raise ValueError('unexpected output')
 gate_path=D/'analysis_gate.py'
 if hashlib.sha256(gate_path.read_bytes()).hexdigest()!=EXPECTED_GATE_SHA:raise ValueError('analysis gate changed')
 gate=load('_multibudget_analysis_gate',gate_path);admission=gate.admit()
 # No prediction-bearing journal is parsed before the complete admissions above.
 plan=gate.read(B/'e4_c_multibudget_expansion/pilot_plan001.json');formal=gate.read(B/'e4_c_multibudget_worker_accept_v3/verification001.json');journal=B/'e4_c_multibudget_worker_v3/run_001/journal.jsonl'
 with journal.open() as f:events=[json.loads(x) for x in f if x.strip()]
 metrics=load('_multibudget_analysis_metrics',D/'metrics.py')
 if metrics.np.__version__!='2.2.6':raise ValueError('NumPy version differs from frozen bootstrap')
 result=metrics.summarize(plan['jobs'],events);annotations=formal['annotations'];model_windows=formal['health_windows']
 observer=load('_multibudget_analysis_watcher',D/'watcher_contract.py');w=B/'e4_c_multibudget_environment';run=B/'e4_c_multibudget_worker_v3';mon=run/'monitor_run001'
 def jsonl(path):
  with path.open() as f:return [json.loads(line) for line in f if line.strip()]
 launch=gate.read(run/'run_001/launch.json');started=gate.read(mon/'started.json');ended=gate.read(mon/'terminal.json')
 observed=observer.validate(jsonl(w/'samples001.jsonl'),jsonl(w/'monitor_watch001/health.jsonl'),jsonl(w/'manual_checks001.jsonl'),launch['time_namespace'],started['created_utc'],ended['utc'])
 if len(annotations)!=1520:raise ValueError('missing health annotations')
 windows={'model_health':model_windows,**{k:observed[k] for k in ('query','query_first_fsync','watcher_health','manual_health')}}
 details=[];health={}
 for i,(job,ann) in enumerate(zip(plan['jobs'],annotations)):
  if ann.get('order')!=i or any(ann.get(k)!=job.get(k) for k in ('phase','task','budget','protocol')):raise ValueError('health annotation/job mismatch')
  record=events[i*3+2]['record'];b=record['boundaries'];t0=b['t0_before_image_read_ns'];t1=b['first_token_selected_ns'];t2=b['generation_complete_ns']
  overlap={k:{'ttft':observer.overlap(v,t0,t1),'generation':observer.overlap(v,t0,t2)} for k,v in windows.items()}
  if overlap['model_health']['ttft']!=ann['ttft_health_overlap_indices'] or overlap['model_health']['generation']!=ann['full_job_health_overlap_indices']:raise ValueError('model health overlap mismatch')
  entry={k:job[k] for k in ('order','phase','task','qid','group_id','budget','protocol','replicate')};entry.update(t0_before_image_read_ns=t0,first_token_selected_ns=t1,generation_complete_ns=t2,token_count=len(record['output_ids']),eos=record['eos'],truncated=record['truncated'],overlap_indices=overlap);details.append(entry)
  if job['phase']=='measure':
   key='/'.join((job['task'],job['protocol'],str(job['budget'])));h=health.setdefault(key,{'measured_jobs':0,'overlap_counts':{k:{'ttft':0,'generation':0} for k in windows}});h['measured_jobs']+=1
   for k in windows:
    for boundary in ('ttft','generation'):h['overlap_counts'][k][boundary]+=bool(overlap[k][boundary])
 first=min(x['t0_before_image_read_ns'] for x in details);last=max(x['generation_complete_ns'] for x in details)
 hourly=observer.hourly_coverage(first,last,windows['manual_health'],windows['model_health']+windows['watcher_health'])
 result.update(admission=admission,analysis_plan_sha256=gate.sha(B/'e4_c_multibudget_expansion/ANALYSIS_PLAN.md'),pilot_plan_sha256=gate.sha(B/'e4_c_multibudget_expansion/pilot_plan001.json'),predictions_opened=True,prediction_scope='accepted completed timing journal parsed for timings and output terminal counts',jobs_detail=details,health_overlap=health,hourly_health_coverage=hourly,manual_check_evidence=('recorded' if windows['manual_health'] else 'unknown: empty manual log'),observer_windows=windows,health_limit='No samples excluded. Direct windows only; watcher second fsync, manual JSON fsync, stop marker, unrecorded and residual activity unknown; UTC coverage is vulnerable to wall-clock adjustments. ps CPU is lifetime average.',author_generation_completion_limit='synchronized generate return before raw token copy/fsync and author text decode; not full end-to-end latency')
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('32-image exploratory timing analysis completed')
if __name__=='__main__':main()
