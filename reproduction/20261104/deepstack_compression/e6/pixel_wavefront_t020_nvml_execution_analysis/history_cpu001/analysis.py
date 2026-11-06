"""Descriptive nine-process NVML analysis; never authorize GPU or read GT.

Only independently accepted complete results may be read. No timing claim.
"""
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];E=B/'e6/pixel_wavefront_t020_nvml_execution_candidate_v8';A=B/'e6/pixel_wavefront_t020_nvml_execution_accept';I=B/'e1_independent_review';H=B/'e6/pixel_wavefront_t020_nvml_host_candidate'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if len(c)<2 or (cwd/c[1]).resolve()!=script.resolve() or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or t.get('worker_pid')!=s['worker_pid'] or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or (out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervision evidence')
def admit():
 p=read(D/'analysis_plan001.json');formal=A/'verification001.json';proof=I/'pixel_t020_nvml_execution_result_review.json';reviewer=I/'review_pixel_t020_nvml_execution_result.py';monitor=I/'monitor_pixel_t020_nvml_execution_result';analysis_review=I/'pixel_t020_nvml_execution_analysis_prelaunch_review.json';ar=I/'review_pixel_t020_nvml_execution_analysis_prelaunch.py';am=I/'monitor_pixel_t020_nvml_execution_analysis_prelaunch'
 if p.get('scope')!='nine_process_pixel_nvml_descriptive' or p.get('GPU_used') is not False or p.get('GT_opened') is not False or any(sha(k)!=v for k,v in p['source_sha256'].items()):raise ValueError('analysis plan/source')
 terminal(am,ar,analysis_review);q=read(analysis_review)
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('GT_opened') is not False or q.get('timing_results_opened') is not False or q.get('plan_sha256')!=sha(D/'analysis_plan001.json') or q.get('reviewer_sha256')!=sha(ar) or q.get('errors')!=[]:raise ValueError('analysis prelaunch review')
 terminal(A/'monitor_verify001',A/'verify.py',formal);terminal(monitor,reviewer,proof);v=read(formal);r=read(proof)
 fields={'formal_sha256':formal,'formal_started_sha256':A/'monitor_verify001/started.json','formal_terminal_sha256':A/'monitor_verify001/terminal.json','run_started_sha256':E/'monitor_run001/started.json','run_terminal_sha256':E/'monitor_run001/terminal.json','reviewer_sha256':reviewer}
 if v.get('status')!='passed' or v.get('jobs')!=9 or v.get('GT_opened') is not False or v.get('scoring_performed') is not False or v.get('errors')!=[] or r.get('status')!='passed' or r.get('independent_review') is not True or r.get('jobs')!=9 or r.get('GT_opened') is not False or r.get('scoring_performed') is not False or r.get('errors')!=[] or any(r.get(k)!=sha(x) for k,x in fields.items()) or p.get('gpu_plan_sha256')!=sha(E/'gpu_plan001.json') or p.get('formal_plan_sha256')!=sha(A/'verification_plan001.json'):raise ValueError('accepted run unavailable')
 return p,v,r

def rows(jobs, formal, get):
 if len(jobs)!=9 or len(formal.get('results',[]))!=9:raise ValueError('nine jobs required')
 out=[]
 for i,job in enumerate(jobs):
  record=formal['results'][i]
  if record.get('job')!=job:raise ValueError('formal job mismatch')
  summary=get(i)
  if summary.get('job')!=job or summary.get('status')!='passed' or summary.get('stage_summary')!=record.get('stage_summary') or any(summary.get(k)!=record.get(k) for k in ('allocator_model_load','allocator_request')):raise ValueError('raw/formal summary mismatch')
  info=record['identity'];out.append({'job':job,'selected_count':info['selected_count'],'prompt_length':info['prompt_length'],'generated_length':info['generated_length'],'sample_count':record['sample_count'],'stage_summary':record['stage_summary'],'allocator_model_load':record['allocator_model_load'],'allocator_request':record['allocator_request'],'container_id':record['container_id'],'host_pid':record['host_pid'],'gpu_uuid':record['gpu_uuid']})
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=(D/'analysis001.json').resolve():raise ValueError('unplanned output')
 plan,formal,review=admit();jobs=read(E/'gpu_plan001.json')['jobs'];run=E/'run_001';result=rows(jobs,formal,lambda i:read(run/('job_%03d'%i)/'memory_summary.json'))
 value={'status':'passed','jobs':9,'results':result,'GT_opened':False,'scoring_performed':False,'timing_claims':False,'analysis_plan_sha256':sha(D/'analysis_plan001.json'),'formal_sha256':sha(A/'verification001.json'),'independent_result_sha256':sha(I/'pixel_t020_nvml_execution_result_review.json'),'limits':['separate process peaks are not per-request allocation differences','sampled MiB maxima may miss transient peaks; overlap queries may cross stage boundaries','device, process and PyTorch allocator quantities are distinct; unknown and absent are not zero','graphics/background and unobserved activity remain possible','three post-hoc DEV anchors, different K, no equal-compute or population inference','CPU copies and synchronization perturb memory; no TTFT claim','endpoint weight stats cannot rule out transient replacement']}
 with a.out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
