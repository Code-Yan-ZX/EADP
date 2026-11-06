"""Future formal pipeline. Caller must validate frozen plans and evidence first."""
import json
from pathlib import Path

def verify(plan,jobs,design_jobs,references,run_dir,core,job_check,raw_check,docker_check,inspect,author,builder,worker_path,monitoring,window_classify):
 if plan.get('jobs')!=jobs or jobs!=design_jobs or len(jobs)!=24:raise ValueError('frozen exact job identity')
 if set(monitoring)!={'supervisor_health','manual_health','nvml','external_watcher'}:raise ValueError('monitoring category inventory')
 root=Path(run_dir);result=core(jobs,references,root,job_check);annotations=[];prefixes={};docker=[]
 if result.get('status')!='passed' or result.get('jobs')!=24:raise ValueError('core failed')
 for j in jobs:
  d=root/('job_%03d'%j['order']);launch=json.loads((d/'launch.json').read_text());ex=json.loads((d/'exit.json').read_text());record=json.loads((d/'record.json').read_text())['record'];ref=references[(j['task'],str(j['qid']),j['arm'])]
  prefixes[str(j['order'])]=raw_check(d/'raw.json',ref)
  docker.append(docker_check(j,launch,ex,inspect(ex['container_identity']['container_id']),author,builder,worker_path))
  begin=record['start_monotonic_ns'];end=record['first_token_monotonic_ns'];ns=record.get('boundaries',{}).get('time_namespace')
  annotations.append({'order':j['order'],'background':{category:window_classify(begin,end,ns,queries) for category,queries in monitoring.items()}})
 return {**result,'prompt_sha256':prefixes,'docker_inspections':docker,'background_annotations':annotations,'all_retained':True,'environment_limitation':'recorded windows provide only direct overlap; unobserved activity is unknown'}
if __name__=='__main__':raise SystemExit('draft has no formal admission')
