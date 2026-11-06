"""No-GT verification and health overlap annotation for fixed 48-job pilot."""
import argparse,importlib.util,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;RUN=BASE/'e4_c_budget_ttft_pilot_v3';V2=BASE/'e4_c_budget_pilot_v2'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
admission=load('_budget_ttft_accept_admission',RUN/'admission.py');identity=admission.identity
def overlap(a,b):return a[0]<=b[1] and b[0]<=a[1]
def job_events(events,job,accepted,tokenizer):
 if len(events)!=3 or [x.get('event') for x in events]!=['started','raw_generation','completed'] or any(x.get('job')!=job for x in events):raise ValueError('missing/unordered timing events')
 raw=events[1]['payload'];result=events[2]['payload'];k=job['budget']
 for key in ('task','qid','group_id','budget','output_ids','generation_scalars','eos_ids'):
  if raw.get(key)!=result.get(key):raise ValueError('raw/result mismatch '+key)
 if (result.get('output_ids')!=accepted['output_ids'] or result.get('answer')!=accepted['answer'] or result.get('generation_scalars')!=accepted['generation_scalars'] or
     result.get('eos_ids')!=accepted['eos_ids'] or result.get('eos')!=accepted['eos'] or result.get('truncated')!=accepted['truncated'] or
     result.get('natural') is not True or result.get('selector_calls')!=(0 if k==1024 else 1)):raise ValueError('same-arm timing result mismatch')
 if tokenizer.decode(result['output_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)!=result['answer']:raise ValueError('independent answer decode mismatch')
 b=result['boundaries'];ticks=[b.get(key) for key in ('t0_before_image_read_ns','first_token_selected_ns','generation_complete_ns')]
 if (any(type(x) is not int or x<0 for x in ticks) or ticks!=sorted(ticks) or b.get('first_criteria_length')!=1 or b.get('criteria_calls')!=len(result['output_ids']) or
     b.get('logits_ready_ns') is not None or b.get('clock_implementation')!='clock_gettime(CLOCK_MONOTONIC)' or
     b.get('explicit_sync_boundaries')!=['before_t0','first_token','generation_end']):raise ValueError('first-token boundary invalid')
 return ticks,b['time_namespace']
def verify():
 out=RUN/'run_001';mon=RUN/'monitor_run001';identity.terminal(mon/'started.json',mon/'terminal.json');approval=admission.admit_gpu();plan=admission.plan()
 launch=json.loads((out/'launch.json').read_text());ex=json.loads((out/'exit.json').read_text());st=ex.get('last_inspect') or {};cid=(ex.get('container_identity') or {}).get('container_id')
 if (launch.get('approval_bundle')!=approval or launch.get('worker_sha256')!=identity.sha(RUN/'worker.py') or launch.get('launcher_sha256')!=identity.sha(RUN/'launch.py') or
     launch.get('run_out')!=str(out) or ex.get('controller')!=launch.get('controller') or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or
     ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('container_stopped_verified') is not True or
     ex.get('client_exited_while_container_running') is not False or ex.get('gpu_pids_after')!=[] or st.get('running') is not False or st.get('pid')!=0 or not cid or st.get('id')!=cid):raise ValueError('timing launcher/exit invalid')
 doc=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=10,check=True).stdout)[0]
 if doc['State']['Running'] is not False or doc['State']['Pid']!=0 or doc['State']['ExitCode']!=0 or doc['Image']!=st['image']:raise ValueError('timing Docker state mismatch')
 snapshot=RUN/'run_001_sources';snap=snapshot/'manifest.json';identity.terminal(HERE/'monitor_snapshot001/started.json',HERE/'monitor_snapshot001/terminal.json');sm=json.loads(snap.read_text())
 if sm.get('status')!='passed' or sm.get('runtime_plan_sha256')!=identity.sha(RUN/'runtime_plan001.json') or sm.get('snapshot_script_sha256')!=identity.sha(HERE/'snapshot.py') or set(sm['sources'])!=set(plan['source_sha256']):raise ValueError('timing source snapshot mismatch')
 for path,digest in plan['source_sha256'].items():
  item=sm['sources'][path];copy=Path(item['snapshot']).resolve()
  if item.get('source')!=path or item.get('sha256')!=digest or not copy.is_relative_to(snapshot.resolve()) or identity.sha(copy)!=digest:raise ValueError('timing source snapshot corrupt')
 complete=json.loads((out/'worker_complete.json').read_text());frozen=json.loads((out/'frozen.json').read_text());journal=out/'journal.jsonl'
 if (complete.get('status')!='passed' or complete.get('jobs')!=48 or complete.get('warmups')!=12 or complete.get('measured')!=36 or
     complete.get('source_unchanged') is not True or complete.get('weights_stat_unchanged') is not True or complete.get('journal_sha256')!=identity.sha(journal) or
     frozen.get('approval')!=approval or frozen.get('source_sha256')!=plan['source_sha256'] or frozen.get('GT_opened') is not False):raise ValueError('timing completion mismatch')
 reference=load('_budget_ttft_accept_reference',V2/'reference.py');refs=reference.load_references();schedule=load('_budget_ttft_accept_contract',RUN/'contract.py');jobs=json.loads((RUN/'pilot_plan001.json').read_text())['jobs'];schedule.validate(jobs,refs['rows'])
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',local_files_only=True,trust_remote_code=False)
 events=[json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
 if len(events)!=48*3:raise ValueError('timing journal denominator')
 health=mon/'health.jsonl';snapshots=[json.loads(line) for line in health.read_text().splitlines() if line.strip()]
 if not snapshots:raise ValueError('missing health windows')
 windows=[]
 for row in snapshots:
  window=row.get('health_snapshot_window_monotonic_ns');namespace=row.get('time_namespace')
  if not isinstance(window,list) or len(window)!=2 or any(type(x) is not int for x in window) or window[0]>window[1] or namespace!=launch['time_namespace']:raise ValueError('health window missing/clock mismatch')
  windows.append(window)
 annotations=[]
 for i,job in enumerate(jobs):
  accepted=refs['records'][(job['task'],job['qid'],job['budget'])];ticks,ns=job_events(events[3*i:3*i+3],job,accepted,tokenizer)
  if ns!=launch['time_namespace']:raise ValueError('sample time namespace mismatch')
  annotations.append({'order':i,'phase':job['phase'],'task':job['task'],'budget':job['budget'],
                      'ttft_health_overlap_indices':[n for n,w in enumerate(windows) if overlap([ticks[0],ticks[1]],w)],
                      'full_job_health_overlap_indices':[n for n,w in enumerate(windows) if overlap([ticks[0],ticks[2]],w)]})
 return {'status':'passed','jobs':48,'warmups':12,'measured':36,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,
  'plan_sha256':identity.sha(RUN/'runtime_plan001.json'),'run_started_sha256':identity.sha(mon/'started.json'),'run_terminal_sha256':identity.sha(mon/'terminal.json'),
  'launch_sha256':identity.sha(out/'launch.json'),'exit_sha256':identity.sha(out/'exit.json'),'worker_complete_sha256':identity.sha(out/'worker_complete.json'),
  'journal_sha256':identity.sha(journal),'health_sha256':identity.sha(health),'snapshot_manifest_sha256':identity.sha(snap),
  'snapshot_started_sha256':identity.sha(HERE/'monitor_snapshot001/started.json'),'snapshot_terminal_sha256':identity.sha(HERE/'monitor_snapshot001/terminal.json'),
  'health_windows':windows,'annotations':annotations,'limits':'Direct health overlap only; unknown background and residual overhead cannot be assigned, no samples excluded; completion boundary includes raw fsync.',
  'verifier_sha256':identity.sha(__file__),'errors':[]}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 try:
  if args.out.resolve()!=HERE/'verification001.json':raise ValueError('unexpected output')
  vp=json.loads((HERE/'verification_plan001.json').read_text())
  for path,digest in vp['source_sha256'].items():
   if identity.sha(path)!=digest:raise ValueError('verification source changed')
  report=verify();report['verification_plan_sha256']=identity.sha(HERE/'verification_plan001.json')
 except Exception as exc:report={'status':'failed','errors':[type(exc).__name__+': '+str(exc)],'GT_opened':False,'scoring_performed':False,'verifier_sha256':identity.sha(__file__)}
 with args.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if report['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
