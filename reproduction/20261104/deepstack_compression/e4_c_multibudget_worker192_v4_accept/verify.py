"""Pre-run formal no-GT verifier for the 7920-job exploratory timing pilot."""
import argparse,importlib.util,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;RUN=BASE/'e4_c_multibudget_worker192_v4'
def load(name,path):
 p=Path(path).resolve();old=sys.modules.get(name)
 if old is not None:
  if not getattr(old,'__file__',None) or Path(old.__file__).resolve()!=p:raise ValueError('foreign verifier module '+name)
  return old
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
admission=load('_multi_timing_accept_admission',RUN/'gpu_gate.py')
def overlap(a,b):return a[0]<=b[1] and b[0]<=a[1]
def job_events(events,job,panel,fixed,tokenizer):
 if len(events)!=3 or [x.get('event') for x in events]!=['job_started','raw_generation','job_completed'] or any(x.get('job')!=job for x in events):raise ValueError('missing or unordered job events')
 raw=events[1]['raw'];rec=events[2]['record'];protocol=job['protocol'];k=job['budget']
 if any(str(raw.get(name))!=str(job.get(name)) or str(rec.get(name))!=str(job.get(name)) for name in ('task','qid','group_id','budget','protocol')):raise ValueError('job identity mismatch')
 for name in ('task','qid','group_id','budget','protocol','output_ids','generation_scalars','min_new_tokens','eos_ids'):
  if raw.get(name)!=rec.get(name):raise ValueError('raw/result mismatch '+name)
 if rec.get('selector_calls')!=(0 if k==1024 else 1):raise ValueError('selector path mismatch')
 if protocol=='natural':
  if any(rec.get(name)!=panel.get(name) for name in ('output_ids','answer','generation_scalars','eos_ids','eos','truncated')) or rec.get('min_new_tokens') is not None:raise ValueError('accepted E4 natural output mismatch')
 elif protocol=='fixed32':
  if any(rec.get(name)!=fixed.get(name) for name in ('output_ids','answer','generation_scalars','min_new_tokens','eos_ids')) or len(rec['output_ids'])!=32 or any(i in rec['eos_ids'] for i in rec['output_ids'][:-1]):raise ValueError('fixed32 reference mismatch')
 else:raise ValueError('unknown protocol')
 if tokenizer.decode(rec['output_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)!=rec['answer']:raise ValueError('independent answer decode mismatch')
 b=rec['boundaries'];ticks=[b.get(k) for k in ('t0_before_image_read_ns','first_token_selected_ns','generation_complete_ns')]
 if (any(type(x) is not int or x<0 for x in ticks) or not ticks[0]<ticks[1]<=ticks[2] or raw.get('generation_complete_ns')!=ticks[2] or b.get('first_criteria_length')!=1 or b.get('criteria_calls')!=len(rec['output_ids']) or b.get('clock_implementation')!='clock_gettime(CLOCK_MONOTONIC)' or b.get('explicit_sync_boundaries')!=['before_t0','first_token','generate_complete']):raise ValueError('timing boundary invalid')
 return ticks,b['time_namespace']
def health_windows(path,namespace):
 rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
 if not rows:raise ValueError('supervisor health windows missing')
 out=[]
 for row in rows:
  window=row.get('health_snapshot_window_monotonic_ns')
  if not isinstance(window,list) or len(window)!=2 or any(type(x) is not int or x<0 for x in window) or window[0]>window[1] or row.get('time_namespace')!=namespace:raise ValueError('bad supervisor health window')
  out.append(window)
 return out
def environment(events,jobs,namespace,model_health):
 env=BASE/'e4_c_multibudget_environment192';review=admission.read(BASE/'e1_independent_review/e4c_environment192_prelaunch_review.json');validator=env/'validate.py'
 if review['source_sha256'].get(str(validator.resolve()))!=admission.sha(validator):raise ValueError('observer validator changed')
 admission.terminal(env/'monitor_watch001',env/'watch.py')
 if not (env/'stop001').is_file():raise ValueError('observer stop marker missing')
 rows=[json.loads(line) for line in (env/'samples001.jsonl').read_text().splitlines() if line.strip()]
 validator_mod=load('_timing192_env_validator',validator);windows=validator_mod.validate(rows,namespace)
 watcher_health=health_windows(env/'monitor_watch001/health.jsonl',namespace);manual=[]
 for line in (env/'manual_checks001.jsonl').read_text().splitlines():
  if not line.strip():continue
  row=json.loads(line);a=row.get('start_monotonic_ns');b=row.get('end_monotonic_ns')
  if row.get('event')!='manual_health_check' or row.get('time_namespace')!=namespace or type(a) is not int or type(b) is not int or a>b or not row.get('start_utc') or not row.get('end_utc') or not isinstance(row.get('details'),dict):raise ValueError('invalid manual health window')
  manual.append([a,b])
 start=events[2]['record']['boundaries']['t0_before_image_read_ns'];end=events[-1]['record']['boundaries']['generation_complete_ns']
 if windows[0][0]>start or rows[-1]['end_ns']<end:raise ValueError('observer did not cover entire run')
 detailpath=BASE/'e4_c_multibudget_analysis192_independent/window_detail.py'
 if admission.sha(detailpath)!='46ee3805e63400c257be1d37c58b052b47ff9bb526908abce32944236ac0398e':raise ValueError('window contract source changed')
 contract=load('_timing192_window_detail',detailpath)
 return contract.report(jobs,events,{'model_health':model_health,'query':[[a,b] for a,b,c in windows],'query_first_fsync':[[a,c] for a,b,c in windows],'watcher_health':watcher_health,'manual_health':manual},namespace)
def verify():
 out=RUN/'run_001';mon=RUN/'monitor_run001';started=admission.terminal(mon,RUN/'launch.py');approval=admission.admit(host=False);plan=admission.plan()
 launch=admission.read(out/'launch.json');ex=admission.read(out/'exit.json');st=ex.get('last_inspect') or {};cid=(ex.get('container_identity') or {}).get('container_id')
 if (started.get('worker_pid')!=(launch.get('controller') or {}).get('pid') or launch.get('approval_bundle')!=approval or launch.get('worker_sha256')!=admission.sha(RUN/'worker.py') or launch.get('launcher_sha256')!=admission.sha(RUN/'launch.py') or launch.get('run_out')!=str(out) or ex.get('controller')!=launch.get('controller') or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('container_stopped_verified') is not True or ex.get('client_exited_while_container_running') is not False or ex.get('gpu_pids_after')!=[] or st.get('running') is not False or st.get('pid')!=0 or not cid or st.get('id')!=cid):raise ValueError('timing launcher/exit invalid')
 doc=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=10,check=True).stdout)[0]
 if doc['State']['Running'] is not False or doc['State']['Pid']!=0 or doc['State']['ExitCode']!=0 or doc['Image']!=st['image'] or doc['HostConfig']['NetworkMode']!='none' or doc['Config']['Cmd']!=[str(RUN/'worker.py'),'--out',str(out)]:raise ValueError('timing Docker state or actual command mismatch')
 snapshot=RUN/'run_001_sources';snap=snapshot/'manifest.json';admission.terminal(HERE/'monitor_snapshot001',HERE/'snapshot.py');sm=admission.read(snap)
 if sm.get('status')!='passed' or sm.get('runtime_plan_sha256')!=admission.sha(RUN/'gpu_plan001.json') or sm.get('snapshot_script_sha256')!=admission.sha(HERE/'snapshot.py') or set(sm['sources'])!=set(plan['source_sha256']):raise ValueError('timing source snapshot mismatch')
 for path,digest in plan['source_sha256'].items():
  item=sm['sources'][path];copy=Path(item['snapshot']).resolve()
  if item.get('source')!=path or item.get('sha256')!=digest or not copy.is_relative_to(snapshot.resolve()) or admission.sha(copy)!=digest:raise ValueError('timing source snapshot corrupt')
 complete=admission.read(out/'worker_complete.json');frozen=admission.read(out/'frozen.json');journal=out/'journal.jsonl'
 if (complete.get('status')!='passed' or complete.get('jobs')!=7920 or complete.get('warmups')!=240 or complete.get('measured')!=7680 or complete.get('fixed32_measure_keys')!=768 or complete.get('source_unchanged') is not True or complete.get('weights_stat_unchanged') is not True or complete.get('journal_sha256')!=admission.sha(journal) or frozen.get('approval')!=approval or frozen.get('GT_opened') is not False):raise ValueError('timing completion mismatch')
 reference=load('_multi_timing_accept_refs',BASE/'e4_c_multibudget_references192/references.py');refs=reference.load_references();sample=admission.read(BASE/'e4_c_multibudget_expansion192/proposal002.json')
 fixed=load('_multi_timing_accept_fixed',RUN/'fixed_reference_gate.py').load(sample);core=load('_multi_timing_accept_core',RUN/'core.py');core.prepare(sample,refs['rows'],refs['records'],fixed);jobs=sample['jobs']
 if frozen.get('schedule')!=jobs:raise ValueError('actual frozen schedule mismatch')
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',local_files_only=True,trust_remote_code=False)
 events=[json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
 if len(events)!=7920*3:raise ValueError('timing journal denominator')
 health=mon/'health.jsonl';snapshots=[json.loads(line) for line in health.read_text().splitlines() if line.strip()]
 windows=health_windows(health,launch['time_namespace'])
 annotations=[]
 for i,job in enumerate(jobs):
  key=(job['task'],str(job['qid']),job['budget']);ticks,ns=job_events(events[3*i:3*i+3],job,refs['records'][key],fixed[key],tokenizer)
  if ns!=launch['time_namespace']:raise ValueError('sample time namespace mismatch')
  annotations.append({'order':i,'phase':job['phase'],'task':job['task'],'budget':job['budget'],'protocol':job['protocol'],'ttft_health_overlap_indices':[n for n,w in enumerate(windows) if overlap([ticks[0],ticks[1]],w)],'full_job_health_overlap_indices':[n for n,w in enumerate(windows) if overlap([ticks[0],ticks[2]],w)]})
 observed=environment(events,jobs,launch['time_namespace'],windows)
 return {'status':'passed','jobs':7920,'warmups':240,'measured':7680,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'plan_sha256':admission.sha(RUN/'gpu_plan001.json'),'run_started_sha256':admission.sha(mon/'started.json'),'run_terminal_sha256':admission.sha(mon/'terminal.json'),'launch_sha256':admission.sha(out/'launch.json'),'exit_sha256':admission.sha(out/'exit.json'),'worker_complete_sha256':admission.sha(out/'worker_complete.json'),'journal_sha256':admission.sha(journal),'health_sha256':admission.sha(health),'snapshot_manifest_sha256':admission.sha(snap),'snapshot_started_sha256':admission.sha(HERE/'monitor_snapshot001/started.json'),'snapshot_terminal_sha256':admission.sha(HERE/'monitor_snapshot001/terminal.json'),'health_windows':windows,'annotations':annotations,'environment_job_details':observed['jobs_detail'],'hourly_health_coverage':observed['hourly_health_coverage'],'environment_limits':observed['unknown'],'samples_excluded':observed['samples_excluded'],'environment_files_sha256':{str(p):admission.sha(p) for p in (BASE/'e4_c_multibudget_environment192/samples001.jsonl',BASE/'e4_c_multibudget_environment192/manual_checks001.jsonl',BASE/'e4_c_multibudget_environment192/monitor_watch001/health.jsonl',BASE/'e4_c_multibudget_environment192/monitor_watch001/started.json',BASE/'e4_c_multibudget_environment192/monitor_watch001/terminal.json',BASE/'e4_c_multibudget_environment192/stop001')},'limits':'Recorded health call direct overlaps only; ps log tails and all other residual host activity remain unknown; all samples retained. Completion boundary is synchronized before raw output fsync and author decoding.','verifier_sha256':admission.sha(__file__),'errors':[]}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 try:
  if args.out.resolve()!=HERE/'verification001.json':raise ValueError('unexpected output')
  vp=admission.read(HERE/'verification_plan001.json')
  for path,digest in vp['source_sha256'].items():
   if admission.sha(path)!=digest:raise ValueError('verification source changed '+path)
  report=verify();report['verification_plan_sha256']=admission.sha(HERE/'verification_plan001.json')
 except Exception as exc:report={'status':'failed','errors':[type(exc).__name__+': '+str(exc)],'GT_opened':False,'scoring_performed':False,'verifier_sha256':admission.sha(__file__)}
 with args.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if report['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
