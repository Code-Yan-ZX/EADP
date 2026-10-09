"""Independent five-arm followup; never mutates or resumes the original queues."""
import argparse,fcntl,hashlib,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=Path('/media/disk2/YZX/research/EADP_amp')
BASE=ROOT/'Qwen_vl/outputs/audit_followup_20261008'
C=BASE/'streamwait_continuation_20261008'
PLAN=OUT/'followup.plan.json';STATE=OUT/'followup.state.json';MARKER=OUT/'followup.finished.json'
WORKER=OUT/'followup_worker.py'
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
OLD=module(C/'schedule_legacy_streamwait_continuation.py','prior_legacy_helpers')
read,sha,atomic,now,process=OLD.read,OLD.sha,OLD.atomic,OLD.now,OLD.process
GROUPS=[dict(group_id='followup_next_gqa_K128_beta2',model='next',task='gqa',budget=128,kind='reused_pair',job_ids=['next_gqa_E128'],reused_job_id='next_gqa_LRMAIN025'),dict(group_id='followup_v15_vizwiz_K128_release',model='v15',task='vizwiz',budget=128,kind='pair',job_ids=['v15_vizwiz_E128','v15_vizwiz_AZ128']),dict(group_id='followup_next_vizwiz_K128_release',model='next',task='vizwiz',budget=128,kind='pair',job_ids=['next_vizwiz_E128','next_vizwiz_AZ128'])]
def verify(plan):
 if sha(__file__)!=plan['controller_sha256'] or sha(WORKER)!=plan['worker_sha256']:raise ValueError('Followup frozen code drift')
 for p,h in plan['source_and_input_sha256'].items():
  if sha(p)!=h:raise ValueError('Followup frozen source/input drift: '+p)
 if plan['groups']!=GROUPS or set(plan['jobs'])!=set(k for g in GROUPS for k in g['job_ids']):raise ValueError('Followup scope changed')
def dependencies():
 cert=read(OUT/'original_batch_certificate.json')
 if not cert['success'] or cert['published_groups_confirmed']!=55 or cert['completed_arms']!=75:raise ValueError('Original certificate incomplete')
 for p,h in cert['frozen_completion_files'].items():
  if sha(p)!=h:raise ValueError('Original completion source changed: '+p)
 for pid in cert['exited_pids']:
  if process(pid)['alive']:raise ValueError('Original completed process PID is occupied; inspect identity before launch')
 return cert

def fairness(group,jobs):
 selected=[jobs[k] for k in group['job_ids']]
 if group['kind']=='reused_pair':selected.append(read(C/'legacy_streamwait_continuation.plan.json')['jobs'][group['reused_job_id']])
 params=[{k:v for k,v in j['parameters'].items() if k!='anchorzip'} for j in selected]
 if len(params)!=2 or params[0]!=params[1]:raise ValueError('Full paired recipe differs')
 protocols=[read(j['control_protocol']) for j in selected]
 for field in ['identity_manifest_sha256','image_manifest_sha256','AST_validation']:
  if protocols[0][field]!=protocols[1][field]:raise ValueError('Paired identity/image/wait differs: '+field)
 if protocols[0]['origin']['native_entry']!=protocols[1]['origin']['native_entry']:raise ValueError('Paired native entry differs')
 # Overlapping production/scorer sources must be byte-identical. New wrapper adds observation only.
 common=set(protocols[0]['source_and_input_sha256'])&set(protocols[1]['source_and_input_sha256'])
 if any(protocols[0]['source_and_input_sha256'][p]!=protocols[1]['source_and_input_sha256'][p] for p in common):raise ValueError('Paired shared source differs')
 return dict(success=True,shared_sources=len(common),same_parameters_except_method=True,same_questions_images_wait=True,additional_observation='New worker records actual model attributes; no pre-vision synchronization or method math change')

def prepare():
 if PLAN.exists() or STATE.exists():raise FileExistsError('Preserve prepared plan/state')
 w=module(WORKER,'new_prepare');jobs={};sources={}
 for g in GROUPS:
  for key in g['job_ids']:
   protocol,paths=w.prepare(key,1);j=dict(w.job_specs()[key],attempt=1,control_protocol=str(paths['control_protocol']),control_protocol_sha256=sha(paths['control_protocol']),output_paths={k:str(v)for k,v in paths.items()},generate_command=protocol['generate_command']);jobs[key]=j
   sources.update(protocol['source_and_input_sha256']);sources[str(paths['control_protocol'])]=sha(paths['control_protocol'])
   for f in ['identity_manifest','image_manifest']:sources[protocol[f]]=protocol[f+'_sha256']
 for p in [C/'legacy_streamwait_worker.py',C/'schedule_legacy_streamwait_continuation.py',C/'schedule_streamwait_repair_continuation.py',BASE/'next_gap_diagnosis_20261008/schedule_next_full_controls.py',OUT/'original_batch_certificate.json',OUT/'POLICY.json',OUT/'inventory.json',OUT/'followup_publisher.py',OUT/'refresh_tables.py',OUT/'pair_statistics.py',BASE/'submission_tables_20261009/build_tables.py',BASE/'submission_tables_20261009/render_tables.py',ROOT/'LLaVA/llava/eval/m4c_evaluator.py',Path('/media/disk2/YZX/research/audit_base_gap_20261003/eadp_official/LLaVA/scripts/eval_vizwiz.py')]:
  sources[str(p)]=sha(p)
 reuse=read(C/'legacy_streamwait_continuation.plan.json')['jobs']['next_gqa_LRMAIN025']
 for p in [reuse['control_protocol'],*reuse['output_paths'].values()]:
  sources[str(p)]=sha(p)
 plan=dict(prepared_utc=now(),n_jobs=5,n_groups=3,n_predictions=29854,groups=GROUPS,jobs=jobs,source_and_input_sha256=sources,controller_sha256=sha(__file__),worker_path=str(WORKER),worker_sha256=sha(WORKER),environment=OLD.FROZEN.UTIL.environment(),fairness={g['group_id']:fairness(g,jobs)for g in GROUPS},reuse=dict(job=reuse,required_original_group='legacy_next_gqa_AZ_K128_streamwait'),scheduling='Fixed declared order; <=2 model/loading reservations. Second requires first real context and complete prediction/runtime. Each pruned job free>=22500MiB. No new FULL.',authorization='User post-batch request, five prespecified arms/three new groups independent of original75/55')
 verify(plan);dependencies();atomic(PLAN,plan);atomic(STATE,dict(status='prepared',started_utc=None,registered_jobs=[],publication_expected_groups=[],jobs={},child_pids=[],plan=str(PLAN),plan_sha256=sha(PLAN),updated_utc=now()))
 print(json.dumps(dict(prepared=True,jobs=5,predictions=29854,plan_sha256=sha(PLAN))))

def validate_job(job,plan):
 verify(plan)
 if sha(job['control_protocol'])!=job['control_protocol_sha256']:raise ValueError('Protocol drift')
 return module(WORKER,'new_validate').validate_finished(read(job['control_protocol']))

def run():
 lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 plan=read(PLAN);state=read(STATE)
 if state['status']!='prepared' or state['plan_sha256']!=sha(PLAN):raise ValueError('Preserve prior registration')
 verify(plan);cert=dependencies();ordered=list(plan['jobs']);active={}
 state.update(status='running',started_utc=now(),supervisor_pid=os.getpid(),supervisor_identity=process(os.getpid()),registered_jobs=ordered,publication_expected_groups=[g['group_id'] for g in GROUPS],jobs={k:dict(j,status='pending')for k,j in plan['jobs'].items()},groups={g['group_id']:dict(g,status='pending')for g in GROUPS},dependency_certificate_sha256=sha(OUT/'original_batch_certificate.json'));atomic(STATE,state)
 def interrupted(signum,frame):raise KeyboardInterrupt('Owned followup supervisor interrupted')
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  while True:
   for key,e in list(active.items()):
    code=e['child'].poll()
    if code is None:continue
    e['log'].close();del active[key]
    if code:state['jobs'][key].update(status='failed',returncode=code);raise RuntimeError('Followup worker exit '+str(code)+': '+key)
    val=validate_job(plan['jobs'][key],plan);state['jobs'][key].update(status='complete',returncode=0,validation=val,finished_utc=now())
   for g in GROUPS:
    if all(state['jobs'][k]['status']=='complete' for k in g['job_ids']):state['groups'][g['group_id']].update(status='complete',finished_utc=now())
   pending=[k for k in ordered if state['jobs'][k]['status']=='pending']
   if not pending and not active:break
   pids={e['child'].pid for e in active.values()};progress={e['child'].pid:all(OLD.FROZEN.first_json(plan['jobs'][k]['output_paths'][f]) for f in ('prediction','runtime'))for k,e in active.items()}
   for k,e in active.items():
    state['jobs'][k]['status']='generating' if progress[e['child'].pid] else 'loading_reserved'
    watched=[Path(plan['jobs'][k]['output_paths'][f])for f in ['prediction','runtime','log']]
    sig=[(p.stat().st_size,p.stat().st_mtime_ns) if p.exists() else None for p in watched]
    if sig!=e.get('progress_signature'):e.update(progress_signature=sig,last_progress=time.monotonic())
    elapsed=time.monotonic()-e['launched_monotonic']
    if (not progress[e['child'].pid] and elapsed>600) or time.monotonic()-e['last_progress']>600:raise RuntimeError('Confirmed followup load/stall timeout; preserve partial: '+k)
   gpu=OLD.FROZEN.UTIL.gpu_snapshot();job=plan['jobs'][pending[0]] if pending else next(iter(plan['jobs'].values()));gate=OLD.launch_gate(job,gpu,pids,progress,True,[plan['jobs'][k]for k in active])
   state.update(status='running' if active else 'waiting_for_GPU_slot',child_pids=sorted(pids),gpu_gate=gate,updated_utc=now());atomic(STATE,state)
   if pending and gate['ready']:
    verify(plan);dependencies();key=pending[0];job=plan['jobs'][key];OLD.PRIOR.protected(job)
    fresh=OLD.launch_gate(job,OLD.FROZEN.UTIL.gpu_snapshot(),pids,progress,True,[plan['jobs'][k]for k in active])
    if not fresh['ready']:time.sleep(1);continue
    log=Path(job['output_paths']['log']).open('x');child=subprocess.Popen(job['generate_command'],cwd=ROOT/'LLaVA',env=dict(os.environ,**plan['environment']),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    active[key]=dict(child=child,log=log,last_progress=time.monotonic(),launched_monotonic=time.monotonic());state['jobs'][key].update(status='loading_reserved',pid=child.pid,process_identity=process(child.pid),started_utc=now(),generation_launch_gate=fresh);state.update(child_pids=sorted(e['child'].pid for e in active.values()),updated_utc=now());atomic(STATE,state)
   time.sleep(1)
  state.update(status='complete',child_pids=[],finished_utc=now(),updated_utc=now());atomic(STATE,state);atomic(MARKER,dict(success=True,complete=True,n_jobs=5,n_groups=3,registered_jobs=ordered,publication_expected_groups=state['publication_expected_groups'],plan_sha256=sha(PLAN),finished_utc=now()))
 except BaseException as exc:
  for k,e in active.items():
   OLD.FROZEN.UTIL.stop_owned(e['child']);e['log'].close();state['jobs'][k].update(status='interrupted_preserved',error=repr(exc))
  state.update(status='failed',error=repr(exc),child_pids=[],updated_utc=now());atomic(STATE,state);raise

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--run',action='store_true');a.add_argument('--prepare',action='store_true');args=a.parse_args();run()if args.run else prepare()
