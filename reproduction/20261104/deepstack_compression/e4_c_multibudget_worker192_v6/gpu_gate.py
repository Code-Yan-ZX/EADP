"""Prospective GPU gate for 192 timing; incomplete evidence fails closed.

No source plan, CPU-container-result review, 888-reference result review,
analysis/environment prelaunch review or final GPU review currently exists.
Nothing here creates proofs or starts a GPU.
"""
import hashlib,importlib.util,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;I=B/'e1_independent_review';ENV=B/'e4_c_multibudget_environment192_v2';AN=B/'e4_c_multibudget_analysis192_exec'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(name,path,digest):
 path=Path(path).resolve()
 if sha(path)!=digest or name in sys.modules:raise ValueError('unreviewed/preloaded module '+name)
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('GPU admission supervisor/command')
 return s
def plan():
 p=read(D/'gpu_plan001.json')
 if p.get('stage')!='192-image four-budget timing' or (p.get('jobs'),p.get('warmups'),p.get('measured'))!=(7920,240,7680) or p.get('GT_opened') is not False or p.get('GPU_authorized_by_plan') is not False or not p.get('source_sha256') or any(sha(k)!=v for k,v in p['source_sha256'].items()):raise ValueError('GPU source plan invalid')
 if any(str((D/name).resolve()) not in p['source_sha256'] for name in ('worker.py','launch.py','gpu_gate.py','cpu_gate.py','core.py','fixed_reference_gate.py')) or any(type(p.get(k)) is not int or p[k]<=0 for k in ('inner_timeout_seconds','outer_timeout_seconds','stale_seconds','minimum_free_bytes','acceptance_reserve_seconds')) or not p['inner_timeout_seconds']+90<=p['outer_timeout_seconds'] or p['outer_timeout_seconds']!=28800 or p['stale_seconds']>p['inner_timeout_seconds'] or p['acceptance_reserve_seconds']<3600:raise ValueError('unsafe resource bounds/source inventory')
 return p
def observer_lifetime(start_ns,now_ns,duration_seconds,outer_seconds,cleanup_seconds):
 if any(type(v) is not int for v in (start_ns,now_ns,duration_seconds,outer_seconds,cleanup_seconds)) or start_ns<0 or start_ns>now_ns or duration_seconds<=0 or outer_seconds<=0 or cleanup_seconds<1200 or start_ns+duration_seconds*1000000000<now_ns+(outer_seconds+cleanup_seconds)*1000000000:raise ValueError('observer duration cannot cover GPU outer timeout and cleanup')
def observer_process(s,actual,state,cwd):
 pid=s.get('worker_pid')
 if type(pid) is not int or pid<=0 or s.get('process_group')!=pid or state=='Z' or actual!=s.get('command') or Path(s['cwd']).resolve()!=Path(cwd).resolve():raise ValueError('observer process identity mismatch')
def active_observer(p):
 # Host-side only: exact watcher plan, actual live process and first fsynced pair.
 planfile=ENV/'plan001.json';pre=I/'e4c_environment192_v2_prelaunch_review.json';reviewer=I/'review_e4c_environment192_v2_prelaunch.py';mon=I/'monitor_e4c_environment192_v2_prelaunch';watch=ENV/'monitor_watch001'
 terminal(mon,reviewer,pre);r=read(pre);w=read(planfile);s=read(watch/'started.json')
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('plan_sha256')!=sha(planfile) or r.get('reviewer_sha256')!=sha(reviewer) or r.get('GT_opened') is not False or r.get('errors')!=[] or w.get('GPU_authorized') is not False or (watch/'terminal.json').exists() or (ENV/'stop001').exists():raise ValueError('watcher prelaunch/active gate')
 if not isinstance(s.get('worker_pid'),int) or s['worker_pid']<=0 or s.get('command')!=w['command'][w['command'].index('--')+1:]:raise ValueError('watcher command/PID')
 pid=s['worker_pid'];stat=Path('/proc')/str(pid)/'stat'
 try:
  value=stat.read_text();tail=value[value.rfind(')')+2:].split()
  actual=[x.decode() for x in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if x]
  observer_process(s,actual,tail[0],B)
 except (OSError,IndexError):raise ValueError('observer not alive')
 with (ENV/'samples001.jsonl').open() as stream:a=json.loads(stream.readline());b=json.loads(stream.readline())
 if not (ENV/'manual_checks001.jsonl').is_file():raise ValueError('manual checks file missing; empty is allowed')
 namespace=os.readlink('/proc/self/ns/time')
 if a.get('event')!='snapshot' or b.get('event')!='logging_complete' or a.get('index')!=0 or b.get('index')!=0 or a.get('returncode')!=0 or a.get('time_namespace')!=namespace or b.get('time_namespace')!=namespace or a.get('query_start_ns')!=b.get('query_start_ns') or not a['query_start_ns']<=a['query_end_ns']<=b['first_fsync_end_ns']:raise ValueError('first observer window incomplete')
 command=s['command']
 if command.count('--duration')!=1:raise ValueError('observer duration unspecified')
 duration=int(command[command.index('--duration')+1]);watch_start=a.get('watch_start_monotonic_ns')
 if type(watch_start) is not int or watch_start>a['query_start_ns']:raise ValueError('observer start monotonic time missing')
 observer_lifetime(watch_start,time.monotonic_ns(),duration,p['outer_timeout_seconds'],1200)
 return {'watcher_started_sha256':sha(watch/'started.json'),'watcher_prelaunch_sha256':sha(pre),'watcher_plan_sha256':sha(planfile)}
def admit(host=False):
 p=plan()
 cg=load('_timing192_gpu_cpu_gate',D/'cpu_gate.py',p['source_sha256'][str((D/'cpu_gate.py').resolve())]);cpu=cg.admit()
 rp=D/'run_preflight001';cm=D/'monitor_preflight001';proof=I/'e4c_worker192_v6_cpu_result_review.json';reviewer=I/'review_e4c_worker192_v6_cpu_result.py';rm=I/'monitor_e4c_worker192_v6_cpu_result'
 terminal(cm,D/'launch.py');terminal(rm,reviewer,proof);r=read(proof);fields={'worker_preflight_sha256':rp/'worker_preflight.json','processor_preflight_sha256':rp/'processor_preflight.json','launch_sha256':rp/'launch.json','exit_sha256':rp/'exit.json','started_sha256':cm/'started.json','terminal_sha256':cm/'terminal.json','reviewer_sha256':reviewer}
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('scope')!='worker192_cpu_preflight_result' or r.get('errors')!=[] or r.get('GT_opened') is not False or r.get('model_loaded') is not False or r.get('cuda_initialized') is not False or any(r.get(k)!=sha(v) for k,v in fields.items()):raise ValueError('CPU container result review')
 w=read(rp/'worker_preflight.json');v=read(rp/'processor_preflight.json');ex=read(rp/'exit.json');st=ex.get('last_inspect') or {}
 if w.get('status')!='passed' or (w.get('inputs'),w.get('jobs'))!=(222,7920) or w.get('GPU_authorized') is not False or w.get('model_loaded') is not False or w.get('cuda_initialized') is not False or v.get('inputs_checked')!=222 or v.get('references_checked')!=888 or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('container_stopped_verified') is not True or ex.get('client_exited_while_container_running') is not False or st.get('running') is not False or st.get('pid')!=0 or st.get('host_devices') or st.get('device_requests'):raise ValueError('actual CPU preflight state')
 proposal=read(B/'e4_c_multibudget_expansion192/proposal002.json')
 if sha(B/'e4_c_multibudget_expansion192/proposal002.json')!=p['proposal_sha256']:raise ValueError('proposal changed')
 fg=load('_timing192_gpu_fixed_gate',D/'fixed_reference_gate.py',p['source_sha256'][str((D/'fixed_reference_gate.py').resolve())]);fixed=fg.load(proposal)
 if len(fixed)!=888:raise ValueError('reference records incomplete')
 analysis=I/'e4c_analysis192_prelaunch_review.json';analysisreviewer=I/'review_e4c_analysis192_prelaunch.py';analysis_mon=I/'monitor_e4c_analysis192_prelaunch';terminal(analysis_mon,analysisreviewer,analysis);ar=read(analysis)
 if ar.get('status')!='passed' or ar.get('independent_review') is not True or ar.get('GT_opened') is not False or ar.get('target_predictions_opened') is not False or ar.get('reviewer_sha256')!=sha(analysisreviewer) or ar.get('plan_sha256')!=sha(AN/'analysis_code_plan001.json') or ar.get('errors')!=[]:raise ValueError('analysis prelaunch')
 if not ar.get('source_sha256') or any(sha(k)!=v for k,v in ar['source_sha256'].items()):raise ValueError('analysis source changed')
 formal=I/'e4c_worker192_v6_accept_prelaunch_review.json';formalreviewer=I/'review_e4c_worker192_v6_accept_prelaunch.py';formalmon=I/'monitor_e4c_worker192_v6_accept_prelaunch';terminal(formalmon,formalreviewer,formal);fr=read(formal)
 if fr.get('status')!='passed' or fr.get('independent_review') is not True or fr.get('GT_opened') is not False or fr.get('reviewer_sha256')!=sha(formalreviewer) or fr.get('plan_sha256')!=sha(B/'e4_c_multibudget_worker192_v6_accept/verification_plan001.json') or fr.get('errors')!=[] or not fr.get('source_sha256') or any(sha(k)!=v for k,v in fr['source_sha256'].items()):raise ValueError('formal verifier prelaunch')
 # The independently reviewed controller/observer plans must be complete; an
 # absent plan or proof means no GPU authority, including in the container.
 env=I/'e4c_environment192_v2_prelaunch_review.json';envreviewer=I/'review_e4c_environment192_v2_prelaunch.py';envmon=I/'monitor_e4c_environment192_v2_prelaunch';terminal(envmon,envreviewer,env);er=read(env)
 if er.get('status')!='passed' or er.get('independent_review') is not True or er.get('GT_opened') is not False or er.get('reviewer_sha256')!=sha(envreviewer) or er.get('plan_sha256')!=sha(ENV/'plan001.json') or er.get('errors')!=[]:raise ValueError('observer prelaunch')
 if not er.get('source_sha256') or any(sha(k)!=v for k,v in er['source_sha256'].items()):raise ValueError('observer source changed')
 gp=I/'e4c_worker192_v6_gpu_prelaunch_review.json';gr=I/'review_e4c_worker192_v6_gpu_prelaunch.py';gm=I/'monitor_e4c_worker192_v6_gpu_prelaunch';terminal(gm,gr,gp);g=read(gp)
 fields={'gpu_plan_sha256':D/'gpu_plan001.json','cpu_preflight_review_sha256':proof,'fixed_formal_sha256':B/'e4_c_multibudget_references192_accept/verification001.json','fixed_review_sha256':I/'e4c_multibudget_references192_result_review.json','analysis_review_sha256':analysis,'environment_review_sha256':env,'formal_prelaunch_review_sha256':formal,'reviewer_sha256':gr}
 if g.get('status')!='passed' or g.get('independent_review') is not True or g.get('GT_opened') is not False or g.get('scoring_performed') is not False or g.get('errors')!=[] or g.get('source_sha256')!=p['source_sha256'] or any(g.get(k)!=sha(v) for k,v in fields.items()):raise ValueError('GPU independent prelaunch')
 if host:active_observer(p)
 return {'status':'passed','gpu_plan_sha256':sha(D/'gpu_plan001.json'),'proposal_sha256':p['proposal_sha256'],'fixed_review_sha256':sha(I/'e4c_multibudget_references192_result_review.json'),'cpu_result_review_sha256':sha(proof),'analysis_review_sha256':sha(analysis),'environment_review_sha256':sha(env),'formal_prelaunch_review_sha256':sha(formal),'gpu_review_sha256':sha(gp)}
