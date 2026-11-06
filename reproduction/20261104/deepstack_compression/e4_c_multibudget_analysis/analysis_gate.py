"""Fail closed before parsing any multi-budget timing journal."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;R=B/'e1_independent_review';RUN=B/'e4_c_multibudget_worker_v3';A=B/'e4_c_multibudget_worker_accept_v3';EX=B/'e4_c_multibudget_expansion'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(directory,script,out=None):
 s=read(directory/'started.json');t=read(directory/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()))):raise ValueError('supervised exit/command mismatch '+str(directory))
 return s
def watcher_prelaunch():
 w=B/'e4_c_multibudget_environment';p=w/'plan001.json';r=R/'e4c_multibudget_environment_prelaunch_review.json';script=R/'review_e4c_multibudget_environment_prelaunch.py';mon=R/'monitor_e4c_multibudget_environment_prelaunch'
 if sha(p)!='3dd79ba0d438946d73b2c3565f72804083b22738a88eec2928e6835256a85a32':raise ValueError('watcher plan changed')
 plan=read(p);terminal(mon,script,r);v=read(r)
 fields={'plan_sha256':p,'cpu_started_sha256':w/'monitor_cpu001/started.json','cpu_terminal_sha256':w/'monitor_cpu001/terminal.json','reviewer_sha256':script}
 if (v.get('status')!='passed' or v.get('independent_review') is not True or v.get('scope')!='multibudget_environment_prelaunch' or v.get('errors')!=[] or v.get('GT_opened') is not False or v.get('timing_results_opened') is not False or v.get('source_sha256')!=plan.get('source_sha256') or any(sha(k)!=x for k,x in plan['source_sha256'].items()) or any(v.get(k)!=sha(x) for k,x in fields.items())):raise ValueError('watcher prelaunch proof')
 return {'watcher_plan_sha256':sha(p),'watcher_prelaunch_sha256':sha(r),'watcher_prelaunch_started_sha256':sha(mon/'started.json'),'watcher_prelaunch_terminal_sha256':sha(mon/'terminal.json'),'watcher_reviewer_sha256':sha(script)}

def prelaunch():
 watch=watcher_prelaunch();plan=read(D/'analysis_code_plan004.json');cpu=read(D/'cpu_proof004.json');mon=D/'monitor_cpu004';terminal(mon,D/'freeze.py')
 if (plan.get('status')!='frozen' or plan.get('GT_opened') is not False or plan.get('timing_results_opened') is not False or cpu.get('status')!='passed' or cpu.get('returncode')!=0 or cpu.get('source_sha256_before')!=cpu.get('source_sha256_after')):raise ValueError('analysis source/CPU proof')
 sources=dict(cpu['source_sha256_after']);sources[str((D/'cpu_proof004.json').resolve())]=sha(D/'cpu_proof004.json')
 if plan.get('source_sha256')!=sources or any(sha(x)!=v for x,v in sources.items()):raise ValueError('analysis source changed')
 terminal(EX/'monitor_freeze_analysis001',EX/'freeze_analysis_plan.py');f=read(EX/'analysis_plan_freeze001.json')
 if f.get('status')!='frozen' or f.get('analysis_plan_sha256')!=sha(EX/'ANALYSIS_PLAN.md') or f.get('pilot_plan_sha256')!=sha(EX/'pilot_plan001.json') or f.get('timing_results_opened') is not False:raise ValueError('analysis plan freeze')
 proof=D/'independent_prelaunch_review.json';reviewer=R/'review_multibudget_analysis_prelaunch.py';reviewmon=R/'monitor_multibudget_analysis_prelaunch'
 if not proof.is_file():raise ValueError('analysis locked: independent prelaunch review missing')
 terminal(reviewmon,reviewer,proof);v=read(proof);fields={'analysis_code_plan_sha256':D/'analysis_code_plan004.json','analysis_plan_sha256':EX/'ANALYSIS_PLAN.md','analysis_freeze_sha256':EX/'analysis_plan_freeze001.json','analysis_freeze_started_sha256':EX/'monitor_freeze_analysis001/started.json','analysis_freeze_terminal_sha256':EX/'monitor_freeze_analysis001/terminal.json','cpu_proof_sha256':D/'cpu_proof004.json','cpu_started_sha256':mon/'started.json','cpu_terminal_sha256':mon/'terminal.json','reviewer_sha256':reviewer,'watcher_delta_plan_sha256':D/'WATCHER_DELTA_PLAN002.md'}
 if (v.get('status')!='passed' or v.get('independent_review') is not True or v.get('scope')!='multibudget_analysis_prelaunch' or v.get('GT_opened') is not False or v.get('timing_results_opened') is not False or v.get('errors')!=[] or v.get('source_sha256')!=sources or any(v.get(k)!=value for k,value in watch.items()) or any(v.get(k)!=sha(path) for k,path in fields.items())):raise ValueError('analysis prelaunch review binding')
 return {**watch,'analysis_code_plan_sha256':sha(D/'analysis_code_plan004.json'),'analysis_prelaunch_sha256':sha(proof)}
def admit():
 pre=prelaunch();formal=A/'verification001.json';review=R/'e4c_multibudget_worker_v3_result_review.json';reviewer=R/'review_e4c_multibudget_worker_v3_result.py';rs=R/'monitor_e4c_multibudget_worker_v3_result';fm=A/'monitor_verify001';rm=RUN/'monitor_run001'
 terminal(fm,A/'verify.py',formal);terminal(rs,reviewer,review);terminal(rm,RUN/'launch.py')
 f=read(formal);v=read(review);journal=RUN/'run_001/journal.jsonl';health=rm/'health.jsonl';snapshot=RUN/'run_001_sources/manifest.json'
 if (f.get('status')!='passed' or (f.get('jobs'),f.get('warmups'),f.get('measured'))!=(1520,240,1280) or f.get('GT_opened') is not False or f.get('scoring_performed') is not False or f.get('errors')!=[] or f.get('verifier_sha256')!=sha(A/'verify.py') or f.get('verification_plan_sha256')!=sha(A/'verification_plan001.json')):raise ValueError('formal timing acceptance')
 fields={'formal_sha256':formal,'formal_started_sha256':fm/'started.json','formal_terminal_sha256':fm/'terminal.json','run_started_sha256':rm/'started.json','run_terminal_sha256':rm/'terminal.json','snapshot_manifest_sha256':snapshot,'reviewer_sha256':reviewer}
 if (v.get('status')!='passed' or v.get('independent_review') is not True or (v.get('jobs'),v.get('warmups'),v.get('measured'))!=(1520,240,1280) or v.get('GT_opened') is not False or v.get('scoring_performed') is not False or v.get('errors')!=[] or any(v.get(k)!=sha(path) for k,path in fields.items())):raise ValueError('independent timing review')
 # Watcher must have actually exited before any timing journal is opened.
 w=B/'e4_c_multibudget_environment';wp=read(w/'plan001.json');wm=w/'monitor_watch001';ws=terminal(wm,w/'watch.py')
 if ws['command']!=wp['command'][wp['command'].index('--')+1:] or Path(ws['cwd']).resolve()!=B.resolve() or not (w/'stop001').is_file():raise ValueError('watcher actual command/stop mismatch')
 artifacts={'watcher_samples_sha256':w/'samples001.jsonl','watcher_health_sha256':wm/'health.jsonl','watcher_started_sha256':wm/'started.json','watcher_terminal_sha256':wm/'terminal.json','manual_checks_sha256':w/'manual_checks001.jsonl'}
 observer={k:sha(p) for k,p in artifacts.items()}
 # Only after both formal and independent proof/exit identities are verified
 # may the prediction-bearing full journal or health log be opened.
 if f.get('launch_sha256')!=sha(RUN/'run_001/launch.json') or f.get('journal_sha256')!=sha(journal) or f.get('health_sha256')!=sha(health) or f.get('snapshot_manifest_sha256')!=sha(snapshot) or v.get('journal_sha256')!=sha(journal) or v.get('health_sha256')!=sha(health):raise ValueError('accepted timing files changed')
 return {**pre,**observer,'formal_sha256':sha(formal),'independent_result_sha256':sha(review),'journal_sha256':sha(journal),'health_sha256':sha(health)}
