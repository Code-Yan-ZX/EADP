"""Fail closed before opening any 192-image timing journal or new result."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;I=B/'e1_independent_review';RUN=B/'e4_c_multibudget_worker192_v6';A=B/'e4_c_multibudget_worker192_v6_accept';REF=B/'e4_c_multibudget_references192';RA=B/'e4_c_multibudget_references192_accept';ENV=B/'e4_c_multibudget_environment192_v2'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or [(cwd/x).resolve() for x in c if Path(x).name==script.name]!=[script.resolve()] or (out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('analysis supervised exit/command')
 return s
def prelaunch():
 p=read(D/'analysis_code_plan001.json');c=read(D/'cpu_proof001.json');terminal(D/'monitor_cpu001',D/'freeze.py')
 if p.get('status')!='frozen' or p.get('GT_opened') is not False or p.get('target_predictions_opened') is not False or c.get('status')!='passed' or c.get('returncode')!=0 or c.get('source_sha256_before')!=c.get('source_sha256_after') or p.get('source_sha256')!=dict(c['source_sha256_after'],**{str((D/'cpu_proof001.json').resolve()):sha(D/'cpu_proof001.json')}) or any(sha(k)!=v for k,v in p['source_sha256'].items()):raise ValueError('analysis CPU/source proof')
 proof=I/'e4c_analysis192_prelaunch_review.json';reviewer=I/'review_e4c_analysis192_prelaunch.py';mon=I/'monitor_e4c_analysis192_prelaunch';terminal(mon,reviewer,proof);v=read(proof)
 fields={'plan_sha256':D/'analysis_code_plan001.json','cpu_proof_sha256':D/'cpu_proof001.json','cpu_started_sha256':D/'monitor_cpu001/started.json','cpu_terminal_sha256':D/'monitor_cpu001/terminal.json','reviewer_sha256':reviewer}
 if v.get('status')!='passed' or v.get('independent_review') is not True or v.get('GT_opened') is not False or v.get('target_predictions_opened') is not False or v.get('errors')!=[] or v.get('source_sha256')!=p['source_sha256'] or any(v.get(k)!=sha(path) for k,path in fields.items()):raise ValueError('analysis independent prelaunch')
 return {'analysis_prelaunch_sha256':sha(proof),'analysis_plan_sha256':sha(D/'analysis_code_plan001.json')}
def admitted_pair(formal,formalmon,formalscript,review,reviewmon,reviewer,runmon,runlauncher,counts):
 terminal(formalmon,formalscript,formal);terminal(reviewmon,reviewer,review);terminal(runmon,runlauncher)
 f=read(formal);r=read(review)
 if f.get('status')!='passed' or r.get('status')!='passed' or r.get('independent_review') is not True or f.get('GT_opened') is not False or r.get('GT_opened') is not False or f.get('scoring_performed') is not False or r.get('scoring_performed') is not False or f.get('errors')!=[] or r.get('errors')!=[] or any(f.get(k)!=v or r.get(k)!=v for k,v in counts.items()) or f.get('verifier_sha256')!=sha(formalscript) or f.get('verification_plan_sha256')!=sha(formalscript.parent/'verification_plan001.json'):raise ValueError('formal/independent result not accepted')
 fields={'formal_sha256':formal,'formal_started_sha256':formalmon/'started.json','formal_terminal_sha256':formalmon/'terminal.json','run_started_sha256':runmon/'started.json','run_terminal_sha256':runmon/'terminal.json','reviewer_sha256':reviewer}
 if any(r.get(k)!=sha(path) for k,path in fields.items()):raise ValueError('independent proof binding')
 return f,r
def admit():
 pre=prelaunch()
 rf,rr=admitted_pair(RA/'verification001.json',RA/'monitor_verify001',RA/'verify.py',I/'e4c_multibudget_references192_result_review.json',I/'monitor_e4c_multibudget_references192_result',I/'review_e4c_multibudget_references192_result.py',REF/'monitor_run001',REF/'launch.py',{'inputs':222,'generations':888})
 f,r=admitted_pair(A/'verification001.json',A/'monitor_verify001',A/'verify.py',I/'e4c_multibudget_worker192_v6_result_review.json',I/'monitor_e4c_multibudget_worker192_v6_result',I/'review_e4c_multibudget_worker192_v6_result.py',RUN/'monitor_run001',RUN/'launch.py',{'jobs':7920,'warmups':240,'measured':7680})
 if rf.get('timing_claims') is not False or rr.get('timing_claims') is not False:raise ValueError('reference timing claim')
 ep=I/'e4c_environment192_v2_prelaunch_review.json';er=I/'review_e4c_environment192_v2_prelaunch.py';em=I/'monitor_e4c_environment192_v2_prelaunch';terminal(em,er,ep);ev=read(ep);wp=read(ENV/'plan001.json')
 if ev.get('status')!='passed' or ev.get('independent_review') is not True or ev.get('plan_sha256')!=sha(ENV/'plan001.json') or ev.get('reviewer_sha256')!=sha(er) or ev.get('GT_opened') is not False or ev.get('errors')!=[] or ev.get('source_sha256')!=wp.get('source_sha256') or any(sha(k)!=v for k,v in wp['source_sha256'].items()):raise ValueError('observer prelaunch proof')
 wm=ENV/'monitor_watch001';s=terminal(wm,ENV/'watch.py')
 if s['command']!=wp['command'][wp['command'].index('--')+1:] or not (ENV/'stop001').is_file():raise ValueError('watcher did not cleanly stop')
 # Only after all complete formal/independent proofs and exits may files that
 # contain predictions or timing be opened even for hashing.
 journal=RUN/'run_001/journal.jsonl';health=RUN/'monitor_run001/health.jsonl';snapshot=RUN/'run_001_sources/manifest.json';records=REF/'run_001/diagnostic/records.jsonl';refjournal=REF/'run_001/diagnostic/journal.jsonl'
 for obj in (rf,rr):
  if obj.get('records_sha256')!=sha(records) or obj.get('journal_sha256')!=sha(refjournal):raise ValueError('accepted fixed32 data changed')
 if f.get('journal_sha256')!=sha(journal) or r.get('journal_sha256')!=sha(journal) or f.get('health_sha256')!=sha(health) or r.get('health_sha256')!=sha(health) or f.get('snapshot_manifest_sha256')!=sha(snapshot) or r.get('snapshot_manifest_sha256')!=sha(snapshot):raise ValueError('accepted timing files changed')
 for path,digest in f.get('environment_files_sha256',{}).items():
  if sha(path)!=digest:raise ValueError('observer evidence changed')
 expected=[ENV/'samples001.jsonl',ENV/'manual_checks001.jsonl',wm/'health.jsonl',wm/'started.json',wm/'terminal.json',ENV/'stop001']
 if set(f.get('environment_files_sha256',{}))!={str(p) for p in expected}:raise ValueError('observer inventory incomplete')
 return {**pre,'reference_formal_sha256':sha(RA/'verification001.json'),'reference_review_sha256':sha(I/'e4c_multibudget_references192_result_review.json'),'timing_formal_sha256':sha(A/'verification001.json'),'timing_review_sha256':sha(I/'e4c_multibudget_worker192_v6_result_review.json'),'journal_sha256':sha(journal),'health_sha256':sha(health)}
