"""Candidate analysis admission; never reads timing journals before both proofs."""
import hashlib,json,time
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def completed(monitor,script,out=None):
 d=Path(monitor);s=read(d/'started.json');t=read(d/'terminal.json');cwd=Path(s['cwd']);cmd=s['command'];actual=[(cwd/x).resolve() for x in cmd if Path(x).name==Path(script).name]
 if (s.get('worker_pid')!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or actual!=[Path(script).resolve()]):raise ValueError('supervisor exit/source')
 if out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=Path(out).resolve()):raise ValueError('supervisor out')
 return s,t

def admit(plan,read_json=read,hash_file=sha,terminal=completed):
 """Only source/metadata and proofs accessed here, never the timing output list."""
 if (plan.get('status')!='frozen' or plan.get('images')!=32 or plan.get('jobs')!=380 or
     plan.get('warmups')!=60 or plan.get('measured')!=320 or plan.get('processes')!=12 or
     plan.get('GT_opened') is not False or time.time()>=plan['budget_end_unix']):raise ValueError('analysis plan/deadline')
 for path,digest in plan['source_sha256'].items():
  if hash_file(path)!=digest:raise ValueError('analysis source changed')
 f=plan['future'];formal=read_json(f['formal']);independent=read_json(f['independent'])
 terminal(f['run_monitor'],f['launcher']);terminal(f['formal_monitor'],f['verifier'],f['formal']);terminal(f['independent_monitor'],f['reviewer'],f['independent'])
 common={'status':'passed','images':32,'jobs':380,'warmups':60,'measured':320,'processes':12,'GT_opened':False,'errors':[]}
 for proof in (formal,independent):
  if any(proof.get(k)!=v for k,v in common.items()):raise ValueError('incomplete acceptance')
 if independent.get('independent_review') is not True or independent.get('formal_sha256')!=hash_file(f['formal']) or independent.get('reviewer_sha256')!=hash_file(f['reviewer']):raise ValueError('independent proof binding')
 if formal.get('execution_plan_sha256')!=hash_file(f['execution_plan']) or independent.get('execution_plan_sha256')!=hash_file(f['execution_plan']) or formal.get('verifier_sha256')!=hash_file(f['verifier']):raise ValueError('execution plan/verifier binding')
 for field,path in (('run_started_sha256',Path(f['run_monitor'])/'started.json'),('run_terminal_sha256',Path(f['run_monitor'])/'terminal.json'),('formal_started_sha256',Path(f['formal_monitor'])/'started.json'),('formal_terminal_sha256',Path(f['formal_monitor'])/'terminal.json')):
  if independent.get(field)!=hash_file(path):raise ValueError('proof exit binding')
 files=formal.get('files_sha256')
 if not isinstance(files,dict) or independent.get('files_sha256')!=files or set(plan['required_result_paths'])-set(files):raise ValueError('incomplete result file binding')
 return {'formal_sha256':hash_file(f['formal']),'independent_sha256':hash_file(f['independent']),'files_sha256':files}

def verify_files(receipt,hash_file=sha):
 """Invoke only after admit succeeds; may read complete results, not GT."""
 for path,digest in receipt['files_sha256'].items():
  if hash_file(path)!=digest:raise ValueError('accepted result changed')
 return True
