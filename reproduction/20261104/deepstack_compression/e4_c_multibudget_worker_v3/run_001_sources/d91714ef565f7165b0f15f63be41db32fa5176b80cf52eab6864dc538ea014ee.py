"""Fail closed until 248 completed references have two independent proofs."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;REV=BASE/'e1_independent_review';RUN=BASE/'e4_c_multibudget_references_v3';ACC=BASE/'e4_c_multibudget_references_accept_v3'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(start,end):
 s=read(start);t=read(end)
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']!=t.get('worker_pid'):raise ValueError('reference supervisor incomplete')
 return s
def verify_evidence_hashes(formal,required):
 for name,path in required.items():
  if formal.get(name)!=sha(path):raise ValueError('248 reference formal evidence changed: '+name)
 return True
def load():
 formal=ACC/'verification001.json';review=REV/'e4c_multibudget_references_v3_result_review.json';reviewer=REV/'review_e4c_multibudget_references_v3_result.py';fm=ACC/'monitor_verify001';rm=REV/'monitor_e4c_multibudget_references_v3_result';snap=RUN/'run_001_sources/manifest.json';records=RUN/'run_001/diagnostic/records.jsonl'
 if not formal.is_file() or not review.is_file():raise ValueError('GPU locked: independent 248 fixed32 reference proof unavailable')
 f=read(formal);r=read(review);terminal(fm/'started.json',fm/'terminal.json');s=terminal(rm/'started.json',rm/'terminal.json');cmd=s['command'];cwd=Path(s['cwd']);paths=[str((cwd/x).resolve()) if not Path(x).is_absolute() else str(Path(x).resolve()) for x in cmd]
 fields={'formal_sha256':formal,'formal_started_sha256':fm/'started.json','formal_terminal_sha256':fm/'terminal.json','run_started_sha256':RUN/'monitor_run001/started.json','run_terminal_sha256':RUN/'monitor_run001/terminal.json','snapshot_manifest_sha256':snap,'records_sha256':records,'reviewer_sha256':reviewer}
 if (f.get('status')!='passed' or f.get('inputs')!=62 or f.get('generations')!=248 or f.get('GT_opened') is not False or f.get('scoring_performed') is not False or f.get('timing_claims') is not False or f.get('errors')!=[] or r.get('status')!='passed' or r.get('independent_review') is not True or r.get('inputs')!=62 or r.get('generations')!=248 or r.get('GT_opened') is not False or r.get('scoring_performed') is not False or r.get('timing_claims') is not False or r.get('errors')!=[] or any(r.get(k)!=sha(v) for k,v in fields.items()) or paths.count(str(reviewer))!=1 or cmd.count('--out')!=1 or paths[cmd.index('--out')+1]!=str(review)):raise ValueError('248 reference proof mismatch')
 terminal(RUN/'monitor_run001/started.json',RUN/'monitor_run001/terminal.json')
 evidence={'plan_sha256':RUN/'execution_plan001.json','run_started_sha256':RUN/'monitor_run001/started.json','run_terminal_sha256':RUN/'monitor_run001/terminal.json','launch_sha256':RUN/'run_001/launch.json','exit_sha256':RUN/'run_001/exit.json','worker_complete_sha256':RUN/'run_001/worker_complete.json','journal_sha256':RUN/'run_001/diagnostic/journal.jsonl','snapshot_manifest_sha256':snap,'snapshot_started_sha256':ACC/'monitor_snapshot001/started.json','snapshot_terminal_sha256':ACC/'monitor_snapshot001/terminal.json','records_sha256':records,'verifier_sha256':ACC/'verify.py','verification_plan_sha256':ACC/'verification_plan001.json'}
 verify_evidence_hashes(f,evidence);terminal(ACC/'monitor_snapshot001/started.json',ACC/'monitor_snapshot001/terminal.json')
 verification_plan=read(ACC/'verification_plan001.json')
 for path,digest in verification_plan['source_sha256'].items():
  if sha(path)!=digest:raise ValueError('248 reference verifier source changed '+path)
 snapshot=read(snap);sources=read(RUN/'execution_plan001.json')['source_sha256'];copies=snapshot.get('sources',{})
 if snapshot.get('status')!='passed' or snapshot.get('execution_plan_sha256')!=sha(RUN/'execution_plan001.json') or snapshot.get('snapshot_script_sha256')!=sha(ACC/'snapshot.py') or set(copies)!=set(sources):raise ValueError('248 reference source snapshot inventory mismatch')
 for original,digest in sources.items():
  item=copies[original];copy=Path(item['snapshot']).resolve()
  if item.get('source')!=original or item.get('sha256')!=digest or not copy.is_relative_to(snap.parent.resolve()) or sha(copy)!=digest:raise ValueError('248 reference source snapshot changed')
 out={}
 with records.open() as fobj:
  for line in fobj:
   rec=json.loads(line);key=(rec['task'],str(rec['qid']),rec['budget'])
   if key in out or rec.get('protocol')!='fixed32' or len(rec.get('output_ids',[]))!=32:raise ValueError('invalid or duplicate fixed32 reference record')
   out[key]=rec
 if len(out)!=248:raise ValueError('incomplete 248 fixed32 reference records')
 return out
