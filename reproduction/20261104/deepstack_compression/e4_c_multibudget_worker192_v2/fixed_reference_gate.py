"""Fail closed until 888 newly generated reference arms have dual acceptance."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;I=B/'e1_independent_review';RUN=B/'e4_c_multibudget_references192';A=B/'e4_c_multibudget_references192_accept'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('reference supervisor identity/exit')
 return s

def load(proposal):
 formal=A/'verification001.json';review=I/'e4c_multibudget_references192_result_review.json';reviewer=I/'review_e4c_multibudget_references192_result.py';fm=A/'monitor_verify001';rm=I/'monitor_e4c_multibudget_references192_result';runmon=RUN/'monitor_run001';snapshot=RUN/'run_001_sources/manifest.json';records=RUN/'run_001/diagnostic/records.jsonl';journal=RUN/'run_001/diagnostic/journal.jsonl'
 if not formal.is_file() or not review.is_file():raise ValueError('GPU locked: 888-reference independent acceptance missing')
 terminal(fm,A/'verify.py',formal);terminal(rm,reviewer,review);terminal(runmon,RUN/'launch.py');terminal(A/'monitor_snapshot001',A/'snapshot.py')
 f=read(formal);r=read(review);fields={'formal_sha256':formal,'formal_started_sha256':fm/'started.json','formal_terminal_sha256':fm/'terminal.json','run_started_sha256':runmon/'started.json','run_terminal_sha256':runmon/'terminal.json','snapshot_manifest_sha256':snapshot,'records_sha256':records,'journal_sha256':journal,'reviewer_sha256':reviewer}
 if f.get('status')!='passed' or (f.get('inputs'),f.get('generations'),f.get('output_tokens_per_arm'),f.get('decode_forwards_after_prefill'))!=(222,888,32,31) or f.get('GT_opened') is not False or f.get('scoring_performed') is not False or f.get('timing_claims') is not False or f.get('errors')!=[] or f.get('verifier_sha256')!=sha(A/'verify.py') or f.get('verification_plan_sha256')!=sha(A/'verification_plan001.json') or r.get('status')!='passed' or r.get('independent_review') is not True or (r.get('inputs'),r.get('generations'))!=(222,888) or r.get('GT_opened') is not False or r.get('scoring_performed') is not False or r.get('timing_claims') is not False or r.get('errors')!=[] or any(r.get(k)!=sha(v) for k,v in fields.items()):raise ValueError('reference formal/independent identity')
 evidence={'plan_sha256':RUN/'execution_plan001.json','run_started_sha256':runmon/'started.json','run_terminal_sha256':runmon/'terminal.json','launch_sha256':RUN/'run_001/launch.json','exit_sha256':RUN/'run_001/exit.json','worker_complete_sha256':RUN/'run_001/worker_complete.json','snapshot_manifest_sha256':snapshot,'snapshot_started_sha256':A/'monitor_snapshot001/started.json','snapshot_terminal_sha256':A/'monitor_snapshot001/terminal.json','journal_sha256':journal,'records_sha256':records}
 if any(f.get(k)!=sha(v) for k,v in evidence.items()):raise ValueError('reference evidence changed')
 plan=read(RUN/'execution_plan001.json');snap=read(snapshot);source=plan['source_sha256']
 if snap.get('status')!='passed' or snap.get('execution_plan_sha256')!=sha(RUN/'execution_plan001.json') or snap.get('snapshot_script_sha256')!=sha(A/'snapshot.py') or set(snap.get('sources',{}))!=set(source):raise ValueError('reference snapshot inventory')
 for original,digest in source.items():
  item=snap['sources'][original];path=Path(item['snapshot']).resolve()
  if item.get('source')!=original or item.get('sha256')!=digest or not path.is_relative_to(snapshot.parent.resolve()) or sha(path)!=digest:raise ValueError('reference snapshot bytes')
 anchors=proposal['measurement_identities']+proposal['warmup_identities'];by={(x['task'],str(x['qid'])):x for x in anchors}
 if len(by)!=222:raise ValueError('proposal input denominator')
 out={}
 with records.open() as stream:
  for line in stream:
   if not line.strip():continue
   x=json.loads(line);qid=(x['task'],str(x['qid']));key=(*qid,x['budget'])
   if qid not in by or key in out or x.get('budget') not in (128,256,512,1024) or x.get('protocol')!='fixed32' or any(str(x.get(k))!=str(by[qid].get(k)) for k in ('task','qid','group_id','image_sha256','message_sha256')):raise ValueError('fixed reference identity')
   out[key]=x
 if len(out)!=888:raise ValueError('fixed reference denominator')
 return out
