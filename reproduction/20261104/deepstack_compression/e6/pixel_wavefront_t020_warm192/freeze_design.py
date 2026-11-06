"""Select only preaccepted input identities, before observing new outcomes."""
import hashlib,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];sys.path.insert(0,str(B/'e4_c_latency'));import input_preflight,design
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if len(c)<2 or (cwd/c[1]).resolve()!=script.resolve() or s['worker_pid']!=t['worker_pid'] or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or (out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('prior supervisor')
def main():
 out=D/'design001.json'
 if out.exists():raise ValueError('design already exists')
 old=B/'e4_c_latency_full';formal=old/'full_001_verification.json';qpath=B/'e1_independent_review/e4c_latency_full001_review.json';reviewer=B/'e1_independent_review/review_e4c_latency_full001.py';fm=old/'monitor_verify_full001';rm=B/'e1_independent_review/monitor_e4c_latency_full001';terminal(fm,old/'full_verify.py',formal);terminal(rm,reviewer,qpath)
 f=read(formal);q=read(qpath);frozen=old/'full_001/frozen.json'
 if f.get('status')!='passed' or f.get('errors')!=[] or f.get('GT_opened') is not False or q.get('status')!='passed' or q.get('independent_review') is not True or q.get('errors')!=[] or q.get('formal_proof_sha256')!=sha(formal) or q.get('formal_started_sha256')!=sha(fm/'started.json') or q.get('formal_terminal_sha256')!=sha(fm/'terminal.json') or q.get('reviewer_sha256')!=sha(reviewer) or f['evidence_sha256'].get(str(frozen.resolve()))!=sha(frozen):raise ValueError('accepted cohort evidence')
 e1=B/'e1';rows,refs=input_preflight.load_accepted_e1_references(e1/'e1full001_completion.json',e1/'monitorverifye1full001/started.json',e1/'monitorverifye1full001/terminal.json');rule=D/'rule001.json';obj=design.build(read(rule),read(frozen)['schedule']['cohort'],rows,refs)
 paths=[rule,D/'design.py',D/'freeze_design.py',frozen,formal,fm/'started.json',fm/'terminal.json',qpath,reviewer,rm/'started.json',rm/'terminal.json',B/'e4_c_latency/input_preflight.py',B/'e4_c_latency/e1_reference_gate.py',e1/'e1full001_completion.json',e1/'monitorverifye1full001/started.json',e1/'monitorverifye1full001/terminal.json']
 obj.update({'scope':'fixed post-hoc DEV paired warm-process timing cohort; no GPU authorization','rule_sha256':sha(rule),'source_sha256':{str(p.resolve()):sha(p) for p in paths},'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'prediction_scope':'previously accepted E1 DEV mixed journal parsed solely for input identity','GPU_used':False,'limits':['source cohort reused after observing earlier three-anchor timing','image identity and processor bytes will be independently checked by actual noGPU preflight','full-generation and full high-dimensional equivalence not established']})
 with out.open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
