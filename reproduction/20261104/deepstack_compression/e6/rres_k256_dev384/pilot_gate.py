"""Require the real three-anchor formal and independent exits; no GPU authority."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];P=B/'e6/rres_k256_execution_v2';I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if len(c)<2 or (cwd/c[1]).resolve()!=script.resolve() or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or (out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervisor did not cleanly exit')
def require():
 proof=I/'rres_k256_v2_result_review.json';reviewer=I/'review_rres_k256_v2_result.py';monitor=I/'monitor_rres_k256_v2_result';formal=P/'verification001.json'
 q=read(proof);v=read(formal);terminal(monitor,reviewer,proof)
 fields={'formal_sha256':formal,'formal_started_sha256':P/'monitor_verify001/started.json','formal_terminal_sha256':P/'monitor_verify001/terminal.json','run_started_sha256':P/'monitor_run001/started.json','run_terminal_sha256':P/'monitor_run001/terminal.json','snapshot_manifest_sha256':P/'source_snapshot001/manifest.json','reviewer_sha256':reviewer}
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('images')!=3 or q.get('GT_opened') is not False or q.get('scoring_performed') is not False or q.get('timing_claims') is not False or q.get('errors')!=[] or any(q.get(k)!=sha(p) for k,p in fields.items()):raise ValueError('independent pilot proof')
 if v.get('status')!='passed' or v.get('images')!=3 or v.get('GT_opened') is not False or v.get('scoring_performed') is not False or v.get('errors')!=[]:raise ValueError('formal pilot result')
 for monitor_path in (P/'monitor_verify001',P/'monitor_run001'):
  s=read(monitor_path/'started.json');t=read(monitor_path/'terminal.json')
  if type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[]:raise ValueError('pilot execution did not cleanly exit')
 return {'formal_sha256':sha(formal),'review_sha256':sha(proof),'reviewer_sha256':sha(reviewer),'review_started_sha256':sha(monitor/'started.json'),'review_terminal_sha256':sha(monitor/'terminal.json')}
