"""Read-only upstream identity checks. GPU execution remains hard locked.

This is not yet a full admission bundle: a separate frozen full-panel code
review, budget and true no-GPU container preflight must be implemented.
"""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def terminal(started,ended,script,out=None):
 s,t=read(started),read(ended);cmd=s.get('command',[]);cwd=Path(s.get('cwd','/'))
 if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or
     s.get('worker_pid')!=t.get('worker_pid') or not isinstance(s.get('worker_pid'),int)):
  raise ValueError('verifier/reviewer did not exit cleanly')
 if [(cwd/x).resolve() for x in cmd if Path(x).name==Path(script).name]!=[Path(script).resolve()]:raise ValueError('wrong verifier/reviewer command')
 if out is not None and (cmd.count('--out')!=1 or cmd.index('--out')+1>=len(cmd) or (cwd/cmd[cmd.index('--out')+1]).resolve()!=Path(out).resolve()):
  raise ValueError('verifier/reviewer output mismatch')

def files_match(entries):
 if not isinstance(entries,dict) or not entries:raise ValueError('missing evidence inventory')
 for p,digest in entries.items():
  if sha(p)!=digest:raise ValueError('evidence changed: '+p)

def smoke_gate():
 d=ROOT/'e2_e3_smoke';r=ROOT/'e1_independent_review';proof=d/'run_001_verification.json';review=r/'e2e3_run001_review.json';p=read(proof);q=read(review)
 if (p.get('status')!='passed' or p.get('questions')!=3 or p.get('natural_verified')!=21 or p.get('forced_verified')!=27 or
     p.get('forced_kv_reference_pairs')!=6 or p.get('kv_fingerprint_steps')!=33 or p.get('GT_opened') is not False or p.get('scoring_performed') is not False):raise ValueError('E2/E3 formal smoke not passed')
 if (q.get('status')!='passed' or q.get('independent_review') is not True or q.get('n_records')!=48 or q.get('GT_opened') is not False or q.get('scoring_performed') is not False or q.get('formal_sha256')!=sha(proof) or q.get('n_kv_reference_pairs')!=6 or q.get('kv_steps')!=33):
  raise ValueError('independent E2/E3 smoke review missing or mismatched')
 terminal(d/'monitor_accept001/started.json',d/'monitor_accept001/terminal.json',d/'accept.py',proof)
 if q.get('accept_terminal_sha256')!=sha(d/'monitor_accept001/terminal.json') or q.get('run_terminal_sha256')!=sha(d/'monitor_001/terminal.json'):raise ValueError('formal smoke supervisor proof mismatch')
 reviewer=r/'review_e2e3_run001.py';rs=r/'monitor_e2e3run001/started.json';rt=r/'monitor_e2e3run001/terminal.json'
 if q.get('reviewer_sha256')!=sha(reviewer):raise ValueError('independent smoke reviewer source changed')
 terminal(rs,rt,reviewer)
 files_match(p['files_sha256']);files_match(p['array_sha256'])
 manifest_path=d/'run_001_sources/manifest.json';manifest=read(manifest_path);frozen=read(d/'run_001/frozen.json')
 for source,entry in manifest.items():
  snapshot=(d/'run_001_sources'/entry['snapshot']).resolve()
  if snapshot.parent!=(d/'run_001_sources').resolve() or sha(source)!=entry['sha256'] or sha(snapshot)!=entry['sha256']:raise ValueError('smoke source snapshot changed')
 for source,digest in frozen['source_sha256'].items():
  if manifest.get(source,{}).get('sha256')!=digest:raise ValueError('smoke source missing from snapshot')
 if p.get('source_sha256')!=sha(d/'accept.py') or manifest.get(str(d/'accept.py'),{}).get('sha256')!=p['source_sha256']:raise ValueError('formal verifier source mismatch')
 return {'proof_sha256':sha(proof),'review_sha256':sha(review),'formal_started_sha256':sha(d/'monitor_accept001/started.json'),'formal_terminal_sha256':sha(d/'monitor_accept001/terminal.json'),'review_started_sha256':sha(rs),'review_terminal_sha256':sha(rt)}

def b_gate():
 d=ROOT/'e4_b_identity';r=ROOT/'e1_independent_review';proof=d/'b_001_verification.json';review=r/'b001_review.json';p=read(proof);q=read(review)
 if (p.get('status')!='passed' or p.get('generations')!=8 or p.get('natural_pairs')!=3 or p.get('forced_pairs')!=1 or p.get('GT_opened') is not False or p.get('scoring_performed') is not False):raise ValueError('E4-B formal proof failed')
 if (q.get('status')!='passed' or q.get('independent_review') is not True or q.get('n_generations')!=8 or q.get('GT_opened') is not False or q.get('scoring_performed') is not False or q.get('formal_proof_sha256')!=sha(proof)):raise ValueError('E4-B independent review missing')
 fs=d/'monitor_verify_b001/started.json';ft=d/'monitor_verify_b001/terminal.json';reviewer=r/'review_b001.py';rs=r/'monitor_b001/started.json';rt=r/'monitor_b001/terminal.json'
 if q.get('formal_started_sha256')!=sha(fs) or q.get('formal_terminal_sha256')!=sha(ft) or q.get('reviewer_sha256')!=sha(reviewer):raise ValueError('E4-B supervisor/source changed')
 terminal(fs,ft,d/'b_verify.py',proof);terminal(rs,rt,reviewer,review);files_match(p['files_sha256']);files_match(p['array_sha256'])
 return {'proof_sha256':sha(proof),'review_sha256':sha(review),'formal_started_sha256':sha(fs),'formal_terminal_sha256':sha(ft),'review_started_sha256':sha(rs),'review_terminal_sha256':sha(rt)}

def check_preflight(f,out):
 out=Path(out);l=read(out/'launch.json');cmd=l['command'];result=f.get('result',{})
 expected={'worker_sha256':HERE/'panel_worker.py','preflight_source_sha256':HERE/'preflight.py','launch_sha256':out/'launch.json','stdout_sha256':out/'stdout.log','stderr_sha256':out/'stderr.log'}
 prior=ROOT/'e4_panel_v2/panel_003/launch.json';old=read(prior)['command'];image='sha256:91d767ae00637d58384a6ca70cbfe7b8c9bfa9f8920bbe4ddaaa9af178100faf'
 def prefix(a):
  a=a[:]
  if a.count('--name')!=1:raise ValueError('wrong preflight container name')
  a[a.index('--name')+1]='<name>'
  while '--device' in a:
   i=a.index('--device');del a[i:i+2]
  a=['CUDA_VISIBLE_DEVICES=' if x=='CUDA_VISIBLE_DEVICES=0' else x for x in a]
  return a[:a.index(image)+1]
 wanted=[str(HERE/'panel_worker.py'),'--execute','--preflight-only','--e1-acceptance',str(ROOT/'e1/e1full001_completion.json'),'--e1-started',str(ROOT/'e1/monitorverifye1full001/started.json'),'--e1-terminal',str(ROOT/'e1/monitorverifye1full001/terminal.json'),'--out',str(out)]
 if (any(f.get(k)!=sha(v) for k,v in expected.items()) or f.get('status')!='passed' or f.get('returncode')!=0 or f.get('timed_out') is not False or f.get('container_running') is not False or f.get('container_exit_code')!=0 or f.get('container_pid')!=0 or f.get('interrupted') is not None or f.get('errors')!=[] or f.get('container_stopped_verified') is not True or f.get('client_exited_while_container_running') is not False or not f.get('container_id') or result.get('status')!='preflight_passed' or result.get('cuda_initialized') is not False or result.get('model_loaded') is not False or result.get('predictions') is not False or read(out/'stdout.log')!=result or
     l.get('upstream_launch_sha256')!=sha(prior) or prefix(old)!=prefix(cmd) or cmd[cmd.index(image)+1:]!=wanted or cmd[cmd.index('--name')+1]!='eadp-e2e3full-preflight-001' or any('/dev/nvidia' in str(x) for x in cmd) or 'CUDA_VISIBLE_DEVICES=' not in cmd):raise ValueError('actual full-panel container preflight failed')

def validate_probe(pv,smoke_proof,checker,reviewer):
 if (pv.get('status')!='passed' or pv.get('GT_opened') is not False or pv.get('scoring_performed') is not False or
     pv.get('formal_smoke_sha256')!=sha(smoke_proof) or pv.get('checker_sha256')!=sha(checker) or
     pv.get('reviewer_sha256')!=sha(reviewer) or pv.get('intervention_records_verified')!=42):raise ValueError('real smoke array probe invalid')

def execution_gate(*,preflight=False):
 """Require CPU/reviewer identity; full runs additionally require real preflight."""
 from full_imports import local
 bundle=HERE/'approval.json'
 if not bundle.is_file():raise ValueError('GPU locked: full-panel independent approval missing')
 b=read(bundle);required={'cpu_proof','cpu_started','cpu_terminal','review_proof','review_started','review_terminal'}
 if set(b)!=required:raise ValueError('GPU locked: incomplete approval bundle')
 c=read(b['cpu_proof']);r=read(b['review_proof']);stage='preflight' if preflight else 'final'
 expected_review=HERE/('independent_review_'+stage+'.json');reviewer=ROOT/'e1_independent_review'/('review_e2e3full_'+stage+'.py');mon=ROOT/'e1_independent_review'/('monitor_e2e3full_'+stage)
 if (Path(b['review_proof']).resolve()!=expected_review.resolve() or Path(b['review_started']).resolve()!=(mon/'started.json').resolve() or Path(b['review_terminal']).resolve()!=(mon/'terminal.json').resolve() or r.get('reviewer_sha256')!=sha(reviewer)):raise ValueError('wrong independent full-panel review')
 terminal(b['review_started'],b['review_terminal'],reviewer,b['review_proof']);terminal(b['cpu_started'],b['cpu_terminal'],HERE/'cpu_checks.py',Path(b['cpu_proof']).parent)
 if (c.get('status')!='passed' or c.get('returncode')!=0 or c.get('timed_out') is not False or r.get('status')!='passed' or r.get('independent_review') is not True or r.get('cpu_proof_sha256')!=sha(b['cpu_proof']) or r.get('cpu_started_sha256')!=sha(b['cpu_started']) or r.get('cpu_terminal_sha256')!=sha(b['cpu_terminal'])):raise ValueError('full-panel CPU/review evidence invalid')
 inventory=local('source_inventory').sources();before=c.get('source_sha256_before',{});after=c.get('source_sha256_after',{})
 if set(before)!=set(inventory) or before!=after or r.get('source_sha256')!=after or any(sha(p)!=d for p,d in after.items()):raise ValueError('full-panel reviewed source changed')
 cpudir=Path(b['cpu_proof']).parent
 if c.get('stdout_sha256')!=sha(cpudir/'stdout.log') or c.get('stderr_sha256')!=sha(cpudir/'stderr.log'):raise ValueError('CPU logs changed')
 probe=HERE/'probe_smoke001.json';probe_started=HERE/'monitor_smokeprobe001/started.json';probe_terminal=HERE/'monitor_smokeprobe001/terminal.json';pv=read(probe)
 if c.get('probe_sha256')!=sha(probe) or c.get('probe_terminal_sha256')!=sha(probe_terminal):raise ValueError('real smoke array probe changed')
 validate_probe(pv,ROOT/'e2_e3_smoke/run_001_verification.json',HERE/'validate_panel_record.py',HERE/'probe_smoke.py')
 terminal(probe_started,probe_terminal,HERE/'probe_smoke.py')
 smoke=smoke_gate();bidentity=b_gate()
 if r.get('smoke_review_sha256')!=smoke['review_sha256'] or r.get('b_review_sha256')!=bidentity['review_sha256']:raise ValueError('upstream identity proof mismatch')
 if not preflight:
  fp=HERE/'container_preflight_001.json';fs=HERE/'monitor_preflight001/started.json';ft=HERE/'monitor_preflight001/terminal.json'
  if r.get('preflight_sha256')!=sha(fp) or r.get('preflight_started_sha256')!=sha(fs) or r.get('preflight_terminal_sha256')!=sha(ft):raise ValueError('independent full preflight review absent')
  terminal(fs,ft,HERE/'preflight.py');check_preflight(read(fp),HERE/'preflight_001')
 return {'approval_sha256':sha(bundle),'cpu_sha256':sha(b['cpu_proof']),'review_sha256':sha(b['review_proof']),'smoke':smoke,'e4b':bidentity}
