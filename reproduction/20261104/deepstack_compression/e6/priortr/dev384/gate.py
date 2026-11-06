"""PriorTR DEV384 admission checks; no GPU or labels."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];PREP=HERE.parent/'preparation';REV=BASE/'e1_independent_review'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def terminal(start,end,script,out=None):
 s=read(start);t=read(end);cmd=s.get('command',[]);cwd=Path(s.get('cwd','/'))
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or [(cwd/x).resolve() for x in cmd if Path(x).name==Path(script).name]!=[Path(script).resolve()]:raise ValueError('supervisor/command failed')
 if out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=Path(out).resolve()):raise ValueError('supervised output differs')
def dev32_evidence():
 """Bind the full accepted DEV32 chain using hashes, without parsing predictions."""
 d=HERE.parent/'dev32';run=d/'run_001';formal=d/'run_001_verification_v2.json';review=d/'run_001_independent_review.json'
 fs=d/'monitor_verify_002/started.json';ft=d/'monitor_verify_002/terminal.json';rs=d/'monitor_review_run001/started.json';rt=d/'monitor_review_run001/terminal.json'
 terminal(fs,ft,d/'panel_verify_v2.py',formal);terminal(rs,rt,d/'review_panel.py',review)
 terminal(d/'monitor_001/started.json',d/'monitor_001/terminal.json',d/'launch.py')
 f=read(formal);r=read(review)
 if (f.get('status')!='passed' or f.get('questions')!=32 or f.get('generations')!=96 or f.get('GT_opened') is not False or f.get('scoring_performed') is not False or f.get('verifier_sha256')!=sha(d/'panel_verify_v2.py') or
     r.get('status')!='passed' or r.get('independent_review') is not True or r.get('questions')!=32 or r.get('generations')!=96 or r.get('GT_opened') is not False or r.get('scoring_performed') is not False or r.get('formal_sha256')!=sha(formal) or r.get('formal_started_sha256')!=sha(fs) or r.get('formal_terminal_sha256')!=sha(ft) or r.get('reviewer_sha256')!=sha(d/'review_panel.py')):
  raise ValueError('PriorTR DEV32 formal or independent acceptance invalid')
 files=f.get('files_sha256',{});arrays=f.get('array_sha256',{})
 required=[d/'monitor_001/started.json',d/'monitor_001/terminal.json']+[run/x for x in ('launch.json','exit.json','frozen.json','worker_complete.json','data/worker_complete.json','data/records.jsonl','data/generated.jsonl')]
 if len(arrays)!=9 or any(files.get(str(p.resolve()))!=sha(p) for p in required):raise ValueError('PriorTR DEV32 artifacts missing or changed')
 if any(sha(p)!=digest for p,digest in {**files,**arrays}.items()):raise ValueError('PriorTR DEV32 accepted artifact changed')
 frozen=read(run/'frozen.json');sources=frozen.get('source_sha256',{})
 if len(sources)!=107 or any(sha(p)!=digest for p,digest in sources.items()):raise ValueError('PriorTR DEV32 execution source changed')
 return {'formal':sha(formal),'review':sha(review),'run_frozen':sha(run/'frozen.json'),'formal_terminal':sha(ft),'review_terminal':sha(rt)}

def upstream():
 sample=HERE/'sample_001.json';sr=HERE/'sample_001_independent_review.json';p=read(sample);r=read(sr)
 if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('questions_verified')!=384 or r.get('sample_sha256')!=sha(sample) or r.get('reviewer_sha256')!=sha(HERE/'review_sample.py') or r.get('freeze_source_sha256')!=sha(HERE/'freeze_panel.py') or r.get('GT_opened') is not False or r.get('predictions_opened') is not False or r.get('scoring_performed') is not False or p.get('GT_opened') is not False or p.get('prediction_opened') is not False or p.get('physical_generations')!=1152):raise ValueError('sample review missing')
 terminal(HERE/'monitor_freeze001/started.json',HERE/'monitor_freeze001/terminal.json',HERE/'freeze_panel.py')
 terminal(HERE/'monitor_review001/started.json',HERE/'monitor_review001/terminal.json',HERE/'review_sample.py')
 if r.get('freeze_started_sha256')!=sha(HERE/'monitor_freeze001/started.json') or r.get('freeze_terminal_sha256')!=sha(HERE/'monitor_freeze001/terminal.json'):raise ValueError('sample freeze terminal changed')
 proof=BASE/'e1/e1full001_completion.json';e1=read(proof);review=REV/'e1full001_review.json';er=read(review)
 if sha(proof)!=p['E1_proof_sha256'] or e1.get('status')!='passed' or e1.get('GT_opened') is not False or e1.get('scoring_performed') is not False or er.get('status')!='passed' or er.get('n_verified')!=384 or er.get('completion_proof_sha256')!=sha(proof) or er.get('reviewer_sha256')!=sha(REV/'review_full.py') or er.get('completion_started_sha256')!=sha(BASE/'e1/monitorverifye1full001/started.json') or er.get('completion_terminal_sha256')!=sha(BASE/'e1/monitorverifye1full001/terminal.json'):raise ValueError('E1 evidence missing')
 for key in ('inputs','panel','journal'):
  z=e1['files'][key]
  if sha(z['path'])!=z['sha256']:raise ValueError('E1 input/journal changed')
 terminal(BASE/'e1/monitorverifye1full001/started.json',BASE/'e1/monitorverifye1full001/terminal.json',BASE/'e1_analysis/completion.py',proof)
 terminal(REV/'monitor_e1full001/started.json',REV/'monitor_e1full001/terminal.json',REV/'review_full.py',review)
 prior=PREP/'smoke_001_verification.json';q=read(prior);pr=REV/'priortr_run003_review.json';w=read(pr)
 if (q.get('status')!='passed' or q.get('verifier_sha256')!=sha(PREP/'smoke_verify.py') or q.get('GT_opened') is not False or q.get('scoring_performed') is not False or w.get('status')!='passed' or w.get('independent_review') is not True or w.get('formal_proof_sha256')!=sha(prior) or w.get('reviewer_sha256')!=sha(REV/'review_priortr_run003.py') or w.get('GT_opened') is not False or w.get('scoring_performed') is not False):raise ValueError('PriorTR identity smoke absent')
 terminal(PREP/'monitor_verify_smoke001/started.json',PREP/'monitor_verify_smoke001/terminal.json',PREP/'smoke_verify.py',prior)
 terminal(REV/'monitor_priortr_run003/started.json',REV/'monitor_priortr_run003/terminal.json',REV/'review_priortr_run003.py',pr)
 if w.get('formal_started_sha256')!=sha(PREP/'monitor_verify_smoke001/started.json') or w.get('formal_terminal_sha256')!=sha(PREP/'monitor_verify_smoke001/terminal.json'):raise ValueError('PriorTR verifier terminal changed')
 for path,digest in q['files_sha256'].items():
  pth=Path(path) if Path(path).is_absolute() else PREP/path
  if sha(pth)!=digest:raise ValueError('PriorTR smoke artifact changed')
 return {'sample':sha(sample),'sample_review':sha(sr),'e1':sha(proof),'e1_review':sha(review),'priortr_smoke':sha(prior),'priortr_review':sha(pr),'dev32':dev32_evidence()}

def check_preflight(f,out):
 from launch import configure
 out=Path(out);launch=read(out/'launch.json');cmd=launch['command'];prior=PREP/'smoke_001/launch.json';want=read(prior)['command'][:];name='eadp-priortr384-preflight-003'
 if want.count('--name')!=1:raise ValueError('prior container name')
 want[want.index('--name')+1]=name
 for device in ('/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'):
  i=want.index(device);del want[i-1:i+1]
 want[want.index('CUDA_VISIBLE_DEVICES=0')]='CUDA_VISIBLE_DEVICES='
 i=want.index(str(PREP/'smoke_worker.py'));want=want[:i]+[str(HERE/'panel_worker.py'),'--execute','--preflight-only','--out',str(out)];want=configure(want,out/'runtime_cache')
 try:parsed=json.loads((out/'stdout.log').read_text().strip().splitlines()[-1])
 except Exception as exc:raise ValueError('preflight stdout') from exc
 result=f.get('result',{})
 expected={'worker_sha256':HERE/'panel_worker.py','preflight_source_sha256':HERE/'preflight.py','launch_sha256':out/'launch.json','stdout_sha256':out/'stdout.log','stderr_sha256':out/'stderr.log'}
 if (cmd!=want or launch.get('prior_launch_sha256')!=sha(prior) or any(f.get(k)!=sha(v) for k,v in expected.items()) or f.get('status')!='passed' or f.get('returncode')!=0 or f.get('timed_out') is not False or f.get('interrupted') is not None or f.get('errors')!=[] or f.get('client_exited_while_container_running') is not False or f.get('container_stopped_verified') is not True or f.get('container_running') is not False or f.get('container_pid')!=0 or f.get('container_exit_code')!=0 or not f.get('container_id') or parsed!=result or result.get('status')!='preflight_passed' or result.get('cuda_initialized') is not False or result.get('model_loaded') is not False or result.get('predictions_parsed') is not False or result.get('labels_opened') is not False or result.get('samples')!=384 or any('/dev/nvidia' in str(x) for x in cmd)):
  raise ValueError('actual no-GPU worker preflight invalid')

def execution_gate(preflight=False):
 """Bind CPU proof, separate source review and, for GPU, actual preflight."""
 import source_inventory
 bundle=HERE/'approval.json'
 if not bundle.is_file():raise ValueError('DEV384 execution locked: approval bundle absent')
 b=read(bundle);required={'cpu_proof','cpu_started','cpu_terminal','review_proof','review_started','review_terminal'}
 if set(b)!=required:raise ValueError('incomplete DEV384 bundle')
 stage='preflight' if preflight else 'final';reviewer=HERE/('review_'+stage+'.py');proof=HERE/('independent_review_'+stage+'.json');monitor=HERE/('monitor_review_'+stage)
 if Path(b['review_proof']).resolve()!=proof.resolve() or Path(b['review_started']).resolve()!=(monitor/'started.json').resolve() or Path(b['review_terminal']).resolve()!=(monitor/'terminal.json').resolve():raise ValueError('wrong reviewer proof')
 terminal(b['review_started'],b['review_terminal'],reviewer,proof);terminal(b['cpu_started'],b['cpu_terminal'],HERE/'cpu_checks.py',Path(b['cpu_proof']).parent)
 c=read(b['cpu_proof']);r=read(proof);before=c.get('source_sha256_before',{});after=c.get('source_sha256_after',{})
 if (c.get('status')!='passed' or c.get('returncode')!=0 or c.get('timed_out') is not False or c.get('GT_opened') is not False or c.get('scoring_performed') is not False or c.get('gpu_used') is not False or r.get('status')!='passed' or r.get('independent_review') is not True or r.get('reviewer_sha256')!=sha(reviewer) or r.get('cpu_proof_sha256')!=sha(b['cpu_proof']) or r.get('cpu_started_sha256')!=sha(b['cpu_started']) or r.get('cpu_terminal_sha256')!=sha(b['cpu_terminal']) or before!=after or set(after)!=set(source_inventory.sources()) or r.get('source_sha256')!=after or any(sha(p)!=h for p,h in after.items())):
  raise ValueError('DEV384 CPU proof/review invalid')
 cpudir=Path(b['cpu_proof']).parent
 if c.get('stdout_sha256')!=sha(cpudir/'stdout.log') or c.get('stderr_sha256')!=sha(cpudir/'stderr.log') or c.get('upstream')!=upstream():raise ValueError('CPU logs or upstream changed')
 if not preflight:
  fp=HERE/'container_preflight_003.json';fs=HERE/'monitor_preflight003/started.json';ft=HERE/'monitor_preflight003/terminal.json'
  if r.get('preflight_sha256')!=sha(fp) or r.get('preflight_started_sha256')!=sha(fs) or r.get('preflight_terminal_sha256')!=sha(ft):raise ValueError('independent preflight review absent')
  terminal(fs,ft,HERE/'preflight.py');check_preflight(read(fp),HERE/'preflight_003')
 return {'approval_sha256':sha(bundle),'cpu_sha256':sha(b['cpu_proof']),'review_sha256':sha(proof),'upstream':c['upstream']}
