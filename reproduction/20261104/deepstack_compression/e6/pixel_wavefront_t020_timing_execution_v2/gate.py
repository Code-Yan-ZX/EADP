"""Preflight-only gate; real timing and GPU remain unconditionally locked."""
import hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];C=B/'e6/pixel_wavefront_t020_model_timing_v2';I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def verified_prior(proof):
 expected=proof['source_sha256_after'];actual={name:sha(name) for name in expected}
 if actual!=expected:raise ValueError('upstream candidate source changed')
 return actual
def sources():
 proof=read(C/'candidate_cpu_proof001.json');items=verified_prior(proof)
 for p in list(D.glob('*.py'))+[C/'candidate_cpu_proof001.json',C/'monitor_candidate_cpu001/started.json',C/'monitor_candidate_cpu001/terminal.json']:items[str(p.resolve())]=sha(p)
 return items
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (cwd/cmd[1]).resolve()!=script.resolve() or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()))):raise ValueError('wrong supervisor')
def preflight():
 plan=D/'preflight_plan001.json';proof=I/'pixel_wavefront_t020_timing_execution_v2_preflight_review.json';reviewer=I/'review_pixel_wavefront_t020_timing_execution_v2_preflight.py';monitor=I/'monitor_pixel_wavefront_t020_timing_execution_v2_preflight'
 if not plan.is_file() or not proof.is_file():raise ValueError('noGPU preflight remains locked')
 p=read(plan);q=read(proof);cpu=read(D/'cpu_check001/verification.json');expected=sources()
 terminal(D/'monitor_cpu001',D/'cpu_check.py',D/'cpu_check001');terminal(monitor,reviewer,proof);terminal(C/'monitor_candidate_cpu001',C/'freeze_candidate.py')
 c=read(C/'candidate_cpu_proof001.json')
 if c.get('status')!='passed' or c.get('source_sha256_before')!=c.get('source_sha256_after') or c.get('GPU_authorized') is not False or c.get('GT_opened') is not False or c.get('existing_DEV_predictions_opened') is not True:raise ValueError('candidate CPU proof')
 if p.get('source_sha256')!=expected or cpu.get('status')!='passed' or cpu.get('source_sha256_before')!=expected or cpu.get('source_sha256_after')!=expected:raise ValueError('execution source changed')
 fields={'plan_sha256':plan,'cpu_sha256':D/'cpu_check001/verification.json','cpu_started_sha256':D/'monitor_cpu001/started.json','cpu_terminal_sha256':D/'monitor_cpu001/terminal.json','reviewer_sha256':reviewer}
 if (q.get('status')!='passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('gpu_authorized') is not False or q.get('GT_opened') is not False or q.get('existing_DEV_predictions_opened') is not True or q.get('source_sha256')!=expected or q.get('errors')!=[] or any(q.get(k)!=sha(v) for k,v in fields.items())):raise ValueError('preflight proof')
 return {'plan_sha256':sha(plan),'review_sha256':sha(proof),'source_sha256':expected}
def gpu():raise RuntimeError('GPU timing is not implemented or authorized')
