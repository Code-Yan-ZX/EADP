"""Analysis source gate and future double result acceptance; no GPU."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review';R=B/'e6/pixel_wavefront_t020_timing_final_v4';A=B/'e6/pixel_wavefront_t020_timing_accept_final'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def sources():return {str(p.resolve()):sha(p) for p in D.glob('*.py')}
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if len(c)<2 or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or (cwd/c[1]).resolve()!=script.resolve() or (out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervisor')
def prelaunch():
 plan=D/'analysis_plan001.json';proof=I/'pixel_wavefront_t020_timing_analysis_v2_prelaunch_review.json';reviewer=I/'review_pixel_wavefront_t020_timing_analysis_v2_prelaunch.py';mon=I/'monitor_pixel_wavefront_t020_timing_analysis_v2_prelaunch'
 if not plan.exists() or not proof.exists():raise ValueError('analysis source gate locked')
 p=read(plan);q=read(proof);cpu=read(D/'cpu_proof001.json');terminal(D/'monitor_cpu001',D/'cpu_check.py');terminal(mon,reviewer,proof);actual=sources()
 fields={'plan_sha256':plan,'cpu_sha256':D/'cpu_proof001.json','cpu_started_sha256':D/'monitor_cpu001/started.json','cpu_terminal_sha256':D/'monitor_cpu001/terminal.json','reviewer_sha256':reviewer,'gpu_plan_sha256':R/'gpu_plan001.json','formal_plan_sha256':A/'verification_plan001.json'}
 if p.get('source_sha256')!=actual or cpu.get('source_sha256_before')!=actual or cpu.get('source_sha256_after')!=actual or cpu.get('status')!='passed' or q.get('status')!='passed' or q.get('independent_review') is not True or q.get('timing_results_opened') is not False or q.get('GT_opened') is not False or q.get('errors')!=[] or q.get('source_sha256')!=actual or any(q.get(k)!=sha(v) for k,v in fields.items()):raise ValueError('analysis source proof')
 return {'source_review_sha256':sha(proof),'plan_sha256':sha(plan)}
def result_admit():
 formal=A/'verification001.json';review=I/'pixel_wavefront_t020_timing_result_review.json';reviewer=I/'review_pixel_wavefront_t020_timing_result.py';mon=I/'monitor_pixel_wavefront_t020_timing_result'
 if not formal.exists() or not review.exists():raise ValueError('formal and independent results required')
 f=read(formal);q=read(review);terminal(A/'monitor_verify001',A/'verify_entry.py',formal);terminal(mon,reviewer,review);terminal(R/'monitor_run001',R/'launch.py')
 fields={'formal_sha256':formal,'formal_started_sha256':A/'monitor_verify001/started.json','formal_terminal_sha256':A/'monitor_verify001/terminal.json','run_started_sha256':R/'monitor_run001/started.json','run_terminal_sha256':R/'monitor_run001/terminal.json','reviewer_sha256':reviewer}
 if f.get('status')!='passed' or f.get('jobs')!=24 or f.get('GT_opened') is not False or q.get('status')!='passed' or q.get('independent_review') is not True or q.get('jobs')!=24 or q.get('GT_opened') is not False or q.get('errors')!=[] or any(q.get(k)!=sha(v) for k,v in fields.items()):raise ValueError('result proof')
 return {'formal_sha256':sha(formal),'independent_review_sha256':sha(review)}
