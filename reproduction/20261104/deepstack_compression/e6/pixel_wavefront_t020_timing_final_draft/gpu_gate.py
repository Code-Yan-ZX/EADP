"""Prospective final GPU gate; fails closed until every future proof exists.
This draft is not frozen and has no execution authority.
"""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(directory,script,out=None):
 s=read(directory/'started.json');t=read(directory/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if (len(cmd)<2 or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or (cwd/cmd[1]).resolve()!=script.resolve() or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()))):raise ValueError('invalid supervisor evidence')
def independent_preflight(proof,reviewer,monitor,expected_fields):
 q=read(proof);terminal(monitor,reviewer,proof)
 if (q.get('status')!='passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('gpu_authorized') is not False or q.get('GT_opened') is not False or q.get('existing_DEV_predictions_opened') is not True or q.get('errors')!=[] or q.get('reviewer_sha256')!=sha(reviewer) or any(q.get(k)!=sha(v) for k,v in expected_fields.items())):raise ValueError('invalid prerequisite noGPU review')
 return q

def prerequisites():
 prior=B/'e6/pixel_wavefront_t020_timing_runtime_v4';name='pixel_wavefront_t020_timing_runtime_v4_preflight_result'
 runtime=independent_preflight(I/(name+'_review.json'),I/('review_'+name+'002.py'),I/('monitor_'+name+'002'),{'plan_sha256':prior/'preflight_plan001.json','worker_preflight_sha256':prior/'run_preflight001/worker_preflight.json','launch_sha256':prior/'run_preflight001/launch.json','identity_sha256':prior/'run_preflight001/container_identity.json','exit_sha256':prior/'run_preflight001/exit.json','started_sha256':prior/'monitor_preflight001/started.json','terminal_sha256':prior/'monitor_preflight001/terminal.json','prelaunch_review_sha256':I/'pixel_wavefront_t020_timing_runtime_v4_preflight_review.json'})
 terminal(prior/'monitor_preflight001',prior/'preflight_launch.py',prior/'run_preflight001')
 run=B/'e6/pixel_wavefront_t020_timing_run_v2';name='pixel_wavefront_t020_timing_run_v2_preflight_result'
 future=independent_preflight(I/(name+'_review.json'),I/('review_'+name+'.py'),I/('monitor_'+name),{'plan_sha256':run/'preflight_plan001.json','worker_preflight_sha256':run/'run_preflight001/worker_preflight.json','launch_sha256':run/'run_preflight001/launch.json','identity_sha256':run/'run_preflight001/container_identity.json','exit_sha256':run/'run_preflight001/exit.json','started_sha256':run/'monitor_preflight001/started.json','terminal_sha256':run/'monitor_preflight001/terminal.json','prelaunch_review_sha256':I/'pixel_wavefront_t020_timing_run_v2_preflight_review.json'})
 terminal(run/'monitor_preflight001',run/'preflight_launch.py',run/'run_preflight001')
 return runtime,future

def admit():
 """No GPU query. Must run in both host and worker before other imports."""
 plan=D/'gpu_plan001.json';proof=I/'pixel_wavefront_t020_timing_final_prelaunch_review.json';reviewer=I/'review_pixel_wavefront_t020_timing_final_prelaunch.py';monitor=I/'monitor_pixel_wavefront_t020_timing_final_prelaunch'
 if not plan.is_file() or not proof.is_file():raise ValueError('final GPU admission locked')
 p=read(plan);q=read(proof)
 prerequisites()
 expected=p.get('source_sha256')
 if not isinstance(expected,dict) or not expected or {k:sha(k) for k in expected}!=expected:raise ValueError('frozen sources changed')
 terminal(monitor,reviewer,proof)
 design=B/'e6/pixel_wavefront_t020_model_timing_v2/design001.json'
 if p.get('design_sha256')!=sha(design) or p.get('jobs')!=read(design).get('jobs'):raise ValueError('frozen exact 24-job schedule mismatch')
 fields={'plan_sha256':plan,'reviewer_sha256':reviewer,'runtime_review_sha256':I/'pixel_wavefront_t020_timing_runtime_v4_preflight_result_review.json','run_preflight_review_sha256':I/'pixel_wavefront_t020_timing_run_v2_preflight_result_review.json','formal_plan_sha256':B/'e6/pixel_wavefront_t020_timing_accept_final/verification_plan001.json'}
 if (p.get('scope')!='pixel_t020_isolated_process_ttft_24' or len(p.get('jobs',[]))!=24
     or q.get('status')!='passed' or q.get('independent_review') is not True or q.get('scope')!='pixel_t020_timing_gpu_prelaunch' or q.get('errors')!=[]
     or q.get('GT_opened') is not False or q.get('scoring_performed') is not False or q.get('source_sha256')!=expected
     or any(q.get(k)!=sha(v) for k,v in fields.items())):raise ValueError('GPU admission proof mismatch')
 return {'plan_sha256':sha(plan),'jobs':p['jobs'],'design_sha256':p['design_sha256'],'worker_sha256':expected[str((D/'worker.py').resolve())]}
if __name__=='__main__':raise SystemExit('unfrozen draft; GPU locked')
