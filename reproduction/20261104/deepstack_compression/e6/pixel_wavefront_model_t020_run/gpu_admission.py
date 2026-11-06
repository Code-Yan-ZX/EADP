"""Fail-closed GPU admission; no GPU interaction or model imports here."""
import json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review'
cached=sys.modules.get('source_inventory')
if cached is not None and (not getattr(cached,'__file__',None) or Path(cached.__file__).resolve()!=D/'source_inventory.py'):raise ValueError('foreign source inventory')
from source_inventory import hashes,sha

def read(p):return json.loads(Path(p).read_text())
def terminal(folder,script,out):
 s=read(folder/'started.json');t=read(folder/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (cwd/c[1]).resolve()!=script.resolve() or c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve():raise ValueError('review supervisor mismatch')

def require_gpu():
 plan=D/'execution_plan001.json';proof=D/'independent_gpu_prelaunch_review.json'
 if not plan.is_file() or not proof.is_file():raise RuntimeError('GPU locked: frozen execution plan and independent GPU prelaunch review missing')
 reviewer=I/'review_pixel_wavefront_t020_run_gpu_prelaunch.py';preflight_review=I/'pixel_wavefront_t020_run_preflight_result_003_review.json';preflight_code=I/'review_pixel_wavefront_t020_run_preflight_result_003.py';cpu=D/'cpu_check007/verification.json';p=read(plan);r=read(proof);sources=hashes()
 terminal(D/'monitor_cpu007',D/'cpu_validation.py',cpu.parent);terminal(I/'monitor_pixel_wavefront_t020_run_preflight_result_003',preflight_code,preflight_review);terminal(I/'monitor_pixel_wavefront_t020_run_gpu_prelaunch',reviewer,proof)
 pre=read(preflight_review);cp=read(cpu);preplan=read(D/'preflight_plan005.json')
 if (cp.get('status')!='passed' or cp.get('source_sha256_before')!=sources or cp.get('source_sha256_after')!=sources or cp.get('gpu_used') is not False or cp.get('model_loaded') is not False or cp.get('existing_DEV_predictions_opened') is not True or preplan.get('source_sha256')!=sources):raise ValueError('CPU/preflight source evidence changed')
 prefiles={'plan_sha256':D/'preflight_plan005.json','worker_preflight_sha256':D/'run_preflight002/worker_preflight.json','launch_sha256':D/'run_preflight002/launch.json','exit_sha256':D/'run_preflight002/exit.json','started_sha256':D/'monitor_preflight002/started.json','terminal_sha256':D/'monitor_preflight002/terminal.json'}
 if pre.get('independent_review') is not True or pre.get('GT_opened') is not False or pre.get('errors')!=[] or any(pre.get(k)!=sha(path) for k,path in prefiles.items()):raise ValueError('reviewed preflight evidence changed')
 if (p.get('stage')!='t020 three-anchor six-container identity run' or p.get('threshold')!=.2 or len(p.get('jobs',[]))!=6 or p.get('source_sha256')!=sources or p.get('GT_opened') is not False or p.get('scoring_performed') is not False or p.get('preflight_result_sha256')!=sha(preflight_review) or pre.get('status')!='passed' or pre.get('gpu_authorized') is not False or pre.get('reviewer_sha256')!=sha(preflight_code)):raise ValueError('identity plan or reviewed preflight')
 if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('scope')!='t020_identity_gpu_prelaunch' or r.get('GT_opened') is not False or r.get('scoring_performed') is not False or r.get('errors')!=[] or r.get('reviewer_sha256')!=sha(reviewer) or r.get('source_sha256')!=sources or r.get('plan_sha256')!=sha(plan) or r.get('preflight_result_sha256')!=sha(preflight_review) or r.get('cpu_sha256')!=sha(cpu) or r.get('cpu_started_sha256')!=sha(D/'monitor_cpu007/started.json') or r.get('cpu_terminal_sha256')!=sha(D/'monitor_cpu007/terminal.json')):raise ValueError('independent GPU prelaunch review mismatch')
 return {'plan_sha256':sha(plan),'review_sha256':sha(proof),'preflight_result_sha256':sha(preflight_review),'source_sha256':sources}
