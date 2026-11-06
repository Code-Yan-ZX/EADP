"""Independent review gate: no-GPU import preflight only."""
import json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review'
cached=sys.modules.get('source_inventory')
if cached is not None and (not getattr(cached,'__file__',None) or Path(cached.__file__).resolve()!=D/'source_inventory.py'):raise ValueError('foreign source inventory')
from source_inventory import hashes,sha

def read(p):return json.loads(Path(p).read_text())
def terminal(folder,script,out):
 s=read(folder/'started.json');t=read(folder/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (cwd/c[1]).resolve()!=script.resolve() or c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve():raise ValueError('supervisor mismatch')

def require_preflight():
 plan=D/'preflight_plan005.json';cpu=D/'cpu_check007/verification.json';proof=D/'independent_preflight_review_003.json';reviewer=I/'review_pixel_wavefront_t020_run_preflight_003.py'
 if not proof.is_file():raise ValueError('independent no-GPU preflight review missing')
 p=read(plan);c=read(cpu);r=read(proof);sources=hashes();terminal(D/'monitor_cpu007',D/'cpu_validation.py',cpu.parent);terminal(I/'monitor_pixel_wavefront_t020_run_preflight_003',reviewer,proof)
 if p.get('gpu_authorized') is not False or p.get('source_sha256')!=sources or c.get('status')!='passed' or c.get('source_sha256_before')!=sources or c.get('source_sha256_after')!=sources or c.get('existing_DEV_predictions_opened') is not True:raise ValueError('plan/CPU proof')
 if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('preflight_only') is not True or r.get('gpu_authorized') is not False or r.get('GT_opened') is not False or r.get('existing_DEV_predictions_opened') is not True or r.get('errors')!=[] or r.get('reviewer_sha256')!=sha(reviewer) or r.get('source_sha256')!=sources or r.get('plan_sha256')!=sha(plan) or r.get('cpu_sha256')!=sha(cpu) or r.get('cpu_started_sha256')!=sha(D/'monitor_cpu007/started.json') or r.get('cpu_terminal_sha256')!=sha(D/'monitor_cpu007/terminal.json')):raise ValueError('review proof')
 return {'plan_sha256':sha(plan),'review_sha256':sha(proof)}
