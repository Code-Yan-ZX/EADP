"""Independent review gate: no-GPU import preflight only."""
import json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review'
# Keep this inventory under a private name: the accepted upstream gate has
# its own different source_inventory module and uses the public name.
import importlib.util
_name='_pixel_t020_run_v2_inventory';_path=D/'source_inventory.py';_inv=sys.modules.get(_name)
if _inv is not None and (not getattr(_inv,'__file__',None) or Path(_inv.__file__).resolve()!=_path):raise ValueError('foreign private inventory')
if _inv is None:
 _spec=importlib.util.spec_from_file_location(_name,_path);_inv=importlib.util.module_from_spec(_spec);sys.modules[_name]=_inv
 try:_spec.loader.exec_module(_inv)
 except BaseException:del sys.modules[_name];raise
hashes,sha=_inv.hashes,_inv.sha

def read(p):return json.loads(Path(p).read_text())
def terminal(folder,script,out):
 s=read(folder/'started.json');t=read(folder/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (cwd/c[1]).resolve()!=script.resolve() or c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve():raise ValueError('supervisor mismatch')

def require_preflight():
 plan=D/'preflight_plan001.json';cpu=D/'cpu_check001/verification.json';proof=D/'independent_preflight_review.json';reviewer=I/'review_pixel_wavefront_t020_run_v2_preflight.py'
 if not proof.is_file():raise ValueError('independent no-GPU preflight review missing')
 p=read(plan);c=read(cpu);r=read(proof);sources=hashes();terminal(D/'monitor_cpu001',D/'cpu_validation.py',cpu.parent);terminal(I/'monitor_pixel_wavefront_t020_run_v2_preflight',reviewer,proof)
 if p.get('gpu_authorized') is not False or p.get('source_sha256')!=sources or c.get('status')!='passed' or c.get('source_sha256_before')!=sources or c.get('source_sha256_after')!=sources or c.get('existing_DEV_predictions_opened') is not True:raise ValueError('plan/CPU proof')
 if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('preflight_only') is not True or r.get('gpu_authorized') is not False or r.get('GT_opened') is not False or r.get('existing_DEV_predictions_opened') is not True or r.get('errors')!=[] or r.get('reviewer_sha256')!=sha(reviewer) or r.get('source_sha256')!=sources or r.get('plan_sha256')!=sha(plan) or r.get('cpu_sha256')!=sha(cpu) or r.get('cpu_started_sha256')!=sha(D/'monitor_cpu001/started.json') or r.get('cpu_terminal_sha256')!=sha(D/'monitor_cpu001/terminal.json')):raise ValueError('review proof')
 return {'plan_sha256':sha(plan),'review_sha256':sha(proof)}
