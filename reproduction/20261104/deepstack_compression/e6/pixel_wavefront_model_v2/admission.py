"""Only the reviewed no-GPU processor/weight preflight can be admitted here."""
import json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review'
cached=sys.modules.get('source_inventory')
if cached is not None and (not getattr(cached,'__file__',None) or Path(cached.__file__).resolve()!=D/'source_inventory.py'):raise ValueError('foreign inventory')
from source_inventory import sha,source_hashes
def read(p):return json.loads(Path(p).read_text())
def terminal(directory,script,out):
 s=read(directory/'started.json');t=read(directory/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (cwd/cmd[1]).resolve()!=script or cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out:raise ValueError('supervisor evidence incomplete')
 return s
def require_preflight():
 plan=read(D/'preflight_plan001.json');cpu=D/'cpu_check001/verification.json';cs=D/'monitor_cpu001/started.json';ct=D/'monitor_cpu001/terminal.json';proof=D/'independent_preflight_review.json';reviewer=I/'review_pixel_wavefront_model_v2_preflight.py';monitor=I/'monitor_pixel_wavefront_model_v2_preflight'
 if not proof.is_file():raise ValueError('no-GPU preflight locked pending independent source review')
 sources=source_hashes();c=read(cpu);r=read(proof)
 terminal(D/'monitor_cpu001',D/'cpu_validation.py',cpu.parent);terminal(monitor,reviewer,proof)
 if plan.get('stage')!='three-anchor wavefront model no-GPU preflight only' or plan.get('source_sha256')!=sources or c.get('source_sha256_before')!=sources or c.get('source_sha256_after')!=sources or c.get('status')!='passed' or c.get('GT_opened') is not False or c.get('gpu_used') is not False or c.get('model_loaded') is not False:raise ValueError('source/CPU mismatch')
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('preflight_only') is not True or r.get('GT_opened') is not False or r.get('gpu_authorized') is not False or r.get('reviewer_sha256')!=sha(reviewer) or r.get('source_sha256')!=sources or r.get('plan_sha256')!=sha(D/'preflight_plan001.json') or r.get('cpu_sha256')!=sha(cpu) or r.get('cpu_started_sha256')!=sha(cs) or r.get('cpu_terminal_sha256')!=sha(ct) or r.get('errors')!=[]:raise ValueError('independent preflight review mismatch')
 return {'plan_sha256':sha(D/'preflight_plan001.json'),'review_sha256':sha(proof),'source_sha256':sources}
def require_gpu():raise ValueError('GPU locked: actual model identity controller and final review not implemented')
