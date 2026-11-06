"""No-GPU container preflight gate only; GPU remains unconditionally locked."""
import hashlib,importlib.util,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review';P=B/'e6/pixel_wavefront_t020_timing_execution_v2'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(name,p):
 p=Path(p).resolve();m=sys.modules.get(name)
 if m is not None:
  if not getattr(m,'__file__',None) or Path(m.__file__).resolve()!=p:raise ValueError('foreign '+name)
  return m
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m

def terminal(directory,script,out=None):
 s=read(directory/'started.json');t=read(directory/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if (len(cmd)<2 or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[]
  or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid')
  or (cwd/cmd[1]).resolve()!=script.resolve() or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()))):raise ValueError('supervisor proof')

UPSTREAM_GATE_SHA='bf4d373e6a23a703bf6d4c7ba54681d003affeeb2a254e8f52efe2d3fe118906'
def pin_upstream(path):
 path=Path(path).resolve()
 if sha(path)!=UPSTREAM_GATE_SHA:raise ValueError('upstream gate source mismatch')
 return path

def prior():
 pg=load('_pixel_t020_timing_execution_gate_prior',pin_upstream(P/'gate.py'));approval=pg.preflight()
 q=read(I/'pixel_wavefront_t020_timing_execution_v2_preflight_result_review.json')
 fields={'plan_sha256':P/'preflight_plan001.json','worker_preflight_sha256':P/'run_preflight001/worker_preflight.json','launch_sha256':P/'run_preflight001/launch.json','exit_sha256':P/'run_preflight001/exit.json','started_sha256':P/'monitor_preflight001/started.json','terminal_sha256':P/'monitor_preflight001/terminal.json','reviewer_sha256':I/'review_pixel_wavefront_t020_timing_execution_v2_preflight_result.py'}
 terminal(P/'monitor_preflight001',P/'preflight_launch.py',P/'run_preflight001')
 terminal(I/'monitor_pixel_wavefront_t020_timing_execution_v2_preflight_result',I/'review_pixel_wavefront_t020_timing_execution_v2_preflight_result.py',I/'pixel_wavefront_t020_timing_execution_v2_preflight_result_review.json')
 if (q.get('status')!='passed' or q.get('independent_review') is not True or q.get('gpu_authorized') is not False or q.get('GT_opened') is not False or q.get('existing_DEV_predictions_opened') is not True or q.get('source_sha256')!=pg.sources() or any(q.get(k)!=sha(v) for k,v in fields.items())):raise ValueError('prior noGPU independent proof')
 return approval,q

def sources():
 pg=load('_pixel_t020_timing_execution_gate_prior',pin_upstream(P/'gate.py'));items=pg.sources()
 extras=list(D.glob('*.py'))+[D/'PLAN_DRAFT.md',D/'cpu_proof003.json',D/'monitor_cpu003/started.json',D/'monitor_cpu003/terminal.json',I/'pixel_wavefront_t020_timing_execution_v2_preflight_result_review.json',I/'review_pixel_wavefront_t020_timing_execution_v2_preflight_result.py',I/'monitor_pixel_wavefront_t020_timing_execution_v2_preflight_result/started.json',I/'monitor_pixel_wavefront_t020_timing_execution_v2_preflight_result/terminal.json']
 for p in extras:items[str(p.resolve())]=sha(p)
 return items

def cpu():
 proof=read(D/'cpu_proof003.json');terminal(D/'monitor_cpu003',D/'cpu_check003.py');actual={p:sha(p) for p in proof['source_sha256_after']}
 if (proof.get('status')!='passed' or proof.get('GPU_authorized') is not False or proof.get('GPU_used') is not False or proof.get('GT_opened') is not False or proof.get('predictions_opened') is not False or proof.get('model_loaded') is not False or proof['source_sha256_before']!=actual or proof['source_sha256_after']!=actual):raise ValueError('CPU proof/source')

def preflight():
 plan=D/'preflight_plan001.json';proof=I/'pixel_wavefront_t020_timing_runtime_v4_preflight_review.json';reviewer=I/'review_pixel_wavefront_t020_timing_runtime_v4_preflight.py';monitor=I/'monitor_pixel_wavefront_t020_timing_runtime_v4_preflight'
 if not plan.exists() or not proof.exists():raise ValueError('noGPU preflight locked pending independent review')
 prior();cpu();p=read(plan);q=read(proof);actual=sources();terminal(monitor,reviewer,proof)
 fields={'plan_sha256':plan,'cpu_sha256':D/'cpu_proof003.json','cpu_started_sha256':D/'monitor_cpu003/started.json','cpu_terminal_sha256':D/'monitor_cpu003/terminal.json','reviewer_sha256':reviewer,'prior_review_sha256':I/'pixel_wavefront_t020_timing_execution_v2_preflight_result_review.json'}
 if (p.get('source_sha256')!=actual or q.get('source_sha256')!=actual or q.get('status')!='passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('gpu_authorized') is not False or q.get('GT_opened') is not False or q.get('existing_DEV_predictions_opened') is not True or q.get('errors')!=[] or any(q.get(k)!=sha(v) for k,v in fields.items())):raise ValueError('runtime preflight proof')
 return {'plan_sha256':sha(plan),'review_sha256':sha(proof)}

def gpu():raise RuntimeError('GPU locked: full runtime and admission are not ready')
