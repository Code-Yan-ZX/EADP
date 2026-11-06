"""Run-source import preflight only. GPU remains unconditionally locked."""
import hashlib,importlib.util,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review';P=B/'e6/pixel_wavefront_t020_timing_run_v2'
UPSTREAM_GATE_SHA='4e7fb864adaeb72e5b5528d700bef75a7c6e55039d9272999e924e1c4424c2bd'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(name,path,expected_sha):
 p=Path(path).resolve()
 if sha(p)!=expected_sha:raise ValueError('source mismatch before import')
 m=sys.modules.get(name)
 if m is not None:
  if not getattr(m,'__file__',None) or Path(m.__file__).resolve()!=p:raise ValueError('foreign module')
  return m
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def terminal(directory,script,out=None):
 s=read(directory/'started.json');t=read(directory/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if len(cmd)<2 or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or (cwd/cmd[1]).resolve()!=script.resolve() or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervisor evidence')
def prior():
 pg=load('_pixel_t020_run_v2_gate_prior',P/'preflight_gate.py',UPSTREAM_GATE_SHA);pg.preflight()
 proof=I/'pixel_wavefront_t020_timing_run_v2_preflight_result_review.json';reviewer=I/'review_pixel_wavefront_t020_timing_run_v2_preflight_result.py';mon=I/'monitor_pixel_wavefront_t020_timing_run_v2_preflight_result'
 if not proof.exists():raise ValueError('prior real noGPU independent review absent')
 q=read(proof);terminal(mon,reviewer,proof);terminal(P/'monitor_preflight001',P/'preflight_launch.py',P/'run_preflight001')
 fields={'plan_sha256':P/'preflight_plan001.json','worker_preflight_sha256':P/'run_preflight001/worker_preflight.json','launch_sha256':P/'run_preflight001/launch.json','identity_sha256':P/'run_preflight001/container_identity.json','exit_sha256':P/'run_preflight001/exit.json','started_sha256':P/'monitor_preflight001/started.json','terminal_sha256':P/'monitor_preflight001/terminal.json','prelaunch_review_sha256':I/'pixel_wavefront_t020_timing_run_v2_preflight_review.json','reviewer_sha256':reviewer}
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('gpu_authorized') is not False or q.get('GT_opened') is not False or q.get('existing_DEV_predictions_opened') is not True or q.get('errors')!=[] or q.get('source_sha256')!=pg.sources() or any(q.get(k)!=sha(v) for k,v in fields.items()):raise ValueError('prior real noGPU independent evidence')
 return pg,q

def sources():
 pg,_=prior();items=pg.sources()
 extras=[B/'e6/pixel_wavefront_t020_timing_controller_draft/job_contract.py',B.parents[2]/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_panel.py',B.parents[2]/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/diagnose.py']+list(D.glob('*.py'))+[D/'cpu_proof001.json',D/'monitor_cpu001/started.json',D/'monitor_cpu001/terminal.json',I/'pixel_wavefront_t020_timing_run_v2_preflight_result_review.json',I/'review_pixel_wavefront_t020_timing_run_v2_preflight_result.py',I/'monitor_pixel_wavefront_t020_timing_run_v2_preflight_result/started.json',I/'monitor_pixel_wavefront_t020_timing_run_v2_preflight_result/terminal.json']
 for p in extras:items[str(p.resolve())]=sha(p)
 return items

def cpu():
 p=read(D/'cpu_proof001.json');terminal(D/'monitor_cpu001',D/'cpu_check001.py');actual={k:sha(k) for k in p['source_sha256_after']}
 if p.get('status')!='passed' or p.get('GPU_used') is not False or p.get('GPU_authorized') is not False or p.get('GT_opened') is not False or p.get('predictions_opened') is not True or p['source_sha256_before']!=actual or p['source_sha256_after']!=actual:raise ValueError('CPU sources')

def preflight():
 plan=D/'preflight_plan001.json';proof=I/'pixel_wavefront_t020_timing_final_v8_preflight_review.json';reviewer=I/'review_pixel_wavefront_t020_timing_final_v8_preflight.py';mon=I/'monitor_pixel_wavefront_t020_timing_final_v8_preflight'
 if not plan.exists() or not proof.exists():raise ValueError('run-source noGPU preflight locked')
 cpu();p=read(plan);q=read(proof);actual=sources();terminal(mon,reviewer,proof)
 fields={'plan_sha256':plan,'cpu_sha256':D/'cpu_proof001.json','cpu_started_sha256':D/'monitor_cpu001/started.json','cpu_terminal_sha256':D/'monitor_cpu001/terminal.json','prior_review_sha256':I/'pixel_wavefront_t020_timing_run_v2_preflight_result_review.json','reviewer_sha256':reviewer}
 if p.get('source_sha256')!=actual or q.get('source_sha256')!=actual or q.get('status')!='passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('gpu_authorized') is not False or q.get('GT_opened') is not False or q.get('existing_DEV_predictions_opened') is not True or q.get('errors')!=[] or any(q.get(k)!=sha(v) for k,v in fields.items()):raise ValueError('independent preflight proof')
 return {'plan_sha256':sha(plan),'review_sha256':sha(proof)}
def gpu():raise RuntimeError('GPU locked: execution plan and admission not complete')
