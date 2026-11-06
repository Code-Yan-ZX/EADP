"""Independent review of timing execution no-GPU preflight only."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent;D=B/'e6/pixel_wavefront_t020_timing_execution_v2';C=B/'e6/pixel_wavefront_t020_model_timing_v2';ROOT=B.parents[2]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def terminal(folder,script,out=None):
 s,t=read(folder/'started.json'),read(folder/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (cwd/cmd[1]).resolve()!=script.resolve() or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervisor mismatch')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out.resolve()
 if out!=R/'pixel_wavefront_t020_timing_execution_v2_preflight_review.json':raise ValueError('output path')
 plan_path=D/'preflight_plan001.json';cpu_path=D/'cpu_check001/verification.json';p,c=read(plan_path),read(cpu_path)
 terminal(D/'monitor_cpu001',D/'cpu_check.py',cpu_path.parent);terminal(D/'monitor_freeze001',D/'freeze_preflight.py',plan_path)
 terminal(C/'monitor_candidate_cpu001',C/'freeze_candidate.py')
 actual={name:sha(name) for name in p['source_sha256']}
 if (len(actual)!=368 or actual!=p['source_sha256'] or c.get('source_sha256_before')!=actual or c.get('source_sha256_after')!=actual or c.get('status')!='passed' or c.get('returncode')!=0 or 'Ran 4 tests' not in c.get('test_stderr','') or c.get('GT_opened') is not False or c.get('existing_DEV_predictions_opened') is not True or c.get('gpu_used') is not False or c.get('model_loaded') is not False):raise ValueError('source/CPU evidence')
 if (p.get('stage')!='t020 natural timing no-GPU import preflight only' or p.get('GT_opened') is not False or p.get('existing_DEV_predictions_opened') is not True or p.get('gpu_authorized') is not False or p.get('cpu_sha256')!=sha(cpu_path) or p.get('cpu_started_sha256')!=sha(D/'monitor_cpu001/started.json') or p.get('cpu_terminal_sha256')!=sha(D/'monitor_cpu001/terminal.json') or p.get('budget_source_sha256')!=sha(B/'initial_plan.json')):raise ValueError('plan contract')
 candidate=read(C/'candidate_cpu_proof001.json')
 if candidate.get('status')!='passed' or candidate.get('GPU_authorized') is not False or candidate.get('GT_opened') is not False or candidate.get('source_sha256_before')!=candidate.get('source_sha256_after') or any(actual.get(name)!=digest for name,digest in candidate['source_sha256_after'].items()):raise ValueError('candidate identity')
 for name in ('gate.py','preflight_entry.py','preflight_launch.py','cpu_check.py','test_preflight.py'):
  if actual.get(str(D/name))!=sha(D/name):raise ValueError('unbound runtime module')
 spec=importlib.util.spec_from_file_location('_timing_execution_review_gate',D/'gate.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
 try:gate.gpu()
 except RuntimeError:pass
 else:raise ValueError('GPU not locked')
 source=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_panel.py';sp=importlib.util.spec_from_file_location('_timing_execution_review_runtime',source);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 cmd=m.container_command(read(source.parent/'run_plan.json'),[str(D/'preflight_entry.py'),'--out',str(D/'run_preflight001/worker_preflight.json')],False,'timing-review-cpu')
 if '--gpus' in cmd or '--device' in cmd or 'CUDA_VISIBLE_DEVICES=' not in cmd or cmd.count('--rm')!=1:raise ValueError('GPU exposure in preflight')
 proof={'status':'passed','independent_review':True,'preflight_only':True,'gpu_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'source_sha256':actual,'plan_sha256':sha(plan_path),'cpu_sha256':sha(cpu_path),'cpu_started_sha256':sha(D/'monitor_cpu001/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu001/terminal.json'),'reviewer_sha256':sha(__file__),'errors':[],'limitations':['Only actual no-GPU import preflight admitted; no real timing or NVML implementation is authorized.','Review author wrote the wavefront candidate, not the timing execution chain.','Existing DEV prediction records were read only for accepted identity.']}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
