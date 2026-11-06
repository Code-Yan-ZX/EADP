"""Independent review of execution-layer no-GPU preflight; never starts Docker."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/rres_eadp_identity3_execution';ROOT=B.parents[2]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def clean(m):
 s=read(m/'started.json');t=read(m/'terminal.json')
 if Path(s['cwd']).resolve()!=B or s['worker_pid']!=t['worker_pid'] or t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[]:raise ValueError('supervisor '+str(m))
 return s
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();dest=I/'rres_eadp_identity3_execution_cpu_review.json'
 if a.out.resolve()!=dest or dest.exists() or (D/'cpu_preflight001').exists():raise ValueError('unplanned output or preflight already started')
 plan=read(D/'cpu_plan001.json');sources=plan['source_sha256']
 if plan['scope']!='identity3_execution_cpu_only' or plan['GPU_authorized'] is not False or plan['GT_opened'] is not False or plan['existing_predictions_opened'] is not True or any(sha(k)!=v for k,v in sources.items()):raise ValueError('plan/sources')
 cpu=clean(D/'monitor_cpu006');freeze=clean(D/'monitor_cpu_freeze001');smoke=clean(D/'host_smoke002/monitor_run001');lifecycle=clean(B/'e6/rres_eadp_paired_timing_assessment/monitor_control_lifecycle001')
 if cpu['command'][1:]!=['-m','unittest','discover','-s','e6/rres_eadp_identity3_execution','-p','test_*.py'] or (Path(freeze['cwd'])/freeze['command'][1]).resolve()!=D/'freeze_cpu.py' or (Path(smoke['cwd'])/smoke['command'][1]).resolve()!=D/'host_entry_smoke.py' or smoke['timeout_seconds']!=3600 or smoke['stale_seconds']!=1200 or smoke['grace_seconds']!=120 or 'two subprocess/device interceptors active' not in (D/'host_smoke002/monitor_run001/stdout.log').read_text():raise ValueError('CPU/smoke commands')
 sys.path.insert(0,str(D));gate=load('_independent_execution_cpu_gate',D/'cpu_gate.py')
 if gate.plan()!=plan:raise ValueError('actual gate')
 try:gate.gpu()
 except RuntimeError:pass
 else:raise ValueError('GPU not locked')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PAIRED_TIMING_CANDIDATE='rres_eadp_timing_pilot_v4',IDENTITY_TIMING_CANDIDATE='rres_eadp_identity3_v2')
 t=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-p','test_*.py'],cwd=B,env=env,text=True,capture_output=True,timeout=180)
 if t.returncode:raise ValueError('independent CPU tests '+t.stderr)
 loader=load('_independent_execution_author',B/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py');runtime=loader.load_author(sources);author=read(ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json');cmd=runtime.container_command(author,[str(D/'cpu_entry.py'),'--out',str(D/'cpu_preflight001')],False,'independent-execution-preview')
 if '--device' in cmd or '--gpus' in cmd or cmd.count('--mount')!=12 or 'CUDA_VISIBLE_DEVICES=' not in cmd:raise ValueError('noGPU command')
 if any(sha(k)!=v for k,v in sources.items()) or (D/'cpu_preflight001').exists():raise ValueError('source/preflight race')
 result={'status':'passed','independent_review':True,'preflight_only':True,'GPU_authorized':False,'GT_opened':False,'existing_predictions_opened':True,'errors':[],'plan_sha256':sha(D/'cpu_plan001.json'),'source_sha256':sources,'reviewer_sha256':sha(__file__),'independent_cpu_tests':t.stderr,'limits':['Only same-image no-device execution imports are authorised; GPU and model remain locked.','Fake cleanup and Docker-identity tests do not prove a real GPU container run.','Diagnostic hooks add overhead; no latency claim.','Accepted EADP reference contains no comparable prefill-logits fingerprint.']}
 with dest.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
