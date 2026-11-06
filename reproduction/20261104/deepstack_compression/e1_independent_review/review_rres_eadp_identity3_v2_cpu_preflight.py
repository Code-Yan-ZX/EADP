"""Independent identity3 CPU/no-device preflight audit; never launches Docker."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/rres_eadp_identity3_v2';T=B/'e6/rres_eadp_timing_pilot_v4';ROOT=B.parents[2]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def clean(mon):
 s=read(mon/'started.json');t=read(mon/'terminal.json')
 if Path(s['cwd']).resolve()!=B or s['worker_pid']!=t['worker_pid'] or t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[]:raise ValueError('supervision '+str(mon))
 return s
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();dest=I/'rres_eadp_identity3_v2_cpu_preflight_review.json'
 if a.out.resolve()!=dest or dest.exists() or (D/'preflight001').exists():raise ValueError('wrong destination or preflight started')
 plan=read(D/'preflight_plan001.json');sources=plan['source_sha256'];gate=load('_independent_identity3_gate',D/'gate.py')
 if plan['scope']!='identity3_cpu_preflight_only' or plan['GPU_authorized'] is not False or plan['GT_opened'] is not False or plan['existing_predictions_opened'] is not True or len(sources)!=314 or any(sha(k)!=v for k,v in sources.items()) or plan['prior']!=gate.prior():raise ValueError('plan/prior/sources')
 cpu=clean(D/'monitor_cpu008');imp=clean(D/'monitor_import_order001');freeze=clean(D/'monitor_preflight_freeze001')
 if cpu['command'][1:]!=['-m','unittest','discover','-s','e6/rres_eadp_identity3_v2','-p','test_*.py'] or (Path(imp['cwd'])/imp['command'][1]).resolve()!=D/'check_import_order.py' or (Path(freeze['cwd'])/freeze['command'][1]).resolve()!=D/'freeze_preflight.py' or 'timing runner bound' not in (D/'monitor_import_order001/stdout.log').read_text():raise ValueError('CPU execution command')
 sys.path.insert(0,str(T));sys.path.insert(0,str(D));schedule=read(T/'schedule001.json');worker=load('_independent_identity_worker',D/'worker.py')
 if tuple(worker.ANCHORS)!=tuple((task,str(next(x['qid'] for x in schedule['measured_inputs'] if x['task']==task))) for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')):raise ValueError('three anchor selection')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PAIRED_TIMING_CANDIDATE='rres_eadp_timing_pilot_v4',IDENTITY_TIMING_CANDIDATE='rres_eadp_identity3_v2');test=[]
 for directory in (D,B/'e6/rres_eadp_paired_timing_assessment'):
  r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(directory),'-p','test_*.py'],cwd=B,env=env,text=True,capture_output=True,timeout=180)
  if r.returncode:raise RuntimeError('independent test '+r.stderr)
  test.append(r.stderr)
 loader=load('_independent_identity_author',B/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py');runtime=loader.load_author(sources);author=read(ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json');cmd=runtime.container_command(author,[str(D/'preflight_entry.py'),'--out',str(D/'preflight001')],False,'independent-identity-preview')
 if '--device' in cmd or '--gpus' in cmd or cmd.count('--mount')!=12 or 'CUDA_VISIBLE_DEVICES=' not in cmd:raise ValueError('noGPU command')
 try:gate.gpu()
 except RuntimeError:pass
 else:raise ValueError('GPU unexpectedly open')
 if any(sha(k)!=v for k,v in sources.items()) or (D/'preflight001').exists():raise ValueError('source drift or preflight race')
 result={'status':'passed','independent_review':True,'preflight_only':True,'GPU_authorized':False,'GT_opened':False,'existing_predictions_opened':True,'errors':[],'plan_sha256':sha(D/'preflight_plan001.json'),'source_sha256':sources,'reviewer_sha256':sha(__file__),'cpu_tests':test,'scope':'only authorise actual same-image no-device diagnostic import preflight','limits':['No GPU, model, processor or DS identity was validated in this audit.','R-res first logits can be compared; accepted E4 logits are unavailable.','Diagnostic hooks alter execution and cannot be used for timing claims.','Prior identity3 review had an uncertain transient source-edit window; the original proof is retained and this separately frozen version binds this new review.']}
 with dest.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
