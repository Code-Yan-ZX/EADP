"""Independent preflight-only review of Pixel NVML nine-process dependency chain."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');D=B/'e6/pixel_wavefront_t020_nvml_execution_preflight_v2';E=B/'e6/pixel_wavefront_t020_nvml_execution_candidate_v8';I=B/'e1_independent_review';R=B/'e6/pixel_wavefront_t020_nvml_runtime_candidate_v2'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(mon,script):
 s=read(mon/'started.json');t=read(mon/'terminal.json');c=s['command'];cwd=Path(s['cwd']);assert len(c)>=2 and (cwd/c[1]).resolve()==script.resolve() and s['worker_pid']==t['worker_pid'] and t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[];return s
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();assert not (D/'run_preflight001').exists()
 terminal(D/'monitor_freeze001',D/'freeze.py');terminal(D/'monitor_cpu002',D/'cpu_check.py');p=read(D/'plan001.json');cpu=read(D/'cpu_proof002.json');src=p['source_sha256']
 assert p['scope']=='pixel nine-reference noGPU imports only' and p['GPU_authorized'] is False and p['GT_opened'] is False and p['existing_DEV_predictions_opened'] is True and len(src)==600 and all(sha(k)==v for k,v in src.items()) and cpu['status']=='passed' and cpu['source_sha256_before']==cpu['source_sha256_after'] and all(src[k]==v for k,v in cpu['source_sha256_after'].items()) and cpu['GPU_used'] is False and cpu['predictions_opened'] is False
 prior=read(E/'cpu_preflight_plan001.json');assert prior['status']=='frozen' and prior['GPU_authorized'] is False and all(src[k]==v for k,v in prior['source_sha256'].items())
 for folder,proof,mon in ((E,'cpu_proof001.json','monitor_cpu001'),(R,'cpu_proof001.json','monitor_cpu001'),(B/'e6/pixel_wavefront_t020_nvml_execution_accept','cpu_proof006.json','monitor_cpu006'),(B/'e6/pixel_wavefront_t020_nvml_execution_integration_v3','cpu_proof001.json','monitor_cpu001')):
  terminal(folder/mon,folder/'cpu_check.py');c=read(folder/proof);assert c['status']=='passed' and c['GPU_used'] is False and c['GT_opened'] is False and c['source_sha256_before']==c['source_sha256_after'] and all(sha(k)==v for k,v in c['source_sha256_after'].items())
 for name,script in (('worker','worker.py'),('controller','controller_entry.py'),('sampler','sampler_entry.py')):
  mon=E/('monitor_'+name+'_smoke001');s=terminal(mon,E/script);v=read(mon/'stdout.log');assert Path(s['cwd']).resolve()==B and s['command'][-1]=='--no-gpu-dependency-smoke' and v['status']=='passed' and v['model_loaded'] is False
  if name=='worker':assert v['references']==9 and v['cuda_initialized'] is False
  else:assert v['gpu_queried'] is False
  for f in ('started.json','terminal.json','stdout.log'):assert src[str((mon/f).resolve())]==sha(mon/f)
 # Independently rerun the frozen local gate tests and demonstrate real entry refuses before a directory or Docker can exist.
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');x=subprocess.run([sys.executable,'-m','unittest','-v','test_gate'],cwd=D,env=env,capture_output=True,text=True,timeout=120);assert x.returncode==0 and 'Ran 3 tests' in x.stderr
 x=subprocess.run([sys.executable,str(D/'launch.py'),'--out',str(D/'run_preflight001')],cwd=B,env=env,capture_output=True,text=True,timeout=60);assert x.returncode!=0 and not (D/'run_preflight001').exists() and 'pixel_t020_nvml_execution_preflight_v2_review.json' in x.stderr
 # Production GPU route itself must remain locked; this invokes no hardware query.
 spec=importlib.util.spec_from_file_location('_independent_pixel_nvml_gpu_gate',E/'gate.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
 try:g.admit_gpu()
 except RuntimeError as ex:assert 'GPU locked' in str(ex)
 else:raise AssertionError('GPU candidate unexpectedly admitted')
 assert all(sha(k)==v for k,v in src.items()) and not (D/'run_preflight001').exists()
 out={'status':'passed','independent_review':True,'preflight_only':True,'GPU_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'errors':[],'plan_sha256':sha(D/'plan001.json'),'reviewer_sha256':sha(__file__),'source_sha256':src,'cpu_sha256':sha(D/'cpu_proof002.json'),'cpu_started_sha256':sha(D/'monitor_cpu002/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu002/terminal.json'),'scope':'allow only retained same-image noGPU dependency container','limitations':['No model or CUDA initialized and no actual NVML sampling was exercised; real sampler/controller/worker handshake needs separate GPU review.','Existing DEV mixed outputs are parsed for accepted input identity.','Current formal plan and GPU plan remain separately gated; this proof is not GPU authority.']}
 with open(a.out,'x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
