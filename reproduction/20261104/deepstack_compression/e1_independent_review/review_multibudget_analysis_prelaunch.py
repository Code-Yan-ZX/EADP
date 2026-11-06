"""CPU-only independent review of frozen 32-image analysis code and watcher."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;I=B/'e1_independent_review';D=B/'e4_c_multibudget_analysis';EX=B/'e4_c_multibudget_expansion'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(folder,script):
 s=read(folder/'started.json');t=read(folder/'terminal.json');c=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and type(s['worker_pid']) is int and s['worker_pid']>0 and s['worker_pid']==t['worker_pid'] and (cwd/c[1]).resolve()==script.resolve()
def main(out):
 plan=read(D/'analysis_code_plan004.json');cpu=read(D/'cpu_proof004.json');term(D/'monitor_cpu004',D/'freeze.py');term(EX/'monitor_freeze_analysis001',EX/'freeze_analysis_plan.py')
 assert plan['status']=='frozen' and plan['GT_opened'] is False and plan['timing_results_opened'] is False and plan['GPU_authorized'] is False and cpu['status']=='passed' and cpu['returncode']==0 and cpu['source_sha256_before']==cpu['source_sha256_after'] and cpu['GT_opened'] is False and cpu['timing_results_opened'] is False and cpu['numpy_version']=='2.2.6'
 sources=dict(cpu['source_sha256_after']);sources[str((D/'cpu_proof004.json').resolve())]=sha(D/'cpu_proof004.json');assert plan['source_sha256']==sources
 for n,h in sources.items():assert sha(n)==h
 f=read(EX/'analysis_plan_freeze001.json');assert f['status']=='frozen' and f['analysis_plan_sha256']==sha(EX/'ANALYSIS_PLAN.md') and f['pilot_plan_sha256']==sha(EX/'pilot_plan001.json') and f['timing_results_opened'] is False
 spec=importlib.util.spec_from_file_location('_independent_multibudget_gate',D/'analysis_gate.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g);watch=g.watcher_prelaunch()
 names=['test_metrics','test_gate','test_report','test_watcher_contract'];r=subprocess.run([sys.executable,'-m','unittest','-v',*names],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=120);assert r.returncode==0 and 'Ran 13 tests' in r.stderr
 for n,h in sources.items():assert sha(n)==h
 fields={'analysis_code_plan_sha256':D/'analysis_code_plan004.json','analysis_plan_sha256':EX/'ANALYSIS_PLAN.md','analysis_freeze_sha256':EX/'analysis_plan_freeze001.json','analysis_freeze_started_sha256':EX/'monitor_freeze_analysis001/started.json','analysis_freeze_terminal_sha256':EX/'monitor_freeze_analysis001/terminal.json','cpu_proof_sha256':D/'cpu_proof004.json','cpu_started_sha256':D/'monitor_cpu004/started.json','cpu_terminal_sha256':D/'monitor_cpu004/terminal.json','reviewer_sha256':Path(__file__),'watcher_delta_plan_sha256':D/'WATCHER_DELTA_PLAN002.md'}
 proof={'status':'passed','independent_review':True,'scope':'multibudget_analysis_prelaunch','GT_opened':False,'timing_results_opened':False,'errors':[],'source_sha256':sources,**watch,**{k:sha(v) for k,v in fields.items()},'limitations':['Pre-run source review only; no timing values or prediction journal opened.','Image-group bootstrap is exploratory, not corrected for multiple comparisons; repeated timings are reduced within image.','Missing manual health windows are explicitly unknown; no observations deleted.','Unobserved background and watcher residual effects remain unknown.']}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==D/'independent_prelaunch_review.json' and not a.out.exists();main(a.out)
