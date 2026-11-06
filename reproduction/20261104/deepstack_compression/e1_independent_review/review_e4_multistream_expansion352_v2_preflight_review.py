"""Independent source/CPU review; authorizes v2 no-GPU container preflight only."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');D=B/'e4_multistream_expansion352_gpu_v2';P=B/'e4_multistream_expansion352';I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);c=s['command'];assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in c if Path(x).name==script.name]==[script.resolve()]
 if out is not None:assert c.count('--out')==1 and (cwd/c[c.index('--out')+1]).resolve()==out.resolve()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();assert not (D/'preflight001').exists() and not (D/'run001').exists();terminal(D/'monitor_cpu_check001',D/'cpu_check.py');c=read(D/'cpu_check001.json');src=c['source_sha256_after'];assert c['status']=='passed' and c['source_sha256_before']==src and len(src)==128 and all(sha(k)==v for k,v in src.items()) and all(c[k] is False for k in ('GT_opened','predictions_opened','feature_values_opened','score_values_opened','GPU_used','model_loaded'))
 prior=I/'e4_multistream_expansion352_preflight001_result_review.json';ps=I/'review_e4_multistream_expansion352_preflight001_result.py';terminal(I/'monitor_e4_multistream_expansion352_preflight001_result',ps,prior);q=read(prior);assert q['status']=='passed' and q['independent_review'] is True and (q['pilot_images'],q['full_inputs'],q['e4_reference_records'])==(9,384,9) and q['reviewer_sha256']==sha(ps) and q['GT_opened'] is False and q['GPU_authorized'] is False and q['errors']==[]
 old=read(P/'cpu_check001.json');assert old['status']=='passed' and all(src.get(k)==v for k,v in old['source_sha256_after'].items()) and sha(P/'selection001.json')==q['source_sha256'][str(P/'selection001.json')]
 plan=read(D/'execution_plan001.json');budget=read(B/'initial_plan.json');assert plan['images']==9 and plan['GT_opened'] is False and plan['scoring_performed'] is False and plan['automatic_retry'] is False and plan['budget_source_sha256']==sha(B/'initial_plan.json') and plan['budget_start_unix']==budget['budget_start_unix'] and plan['budget_end_unix']==budget['budget_end_unix'] and time.time()+plan['preflight_outer_seconds']<budget['budget_end_unix'] and plan['preflight_inner_seconds']+plan['grace_seconds']<=plan['preflight_outer_seconds'] and plan['gpu_inner_seconds']+plan['grace_seconds']<=plan['gpu_outer_seconds']
 fp=read(D/'formal_plan001.json');assert fp['images']==9 and fp['GT_opened'] is False and fp['scoring_performed'] is False and fp['automatic_retry'] is False and 'not TTFT' in fp['timing_boundary']
 tests=0
 for directory,pattern,expected in [(P,'test_norm_contract.py',4),(P,'test_pilot_engine.py',3),(P,'test_pilot_contract.py',3),(P,'test_selection.py',5),(D,'test_*.py',5)]:
  r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(directory),'-p',pattern,'-v'],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120);assert r.returncode==0 and f'Ran {expected} tests' in r.stderr,(pattern,r.stderr);tests+=expected
 assert tests==20 and all(sha(k)==v for k,v in src.items())
 # Exercise the real CLI with missing review; source admission must reject before creating Docker/output.
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');r=subprocess.run([sys.executable,str(D/'launch.py'),'--run-id','001','--preflight-only'],cwd=B,env=env,capture_output=True,text=True,timeout=120);assert r.returncode!=0 and 'independent review missing' in r.stderr and not (D/'preflight001').exists()
 r=subprocess.run([sys.executable,str(D/'worker.py'),'--out',str(D/'run001')],cwd=B,env=env,capture_output=True,text=True,timeout=120);assert r.returncode!=0 and 'independent review missing' in r.stderr and not (D/'run001').exists()
 out={'status':'passed','independent_review':True,'preflight_only':True,'GT_opened':False,'feature_values_opened':False,'score_values_opened':False,'GPU_used':False,'model_loaded':False,'errors':[],'source_sha256':src,'reviewer_sha256':sha(__file__),'cpu_sha256':sha(D/'cpu_check001.json'),'cpu_started_sha256':sha(D/'monitor_cpu_check001/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu_check001/terminal.json'),'selection_sha256':sha(P/'selection001.json'),'prior_container_review_sha256':sha(prior),'prior_container_review_started_sha256':sha(I/'monitor_e4_multistream_expansion352_preflight001_result/started.json'),'prior_container_review_terminal_sha256':sha(I/'monitor_e4_multistream_expansion352_preflight001_result/terminal.json'),'independent_cpu_tests':tests,'scope':'only authorize v2 actual no-GPU preflight; no GPU/model permission','limitations':['Nine actual author processor executions and four-shard hashes still must be performed and independently checked in the no-GPU container.','Future GPU gate requires separate review binding actual image, command, environment, full mounts and independent no-GPU result.','Nine-image identity/cost capture is not a TTFT benchmark and retained norm vectors do not independently reconstruct full high-dimensional tensors.']}
 with open(a.out,'x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
