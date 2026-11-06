"""Prelaunch audit using synthetic arrays only; never load real tensor archives."""
import argparse,hashlib,importlib.util,json,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e4_multistream_local_minimax_v2';OLD=B/'e4_multistream_local_minimax';E=B/'e4_c_microbench_full'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def clean(mon,script):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd'])
 if s['worker_pid']!=t['worker_pid'] or t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or [(cwd/x).resolve() for x in s['command'] if Path(x).name==script.name]!=[script.resolve()]:raise ValueError('supervision '+str(mon))
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();dest=I/'local_minimax_v2_prelaunch_review.json'
 if a.out.resolve()!=dest or dest.exists() or (D/'result001.json').exists() or (D/'rows001.jsonl').exists():raise ValueError('output or already run')
 plan=read(D/'plan001.json');inv=read(D/'inventory001.json');cpu=read(D/'cpu_proof001.json');clean(D/'monitor_cpu001',D/'cpu_check.py');clean(D/'monitor_freeze001',D/'freeze.py');clean(OLD/'monitor_inventory001',OLD/'inventory.py');clean(E/'monitor_verify_full001',E/'full_verify.py');clean(I/'monitor_component_full001',I/'review_component_full001.py')
 if plan['status']!='frozen' or plan['questions']!=32 or plan['GT_opened'] is not False or plan['real_tensors_deserialized'] is not False or cpu['status']!='passed' or cpu['real_tensors_deserialized'] is not False or inv['status']!='passed' or inv['images']!=32 or inv['real_tensors_deserialized'] is not False or sha(D/'inventory001.json')!=sha(OLD/'inventory001.json') or any(sha(k)!=v for k,v in plan['source_sha256'].items()) or any(sha(k)!=v for k,v in cpu['source_sha256'].items()) or read(B/'initial_plan.json')['budget_end_unix']<=time.time():raise ValueError('source/CPU/plan')
 formal=read(E/'full_001_verification.json');prior=read(I/'component_full001_review.json');schedule=read(B/'e4_c_microbench/schedule001.json')['input_identities']
 if formal['status']!='passed' or formal['images']!=32 or prior['status']!='passed' or prior['independent_review'] is not True or prior['images']!=32 or prior['reviewer_sha256']!=sha(I/'review_component_full001.py') or prior['formal_sha256']!=sha(E/'full_001_verification.json') or prior['array_sha256']!=formal['array_sha256'] or len(schedule)!=32:raise ValueError('accepted historical archives')
 archives=inv['archives'];expected=formal['array_sha256']
 if len(archives)!=32 or {x['path']:x['sha256'] for x in archives}!=expected or plan['archives_sha256']!=expected:raise ValueError('archive inventory')
 for i,x in enumerate(archives):
  if x['index']!=i or any(x[k]!=schedule[i][k] for k in ('task','qid','group_id','image_sha256','message_sha256')) or Path(x['path']).stat().st_size!=x['size_bytes']:raise ValueError('archive identity')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-p','test_*.py'],cwd=B,env=env,text=True,capture_output=True,timeout=180)
 if r.returncode:raise RuntimeError('CPU tests '+r.stderr)
 sp=importlib.util.spec_from_file_location('_independent_minimax_v2',D/'core.py');core=importlib.util.module_from_spec(sp);sp.loader.exec_module(core)
 # Independently brute force all four candidates of one nontrivial 2x2 block.
 arrays=[np.zeros((32,32,2),dtype=np.float64) for _ in range(4)];points=[(0.,0.),(1.,0.),(4.,0.),(4.,2.)]
 for k,x in enumerate(arrays):
  for i,(y,z) in enumerate(points):x[i//2,i%2,:]=[(k+1)*y,z+k]
 got=core.select(arrays,7);scales=[math.sqrt(sum(float(np.dot(v,v)) for v in x.reshape(1024,-1))/1024)+1e-6 for x in arrays];candidates=[0,1,32,33];radius=[]
 for j in candidates:
  worst=0.
  for x,scale in zip(arrays,scales):
   v=x.reshape(1024,-1)
   for i in candidates:worst=max(worst,math.sqrt(sum(float(y*y) for y in v[i]-v[j]))/scale)
  radius.append(worst)
 expected_id=candidates[min(range(4),key=lambda i:(radius[i],candidates[i]))]
 if got['block_ids']['minimax'][0]!=expected_id or got['ids']['minimax']!=sorted(got['block_ids']['minimax']) or len(got['ids']['minimax'])!=256 or not math.isclose(got['radii']['minimax'][0],min(radius),rel_tol=1e-12):raise ValueError('independent brute force')
 if any(sha(k)!=v for k,v in plan['source_sha256'].items()) or (D/'result001.json').exists() or (D/'rows001.jsonl').exists():raise ValueError('source drift or premature analysis')
 proof={'status':'passed','independent_review':True,'real_tensors_deserialized':False,'GT_opened':False,'errors':[],'plan_sha256':sha(D/'plan001.json'),'inventory_sha256':sha(D/'inventory001.json'),'cpu_proof_sha256':sha(D/'cpu_proof001.json'),'reviewer_sha256':sha(__file__),'cpu_test_stderr':r.stderr,'limitations':['Reused earlier supervised archive byte hashes without re-reading 2.83GB or deserializing real tensors.','Posthoc feature-only diagnostic; no quality, timing, causal or novelty claim.','Actual tensor execution requires separate resource scheduling and complete result review.']}
 with dest.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
