import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression')
D=B/'e6/pixel_wavefront_t020_timing_observer_v2'; S=B/'e6/pixel_wavefront_t020_timing_exec_supplement_v5'; R=B/'e6/pixel_wavefront_t020_timing_final_v9'; A=B/'e6/pixel_wavefront_t020_timing_accept_v5'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())
def terminal(d,script):
 t=load(d/'terminal.json'); s=load(d/'started.json')
 assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[]
 assert any(str(x).endswith(script) for x in s['command']), s
 return {'started_sha256':sha(d/'started.json'),'terminal_sha256':sha(d/'terminal.json')}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args(); assert not (R/'run_001').exists()
 plan=load(D/'plan001.json'); sp=load(S/'plan001.json'); gp=R/'gpu_plan001.json'; fp=A/'verification_plan001.json'
 expected={str(D/n):sha(D/n) for n in ('observe.py','test_observe.py','freeze.py')}
 assert plan['source_sha256']==expected and plan['gpu_plan_sha256']==sha(gp) and plan['formal_plan_sha256']==sha(fp)
 assert plan['interval_seconds']==60 and plan['duration_seconds']==30000 and plan['output']==str(D/'observations001.jsonl') and plan['stop']==str(D/'stop001')
 assert plan['GT_opened'] is False and plan['GPU_used'] is False and not (D/'observations001.jsonl').exists() and not (D/'stop001').exists()
 assert sp['gpu_plan_sha256']==sha(gp) and sp['formal_plan_sha256']==sha(fp) and load(fp)['gpu_plan_sha256']==sha(gp)
 assert sp['freeze_source_sha256']==sha(S/'freeze.py') and sp['snapshot_source_sha256']==sha(A/'snapshot_entry.py') and sp['verifier_source_sha256']==sha(A/'verify_entry.py')
 assert sp['cwd']==str(B) and sp['automatic_retry'] is False and sp['GPU_used'] is False and sp['GT_opened'] is False and sp['predictions_opened'] is False
 py='/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python'
 def cmd(run,script,out): return [py,str(B/'monitor/supervise.py'),'--run-dir',str(run),'--timeout','1800','--snapshot-seconds','3600','--poll-seconds','5','--grace-seconds','30','--',py,str(script),'--out',str(out)]
 assert sp['snapshot_command']==cmd(A/'monitor_snapshot001',A/'snapshot_entry.py',R/'run_001_sources')
 assert sp['verify_command']==cmd(A/'monitor_verify001',A/'verify_entry.py',A/'verification001.json')
 formal=load(fp); assert formal['source_sha256'][str(A/'snapshot.py')]==sha(A/'snapshot.py') and formal['source_sha256'][str(A/'verify.py')]==sha(A/'verify.py')
 terminals={'observer_freeze':terminal(D/'monitor_freeze001','freeze.py'),'supplement_freeze':terminal(S/'monitor_freeze001','freeze.py')}
 r=subprocess.run([sys.executable,'-m','unittest','-v','test_observe'],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=30)
 assert r.returncode==0 and 'Ran 2 tests' in r.stderr,(r.stdout,r.stderr)
 assert {str(D/n):sha(D/n) for n in ('observe.py','test_observe.py','freeze.py')}==expected
 proof={'status':'passed','independent_review':True,'scope':'supplementary observer and frozen post-run commands only; no GPU authorization','GT_opened':False,'GPU_used':False,'model_loaded':False,'samples_used_for_filtering':False,'observer_plan_sha256':sha(D/'plan001.json'),'supplement_plan_sha256':sha(S/'plan001.json'),'gpu_plan_sha256':sha(gp),'formal_plan_sha256':sha(fp),'source_sha256':expected,'supplement_sources':{str(p):sha(p) for p in (S/'freeze.py',A/'snapshot_entry.py',A/'verify_entry.py',A/'snapshot.py',A/'verify.py')},'freeze_supervision':terminals,'independent_cpu_tests':2,'limitations':['Queries and first fsync have recorded boundaries; second fsync and other background actions have unknown windows.','Point observations cannot prove machine isolation or identify all interference; retain all samples.','Observer can disturb timing and query failures or missing coverage cannot be interpreted as no interference.','Supplement is not automatically enforced by the GPU gate; actual run, observer coverage and post-run proofs require separate verification.'],'reviewer_sha256':sha(__file__),'errors':[]}
 with open(a.out,'x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
