"""Independent static/CPU review of future random-smoke acceptance, no run read."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;A=B/'e6/random_smoke_accept';D=B/'e6/random_smoke';I=B/'e1_independent_review'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main(out):
 plan=read(A/'verification_plan001.json');s=read(A/'monitor_freeze001/started.json');t=read(A/'monitor_freeze001/terminal.json')
 assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and t['worker_pid']==s['worker_pid'] and (Path(s['cwd'])/s['command'][1]).resolve()==A/'freeze.py'
 assert plan['status']=='prepared' and plan['GT_opened'] is False and plan['scoring_performed'] is False and plan['GPU_authorized'] is False and plan['execution_plan_sha256']==sha(D/'execution_plan009.json') and len(plan['source_sha256'])==7
 for name,h in plan['source_sha256'].items():assert sha(name)==h
 r=subprocess.run([sys.executable,'-m','unittest','-v','test_verify','test_contract','test_inventory_contract'],cwd=A,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=60);assert r.returncode==0 and 'Ran 8 tests' in r.stderr
 for name,h in plan['source_sha256'].items():assert sha(name)==h
 p={'status':'passed','independent_review':True,'scope':'pre-run formal verifier source review only','GT_opened':False,'predictions_opened':False,'scoring_performed':False,'GPU_authorized':False,'plan_sha256':sha(A/'verification_plan001.json'),'plan_started_sha256':sha(A/'monitor_freeze001/started.json'),'plan_terminal_sha256':sha(A/'monitor_freeze001/terminal.json'),'source_sha256':plan['source_sha256'],'reviewer_sha256':sha(__file__),'errors':[],'limitations':['No run output has been read or validated.','GPU run and resulting independent review require separate supervision.','Full KV tensors and separate author greedy IDs are not archived; only runtime fingerprints and online checks.']}
 with out.open('x') as f:json.dump(p,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==I/'random_smoke_accept_prelaunch_review.json' and not a.out.exists();main(a.out)
