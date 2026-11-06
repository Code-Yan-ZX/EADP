"""Independent CPU review, authorizing only the retained no-GPU preflight."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e4_c_multibudget_fixed32';R=B/'e1_independent_review'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def get(path):return json.loads(Path(path).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 if args.out.resolve()!=(R/'e4c_multibudget_fixed32_worker_preflight_review.json').resolve() or args.out.exists():raise ValueError('wrong/existing proof')
 plan=get(D/'execution_plan005.json');cpu=get(D/'execution_cpu005.json');start=get(D/'monitor_execution_cpu005/started.json');end=get(D/'monitor_execution_cpu005/terminal.json')
 assert plan['stage']=='4-budget fixed32 24-generation diagnostic' and plan['budgets']==[128,256,512,1024] and plan['GT_opened'] is False and plan['scoring_performed'] is False and plan['GPU_authorized_by_plan'] is False
 assert (cpu['status'],cpu['test_returncode'],cpu['GT_opened'],cpu['predictions_opened'],cpu['model_loaded'],cpu['GPU_authorized'])==('passed',0,False,True,False,False)
 assert 'hash only' in cpu['prediction_scope'] and 'Ran 19 tests' in cpu['test_output'] and cpu['plan_sha256']==sha(D/'execution_plan005.json') and cpu['source_sha256']==plan['source_sha256']
 assert end['reason']=='completed' and end['worker_returncode']==0 and end['remaining_process_group']==[] and start['worker_pid']==end['worker_pid'] and Path(start['cwd']).resolve()==D
 assert str(D/'freeze_execution.py') in [str((Path(start['cwd'])/x).resolve()) if not x.startswith('-') else x for x in start['command']]
 source=plan['source_sha256'];assert len(source)==164
 assert all(sha(path)==h for path,h in source.items())
 run=subprocess.run([sys.executable,'-m','unittest','-v','test_fixed','test_real_arm_schema','test_core','test_execution'],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True,timeout=120)
 assert run.returncode==0 and 'Ran 19 tests' in run.stderr and '\nOK' in run.stderr
 assert all(sha(path)==h for path,h in source.items())
 proof={'status':'passed','independent_review':True,'scope':'fixed32_cpu_preflight_only','source_sha256':source,'plan_sha256':sha(D/'execution_plan005.json'),'execution_cpu_sha256':sha(D/'execution_cpu005.json'),'execution_cpu_started_sha256':sha(D/'monitor_execution_cpu005/started.json'),'execution_cpu_terminal_sha256':sha(D/'monitor_execution_cpu005/terminal.json'),'reviewer_sha256':sha(__file__),'independent_tests':19,'GT_opened':False,'predictions_opened':True,'prediction_scope':'previously accepted DEV prediction files opened for source SHA only; not parsed','scoring_performed':False,'GPU_authorized':False,'limitations':['authorizes only retained no-GPU worker preflight; real model and GPU paths remain unverified','fixed32 has 32 output tokens and 31 decode forwards, not forced33/32 decode','KV contents will be fingerprinted, not archived for full reconstruction'],'errors':[]}
 with args.out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('independent CPU-only preflight review passed')
if __name__=='__main__':main()
