"""Independent review of frozen random-full score v3; no target predictions/GT."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;I=B/'e1_independent_review';D=B/'e6/random_full_analysis_v3'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script):
 s=read(d/'started.json');t=read(d/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()]:raise ValueError('supervisor evidence')
 return s
def main(out):
 terminal(D/'monitor_cpu001',D/'cpu_check.py');terminal(D/'monitor_freeze_score001',D/'freeze_score.py')
 plan=read(D/'score_plan001.json');cpu=read(D/'cpu_proof001.json');source=plan['source_sha256']
 if (plan.get('status')!='frozen' or plan.get('GT_opened') is not False or plan.get('full_panel_predictions_opened') is not False or cpu.get('status')!='passed' or cpu.get('tests')!=11 or cpu.get('integration_returncode')!=0 or cpu.get('GT_opened') is not False or cpu.get('full_panel_predictions_opened') is not False or cpu.get('source_sha256_before')!=cpu.get('source_sha256_after') or cpu.get('source_sha256_after')!=source or plan.get('cpu_proof_sha256')!=sha(D/'cpu_proof001.json')):raise ValueError('plan/CPU proof')
 for path,digest in source.items():
  p=Path(path).resolve()
  if '/run_001/' in str(p) or p.suffix in ('.tsv','.xlsx') or sha(p)!=digest:raise ValueError('unsafe/changed source '+str(p))
 # Execute frozen tests in separate interpreters; synthetic inputs only.
 logs=[]
 for args in ([sys.executable,'-m','unittest','-v','test_paired','test_score'],[sys.executable,'test_main_synthetic.py','-v']):
  r=subprocess.run(args,cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=120)
  if r.returncode!=0:raise ValueError('synthetic tests failed '+r.stderr[-1500:])
  logs.append(r.stderr)
 if 'Ran 11 tests' not in logs[0] or 'Ran 3 tests' not in logs[1] or any(sha(path)!=digest for path,digest in source.items()):raise ValueError('test count/source changed')
 result={'status':'passed','independent_review':True,'scope':'random_full_v2_score_prelaunch','errors':[],'GT_opened':False,'target_predictions_opened':False,'score_plan_sha256':sha(D/'score_plan001.json'),'cpu_proof_sha256':sha(D/'cpu_proof001.json'),'cpu_started_sha256':sha(D/'monitor_cpu001/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu001/terminal.json'),'freeze_started_sha256':sha(D/'monitor_freeze_score001/started.json'),'freeze_terminal_sha256':sha(D/'monitor_freeze_score001/terminal.json'),'reviewer_sha256':sha(__file__),'source_sha256':source,'tests':11,'integration_tests':3,'limitations':['Only scorer prelaunch reviewed; no random result or labels read.','task×seed XLSX inference isolated; paired image-group bootstrap conditional on three fixed seeds.','Full formal and independent result proofs and actual exits are still mandatory before scoring.']}
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
 if a.out.resolve()!=I/'random_full_v2_score_prelaunch_review.json' or a.out.exists():raise ValueError('fixed output required')
 main(a.out)
