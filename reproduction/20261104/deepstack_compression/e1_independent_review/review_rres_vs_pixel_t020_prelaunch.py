"""Independent static and CPU prelaunch review; never parse scores or labels."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/rres_vs_pixel_t020'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def clean(mon,script):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd'])
 if cwd.resolve()!=B or len(s['command'])<2 or (cwd/s['command'][1]).resolve()!=script.resolve() or s['worker_pid']!=t['worker_pid'] or t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[]:raise ValueError('unclean supervisor '+str(mon))
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();dest=I/'rres_vs_pixel_t020_prelaunch_review.json'
 if a.out.resolve()!=dest or dest.exists() or (D/'result001.json').exists():raise ValueError('already run or incorrect output')
 plan=read(D/'plan001.json');cpu=read(D/'cpu_proof001.json');clean(D/'monitor_freeze001',D/'freeze.py');clean(D/'monitor_cpu001',D/'cpu_check.py')
 if plan['status']!='frozen' or plan['GT_opened'] is not False or plan['target_predictions_parsed'] is not False or plan['questions']!=384 or plan['replicates']!=10000 or plan['seed']!=20261104 or read(B/'initial_plan.json')['budget_end_unix']<=time.time() or any(sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('plan, deadline or source')
 if cpu['status']!='passed' or cpu['GT_opened'] is not False or cpu['scores_parsed'] is not False or cpu['plan_sha256']!=sha(D/'plan001.json') or cpu['source_sha256']!=plan['source_sha256'] or cpu['log_sha256']!=sha(D/'cpu001.log'):raise ValueError('CPU evidence')
 for n in ('pixel_t020_dev384_result','pixel_t020_dev384_scores','rres_k256_dev384_result','rres_k256_dev384_scores'):
  q=read(I/(n+'_review.json'));source=I/('review_'+n+'.py');clean(I/('monitor_'+n),source)
  if q['status']!='passed' or q['independent_review'] is not True or q['errors']!=[] or q['reviewer_sha256']!=sha(source):raise ValueError('prior independent review '+n)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');r=subprocess.run([sys.executable,'-m','unittest','-v','test_compare'],cwd=D,env=env,capture_output=True,text=True,timeout=180)
 if r.returncode:raise RuntimeError('independent CPU test '+r.stderr)
 if any(sha(k)!=v for k,v in plan['source_sha256'].items()) or (D/'result001.json').exists():raise ValueError('source drift or premature result')
 proof={'status':'passed','independent_review':True,'GT_opened':False,'scores_parsed':False,'existing_score_files_opened_for_hash':True,'errors':[],'plan_sha256':sha(D/'plan001.json'),'cpu_proof_sha256':sha(D/'cpu_proof001.json'),'cpu_started_sha256':sha(D/'monitor_cpu001/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu001/terminal.json'),'reviewer_sha256':sha(__file__),'cpu_test_stderr':r.stderr,'scope':'prelaunch code and synthetic tests only','limitations':['Posthoc comparison after observing both methods and tau=0; intervals uncorrected.','R-res six kwargs and EOS checked online but per-request observed kwargs and full effective config not archived.','Different visual resolutions and dynamic K; no equal-compute or speed claim.']}
 with dest.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
