"""Independent pure-statistics audit; no real timing data or GPU."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/rres_eadp_timing_analysis_proposal_v2';A=B/'e6/rres_eadp_paired_timing_assessment'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def check_monitor(d):
 s=read(d/'started.json');t=read(d/'terminal.json')
 if Path(s['cwd']).resolve()!=B or s['worker_pid']!=t['worker_pid'] or t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[]:raise ValueError('supervision')
 return s
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=I/'rres_eadp_timing_statistics_v2_prelaunch_review.json'
 if a.out.resolve()!=out or out.exists():raise ValueError('unplanned out')
 plan=read(D/'statistics_plan001.json');s=check_monitor(D/'monitor_statistics_freeze001');check_monitor(A/'monitor_stats_v2_independent001');check_monitor(A/'monitor_observation_cpu002')
 if (plan['status']!='frozen' or plan['images']!=32 or plan['processes']!=12 or plan['warmups']!=60 or plan['measured']!=320 or plan['bootstrap_draws']!=10000 or plan['seed']!=20261106 or plan['bootstrap_rng']!='numpy.random.default_rng' or plan['percentile_method']!='linear' or plan['GT_opened'] is not False or plan['timing_results_opened'] is not False or plan['GPU_used'] is not False or any(sha(k)!=v for k,v in plan['source_sha256'].items())):raise ValueError('statistics plan or sources')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');test=[]
 for d,pat in ((D,'test_metrics.py'),(D,'test_windows.py'),(A,'test_analysis_v2_independent.py'),(A,'test_observation_windows.py')):
  x=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(d),'-p',pat],cwd=B,env=env,text=True,capture_output=True,timeout=120)
  if x.returncode:raise ValueError('test failure '+x.stderr)
  test.append(x.stderr)
 if any(sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('source drift')
 r={'status':'passed','independent_review':True,'scope':'pure statistics and normalized window CPU candidate only; not GPU or result-read admission','plan_sha256':sha(D/'statistics_plan001.json'),'source_sha256':plan['source_sha256'],'freeze_started_sha256':sha(D/'monitor_statistics_freeze001/started.json'),'freeze_terminal_sha256':sha(D/'monitor_statistics_freeze001/terminal.json'),'independent_test_sha256':sha(A/'test_analysis_v2_independent.py'),'window_test_sha256':sha(A/'test_observation_windows.py'),'reviewer_sha256':sha(__file__),'GT_opened':False,'timing_results_opened':False,'GPU_used':False,'errors':[],'tests':test,'limitations':['Read gate, actual observer and formal result interfaces require separate integration and independent review before real data.','Known observed overlap cannot establish absence of unobserved disturbances.','32 images are a small, exploratory DEV sample.']}
 with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
