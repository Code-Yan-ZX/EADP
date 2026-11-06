import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');D=B/'e4_c_multibudget_environment192_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',required=True);a=a.parse_args();p=load(D/'plan001.json')
 paths=[D/'watch.py',D/'validate.py',D/'test_validate.py',D/'freeze.py',B/'initial_plan.json',B/'monitor/supervise.py',B/'e4_c_latency/timing_supervise.py']; src={str(x):sha(x) for x in paths}
 assert p['source_sha256']==src and p['status']=='prepared' and p['GPU_authorized'] is False and p['GT_opened'] is False and p['scoring_performed'] is False
 py='/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python'; ctl='/home/dell/miniconda3/bin/python'
 command=[ctl,str(B/'e4_c_latency/timing_supervise.py'),'--run-dir',str(D/'monitor_watch001'),'--timeout','36120','--snapshot-seconds','3600','--poll-seconds','5','--grace-seconds','30','--disk','/media/disk2','--',py,str(D/'watch.py'),'--output',str(D/'samples001.jsonl'),'--stop',str(D/'stop001'),'--interval','60','--duration','36000']
 assert p['command']==command and p['gpu_outer_timeout_seconds']==28800 and p['minimum_cleanup_seconds']>=1200 and p['watcher_duration_seconds']==36000 and p['watcher_outer_timeout_seconds']==36120 and p['interval_seconds']==60 and p['watcher_duration_seconds']>=p['gpu_outer_timeout_seconds']+p['minimum_cleanup_seconds']
 assert p['manual_path']==str(D/'manual_checks001.jsonl') and not any((D/n).exists() for n in ('samples001.jsonl','stop001','monitor_watch001'))
 s=load(D/'monitor_freeze001/started.json');t=load(D/'monitor_freeze001/terminal.json')
 assert Path(s['cwd']).resolve()==D and s['command'][-1]=='freeze.py' and t['worker_pid']==s['worker_pid'] and t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[]
 r=subprocess.run([sys.executable,'-m','unittest','-v','test_validate'],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=60)
 assert r.returncode==0 and 'Ran 3 tests' in r.stderr,(r.stdout,r.stderr)
 assert {str(x):sha(x) for x in paths}==src
 out={'status':'passed','independent_review':True,'GT_opened':False,'scoring_performed':False,'GPU_used':False,'GPU_authorized':False,'plan_sha256':sha(D/'plan001.json'),'reviewer_sha256':sha(__file__),'source_sha256':src,'freeze_started_sha256':sha(D/'monitor_freeze001/started.json'),'freeze_terminal_sha256':sha(D/'monitor_freeze001/terminal.json'),'independent_cpu_tests':3,'scope':'observer prelaunch only; does not start observer or authorize GPU','limitations':['At launch the live gate must ensure the observer still covers GPU timeout plus cleanup and time namespaces match.','Hourly supervisor health windows, manual checks and complete watcher lifecycle require actual post-run verification.','Unlogged background, second fsync, manual log and stop-marker writes have unknown overlap; do not exclude or select samples.','Process observations cannot establish full device or host isolation.'],'errors':[]}
 with open(a.out,'x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
