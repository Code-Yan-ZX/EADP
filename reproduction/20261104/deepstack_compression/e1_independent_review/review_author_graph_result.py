"""Independently audit the five recorded failed author CUDA graph captures."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent;D=B/'e5/author_graph';O=D/'run_smoke001/observations'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for z in iter(lambda:f.read(1048576),b''):h.update(z)
 return h.hexdigest()
def req(x,msg):
 if not x:raise ValueError(msg)
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();plan=read(D/'plan001.json');s=read(D/'monitor_smoke001/started.json');t=read(D/'monitor_smoke001/terminal.json');launch=read(D/'run_smoke001/launch.json');ex=read(D/'run_smoke001/exit.json');done=read(O/'worker_complete.json');pre=read(D/'independent_prelaunch_review.json')
 req(plan['budgets']==[1,16,128,256,512] and launch['plan_sha256']==sha(D/'plan001.json'),'plan')
 for f,h in plan['sha256'].items():req(sha(f)==h,'source mismatch '+f)
 req(t['reason']=='worker_failed' and t['worker_returncode']==1 and not t['remaining_process_group'] and s['worker_pid']==t['worker_pid']==launch['controller']['pid']==ex['controller']['pid'],'supervisor exit')
 req(ex['returncode']==ex['container_exit_code']==1 and ex['container_stopped_verified'] is True and ex['interrupted'] is None and ex['timed_out'] is False and ex['gpu_pids_after']==[],'failure cleanup')
 obj=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],text=True,capture_output=True,check=True,timeout=15).stdout)[0];st=obj['State'];req(obj['Id']==ex['container_identity']['container_id'] and obj['Image']==ex['container_identity']['image'] and not st['Running'] and st['Pid']==0 and st['ExitCode']==1 and not st['OOMKilled'],'actual Docker state')
 req(pre['status']=='passed' and pre['independent_review'] is True and pre['plan_sha256']==sha(D/'plan001.json') and pre['reviewer_sha256']==sha(pre['reviewer_path']),'prelaunch')
 cases=done['cases'];req(done['status']=='attempts_recorded' and done['GT_opened'] is False and done['scoring_performed'] is False and [x['k'] for x in cases]==plan['budgets'] and all(x['returncode']==1 and x['timed_out'] is False for x in cases),'cases')
 progress=[json.loads(z) for z in (O/'progress.jsonl').read_text().splitlines()];expected=[{'event':'worker_started'}]
 for k in plan['budgets']:expected.extend([{'event':'k_started','k':k},{'event':'k_finished','k':k,'returncode':1,'timed_out':False}])
 req(progress==expected,'progress')
 files={};results=[]
 for k in plan['budgets']:
  path=O/f'k_{k:03d}.jsonl';events=[json.loads(z) for z in path.read_text().splitlines()];req(len(events)==1 and events[0]['event']=='failed' and events[0]['k']==k and events[0]['error_type']=='AcceleratorError','failure event')
  tr=events[0]['traceback'];req('operation not permitted when stream is capturing' in tr and 'selected_mask[batch_indices, best_idx] = True' in tr and 'previous error during capture' in events[0]['error'],'capture failure evidence')
  results.append({'k':k,'capture_completed':False,'replay_count':0,'timing_count':0,'failed_at_reported_author_line':186,'CUDA_error_reported':'operation not permitted when stream is capturing'});files[str(path)]=sha(path)
 for path in (O/'worker_complete.json',O/'progress.jsonl',D/'run_smoke001/launch.json',D/'run_smoke001/exit.json',D/'monitor_smoke001/started.json',D/'monitor_smoke001/terminal.json',D/'plan001.json',D/'independent_prelaunch_review.json'):files[str(path)]=sha(path)
 out={'status':'passed','independent_review':True,'scope':'audit of failed probe; not a successful GPU experiment','attempted_k':plan['budgets'],'results':results,'files_sha256':files,'reviewer_sha256':sha(__file__),'actual_container':{'id':obj['Id'],'running':st['Running'],'pid':st['Pid'],'exit_code':st['ExitCode'],'oom_killed':st['OOMKilled']},'GT_opened':False,'scoring_performed':False,'limitations':['No graph capture or replay completed; no equivalence, latency, or memory comparison can be claimed.','CUDA errors can be asynchronous; the reported author source line is evidence of the surfaced error, not a definitive causal localization.','A failure of this exact unmodified implementation, environment and capture setup does not establish that all author-like implementations or environments are ungraphable.'],'errors':[]}
 with a.out.open('x') as h:json.dump(out,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 print('failure evidence reviewed')
if __name__=='__main__':main()
