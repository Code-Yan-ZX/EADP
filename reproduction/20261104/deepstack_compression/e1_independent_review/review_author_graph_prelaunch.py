"""Independent author-only CUDA graph feasibility prelaunch review."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e5/author_graph';R=B/'e1_independent_review';M=R/'monitor_author_graph_prelaunch'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def terminal(mon):
 s=read(mon/'started.json');t=read(mon/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'];return s
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert a.out.resolve()==D/'independent_prelaunch_review.json';plan=read(D/'plan001.json');assert sha(D/'plan001.json')=='3caf0b12070fc0b4168fc342fba99295b636d683a968730e4adfe593d94afb8c' and plan['budgets']==[1,16,128,256,512] and plan['child_timeout_seconds']==300 and plan['stale_seconds']==420 and plan['inner_timeout_seconds']==1800 and plan['outer_timeout_seconds']==1920
assert all(sha(p)==h for p,h in plan['sha256'].items());terminal(D/'monitor_cpu_test004');assert 'Ran 7 tests' in (D/'monitor_cpu_test004/stderr.log').read_text();terminal(D/'monitor_cpu_preflight003');s=terminal(D/'monitor_preflight001');assert s['timeout_seconds']==720 and s['grace_seconds']==90 and (Path(s['cwd'])/s['command'][1]).resolve()==D/'launch.py' and s['command'][-2:]==['--stage','preflight']
pref=read(D/'preflight001.json');launch=read(D/'run_preflight001/launch.json');ex=read(D/'run_preflight001/exit.json');assert pref['status']=='passed' and pref['source_plan_sha256']==sha(D/'plan001.json') and pref['cuda_initialized'] is False and pref['model_loaded'] is False and pref['GT_opened'] is False and pref['scoring_performed'] is False and pref['case_sha256']==plan['sha256'][str(B/'e5/selector_real32/run_001/image_017/case_512.pt')] and s['worker_pid']==launch['controller']['pid']==ex['controller']['pid'] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] is True and ex['client_exited_while_container_running'] is False
obj=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];st=obj['State'];cmd=launch['command'];assert obj['Id']==ex['container_identity']['container_id']==ex['last_inspect']['id'] and obj['Image']==ex['container_identity']['image'] and not st['Running'] and st['Pid']==0 and st['ExitCode']==0 and st['OOMKilled'] is False and obj['HostConfig']['NetworkMode']=='none' and not obj['HostConfig']['Devices'] and not obj['HostConfig']['DeviceRequests']
assert obj['Config']['User']==cmd[cmd.index('--user')+1] and obj['Config']['WorkingDir']==cmd[cmd.index('--workdir')+1] and obj['Config']['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]] and obj['Config']['Cmd']==[str(D/'worker.py'),'--preflight-only','--out',str(D/'preflight001.json')]
mount=[]
for i,c in enumerate(cmd[:-1]):
 if c=='--mount':
  v=dict(z.split('=',1) if '=' in z else (z,True) for z in cmd[i+1].split(','));mount.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in v))
 if c=='--env':assert cmd[i+1] in obj['Config']['Env']
assert sorted(mount)==sorted((str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in obj['Mounts'] if x['Type']=='bind') and 'CUDA_VISIBLE_DEVICES=' in obj['Config']['Env']
started=M/'started.json';assert started.exists();out={'status':'passed','independent_review':True,'plan_sha256':sha(D/'plan001.json'),'preflight_sha256':sha(D/'preflight001.json'),'preflight_started_sha256':sha(D/'monitor_preflight001/started.json'),'preflight_terminal_sha256':sha(D/'monitor_preflight001/terminal.json'),'preflight_exit_sha256':sha(D/'run_preflight001/exit.json'),'reviewer_path':str(Path(__file__).resolve()),'reviewer_sha256':sha(__file__),'review_started_sha256':sha(started),'review_terminal_path':str(M/'terminal.json'),'GT_opened':False,'scoring_performed':False,'scope':'author-greedy graph feasibility prelaunch; GPU only by root scheduler','limitations':['no GPU capture or correctness established in preflight','each K isolated and capture failure or OOM retained','graph setup/pool allocations separate from replay; not whole-device peak or TTFT','component timing may include input copies and is not a model integration result'],'errors':[]};a.out.write_text(json.dumps(out,indent=2)+'\n');print('passed')
