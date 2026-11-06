"""Final independent E4-C code/preflight review; never runs a model or GPU."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent; HERE=BASE/'e4_c_timing'
sys.path.insert(0,str(HERE))
from light_admission import admit,sources
from real_worker import PLAN,WEIGHTS

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 assert a.out.resolve()==BASE/'e1_independent_review/e4c_light_prelaunch_review.json'
 temp=BASE/'e1_independent_review/e4c_light_worker_preflight_review.json';proof=read(temp)
 assert proof['status']=='passed' and proof['scope']=='worker_preflight_only' and proof['GT_opened'] is False and proof['scoring_performed'] is False
 prior=admit(preflight_only=True); assert prior['review_sha256']==sha(temp)
 sources_now={str(x):sha(x) for x in sources()};assert len(sources_now)==36 and proof['source_sha256']==sources_now
 paths={name:Path(proof[name+'_path']) for name in ('reference','reference_started','reference_terminal','cpu_proof','cpu_started','cpu_terminal','preflight_launch','preflight_exit','preflight_started','preflight_terminal')}
 assert all(sha(path)==proof[name+'_sha256'] for name,path in paths.items())
 paths.update({'worker_preflight_launch':HERE/'light_preflight012/launch.json','worker_preflight_exit':HERE/'light_preflight012/exit.json',
               'worker_preflight_started':HERE/'monitor_light_preflight012/started.json','worker_preflight_terminal':HERE/'monitor_light_preflight012/terminal.json',
               'worker_preflight_result':HERE/'light_preflight012/worker_preflight.json'})
 start,end=read(paths['worker_preflight_started']),read(paths['worker_preflight_terminal']);cmd=start['command'];cwd=Path(start['cwd'])
 assert end['reason']=='completed' and end['worker_returncode']==0 and end['remaining_process_group']==[] and end['worker_pid']==start['worker_pid']==start['process_group']
 assert [(cwd/x).resolve() for x in cmd if Path(x).name=='light_preflight_launch.py']==[(HERE/'light_preflight_launch.py').resolve()]
 assert cmd.count('--run-id')==1 and cmd[cmd.index('--run-id')+1]=='012' and cmd.count('--worker-preflight')==1
 assert cmd.count('--references')==1 and (cwd/cmd[cmd.index('--references')+1]).resolve()==paths['reference'].resolve()
 launch,exit,result=(read(paths[x]) for x in ('worker_preflight_launch','worker_preflight_exit','worker_preflight_result'))
 cmd=launch['command']; assert launch['controller']['pid']==start['worker_pid'] and launch['approval_bundle']==prior and launch['worker_preflight'] is True
 assert launch['source_sha256']==sources_now and launch['worker_sha256']==sources_now[str(HERE/'light_smoke_worker.py')] and launch['gpu_requested'] is False
 assert '--gpus' not in cmd and '--device' not in cmd and cmd.count('--preflight-only')==1 and cmd.count(str(HERE/'light_smoke_worker.py'))==1
 assert cmd.count('--references')==1 and Path(cmd[cmd.index('--references')+1]).resolve()==paths['reference'].resolve()
 assert cmd.count('--out')==1 and Path(cmd[cmd.index('--out')+1]).resolve()==(HERE/'light_preflight012').resolve()
 assert exit['returncode']==0 and exit['error'] is None and exit['interrupted'] is None and exit['timed_out'] is False and exit['source_unchanged'] is True
 assert exit['container_stopped_verified'] is True and exit['client_exited_while_container_running'] is False
 ident,old=exit['identity'],exit['inspect'];actual=json.loads(subprocess.check_output(['docker','inspect',ident['container_id']],text=True))[0];state=actual['State'];plan=read(PLAN)
 assert actual['Id']==ident['container_id']==old['id'] and actual['Image']==old['image']==plan['container_image']
 assert state['Running'] is False and state['Pid']==0 and state['ExitCode']==0 and not state['OOMKilled']
 assert not actual['HostConfig']['Devices'] and not actual['HostConfig']['DeviceRequests']
 assert actual['Config']['Entrypoint']==old['config']['Entrypoint'] and actual['Config']['Cmd']==old['config']['Cmd']
 assert actual['Config']['Cmd']==[str(HERE/'light_smoke_worker.py'),'--preflight-only','--references',str(paths['reference']), '--out',str(HERE/'light_preflight012')]
 assert result['status']=='passed' and result['cuda_initialized'] is False and result['model_loaded'] is False and result['predictions'] is False
 assert result['reference_sha256']==sha(paths['reference']) and result['source_sha256']==sources_now and result['plan_sha256']==sha(PLAN) and result['weights_identity_sha256']==sha(WEIGHTS) and result['versions']==plan['expected_versions']
 weights=read(WEIGHTS)['files'];assert len(weights)==4 and set(result['weight_stats_before'])=={x['path'] for x in weights}
 for w in weights:
  st=Path(w['path']).stat();assert result['weight_stats_before'][w['path']]==[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns] and st.st_size==w['bytes']
 log=HERE/'light_preflight012/container.log';last=json.loads([x for x in log.read_text().splitlines() if x.startswith('{')][-1]);assert last=={'status':'passed','cuda_initialized':False,'model_loaded':False}
 q={'status':'passed','independent_review':True,'GT_opened':False,'scoring_performed':False,'errors':[],'scope':'gpu_diagnostic_prelaunch',
    'reviewer_sha256':sha(__file__),'source_sha256':sources_now,'worker_preflight_container_log_sha256':sha(log),'temporary_review_sha256':sha(temp),
    'limits':['Worker preflight rehashed four payloads; this independent review checks their current stat rather than rehashing payloads.',
              'Old E4 ON evidence has no KV contents; only fresh ON and lightweight ON can compare KV fingerprints.',
              'This review permits scheduling the diagnostic only; no timing or accuracy conclusion, and GPU launch remains subject to root coordination.']}
 for name,path in paths.items():q[name+'_path']=str(path);q[name+'_sha256']=sha(path)
 with a.out.open('x') as f:json.dump(q,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','scope':q['scope'],'source_count':len(sources_now)}))
if __name__=='__main__':main()
