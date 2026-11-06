"""Bounded, retained-container CPU import probe. Never requests GPU devices."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from real_worker import PLAN,sha,HERE

def save(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def inspect_container(name,result):
 try:
  x=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10,check=False)
  if x.returncode!=0:raise RuntimeError('docker inspect exit '+str(x.returncode))
  item=json.loads(x.stdout)[0];snapshot={'id':item['Id'],'image':item['Image'],'state':item['State'],'config':{'Image':item['Config']['Image'],'Entrypoint':item['Config']['Entrypoint'],'Cmd':item['Config']['Cmd']},'host_devices':item['HostConfig']['Devices'],'device_requests':item['HostConfig']['DeviceRequests']}
  result['inspect']=snapshot;return snapshot
 except Exception as exc:
  result.setdefault('inspect_errors',[]).append({'type':type(exc).__name__,'message':str(exc)});return None

def cleanup_container(name,result):
 """Always inspect, attempt kill on failed stop or unknown/running state."""
 def call(action,timeout):
  item={'action':action}
  try:
   x=subprocess.run(['docker',action]+(['--time','10'] if action=='stop' else [])+[name],capture_output=True,text=True,timeout=timeout,check=False)
   item.update(returncode=x.returncode,stdout=x.stdout[-2000:],stderr=x.stderr[-2000:])
  except Exception as exc:item.update(error_type=type(exc).__name__,error=str(exc))
  result.setdefault('cleanup_attempts',[]).append(item);return item.get('returncode')==0
 stopped=call('stop',30);after=inspect_container(name,result)
 if not stopped or after is None or after['state']['Running']:
  call('kill',15);after=inspect_container(name,result)
 return after

def stop_reason(interrupted,elapsed,timeout):
 if interrupted is not None:return 'interrupted'
 if elapsed>=timeout:return 'timed_out'
 return None

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--worker-preflight',action='store_true');p.add_argument('--references',type=Path);args=p.parse_args()
 if not args.run_id.isalnum():raise ValueError('invalid preflight identity')
 if args.worker_preflight!=(args.references is not None):raise ValueError('worker preflight requires exact references')
 approval=None
 if args.worker_preflight:
  from light_admission import admit
  approval=admit(preflight_only=True)
  if args.references.resolve()!=Path(approval['reference_path']).resolve() or sha(args.references)!=approval['reference_sha256']:raise ValueError('preflight reference mismatch')
 timeout_seconds=1200 if args.worker_preflight else 180
 out=HERE/('light_preflight'+args.run_id);out.mkdir(exist_ok=False);name='eadp-light-cpu-'+args.run_id+'-'+str(os.getpid())
 sys.path.insert(0,str(PLAN.parent));from run_panel import container_command,identity
 plan=json.loads(PLAN.read_text());probe=HERE/'light_container_probe.py';worker=HERE/'light_smoke_worker.py'
 target=[str(worker),'--preflight-only','--references',str(args.references.resolve()),'--out',str(out)] if args.worker_preflight else [str(probe)]
 cmd=container_command(plan,target,False,name)
 if cmd.count('--rm')!=1 or '--device' in cmd or '--gpus' in cmd:raise ValueError('CPU-only container command invalid')
 cmd.remove('--rm');controller=identity(os.getpid())
 from light_admission import sources
 source=sources()
 source_hash={str(p):sha(p) for p in source}
 save(out/'launch.json',{'command':cmd,'name':name,'controller':controller,'source_sha256':source_hash,'timeout_seconds':timeout_seconds,'gpu_requested':False,
                          'worker_preflight':args.worker_preflight,'approval_bundle':approval,'worker_sha256':sha(worker)})
 interrupted=[None]
 def handle(signum,_frame):interrupted[0]=signum
 old={s:signal.signal(s,handle) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
 result={'returncode':None,'interrupted':None,'timed_out':False,'identity':None,'inspect':None,'error':None,
         'client_exited_while_container_running':False,'container_stopped_verified':False,'cleanup_attempts':[],'inspect_errors':[]};p=None
 try:
  with (out/'container.log').open('x') as log:
   p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic()
   while p.poll() is None:
    state=inspect_container(name,result)
    if state is not None:
     pid=state['state']['Pid']
     if pid and result['identity'] is None:result['identity']={'container_id':state['id'],'image':state['image'],'host_init':identity(pid)}
    reason=stop_reason(interrupted[0],time.monotonic()-start,timeout_seconds)
    if reason is not None:
     result['interrupted']=interrupted[0];result['timed_out']=reason=='timed_out';cleanup_container(name,result)
     try:p.wait(timeout=30)
     except subprocess.TimeoutExpired:p.kill();p.wait()
     break
    time.sleep(.5)
   result['returncode']=p.wait()
   state=inspect_container(name,result)
   if state is None:raise RuntimeError('container final inspect failed')
   if state['state']['Running']:
    result['client_exited_while_container_running']=True;cleanup_container(name,result)
    raise RuntimeError('docker client exited while container still running')
 except BaseException as exc:
  result['error']={'type':type(exc).__name__,'message':str(exc)}
 finally:
  try:
   if p is not None and (p.poll() is None or result['error'] is not None):cleanup_container(name,result)
   if p is not None and p.poll() is None:p.kill();p.wait(timeout=15)
  except Exception as exc:result['cleanup_error']={'type':type(exc).__name__,'message':str(exc)}
  if p is not None:
   last=inspect_container(name,result);identity_seen=result.get('identity') or {}
   result['container_stopped_verified']=bool(last and last['state'].get('Running') is False and last['state'].get('Pid')==0 and last.get('id')==identity_seen.get('container_id'))
  try:result['source_unchanged']=all(sha(path)==digest for path,digest in source_hash.items())
  except Exception as exc:result['source_unchanged']=False;result['source_error']={'type':type(exc).__name__,'message':str(exc)}
  result['interrupted']=interrupted[0];result['finished_unix']=time.time()
  try:save(out/'exit.json',result)
  finally:
   for s,handler in old.items():signal.signal(s,handler)
 state=(result.get('inspect') or {}).get('state') or {};ident=result.get('identity') or {};insp=result.get('inspect') or {}
 ok=(result['returncode']==0 and not result['timed_out'] and result['interrupted'] is None and result['error'] is None and result['source_unchanged'] and
     result['client_exited_while_container_running'] is False and result['container_stopped_verified'] is True and not result.get('cleanup_error') and
     state.get('Running') is False and state.get('Pid')==0 and state.get('ExitCode')==0 and ident.get('container_id')==insp.get('id') and
     insp.get('image')==plan['container_image'] and not insp.get('host_devices') and not insp.get('device_requests'))
 return 0 if ok else 1

if __name__=='__main__':sys.exit(main())
