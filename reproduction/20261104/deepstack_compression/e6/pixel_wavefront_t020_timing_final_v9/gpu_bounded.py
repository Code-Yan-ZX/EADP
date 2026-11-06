"""Bounded E4-C Docker launch draft; unconditionally locked before GPU work.

 Real launch requires independently reviewed B/lightweight identity, time budget,
 model/device exclusivity, source binding and an external timing supervisor.
 Memory runs also require a host sampler handshake and remain separately locked.
"""
import argparse,importlib.util,json,os,signal,subprocess,time
from pathlib import Path
import hashlib
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()

def process_attestation(hostpid,container_id):
 proc=Path('/proc')/str(hostpid)
 argv=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
 cgroup=(proc/'cgroup').read_text()
 if not argv or container_id not in cgroup:raise RuntimeError('container process identity/cgroup mismatch')
 return {'host_argv':argv,'host_cgroup':cgroup}

def exit_code(result,require_gpu_idle=True):
 container=result.get('container_identity') or {};host=container.get('host_init') or {}
 if result.get('timed_out'):return 124
 if result.get('interrupted') is not None:return 128+result['interrupted']
 if (result.get('failure_type') is not None or result.get('client_exited_while_container_running') is not False or
     result.get('container_stopped_verified') is not True or not container.get('container_id') or
     type(host.get('pid')) is not int or host['pid']<=0 or type(host.get('start_ticks')) is not int or host['start_ticks']<=0 or not host.get('boot_id') or
     (require_gpu_idle and result.get('gpu_pids_after')!=[]) or result.get('container_exit_code')!=0):return 1
 return result.get('returncode') if result.get('returncode') is not None else 1

def validate_budget_window(plan,now,inner_timeout,outer_timeout,grace_seconds):
 """Bind explicit outer deadline to user's immutable three-day budget."""
 source=Path(plan['budget_source'])
 if sha(source)!=plan.get('budget_source_sha256'):raise ValueError('budget source changed')
 original=json.loads(source.read_text())
 if (original.get('budget_start_unix')!=plan.get('budget_start_unix') or original.get('budget_end_unix')!=plan.get('budget_end_unix') or
     type(inner_timeout) is not int or type(outer_timeout) is not int or type(grace_seconds) is not int or
     not 60<=inner_timeout<=43200 or grace_seconds<60 or outer_timeout<inner_timeout+grace_seconds or
     now<plan['budget_start_unix'] or now+outer_timeout>plan['budget_end_unix']):raise ValueError('insufficient bounded timing budget')
 return {'budget_source_sha256':sha(source),'inner_timeout_seconds':inner_timeout,'outer_timeout_seconds':outer_timeout,'grace_seconds':grace_seconds}

def run_bounded(cmd,name,timeout,gpu_pids,identity,artifacts,require_gpu_idle=True,on_identity=None):
 """Preserve structured failure and cleanup observations whenever storage works."""
 interrupted=[None]
 def on_signal(signum,_frame):interrupted[0]=signum
 previous={s:signal.signal(s,on_signal) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
 result={'returncode':None,'timed_out':False,'interrupted':None,'client_exited_while_container_running':False,
         'container_stopped_verified':False,'container_exit_code':None,'container_identity':None,
         'failure_type':None,'failure':None,'cleanup_attempts':[],'gpu_pids_after':None}
 process=None;log=None
 def cleanup(action,seconds):
  observation={'action':action}
  try:
   value=subprocess.run(['docker',action]+(['--time','10'] if action=='stop' else [])+[name],capture_output=True,text=True,timeout=seconds,check=False)
   observation.update(returncode=value.returncode,stdout=value.stdout[-2000:],stderr=value.stderr[-2000:])
  except Exception as exc:observation.update(error_type=type(exc).__name__,error=str(exc))
  result['cleanup_attempts'].append(observation)
  return observation.get('returncode')==0
 def inspect():
  value=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10,check=False)
  info={'returncode':value.returncode,'stderr':value.stderr[-2000:]}
  if value.returncode==0:
   data=json.loads(value.stdout)[0]
   info.update(id=data['Id'],image=data['Image'],running=data['State']['Running'],pid=data['State']['Pid'],exit_code=data['State']['ExitCode'],
               host_devices=data['HostConfig']['Devices'],device_requests=data['HostConfig']['DeviceRequests'],pid_mode=data['HostConfig']['PidMode'],
               entrypoint=data['Config']['Entrypoint'],command=data['Config']['Cmd'])
  result['last_inspect']=info
  if value.returncode!=0:raise RuntimeError('container inspect failed')
  return data
 def cleanup_and_inspect():
  stopped=cleanup('stop',30)
  try:state=inspect()
  except Exception:state=None
  if not stopped or state is None or state['State']['Running']:
   cleanup('kill',15)
   try:state=inspect()
   except Exception:state=None
  return state
 try:
  try:
   log=(artifacts/'docker.log').open('x')
   process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);begin=time.monotonic()
   while process.poll() is None:
    if result['container_identity'] is None:
     try:item=inspect()
     except RuntimeError:item=None # container may not exist immediately after launch
     if item:
      hostpid=item['State']['Pid']
      if hostpid:
       host=identity(hostpid)
       if host is None:raise RuntimeError('container host identity unavailable')
       result['container_identity']={'container_id':item['Id'],'image':item['Image'],'host_init':host,**process_attestation(hostpid,item['Id']),'observed_unix':time.time()}
       with (artifacts/'container_identity.json').open('x') as f:json.dump(result['container_identity'],f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
       if on_identity is not None:on_identity(result['container_identity'])
    if time.monotonic()-begin>=timeout or interrupted[0] is not None:
     result['timed_out']=interrupted[0] is None;result['interrupted']=interrupted[0]
     cleanup_and_inspect()
     try:process.wait(timeout=30)
     except subprocess.TimeoutExpired:process.kill();process.wait()
     break
    time.sleep(1)
   result['returncode']=process.wait()
   if interrupted[0] is not None:result['interrupted']=interrupted[0]
   if time.monotonic()-begin>=timeout and interrupted[0] is None:result['timed_out']=True
   state=inspect()
   if state['State']['Running']:
    result['client_exited_while_container_running']=True
    state=cleanup_and_inspect()
    if state is None:raise RuntimeError('container state unknown after cleanup')
   if state['State']['Running']:raise RuntimeError('container still running')
   if not result['container_identity'] or state['Id']!=result['container_identity']['container_id']:raise RuntimeError('container identity unavailable or changed')
   result['container_stopped_verified']=True;result['container_exit_code']=state['State']['ExitCode']
  except BaseException as exc:
   result['failure_type']=type(exc).__name__;result['failure']=str(exc)
   if process is not None:
    cleanup_and_inspect()
    if process.poll() is None:
     try:process.kill();process.wait(timeout=15)
     except Exception as kill_exc:result['cleanup_attempts'].append({'action':'client_kill','error_type':type(kill_exc).__name__,'error':str(kill_exc)})
  finally:
   if log is not None:log.close()
   if process is not None:
    try:
     last=inspect();known=result['container_identity'] or {}
     result['container_stopped_verified']=bool(last['State']['Running'] is False and last['State']['Pid']==0 and last['Id']==known.get('container_id'))
     result['container_exit_code']=last['State']['ExitCode']
    except Exception as exc:
     result['container_stopped_verified']=False;result['final_inspect_error']={'type':type(exc).__name__,'message':str(exc)}
   result['finished_unix']=time.time();result['interrupted']=interrupted[0] if interrupted[0] is not None else result['interrupted']
   if gpu_pids is not None:
    try:result['gpu_pids_after']=gpu_pids()
    except Exception as exc:result['gpu_query_error']={'type':type(exc).__name__,'message':str(exc)}
   try:result['controller']=identity(os.getpid())
   except Exception as exc:result['controller_error']={'type':type(exc).__name__,'message':str(exc)}
   with (artifacts/'exit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  return exit_code(result,require_gpu_idle=require_gpu_idle)
 finally:
  for sig,handler in previous.items():signal.signal(sig,handler)

