"""Bounded E4-C Docker launch draft; unconditionally locked before GPU work.

 Real launch requires independently reviewed B/lightweight identity, time budget,
 model/device exclusivity, source binding and an external timing supervisor.
 Memory runs also require a host sampler handshake and remain separately locked.
"""
import argparse,importlib.util,json,os,signal,subprocess,time
from pathlib import Path
from identity_gate import require_gpu_identity
from real_worker import PLAN,HERE,BASE,sha

def exit_code(result):
 container=result.get('container_identity') or {};host=container.get('host_init') or {}
 if result.get('timed_out'):return 124
 if result.get('interrupted') is not None:return 128+result['interrupted']
 if (result.get('failure_type') is not None or result.get('client_exited_while_container_running') is not False or
     result.get('container_stopped_verified') is not True or not container.get('container_id') or
     type(host.get('pid')) is not int or host['pid']<=0 or type(host.get('start_ticks')) is not int or host['start_ticks']<=0 or not host.get('boot_id') or
     result.get('gpu_pids_after')!=[] or result.get('container_exit_code')!=0):return 1
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

def run_bounded(cmd,name,timeout,gpu_pids,identity,artifacts):
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
   info.update(id=data['Id'],running=data['State']['Running'],pid=data['State']['Pid'],exit_code=data['State']['ExitCode'])
  result['last_inspect']=info
  if value.returncode!=0:raise RuntimeError('container inspect failed')
  return data
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
       result['container_identity']={'container_id':item['Id'],'image':item['Image'],'host_init':identity(hostpid),'observed_unix':time.time()}
       with (artifacts/'container_identity.json').open('x') as f:json.dump(result['container_identity'],f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    if time.monotonic()-begin>=timeout or interrupted[0] is not None:
     result['timed_out']=interrupted[0] is None;result['interrupted']=interrupted[0]
     if not cleanup('stop',30):cleanup('kill',15)
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
    if not cleanup('stop',30):cleanup('kill',15)
    state=inspect()
   if state['State']['Running']:raise RuntimeError('container still running')
   if not result['container_identity'] or state['Id']!=result['container_identity']['container_id']:raise RuntimeError('container identity unavailable or changed')
   result['container_stopped_verified']=True;result['container_exit_code']=state['State']['ExitCode']
  except BaseException as exc:
   result['failure_type']=type(exc).__name__;result['failure']=str(exc)
   if process is not None:
    if not cleanup('stop',30):cleanup('kill',15)
    if process.poll() is None:
     try:process.kill();process.wait(timeout=15)
     except Exception as kill_exc:result['cleanup_attempts'].append({'action':'client_kill','error_type':type(kill_exc).__name__,'error':str(kill_exc)})
  finally:
   if log is not None:log.close()
   result['finished_unix']=time.time();result['interrupted']=interrupted[0] if interrupted[0] is not None else result['interrupted']
   try:result['gpu_pids_after']=gpu_pids()
   except Exception as exc:result['gpu_query_error']={'type':type(exc).__name__,'message':str(exc)}
   try:result['controller']=identity(os.getpid())
   except Exception as exc:result['controller_error']={'type':type(exc).__name__,'message':str(exc)}
   with (artifacts/'exit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  return exit_code(result)
 finally:
  for sig,handler in previous.items():signal.signal(sig,handler)

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--role',choices=('latency_interleaved','memory_isolated'),required=True);p.add_argument('--mode',choices=('C-ON','C-DROP','C-SKIP'));p.add_argument('--timeout',type=int,required=True)
 for key in ('e1-acceptance','e1-started','e1-terminal','out'):p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args()
 if not a.run_id.isalnum() or not 60<=a.timeout<=43200 or (a.role=='memory_isolated')!=(a.mode is not None):raise ValueError('invalid bounded run identity')
 require_gpu_identity() # must reject before importing GPU tools or using GPU
 if a.role=='memory_isolated':raise ValueError('memory launch locked: sampler identity handshake incomplete')
 raise ValueError('timing launch locked: supervisor/budget/source/identity admission incomplete')
 # Future reviewed admission must pass identical args to real_worker.py in the
 # fixed author container, freeze launch and source hashes, then call run_bounded.
if __name__=='__main__':raise SystemExit(main())
