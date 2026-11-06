"""Owned-container cleanup after a failed supervised job; no standalone CLI."""
import json,re,os,signal,time
from pathlib import Path
def recover(job,jobdir,plan_sha,guard_class):
 """Only touch an exact nonce/name/label-owned container; preserve uncertainty."""
 d=Path(jobdir);path=d/'launch.json'
 if not path.exists():return {'status':'not_created','note':'launch record absent before create'}
 launch=json.loads(path.read_text());nonce=launch.get('nonce');command=launch.get('docker_create')
 if launch.get('job')!=job or launch.get('plan_sha256')!=plan_sha or not isinstance(nonce,str) or not re.fullmatch(r'[0-9a-f]{32}',nonce) or not isinstance(command,list):raise ValueError('unsafe recovery evidence')
 name='eadp-memory-pixel-'+nonce;guard=guard_class(name,nonce,command);guard.create_submitted=True;cid=guard.recover_owned_container()
 if cid is None:return {'status':'absent','container_id':None}
 errors=[]
 try:state=guard.inspect(cid)
 except BaseException as e:return {'status':'unknown','container_id':cid,'errors':[type(e).__name__+': '+str(e)]}
 if state.get('running') is True or state.get('pid')!=0:
  try:guard.stop_container(cid)
  except BaseException as e:errors.append('stop '+type(e).__name__+': '+str(e))
  try:state=guard.inspect(cid)
  except BaseException as e:errors.append('inspect '+type(e).__name__+': '+str(e));state=None
  if not state or state.get('running') is not False or state.get('pid')!=0:
   try:guard.kill_container(cid)
   except BaseException as e:errors.append('kill '+type(e).__name__+': '+str(e))
   try:state=guard.inspect(cid)
   except BaseException as e:errors.append('final inspect '+type(e).__name__+': '+str(e));state=None
 ok=bool(state and state.get('container_id')==cid and state.get('running') is False and state.get('pid')==0)
 return {'status':'stopped' if ok else 'unknown','container_id':cid,'errors':errors,'sampler_status':'must independently inspect sampler supervisor and process group'}

def stop_sampler(jobdir,*,proc=Path('/proc'),kill=os.kill,getsid=os.getsid,clock=time.monotonic,sleep=time.sleep,timeout=90):
 """Signal only the exact supervised sampler for this job; unknown stays unknown."""
 d=Path(jobdir);launch=json.loads((d/'launch.json').read_text());monitor=d/'sampler_monitor';started=monitor/'started.json';terminal=monitor/'terminal.json'
 if not started.exists():return {'status':'unknown','reason':'sampler supervisor has no started or terminal record'}
 s=json.loads(started.read_text());pid=s.get('worker_pid');expected_worker=launch.get('sampler_worker');expected_supervisor=launch.get('sampler_supervisor')
 if type(pid) is not int or pid<=0 or s.get('process_group')!=pid or s.get('command')!=expected_worker:return {'status':'unknown','reason':'sampler started identity mismatch'}
 def terminal_state():
  try:t=json.loads(terminal.read_text())
  except FileNotFoundError:return None
  return t if t.get('worker_pid')==pid and t.get('remaining_process_group')==[] else None
 done=terminal_state()
 if done is not None:return {'status':'stopped','terminal':done}
 def argv(pid):return [x.decode() for x in (proc/str(pid)/'cmdline').read_bytes().split(b'\0') if x]
 try:
  fields=(proc/str(pid)/'stat').read_text().rsplit(')',1)[1].split();parent=int(fields[1])
  if argv(pid)!=expected_worker or argv(parent)!=expected_supervisor or getsid(parent)!=parent:return {'status':'unknown','reason':'live sampler argv/session mismatch'}
  kill(parent,signal.SIGTERM)
 except (FileNotFoundError,ProcessLookupError,PermissionError,ValueError,IndexError):return {'status':'unknown','reason':'live sampler identity unavailable'}
 deadline=clock()+timeout
 while clock()<deadline:
  done=terminal_state()
  if done is not None:return {'status':'stopped','terminal':done}
  sleep(.1)
 return {'status':'unknown','reason':'sampler supervisor did not establish terminal after TERM'}
if __name__=='__main__':raise SystemExit('no CLI or GPU admission')
