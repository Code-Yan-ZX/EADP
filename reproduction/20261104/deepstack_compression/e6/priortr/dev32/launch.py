"""Bounded single-GPU Docker launch for PriorTR DEV32; gate remains locked."""
import argparse,importlib.util,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
import gate
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];PREP=HERE.parent/'preparation';PYTHON=HERE.parent/'install001/env/bin/python'

def validate_supervision(started,run,budget,now,initial):
 if (gate.sha(BASE/'initial_plan.json')!=budget['budget_source_sha256'] or now+budget['outer_timeout_seconds']>initial['budget_end_unix'] or
     started.get('worker_pid')!=os.getpid() or started.get('timeout_seconds')!=budget['outer_timeout_seconds'] or
     started.get('stale_seconds')!=budget['journal_stale_seconds'] or started.get('snapshot_seconds')!=budget['snapshot_seconds'] or
     started.get('poll_seconds')!=budget['poll_seconds'] or started.get('grace_seconds')!=budget['grace_seconds'] or
     started.get('progress_paths')!=[str(run/'data/generated.jsonl')]):raise ValueError('supervision/deadline mismatch')

def configure(command,cache):
 command=list(command);cache=Path(cache).resolve();overrides={'PYTHONPATH':str(PREP),'HF_HOME':str(cache/'hf'),'XDG_CACHE_HOME':str(cache/'xdg'),'MPLCONFIGDIR':str(cache/'mpl'),'TMPDIR':str(cache),'TORCH_EXTENSIONS_DIR':str(cache/'extensions'),'TORCHINDUCTOR_CACHE_DIR':str(cache/'inductor'),'USER':'eadp','LOGNAME':'eadp'};seen=set()
 for i,x in enumerate(command):
  key=x.split('=',1)[0]
  if key in overrides:
   if key in seen:raise ValueError('duplicate container environment')
   command[i]=key+'='+overrides[key];seen.add(key)
 at=command.index('--workdir');extra=[]
 for key,value in overrides.items():
  if key not in seen:extra.extend(['--env',key+'='+value])
 command[at:at]=extra;command[command.index('--entrypoint')+1]=str(PYTHON)
 return command

def clean_exit(r):
 return r.get('returncode')==0 and r.get('container_exit_code')==0 and r.get('timed_out') is False and r.get('interrupted') is None and r.get('failure') is None and r.get('container_stopped_verified') is True and r.get('client_exited_while_container_running') is False and r.get('gpu_pids_after')==[] and isinstance(r.get('container_identity'),dict)

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
 if not a.run_id.isalnum():raise ValueError('invalid run id')
 approval=gate.execution_gate();gate.upstream();budget=gate.read(HERE/'resource_plan.json');initial=gate.read(BASE/'initial_plan.json');run=HERE/('run_'+a.run_id);started=gate.read(HERE/('monitor_'+a.run_id)/'started.json')
 validate_supervision(started,run,budget,time.time(),initial)
 if shutil.disk_usage(HERE).free<budget['minimum_disk_free_bytes']:raise ValueError('insufficient disk')
 planpath=BASE.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';plan=gate.read(planpath)
 for group in ('source_sha256','implementation_sha256'):
  for path,digest in plan[group].items():
   if gate.sha(path)!=digest:raise ValueError('upstream source changed')
 module_path=planpath.parent/'run_panel.py';spec=importlib.util.spec_from_file_location('_priortr_dev32_container_runtime',module_path);module=importlib.util.module_from_spec(spec);sys.path.insert(0,str(module_path.parent));spec.loader.exec_module(module)
 if Path(module.__file__).resolve()!=module_path.resolve() or module.gpu_pids():raise ValueError('container module or GPU occupancy')
 run.mkdir(exist_ok=False);cache=run/'runtime_cache';cache.mkdir();name='eadp-priortr32-'+a.run_id+'-'+str(os.getpid());worker=HERE/'panel_worker.py'
 cmd=configure(module.container_command(plan,[str(worker),'--execute','--out',str(run)],True,name),cache)
 if cmd.count('--rm')!=1:raise ValueError('unexpected container removal')
 cmd.remove('--rm');ctrl=module.identity(os.getpid());now=time.time();validate_supervision(started,run,budget,now,initial)
 launch={'command':cmd,'controller':ctrl,'worker_sha256':gate.sha(worker),'launcher_sha256':gate.sha(__file__),'budget_sha256':gate.sha(HERE/'resource_plan.json'),'approval':approval,'started_unix':now}
 with (run/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 interrupted=[None]
 def handle(sig,_frame):interrupted[0]=sig
 for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,handle)
 result={'returncode':None,'container_exit_code':None,'timed_out':False,'interrupted':None,'container_stopped_verified':False,'client_exited_while_container_running':False,'container_identity':None,'controller':ctrl,'name':name,'failure':None,'stop_attempts':[]};client=None;state=None
 with (run/'launcher.log').open('x') as log:
  def inspect():
   q=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10);return json.loads(q.stdout)[0] if q.returncode==0 else None
  def stop():
   for method in ('stop','kill'):
    try:
     cmd=['docker','stop','--time','10',name] if method=='stop' else ['docker','kill',name]
     q=subprocess.run(cmd,stdout=log,stderr=log,timeout=30,check=False);result['stop_attempts'].append({'method':method,'returncode':q.returncode});current=inspect()
     if current is not None and not current['State']['Running']:return current
    except Exception as exc:result['stop_attempts'].append({'method':method,'error':repr(exc)})
   return None
  try:
   client=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);begin=time.monotonic()
   while client.poll() is None:
    if result['container_identity'] is None:
     state=inspect()
     if state is not None and state['State']['Pid']:result['container_identity']={'container_id':state['Id'],'image':state['Image'],'host_init':module.identity(state['State']['Pid'])}
    if interrupted[0] is not None or time.monotonic()-begin>=budget['inner_timeout_seconds']:
     result['interrupted']=interrupted[0];result['timed_out']=interrupted[0] is None;state=stop();break
    time.sleep(1)
   try:result['returncode']=client.wait(timeout=30)
   except subprocess.TimeoutExpired:client.kill();result['returncode']=client.wait()
   state=inspect()
   if state is not None and state['State']['Running']:result['client_exited_while_container_running']=True;state=stop()
   if state is None or state['State']['Running'] or result['container_identity'] is None or state['Id']!=result['container_identity']['container_id']:raise RuntimeError('container identity/exit unverified')
   result['container_exit_code']=state['State']['ExitCode'];result['container_stopped_verified']=True
  except Exception as exc:result['failure']=repr(exc)
  finally:
   if not result['container_stopped_verified']:
    stopped=stop()
    if stopped is not None and result['container_identity'] is not None and stopped['Id']==result['container_identity']['container_id']:result['container_exit_code']=stopped['State']['ExitCode'];result['container_stopped_verified']=True
   if client is not None:
    try:
     if client.poll() is None:client.kill();client.wait(timeout=10)
    except Exception as exc:result['failure']=result.get('failure') or ('client cleanup: '+repr(exc))
   result['interrupted']=interrupted[0];result['finished_unix']=time.time()
   try:result['gpu_pids_after']=module.gpu_pids()
   except Exception as exc:result['gpu_pids_after']=None;result['gpu_query_error']=repr(exc)
   with (run/'exit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if clean_exit(result) else 1
if __name__=='__main__':sys.exit(main())
