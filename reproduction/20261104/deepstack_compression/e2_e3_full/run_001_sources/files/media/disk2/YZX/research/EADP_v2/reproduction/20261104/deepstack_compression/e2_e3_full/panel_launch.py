"""Prepared bounded Docker launcher; execution is unconditionally locked."""
import argparse,importlib.util,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
from full_imports import local
_gate=local('panel_gate');execution_gate=_gate.execution_gate;sha=_gate.sha
HERE=Path(__file__).resolve().parent;BASE=HERE.parent

def validate_supervision(started,run,budget,now,initial):
 if (sha(initial)!=budget['budget_source_sha256'] or json.loads(initial.read_text())['budget_end_unix']!=budget['budget_end_unix'] or
     now+budget['outer_timeout_seconds']>budget['budget_end_unix'] or
     started.get('worker_pid')!=os.getpid() or started.get('timeout_seconds')!=budget['outer_timeout_seconds'] or
     started.get('stale_seconds')!=budget['journal_stale_seconds'] or started.get('snapshot_seconds')!=budget['snapshot_seconds'] or
     started.get('poll_seconds')!=budget['poll_seconds'] or started.get('grace_seconds')!=budget['grace_seconds'] or
     started.get('progress_paths')!=[str(run/'data/generated.jsonl')]):raise ValueError('supervision or deadline mismatch')

def clean_exit(result):
 return (result.get('returncode')==0 and result.get('container_exit_code')==0 and result.get('timed_out') is False and
         result.get('interrupted') is None and result.get('client_exited_while_container_running') is False and
         result.get('container_stopped_verified') is True and result.get('gpu_pids_after')==[] and
         isinstance(result.get('container_identity'),dict) and result.get('failure') is None)

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True)
 for name in ('e1-acceptance','e1-started','e1-terminal'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args()
 if not a.run_id.isalnum():raise ValueError('invalid run id')
 approval=execution_gate()
 budget=json.loads((HERE/'budget_plan.json').read_text());initial=BASE/'initial_plan.json';run=HERE/('run_'+a.run_id)
 started=json.loads((HERE/('monitor_'+a.run_id)/'started.json').read_text());validate_supervision(started,run,budget,time.time(),initial)
 if shutil.disk_usage(HERE).free<budget['minimum_free_bytes']:raise ValueError('insufficient free disk')
 planpath=BASE.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';plan=json.loads(planpath.read_text())
 for group in ('source_sha256','implementation_sha256'):
  for path,digest in plan[group].items():
   if sha(path)!=digest:raise ValueError('upstream source changed')
 module_path=planpath.parent/'run_panel.py';spec=importlib.util.spec_from_file_location('_eadp_e2e3_container_runtime',module_path)
 module=importlib.util.module_from_spec(spec);sys.path.insert(0,str(module_path.parent));spec.loader.exec_module(module)
 if Path(module.__file__).resolve()!=module_path.resolve():raise ImportError('unexpected container runtime')
 if module.gpu_pids():raise ValueError('GPU occupied')
 run.mkdir(exist_ok=False);name='eadp-e2e3full-'+a.run_id+'-'+str(os.getpid());worker=HERE/'panel_worker.py'
 args=[str(worker),'--execute','--e1-acceptance',str(a.e1_acceptance.resolve()),'--e1-started',str(a.e1_started.resolve()),'--e1-terminal',str(a.e1_terminal.resolve()),'--out',str(run)]
 cmd=module.container_command(plan,args,True,name)
 if cmd.count('--rm')!=1:raise ValueError('unexpected removal policy')
 cmd.remove('--rm');ctrl=module.identity(os.getpid());now=time.time();validate_supervision(started,run,budget,now,initial)
 launch={'command':cmd,'controller':ctrl,'worker_sha256':sha(worker),'launcher_sha256':sha(__file__),'budget_sha256':sha(HERE/'budget_plan.json'),'approval':approval,'started_unix':now}
 with (run/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 interrupted=[None]
 def handle(sig,_frame):interrupted[0]=sig
 for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,handle)
 result={'returncode':None,'container_exit_code':None,'timed_out':False,'interrupted':None,'container_stopped_verified':False,
         'client_exited_while_container_running':False,'container_identity':None,'controller':ctrl,'name':name,'failure':None,'stop_attempts':[]}
 client=None;state=None
 with (run/'launcher.log').open('x') as log:
  def inspect():
   p=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10)
   return json.loads(p.stdout)[0] if p.returncode==0 else None
  def stop_container():
   for method in ('stop','kill'):
    try:
     command=['docker','stop','--time','10',name] if method=='stop' else ['docker','kill',name]
     p=subprocess.run(command,stdout=log,stderr=log,timeout=30,check=False);result['stop_attempts'].append({'method':method,'returncode':p.returncode})
     current=inspect()
     if current is not None and not current['State']['Running']:return current
    except Exception as e:result['stop_attempts'].append({'method':method,'error':repr(e)})
   return None
  try:
   client=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);begin=time.monotonic()
   while client.poll() is None:
    if result['container_identity'] is None:
     state=inspect()
     if state is not None and state['State']['Pid']:
      result['container_identity']={'container_id':state['Id'],'image':state['Image'],'host_init':module.identity(state['State']['Pid'])}
    if interrupted[0] is not None or time.monotonic()-begin>=budget['inner_timeout_seconds']:
     result['interrupted']=interrupted[0];result['timed_out']=interrupted[0] is None;state=stop_container();break
    time.sleep(1)
   try:result['returncode']=client.wait(timeout=30)
   except subprocess.TimeoutExpired:client.kill();result['returncode']=client.wait()
   state=inspect()
   if state is not None and state['State']['Running']:
    result['client_exited_while_container_running']=True;state=stop_container()
   if state is None or state['State']['Running'] or result['container_identity'] is None or state['Id']!=result['container_identity']['container_id']:raise RuntimeError('container identity/exit unverified')
   result['container_exit_code']=state['State']['ExitCode'];result['container_stopped_verified']=True
  except Exception as e:result['failure']=repr(e)
  finally:
   if not result['container_stopped_verified']:
    stopped=stop_container()
    if stopped is not None and result['container_identity'] is not None and stopped['Id']==result['container_identity']['container_id']:
     result['container_exit_code']=stopped['State']['ExitCode'];result['container_stopped_verified']=True
   if client is not None:
    try:
     if client.poll() is None:client.kill();client.wait(timeout=10)
    except Exception as e:result['failure']=result.get('failure') or ('client cleanup: '+repr(e))
   result['interrupted']=interrupted[0];result['finished_unix']=time.time()
   try:result['gpu_pids_after']=module.gpu_pids()
   except Exception as e:result['gpu_pids_after']=None;result['gpu_query_error']=repr(e)
   with (run/'exit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if clean_exit(result) else 1
if __name__=='__main__':sys.exit(main())
