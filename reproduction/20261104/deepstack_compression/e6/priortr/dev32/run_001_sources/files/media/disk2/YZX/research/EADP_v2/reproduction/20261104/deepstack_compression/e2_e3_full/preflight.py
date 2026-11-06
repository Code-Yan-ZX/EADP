"""Actual bounded same-image no-GPU pre-model full-panel entry preflight."""
import json,os,signal,subprocess,time
from pathlib import Path
from full_imports import local
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;gate=local('panel_gate');sha=gate.sha

def supervise_container(cmd,name,out,interrupted,timeout=600):
 """Own client and container lifetime even on timeout, SIGTERM, or errors."""
 result={'returncode':None,'timed_out':False,'interrupted':None,'client_exited_while_container_running':False,
         'container_stopped_verified':False,'errors':[],'container_id':None,'container_exit_code':None,'container_running':None}
 client=None;state=None
 def note(where,e):result['errors'].append({'where':where,'error':str(e)})
 def inspect():
  try:
   p=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=20)
   if p.returncode:return None
   xs=json.loads(p.stdout)
   if len(xs)!=1:raise ValueError('bad inspect result')
   return xs[0]
  except Exception as e:note('inspect',e);return None
 def stop():
  for method in ('stop','kill'):
   try:
    command=['docker','stop','--time','5',name] if method=='stop' else ['docker','kill',name]
    p=subprocess.run(command,capture_output=True,text=True,timeout=20,check=False)
    if p.returncode:note(method,'exit '+str(p.returncode))
   except Exception as e:note(method,e)
   obj=inspect()
   if obj is not None and obj.get('State',{}).get('Running') is False:return obj
  return None
 with (out/'stdout.log').open('x') as stdout,(out/'stderr.log').open('x') as stderr:
  try:
   client=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,start_new_session=True);begin=time.monotonic()
   while client.poll() is None:
    if interrupted[0] is not None or time.monotonic()-begin>=timeout:
     result['interrupted']=interrupted[0];result['timed_out']=interrupted[0] is None;state=stop();break
    time.sleep(1)
   try:result['returncode']=client.wait(timeout=30)
   except Exception as e:note('client_wait',e)
   if state is None:state=inspect()
   if state is not None and state.get('State',{}).get('Running') is True:
    result['client_exited_while_container_running']=True;state=stop()
  except BaseException as e:note('supervision',repr(e))
  finally:
   if state is None or state.get('State',{}).get('Running') is not False:state=stop()
   if client is not None:
    try:
     if client.poll() is None:client.kill()
     result['returncode']=client.wait(timeout=10)
    except Exception as e:note('client_cleanup',e)
   if state is not None:
    st=state.get('State',{});result['container_id']=state.get('Id');result['container_running']=st.get('Running');result['container_exit_code']=st.get('ExitCode');result['container_pid']=st.get('Pid')
    result['container_stopped_verified']=bool(state.get('Id') and st.get('Running') is False and st.get('Pid')==0)
   if interrupted[0] is not None:result['interrupted']=interrupted[0]
 return result

def main():
 approval=gate.execution_gate(preflight=True);out=HERE/'preflight_001';out.mkdir(exist_ok=False)
 old=BASE/'e4_panel_v2/panel_003/launch.json';cmd=json.loads(old.read_text())['command'][:];name='eadp-e2e3full-preflight-001';cmd[cmd.index('--name')+1]=name
 for device in ('/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'):
  i=cmd.index(device);del cmd[i-1:i+1]
 cmd[cmd.index('CUDA_VISIBLE_DEVICES=0')]='CUDA_VISIBLE_DEVICES='
 index=cmd.index(str(BASE/'e4_panel_v2/panel_worker.py'));cmd=cmd[:index]+[str(HERE/'panel_worker.py'),'--execute','--preflight-only',
  '--e1-acceptance',str(BASE/'e1/e1full001_completion.json'),'--e1-started',str(BASE/'e1/monitorverifye1full001/started.json'),
  '--e1-terminal',str(BASE/'e1/monitorverifye1full001/terminal.json'),'--out',str(out)]
 launch={'command':cmd,'upstream_launch_sha256':sha(old),'approval':approval}
 with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 interrupted=[None]
 def handle(sig,_frame):interrupted[0]=sig
 for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,handle)
 start=time.time();outcome=supervise_container(cmd,name,out,interrupted)
 try:result=json.loads((out/'stdout.log').read_text().strip().splitlines()[-1])
 except Exception:result=None
 passed=(not outcome['timed_out'] and outcome['interrupted'] is None and not outcome['errors'] and outcome['returncode']==0 and outcome['container_stopped_verified'] and outcome['container_exit_code']==0 and not outcome['client_exited_while_container_running'] and isinstance(result,dict) and result.get('status')=='preflight_passed' and result.get('cuda_initialized') is False and result.get('model_loaded') is False and result.get('predictions') is False and not (out/'frozen.json').exists() and not (out/'data').exists())
 proof={'status':'passed' if passed else 'failed',**outcome,'worker_sha256':sha(HERE/'panel_worker.py'),'preflight_source_sha256':sha(__file__),'launch_sha256':sha(out/'launch.json'),'stdout_sha256':sha(out/'stdout.log'),'stderr_sha256':sha(out/'stderr.log'),'result':result,'started_unix':start,'finished_unix':time.time()}
 with (HERE/'container_preflight_001.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':proof['status'],'returncode':outcome['returncode']}),flush=True);return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
