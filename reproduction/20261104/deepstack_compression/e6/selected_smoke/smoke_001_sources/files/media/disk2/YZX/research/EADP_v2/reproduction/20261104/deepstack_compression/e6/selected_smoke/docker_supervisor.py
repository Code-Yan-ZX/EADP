"""Process-level Docker supervisor adapted from reviewed M3 smoke; no launch authority."""
import json,os,subprocess,time

def supervise(command,name,out,controller,timeout,gpu_pids,identity,interrupted):
    """Preserve exit evidence even if Docker/client operations fail; never assume unknown state is stopped."""
    result={'returncode':None,'timed_out':False,'interrupted':None,'container_stopped_verified':False,
            'client_exited_while_container_running':False,'container_identity':None,'errors':[]}
    errors=result['errors'];client=None;state=None
    def note(stage,exc):errors.append(stage+': '+str(exc))
    def inspect(allow_missing=False):
        try:
            q=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10)
            if q.returncode:
                if not allow_missing:note('inspect',f'exit {q.returncode}')
                return None
            values=json.loads(q.stdout)
            if len(values)!=1:raise ValueError('unexpected inspect result')
            return values[0]
        except Exception as e:note('inspect',e);return None
    def stop_and_kill(log):
        try:
            q=subprocess.run(['docker','stop','--time','10',name],stdout=log,stderr=log,timeout=30,check=False)
            if q.returncode:note('stop',f'exit {q.returncode}')
        except Exception as e:note('stop',e)
        v=inspect()
        if v is None or v.get('State',{}).get('Running') is not False:
            try:
                q=subprocess.run(['docker','kill',name],stdout=log,stderr=log,timeout=15,check=False)
                if q.returncode:note('kill',f'exit {q.returncode}')
            except Exception as e:note('kill',e)
            v=inspect()
        return v
    with (out/'launcher.log').open('x') as log:
        try:
            client=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);begin=time.monotonic()
            while client.poll() is None:
                if result['container_identity'] is None:
                    v=inspect(allow_missing=True)
                    if v is not None:
                        pid=v['State']['Pid']
                        if pid:result['container_identity']={'container_id':v['Id'],'image':v['Image'],'host_init':identity(pid)}
                if interrupted[0] is not None or time.monotonic()-begin>=timeout:
                    result['interrupted']=interrupted[0];result['timed_out']=interrupted[0] is None
                    state=stop_and_kill(log);break
                time.sleep(1)
            try:result['returncode']=client.wait(timeout=30)
            except Exception as e:note('client wait',e)
            if state is None:state=inspect()
            if state is not None and state.get('State',{}).get('Running') is True:
                result['client_exited_while_container_running']=True;state=stop_and_kill(log)
        except BaseException as e:note('supervision',repr(e))
        finally:
            if state is None or state.get('State',{}).get('Running') is not False:state=stop_and_kill(log)
            if client is not None:
                try:
                    if client.poll() is None:client.kill()
                    result['returncode']=client.wait(timeout=10)
                except Exception as e:note('client cleanup',e)
            try:
                ident=result['container_identity']
                if state is None or state.get('State',{}).get('Running') is not False:raise ValueError('container stop not confirmed')
                if ident is None or state.get('Id')!=ident['container_id']:raise ValueError('container identity mismatch')
                result['container_exit_code']=state['State']['ExitCode']
                if result['container_exit_code']!=result['returncode']:raise ValueError('client/container exit mismatch')
                result['container_stopped_verified']=True
            except Exception as e:note('termination',e)
            result.update(controller=controller,name=name,finished_unix=time.time())
            if interrupted[0] is not None:result['interrupted']=interrupted[0]
            try:result['gpu_pids_after']=gpu_pids()
            except Exception as e:result['gpu_pids_after']=None;note('gpu snapshot',e)
            with (out/'exit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return result

