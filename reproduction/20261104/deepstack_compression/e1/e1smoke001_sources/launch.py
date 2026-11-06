"""Bounded Docker launcher for E1 legacy anchor; no automatic retries."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from runner import ROOT, PROJECT, E0, PLAN, validate_gate, validate_legacy_gate, verified_inputs, sha

HERE=Path(__file__).resolve().parent
OLD=PLAN.parent
sys.path.insert(0,str(OLD))
from run_panel import container_command, gpu_pids, identity


def exit_code(result):
    if result['timed_out']: return 124
    if result['interrupted'] is not None: return 128+result['interrupted']
    if result.get('client_exited_while_container_running'): return 1
    return result['returncode'] if result['returncode'] is not None else 1


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-id',required=True); p.add_argument('--e0-proof',type=Path,required=True)
    p.add_argument('--manifest-dir',type=Path,required=True); p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--input-proof',type=Path,required=True); p.add_argument('--timeout',type=int,default=3600)
    p.add_argument('--mode',choices=('legacy','smoke'),default='legacy')
    p.add_argument('--limit',type=int,choices=(1,16))
    a=p.parse_args()
    if not a.run_id.isalnum() or not 60<=a.timeout<=86400: raise ValueError('invalid id or bounded timeout')
    for name in ('e0_proof','manifest_dir','inputs','input_proof'):
        setattr(a,name,getattr(a,name).resolve())
    validate_gate(a.e0_proof); verified_inputs(a.manifest_dir,a.inputs,a.input_proof)
    if a.mode=='legacy' and a.limit is None: raise ValueError('legacy requires --limit 1 or 16')
    if a.mode=='smoke': validate_legacy_gate()
    if gpu_pids(): raise RuntimeError('GPU compute process already exists')
    out=HERE/a.run_id
    if out.exists() or (HERE/(a.run_id+'_launch.json')).exists(): raise ValueError('refusing overwrite')
    name='eadp-e1legacy-'+a.run_id+'-'+str(os.getpid())
    plan=json.loads(PLAN.read_text())
    worker=HERE/('legacy_anchor.py' if a.mode=='legacy' else 'runner.py')
    worker_args=[str(worker),'--execute','--e0-proof',str(a.e0_proof), '--manifest-dir',str(a.manifest_dir),
                 '--inputs',str(a.inputs),'--input-proof',str(a.input_proof),'--out',str(out)]
    worker_args += ['--limit',str(a.limit)] if a.mode=='legacy' else ['--stage','smoke']
    cmd=container_command(plan,worker_args,True,name)
    files=[HERE/'legacy_anchor.py',HERE/'position_adapter.py',HERE/'runner.py',HERE/'contract.py',HERE/'launch.py',E0/'adapter.py']
    launch={'command':cmd,'controller':identity(os.getpid()),'started_unix':time.time(),
            'prior_gpu_pids':gpu_pids(),'timeout_seconds':a.timeout,'source_sha256':{str(x):sha(x) for x in files}}
    with (HERE/(a.run_id+'_launch.json')).open('x') as f: json.dump(launch,f,indent=2); f.write('\n')
    interrupted=[None]
    def stop_signal(signum,_frame): interrupted[0]=signum
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP): signal.signal(sig,stop_signal)
    container_identity=None
    result={'returncode':None,'timed_out':False,'interrupted':None,'client_exited_while_container_running':False}
    with (HERE/(a.run_id+'.log')).open('x') as log:
        # Keep the Docker client outside the external supervisor's process
        # group: a supervisor TERM reaches this launcher, which stops and
        # waits for the real container before returning.
        process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        begin=time.monotonic()
        try:
            while process.poll() is None:
                if container_identity is None:
                    probe=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=5)
                    if probe.returncode==0:
                        item=json.loads(probe.stdout)[0]; hostpid=item['State']['Pid']
                        if hostpid:
                            container_identity={'container_id':item['Id'],'image':item['Image'],'host_init':identity(hostpid),'observed_unix':time.time()}
                            with (HERE/(a.run_id+'_container_identity.json')).open('x') as f: json.dump(container_identity,f,indent=2)
                if time.monotonic()-begin>=a.timeout or interrupted[0] is not None:
                    result['timed_out']=interrupted[0] is None; result['interrupted']=interrupted[0]
                    subprocess.run(['docker','stop','--time','10',name],check=False,stdout=log,stderr=log,timeout=30)
                    try: process.wait(timeout=30)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
                    break
                time.sleep(1)
            result['returncode']=process.wait()
            # A detached/abruptly exiting Docker client is not proof that the
            # daemon-owned container stopped. Inspect and stop if necessary.
            check=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10)
            if check.returncode==0:
                item=json.loads(check.stdout)[0]
                if item['State']['Running']:
                    result['client_exited_while_container_running']=True
                    subprocess.run(['docker','stop','--time','10',name],check=False,stdout=log,stderr=log,timeout=30)
                    check=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10)
                if check.returncode==0 and json.loads(check.stdout)[0]['State']['Running']:
                    raise RuntimeError('container still running after stop')
            if check.returncode!=0 and 'No such object' not in check.stderr and 'No such container' not in check.stderr:
                raise RuntimeError('container stop status could not be confirmed: '+check.stderr.strip())
            result['container_stopped_verified']=True
        except BaseException:
            subprocess.run(['docker','stop','--time','10',name],check=False,stdout=log,stderr=log,timeout=30)
            if process.poll() is None: process.kill(); process.wait()
            raise
    result.update(container_identity=container_identity,finished_unix=time.time(),gpu_pids_after=gpu_pids(),name=name,controller=identity(os.getpid()))
    with (HERE/(a.run_id+'_exit.json')).open('x') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result),flush=True)
    return exit_code(result)


if __name__=='__main__': raise SystemExit(main())
