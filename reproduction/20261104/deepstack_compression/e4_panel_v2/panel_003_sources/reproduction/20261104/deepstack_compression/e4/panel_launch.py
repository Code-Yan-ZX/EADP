"""Future hard-bounded E4 full DEV Docker launcher, locked pending review."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from identity_worker import HERE,PLAN,sha
from panel_gate import validated_dev,validate_smoke_gate

PANEL_APPROVAL_PATH=HERE/'panel_approval.json'

def check_panel_approval():
    if not PANEL_APPROVAL_PATH.is_file():raise ValueError('independent full-panel review is not complete')
    bundle=json.loads(PANEL_APPROVAL_PATH.read_text())
    if set(bundle)!={'cpu_proof','review_proof'}:raise ValueError('incomplete panel approval')
    from smoke_launch import check_approval
    smoke=check_approval()
    cpu=Path(bundle['cpu_proof']);review=Path(bundle['review_proof'])
    c=json.loads(cpu.read_text());r=json.loads(review.read_text())
    if (c.get('status')!='passed' or c.get('returncode')!=0 or c.get('timeout') is not False or
        r.get('status')!='passed' or r.get('independent_review') is not True or r.get('cpu_proof_sha256')!=sha(cpu)):
        raise ValueError('panel CPU/review proof failed')
    paths=[HERE/x for x in ('panel_worker.py','panel_launch.py','panel_gate.py','panel_verify.py','test_panel.py')]
    for path in paths:
        digest=sha(path);name=str(path)
        if (c.get('source_sha256_before',{}).get(name)!=digest or c.get('source_sha256_after',{}).get(name)!=digest or
            r.get('source_sha256',{}).get(name)!=digest):raise ValueError('unreviewed panel source: '+name)
    return {'smoke':smoke,'approval_manifest':{'path':str(PANEL_APPROVAL_PATH.resolve()),'sha256':sha(PANEL_APPROVAL_PATH)},
            'cpu':{'path':str(cpu.resolve()),'sha256':sha(cpu)},
            'review':{'path':str(review.resolve()),'sha256':sha(review)}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--timeout',type=int,required=True)
    for name in ('e1-acceptance','e1-started','e1-terminal','smoke-dir','smoke-proof','smoke-started','smoke-terminal'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if not a.run_id.isalnum() or not 60<=a.timeout<=172800:raise ValueError('invalid id or deadline')
    approval=check_panel_approval()
    validated_dev(a.e1_acceptance,a.e1_started,a.e1_terminal)
    validate_smoke_gate(a.e1_acceptance,a.smoke_dir,a.smoke_proof,a.smoke_started,a.smoke_terminal)
    plan=json.loads(PLAN.read_text())
    for group in ('source_sha256','implementation_sha256'):
        for path,digest in plan[group].items():
            if sha(path)!=digest:raise ValueError('pre-launch source changed: '+path)
    sys.path.insert(0,str(PLAN.parent))
    from run_panel import container_command,gpu_pids,identity
    if gpu_pids():raise ValueError('GPU occupied')
    out=HERE/('panel_'+a.run_id);out.mkdir(exist_ok=False)
    name='eadp-e4full-'+a.run_id+'-'+str(os.getpid());worker=HERE/'panel_worker.py'
    args=[str(worker),'--execute']
    for field in ('e1_acceptance','e1_started','e1_terminal','smoke_dir','smoke_proof','smoke_started','smoke_terminal'):
        args += ['--'+field.replace('_','-'),str(getattr(a,field).resolve())]
    args += ['--out',str(out)]
    command=container_command(plan,args,True,name)
    if command.count('--rm')!=1:raise ValueError('unexpected Docker cleanup policy')
    command.remove('--rm')
    controller=identity(os.getpid());launch={'command':command,'controller':controller,'timeout_seconds':a.timeout,
        'approval_bundle':approval,'launcher_sha256':sha(__file__),'worker_sha256':sha(worker),'started_unix':time.time()}
    with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    interrupted=[None]
    def stop(signum,_frame):interrupted[0]=signum
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
    result={'returncode':None,'timed_out':False,'interrupted':None,'container_stopped_verified':False,
            'client_exited_while_container_running':False,'container_identity':None}
    with (out/'launcher.log').open('x') as log:
        client=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);begin=time.monotonic()
        try:
            while client.poll() is None:
                if result['container_identity'] is None:
                    probe=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=5)
                    if probe.returncode==0:
                        state=json.loads(probe.stdout)[0];pid=state['State']['Pid']
                        if pid:result['container_identity']={'container_id':state['Id'],'image':state['Image'],'host_init':identity(pid)}
                if interrupted[0] is not None or time.monotonic()-begin>=a.timeout:
                    result['interrupted']=interrupted[0];result['timed_out']=interrupted[0] is None
                    subprocess.run(['docker','stop','--time','10',name],stdout=log,stderr=log,timeout=30,check=False)
                    try:client.wait(timeout=30)
                    except subprocess.TimeoutExpired:client.kill();client.wait()
                    break
                time.sleep(1)
            result['returncode']=client.wait()
            probe=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10)
            if probe.returncode==0 and json.loads(probe.stdout)[0]['State']['Running']:
                result['client_exited_while_container_running']=True
                subprocess.run(['docker','stop','--time','10',name],stdout=log,stderr=log,timeout=30,check=False)
                probe=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10)
            if probe.returncode!=0 or json.loads(probe.stdout)[0]['State']['Running']:
                raise RuntimeError('container stop not confirmed')
            if result['container_identity'] is None or json.loads(probe.stdout)[0]['Id']!=result['container_identity']['container_id']:
                raise RuntimeError('container identity mismatch')
            result['container_exit_code']=json.loads(probe.stdout)[0]['State']['ExitCode']
            if result['container_exit_code']!=result['returncode']:raise RuntimeError('client/container exit mismatch')
            result['container_stopped_verified']=True
        finally:
            if not result['container_stopped_verified']:
                subprocess.run(['docker','stop','--time','10',name],stdout=log,stderr=log,timeout=30,check=False)
            if client.poll() is None:client.kill();client.wait()
            result.update(controller=controller,name=name,finished_unix=time.time())
            if interrupted[0] is not None:result['interrupted']=interrupted[0]
            try:result['gpu_pids_after']=gpu_pids()
            except Exception:result['gpu_pids_after']=None
            with (out/'exit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    if (result['returncode']!=0 or result['timed_out'] or result['interrupted'] is not None or
        result['client_exited_while_container_running'] or result['gpu_pids_after']!=[]):return 1
    return 0

if __name__=='__main__':sys.exit(main())
