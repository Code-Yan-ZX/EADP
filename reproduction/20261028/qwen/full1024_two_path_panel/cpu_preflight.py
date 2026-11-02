"""Create a no-device container and verify actual binds before and after CPU guards."""
import json
import subprocess
import sys
import common as c
sys.path.insert(0,str(c.PREV))
from run_panel import container_command
from namespace_guards import verify_mounts

def main():
    p=c.read(c.PLAN);c.plan_checks(p)
    target=c.HERE/'CPU_preflight_container_record.json';assert not target.exists()
    old=c.read(c.PREV/'run_plan.json')
    command=container_command(old,[str(c.HERE/'cpu_container_check.py')],False,'eadp-qwen-1024-paths-cpu-preflight')
    assert '--device' not in command
    command.remove('--rm');command[1]='create'
    record=dict(command=command,plan_sha256=c.sha(c.PLAN),GPU_requested=False)
    created=subprocess.run(command,capture_output=True,text=True)
    record.update(create_exit=created.returncode,create_stdout=created.stdout,create_stderr=created.stderr);c.save(target,record)
    assert created.returncode==0
    cid=created.stdout.strip();record['container_id']=cid
    try:
        inspected=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
        record['mount_proof']=verify_mounts(old,inspected,gpu=False)
        assert inspected['Config']['Entrypoint']==[old['python']] and inspected['Config']['Cmd']==[str(c.HERE/'cpu_container_check.py')]
        assert 'CUDA_VISIBLE_DEVICES=' in inspected['Config']['Env']
        c.save(target,record)
        run=subprocess.run(['docker','start','--attach',cid],capture_output=True,text=True)
        state=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]['State']
        record.update(actual_exit=run.returncode,stdout=run.stdout,stderr=run.stderr,terminal_state=state);c.save(target,record)
        assert run.returncode==state['ExitCode']==0 and not state['Running']
        actual=json.loads(run.stdout);assert actual['status']=='passed' and actual['plan_sha256']==c.sha(c.PLAN)
        assert not actual['torch_imported'] and not actual['GPU_devices_present']
        record['result']=actual;record['status']='passed'
    finally:
        removed=subprocess.run(['docker','rm',cid],capture_output=True,text=True)
        record.update(remove_exit=removed.returncode,remove_stderr=removed.stderr);c.save(target,record)
    assert record['status']=='passed' and record['remove_exit']==0
    c.save(c.HERE/'CPU_preflight_pass.json',dict(status='passed',plan_sha256=c.sha(c.PLAN),
        container_record_sha256=c.sha(target),torch_imported=False,GPU_requested=False,model_loaded=False))
    print(json.dumps(dict(status='passed',GPU_requested=False,plan_sha256=c.sha(c.PLAN))))
if __name__=='__main__':main()
