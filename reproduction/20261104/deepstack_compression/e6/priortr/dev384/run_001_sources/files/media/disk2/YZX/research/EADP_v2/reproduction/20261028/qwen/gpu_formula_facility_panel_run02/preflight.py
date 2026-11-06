"""One explicit CPU-only Docker preflight; never requests a GPU device."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diagnose import PLAN, validate, sha, save
from run_panel import container_command
from namespace_guards import verify_mounts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-cpu-container', action='store_true')
    args=parser.parse_args();plan=json.loads(PLAN.read_text());validate(plan)
    if not args.run_cpu_container:
        print(json.dumps(dict(status='CPU_preflight_prepared_only', GPU_requested=False, container_started=False)))
        return
    target=HERE/'CPU_preflight_container_record.json'
    assert not target.exists()
    command=container_command(plan,[str(HERE/'cpu_preflight.py')],False,plan['container_name']+'-cpu-preflight')
    assert '--device' not in command
    command.remove('--rm');command[1]='create'
    record=dict(create_command=command,run_plan_sha256=sha(PLAN),GPU_requested=False,automatic_GPU_retry=False)
    created=subprocess.run(command,capture_output=True,text=True)
    record.update(create_exit_code=created.returncode,create_stdout=created.stdout,create_stderr=created.stderr)
    save(target,record);assert created.returncode==0,created.stderr
    cid=created.stdout.strip();record['container_id']=cid
    try:
        before=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
        record['before_start_mount_proof']=verify_mounts(plan,before,gpu=False)
        record['actual_image']=before['Image']
        record['actual_entrypoint']=before['Config']['Entrypoint']
        record['actual_cmd']=before['Config']['Cmd']
        assert before['State']['Pid']==0 and not before['State']['Running']
        assert before['Config']['Entrypoint']==[plan['python']]
        assert before['Config']['Cmd']==[str(HERE/'cpu_preflight.py')]
        assert 'CUDA_VISIBLE_DEVICES=' in before['Config']['Env']
        save(target,record)
        started=subprocess.run(['docker','start','--attach',cid],capture_output=True,text=True)
        after=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
        record.update(start_attach_exit_code=started.returncode,stdout=started.stdout,stderr=started.stderr,terminal_state=after['State'])
        save(target,record)
        assert started.returncode==0 and after['State']['ExitCode']==0 and not after['State']['Running']
        result=json.loads(started.stdout)
        assert result['status']=='all_frozen_guards_passed_in_CPU_container'
        assert result['run_plan_sha256']==sha(PLAN) and not result['torch_imported'] and not result['GPU_devices_present']
        record['result']=result;record['status']='passed'
    finally:
        removed=subprocess.run(['docker','rm',cid],capture_output=True,text=True)
        record.update(remove_exit_code=removed.returncode,remove_stdout=removed.stdout,remove_stderr=removed.stderr)
        save(target,record)
    assert record['status']=='passed' and record['remove_exit_code']==0
    save(HERE/'CPU_preflight_pass.json',dict(status='passed',run_plan_sha256=sha(PLAN),
        container_record_sha256=sha(target),torch_imported=False,GPU_requested=False,model_loaded=False,
        actual_bind_sources_readonly_verified=True,all_model_numeric_stats_unchanged=True))
    print(json.dumps(dict(status='passed',run_plan_sha256=sha(PLAN),GPU_requested=False,torch_imported=False)))


if __name__=='__main__':main()
