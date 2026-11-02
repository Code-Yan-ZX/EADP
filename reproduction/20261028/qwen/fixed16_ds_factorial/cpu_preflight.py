"""One no-device Docker container, actual read-only mounts and frozen guards."""
import json,subprocess,sys
import common as c
sys.path.insert(0,str(c.OLD));from run_panel import container_command
from namespace_guards import verify_mounts

def main():
 p=c.read(c.PLAN);c.check(p);target=c.HERE/'CPU_preflight_container_record.json';assert not target.exists()
 old=c.read(c.OLD/'run_plan.json');cmd=container_command(old,[str(c.HERE/'cpu_container_check.py')],False,'eadp-qwen-fixed16-ds-cpu')
 assert '--device' not in cmd;cmd.remove('--rm');cmd[1]='create'
 record=dict(command=cmd,plan_sha256=c.sha(c.PLAN),GPU_requested=False)
 create=subprocess.run(cmd,capture_output=True,text=True);record.update(create_exit=create.returncode,stdout=create.stdout,stderr=create.stderr);c.save(target,record)
 assert create.returncode==0;cid=create.stdout.strip();record['container_id']=cid
 try:
  info=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0];record['mount_proof']=verify_mounts(old,info,gpu=False)
  assert info['Config']['Entrypoint']==[old['python']] and info['Config']['Cmd']==[str(c.HERE/'cpu_container_check.py')]
  assert 'CUDA_VISIBLE_DEVICES=' in info['Config']['Env'];c.save(target,record)
  run=subprocess.run(['docker','start','--attach',cid],capture_output=True,text=True)
  state=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]['State']
  record.update(actual_exit=run.returncode,run_stdout=run.stdout,run_stderr=run.stderr,terminal_state=state);c.save(target,record)
  assert run.returncode==state['ExitCode']==0 and not state['Running']
  result=json.loads(run.stdout);assert result['status']=='passed' and result['plan_sha256']==c.sha(c.PLAN)
  assert not result['torch_imported'] and not result['GPU_devices_present'];record['status']='passed'
 finally:
  rm=subprocess.run(['docker','rm',cid],capture_output=True,text=True);record['remove_exit']=rm.returncode;c.save(target,record)
 assert record['status']=='passed' and record['remove_exit']==0
 c.save(c.HERE/'CPU_preflight_pass.json',dict(status='passed',plan_sha256=c.sha(c.PLAN),container_record_sha256=c.sha(target),GPU_requested=False,model_loaded=False))
 print(json.dumps(dict(status='passed',plan_sha256=c.sha(c.PLAN),GPU_requested=False)))
if __name__=='__main__':main()
