"""Execute synthetic hook test in the locked author container without devices."""
import json,subprocess,sys
import common as c
sys.path.insert(0,str(c.OLD))
from run_panel import container_command
from namespace_guards import verify_mounts

def main():
 p=c.read(c.PLAN);c.check(p);record_path=c.HERE/'cpu_torch_hook_container_record.json';assert not record_path.exists()
 old=c.read(c.OLD/'run_plan.json');command=container_command(old,[str(c.HERE/'cpu_torch_hook_test.py'),'--author'],False,'eadp-qwen-doc-ds-toggles-hook-cpu')
 assert '--device' not in command;command.remove('--rm');command[1]='create'
 record=dict(command=command,plan_sha256=c.sha(c.PLAN),GPU_requested=False)
 create=subprocess.run(command,capture_output=True,text=True);record.update(create_exit=create.returncode,create_stdout=create.stdout,create_stderr=create.stderr);c.save(record_path,record);assert create.returncode==0
 cid=create.stdout.strip();record['container_id']=cid
 try:
  info=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0];record['mount_proof']=verify_mounts(old,info,gpu=False)
  assert 'CUDA_VISIBLE_DEVICES=' in info['Config']['Env'];c.save(record_path,record)
  result=subprocess.run(['docker','start','--attach',cid],capture_output=True,text=True)
  state=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]['State']
  record.update(actual_exit=result.returncode,stdout=result.stdout,stderr=result.stderr,terminal_state=state);c.save(record_path,record)
  assert result.returncode==state['ExitCode']==0 and not state['Running']
  result=c.read(c.HERE/'cpu_torch_hook_test_author.json');assert result['success'] and result['plan_sha256']==c.sha(c.PLAN)
  assert result['torch_package_version']==old['expected_versions']['torch'] and not result['CUDA_initialized']
  record['result_sha256']=c.sha(c.HERE/'cpu_torch_hook_test_author.json');record['status']='passed'
 finally:
  rm=subprocess.run(['docker','rm',cid],capture_output=True,text=True);record['remove_exit']=rm.returncode;c.save(record_path,record)
 assert record['status']=='passed' and record['remove_exit']==0
 print(json.dumps(dict(status='passed',plan_sha256=c.sha(c.PLAN),GPU_requested=False)))
if __name__=='__main__':main()
