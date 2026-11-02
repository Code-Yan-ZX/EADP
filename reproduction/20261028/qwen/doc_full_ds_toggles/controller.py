"""One explicit root allocation; single model, no waiting queue or retry."""
import argparse,json,subprocess,sys,time
from pathlib import Path
import common as c
sys.path.insert(0,str(c.OLD))
from run_panel import container_command,gpu_pids,identity
from namespace_guards import verify_mounts

def run(note_path):
 p=c.read(c.PLAN);c.check(p);note=c.read(note_path)
 assert note['authorization']=='explicit_root_gpu_allocation' and note['run_id']==c.RUN_ID
 assert note['plan_sha256']==c.sha(c.PLAN) and note['GPU_device']==0 and note['GPU_slot'] in ('A','B')
 assert note['single_attempt'] and not note['automatic_retry'] and note['maximum_concurrent_models_including_loading']==1
 assert note['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 pre=c.read(c.HERE/'CPU_preflight_pass.json');assert pre['status']=='passed' and pre['plan_sha256']==c.sha(c.PLAN)
 assert note['CPU_preflight_sha256']==c.sha(c.HERE/'CPU_preflight_pass.json')
 tests=c.read(c.HERE/'cpu_tests.json');assert tests['success'] and tests['plan_sha256']==c.sha(c.PLAN)
 assert note['CPU_tests_sha256']==c.sha(c.HERE/'cpu_tests.json')
 assert pre['container_record_sha256']==c.sha(c.HERE/'CPU_preflight_container_record.json')
 hook=c.read(c.HERE/'cpu_torch_hook_test_author.json');assert hook['success'] and hook['plan_sha256']==c.sha(c.PLAN)
 hook_container=c.read(c.HERE/'cpu_torch_hook_container_record.json');assert hook_container['status']=='passed' and hook_container['actual_exit']==hook_container['remove_exit']==0 and hook_container['result_sha256']==c.sha(c.HERE/'cpu_torch_hook_test_author.json') and hook_container['plan_sha256']==c.sha(c.PLAN) and not hook_container['GPU_requested'] and not hook_container['terminal_state']['Running']
 assert note['CPU_torch_hook_sha256']==c.sha(c.HERE/'cpu_torch_hook_test_author.json') and hook['torch_package_version']==c.read(c.OLD/'run_plan.json')['expected_versions']['torch']
 others=note['allowed_other_gpu_processes'];assert len(others)==0 and gpu_pids()==sorted(x['pid'] for x in others)
 for x in others:
  actual=identity(x['pid']);assert actual and all(actual[k]==x[k] for k in ('pid','start_ticks','boot_id'))
 assert not (c.HERE/'execution').exists() and not (c.HERE/'state.json').exists()
 old=c.read(c.OLD/'run_plan.json');name='eadp-qwen-doc-ds-toggles'
 cmd=container_command(old,[str(c.HERE/'worker.py'),'--allocation-note',str(note_path)],True,name)
 controller=identity(__import__('os').getpid())
 with (c.HERE/'launch.claim.json').open('x') as f:json.dump(dict(controller=controller,plan_sha256=c.sha(c.PLAN),allocation_sha256=c.sha(note_path)),f,indent=2)
 state=dict(status='starting',controller=controller,plan_sha256=c.sha(c.PLAN),allocation_sha256=c.sha(note_path),command=cmd,automatic_retry=False)
 c.save(c.HERE/'state.json',state);child=None
 try:
  with (c.HERE/'execution.log').open('x') as log:
   child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);state.update(status='running',docker_client=identity(child.pid));c.save(c.HERE/'state.json',state)
   while child.poll() is None:
    if 'container_identity' not in state:
     probe=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
     if probe.returncode==0:
      info=json.loads(probe.stdout)[0];pid=info['State']['Pid'];actual=identity(pid) if pid else None
      if actual:
       assert info['Image']==old['container_image'];group=Path('/proc',str(pid),'cgroup').read_text();assert info['Id'] in group
       state['container_identity']=dict(container_id=info['Id'],host_init=actual,host_cgroup=group)
       state['mount_proof']=verify_mounts(old,info,gpu=True);c.save(c.HERE/'state.json',state)
    time.sleep(1)
   code=child.wait()
  state.update(status='native_exited_validation_pending',native_exit_code=code);c.save(c.HERE/'state.json',state);assert code==0
  original=state['container_identity']['host_init'];now=identity(original['pid'])
  assert now is None or now['start_ticks']!=original['start_ticks'];assert original['pid'] not in gpu_pids()
  probe=subprocess.run(['docker','inspect',state['container_identity']['container_id']],capture_output=True,text=True)
  assert probe.returncode!=0 or not json.loads(probe.stdout)[0]['State']['Running']
  state.update(original_container_exited=True,GPU_pid_absent=True);c.save(c.HERE/'state.json',state)
  c.check(p)
  verify_cmd=container_command(old,[str(c.HERE/'verify_generation.py')],False,'eadp-qwen-doc-ds-toggles-verifier')
  assert '--device' not in verify_cmd
  with (c.HERE/'generation_verification.log').open('x') as log:
   ver=subprocess.run(verify_cmd,stdout=log,stderr=subprocess.STDOUT)
  state['CPU_verifier_exit_code']=ver.returncode;c.save(c.HERE/'state.json',state);assert ver.returncode==0
  report=c.read(c.HERE/'execution/generation_verification.json');assert report['success'] and report['n_generations']==10698
  c.save(c.HERE/'execution/release.json',dict(success=True,plan_sha256=c.sha(c.PLAN),allocation_sha256=c.sha(note_path),
   generation_verification_sha256=c.sha(c.HERE/'execution/generation_verification.json'),native_container_exited=True,
   controller_actual_exit_requires_root_pty_proof=True,GT_or_scoring_started=False))
  state.update(status='complete',release_sha256=c.sha(c.HERE/'execution/release.json'));c.save(c.HERE/'state.json',state)
 except BaseException as error:
  state.update(status='terminal_failure_no_retry',error=repr(error));c.save(c.HERE/'state.json',state)
  if child is not None and child.poll() is None:
   stopped=subprocess.run(['docker','stop','--time','10',name],capture_output=True,text=True)
   state['emergency_stop']=dict(exit_code=stopped.returncode,stdout=stopped.stdout,stderr=stopped.stderr)
   try:state['docker_client_exit_after_stop']=child.wait(timeout=30)
   except subprocess.TimeoutExpired:state['docker_client_still_running']=True
  probe=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
  state['container_after_failure']=dict(inspect_exit=probe.returncode,stdout=probe.stdout,stderr=probe.stderr)
  state['failure_does_not_prove_GPU_released']=True;c.save(c.HERE/'state.json',state);c.save(c.HERE/'failure.json',state);raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['check','run']);ap.add_argument('--allocation-note',type=Path);args=ap.parse_args()
 if args.action=='check':c.check(c.read(c.PLAN));print(json.dumps(dict(status='CPU_only',GPU_started=False)))
 else:assert args.allocation_note;run(args.allocation_note.resolve())
