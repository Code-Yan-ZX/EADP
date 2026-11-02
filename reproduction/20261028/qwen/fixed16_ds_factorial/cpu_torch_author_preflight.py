"""Run synthetic hook test in the author's no-device container only."""
import json,subprocess,sys
import common as c
sys.path.insert(0,str(c.OLD));from run_panel import container_command
p=c.read(c.PLAN);c.check(p)
old=c.read(c.OLD/'run_plan.json');target=c.HERE/'cpu_torch_author_container_record.json';assert not target.exists()
cmd=container_command(old,[str(c.HERE/'cpu_torch_hook_test.py'),'--author'],False,'eadp-qwen-fixed16-ds-hooks-cpu')
assert '--device' not in cmd
run=subprocess.run(cmd,capture_output=True,text=True);c.save(target,dict(command=cmd,actual_exit=run.returncode,stdout=run.stdout,stderr=run.stderr,GPU_requested=False))
assert run.returncode==0
result=c.read(c.HERE/'cpu_torch_hook_test_author.json')
assert result['success'] and result['torch_package_version']==old['expected_versions']['torch'] and result['plan_sha256']==c.sha(c.PLAN)
assert not result['CUDA_initialized'] and not result['model_loaded']
print(json.dumps(dict(status='passed',torch=result['torch_package_version'],GPU_requested=False)))
