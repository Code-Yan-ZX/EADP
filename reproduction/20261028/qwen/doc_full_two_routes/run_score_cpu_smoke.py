"""Launch only synthetic author score in a no-device container."""
import json,subprocess,sys
import common as c
sys.path.insert(0,str(c.OLD));from run_panel import container_command
old=c.read(c.OLD/'run_plan.json');command=container_command(old,[str(c.HERE/'score_cpu_smoke.py')],False,'eadp-qwen-doc-score-synthetic')
assert '--device' not in command
run=subprocess.run(command,capture_output=True,text=True)
record=dict(actual_exit=run.returncode,stdout=run.stdout,stderr=run.stderr,GPU_requested=False,command=command)
c.save(c.HERE/'score_cpu_smoke_result.json',record)
assert run.returncode==0
result=json.loads(run.stdout.splitlines()[-1]);assert result['success'] and not result['CUDA_initialized'] and not result['real_GT_read']
print(json.dumps(result))
