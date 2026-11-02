"""No-device container tests of draft capture code; never allocate a GPU."""
import hashlib,json,subprocess,sys
import common as c
from pathlib import Path
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'gpu_formula_facility_panel_run02'
sys.path.insert(0,str(OLD))
from run_panel import container_command
from namespace_guards import verify_mounts
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 plan=c.read(c.PLAN);c.check(plan);p=json.loads((OLD/'run_plan.json').read_text());target=HERE/'cpu_container_test_report.json';assert not target.exists()
 args=container_command(p,[str(HERE/'cpu_container_entry.py')],False,'eadp-k256-fixed16-ds-draft-cpu');assert '--device' not in args
 args.remove('--rm');args[1]='create'
 files=['contracts.py','selection_observer.py','intervention.py','worker_draft.py','historical_anchor.py','cpu_feasibility.py','cpu_tests.py','cpu_worker_hook_test.py','cpu_old_anchor_test.py','cpu_container_entry.py','run_cpu_container_tests.py','common.py','worker.py','controller.py','prepare_plan.py','verify_terminal.py','test_verify_terminal.py']
 record=dict(status='started',plan_sha256=c.sha(c.PLAN),source_sha256={x:sha(HERE/x) for x in files},old_plan_sha256=sha(OLD/'run_plan.json'),GPU_requested=False)
 def save():target.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 save();created=subprocess.run(args,text=True,capture_output=True);record.update(create_exit=created.returncode,create_stdout=created.stdout,create_stderr=created.stderr);save();assert created.returncode==0
 cid=created.stdout.strip();record['container_id']=cid
 try:
  info=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0];record['mount_proof']=verify_mounts(p,info,gpu=False)
  assert 'CUDA_VISIBLE_DEVICES=' in info['Config']['Env'];save()
  result=subprocess.run(['docker','start','--attach',cid],text=True,capture_output=True)
  state=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]['State']
  record.update(actual_exit=result.returncode,stdout=result.stdout,stderr=result.stderr,terminal_state=state);save()
  assert result.returncode==state['ExitCode']==0 and not state['Running'];output=json.loads(result.stdout.strip().splitlines()[-1])
  assert output['success'] and output['tests']>=12 and output['plan_sha256']==c.sha(c.PLAN) and not output['CUDA_initialized'] and not output['GPU_devices_present']
  assert record['source_sha256']=={x:sha(HERE/x) for x in files};record['status']='passed'
 finally:
  removed=subprocess.run(['docker','rm',cid],text=True,capture_output=True);record['remove_exit']=removed.returncode;save()
 assert record['status']=='passed' and record['remove_exit']==0
 print(json.dumps(dict(status='passed',tests=output['tests'],GPU_requested=False,report_sha256=sha(target))))
if __name__=='__main__':main()
