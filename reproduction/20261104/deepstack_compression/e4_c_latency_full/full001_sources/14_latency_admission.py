"""Separate two-stage review gates for CPU preflight and latency smoke."""
import json
from pathlib import Path
from identity_gate import sha,supervised
from latency_gate import require_light_identity
from real_worker import PLAN,WEIGHTS
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;REVIEW=BASE/'e1_independent_review'

def terminal(started,ended):
 s=json.loads(Path(started).read_text());t=json.loads(Path(ended).read_text())
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid'):raise ValueError('latency supervisor incomplete')
 return s

def supervised_run_id(started,script,prefix,output_path):
 cmd=started.get('command',[]);cwd=Path(started.get('cwd','/'));out=Path(output_path).resolve().parent
 if ([(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or cmd.count('--run-id')!=1 or cmd.index('--run-id')+1>=len(cmd) or
     out!=HERE/(prefix+cmd[cmd.index('--run-id')+1])):raise ValueError('latency preflight supervisor command mismatch')

def cpu_container(launch,ex,started,target):
 cmd=launch.get('command',[]);identity=ex.get('container_identity') or {};last=ex.get('last_inspect') or {};author=json.loads(PLAN.read_text())
 if (launch.get('gpu_requested') is not False or '--device' in cmd or '--gpus' in cmd or cmd.count(str(target))!=1 or
     (launch.get('controller') or {}).get('pid')!=started.get('worker_pid') or ex.get('controller')!=launch.get('controller') or
     ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or
     ex.get('failure_type') is not None or ex.get('client_exited_while_container_running') is not False or ex.get('container_stopped_verified') is not True or
     ex.get('gpu_pids_after') is not None or not identity.get('host_init') or identity.get('image')!=author['container_image'] or
     str(target) not in identity.get('host_argv',[]) or identity.get('container_id','') not in identity.get('host_cgroup','') or
     last.get('id')!=identity.get('container_id') or last.get('image')!=author['container_image'] or last.get('running') is not False or last.get('pid')!=0 or last.get('exit_code')!=0 or
     last.get('host_devices') or last.get('device_requests')):raise ValueError('latency CPU container exit invalid')

def admit(stage,preflight_only=False):
 if stage!='smoke':raise ValueError('full latency locked until reviewed timing smoke and bounded budget')
 identity=require_light_identity()
 if preflight_only:
  proof=REVIEW/'e4c_latency_worker_preflight_review.json';reviewer=REVIEW/'review_e4c_latency_worker_preflight.py';monitor=REVIEW/'monitor_e4c_latency_worker_preflight';scope='latency_worker_cpu_preflight'
 else:
  proof=REVIEW/'e4c_latency_smoke_prelaunch_review.json';reviewer=REVIEW/'review_e4c_latency_smoke_prelaunch.py';monitor=REVIEW/'monitor_e4c_latency_smoke_prelaunch';scope='latency_smoke_launch'
 if not proof.is_file():raise ValueError('latency locked: independent review missing')
 supervised(monitor/'started.json',monitor/'terminal.json',reviewer,proof);r=json.loads(proof.read_text())
 from latency_worker import sources
 expected={str(p):sha(p) for p in sources()}
 if (r.get('status')!='passed' or r.get('scope')!=scope or r.get('independent_review') is not True or r.get('GT_opened') is not False or
     r.get('scoring_performed') is not False or r.get('errors')!=[] or r.get('light_identity')!=identity or r.get('source_sha256')!=expected or r.get('reviewer_sha256')!=sha(reviewer)):
  raise ValueError('latency review mismatch')
 names=['cpu_proof','cpu_started','cpu_terminal','import_launch','import_exit','import_started','import_terminal']
 if not preflight_only:names+=['worker_launch','worker_exit','worker_started','worker_terminal','worker_result']
 for name in names:
  path=Path(r[name+'_path']).resolve()
  if not path.is_relative_to(HERE) or r.get(name+'_sha256')!=sha(path):raise ValueError('latency review input mismatch: '+name)
 cstarted=terminal(r['cpu_started_path'],r['cpu_terminal_path']);istarted=terminal(r['import_started_path'],r['import_terminal_path'])
 supervised_run_id(cstarted,HERE/'cpu_check.py','cpu_check',r['cpu_proof_path']);supervised_run_id(istarted,HERE/'preflight_import.py','import_preflight',r['import_launch_path'])
 cpu=json.loads(Path(r['cpu_proof_path']).read_text());il=json.loads(Path(r['import_launch_path']).read_text());ie=json.loads(Path(r['import_exit_path']).read_text())
 cpu_container(il,ie,istarted,HERE/'import_probe.py')
 if (cpu.get('status')!='passed' or cpu.get('returncode')!=0 or cpu.get('source_sha256_before')!=cpu.get('source_sha256_after') or
     any(cpu.get('source_sha256_after',{}).get(k)!=v for k,v in expected.items()) or il.get('gpu_requested') is not False or
     '--device' in il.get('command',[]) or '--gpus' in il.get('command',[]) or (il.get('controller') or {}).get('pid')!=istarted.get('worker_pid') or
     any(il.get('source_sha256',{}).get(k)!=v for k,v in expected.items()) or Path(r['import_exit_path']).resolve().parent!=Path(r['import_launch_path']).resolve().parent):raise ValueError('latency CPU or container import proof invalid')
 if not preflight_only:
  wstarted=terminal(r['worker_started_path'],r['worker_terminal_path']);wl=json.loads(Path(r['worker_launch_path']).read_text());we=json.loads(Path(r['worker_exit_path']).read_text());wr=json.loads(Path(r['worker_result_path']).read_text())
  supervised_run_id(wstarted,HERE/'preflight_worker.py','worker_preflight',r['worker_launch_path']);cpu_container(wl,we,wstarted,HERE/'latency_worker.py')
  prior=admit(stage,preflight_only=True);cmd=wl.get('command',[]);workdir=Path(r['worker_launch_path']).resolve().parent
  if (wl.get('approval_bundle')!=prior or wl.get('gpu_requested') is not False or (wl.get('controller') or {}).get('pid')!=wstarted.get('worker_pid') or
      '--device' in cmd or '--gpus' in cmd or cmd.count('--preflight-only')!=1 or cmd.count(str(HERE/'latency_worker.py'))!=1 or
      any(wl.get('source_sha256',{}).get(k)!=v for k,v in expected.items()) or we.get('returncode')!=0 or we.get('container_exit_code')!=0 or
      we.get('timed_out') is not False or we.get('interrupted') is not None or we.get('failure_type') is not None or we.get('container_stopped_verified') is not True or
      we.get('client_exited_while_container_running') is not False or we.get('gpu_pids_after') is not None or
      Path(r['worker_exit_path']).resolve().parent!=workdir or Path(r['worker_result_path']).resolve()!=workdir/'worker_preflight.json' or
      wr.get('status')!='passed' or wr.get('cuda_initialized') is not False or wr.get('model_loaded') is not False or wr.get('predictions') is not False or
      wr.get('source_sha256')!=expected or wr.get('light_identity')!=identity or wr.get('stage')!='smoke' or len(wr.get('weight_stats_before',{}))!=4 or
      cmd.count('--stage')!=1 or cmd[cmd.index('--stage')+1]!='smoke' or cmd.count('--out')!=1 or Path(cmd[cmd.index('--out')+1]).resolve()!=workdir):raise ValueError('latency full worker preflight invalid')
 return {'scope':scope,'review_sha256':sha(proof),'review_started_sha256':sha(monitor/'started.json'),'review_terminal_sha256':sha(monitor/'terminal.json'),
         'light_identity':identity,'source_sha256':expected,'preflight_only':preflight_only}
