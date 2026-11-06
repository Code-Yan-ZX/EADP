"""GPU admission predicate; a future independent reviewer must create proof.

Runs before any CUDA query/model construction. Never synthesizes a proof.
"""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
spec=importlib.util.spec_from_file_location('_pilot_bootstrap_admission_v4',HERE/'bootstrap.py');boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.pin_all();pre=boot.pin('preflight',HERE/'preflight.py')

def valid_terminal(start,terminal):
 return (terminal.get('reason')=='completed' and terminal.get('worker_returncode')==0 and terminal.get('remaining_process_group')==[] and
         type(start.get('worker_pid')) is int and start.get('worker_pid')==terminal.get('worker_pid'))

def called(start,script):
 cwd=Path(start.get('cwd','/'));cmd=start.get('command',[])
 resolved=[str((cwd/x).resolve()) if not Path(x).is_absolute() else str(Path(x).resolve()) for x in cmd]
 return resolved.count(str(script.resolve()))==1

def validate(plan_hash,cpu,cs,ct,pref,launch,exit,ps,pt,handshake,hl,he,hs,ht,review,rs,rt,paths,sha):
 if (not valid_terminal(cs,ct) or not called(cs,HERE/'cpu_check.py') or not called(ps,HERE/'preflight_launch.py') or cpu.get('status')!='passed' or cpu.get('plan_sha256')!=plan_hash or cpu.get('GT_opened') is not False or cpu.get('gpu_used') is not False or
     pref.get('status')!='passed' or pref.get('source_plan_sha256')!=plan_hash or pref.get('cuda_initialized') is not False or pref.get('model_loaded') is not False or pref.get('weight_sha256_verified')!=4 or pref.get('same_arm_references')!=9 or pref.get('GT_opened') is not False or
     not valid_terminal(ps,pt) or ps.get('worker_pid')!=launch.get('controller',{}).get('pid') or ps.get('worker_pid')!=exit.get('controller',{}).get('pid') or
     launch.get('plan_sha256')!=plan_hash or exit.get('returncode')!=0 or exit.get('container_exit_code')!=0 or exit.get('timed_out') is not False or exit.get('interrupted') is not None or
     exit.get('failure_type') is not None or exit.get('client_exited_while_container_running') is not False or exit.get('container_stopped_verified') is not True or
     exit.get('last_inspect',{}).get('device_requests') is not None or exit.get('last_inspect',{}).get('host_devices')!=[]):raise ValueError('CPU/no-GPU preflight evidence failed')
 if (not valid_terminal(hs,ht) or not called(hs,HERE/'handshake_launch.py') or hs.get('worker_pid')!=hl.get('controller',{}).get('pid') or hs.get('worker_pid')!=he.get('controller',{}).get('pid') or
     hl.get('plan_sha256')!=plan_hash or handshake.get('status')!='passed' or handshake.get('cuda_initialized') is not False or handshake.get('model_loaded') is not False or
     handshake.get('launch_sha256')!=sha(paths['hl']) or handshake.get('release_sha256')!=sha(paths['release']) or
     handshake.get('container_id')!=(he.get('container_identity') or {}).get('container_id') or he.get('returncode')!=0 or he.get('container_exit_code')!=0 or
     he.get('timed_out') is not False or he.get('interrupted') is not None or he.get('failure_type') is not None or he.get('client_exited_while_container_running') is not False or
     he.get('container_stopped_verified') is not True or he.get('last_inspect',{}).get('device_requests') is not None or he.get('last_inspect',{}).get('host_devices')!=[]):raise ValueError('no-GPU container handshake failed')
 if (review.get('status')!='passed' or review.get('independent_review') is not True or review.get('GT_opened') is not False or review.get('scoring_performed') is not False or review.get('plan_sha256')!=plan_hash or
     review.get('cpu_sha256')!=sha(paths['cpu']) or review.get('cpu_started_sha256')!=sha(paths['cs']) or review.get('cpu_terminal_sha256')!=sha(paths['ct']) or review.get('preflight_launch_sha256')!=sha(paths['launch']) or review.get('preflight_sha256')!=sha(paths['pref']) or review.get('preflight_started_sha256')!=sha(paths['ps']) or
     review.get('preflight_terminal_sha256')!=sha(paths['pt']) or review.get('preflight_exit_sha256')!=sha(paths['exit']) or
     review.get('handshake_sha256')!=sha(paths['handshake']) or review.get('handshake_launch_sha256')!=sha(paths['hl']) or review.get('handshake_exit_sha256')!=sha(paths['he']) or review.get('handshake_started_sha256')!=sha(paths['hs']) or review.get('handshake_terminal_sha256')!=sha(paths['ht']) or review.get('handshake_release_sha256')!=sha(paths['release']) or
     review.get('reviewer_sha256')!=sha(paths['reviewer']) or not valid_terminal(rs,rt)):raise ValueError('independent prelaunch review failed')
 cmd=rs.get('command',[]);cwd=Path(rs.get('cwd','/'));resolved=[str((cwd/x).resolve()) if not Path(x).is_absolute() else str(Path(x).resolve()) for x in cmd]
 if resolved.count(str(paths['reviewer'].resolve()))!=1 or cmd.count('--out')!=1 or cmd.index('--out')+1>=len(cmd) or resolved[cmd.index('--out')+1]!=str(paths['review'].resolve()):raise ValueError('reviewer command/output mismatch')
 return True

def require():
 plan=pre.read(HERE/'plan001.json');pre.verify_source_plan(plan);h=pre.sha(HERE/'plan001.json')
 paths={'cpu':HERE/'cpu001.json','cs':HERE/'monitor_cpu001/started.json','ct':HERE/'monitor_cpu001/terminal.json','pref':HERE/'preflight001.json','ps':HERE/'monitor_preflight001/started.json','pt':HERE/'monitor_preflight001/terminal.json',
        'launch':HERE/'run_preflight001/launch.json','exit':HERE/'run_preflight001/exit.json','review':HERE/'independent_prelaunch_review.json',
        'rs':BASE/'e1_independent_review/monitor_pixel_timing_v4_prelaunch/started.json','rt':BASE/'e1_independent_review/monitor_pixel_timing_v4_prelaunch/terminal.json',
        'reviewer':BASE/'e1_independent_review/review_pixel_timing_v4_prelaunch.py',
        'handshake':HERE/'run_handshake001/job_000/handshake.json','hl':HERE/'run_handshake001/job_000/launch.json',
        'he':HERE/'run_handshake001/job_000/exit.json','release':HERE/'run_handshake001/job_000/release.json',
        'hs':HERE/'monitor_handshake001/started.json','ht':HERE/'monitor_handshake001/terminal.json'}
 objects={k:pre.read(v) for k,v in paths.items() if k!='reviewer'}
 if objects['cpu'].get('source_sha256')!=plan.get('sha256'):raise ValueError('CPU source inventory mismatch')
 validate(h,objects['cpu'],objects['cs'],objects['ct'],objects['pref'],objects['launch'],objects['exit'],objects['ps'],objects['pt'],objects['handshake'],objects['hl'],objects['he'],objects['hs'],objects['ht'],objects['review'],objects['rs'],objects['rt'],paths,pre.sha)
 return {'plan_sha256':h,'preflight_sha256':pre.sha(paths['pref']),'review_sha256':pre.sha(paths['review'])}
