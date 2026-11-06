"""Independent actual retained no-GPU container preflight review."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/pixel_wavefront_t020_timing_run_v2';P=B/'e6/pixel_wavefront_t020_timing_runtime_v4'
OUT=I/'pixel_wavefront_t020_timing_run_v2_preflight_result_review.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def supervised(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or len(cmd)<2 or (cwd/cmd[1]).resolve()!=script.resolve() or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervisor identity/exit')
 return s
def inspect(cid):return json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=15,check=True).stdout)[0]
def mounts(obj):return sorted((m['Type'],m['Source'],m['Destination'],m['RW']) for m in obj['Mounts'])
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);out=a.parse_args().out.resolve()
 if out!=OUT.resolve() or out.exists():raise ValueError('unexpected output')
 mon=D/'monitor_preflight001';run=D/'run_preflight001';plan=read(D/'preflight_plan001.json');w=read(run/'worker_preflight.json');launch=read(run/'launch.json');ident=read(run/'container_identity.json');ex=read(run/'exit.json')
 s=supervised(mon,D/'preflight_launch.py',run);supervised(I/'monitor_pixel_wavefront_t020_timing_run_v2_preflight',I/'review_pixel_wavefront_t020_timing_run_v2_preflight.py',I/'pixel_wavefront_t020_timing_run_v2_preflight_review.json')
 if (s['worker_pid']!=launch.get('controller',{}).get('pid') or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('client_exited_while_container_running') is not False or ex.get('failure_type') is not None or ex.get('container_identity')!=ident):raise ValueError('actual controller or exit')
 cid=ident['container_id'];obj=inspect(cid);prior=inspect(read(P/'run_preflight001/exit.json')['container_identity']['container_id']);state=obj['State'];host=obj['HostConfig'];config=obj['Config'];pconfig=prior['Config'];phost=prior['HostConfig']
 if (obj['Id']!=cid or obj['Image']!=ident.get('image') or obj['Image']!=prior['Image'] or state['Running'] is not False or state['Pid']!=0 or state['ExitCode']!=0 or state['OOMKilled'] is not False or host['NetworkMode']!='none' or host['Devices'] not in (None,[]) or host['DeviceRequests'] not in (None,[]) or 'CUDA_VISIBLE_DEVICES=' not in config['Env'] or config['Cmd']!=[str(D/'preflight_entry.py'),'--out',str(run/'worker_preflight.json')]):raise ValueError('Docker state, device, command or image')
 if any(config.get(k)!=pconfig.get(k) for k in ('Entrypoint','User','WorkingDir')) or len({v.split('=',1)[0] for v in config['Env']})!=len(config['Env']) or len({v.split('=',1)[0] for v in pconfig['Env']})!=len(pconfig['Env']) or sorted(config['Env'])!=sorted(pconfig['Env']) or any(host.get(k)!=phost.get(k) for k in ('NetworkMode','CapDrop','SecurityOpt','CgroupnsMode')) or mounts(obj)!=mounts(prior):raise ValueError('Docker configuration or mounts differ from accepted baseline')
 review=I/'pixel_wavefront_t020_timing_run_v2_preflight_review.json'
 if (w.get('status')!='passed' or w.get('references')!=6 or w.get('GPU_authorized') is not False or w.get('model_loaded') is not False or w.get('cuda_initialized') is not False or w.get('GT_opened') is not False or w.get('existing_DEV_predictions_opened') is not True or w.get('container_id')!=cid or w.get('container_identity_sha256')!=sha(run/'container_identity.json') or w.get('approval')!=launch.get('approval') or w['approval'].get('plan_sha256')!=sha(D/'preflight_plan001.json') or w['approval'].get('review_sha256')!=sha(review)):raise ValueError('worker approval/scope')
 expected={'worker':P/'worker.py','controller':P/'controller.py','host_runner':P/'host_runner.py','nvml_stages':P/'nvml_stages.py','check':B/'e6/pixel_wavefront_t020_timing_controller_draft/job_contract.py','model':B/'e6/pixel_wavefront_t020_model_timing_v2/model_candidate.py'}
 if w.get('gpu_worker_path')!=str(D/'gpu_worker.py') or w.get('gpu_launcher_path')!=str(D/'gpu_launch.py') or w.get('module_paths')!={k:str(v) for k,v in expected.items()}:raise ValueError('actual imported bridge source mismatch')
 baseline=read(P/'run_preflight001/worker_preflight.json')
 if w.get('model_module_paths')!=baseline.get('model_module_paths'):raise ValueError('model import source mismatch')
 sources=plan['source_sha256']
 if len(sources)!=409 or any(sha(p)!=v for p,v in sources.items()):raise ValueError('source identity changed')
 proof={'status':'passed','independent_review':True,'preflight_only':True,'gpu_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'plan_sha256':sha(D/'preflight_plan001.json'),'worker_preflight_sha256':sha(run/'worker_preflight.json'),'launch_sha256':sha(run/'launch.json'),'identity_sha256':sha(run/'container_identity.json'),'exit_sha256':sha(run/'exit.json'),'started_sha256':sha(mon/'started.json'),'terminal_sha256':sha(mon/'terminal.json'),'prelaunch_review_sha256':sha(review),'reviewer_sha256':sha(__file__),'source_sha256':sources,'errors':[],'limitations':['No model, GPU or NVML sampling run was authorized.','Existing accepted DEV mixed predictions parsed for identity; no GT.','Review compares retained container with the previously accepted baseline, not a performance measurement.']}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
