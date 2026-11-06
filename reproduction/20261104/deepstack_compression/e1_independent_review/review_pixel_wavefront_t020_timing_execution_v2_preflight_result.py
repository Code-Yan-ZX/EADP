"""Independent review of actual timing-layer no-GPU import preflight."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent;D=B/'e6/pixel_wavefront_t020_timing_execution_v2';ROOT=B.parents[2]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out.resolve()
 if out!=R/'pixel_wavefront_t020_timing_execution_v2_preflight_result_review.json':raise ValueError('unplanned output')
 run=D/'run_preflight001';mon=D/'monitor_preflight001';plan=read(D/'preflight_plan001.json');w=read(run/'worker_preflight.json');s=read(mon/'started.json');t=read(mon/'terminal.json');launch=read(run/'launch.json');ex=read(run/'exit.json');ident=read(run/'container_identity.json')
 if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or s.get('worker_pid')!=launch['controller']['pid'] or (Path(s['cwd'])/s['command'][1]).resolve()!=D/'preflight_launch.py'):raise ValueError('supervisor')
 if (ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('container_identity')!=ident or ex.get('controller')!=launch.get('controller') or ex['last_inspect'].get('host_devices') not in (None,[]) or ex['last_inspect'].get('device_requests') is not None):raise ValueError('exit or device')
 obj=json.loads(subprocess.run(['docker','inspect',ident['container_id']],capture_output=True,text=True,timeout=15,check=True).stdout)[0];state=obj['State'];host=obj['HostConfig']
 if (obj['Id']!=ident['container_id'] or obj['Image']!=ident['image'] or state['Running'] or state['Pid']!=0 or state['ExitCode']!=0 or state['OOMKilled'] or host['NetworkMode']!='none' or host['Devices'] not in (None,[]) or host['DeviceRequests'] not in (None,[]) or 'CUDA_VISIBLE_DEVICES=' not in obj['Config']['Env'] or obj['Config']['Cmd']!=[str(D/'preflight_entry.py'),'--out',str(run/'worker_preflight.json')]):raise ValueError('actual Docker state')
 pre=R/'pixel_wavefront_t020_timing_execution_v2_preflight_review.json'
 if (w.get('status')!='passed' or w.get('gpu_authorized') is not False or w.get('cuda_initialized') is not False or w.get('model_loaded') is not False or w.get('GT_opened') is not False or w.get('existing_DEV_predictions_opened') is not True or w.get('references')!=6 or w.get('approval')!=launch.get('approval') or w['approval'].get('plan_sha256')!=sha(D/'preflight_plan001.json') or w['approval'].get('review_sha256')!=sha(pre) or w['approval'].get('source_sha256')!=plan['source_sha256']):raise ValueError('worker scope or approval')
 pixel=B/'e6/external_sources/pixelprune_8d347eb';author=ROOT/'_reproduction/author_eadp/Qwen_vl';paths={'pixelprune':str(pixel/'pixelprune/__init__.py'),'pixelprune.methods.pred_2d':str(pixel/'pixelprune/methods/pred_2d.py'),'pixelprune.patches.qwen3_vl_hf':str(pixel/'pixelprune/patches/qwen3_vl_hf.py'),'vlmeval.vlm.qwen3_vl.model_fixed_res':str(author/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py'),'transformers.models.qwen3_vl.modeling_qwen3_vl':str(ROOT/'_reproduction/envs/qwen_author/lib/python3.10/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py')}
 if w.get('module_paths')!=paths or len(plan['source_sha256'])!=368 or any(sha(p)!=h for p,h in plan['source_sha256'].items()):raise ValueError('module path/source mismatch')
 identity=w['identity']
 if identity.get('formal_sha256')!=sha(B/'e6/pixel_wavefront_t020_accept_v2/verification001.json') or identity.get('review_sha256')!=sha(R/'pixel_wavefront_t020_run_v2_result_review.json') or identity.get('design_sha256')!=sha(B/'e6/pixel_wavefront_t020_model_timing_v2/design001.json'):raise ValueError('identity binding')
 proof={'status':'passed','independent_review':True,'scope':'timing execution no-GPU import preflight only','gpu_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'plan_sha256':sha(D/'preflight_plan001.json'),'worker_preflight_sha256':sha(run/'worker_preflight.json'),'launch_sha256':sha(run/'launch.json'),'exit_sha256':sha(run/'exit.json'),'started_sha256':sha(mon/'started.json'),'terminal_sha256':sha(mon/'terminal.json'),'reviewer_sha256':sha(__file__),'source_sha256':plan['source_sha256'],'errors':[],'limitations':['No actual timing or GPU model execution authorized.','Reviewer authored wavefront candidate but not this execution layer.','Prior DEV identity files include predictions; no GT read.']}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
