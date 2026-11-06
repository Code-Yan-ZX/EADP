"""Fail-closed full-panel GPU admission; verifies real noGPU and reviews."""
import hashlib,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];F=B/'e6/rres_k256_dev384';I=B/'e1_independent_review'
sys.path.insert(0,str(F));import pilot_gate
sha=pilot_gate.sha;read=pilot_gate.read;terminal=pilot_gate.terminal
def admit_gpu():
 paths=[F/'gpu_plan001.json',F/'verification_plan001.json',I/'rres_k256_dev384_preflight_result_review.json',I/'rres_k256_dev384_gpu_prelaunch_review.json',I/'rres_k256_full_score_prelaunch_review.json',I/'rres_k256_dev384_runtime_result_review.json']
 if not all(p.exists() for p in paths):raise RuntimeError('GPU locked until noGPU result and all independent reviews')
 pilot=pilot_gate.require();plan,formal,pre,q,score,runtime=[read(p) for p in paths]
 pre_path=paths[2];pr=I/'review_rres_k256_dev384_preflight_result.py';terminal(I/'monitor_rres_k256_dev384_preflight_result',pr,pre_path);terminal(F/'monitor_preflight001',F/'preflight_launch.py',F/'run_preflight001')
 fields={'plan_sha256':F/'preflight_plan001.json','worker_preflight_sha256':F/'run_preflight001/worker_preflight.json','processor_probe_sha256':F/'run_preflight001/probe/processor_probe.json','launch_sha256':F/'run_preflight001/launch.json','identity_sha256':F/'run_preflight001/container_identity.json','exit_sha256':F/'run_preflight001/exit.json','started_sha256':F/'monitor_preflight001/started.json','terminal_sha256':F/'monitor_preflight001/terminal.json','reviewer_sha256':pr}
 if pre.get('status')!='passed' or pre.get('independent_review') is not True or pre.get('preflight_only') is not True or pre.get('GPU_authorized') is not False or pre.get('images')!=384 or pre.get('GT_opened') is not False or pre.get('existing_DEV_predictions_opened') is not True or pre.get('errors')!=[] or any(pre.get(k)!=sha(p) for k,p in fields.items()):raise ValueError('independent noGPU result')
 rr=I/'review_rres_k256_dev384_runtime_result.py';terminal(I/'monitor_rres_k256_dev384_runtime_result',rr,paths[5]);terminal(D/'monitor_runtime_preflight001',D/'runtime_preflight_launch.py',D/'runtime_preflight001')
 runtime_fields={'plan_sha256':D/'runtime_preflight_plan001.json','worker_preflight_sha256':D/'runtime_preflight001/worker_preflight.json','launch_sha256':D/'runtime_preflight001/launch.json','identity_sha256':D/'runtime_preflight001/container_identity.json','exit_sha256':D/'runtime_preflight001/exit.json','started_sha256':D/'monitor_runtime_preflight001/started.json','terminal_sha256':D/'monitor_runtime_preflight001/terminal.json','reviewer_sha256':rr}
 if runtime.get('status')!='passed' or runtime.get('independent_review') is not True or runtime.get('preflight_only') is not True or runtime.get('GPU_authorized') is not False or runtime.get('GT_opened') is not False or runtime.get('errors')!=[] or any(runtime.get(k)!=sha(p) for k,p in runtime_fields.items()):raise ValueError('runtime noGPU result')
 reviewer=I/'review_rres_k256_dev384_gpu_prelaunch.py';terminal(I/'monitor_rres_k256_dev384_gpu_prelaunch',reviewer,paths[3]);score_reviewer=I/'review_rres_k256_full_score_prelaunch.py';terminal(I/'monitor_rres_k256_full_score_prelaunch',score_reviewer,paths[4])
 A=B/'e6/rres_k256_full_analysis_v2'
 if score.get('status')!='passed' or score.get('scope')!='rres_k256_full_score_prelaunch' or score.get('independent_review') is not True or score.get('GT_opened') is not False or score.get('target_predictions_opened') is not False or score.get('errors')!=[] or score.get('reviewer_sha256')!=sha(score_reviewer) or score.get('score_plan_sha256')!=sha(A/'score_plan001.json') or score.get('cpu_proof_sha256')!=sha(A/'cpu_proof001.json'):raise ValueError('score prelaunch proof')
 if plan.get('scope')!='rres384_gpu_generation' or plan.get('images')!=384 or plan.get('GT_opened') is not False or plan.get('pilot_review_sha256')!=pilot['review_sha256'] or plan.get('preflight_result_sha256')!=sha(pre_path) or plan.get('runtime_result_sha256')!=sha(paths[5]) or plan.get('score_review_sha256')!=sha(paths[4]) or formal.get('gpu_plan_sha256')!=sha(paths[0]) or any(sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('GPU plan')
 expected={'plan_sha256':paths[0],'formal_plan_sha256':paths[1],'preflight_result_sha256':pre_path,'runtime_result_sha256':paths[5],'score_review_sha256':paths[4],'reviewer_sha256':reviewer}
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('scope')!='rres384_gpu_prelaunch' or q.get('GT_opened') is not False or q.get('scoring_performed') is not False or q.get('errors')!=[] or q.get('source_sha256')!=plan['source_sha256'] or any(q.get(k)!=sha(p) for k,p in expected.items()):raise ValueError('GPU independent prelaunch')
 weights=read(F/'run_preflight001/worker_preflight.json')['weights']
 if len(weights)!=4:raise ValueError('weight inventory')
 for x in weights:
  s=Path(x['path']).stat()
  if x['stat']!=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]:raise ValueError('weight endpoint mismatch')
 return {'plan_sha256':sha(paths[0]),'source_sha256':plan['source_sha256']}
