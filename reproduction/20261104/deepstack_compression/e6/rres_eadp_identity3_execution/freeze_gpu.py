"""Freeze six-case GPU identity + post-run validation, never launch GPU."""
import json,os
from pathlib import Path
import gpu_gate
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REV=ROOT/'e1_independent_review';IDENTITY=ROOT/'e6/rres_eadp_identity3_v2';PY='/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python'
def put(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 proof=REV/'rres_eadp_identity3_execution_preflight_result_review.json';q=gpu_gate.read(proof);reviewer=REV/'review_rres_eadp_identity3_execution_preflight_result.py';revmon=REV/'monitor_rres_eadp_identity3_execution_preflight_result';run=HERE/'cpu_preflight001';mon=HERE/'monitor_cpu_preflight001'
 files={'plan_sha256':HERE/'cpu_plan001.json','worker_sha256':run/'worker_preflight.json','launch_sha256':run/'launch.json','identity_sha256':run/'container_identity.json','exit_sha256':run/'exit.json','started_sha256':mon/'started.json','terminal_sha256':mon/'terminal.json','reviewer_sha256':reviewer}
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('GPU_authorized') is not False or q.get('GT_opened') is not False or q.get('existing_predictions_opened') is not True or q.get('errors')!=[] or any(q.get(k)!=gpu_gate.sha(v) for k,v in files.items()) or not gpu_gate.ended(revmon/'started.json',revmon/'terminal.json') or not gpu_gate.ended(mon/'started.json',mon/'terminal.json'):raise ValueError('execution noGPU independent proof')
 sources=gpu_gate.read(HERE/'cpu_plan001.json')['source_sha256'].copy()
 extra=[HERE/'cpu_plan001.json',HERE/'freeze_gpu.py',proof,reviewer,revmon/'started.json',revmon/'terminal.json',run/'worker_preflight.json',run/'launch.json',run/'container_identity.json',run/'exit.json',mon/'started.json',mon/'terminal.json']
 for path in extra:sources[str(path.resolve())]=gpu_gate.sha(path)
 if any(gpu_gate.sha(path)!=digest for path,digest in sources.items()):raise ValueError('source drift')
 progress=[str((HERE/'run001'/arm/'observations/progress.jsonl').resolve()) for arm in ('rres','eadp')]
 cmd=[PY,str(ROOT/'monitor/supervise.py'),'--run-dir',str(HERE/'monitor_run001'),'--timeout','3600','--stale','1200']
 for path in progress:cmd.extend(['--progress',path])
 cmd.extend(['--grace-seconds','120','--',PY,str(HERE/'controller.py'),'--run-id','001'])
 plan={'status':'frozen','scope':'rres_eadp_identity3_gpu','source_sha256':dict(sorted(sources.items())),'preflight_review_sha256':gpu_gate.sha(REV/'rres_eadp_identity3_v2_preflight_result_review.json'),'execution_preflight_review_sha256':gpu_gate.sha(proof),'anchors':[['DocVQA_VAL','64588'],['TextVQA_VAL','36511'],['AI2D_TEST','207406']],'arms':['rres','eadp'],'run_command':cmd,'container_timeout_seconds':1600,'stale_seconds':1200,'outer_timeout_seconds':3600,'outer_grace_seconds':120,'GT_opened':False,'existing_predictions_opened':True,'timing_claims':False,'limits':['Diagnostic hooks and synchronizations prohibit latency claims','Post-run source snapshot cannot exclude transient changes','Weight stat endpoints do not exclude transient replacement','EADP prior prefill logits not archived']}
 put(HERE/'gpu_plan001.json',plan)
 formal={'status':'frozen','scope':'six-case non-timing identity formal verification','gpu_plan_sha256':gpu_gate.sha(HERE/'gpu_plan001.json'),'verifier_sha256':gpu_gate.sha(HERE/'verify.py'),'snapshot_sha256':gpu_gate.sha(HERE/'snapshot.py'),'snapshot_command':[PY,str(ROOT/'monitor/supervise.py'),'--run-dir',str(HERE/'monitor_snapshot001'),'--timeout','900','--',PY,str(HERE/'snapshot.py')],'verify_command':[PY,str(ROOT/'monitor/supervise.py'),'--run-dir',str(HERE/'monitor_verify001'),'--timeout','900','--',PY,str(HERE/'verify.py'),'--out',str(HERE/'verification001.json')],'GT_opened':False,'timing_claims':False}
 put(HERE/'verification_plan001.json',formal)
if __name__=='__main__':main()
