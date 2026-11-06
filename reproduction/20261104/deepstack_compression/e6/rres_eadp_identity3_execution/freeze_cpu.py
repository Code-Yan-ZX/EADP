"""Freeze execution-layer no-device import test, not GPU authorization."""
import json,os
from pathlib import Path
import gpu_gate
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];IDENTITY=ROOT/'e6/rres_eadp_identity3_v2';REV=ROOT/'e1_independent_review'
def main():
 if not gpu_gate.ended(HERE/'monitor_cpu006/started.json',HERE/'monitor_cpu006/terminal.json'):raise ValueError('CPU tests')
 lifecycle=ROOT/'e6/rres_eadp_paired_timing_assessment'
 if not gpu_gate.ended(lifecycle/'monitor_control_lifecycle001/started.json',lifecycle/'monitor_control_lifecycle001/terminal.json'):raise ValueError('external lifecycle test')
 smoke=HERE/'host_smoke002';sm=smoke/'monitor_run001';ss=gpu_gate.read(sm/'started.json')
 if not gpu_gate.ended(sm/'started.json',sm/'terminal.json') or ss.get('timeout_seconds')!=3600 or ss.get('stale_seconds')!=1200 or ss.get('grace_seconds')!=120 or gpu_gate.read(smoke/'run001/controller_complete.json').get('status')!='passed' or 'two subprocess/device interceptors active' not in (sm/'stdout.log').read_text():raise ValueError('actual host entry smoke')
 sources=gpu_gate.read(IDENTITY/'preflight_plan001.json')['source_sha256'].copy()
 names=['gpu_gate.py','controller.py','gpu_worker.py','snapshot.py','verify.py','docker_contract.py','cpu_gate.py','cpu_entry.py','cpu_launch.py','freeze_cpu.py','host_entry_smoke.py','test_controller.py','test_lock.py','test_verify.py','test_cpu_imports.py','test_docker_contract.py','monitor_cpu006/started.json','monitor_cpu006/terminal.json','host_smoke002/monitor_run001/started.json','host_smoke002/monitor_run001/terminal.json','host_smoke002/monitor_run001/stdout.log','host_smoke002/run001/controller_complete.json']
 extra=[HERE/x for x in names]+[IDENTITY/'preflight_plan001.json',IDENTITY/'preflight001/worker_preflight.json',IDENTITY/'preflight001/launch.json',IDENTITY/'preflight001/container_identity.json',IDENTITY/'preflight001/exit.json',IDENTITY/'monitor_preflight001/started.json',IDENTITY/'monitor_preflight001/terminal.json',REV/'rres_eadp_identity3_v2_cpu_preflight_review.json',REV/'rres_eadp_identity3_v2_preflight_result_review.json',REV/'review_rres_eadp_identity3_v2_preflight_result.py',REV/'monitor_rres_eadp_identity3_v2_preflight_result/started.json',REV/'monitor_rres_eadp_identity3_v2_preflight_result/terminal.json',ROOT/'e6/rres_k256_dev384/run001/launch.json',lifecycle/'test_control_lifecycle.py',lifecycle/'monitor_control_lifecycle001/started.json',lifecycle/'monitor_control_lifecycle001/terminal.json']
 for p in extra:sources[str(p.resolve())]=gpu_gate.sha(p)
 if any(gpu_gate.sha(path)!=digest for path,digest in sources.items()):raise ValueError('frozen source drift')
 result={'status':'frozen','scope':'identity3_execution_cpu_only','source_sha256':dict(sorted(sources.items())),'GPU_authorized':False,'GT_opened':False,'existing_predictions_opened':True,'limits':['No model, GPU or sampling in import preflight','Identity diagnosis remains subject to separate GPU plan and independent review','Scores/predictions from accepted previous outputs are opened for identity only']}
 with (HERE/'cpu_plan001.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
