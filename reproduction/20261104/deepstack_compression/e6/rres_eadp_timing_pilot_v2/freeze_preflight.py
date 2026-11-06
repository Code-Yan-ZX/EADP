"""Freeze no-device probe sources after CPU and actual-reference checks."""
import json,os
from pathlib import Path
import preflight_gate as g
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def main():
 if not g.ended(HERE/'monitor_cpu004/started.json',HERE/'monitor_cpu004/terminal.json') or not g.ended(HERE/'monitor_references001/started.json',HERE/'monitor_references001/terminal.json') or not g.ended(HERE/'monitor_schedule001/started.json',HERE/'monitor_schedule001/terminal.json'):raise ValueError('CPU or real-reference checks unfinished')
 prior=ROOT/'e6/rres_k256_dev384_exec/runtime_preflight_plan001.json';sources=g.read(prior)['source_sha256'].copy()
 names=['PLAN.md','schedule.py','freeze_schedule.py','schedule001.json','runner.py','generation_capture.py','qwen_request.py','process_core.py','process_worker.py','reference_inputs.py','preflight_entry.py','preflight_launch.py','preflight_gate.py','freeze_preflight.py','test_schedule.py','test_runner.py','test_hf_cpu.py','test_qwen_request.py','test_process_core.py']
 extra=[HERE/x for x in names]+[prior,ROOT/'e4_c_multibudget_worker_v3/references.py',ROOT/'e4_c_budget_pilot_v2/adapter.py',ROOT/'e6/rres_k256_pilot_v2/resolution.py',ROOT/'e1_independent_review/rres_k256_dev384_result_review.json',ROOT/'e1_independent_review/review_rres_k256_dev384_result.py',ROOT/'e1_independent_review/monitor_rres_k256_dev384_result/started.json',ROOT/'e1_independent_review/monitor_rres_k256_dev384_result/terminal.json',ROOT/'e6/rres_k256_dev384/verification001.json',ROOT/'e6/rres_k256_dev384/monitor_verify001/started.json',ROOT/'e6/rres_k256_dev384/monitor_verify001/terminal.json']
 for monitor in ('monitor_cpu004','monitor_references001','monitor_schedule001'):
  extra.extend([HERE/monitor/'started.json',HERE/monitor/'terminal.json'])
 for path in extra:sources[str(path.resolve())]=g.sha(path)
 if any(g.sha(path)!=digest for path,digest in sources.items()):raise ValueError('historical source changed')
 data={'status':'frozen','scope':'paired_timing_cpu_preflight_only','source_sha256':dict(sorted(sources.items())),'inputs':62,'references':124,'GPU_authorized':False,'GT_opened':False,'existing_predictions_opened':True,'scoring_performed':False,'limitations':['Imports and prior outputs only; no model/GPU/identity diagnostic or timing measured','Existing prediction-containing sources are opened for identity and hashing','Generated output and full effective config must be validated on actual Qwen before timing','No GPU execution authorized']}
 with (HERE/'preflight_plan001.json').open('x') as f:json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
