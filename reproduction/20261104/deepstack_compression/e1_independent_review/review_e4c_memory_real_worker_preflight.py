"""Independent review of CPU-only real-sampler preparation; no GPU admission."""
import argparse,json,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e4_c_memory_real';R=B/'e1_independent_review';sys.path.insert(0,str(D));import gate
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert a.out.resolve()==R/'e4c_memory_real_worker_preflight_review.json'
sha=gate.runtime.sha
for mon in (D/'monitor_cpu004',R/'monitor_memory_real_cpu_review'):
 s=json.loads((mon/'started.json').read_text());t=json.loads((mon/'terminal.json').read_text());assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid']
assert 'Ran 9 tests' in (R/'monitor_memory_real_cpu_review/stderr.log').read_text()
up=gate.admit_cpu();source=gate.sources();c=json.loads((D/'cpu_check004/verification.json').read_text());assert c['status']=='passed' and c['gpu_used'] is False and c['gpu_authorized'] is False and c['source_sha256_before']==source==c['source_sha256_after']
plan=json.loads((D/'preflight_plan.json').read_text());assert plan['gpu_authorized'] is False and (plan['inner_timeout_seconds'],plan['outer_timeout_seconds'],plan['grace_seconds'])==(1800,2100,120) and 'Existing accepted E1 DEV prediction journal' in plan['limits']
correction=R/'e4c_memory_runtime_container001_scope_correction.json';assert json.loads(correction.read_text())['predictions_opened'] is True
result={'status':'passed','independent_review':True,'scope':'real NVML CPU worker preflight only','gpu_authorized':False,'GT_opened':False,'scoring_performed':False,'existing_DEV_predictions_opened':True,'new_predictions_generated':False,'errors':[],'source_sha256':source,'cpu_sha256':sha(D/'cpu_check004/verification.json'),'cpu_started_sha256':sha(D/'monitor_cpu004/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu004/terminal.json'),'fake_review_sha256':up['fake_review_sha256'],'reviewer_sha256':sha(__file__),'prior_scope_correction_sha256':sha(correction),'independent_cpu_test_terminal_sha256':sha(R/'monitor_memory_real_cpu_review/terminal.json'),'limitations':['only CPU preflight approved; no actual GPU/NVML sampling','actual retained CPU container must separately be independently reviewed','accepted DEV prediction journal parsed solely for input identity']}
a.out.write_text(json.dumps(result,indent=2)+'\n');print('passed')
