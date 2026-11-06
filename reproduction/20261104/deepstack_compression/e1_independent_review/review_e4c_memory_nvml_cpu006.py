"""CPU-only NVML parser/sampler audit; no GPU permission."""
import argparse,json,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e4_c_memory_nvml';R=B/'e1_independent_review';sys.path.insert(0,str(D));import admission
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert a.out.resolve()==R/'e4c_memory_nvml_cpu006_review.json'
sha=admission.prior.runtime.sha
for mon in (D/'monitor_cpu006',R/'monitor_nvml_cpu006_review_tests',R/'monitor_e4c_memory_real_preflight001'):
 s=json.loads((mon/'started.json').read_text());t=json.loads((mon/'terminal.json').read_text());assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid']
assert 'Ran 9 tests' in (R/'monitor_nvml_cpu006_review_tests/stderr.log').read_text();up=admission.admit_cpu();source=admission.sources();cpu=json.loads((D/'cpu_check006/verification.json').read_text());assert cpu['status']=='passed' and cpu['gpu_used'] is False and cpu['gpu_authorized'] is False and cpu['source_sha256_before']==source==cpu['source_sha256_after']
r={'status':'passed','independent_review':True,'scope':'NVML device/process sampler CPU-only review','gpu_authorized':False,'GT_opened':False,'scoring_performed':False,'errors':[],'source_sha256':source,'cpu_sha256':sha(D/'cpu_check006/verification.json'),'cpu_started_sha256':sha(D/'monitor_cpu006/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu006/terminal.json'),'prior_review_sha256':up['prior_review_sha256'],'reviewer_sha256':sha(__file__),'independent_tests_terminal_sha256':sha(R/'monitor_nvml_cpu006_review_tests/terminal.json'),'limitations':['synthetic tests only; no actual NVML query or GPU sampler operation authorized','process/device distinction, missing observations, graphics background, MiB rounding and transient missed peaks must remain explicit']};a.out.write_text(json.dumps(r,indent=2)+'\n');print('passed')
