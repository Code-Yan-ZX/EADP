"""Freeze new minimal handshake sources, upstream proofs and limits."""
import hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;ROOT=B.parents[2];R=B/'e1_independent_review';AP=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
proofs=[('e4c_memory_fake_handshake001_review.json','review_e4c_memory_fake_handshake001.py','monitor_e4c_memory_fake_handshake001'),('e4c_memory_nvml_cpu006_review.json','review_e4c_memory_nvml_cpu006.py','monitor_e4c_memory_nvml_cpu006'),('e4c_memory_real_preflight001_review.json','review_e4c_memory_real_preflight001.py','monitor_e4c_memory_real_preflight001')]
sources=list(D.glob('*.py'))+[D/'PLAN.md',B/'initial_plan.json',B/'monitor/supervise.py',B/'e4_c_latency/timing_launch.py',AP,AP.parent/'run_panel.py',Path(json.loads(AP.read_text())['container_runtime_source'])]
for name in ('e4_c_memory','e4_c_memory_runtime','e4_c_memory_real','e4_c_memory_nvml'):sources.extend((B/name).glob('*.py'))
for proof,reviewer,monitor in proofs:
 p=R/proof;r=json.loads(p.read_text());s=json.loads((R/monitor/'started.json').read_text());t=json.loads((R/monitor/'terminal.json').read_text())
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('reviewer_sha256')!=sha(R/reviewer) or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or t.get('worker_pid')!=s.get('worker_pid'):raise ValueError('upstream independent proof incomplete')
 sources.extend([p,R/reviewer,R/monitor/'started.json',R/monitor/'terminal.json'])
plan={'scope':'minimal real CUDA allocation and host NVML handshake','allocation_bytes':16*1024*1024,'hold_seconds':8,'sample_interval_ns':500000000,'max_sample_gap_ns':5000000000,'worker_timeout_seconds':60,'sampler_timeout_seconds':150,'inner_timeout_seconds':180,'outer_timeout_seconds':300,'grace_seconds':90,'budget_source_sha256':sha(B/'initial_plan.json'),'source_sha256':{str(p.resolve()):sha(p) for p in sorted(set(sources))},'GPU_authorization':'independent review and root scheduling required','GT_opened':False,'model_loaded':False,'limitations':['sampled MiB values can miss transient peaks','16 MiB tensor is not total process/device memory','no model, performance or per-request claims']}
with (D/'plan001.json').open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
