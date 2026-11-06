"""Synthetic-only source review; permits CPU preflight, never a GPU run."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];H=R/'e4_c_memory_runtime';I=R/'e1_independent_review'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
a=argparse.ArgumentParser();a.add_argument('--out',required=True);out=Path(a.parse_args().out).resolve();assert out==(I/'e4c_memory_runtime_preflight_review.json').resolve()
spec=importlib.util.spec_from_file_location('audit_memory_runtime_admission',H/'runtime_admission.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
cpu=read(H/'cpu_check001/verification.json');s=read(H/'monitor_runtime_cpu001/started.json');t=read(H/'monitor_runtime_cpu001/terminal.json');plan=read(H/'preflight_plan.json')
assert cpu['status']=='passed' and cpu['tests']==48 and all(cpu[k] is False for k in ('gpu_used','model_loaded','GT_opened','scoring_performed'))
assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'] and s['cwd']==str(R) and s['command'][1:]==['e4_c_memory_runtime/runtime_cpu_check.py','--run-id','001']
assert sha(H/'cpu_check001/stderr.log')==cpu['stderr_sha256'];gate.prior();src=gate.source_hashes();assert src==cpu['source_sha256_before']==cpu['source_sha256_after'] and len(src)==88
assert plan['gpu_authorized'] is False and plan['inner_timeout_seconds']<plan['outer_timeout_seconds'] and plan['grace_seconds']>0
assert 'if __name__' in (H/'worker_candidate.py').read_text() and 'independently reviewed worker/controller/sampler admission is not implemented' in (H/'worker_candidate.py').read_text()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1');p=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(H),'-v'],cwd=R,env=env,capture_output=True,text=True,timeout=120)
assert p.returncode==0 and 'Ran 48 tests' in p.stderr and 'OK' in p.stderr and gate.source_hashes()==src
proof={'status':'passed','independent_review':True,'scope':'memory_runtime_cpu_preflight_only','gpu_authorized':False,'GT_opened':False,'scoring_performed':False,'errors':[],'source_sha256':src,'cpu_proof_sha256':sha(H/'cpu_check001/verification.json'),'cpu_started_sha256':sha(H/'monitor_runtime_cpu001/started.json'),'cpu_terminal_sha256':sha(H/'monitor_runtime_cpu001/terminal.json'),'prior_review_sha256':sha(I/'e4c_memory_preflight001_review.json'),'reviewer_sha256':sha(__file__),'synthetic_tests':48,'limitations':plan['limitations']}
with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
