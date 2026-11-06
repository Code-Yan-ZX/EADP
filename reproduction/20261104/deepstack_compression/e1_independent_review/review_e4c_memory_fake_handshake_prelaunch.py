"""Independent review of no-GPU fabricated sampler handshake only."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];H=R/'e4_c_memory_handshake';I=R/'e1_independent_review';sys.path.insert(0,str(H));import gate

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=argparse.ArgumentParser();a.add_argument('--out',required=True);out=Path(a.parse_args().out).resolve();assert out==(I/'e4c_memory_fake_handshake_prelaunch_review.json').resolve()
cpu=read(H/'cpu_check005/verification.json');s=read(H/'monitor_cpu005/started.json');t=read(H/'monitor_cpu005/terminal.json');plan=read(H/'launch_plan.json')
assert cpu['status']=='passed' and cpu['gpu_used'] is False and cpu['model_loaded'] is False and cpu['test_exit_code']==0
assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'] and s['cwd']==str(R) and s['command'][1:]==['e4_c_memory_handshake/cpu_check.py']
source=gate.sources();assert len(source)==9 and cpu['source_sha256_before']==cpu['source_sha256_after']==source
assert plan['scope']=='synthetic fake sampler and no-GPU Docker handshake only' and plan['gpu_authorized'] is False and plan['supervisor_timeout_seconds']==360 and plan['supervisor_grace_seconds']==90
upstream=gate.runtime.admit();gate.runtime.supervisor(gate.PRIOR_MON,gate.PRIOR_REVIEWER);prior=read(gate.PRIOR);assert prior['status']=='passed' and prior['independent_review'] is True and prior['gpu_authorized'] is False and prior['reviewer_sha256']==sha(gate.PRIOR_REVIEWER)
for k,v in {'launch_sha256':R/'e4_c_memory_runtime/container_preflight001/launch.json','exit_sha256':R/'e4_c_memory_runtime/container_preflight001/exit.json','worker_preflight_sha256':R/'e4_c_memory_runtime/container_preflight001/worker_preflight.json','started_sha256':R/'e4_c_memory_runtime/monitor_container_preflight001/started.json','terminal_sha256':R/'e4_c_memory_runtime/monitor_container_preflight001/terminal.json'}.items():assert prior[k]==sha(v)
env={**os.environ,'PYTHONPATH':str(H),'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run([sys.executable,'-m','unittest','-v','test_handshake'],cwd=R,env=env,capture_output=True,text=True,timeout=60);assert p.returncode==0 and 'Ran 10 tests' in p.stderr and 'OK' in p.stderr and gate.sources()==source
proof={'status':'passed','independent_review':True,'scope':'no-GPU fake handshake only','gpu_authorized':False,'GT_opened':False,'scoring_performed':False,'errors':[],'source_sha256':source,'cpu_sha256':sha(H/'cpu_check005/verification.json'),'cpu_started_sha256':sha(H/'monitor_cpu005/started.json'),'cpu_terminal_sha256':sha(H/'monitor_cpu005/terminal.json'),'prior_review_sha256':sha(gate.PRIOR),'reviewer_sha256':sha(__file__),'synthetic_tests':10,'limitation':'Fabricated 1 MiB samples; no real NVML or GPU memory claim.'}
with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
