"""Freeze and test execution and formal source without starting GPU."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
NAMES=('PLAN_delta002.md','PLAN_delta003.md','worker.py','gpu_gate.py','launch.py','gpu_dispatch.py','snapshot.py','verify.py','test_execution.py','test_formal_main.py','runtime_admit.py','runtime_probe.py','runtime_preflight_launch.py','host_entry_smoke004.py','host_entry_smoke004.json','cpu_check.py','freeze_runtime.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
before={str((D/n).resolve()):sha(D/n) for n in NAMES}
p=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-p','test_*.py'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120,check=False)
after={str((D/n).resolve()):sha(D/n) for n in NAMES}
if p.returncode or before!=after:raise RuntimeError('CPU tests or source mismatch: '+p.stdout+p.stderr)
with (D/'cpu_proof001.json').open('x') as f:json.dump({'status':'passed','source_sha256_before':before,'source_sha256_after':after,'test_stdout':p.stdout,'test_stderr':p.stderr,'GPU_used':False,'GT_opened':False,'new_predictions_opened':False,'errors':[]},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
