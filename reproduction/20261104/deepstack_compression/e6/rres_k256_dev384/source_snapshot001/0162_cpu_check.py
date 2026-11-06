"""Re-run synthetic tests with source before/after; no data or GPU execution."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
NAMES=('PLAN.md','design.py','freeze_design.py','design001.json','test_design.py','processor_contract.py','test_processor_contract.py','panel_core.py','test_panel_core.py','probe_core.py','test_probe_core.py','pilot_gate.py','identity_wait.py','test_identity_wait.py','preflight_gate.py','test_gate.py','probe_processor.py','preflight_worker.py','preflight_launch.py','cpu_check.py','freeze_preflight.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
before={str((D/n).resolve()):sha(D/n) for n in NAMES}
p=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-p','test_*.py'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=100,check=False)
after={str((D/n).resolve()):sha(D/n) for n in NAMES}
if p.returncode or before!=after:raise RuntimeError('CPU tests or source mismatch: '+p.stdout+p.stderr)
out={'status':'passed','source_sha256_before':before,'source_sha256_after':after,'test_stdout':p.stdout,'test_stderr':p.stderr,'GPU_used':False,'GT_opened':False,'new_predictions_opened':False,'errors':[]}
with (D/'cpu_proof001.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
