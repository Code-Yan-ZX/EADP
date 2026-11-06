import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=[D/n for n in ('PLAN.md','core.py','test_core.py','run.py','test_run_entry.py','cpu_check.py','freeze.py','inventory001.json')];before={str(p):sha(p) for p in files}
r=subprocess.run([sys.executable,'-m','unittest','-v','test_core','test_run_entry'],cwd=D,capture_output=True,text=True,timeout=120,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''});(D/'cpu001.log').write_text(r.stdout+r.stderr)
if r.returncode or any(sha(k)!=v for k,v in before.items()):raise ValueError('tests or source drift')
with (D/'cpu_proof001.json').open('x') as f:json.dump({'status':'passed','source_sha256':before,'GPU_used':False,'GT_opened':False,'real_tensors_deserialized':False,'cpu_log_sha256':sha(D/'cpu001.log')},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
