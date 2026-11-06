"""Test locked CLIs and fake worker/controller/scheduler; no model or GPU."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.glob('*.py'))}
before=hashes();r=subprocess.run([sys.executable,'-m','unittest','-v','test_locks','test_worker_flow','test_controller_flow','test_scheduler','test_recovery'],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120);after=hashes();(D/'cpu_test001.log').write_text(r.stdout+r.stderr)
v={'status':'passed' if r.returncode==0 and before==after else 'failed','source_sha256_before':before,'source_sha256_after':after,'GPU_used':False,'model_loaded':False,'GT_opened':False,'predictions_opened':False,'limitations':['fake components only; executable gate remains hard locked','no actual GPU, sampler process or Docker lifecycle']}
with (D/'cpu_proof001.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if v['status']!='passed':raise SystemExit(1)
