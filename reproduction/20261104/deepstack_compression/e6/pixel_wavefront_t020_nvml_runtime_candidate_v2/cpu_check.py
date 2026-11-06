"""Synthetic adapted PID1/NVML protocol tests; no actual GPU or Docker."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
TESTS=['test_contract','test_barrier','test_sampler_io','test_host_evidence','test_lifecycle','test_worker_barrier','test_controller_adapter','test_interruptible','test_docker_guard','test_sampler_supervisor','test_controller_io','test_nvml_source','test_guard','test_pixel_modes','test_atomic_publish']
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.glob('*.py'))}
before=hashes();r=subprocess.run([sys.executable,'-m','unittest','-v',*TESTS],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120);after=hashes();(D/'cpu_test001.log').write_text(r.stdout+r.stderr)
v={'status':'passed' if r.returncode==0 and before==after else 'failed','source_sha256_before':before,'source_sha256_after':after,'GPU_used':False,'model_loaded':False,'GT_opened':False,'predictions_opened':False,'scope':'synthetic Pixel-mode lifecycle only, not real GPU admission','prior_test_failure':'FAILED_CPU001.md'}
with (D/'cpu_proof001.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if v['status']!='passed':raise SystemExit(1)
