"""Bounded CPU-only evidence for runtime draft, not an admission proof."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
NAMES=('worker.py','host_runner.py','controller.py','nvml_stages.py','test_runtime.py','test_nvml_stages.py','cpu_check.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sources():return {str((D/n).resolve()):sha(D/n) for n in NAMES}
def main():
 out=D/'cpu_proof001.json'
 if out.exists():raise ValueError('existing proof')
 before=sources();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
 run=subprocess.run([sys.executable,'-m','unittest','-v','test_runtime','test_nvml_stages'],cwd=D,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
 after=sources();(D/'cpu_test001.log').write_text(run.stdout)
 value={'status':'passed' if run.returncode==0 and before==after else 'failed','scope':'CPU-only draft interfaces; not GPU or no-GPU container admission','source_sha256_before':before,'source_sha256_after':after,'tests_returncode':run.returncode,'GPU_authorized':False,'GPU_used':False,'GT_opened':False,'predictions_opened':False,'model_loaded':False}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if value['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
