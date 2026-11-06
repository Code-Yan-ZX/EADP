"""CPU-only run draft checks; does not authorize Docker or GPU."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];R=B/'e6/pixel_wavefront_t020_timing_runtime_v4';C=B/'e6/pixel_wavefront_t020_model_timing_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sources():
 files=[D/x for x in ('gpu_worker.py','gpu_launch.py','test_entry_locks.py','test_worker_success.py','preflight_gate.py','preflight_entry.py','preflight_launch.py','test_preflight_gate.py','cpu_check001.py')]+[R/x for x in ('worker.py','host_runner.py','controller.py','bridge.py','nvml_stages.py')]+[C/'design001.json',B/'e6/pixel_wavefront_t020_timing_controller_draft/job_contract.py']
 return {str(p.resolve()):sha(p) for p in files}
def main():
 before=sources();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
 result=subprocess.run([sys.executable,'-m','unittest','-v','test_entry_locks','test_worker_success','test_preflight_gate'],cwd=D,env=env,capture_output=True,text=True,timeout=60)
 after=sources();(D/'cpu_test001.log').write_text(result.stdout+result.stderr)
 value={'status':'passed' if result.returncode==0 and before==after else 'failed','scope':'draft 24-job launcher fake-Docker tests; no GPU admission','source_sha256_before':before,'source_sha256_after':after,'returncode':result.returncode,'GPU_used':False,'GPU_authorized':False,'GT_opened':False,'predictions_opened':False}
 with (D/'cpu_proof001.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if value['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
