"""CPU-only tests before an independently reviewed noGPU preflight."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];R=B/'e6/pixel_wavefront_t020_timing_runtime_v4';C=B/'e6/pixel_wavefront_t020_model_timing_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sources():
 files=list(D.glob('*.py'))+[R/x for x in ('worker.py','host_runner.py','controller.py','bridge.py','nvml_stages.py')]+[C/'design001.json',B/'e6/pixel_wavefront_t020_timing_controller_draft/job_contract.py']
 return {str(p.resolve()):sha(p) for p in files}
def main():
 before=sources();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');r=subprocess.run([sys.executable,'-m','unittest','-v','test_locks','test_raw_prefix','test_preflight_gate','test_final_main'],cwd=D,env=env,capture_output=True,text=True,timeout=60);after=sources();(D/'cpu_test001.log').write_text(r.stdout+r.stderr)
 value={'status':'passed' if r.returncode==0 and before==after else 'failed','scope':'final candidate source noGPU CPU tests, GPU not authorized','source_sha256_before':before,'source_sha256_after':after,'GPU_used':False,'GPU_authorized':False,'GT_opened':False,'predictions_opened':True,'prediction_scope':'existing accepted DEV mixed journals hashed or parsed for identity only'}
 with (D/'cpu_proof001.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if value['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
