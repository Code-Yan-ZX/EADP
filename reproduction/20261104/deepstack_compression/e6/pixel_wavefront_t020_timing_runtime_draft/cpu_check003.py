"""CPU-only bridge/preflight regression proof. No real Docker or GPU."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1]
NAMES=('worker.py','host_runner.py','controller.py','nvml_stages.py','bridge.py','preflight_gate.py','preflight_entry.py','preflight_launch.py','test_runtime.py','test_nvml_stages.py','test_bridge.py','test_preflight_gate.py','cpu_check003.py','PLAN_DRAFT.md')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sources():
 paths=[D/n for n in NAMES]+[B/'e6/pixel_wavefront_t020_timing_controller_draft/job_contract.py',B/'e6/pixel_wavefront_t020_model_timing_v2/nvml_contract.py',B/'e6/pixel_wavefront_t020_model_timing_v2/model_candidate.py']
 return {str(p.resolve()):sha(p) for p in paths}
def main():
 out=D/'cpu_proof003.json'
 if out.exists():raise ValueError('existing proof')
 before=sources();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
 run=subprocess.run([sys.executable,'-m','unittest','-v','test_runtime','test_nvml_stages','test_bridge','test_preflight_gate'],cwd=D,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
 after=sources();(D/'cpu_test003.log').write_text(run.stdout)
 value={'status':'passed' if run.returncode==0 and before==after else 'failed','scope':'CPU-only draft bridge, admission lock and interfaces; no Docker or GPU admission','source_sha256_before':before,'source_sha256_after':after,'tests_returncode':run.returncode,'GPU_authorized':False,'GPU_used':False,'GT_opened':False,'predictions_opened':False,'model_loaded':False}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if value['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
