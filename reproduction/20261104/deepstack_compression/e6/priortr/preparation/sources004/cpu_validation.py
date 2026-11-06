"""Run synthetic real-model/contract tests and freeze exact tested sources."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;INSTALL=HERE.parent/'install001'
NAMES=('admission.py','preprocess.py','trace.py','smoke_worker.py','smoke_launch.py','smoke_verify.py','test_trace.py','test_env_contract.py','test_launch.py','env_contract.py','cpu_validation.py')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 before={n:sha(HERE/n) for n in NAMES};env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1');start=time.time()
 r=subprocess.run([sys.executable,'-m','unittest','-v','test_trace','test_env_contract','test_launch'],cwd=HERE,env=env,text=True,capture_output=True,timeout=120)
 (HERE/'cpu_validation004.stdout').write_text(r.stdout);(HERE/'cpu_validation004.stderr').write_text(r.stderr)
 after={n:sha(HERE/n) for n in NAMES};passed=r.returncode==0 and before==after and 'Ran 3 tests' in r.stderr and r.stderr.rstrip().endswith('OK')
 d={'status':'passed' if passed else 'failed','returncode':r.returncode,'source_sha256_before':before,'source_sha256_after':after,'plan_sha256':sha(HERE/'smoke_single_plan002.json'),'processor_proof_sha256':sha(HERE/'processor_check002.json'),'env_fingerprint_sha256':sha(INSTALL/'env_fingerprint.json'),'environment_provenance_sha256':sha(INSTALL/'provenance.json'),'stdout_sha256':sha(HERE/'cpu_validation004.stdout'),'stderr_sha256':sha(HERE/'cpu_validation004.stderr'),'started_unix':start,'finished_unix':time.time(),'scope':'CPU synthetic tiny multimodal, env mutation, bounded-launch exit contracts; no model weights or GPU'}
 with (HERE/'cpu_validation004.json').open('x') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':d['status'],'returncode':r.returncode}));return 0 if passed else 1
if __name__=='__main__':sys.exit(main())
