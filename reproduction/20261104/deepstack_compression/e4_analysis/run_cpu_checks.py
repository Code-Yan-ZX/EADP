"""CPU synthetic tests; records exact source identity and process result."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
score_e4_v2_sources=('panel_worker.py','panel_launch.py','panel_gate.py','panel_verify.py','native_panel_gate.py','panel_v2_imports.py','budget_plan.json')
score_e4_e4_sources=('smoke_contract.py','budget_runner.py','budget_adapter.py','decode_observer.py','off_adapter.py','skip_deepstack.py','identity_records.py','identity_worker.py','verify_k256.py')
FILES=[HERE/x for x in ('plan.json','paired.py','score_e4.py','test_e4_analysis.py','run_cpu_checks.py')]+[BASE/'e1_analysis'/x for x in ('completion.py','metrics.py','score.py')]+[BASE/'e4_panel_v2'/x for x in score_e4_v2_sources]+[BASE/'e4'/x for x in score_e4_e4_sources]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def main():
 out=HERE/'cpu_check004';out.mkdir(exist_ok=False);before={str(x):sha(x) for x in FILES};env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1');start=time.time()
 r=subprocess.run([sys.executable,'-m','unittest','-v','test_e4_analysis'],cwd=HERE,env=env,text=True,capture_output=True,timeout=120);(out/'stdout.log').write_text(r.stdout);(out/'stderr.log').write_text(r.stderr)
 after={str(x):sha(x) for x in FILES};ok=r.returncode==0 and before==after and 'Ran 12 tests' in r.stderr and r.stderr.rstrip().endswith('OK')
 d={'status':'passed' if ok else 'failed','returncode':r.returncode,'source_sha256_before':before,'source_sha256_after':after,'stdout_sha256':sha(out/'stdout.log'),'stderr_sha256':sha(out/'stderr.log'),'started_unix':start,'finished_unix':time.time(),'scope':'synthetic data only; no actual predictions or reference answers'}
 with (out/'verification.json').open('x') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':d['status'],'returncode':r.returncode}));return 0 if ok else 1
if __name__=='__main__':sys.exit(main())
