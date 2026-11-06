"""Freeze and test generic baseline P3/DS diagnostics on synthetic CPU data."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];AUTHOR=BASE.parents[2]/'_reproduction/author_eadp/Qwen_vl'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def main():
 out=HERE/'cpu_check_003';out.mkdir(exist_ok=False)
 files=[HERE/x for x in ('adapter.py','recover_indices.py','test_adapter.py','test_recover_indices.py','cpu_checks.py')]+[BASE/'e4'/x for x in ('off_adapter.py','budget_adapter.py','skip_deepstack.py')]+[AUTHOR/x for x in ('model/cdpruner.py','model/divpruner.py','VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py')]
 before={str(x.resolve()):sha(x) for x in files};env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');start=time.time()
 try:r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],env=env,text=True,capture_output=True,timeout=120);code=r.returncode;stdout=r.stdout;stderr=r.stderr;timed=False
 except subprocess.TimeoutExpired as exc:code=None;stdout=str(exc.stdout or '');stderr=str(exc.stderr or '');timed=True
 (out/'stdout.log').write_text(stdout);(out/'stderr.log').write_text(stderr);after={str(x.resolve()):sha(x) for x in files};ok=code==0 and not timed and before==after and 'Ran 4 tests' in stderr and stderr.rstrip().endswith('OK')
 result={'status':'passed' if ok else 'failed','returncode':code,'timed_out':timed,'source_sha256_before':before,'source_sha256_after':after,'stdout_sha256':sha(out/'stdout.log'),'stderr_sha256':sha(out/'stderr.log'),'GT_opened':False,'scoring_performed':False,'gpu_used':False,'start_unix':start,'end_unix':time.time(),'limits':['Synthetic CPU only; actual model hooks, selector identity, P3/DS, KV and generation need GPU smoke.','Cannot interpret GPU speed or benchmark quality.']}
 with (out/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':result['status']}));return 0 if ok else 1
if __name__=='__main__':raise SystemExit(main())
