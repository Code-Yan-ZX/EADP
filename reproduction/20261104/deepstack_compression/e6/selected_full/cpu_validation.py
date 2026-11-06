"""Source-bound synthetic CPU tests; no prediction, GT or real model."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from freeze_plan import HERE,ROOT,SMOKE,REVIEW,sha
def sources():
 prev=json.loads((SMOKE/'smoke_001_sources/manifest.json').read_text())['source_sha256']
 files=list(HERE.glob('*.py'))+[HERE/'plan.json',HERE/'plan_001_pre_controls.json',REVIEW/'review_selected_smoke.py',ROOT/'monitor/supervise.py',ROOT/'m3_gpu_smoke/preflight_001/launch.json']
 for p,digest in prev.items():
  if sha(p)!=digest:raise ValueError('accepted smoke source changed: '+p)
  files.append(Path(p))
 return sorted(set(x.resolve() for x in files))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve();out.mkdir(exist_ok=False)
 paths=sources();before={str(p):sha(p) for p in paths};env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 tests=['test_full','test_full_contract','test_gate','test_imports','test_container_identity','test_generation_reference'];begin=time.time();reports=[]
 # Each module starts in a fresh interpreter: imported legacy test helpers
 # mutate sys.path and otherwise may shadow our identically named test files.
 for test in tests:
  cmd=[sys.executable,str(HERE/(test+'.py')),'-v']
  try:r=subprocess.run(cmd,cwd=HERE,env=env,capture_output=True,text=True,timeout=180);rc=r.returncode;stdout=r.stdout;stderr=r.stderr;timed=False
  except subprocess.TimeoutExpired as exc:rc=None;stdout=str(exc.stdout or '');stderr=str(exc.stderr or '');timed=True
  op=out/(test+'.stdout');ep=out/(test+'.stderr');op.write_text(stdout);ep.write_text(stderr)
  reports.append({'command':cmd,'returncode':rc,'timeout':timed,'stdout_sha256':sha(op),'stderr_sha256':sha(ep)})
 after={str(p):sha(p) for p in paths};ok=all(x['returncode']==0 and not x['timeout'] for x in reports) and before==after
 proof={'status':'passed' if ok else 'failed','source_sha256_before':before,'source_sha256_after':after,'tests':reports,
        'started_unix':begin,'finished_unix':time.time(),'python_version':sys.version,
        'GT_opened':False,'predictions_opened':False,
        'real_model_loaded':False,'gpu_used':False,'scope':'CPU-only synthetic full-panel and admission contracts'}
 with (out/'verification.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if ok else 1
if __name__=='__main__':sys.exit(main())
