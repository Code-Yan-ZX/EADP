"""Source-frozen synthetic batch-2 CPU tests; no prediction or answer reads."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(HERE))
import batch_gate
def sources():
 old=json.loads((ROOT/'m3_oracle_exec/cpu_check_009/verification.json').read_text())['source_sha256_after']
 for p,d in old.items():
  if batch_gate.sha(p)!=d:raise ValueError('first batch source changed: '+p)
 plan=json.loads((HERE/'execution_plan_002.json').read_text());files=list(HERE.glob('*.py'))+[HERE/'execution_plan_002.json']+[Path(p) for p in plan['source_sha256']]+[Path(p) for p in old]+[ROOT/'m3_oracle_verify_v2/environment_002_v2.json',Path(sys.executable)]
 return sorted(set(p.resolve() for p in files))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve();out.mkdir(exist_ok=False)
 files=sources();before={str(p):batch_gate.sha(p) for p in files};env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');begin=time.time()
 try:r=subprocess.run([sys.executable,'-m','unittest','-v','test_lock','test_gate'],cwd=HERE,env=env,capture_output=True,text=True,timeout=120);rc=r.returncode;stdout=r.stdout;stderr=r.stderr;timed=False
 except subprocess.TimeoutExpired as exc:rc=None;stdout=str(exc.stdout or '');stderr=str(exc.stderr or '');timed=True
 (out/'stdout.log').write_text(stdout);(out/'stderr.log').write_text(stderr);after={str(p):batch_gate.sha(p) for p in files};ok=rc==0 and not timed and before==after
 proof={'status':'passed' if ok else 'failed','returncode':rc,'timed_out':timed,'source_sha256_before':before,'source_sha256_after':after,'stdout_sha256':batch_gate.sha(out/'stdout.log'),'stderr_sha256':batch_gate.sha(out/'stderr.log'),
        'GT_opened':False,'predictions_opened':False,'real_model_loaded':False,'gpu_used':False,'started_unix':begin,'finished_unix':time.time(),'scope':'synthetic CPU only; no second-batch prediction or label'}
 with (out/'verification.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if ok else 1
if __name__=='__main__':sys.exit(main())
