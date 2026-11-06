"""Synthetic runtime/controller tests only; never initializes CUDA."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import runtime_admission as a
HERE=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);arg=p.parse_args()
 if not arg.run_id.isalnum():raise ValueError('run ID')
 out=HERE/('cpu_check'+arg.run_id);out.mkdir(exist_ok=False);a.prior();before=a.source_hashes();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
 proc=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-v'],cwd=HERE.parent,env=env,text=True,capture_output=True,timeout=120)
 (out/'stdout.log').write_text(proc.stdout);(out/'stderr.log').write_text(proc.stderr);after=a.source_hashes()
 if proc.returncode or 'Ran 48 tests' not in proc.stderr or before!=after:raise ValueError('runtime CPU fake tests or source identity')
 result={'status':'passed','tests':48,'source_sha256_before':before,'source_sha256_after':after,'gpu_used':False,'model_loaded':False,'GT_opened':False,'scoring_performed':False,'stderr_sha256':a.sha(out/'stderr.log'),'scope':'synthetic-only runtime and lifecycle tests'}
 with (out/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed','tests':48,'source_count':len(before)}))
if __name__=='__main__':main()
