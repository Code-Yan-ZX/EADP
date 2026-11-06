"""Run fake CPU tests in separate processes to avoid contract.py collisions."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from admission import HERE,sources,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
 if not a.run_id.isalnum():raise ValueError('invalid run ID')
 out=HERE/('cpu_check'+a.run_id);out.mkdir(exist_ok=False);before={str(p):sha(p) for p in sources()}
 suites=[('sampler',['test_contract','test_barrier','test_host_evidence','test_sampler_io'],13),('worker',['test_plan','test_worker','test_worker_preflight','test_admission'],10)]
 for name,modules,count in suites:
  env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';proc=subprocess.run([sys.executable,'-m','unittest','-v',*modules],cwd=HERE,env=env,text=True,capture_output=True,timeout=120)
  (out/(name+'_stdout.log')).write_text(proc.stdout);(out/(name+'_stderr.log')).write_text(proc.stderr)
  if proc.returncode or ('Ran '+str(count)+' tests') not in proc.stderr:raise ValueError('CPU fake test suite failed: '+name)
 after={str(p):sha(p) for p in sources()}
 if before!=after:raise ValueError('source changed during CPU fake tests')
 r={'status':'passed','tests':23,'source_sha256_before':before,'source_sha256_after':after,'sampler_stderr_sha256':sha(out/'sampler_stderr.log'),'worker_stderr_sha256':sha(out/'worker_stderr.log'),'GT_opened':False,'predictions_opened':False,'gpu_used':False,'model_loaded':False}
 with (out/'verification.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed','tests':23}))
if __name__=='__main__':main()
