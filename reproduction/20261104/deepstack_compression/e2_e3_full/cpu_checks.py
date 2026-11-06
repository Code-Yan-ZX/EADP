"""Bounded CPU tests and source hashes for locked full-panel preparation."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from full_imports import local
HERE=Path(__file__).resolve().parent;sha=local('panel_gate').sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir(exist_ok=False)
 sources=local('source_inventory').sources();before={path:sha(path) for path in sources};probe=json.loads((HERE/'probe_smoke001.json').read_text());pt=json.loads((HERE/'monitor_smokeprobe001/terminal.json').read_text())
 local('panel_gate').validate_probe(probe,HERE.parent/'e2_e3_smoke/run_001_verification.json',HERE/'validate_panel_record.py',HERE/'probe_smoke.py')
 if pt.get('reason')!='completed' or pt.get('worker_returncode')!=0 or pt.get('remaining_process_group')!=[]:raise ValueError('real smoke checker probe missing')
 upstream={'smoke':local('panel_gate').smoke_gate(),'e4b':local('panel_gate').b_gate()}
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');cmd=[sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'];start=time.time()
 try:r=subprocess.run(cmd,cwd=HERE,env=env,text=True,capture_output=True,timeout=180);code=r.returncode;stdout=r.stdout;stderr=r.stderr;timed=False
 except subprocess.TimeoutExpired as e:code=None;stdout=str(e.stdout or '');stderr=str(e.stderr or '');timed=True
 (out/'stdout.log').write_text(stdout);(out/'stderr.log').write_text(stderr);after={path:sha(path) for path in sources}
 passed=code==0 and not timed and before==after and 'Ran 23 tests' in stderr and stderr.rstrip().endswith('OK')
 result={'status':'passed' if passed else 'failed','returncode':code,'timed_out':timed,'source_sha256_before':before,'source_sha256_after':after,'stdout_sha256':sha(out/'stdout.log'),'stderr_sha256':sha(out/'stderr.log'),'probe_sha256':sha(HERE/'probe_smoke001.json'),'probe_terminal_sha256':sha(HERE/'monitor_smokeprobe001/terminal.json'),'upstream':upstream,'started_unix':start,'finished_unix':time.time(),'GT_opened':False,'scoring_performed':False,'gpu_used':False}
 with (out/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':result['status']}),flush=True);return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
