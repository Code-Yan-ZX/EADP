"""Synthetic-only memory pilot component checks; never opens GPU or labels."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();files=[D/x for x in ('PLAN.md','summary.py','observe.py','test_summary.py','test_observe.py','model_run.py','test_model_run.py','worker.py','cpu_check.py')]
def sha(x):return hashlib.sha256(Path(x).read_bytes()).hexdigest()
before={str(x):sha(x) for x in files};env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-p','test_*.py','-v'],capture_output=True,text=True,env=env,timeout=60);after={str(x):sha(x) for x in files};a.out.mkdir(exist_ok=False);(a.out/'stderr.log').write_text(r.stderr);(a.out/'stdout.log').write_text(r.stdout)
result={'status':'passed' if r.returncode==0 and before==after else 'failed','source_sha256_before':before,'source_sha256_after':after,'tests_log_sha256':sha(a.out/'stderr.log'),'GPU_used':False,'GT_opened':False,'model_loaded':False,'limitations':['synthetic helpers only; no real NVML, CUDA, model or endpoint verified','no GPU launcher or approval exists']}
with (a.out/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
raise SystemExit(0 if result['status']=='passed' else 1)
