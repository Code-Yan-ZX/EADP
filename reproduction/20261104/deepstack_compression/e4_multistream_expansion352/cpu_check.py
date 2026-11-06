"""CPU-only source freeze for real-worker preflight; GPU remains locked."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 out=D/'cpu_check001.json'
 if out.exists():raise ValueError('existing CPU proof')
 c=json.loads((D/'contract_freeze001.json').read_text());s=json.loads((D/'selection001.json').read_text());prior=json.loads((B/'e5/selector_real32/cpu_check_005/verification.json').read_text())
 if c.get('status')!='frozen' or any(sha(k)!=v for k,v in c['source_sha256'].items()) or s.get('status')!='frozen' or len(s.get('pilot',[]))!=9 or prior.get('status')!='passed':raise ValueError('frozen inputs')
 # Prior capture source inventory plus all new entrypoints, selection and proofs.
 paths={Path(x).resolve() for x in prior['source_sha256_after']}|{x.resolve() for x in D.glob('*.py')}|{(D/x).resolve() for x in ('PLAN.md','plan_freeze001.json','selection001.json','contract_freeze001.json','pilot_cpu001.json')}
 paths|={(B/'e4_c_microbench/capture.py').resolve(),(B/'e4_c_microbench_pilot/capture_guard.py').resolve(),(B/'e4_c_microbench_pilot/identity_contract.py').resolve(),(B/'e5/selector_real32/panel_reference.py').resolve(),(B/'e4_c_latency/timing_launch.py').resolve(),(B/'monitor/supervise.py').resolve()}
 before={str(p):sha(p) for p in sorted(paths)}
 # Unlike historical source inventories, do not silently accept changes in the capture implementation.
 for k,v in prior['source_sha256_after'].items():
  if before[k]!=v:raise ValueError('accepted capture source changed: '+k)
 run=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-p','test_*.py','-v'],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
 after={str(p):sha(p) for p in sorted(paths)}
 if run.returncode or before!=after:raise ValueError('CPU tests or source drift '+run.stderr[-2000:])
 value={'status':'passed','source_sha256_before':before,'source_sha256_after':after,'tests':run.stdout+run.stderr,'GT_opened':False,'predictions_opened':False,'feature_values_opened':False,'score_values_opened':False,'GPU_used':False,'model_loaded':False,'scope':'CPU contracts and preflight source freeze; real container and GPU remain locked','errors':[]}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
