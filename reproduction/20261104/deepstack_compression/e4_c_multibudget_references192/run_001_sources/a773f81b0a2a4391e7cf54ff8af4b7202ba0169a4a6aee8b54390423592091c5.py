"""Freeze and exercise fixed32 diagnostic CPU-only candidate; no GPU admission."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def sources():
 local=[HERE/x for x in ('fresh_fixed.py','light_fixed.py','fixed_common.py','compare_fixed.py','core.py','test_fixed.py','test_real_arm_schema.py','test_core.py','cpu_check.py')]
 deps=[BASE/x for x in ('e4/budget_runner.py','e4/budget_adapter.py','e4/off_adapter.py','e4/decode_observer.py','e4/identity_records.py','e4_c_budget_pilot_v2/budget_arm.py','e4_c_budget_pilot_v2/budget_capture.py','e4_c_budget_pilot_v2/adapter.py','e4_c_budget_pilot_v2/reference.py','e4_c_budget_pilot_v2/budget_compare.py','e4_c_budget_execution/run_001_sources/manifest.json','e1_independent_review/e4c_budget_identity001_review.json','e1_independent_review/monitor_e4c_budget_identity001/started.json','e1_independent_review/monitor_e4c_budget_identity001/terminal.json','e4_c_multibudget_expansion/pilot_plan001.json','e4_c_multibudget_expansion/fixed32_contract.py','monitor/supervise.py')]
 return local+deps

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
 if a.out.exists():raise ValueError('CPU output exists')
 before={str(x.resolve()):sha(x) for x in sources()};env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
 test=subprocess.run([sys.executable,'-m','unittest','-v','test_fixed','test_real_arm_schema','test_core'],cwd=HERE,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
 after={str(x.resolve()):sha(x) for x in sources()}
 if before!=after:raise ValueError('source changed during CPU test')
 result={'status':'passed' if test.returncode==0 else 'failed','test_returncode':test.returncode,'test_output':test.stdout,
         'source_sha256':before,'GT_opened':False,'predictions_opened':False,'scoring_performed':False,
         'cuda_visible_devices':'','model_loaded':False,'GPU_authorized':False,
         'scope':'synthetic CPU schema and failure tests only; no real fixed32 model capture; KV content not archived'}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if test.returncode:raise SystemExit(test.returncode)
 print('fixed32 CPU candidate tests passed',flush=True)
if __name__=='__main__':main()
