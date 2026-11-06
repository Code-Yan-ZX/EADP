"""Independent no-GPU preflight source audit for 248 fixed references."""
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path
B=Path(__file__).resolve().parents[1];D=B/'e4_c_multibudget_references';I=B/'e1_independent_review';OUT=I/'e4c_multibudget_references_worker_preflight_review.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def run():
 path=D/'gate.py';spec=importlib.util.spec_from_file_location('_independent_multibudget_reference_gate',path);gate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=gate;spec.loader.exec_module(gate)
 plan=gate.plan();prior=gate.prior();cpu=gate.execution_cpu(plan);cpu_path=D/'execution_cpu003.json';cs=D/'monitor_execution_cpu003/started.json';ct=D/'monitor_execution_cpu003/terminal.json'
 if (plan.get('jobs')!=248 or plan.get('GT_opened') is not False or cpu.get('status')!='passed' or cpu.get('predictions_opened') is not True or cpu.get('GPU_authorized') is not False or cpu.get('plan_sha256')!=sha(D/'execution_plan003.json') or 'Ran 12 tests' not in cpu.get('test_output','')):raise ValueError('plan/CPU test')
 try:gate.admit_cpu_preflight(plan)
 except ValueError as exc:
  if 'review missing' not in str(exc):raise
 else:raise ValueError('unreviewed CPU container not locked')
 try:gate.admit_fixed_identity()
 except ValueError as exc:
  if 'GPU locked' not in str(exc):raise
 else:raise ValueError('GPU fixed identity is not locked')
 if not prior.get('review_sha256'):raise ValueError('upstream identity missing')
 return {'status':'passed','independent_review':True,'scope':'references_cpu_preflight_only','source_sha256':plan['source_sha256'],'plan_sha256':sha(D/'execution_plan003.json'),'execution_cpu_sha256':sha(cpu_path),'execution_cpu_started_sha256':sha(cs),'execution_cpu_terminal_sha256':sha(ct),'reviewer_sha256':sha(__file__),'GT_opened':False,'predictions_opened':True,'scoring_performed':False,'errors':[],'limitations':['Only source-bound no-GPU worker/processor preflight; it hashes prior mixed DEV predictions.','All 12 fixed32 identity pairs and full 248-reference result still require actual model generation and separate formal/independent reviews; GPU remains locked.']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=OUT:raise ValueError('unexpected output')
 v=run()
 with a.out.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
