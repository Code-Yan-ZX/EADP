"""Freeze pure-CPU selection and sufficient-statistic contract, not a GPU runner."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=D/'contract_freeze001.json'
 if out.exists():raise ValueError('output exists')
 plan=json.loads((D/'plan_freeze001.json').read_text());selection=json.loads((D/'selection001.json').read_text());s=json.loads((D/'monitor_selection001/started.json').read_text());t=json.loads((D/'monitor_selection001/terminal.json').read_text())
 if plan.get('status')!='frozen' or selection.get('status')!='frozen' or selection.get('plan_freeze_sha256')!=sha(D/'plan_freeze001.json') or selection.get('runner_sha256')!=sha(D/'select_metadata.py') or selection.get('GT_opened') is not False or selection.get('predictions_opened') is not False or selection.get('feature_values_opened') is not False or selection.get('score_values_opened') is not False or len(selection['remaining'])!=352 or len(selection['pilot'])!=9 or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (Path(s['cwd'])/s['command'][1]).resolve()!=D/'select_metadata.py':raise ValueError('selection proof')
 paths=[D/'norm_contract.py',D/'test_norm_contract.py',D/'selection.py',D/'test_selection.py',D/'select_metadata.py',D/'freeze_contract.py',D/'plan_freeze001.json',D/'selection001.json',B/'e4_multistream_budget_diagnostic_v2/features.py']
 before={str(p):sha(p) for p in paths};tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-p','test_*.py','-v'],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
 if tests.returncode or before!={str(p):sha(p) for p in paths}:raise ValueError('test/source drift')
 value={'status':'frozen','source_sha256':before,'selection_started_sha256':sha(D/'monitor_selection001/started.json'),'selection_terminal_sha256':sha(D/'monitor_selection001/terminal.json'),'tests':tests.stdout+tests.stderr,'GT_opened':False,'predictions_opened':False,'GPU_used':False,'feature_values_opened':False,'score_values_opened':False,'scope':'CPU selection and norm contract only; no runner, preflight, model or GPU authorization','errors':[]}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
