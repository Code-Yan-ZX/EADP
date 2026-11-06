"""Freeze analysis source and pure-synthetic tests before timing execution."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;EX=B/'e4_c_multibudget_expansion'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 cpu=D/'cpu_proof001.json';plan=D/'analysis_code_plan001.json'
 if cpu.exists() or plan.exists():raise ValueError('already frozen')
 files=[D/x for x in ('analysis_gate.py','report.py','metrics.py','test_metrics.py','test_gate.py','test_report.py','freeze.py')]+[EX/'ANALYSIS_PLAN.md',EX/'analysis_plan_freeze001.json',EX/'monitor_freeze_analysis001/started.json',EX/'monitor_freeze_analysis001/terminal.json',EX/'pilot_plan001.json']
 before={str(x.resolve()):sha(x) for x in files};f=json.loads((EX/'analysis_plan_freeze001.json').read_text());t=json.loads((EX/'monitor_freeze_analysis001/terminal.json').read_text());st=json.loads((EX/'monitor_freeze_analysis001/started.json').read_text())
 if f.get('status')!='frozen' or f.get('analysis_plan_sha256')!=sha(EX/'ANALYSIS_PLAN.md') or f.get('pilot_plan_sha256')!=sha(EX/'pilot_plan001.json') or f.get('timing_results_opened') is not False or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or st.get('worker_pid')!=t.get('worker_pid'):raise ValueError('analysis plan freeze invalid')
 import numpy as np
 if np.__version__!='2.2.6':raise ValueError('frozen NumPy 2.2.6 required')
 r=subprocess.run([sys.executable,'-m','unittest','-v','test_metrics','test_gate','test_report'],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True,timeout=120);after={str(x.resolve()):sha(x) for x in files};status='passed' if r.returncode==0 and before==after else 'failed'
 p={'status':status,'returncode':r.returncode,'test_output':r.stdout+r.stderr,'source_sha256_before':before,'source_sha256_after':after,'numpy_version':np.__version__,'GT_opened':False,'timing_results_opened':False,'GPU_authorized':False}
 with cpu.open('x') as f:json.dump(p,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if status!='passed':raise SystemExit(1)
 sources=dict(after);sources[str(cpu.resolve())]=sha(cpu)
 with plan.open('x') as f:json.dump({'status':'frozen','source_sha256':sources,'GT_opened':False,'timing_results_opened':False,'GPU_authorized':False,'scope':'exploratory 32-image analysis code; complete formal and independent GPU-result acceptance required before journal access'},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('analysis source frozen, no timing results opened')
if __name__=='__main__':main()
