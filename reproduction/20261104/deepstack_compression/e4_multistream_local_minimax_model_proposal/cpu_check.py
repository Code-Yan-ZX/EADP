"""Bounded CPU proof for the preliminary candidate; no GPU/model authority."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def main():
 if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')) or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('not single-thread CPU environment')
 if (D/'cpu_proof001.json').exists():raise ValueError('proof exists')
 c=read(D/'cohort001.json');s=read(D/'monitor_cohort001/started.json');t=read(D/'monitor_cohort001/terminal.json')
 if c.get('status')!='passed' or c.get('images')!=32 or c.get('jobs')!=64 or c.get('errors')!=[] or c.get('GT_opened') is not False or c.get('real_feature_archives_opened') is not False or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid'):raise ValueError('cohort proof/exit')
 cwd=Path(s['cwd']);cmd=s['command'];expected=(D/'cohort.py').resolve()
 if [(cwd/x).resolve() for x in cmd if Path(x).name=='cohort.py']!=[expected] or cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=(D/'cohort001.json').resolve():raise ValueError('cohort command')
 sources=[D/'PLAN.md',D/'cohort.py',D/'cohort001.json',D/'arm.py',D/'test_cpu.py',Path(__file__),B/'m3_oracle_exec/selection_adapter.py',B/'e4/budget_runner.py',B/'e4/budget_adapter.py',B/'e4/off_adapter.py',B/'e4/skip_deepstack.py',B/'e4/identity_records.py',B/'e4/decode_observer.py',B/'e4/test_off_adapter.py',B/'e4/test_skip_deepstack.py']
 before={str(p):sha(p) for p in sources}
 p=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-p','test_cpu.py','-v'],capture_output=True,text=True,timeout=90)
 if p.returncode or 'Ran 12 tests' not in p.stderr or '\nOK\n' not in p.stderr:raise RuntimeError('CPU tests failed: '+p.stderr[-3000:])
 if {str(x):sha(x) for x in sources}!=before:raise ValueError('source drift')
 q={'status':'passed','scope':'preliminary CPU preparation only; not GPU or model execution admission', 'tests':12,'images':32,'jobs':64,'GT_opened':False,'GPU_used':False,'model_executed':False,'real_feature_archives_opened':False,'cohort_started_sha256':sha(D/'monitor_cohort001/started.json'),'cohort_terminal_sha256':sha(D/'monitor_cohort001/terminal.json'),'source_sha256':before,'limitations':['Simulated candidate entry and real selected-adapter CPU hook chain; actual model generation not tested.', 'Future worker, actual source-pinned imports, raw output preservation, GPU admission, container preflight, formal/independent acceptance and score gate remain to be implemented and reviewed.'], 'errors':[]}
 with (D/'cpu_proof001.json').open('x') as f:json.dump(q,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(p.stderr)
if __name__=='__main__':main()
