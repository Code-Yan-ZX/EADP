import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=[D/x for x in ('PLAN.md','freeze_plan.py','freeze_code.py','gate.py','stats.py','analyze.py','test_stats.py','test_gate.py')];before={str(p):sha(p) for p in paths};r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-p','test_*.py','-v'],text=True,capture_output=True,timeout=180);after={str(p):sha(p) for p in paths}
if r.returncode!=0 or before!=after:raise ValueError('CPU tests/source changed: '+r.stdout+r.stderr)
cpu={'status':'passed','source_before':before,'source_after':after,'tests':r.stdout+r.stderr,'GT_opened':False,'new_343_values_opened':False,'scoring_performed':False,'GPU_used':False}
with (D/'cpu_proof001.json').open('x') as f:json.dump(cpu,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
code={'status':'frozen','source_sha256':after,'plan_sha256':sha(D/'plan_freeze001.json'),'cpu_sha256':sha(D/'cpu_proof001.json')}
with (D/'code_plan001.json').open('x') as f:json.dump(code,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
