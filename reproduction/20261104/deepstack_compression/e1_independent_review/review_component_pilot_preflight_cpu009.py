"""Independent synthetic review for CPU component-worker preflight only."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];H=R/'e4_c_microbench_pilot';I=R/'e1_independent_review';sys.path.insert(0,str(H));import pilot_admission as gate

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=argparse.ArgumentParser();a.add_argument('--out',required=True);out=Path(a.parse_args().out).resolve();assert out==(H/'independent_preflight_review.json').resolve()
cpu=read(H/'cpu_check_009/verification.json');s=read(H/'monitor_cpu009/started.json');t=read(H/'monitor_cpu009/terminal.json')
assert cpu['status']=='passed' and cpu['gpu_used'] is False and cpu['GT_opened'] is False and cpu['model_loaded'] is False
assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'] and s['cwd']==str(R) and s['command'][1:]==['e4_c_microbench_pilot/cpu_validation.py','--out','e4_c_microbench_pilot/cpu_check_009']
source={str(p):sha(p) for p in gate.sources()};assert len(source)==58 and cpu['source_sha256_before']==cpu['source_sha256_after']==source and cpu['stderr_sha256']==sha(H/'cpu_check_009/stderr.log')
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(H),'-p','test_*.py','-v'],cwd=R,env=env,capture_output=True,text=True,timeout=90);assert p.returncode==0 and 'Ran 16 tests' in p.stderr and 'OK' in p.stderr and {str(q):sha(q) for q in gate.sources()}==source
proof={'status':'passed','independent_review':True,'preflight_only':True,'reviewer_sha256':sha(__file__),'source_sha256':source,'cpu_sha256':sha(H/'cpu_check_009/verification.json'),'cpu_started_sha256':sha(H/'monitor_cpu009/started.json'),'cpu_terminal_sha256':sha(H/'monitor_cpu009/terminal.json'),'GT_opened':False,'gpu_authorized':False,'synthetic_tests':16,'limitations':['Only CPU container preflight is authorized; no GPU measurement.','DEV mixed prediction journal is parsed for accepted identity fields by real preflight; no GT is opened.','DS staging excludes decoder addition and does not represent total DS cost.'],'errors':[]}
with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
