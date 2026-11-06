"""Synthetic-only independent review of the post-hoc comparison source."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];H=R/'e6/selected_vs_eadp'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
a=argparse.ArgumentParser();a.add_argument('--out',required=True);o=Path(a.parse_args().out).resolve();assert o==(H/'independent_prelaunch_review.json').resolve()
cpu=read(H/'cpu_check_005/verification.json');s=read(H/'monitor_cpu005/started.json');t=read(H/'monitor_cpu005/terminal.json');plan=read(H/'plan_003.json')
assert cpu['status']=='passed' and cpu['GT_opened'] is False and cpu['predictions_opened'] is False and cpu['gpu_used'] is False
assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'] and s['cwd']==str(R)
assert s['command'][1:]==['e6/selected_vs_eadp/cpu_validation.py','--out','e6/selected_vs_eadp/cpu_check_005']
required={str(H/n) for n in ('plan.json','plan_002.json','plan_003.json','compare.py','test_compare.py','cpu_validation.py')};src=cpu['source_sha256_before'];assert src==cpu['source_sha256_after'] and set(src)==required and all(sha(p)==h for p,h in src.items())
assert sha(H/'cpu_check_005/stderr.log')==cpu['stderr_sha256'] and sha(H/'plan.json')==plan['prior_plan_sha256'] and sha(H/'plan_002.json')==plan['revision_002_sha256']
assert plan['numpy_version']=='2.2.6' and plan['questions']==384 and plan['bootstrap']=={'replicates':10000,'seed':20261104,'unit':'image_group'}
assert plan['GT_opened'] is False and plan['new_predictions_opened'] is False
p=subprocess.run([sys.executable,str(H/'test_compare.py')],capture_output=True,text=True,timeout=90);assert p.returncode==0 and 'Ran 5 tests' in p.stderr and 'OK' in p.stderr
assert all(sha(p)==h for p,h in src.items())
proof={'status':'passed','independent_review':True,'source_sha256':src,'reviewer_sha256':sha(__file__),'cpu_sha256':sha(H/'cpu_check_005/verification.json'),'cpu_started_sha256':sha(H/'monitor_cpu005/started.json'),'cpu_terminal_sha256':sha(H/'monitor_cpu005/terminal.json'),'plan_sha256':sha(H/'plan_003.json'),'synthetic_tests':5,'GT_opened':False,'predictions_opened':False,'gpu_used':False,'limitations':plan['constraints'],'errors':[]}
with o.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
