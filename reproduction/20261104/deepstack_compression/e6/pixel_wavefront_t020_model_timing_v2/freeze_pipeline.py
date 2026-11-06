import hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
s=read(D/'monitor_pipeline_cpu001/started.json');t=read(D/'monitor_pipeline_cpu001/terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and (Path(s['cwd'])/s['command'][1]).resolve()==D/'test_pipeline.py'
r={'status':'passed','scope':'CPU fake pipeline only','sources':{str((D/n).resolve()):sha(D/n) for n in ('contract.py','test_contract.py','admission.py','synthetic_pipeline.py','test_pipeline.py')},'design_sha256':sha(D/'design001.json'),'cpu_started_sha256':sha(D/'monitor_pipeline_cpu001/started.json'),'cpu_terminal_sha256':sha(D/'monitor_pipeline_cpu001/terminal.json'),'GT_opened':False,'predictions_opened':False,'model_loaded':False,'GPU_authorized':False,'limitations':['24 fake samples including one deliberate mismatch retained; no real timer or sampler.','Positive identity admission awaits future independently reviewed real identity proof.']}
with (D/'synthetic_pipeline_proof001.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
