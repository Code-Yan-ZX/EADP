"""Bind corrected candidate and reuse previously completed archive byte inventory."""
import datetime,hashlib,importlib.metadata,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;OLD=B/'e4_multistream_local_minimax';I=B/'e1_independent_review';E=B/'e4_c_microbench_full';END=1794065414
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(mon,script):
 a=read(mon/'started.json');b=read(mon/'terminal.json');cwd=Path(a['cwd'])
 if a.get('worker_pid')!=b.get('worker_pid') or b.get('reason')!='completed' or b.get('worker_returncode')!=0 or b.get('remaining_process_group')!=[] or [(cwd/x).resolve() for x in a['command'] if Path(x).name==script.name]!=[script.resolve()]:raise ValueError('supervisor')
if datetime.datetime.now(datetime.timezone.utc).timestamp()>=END or read(B/'initial_plan.json')['budget_end_unix']!=END:raise ValueError('deadline')
terminal(D/'monitor_cpu001',D/'cpu_check.py');terminal(OLD/'monitor_inventory001',OLD/'inventory.py');cpu=read(D/'cpu_proof001.json');inv=read(D/'inventory001.json')
if cpu.get('status')!='passed' or inv.get('status')!='passed' or inv.get('images')!=32 or sha(D/'inventory001.json')!=sha(OLD/'inventory001.json') or any(x.get('real_tensors_deserialized') is not False or x.get('GT_opened') is not False for x in (cpu,inv)) or any(sha(k)!=v for k,v in cpu['source_sha256'].items()):raise ValueError('previous evidence')
files=[B/'initial_plan.json',D/'PLAN.md',D/'core.py',D/'test_core.py',D/'cpu_check.py',D/'cpu_proof001.json',D/'inventory001.json',D/'run.py',D/'test_run_entry.py',D/'freeze.py',OLD/'inventory.py',OLD/'inventory001.json',E/'full_001_verification.json',I/'component_full001_review.json',I/'review_component_full001.py',B/'e4_c_microbench/schedule001.json']
for mon in (D/'monitor_cpu001',OLD/'monitor_inventory001',E/'monitor_full001',E/'monitor_verify_full001',I/'monitor_component_full001'):files.extend([mon/'started.json',mon/'terminal.json'])
value={'status':'frozen','source_sha256':{str(p):sha(p) for p in files},'archives_sha256':{x['path']:x['sha256'] for x in inv['archives']},'questions':32,'numpy_version':importlib.metadata.version('numpy'),'GT_opened':False,'real_tensors_deserialized':False,'GPU_used':False,'full_runtime_pending':True}
with (D/'plan001.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
