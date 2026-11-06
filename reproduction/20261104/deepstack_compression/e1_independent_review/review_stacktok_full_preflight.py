"""Independent source and synthetic review; authorizes no-GPU preflight only."""
import argparse,json,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e6/stacktok_full';sys.path.insert(0,str(D))
from admission import sources,prior
from real_worker import sha
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==D/'independent_preflight_review.json'
def read(x):return json.loads(Path(x).read_text())
m=D/'monitor_cpu003';s=read(m/'started.json');t=read(m/'terminal.json');cmd=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in cmd if Path(x).name=='cpu_validation.py']==[D/'cpu_validation.py'] and cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==D/'cpu_check003'
cpu=D/'cpu_check003/verification.json';c=read(cpu);source={str(x.resolve()):sha(x) for x in sources()};assert c['status']=='passed' and c['source_sha256_before']==c['source_sha256_after']==source and c['test_returncodes']==[0,0] and c['GPU_used'] is False and c['model_loaded'] is False and c['GT_opened'] is False and c['predictions_opened'] is False
assert 'Ran 7 tests' in (cpu.parent/'test0.stderr').read_text() and 'Ran 13 tests' in (cpu.parent/'test1.stderr').read_text();identity=prior();assert c['prior_identity']==identity
plan=read(D/'plan.json');budget=read(D/'budget_plan.json');assert plan['questions']==384 and plan['budget']==256 and budget['inner_timeout_seconds']==7200 and budget['outer_timeout_seconds']==7320 and budget['stale_seconds']==1200
r={'status':'passed','independent_review':True,'preflight_only':True,'reviewer_sha256':sha(__file__),'source_sha256':source,'cpu_sha256':sha(cpu),'cpu_started_sha256':sha(m/'started.json'),'cpu_terminal_sha256':sha(m/'terminal.json'),'GT_opened':False,'prior_identity':identity,'predictions_opened':False,'scope':'source review permitting only the no-GPU worker preflight','limitations':['preflight will parse accepted E1/E4 DEV mixed prediction journals for processor/visual identity','full model and 384-question behavior remain unverified; same-container no-GPU preflight and independent final review required before any GPU','local Qwen3 adaptation around upstream selector, not the upstream Qwen3 implementation','non-anchor tensor records are summaries and cannot be reconstructed offline'],'errors':[]};a.out.write_text(json.dumps(r,indent=2)+'\n');print('passed',len(source))
