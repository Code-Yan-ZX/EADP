"""Source and synthetic review: authorizes only a no-GPU worker preflight."""
import argparse,json,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e6/stacktok_smoke';sys.path.insert(0,str(D))
from admission import sources,prior
from real_worker import sha
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert a.out.resolve()==D/'independent_preflight_review.json'
def read(x):return json.loads(Path(x).read_text())
m=D/'monitor_cpu008';s=read(m/'started.json');t=read(m/'terminal.json');cmd=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'];assert [(cwd/x).resolve() for x in cmd if Path(x).name=='cpu_validation.py']==[D/'cpu_validation.py'];assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==D/'cpu_check008'
cpu=D/'cpu_check008/verification.json';c=read(cpu);src={str(x.resolve()):sha(x) for x in sources()};assert c['status']=='passed' and c['source_sha256_before']==c['source_sha256_after']==src and c['test_returncodes']==[0,0] and c['GPU_used'] is False and c['model_loaded'] is False and c['GT_opened'] is False and c['predictions_opened'] is False
assert 'Ran 10 tests' in (cpu.parent/'test0.stderr').read_text() and 'Ran 13 tests' in (cpu.parent/'test1.stderr').read_text();identity=prior();assert c['prior_identity']==identity
plan=read(D/'plan001.json');budget=read(D/'budget_plan.json');assert plan['GT_opened'] is False and plan['gpu_authorized'] is False and len(plan['anchors'])==3 and plan['natural_generations']==plan['forced_generations']==9 and plan['forced_steps']==32 and plan['budgets']==[1024,256] and budget['inner_timeout_seconds']==5400 and budget['outer_timeout_seconds']==5520
assert (B/'e4_native/native_capture.py').read_text().replace('from native_contract import E4\n','')==(D/'stock_capture.py').read_text()
assert "if path=='local_stacktok'" in (D/'worker.py').read_text() and 'compare_keepall(*pair)' in (D/'worker.py').read_text()
value={'status':'passed','independent_review':True,'preflight_only':True,'reviewer_sha256':sha(__file__),'source_sha256':src,'cpu_sha256':sha(cpu),'cpu_started_sha256':sha(m/'started.json'),'cpu_terminal_sha256':sha(m/'terminal.json'),'GT_opened':False,'prior_identity':identity,'predictions_opened':False,'scope':'CPU source review permitting only a same-image no-GPU worker preflight','limitations':['the actual worker preflight will open accepted E1 DEV mixed prediction journals solely for input identity','real Qwen3 behavior, selector, prefill/three DS, 36-layer cache, forced/natural generation and identity are not established by synthetic tests','local Qwen3 integration, not upstream author Qwen3 code; GPU requires final independent review and root scheduling'],'errors':[]};a.out.write_text(json.dumps(value,indent=2)+'\n');print('passed; sources',len(src))
