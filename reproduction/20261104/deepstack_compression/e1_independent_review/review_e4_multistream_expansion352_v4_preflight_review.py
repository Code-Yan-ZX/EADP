"""Independent v4 schema/source prelaunch: authorize only no-GPU preflight."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');D=B/'e4_multistream_expansion352_gpu_v4';P=B/'e4_multistream_expansion352';I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);cmd=s['command'];assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()]
 if out is not None:assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();assert not (D/'preflight001').exists() and not (D/'run001').exists();terminal(D/'monitor_cpu_check002',D/'cpu_check.py');terminal(D/'monitor_freeze_schema001',D/'freeze_schema_fixture.py');c=read(D/'cpu_check002.json');src=c['source_sha256_after'];assert c['status']=='passed' and c['source_sha256_before']==src and len(src)==136 and all(sha(k)==v for k,v in src.items()) and c['GT_opened'] is False and c['predictions_opened'] is True and c['feature_values_opened'] is False and c['score_values_opened'] is False and c['GPU_used'] is False and c['model_loaded'] is False and c['errors']==[]
 old=read(P/'cpu_check001.json');assert old['status']=='passed' and all(src.get(k)==v for k,v in old['source_sha256_after'].items())
 plan=read(D/'execution_plan001.json');budget=read(B/'initial_plan.json');assert plan['images']==9 and plan['GT_opened'] is False and plan['scoring_performed'] is False and plan['automatic_retry'] is False and plan['budget_source_sha256']==sha(B/'initial_plan.json') and plan['budget_end_unix']==budget['budget_end_unix'] and time.time()+plan['preflight_outer_seconds']<budget['budget_end_unix'] and plan['preflight_inner_seconds']+plan['grace_seconds']<=plan['preflight_outer_seconds']
 # Independently rebind the identity-only fixture to already accepted E1/E4 records.
 fixture=read(D/'schema_fixture001.json');selection=read(P/'selection001.json')['pilot'];assert fixture['status']=='frozen' and fixture['selected']==selection and len(selection)==9 and fixture['GT_opened'] is False and fixture['scoring_performed'] is False and fixture['errors']==[] and fixture['selection_sha256']==sha(P/'selection001.json') and fixture['freeze_script_sha256']==sha(D/'freeze_schema_fixture.py')
 e1p=B/'e1/e1full001_completion.json';e4p=B/'e4_panel_v2/panel_003_verification.json';e1=read(e1p);e4=read(e4p);journal=Path(e1['files']['journal']['path']);panel=B/'e4_panel_v2/panel_003/records.jsonl';assert fixture['e1_proof_sha256']==sha(e1p) and fixture['e4_proof_sha256']==sha(e4p) and fixture['e1_journal_sha256']==e1['files']['journal']['sha256']==sha(journal) and fixture['e4_journal_sha256']==e4['files_sha256'][str(panel)]==sha(panel)
 want={(x['task'],str(x['qid'])) for x in selection};actual1={};actual4={}
 with journal.open() as f:
  for line in f:
   r=json.loads(line);key=(r.get('task'),str(r.get('qid')))
   if key in want and r.get('arm')=='P3-D1' and r.get('forced') is False:assert key not in actual1;actual1[key]=r
 with panel.open() as f:
  for line in f:
   r=json.loads(line);key=(r.get('task'),str(r.get('qid')))
   if key in want and r.get('budget')==256 and r.get('ds_on') is True:assert key not in actual4;actual4[key]=r
 assert set(actual1)==set(actual4)==want and len(fixture['e1_fields'])==len(fixture['e4_fields'])==9
 for group,actual in ((fixture['e1_fields'],actual1),(fixture['e4_fields'],actual4)):
  assert {(r['task'],str(r['qid'])) for r in group}==want
  for r in group:
   assert 'answer' not in r and 'output_ids' not in r;srcrow=actual[(r['task'],str(r['qid']))];assert all(srcrow.get(k)==v for k,v in r.items())
 tests=0
 for directory,pattern,n in [(P,'test_norm_contract.py',4),(P,'test_pilot_engine.py',3),(P,'test_pilot_contract.py',3),(P,'test_selection.py',5),(D,'test_*.py',13)]:
  r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(directory),'-p',pattern,'-v'],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120);assert r.returncode==0 and f'Ran {n} tests' in r.stderr,(pattern,r.stderr);tests+=n
 assert tests==28 and all(sha(k)==v for k,v in src.items())
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');r=subprocess.run([sys.executable,str(D/'launch.py'),'--run-id','001','--preflight-only'],cwd=B,env=env,capture_output=True,text=True,timeout=120);assert r.returncode!=0 and 'independent review missing' in r.stderr and not (D/'preflight001').exists()
 r=subprocess.run([sys.executable,str(D/'worker.py'),'--out',str(D/'run001')],cwd=B,env=env,capture_output=True,text=True,timeout=120);assert r.returncode!=0 and 'independent review missing' in r.stderr and not (D/'run001').exists()
 out={'status':'passed','independent_review':True,'preflight_only':True,'GT_opened':False,'predictions_opened':True,'feature_values_opened':False,'score_values_opened':False,'GPU_used':False,'model_loaded':False,'errors':[],'source_sha256':src,'reviewer_sha256':sha(__file__),'cpu_sha256':sha(D/'cpu_check002.json'),'cpu_started_sha256':sha(D/'monitor_cpu_check002/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu_check002/terminal.json'),'selection_sha256':sha(P/'selection001.json'),'schema_fixture_sha256':sha(D/'schema_fixture001.json'),'independent_cpu_tests':tests,'scope':'source/schema review authorizes only v4 no-GPU preflight','limitations':['Actual nine author processor calls and four-shard hashes still require the real no-GPU container and independent review.','The host v4 formal verifier is a separate future frozen/independent gate and must independently compare saved actual identity fields.','Existing accepted E1/E4 mixed records parsed only for identity metadata; no GT or scores.']}
 with open(a.out,'x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
