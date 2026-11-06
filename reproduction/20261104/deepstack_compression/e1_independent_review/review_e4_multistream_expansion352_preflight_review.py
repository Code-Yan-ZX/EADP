"""Independent CPU review for nine-image no-GPU preflight, not GPU authority."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e4_multistream_expansion352'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for a in iter(lambda:f.read(1<<20),b''):h.update(a)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(p,script):
 s=read(p/'started.json');t=read(p/'terminal.json');cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in s['command'] if Path(x).name==script.name]==[script.resolve()]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=I/'e4_multistream_expansion352_preflight_review.json'
 if a.out.resolve()!=out or out.exists() or (D/'preflight001').exists():raise ValueError('unexpected output')
 terminal(D/'monitor_cpu_check001',D/'cpu_check.py');c=read(D/'cpu_check001.json');s=read(D/'selection001.json');q=read(I/'multistream_expansion352_review.json')
 if c.get('status')!='passed' or c.get('GT_opened') is not False or c.get('GPU_used') is not False or c.get('feature_values_opened') is not False or c.get('score_values_opened') is not False or c['source_sha256_before']!=c['source_sha256_after'] or any(sha(k)!=v for k,v in c['source_sha256_after'].items()):raise ValueError('source or CPU evidence')
 if q.get('status')!='passed' or len(s.get('pilot',[]))!=9:raise ValueError('selection evidence')
 required=[D/x for x in ('preflight_launch.py','pilot_worker.py','pilot_engine.py','admission.py','pilot_contract.py','norm_contract.py','selection001.json','PLAN.md')]+[B/'e4_c_microbench_pilot/capture_guard.py',B/'e4_c_microbench_pilot/identity_contract.py',B/'e5/selector_real32/panel_reference.py']
 if any(str(x.resolve()) not in c['source_sha256_after'] for x in required):raise ValueError('missing execution source')
 worker=(D/'pilot_worker.py').read_text();host=(D/'preflight_launch.py').read_text();engine=(D/'pilot_engine.py').read_text();gate=(D/'admission.py').read_text()
 for fragment in ('adm.admit(a.preflight_only)','torch.cuda.is_initialized()','len(selected)!=9','model_loaded\':False','scoring_performed\':False'):
  if fragment not in worker:raise ValueError('worker requirement '+fragment)
 for fragment in ('gate.admit(True)',"'--preflight-only'", "'--gpus'", "'--device'",'run_bounded'):
  if fragment not in host:raise ValueError('preflight requirement '+fragment)
 for fragment in ('validate_e1(actual,ref,tensor_meta)','identity_check(target,meta,e4)','image_not_evaluated','preflight_seconds','capture_seconds','serialization_seconds'):
  if fragment not in engine:raise ValueError('engine requirement '+fragment)
 if 'GPU runtime admission not implemented' not in gate:raise ValueError('GPU not locked')
 z=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-p','test_*.py','-v'],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),text=True,capture_output=True,timeout=180)
 if z.returncode or 'Ran 17 tests' not in z.stderr:raise ValueError('tests failed '+z.stderr[-2000:])
 z2=subprocess.run([sys.executable,str(D/'preflight_launch.py'),'--run-id','001'],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True,timeout=30)
 if z2.returncode==0 or 'independent review missing' not in z2.stderr or (D/'preflight001').exists():raise ValueError('real host entry not closed before proof')
 value={'status':'passed','independent_review':True,'preflight_only':True,'GT_opened':False,'feature_values_opened':False,'score_values_opened':False,'GPU_used':False,'model_loaded':False,'errors':[],'source_sha256':c['source_sha256_after'],'reviewer_sha256':sha(__file__),'cpu_sha256':sha(D/'cpu_check001.json'),'cpu_started_sha256':sha(D/'monitor_cpu_check001/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu_check001/terminal.json'),'selection_sha256':sha(D/'selection001.json'),'test_log':z.stderr,'real_entry_before_review':'correctly refused before Docker','limitations':['CPU review authorizes only real no-GPU preflight; GPU is explicitly locked','actual preflight hashes weights and parses already accepted mixed DEV records for identity','no model output, feature results or performance conclusions']}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
