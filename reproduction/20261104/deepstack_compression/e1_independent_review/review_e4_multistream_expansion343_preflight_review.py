"""Independent preflight-only review for 343-image residual extraction."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e4_multistream_expansion343_gpu';P=B/'e4_multistream_expansion352'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for a in iter(lambda:f.read(1<<20),b''):h.update(a)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(p,script):
 s=read(p/'started.json');t=read(p/'terminal.json');cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in s['command'] if Path(x).name==script.name]==[script.resolve()]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=I/'e4_multistream_expansion343_preflight_review.json'
 if a.out.resolve()!=out or out.exists() or (D/'preflight001').exists() or (D/'run001').exists():raise ValueError('unexpected output')
 terminal(D/'monitor_cpu_check002',D/'cpu_check.py');c=read(D/'cpu_check002.json');p=read(D/'execution_plan001.json');s=read(P/'selection001.json');past=read(I/'e4_multistream_expansion352_pilot_v4_result_review.json')
 if c.get('status')!='passed' or c.get('GT_opened') is not False or c.get('GPU_used') is not False or c.get('feature_values_opened') is not False or c.get('score_values_opened') is not False or c['source_sha256_before']!=c['source_sha256_after'] or any(sha(k)!=v for k,v in c['source_sha256_after'].items()):raise ValueError('source or CPU evidence')
 if p.get('images')!=343 or p.get('GT_opened') is not False or p.get('automatic_retry') is not False or past.get('status')!='passed' or past.get('images')!=9:raise ValueError('plan or accepted pilot')
 required=[D/x for x in ('launch.py','worker.py','pilot_engine.py','admission.py','cohort.py','full_reference.py','processor_preflight.py','execution_plan001.json','formal_plan001.json')]+[P/'selection001.json',P/'norm_contract.py']
 if any(str(x.resolve()) not in c['source_sha256_after'] for x in required):raise ValueError('missing source')
 sys.path.insert(0,str(D));import cohort,admission
 r=cohort.build(s)
 if len(r)!=343 or {t:sum(z['task']==t for z in r) for t in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')}!={'DocVQA_VAL':114,'TextVQA_VAL':114,'AI2D_TEST':115}:raise ValueError('cohort')
 admission.prior_pilot()
 w=(D/'worker.py').read_text();h=(D/'launch.py').read_text();e=(D/'pilot_engine.py').read_text()
 for frag in ('len(selected)!=343','processor_inputs_checked','e1_p3d1_complete_references','torch.cuda.is_initialized()','engine.run(selected'):
  if frag not in w:raise ValueError('worker '+frag)
 for frag in ('gate.admit(a.preflight_only,host=True)','run_bounded','gpu_pids()','--preflight-only'):
  if frag not in h:raise ValueError('launcher '+frag)
 for frag in ('validate_e1(actual,ref,tensor_meta)','identity_check(target,meta,e4)','image_not_evaluated','timing_seconds'):
  if frag not in e:raise ValueError('engine '+frag)
 logs=[]
 for directory,pattern in ((P,'test_norm_contract.py'),(P,'test_pilot_contract.py'),(P,'test_selection.py'),(D,'test_*.py')):
  z=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(directory),'-p',pattern,'-v'],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),text=True,capture_output=True,timeout=180)
  if z.returncode or 'OK' not in z.stderr:raise ValueError('tests '+z.stderr[-1500:])
  logs.append(z.stderr)
 z=subprocess.run([sys.executable,str(D/'launch.py'),'--run-id','001','--preflight-only'],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True,timeout=60)
 if z.returncode==0 or 'independent review missing' not in z.stderr or (D/'preflight001').exists():raise ValueError('real entry failed to deny before review')
 value={'status':'passed','independent_review':True,'preflight_only':True,'GT_opened':False,'feature_values_opened':False,'score_values_opened':False,'GPU_used':False,'errors':[],'source_sha256':c['source_sha256_after'],'reviewer_sha256':sha(__file__),'cpu_sha256':sha(D/'cpu_check002.json'),'cpu_started_sha256':sha(D/'monitor_cpu_check002/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu_check002/terminal.json'),'selection_sha256':sha(P/'selection001.json'),'tests':logs,'cohort_counts':{'DocVQA_VAL':114,'TextVQA_VAL':114,'AI2D_TEST':115},'limitations':['authorizes actual no-GPU preflight only; GPU admission and independent review remain','full 343 processor/identity checks and four-shard hashes must occur in real preflight','source identity includes hashes of previously accepted mixed DEV records; no new features or scores opened']}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
