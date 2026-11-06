"""Independent review of 384-input no-device preflight; no new inference or labels."""
import argparse,collections,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/rres_k256_dev384';ROOT=B.parents[2]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def terminal(d,script):
 s=read(d/'started.json');t=read(d/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 assert type(s['worker_pid']) is int and s['worker_pid']>0 and s['worker_pid']==t['worker_pid'] and t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()],str(d)
 return s

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists()
 plan=read(D/'preflight_plan001.json');cpu=read(D/'cpu_proof001.json');terminal(D/'monitor_cpu_freeze001',D/'cpu_check.py');terminal(D/'monitor_preflight_freeze001',D/'freeze_preflight.py');terminal(D/'monitor_design001',D/'freeze_design.py')
 assert plan['scope']=='rres384_noGPU_processor_preflight' and plan['images']==384 and plan['resolution']==512 and plan['visual_tokens']==256 and plan['GPU_authorized'] is False and plan['GT_opened'] is False and plan['existing_DEV_predictions_opened'] is True and plan['scoring_performed'] is False and plan['errors']==[]
 source=plan['source_sha256'];assert cpu['status']=='passed' and cpu['GPU_used'] is False and cpu['GT_opened'] is False and cpu['source_sha256_before']==cpu['source_sha256_after'] and all(source.get(k)==v for k,v in cpu['source_sha256_after'].items()) and not any(x.startswith('/data/models/huggingface/') for x in source)
 assert all(sha(k)==v for k,v in source.items())
 design=read(D/'design001.json');parent=B/'e6/pixelprune_t020_dev384/plan001.json';prior=read(parent);assert design['parent_plan_sha256']==sha(parent) and design['design_source_sha256']==sha(D/'design.py') and design['freeze_source_sha256']==sha(D/'freeze_design.py') and plan['design_sha256']==sha(D/'design001.json') and design['status']=='passed' and design['GT_opened'] is False and design['predictions_opened'] is False
 fields=('task','qid','group_id','message_sha256','image_sha256','processor','image_start');expected=[{**{k:r[k] for k in fields},'qid':str(r['qid'])} for r in prior['anchors']];actual=design['anchors'];keys=[(r['task'],str(r['qid'])) for r in actual]
 assert actual==expected and len(keys)==len(set(keys))==384 and len({x['group_id'] for x in actual})==384 and collections.Counter(x[0] for x in keys)=={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128}
 sys.path.insert(0,str(D));import pilot_gate
 assert Path(pilot_gate.__file__).resolve()==D/'pilot_gate.py';prior=pilot_gate.require();assert prior['review_sha256']==plan['pilot_review_sha256']
 run=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-p','test_*.py'],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=180)
 assert run.returncode==0 and 'Ran 17 tests' in run.stderr,(run.stdout,run.stderr)
 gatepath=D/'preflight_gate.py';assert sha(gatepath) in (D/'preflight_launch.py').read_text() and sha(gatepath) in (D/'preflight_worker.py').read_text()
 # Direct real host entry must fail closed while this reviewer has not exited.
 out=D/'run_preflight001';assert not out.exists();entry=subprocess.run([sys.executable,str(D/'preflight_launch.py'),'--out',str(out)],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
 assert entry.returncode!=0 and not out.exists() and ('terminal.json' in entry.stderr or 'supervision' in entry.stderr),entry.stderr
 loaderpath=B/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py';spec=importlib.util.spec_from_file_location('_rres384_review_loader',loaderpath);loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader);runtime=loader.load_author(source);author=read(ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json');cmd=runtime.container_command(author,[str(D/'preflight_worker.py'),'--out',str(out/'worker_preflight.json')],False,'synthetic-review-only')
 assert cmd.count('--rm')==1 and '--gpus' not in cmd and '--device' not in cmd and 'CUDA_VISIBLE_DEVICES=' in cmd and '--network' in cmd and cmd[cmd.index('--network')+1]=='none' and author['container_image'] in cmd
 for name in ('preflight_launch.py','preflight_worker.py','probe_processor.py','probe_core.py','processor_contract.py','panel_core.py','identity_wait.py','pilot_gate.py','preflight_gate.py'):
  assert str((D/name).resolve()) in source,name
 assert all(sha(k)==v for k,v in source.items())
 result={'status':'passed','independent_review':True,'preflight_only':True,'GPU_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'source_sha256':source,'plan_sha256':sha(D/'preflight_plan001.json'),'cpu_sha256':sha(D/'cpu_proof001.json'),'cpu_started_sha256':sha(D/'monitor_cpu_freeze001/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu_freeze001/terminal.json'),'pilot_review_sha256':prior['review_sha256'],'reviewer_sha256':sha(__file__),'questions':384,'synthetic_tests':17,'GPU_used':False,'model_loaded':False,'scope':'only no-device processor/weight preflight; not GPU or model authorization','limitations':['accepted E1 mixed DEV journal and existing DEV predictions may be hashed/parsed for identity','384 actual processor, weights and real container not yet run','384 GPU worker and formal acceptance not yet frozen'],'errors':[]}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('384 noGPU preflight independent review passed')
if __name__=='__main__':main()
