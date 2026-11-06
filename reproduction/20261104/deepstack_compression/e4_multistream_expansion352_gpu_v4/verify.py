"""Formal no-GT, post-completion nine-image verification."""
import argparse,hashlib,importlib.util,json,math,os,subprocess,sys
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;B=D.parent;P=B/'e4_multistream_expansion352'
ADMISSION_SHA='6ab5a522548a0b6bcf6fe7a67c2e3971c83e736a92a37ee0b8ab42b412fd8b5f'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path,digest):
 if sha(path)!=digest or name in sys.modules:raise ValueError('unapproved module')
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m
 try:s.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve()
 if out!=D/'verification001.json' or out.exists():raise ValueError('unexpected proof')
 gate=load('_multistream352v2_verify_admission',D/'admission.py',ADMISSION_SHA);approval=gate.admit(False,host=True);gate.terminal(D/'monitor_run001',D/'launch.py');gate.terminal(D/'monitor_snapshot001',D/'snapshot.py',D/'run001_sources')
 run=D/'run001';la=gate.read(run/'launch.json');ex=gate.read(run/'exit.json');done=gate.read(run/'worker_complete.json');snap=gate.read(D/'run001_sources/manifest.json');selection=gate.read(P/'selection001.json')['pilot']
 if (la.get('approval_bundle')!=approval or la.get('gpu_requested') is not True or ex.get('container_stopped_verified') is not True or ex.get('container_exit_code')!=0 or ex.get('gpu_pids_after')!=[] or ex.get('failure_type') is not None or done.get('status')!='completed' or done.get('images')!=9 or done.get('source_sha256')!=approval['source_sha256'] or done.get('weight_stats_before')!=done.get('weight_stats_after') or snap.get('status')!='passed' or snap.get('gpu_review_sha256')!=approval['review_sha256'] or snap.get('run_terminal_sha256')!=sha(D/'monitor_run001/terminal.json') or {k:v['sha256'] for k,v in snap['sources'].items()}!=approval['source_sha256'] or any(sha(v['snapshot'])!=v['sha256'] for v in snap['sources'].values())):raise ValueError('completed run/source snapshot')
 obj=json.loads(subprocess.run(['docker','inspect',la['name']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];state=obj['State']
 if obj['Id']!=ex['container_identity']['container_id'] or state['Running'] is not False or state['Pid']!=0 or state['ExitCode']!=0 or state['OOMKilled'] is not False:raise ValueError('actual GPU container exit')
 norm=load('_multistream352v2_verify_norm',P/'norm_contract.py',approval['source_sha256'][str((P/'norm_contract.py').resolve())]);features=norm.original();contract=load('_multistream352v2_verify_contract',P/'pilot_contract.py',approval['source_sha256'][str((P/'pilot_contract.py').resolve())]);journal=[json.loads(x) for x in (run/'journal.jsonl').read_text().splitlines() if x.strip()];finished=[e for e in journal if e.get('status')=='image_completed'];generated=[e for e in journal if e.get('status')=='capture_generated'];cleanup=[e for e in journal if e.get('status')=='capture_cleanup']
 if len(finished)!=9 or len(generated)!=9 or len(cleanup)!=9 or any(e.get('passed') is not True or e.get('capture_completed') is not True for e in cleanup) or any(e.get('status') in ('image_failed','image_not_evaluated') for e in journal):raise ValueError('journal coverage/cleanup')
 # Re-open only the already accepted mixed E4 records and independently match
 # the runtime-recorded metadata against the same task/question reference.
 for name in ('identity_gate','latency_admission','panel_reference'):
  if name in sys.modules:raise ValueError('preloaded reference dependency')
 sys.path[:0]=[str(B/'e4_c_latency'),str(B/'e5/selector_real32')]
 refpath=B/'e5/selector_real32/panel_reference.py';panel=load('_multistream352v2_verify_panel_reference',refpath,approval['source_sha256'][str(refpath.resolve())]);worker=load('_multistream352v2_verify_worker',D/'worker.py',approval['source_sha256'][str((D/'worker.py').resolve())]);e4=worker.references(selection,panel.accepted())
 results=[];files={}
 for i,target in enumerate(selection):
  record=run/'captures'/('image_%03d.json'%i);array=run/'captures'/('image_%03d_norms.npy'%i);v=gate.read(record);n=np.load(array,allow_pickle=False)
  if v.get('status')!='passed' or v.get('index')!=i or any(str(v.get(k))!=str(target[k]) for k in ('task','qid','group_id','image_sha256','message_sha256')) or v.get('norms_sha256')!=sha(array) or v.get('norms_path')!=str(array) or v.get('hooks_clean') is not True or v.get('e1_identity',{}).get('status')!='passed' or v.get('features')!=norm.features_from_norms(n,features) or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in v['timing_seconds'].values()):raise ValueError('image identity/norm/features')
  contract.check_identity(target,v['e4_meta'],e4[(target['task'],str(target['qid']))])
  if finished[i].get('index')!=i or str(finished[i].get('qid'))!=str(target['qid']) or str(generated[i].get('qid'))!=str(target['qid']) or generated[i].get('task')!=target['task']:raise ValueError('journal identity')
  files[str(record)]=sha(record);files[str(array)]=sha(array);results.append({'task':target['task'],'qid':str(target['qid']),'status':'passed'})
 status=gate.read(run/'captures/status.json')
 if not contract.check_complete(selection,status['rows']) or status.get('images_passed')!=9 or status.get('status')!='passed':raise ValueError('complete denominator')
 files.update({str(p):sha(p) for p in (run/'journal.jsonl',run/'worker_complete.json',run/'launch.json',run/'exit.json',run/'captures/status.json')})
 proof={'status':'passed','images':9,'rows':results,'files_sha256':files,'source_manifest_sha256':sha(D/'run001_sources/manifest.json'),'formal_plan_sha256':sha(D/'formal_plan001.json'),'run_started_sha256':sha(D/'monitor_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'verifier_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'limits':['All eight features are recomputed from retained norm vectors; full visual tensors were not retained, and model tensor association remains runtime identity/fingerprint evidence.','Nine cases are exploratory, post hoc cost/identity diagnostics, not TTFT.','Accepted E1/E4 DEV mixed journals were parsed for identity and nine one-token diagnostic generations are recorded.'],'errors':[]}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
