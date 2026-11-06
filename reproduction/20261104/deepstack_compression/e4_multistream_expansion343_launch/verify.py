"""Formal no-GT, post-completion nine-image verification."""
import argparse,hashlib,importlib.util,json,math,os,subprocess,sys
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;B=D.parent;P=B/'e4_multistream_expansion352';V=B/'e4_multistream_expansion343_gpu'
GATE_SHA='3b076731b71aeba5ae80e4cf02a34ee9849bd4508b6cae845fb62c61a0e59d96'
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
 gate=load('_multistream352v2_verify_admission',D/'gpu_gate.py',GATE_SHA);guard=gate.admit();approval=guard['full_approval'];gate.terminal(D/'monitor_run001',D/'launch.py');gate.terminal(D/'monitor_snapshot001',D/'snapshot.py',D/'run001_sources')
 run=V/'run001';la=gate.read(run/'launch.json');ex=gate.read(run/'exit.json');done=gate.read(run/'worker_complete.json');snap=gate.read(D/'run001_sources/manifest.json');cohort=load('_multistream343_verify_cohort',V/'cohort.py',approval['source_sha256'][str((V/'cohort.py').resolve())]);selection=cohort.build(gate.read(P/'selection001.json'))
 if (la.get('approval_bundle')!=approval or la.get('full_guard')!=guard or la.get('gpu_requested') is not True or ex.get('container_stopped_verified') is not True or ex.get('container_exit_code')!=0 or ex.get('gpu_pids_after')!=[] or ex.get('failure_type') is not None or done.get('status')!='completed' or done.get('images')!=343 or done.get('source_sha256')!=approval['source_sha256'] or done.get('weight_stats_before')!=done.get('weight_stats_after') or snap.get('status')!='passed' or snap.get('gpu_review_sha256')!=approval['review_sha256'] or snap.get('run_terminal_sha256')!=sha(D/'monitor_run001/terminal.json') or {k:v['sha256'] for k,v in snap['sources'].items()}!=dict(approval['source_sha256'],**guard['guard_source_sha256']) or snap.get('guard_review_sha256')!=guard['guard_review_sha256'] or any(sha(v['snapshot'])!=v['sha256'] for v in snap['sources'].values())):raise ValueError('completed run/source snapshot')
 gate.inspect_gpu(la,ex)
 norm=load('_multistream352v2_verify_norm',P/'norm_contract.py',approval['source_sha256'][str((P/'norm_contract.py').resolve())]);features=norm.original();contract=load('_multistream352v2_verify_contract',P/'pilot_contract.py',approval['source_sha256'][str((P/'pilot_contract.py').resolve())]);journal=[json.loads(x) for x in (run/'journal.jsonl').read_text().splitlines() if x.strip()];finished=[e for e in journal if e.get('status')=='image_completed'];generated=[e for e in journal if e.get('status')=='capture_generated'];cleanup=[e for e in journal if e.get('status')=='capture_cleanup']
 if len(finished)!=343 or len(generated)!=343 or len(cleanup)!=343 or any(e.get('passed') is not True or e.get('capture_completed') is not True for e in cleanup) or any(e.get('status') in ('image_failed','image_not_evaluated') for e in journal):raise ValueError('journal coverage/cleanup')
 # Re-open only the already accepted mixed E4 records and independently match
 # the runtime-recorded metadata against the same task/question reference.
 for name in ('identity_gate','latency_admission','panel_reference'):
  if name in sys.modules:raise ValueError('preloaded reference dependency')
 sys.path[:0]=[str(B/'e4_c_latency'),str(B/'e5/selector_real32')]
 refpath=B/'e5/selector_real32/panel_reference.py';panel=load('_multistream352v2_verify_panel_reference',refpath,approval['source_sha256'][str(refpath.resolve())]);worker=load('_multistream352v2_verify_worker',V/'worker.py',approval['source_sha256'][str((V/'worker.py').resolve())]);e4=worker.references(selection,panel.accepted())
 # Independent full E1 P3-D1 versus E4 mapping, using the same accepted
 # journal only for identity metadata and validating its hash before/after.
 import input_preflight
 if Path(input_preflight.__file__).resolve()!=(B/'e4_c_latency/input_preflight.py').resolve():raise ValueError('E1 reference import path')
 e1=B/'e1';rows,slim=input_preflight.load_accepted_e1_references(e1/'e1full001_completion.json',e1/'monitorverifye1full001/started.json',e1/'monitorverifye1full001/terminal.json');proof=gate.read(e1/'e1full001_completion.json');entry=proof['files']['journal'];path=Path(entry['path'])
 if gate.sha(path)!=entry['sha256']:raise ValueError('E1 journal changed')
 fullmod=load('_multistream_v4_verify_fullref',V/'full_reference.py',approval['source_sha256'][str((V/'full_reference.py').resolve())])
 with path.open() as f:fullrefs=fullmod.collect(selection,(json.loads(x) for x in f if x.strip()),slim,e4)
 if gate.sha(path)!=entry['sha256']:raise ValueError('E1 journal changed during read')
 verification=load('_multistream_v4_verify_contract',D/'verify_contract.py',guard['guard_source_sha256'][str((D/'verify_contract.py').resolve())])
 results=[];files={}
 for i,target in enumerate(selection):
  record=run/'captures'/('image_%03d.json'%i);array=run/'captures'/('image_%03d_norms.npy'%i);v=gate.read(record);n=np.load(array,allow_pickle=False)
  if v.get('status')!='passed' or v.get('index')!=i or any(str(v.get(k))!=str(target[k]) for k in ('task','qid','group_id','image_sha256','message_sha256')) or v.get('norms_sha256')!=sha(array) or v.get('norms_path')!=str(array) or v.get('hooks_clean') is not True or v.get('e1_identity',{}).get('status')!='passed' or v.get('features')!=norm.features_from_norms(n,features) or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in v['timing_seconds'].values()):raise ValueError('image identity/norm/features')
  contract.check_identity(target,v['e4_meta'],e4[(target['task'],str(target['qid']))]);ref=fullrefs[(target['task'],str(target['qid']))];actual=v.get('actual_identity')
  verification.actual_identity(actual,ref)
  if finished[i].get('index')!=i or str(finished[i].get('qid'))!=str(target['qid']) or str(generated[i].get('qid'))!=str(target['qid']) or generated[i].get('task')!=target['task']:raise ValueError('journal identity')
  files[str(record)]=sha(record);files[str(array)]=sha(array);results.append({'task':target['task'],'qid':str(target['qid']),'status':'passed'})
 status=gate.read(run/'captures/status.json')
 if not verification.complete(selection,status['rows']) or status.get('images_passed')!=343 or status.get('status')!='passed':raise ValueError('complete denominator')
 files.update({str(p):sha(p) for p in (run/'journal.jsonl',run/'worker_complete.json',run/'launch.json',run/'exit.json',run/'captures/status.json')})
 proof={'status':'passed','images':343,'rows':results,'files_sha256':files,'source_manifest_sha256':sha(D/'run001_sources/manifest.json'),'formal_plan_sha256':sha(D/'formal_plan001.json'),'run_started_sha256':sha(D/'monitor_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'verifier_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'limits':['Eight features are recomputed from retained norm vectors; stored actual processor/visual/DS/prefill fingerprints and coordinates match accepted E1/E4. Full visual tensors were not retained and association remains runtime capture evidence.','These 343 posthoc exploratory captures are not TTFT.','Accepted E1/E4 DEV mixed journals were parsed for identity and 343 one-token diagnostic generations are recorded.'],'errors':[]}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
