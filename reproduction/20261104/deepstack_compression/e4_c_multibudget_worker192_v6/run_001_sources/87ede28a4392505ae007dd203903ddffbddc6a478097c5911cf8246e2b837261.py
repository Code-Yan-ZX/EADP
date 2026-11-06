"""Existing E4 DEV P3/DS-on natural endpoints, never GT or new predictions."""
import hashlib,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;PILOT=BASE/'e6/pixel_natural_timing_pilot_v4'

def load(name,path):
 path=Path(path).resolve();existing=sys.modules.get(name)
 if existing is not None:
  if not getattr(existing,'__file__',None) or Path(existing.__file__).resolve()!=path:raise ValueError('foreign preloaded module '+name)
  return existing
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
 try:spec.loader.exec_module(mod)
 except BaseException:del sys.modules[name];raise
 return mod

if hashlib.sha256((PILOT/'bootstrap.py').read_bytes()).hexdigest()!='ab81e2f48f995403afc21ab6ab4875c8b4ae4e9bcca72be063ab8fc42a53fdbc':raise ValueError('bootstrap source changed')
spec=importlib.util.spec_from_file_location('_budget_reference_bootstrap',PILOT/'bootstrap.py');boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.pin_all()

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(start,end):
 return end.get('reason')=='completed' and end.get('worker_returncode')==0 and end.get('remaining_process_group')==[] and type(start.get('worker_pid')) is int and start['worker_pid']==end.get('worker_pid')

def load_references():
 p=BASE/'e4_panel_v2';r=BASE/'e1_independent_review';formal=p/'panel_003_verification.json';review=r/'e4panel003_review.json'
 fs=p/'monitor_verify_panel003/started.json';ft=p/'monitor_verify_panel003/terminal.json';rs=r/'monitor_e4panel003/started.json';rt=r/'monitor_e4panel003/terminal.json';reviewer=r/'review_e4panel.py'
 f=read(formal);q=read(review)
 if (f.get('status')!='passed' or f.get('n_questions')!=384 or f.get('all_arms_validated')!=3072 or f.get('GT_opened') is not False or f.get('scoring_performed') is not False or
     q.get('status')!='passed' or q.get('n_verified')!=384 or q.get('arms_verified')!=3072 or q.get('errors')!=[] or q.get('formal_proof_sha256')!=sha(formal) or q.get('formal_verifier_started_sha256')!=sha(fs) or q.get('formal_verifier_terminal_sha256')!=sha(ft) or q.get('reviewer_sha256')!=sha(reviewer) or
     not terminal(read(fs),read(ft)) or not terminal(read(rs),read(rt))):raise ValueError('E4 independent acceptance gate')
 path=(p/'panel_003/records.jsonl').resolve()
 if f['files_sha256'].get(str(path))!=sha(path):raise ValueError('E4 records identity')
 if sha(HERE/'proposal.py')!='4006b7b9ee63c00ac060bdf3fcd629b4d41a1e76ccc5ecb014431d93e94f2df4':raise ValueError('proposal source changed before import')
 proposal=load('_budget_192_proposal',HERE/'proposal.py');anchors=proposal.load();needed={(a['task'],str(a['qid'])):a for a in anchors}
 if len(needed)!=222:raise ValueError('invalid frozen input set')
 e1=BASE/'e1';input_module=boot.pin('input_preflight',BASE/'e4_c_latency/input_preflight.py');rows,refs=input_module.load_accepted_e1_references(e1/'e1full001_completion.json',e1/'monitorverifye1full001/started.json',e1/'monitorverifye1full001/terminal.json');bykey={(x['task'],str(x['qid'])):x for x in rows}
 chosen=[];records={}
 for key,anchor in needed.items():
  row=bykey[key];ref=refs[key]
  if anchor['processor_reference']!=ref['processor'] or anchor['image_start_reference']!=ref['image_start']:raise ValueError('proposal processor identity')
  if row.get('split')!='DEV':raise ValueError('non-DEV input')
  if any(row.get(k)!=anchor.get(k) for k in ('task','group_id','message_sha256','image_sha256')) or sha(row['image_path'])!=anchor['image_sha256']:raise ValueError('E4 input identity')
  chosen.append({**row,'image_start':ref['image_start'],'processor_expected':ref['processor']})
 with path.open() as fobj:
  for line in fobj:
   rec=json.loads(line);key=(rec['task'],str(rec['qid']))
   if key not in needed or rec.get('ds_on') is not True or rec.get('budget') not in (128,256,512,1024) or rec.get('forced') is not False:continue
   anchor=needed[key];identity=(*key,rec['budget'])
   if identity in records or any(rec.get(k)!=anchor.get(k) for k in ('group_id','message_sha256','image_sha256')) or rec.get('processor')!=refs[key]['processor'] or rec.get('image_start')!=refs[key]['image_start']:raise ValueError('E4 endpoint binding')
   ids=rec.get('output_ids');eos=rec.get('eos_ids');k=rec['budget'];mode='keep_all' if k==1024 else 'author_greedy'
   if not isinstance(ids,list) or not ids or not eos or not (ids[-1] in eos or len(ids)==2048) or rec.get('actual_injection_verified') is not True or rec.get('selection_mode')!=mode or rec.get('greedy_calls')!=(0 if k==1024 else 1) or len(rec.get('ordered_ids',[]))!=k:raise ValueError('E4 endpoint incomplete')
   records[identity]=rec
 if len(records)!=888:raise ValueError('missing 222x4 accepted E4 natural endpoints')
 return {'rows':chosen,'records':records,'formal_sha256':sha(formal),'review_sha256':sha(review),'records_sha256':sha(path),'GT_opened':False,'predictions_opened':True}
