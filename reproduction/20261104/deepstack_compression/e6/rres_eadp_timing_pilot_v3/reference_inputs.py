"""Load only already accepted input identities and same-arm natural outputs."""
import hashlib,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def terminal(start,end):
 s=read(start);e=read(end)
 return e.get('reason')=='completed' and e.get('worker_returncode')==0 and e.get('remaining_process_group')==[] and type(s.get('worker_pid')) is int and s['worker_pid']==e.get('worker_pid')
def load():
 review=ROOT/'e1_independent_review/rres_k256_dev384_result_review.json';q=read(review)
 formal=ROOT/'e6/rres_k256_dev384/verification001.json';fs=ROOT/'e6/rres_k256_dev384/monitor_verify001/started.json';ft=fs.parent/'terminal.json';rs=ROOT/'e1_independent_review/monitor_rres_k256_dev384_result/started.json';rt=rs.parent/'terminal.json'
 reviewer=ROOT/'e1_independent_review/review_rres_k256_dev384_result.py'
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('images')!=384 or q.get('errors')!=[] or q.get('GT_opened') is not False or q.get('scoring_performed') is not False or q.get('formal_sha256')!=sha(formal) or q.get('formal_started_sha256')!=sha(fs) or q.get('formal_terminal_sha256')!=sha(ft) or q.get('reviewer_sha256')!=sha(reviewer) or not terminal(fs,ft) or not terminal(rs,rt):raise ValueError('R-res independent acceptance')
 path=ROOT/'e4_c_multibudget_worker_v3/references.py';spec=importlib.util.spec_from_file_location('_rres_eadp_existing_references',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 e4=mod.load_references();schedule=read(HERE/'schedule001.json');cohort=schedule['measured_inputs']+schedule['warmup_inputs'];rows={(r['task'],str(r['qid'])):r for r in e4['rows']};refs={}
 if len(cohort)!=62 or len(rows)!=62:raise ValueError('cohort denominator')
 for a in cohort:
  key=(a['task'],str(a['qid']));row=rows[key]
  if any(a[k]!=row[k] for k in ('group_id','message_sha256','image_sha256')):raise ValueError('input changed')
  e=e4['records'][(*key,256)];refs[('eadp',*key)]=e
  base=ROOT/'e6/rres_k256_dev384/run001/observations'/('_'.join(key))
  rp=Path(str(base)+'.json');rawp=Path(str(base)+'_generated.json')
  for path in (rp,rawp):
   if q['files_sha256'].get(str(path.resolve()))!=sha(path):raise ValueError('R-res file identity')
  r=read(rp);raw=read(rawp)
  if any(r[k]!=a[k] for k in ('group_id','message_sha256','image_sha256')) or r['output_ids']!=raw['full_sequence'][0][raw['input_length']:] or r.get('visual_tokens')!=256:raise ValueError('R-res output identity')
  refs[('rres',*key)]=r
 if len(refs)!=124:raise ValueError('same-arm references incomplete')
 return {'rows':rows,'references':refs,'GT_opened':False,'existing_predictions_opened':True,'rres_review_sha256':sha(review),'eadp_review_sha256':e4['review_sha256'],'eadp_records_sha256':e4['records_sha256']}
