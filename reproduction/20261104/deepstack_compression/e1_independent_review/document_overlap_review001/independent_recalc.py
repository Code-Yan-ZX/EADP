"""Independent exact document-set calculations; synthetic fixtures until admitted."""
from collections import Counter
import argparse,hashlib,json,os,re
from pathlib import Path

SPLITS=('fit','validation','excluded_eval')

def recalculate(eval_ids, source_rows):
 if any(not isinstance(x,str) or not x or x!=x.strip() for x in eval_ids):raise ValueError('bad eval id')
 source=list(source_rows);sets={k:set() for k in SPLITS};counts=Counter();same=Counter();qid_seen=set()
 for row in source:
  doc,split=row.get('document_id'),row.get('split');qid=row.get('question_id')
  if not isinstance(doc,str) or not doc or doc!=doc.strip() or split not in SPLITS:raise ValueError('bad source metadata')
  if type(qid) is not int or qid in qid_seen:raise ValueError('duplicate or malformed qid')
  qid_seen.add(qid);sets[split].add(doc);counts[split]+=1
  if doc in eval_ids:same[split]+=1
 combined=set().union(*sets.values());shared=sets['fit'] & sets['validation']
 fv=Counter(r['split'] for r in source if r['document_id'] in shared and r['split'] in ('fit','validation'))
 return {'source_rows_all':len(source),'source_rows_by_split':{k:counts[k] for k in SPLITS},'eligible_rows_fit_plus_validation':counts['fit']+counts['validation'],'source_document_ids':len(combined),'evaluation_docvqa_document_ids':len(eval_ids),'source_documents_also_in_eval':len(combined & eval_ids),'source_rows_from_eval_documents_by_split':{k:same[k] for k in SPLITS},'source_documents_also_in_eval_by_split':{k:len(sets[k]&eval_ids) for k in SPLITS},'fit_validation_shared_documents':len(shared),'rows_from_fit_validation_shared_documents_by_split':{k:fv[k] for k in ('fit','validation')}}

ROOT=Path(__file__).resolve().parents[2];D=ROOT/'m3_train_data';R=ROOT/'e1_independent_review'
def load(p):return json.loads(Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def supervised(directory,script,args):
 s=load(directory/'started.json');t=load(directory/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if not (t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'] and cwd.resolve()==ROOT and len(cmd)==2+len(args) and (cwd/cmd[1]).resolve()==script.resolve() and cmd[2:]==args):raise ValueError('supervisor evidence')
 return s,t
def jsonlines(path):
 with path.open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if out!=(R/'document_overlap002_review.json').resolve():raise ValueError('unexpected output')
 # Verify all three terminal chains before opening the completed source rows.
 cfg=load(D/'train_audit_execution_001.json');supervised(D/'monitor_train_audit001',D/'train_audit.py',[cfg['run_id']])
 audit=D/'train_audit_001/result.json';review=R/'train_audit001_review.json'
 supervised(R/'monitor_train_audit001',R/'review_train_audit_full.py',['--out','e1_independent_review/train_audit001_review.json'])
 pre=R/'train_audit_prelaunch_review.json';supervised(R/'monitor_train_audit_prelaunch',R/'review_train_audit_prelaunch.py',['--out','e1_independent_review/train_audit_prelaunch_review.json']);pp=load(pre)
 if pp['status']!='passed' or pp['independent_review'] is not True or pp['reviewer_sha256']!=digest(R/'review_train_audit_prelaunch.py') or len(pp['source_sha256'])!=6 or any(digest(p)!=v for p,v in pp['source_sha256'].items()):raise ValueError('prelaunch review changed')
 supervised(D/'monitor_document_overlap002',D/'document_overlap_analysis_v2.py',[])
 ar=load(audit);rp=load(review);protocol=load(D/'train_audit_protocol_001.json');summary=load(ROOT/'data_manifest/manifest_summary.json');diag=load(D/'document_overlap_002.json')
 if not (ar['status']=='provisional_completed' and ar['counts']['rows']==39463 and ar['answer_or_question_values_extracted'] is False and ar['predictions_opened'] is False and ar['training_performed'] is False and ar['inputs']['execution_config_sha256']==digest(D/'train_audit_execution_001.json') and ar['inputs']['protocol_sha256']==digest(D/'train_audit_protocol_001.json') and cfg['protocol_sha256']==digest(D/'train_audit_protocol_001.json')):raise ValueError('audit proof')
 if not (rp['status']=='passed' and rp['independent_review'] is True and rp['reviewer_sha256']==digest(R/'review_train_audit_full.py') and rp['formal_result_sha256']==digest(audit) and rp['run_started_sha256']==digest(D/'monitor_train_audit001/started.json') and rp['run_terminal_sha256']==digest(D/'monitor_train_audit001/terminal.json') and rp['rows_verified']==39463 and rp['outputs_sha256']==ar['outputs'] and rp['labels_extracted'] is False and rp['predictions_opened'] is False and rp['training_performed'] is False):raise ValueError('independent audit proof')
 man=ROOT/'data_manifest/full_manifest.jsonl';sem=D/'eval_path_semantics_001.json';supervised(D/'monitor_eval_paths001',D/'validate_eval_paths.py',[]);semantic=load(sem)
 if not (digest(man)==protocol['inputs_sha256'][str(man)]==summary['outputs_sha256']['full_manifest.jsonl'] and digest(ROOT/'data_manifest/manifest_summary.json')==protocol['inputs_sha256'][str(ROOT/'data_manifest/manifest_summary.json')] and digest(sem)==protocol['inputs_sha256'][str(sem)] and semantic['source_sha256']==digest(D/'validate_eval_paths.py') and semantic['status']=='passed' and semantic['counts']=={'rows':5349,'same_doc':5349,'page_delta_-1':5349}):raise ValueError('eval provenance')
 rowsfile=D/'train_audit_001/rows.jsonl'
 if digest(rowsfile)!=ar['outputs']['rows.jsonl']:raise ValueError('audit rows changed')
 sources={'rows_sha256':digest(rowsfile),'audit_result_sha256':digest(audit),'independent_review_sha256':digest(review),'eval_manifest_sha256':digest(man),'protocol_sha256':digest(D/'train_audit_protocol_001.json'),'source_sha256':digest(D/'document_overlap_analysis_v2.py')}
 if diag['status']!='diagnostic_completed' or any(diag['inputs'].get(k)!=v for k,v in sources.items()):raise ValueError('diagnostic provenance')
 evaldocs=set();n=0
 for row in jsonlines(man):
  if row['task']=='DocVQA_VAL':
   n+=1;match=re.fullmatch(r'(.+)_p(\d+)',Path(row['source_image_path']).stem)
   if not match:raise ValueError('eval path')
   evaldocs.add(match.group(1))
 if n!=5349:raise ValueError('eval rows')
 result=recalculate(evaldocs,jsonlines(rowsfile));expected={k:ar['counts']['rows_by_split'].get(k,0) for k in SPLITS}
 if result['source_rows_all']!=39463 or result['source_rows_by_split']!=expected or result!=diag['counts']:raise ValueError('independent counts differ')
 proof={'status':'passed','independent_review':True,'counts':result,'inputs':sources,'diagnostic_sha256':digest(D/'document_overlap_002.json'),'diagnostic_started_sha256':digest(D/'monitor_document_overlap002/started.json'),'diagnostic_terminal_sha256':digest(D/'monitor_document_overlap002/terminal.json'),'audit_review_started_sha256':digest(R/'monitor_train_audit001/started.json'),'audit_review_terminal_sha256':digest(R/'monitor_train_audit001/terminal.json'),'reviewer_sha256':digest(__file__),'GT_opened':False,'predictions_opened':False,'training_performed':False,'limitations':['Exact document-ID equality only; not evidence that pages are duplicates.','Evaluation document metadata is limited to DocVQA.','This diagnostic does not authorize training.']}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
