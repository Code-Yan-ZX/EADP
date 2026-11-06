"""Frozen document-ID risk diagnostic after full independent audit, no labels.

Does not alter page-level graph assignment or establish that pages are duplicates.
"""
import collections,hashlib,json,os,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;REVIEW=BASE/'e1_independent_review'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(path):return json.loads(path.read_text())
def rows(path):
 with path.open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def document_stats(eval_docs,train_rows):
 docs=collections.defaultdict(set);row_counts=collections.Counter();same_eval_rows=collections.Counter();same_eval_docs=collections.defaultdict(set);all_rows=[]
 for row in train_rows:
  doc=row.get('document_id');split=row.get('split')
  if not isinstance(doc,str) or not doc.strip() or split not in ('fit','validation','excluded_eval'):raise ValueError('invalid document/split')
  doc=doc.strip();docs[doc].add(split);row_counts[split]+=1;all_rows.append((doc,split))
  if doc in eval_docs:same_eval_rows[split]+=1;same_eval_docs[split].add(doc)
 fitval={d for d,s in docs.items() if 'fit' in s and 'validation' in s}
 fitval_rows=collections.Counter(split for doc,split in all_rows if doc in fitval and split in ('fit','validation'))
 return {'source_rows_all':len(all_rows),'source_rows_by_split':{k:row_counts[k] for k in ('fit','validation','excluded_eval')},'eligible_rows_fit_plus_validation':row_counts['fit']+row_counts['validation'],'source_document_ids':len(docs),'evaluation_docvqa_document_ids':len(eval_docs),'source_documents_also_in_eval':sum(d in eval_docs for d in docs),'source_rows_from_eval_documents_by_split':{k:same_eval_rows[k] for k in ('fit','validation','excluded_eval')},'source_documents_also_in_eval_by_split':{k:len(same_eval_docs[k]) for k in ('fit','validation','excluded_eval')},'fit_validation_shared_documents':len(fitval),'rows_from_fit_validation_shared_documents_by_split':{k:fitval_rows[k] for k in ('fit','validation')}}
def completed(start,end):
 return end.get('reason')=='completed' and end.get('worker_returncode')==0 and end.get('remaining_process_group')==[] and end.get('worker_pid')==start.get('worker_pid')==start.get('process_group')
def main():
 aud=HERE/'train_audit_001';mon=HERE/'monitor_train_audit001';started=load(mon/'started.json');terminal=load(mon/'terminal.json');cfg=load(HERE/'train_audit_execution_001.json')
 if not completed(started,terminal) or started.get('cwd')!=str(BASE) or started.get('command')!=[cfg['python_executable'],cfg['worker'],cfg['run_id']]:raise ValueError('audit incomplete or wrong command')
 rm=REVIEW/'monitor_train_audit001';rs=load(rm/'started.json');rt=load(rm/'terminal.json');reviewer=REVIEW/'review_train_audit_full.py'
 if not completed(rs,rt) or rs.get('cwd')!=str(BASE) or rs.get('command')!=['/home/dell/miniconda3/bin/python','e1_independent_review/review_train_audit_full.py','--out','e1_independent_review/train_audit001_review.json']:raise ValueError('independent review incomplete or wrong command')
 pm=REVIEW/'monitor_train_audit_prelaunch';ps=load(pm/'started.json');pt=load(pm/'terminal.json');pre=load(REVIEW/'train_audit_prelaunch_review.json');pr=REVIEW/'review_train_audit_prelaunch.py'
 if not completed(ps,pt) or ps.get('cwd')!=str(BASE) or ps.get('command')!=['/home/dell/miniconda3/bin/python','e1_independent_review/review_train_audit_prelaunch.py','--out','e1_independent_review/train_audit_prelaunch_review.json'] or pre.get('status')!='passed' or pre.get('independent_review') is not True or pre.get('reviewer_sha256')!=sha(pr):raise ValueError('audit prelaunch independent proof')
 for path,digest in pre.get('source_sha256',{}).items():
  if sha(Path(path))!=digest:raise ValueError('prelaunch source changed')
 if len(pre.get('source_sha256',{}))!=6:raise ValueError('incomplete prelaunch source proof')
 result=load(aud/'result.json');protocol=load(HERE/'train_audit_protocol_001.json');proof=load(REVIEW/'train_audit001_review.json')
 if result.get('status')!='provisional_completed' or result.get('answer_or_question_values_extracted') is not False or result.get('counts',{}).get('rows')!=39463:raise ValueError('audit result identity')
 if result['inputs']['protocol_sha256']!=sha(HERE/'train_audit_protocol_001.json') or result['inputs']['execution_config_sha256']!=sha(HERE/'train_audit_execution_001.json'):raise ValueError('protocol or execution config binding')
 if proof.get('status')!='passed' or proof.get('independent_review') is not True or proof.get('reviewer_sha256')!=sha(reviewer) or proof.get('formal_result_sha256')!=sha(aud/'result.json') or proof.get('run_started_sha256')!=sha(mon/'started.json') or proof.get('run_terminal_sha256')!=sha(mon/'terminal.json') or proof.get('rows_verified')!=39463 or proof.get('outputs_sha256')!=result['outputs'] or proof.get('labels_extracted') is not False or proof.get('predictions_opened') is not False or proof.get('training_performed') is not False:raise ValueError('independent proof binding')
 if sha(aud/'rows.jsonl')!=result['outputs']['rows.jsonl']:raise ValueError('source rows changed')
 man=BASE/'data_manifest/full_manifest.jsonl';summary=load(BASE/'data_manifest/manifest_summary.json');sem=HERE/'eval_path_semantics_001.json';v=load(sem);st=load(HERE/'monitor_eval_paths001/started.json');en=load(HERE/'monitor_eval_paths001/terminal.json')
 if sha(man)!=protocol['inputs_sha256'][str(man)] or sha(man)!=summary['outputs_sha256']['full_manifest.jsonl'] or sha(sem)!=protocol['inputs_sha256'][str(sem)] or v.get('status')!='passed' or v.get('counts')!={'rows':5349,'same_doc':5349,'page_delta_-1':5349} or v.get('source_sha256')!=sha(HERE/'validate_eval_paths.py') or st.get('cwd')!=str(BASE) or st.get('command')!=['/home/dell/miniconda3/bin/python','m3_train_data/validate_eval_paths.py'] or not completed(st,en):raise ValueError('eval manifest or mapping proof')
 docs=set();n=0
 for row in rows(man):
  if row['task']=='DocVQA_VAL':
   n+=1;m=re.fullmatch(r'(.+)_p(\d+)',Path(row['source_image_path']).stem)
   if not m:raise ValueError('unparseable eval page')
   docs.add(m[1])
 if n!=5349:raise ValueError('eval row denominator')
 counts=document_stats(docs,rows(aud/'rows.jsonl'))
 if counts['source_rows_all']!=39463 or any(counts['source_rows_by_split'][k]!=result['counts']['rows_by_split'].get(k,0) for k in ('fit','validation','excluded_eval')):raise ValueError('source row denominator or split counts')
 out={'status':'diagnostic_completed','scope':'distinct DocVQA document IDs only; source rows include excluded records; no questions, answers or predictions extracted; frozen image groups unchanged','counts':counts,'inputs':{'rows_sha256':sha(aud/'rows.jsonl'),'audit_result_sha256':sha(aud/'result.json'),'independent_review_sha256':sha(REVIEW/'train_audit001_review.json'),'eval_manifest_sha256':sha(man),'protocol_sha256':sha(HERE/'train_audit_protocol_001.json'),'source_sha256':sha(Path(__file__))},'limitations':['Same-document pages are not automatically duplicate images, but dependence can affect generalization.','Evaluation document identifiers here are available for DocVQA only; the image graph covers all three tasks.','Third-party source provenance remains unverified.','This diagnostic requires independent review before any training decision.']}
 with (HERE/'document_overlap_002.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(counts))
if __name__=='__main__':main()
