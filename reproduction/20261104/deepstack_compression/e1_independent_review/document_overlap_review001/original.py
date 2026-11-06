"""Post-audit metadata diagnostic, never questions, answers, or predictions.

Predeclared to quantify leakage risks beyond page-level graph grouping. It does
not alter the frozen group split or imply that shared documents are duplicates.
"""
import collections,hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def rows(path):
 with path.open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def document_stats(eval_docs,train_rows):
 docs=collections.defaultdict(set);per_split=collections.Counter();overlap=collections.Counter();n=0
 for r in train_rows:
  doc=r['document_id'];split=r['split'];docs[doc].add(split);per_split[split]+=1;n+=1
  if doc in eval_docs:overlap[split]+=1
 return {'train_rows':n,'train_documents':len(docs),'evaluation_docvqa_documents':len(eval_docs),'train_documents_shared_with_eval':sum(doc in eval_docs for doc in docs),'train_rows_from_eval_documents_by_split':dict(overlap),'rows_by_split':dict(per_split),'fit_validation_shared_documents':sum('fit' in s and 'validation' in s for s in docs.values()),'fit_validation_or_eval_connected_documents':sum(len(s)>1 for s in docs.values())}
def main():
 aud=HERE/'train_audit_001';result=json.loads((aud/'result.json').read_text());terminal=json.loads((HERE/'monitor_train_audit001/terminal.json').read_text());started=json.loads((HERE/'monitor_train_audit001/started.json').read_text())
 if result.get('status')!='provisional_completed' or terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[] or terminal.get('worker_pid')!=started.get('worker_pid') or sha(aud/'rows.jsonl')!=result['outputs']['rows.jsonl']:raise ValueError('full provisional audit has not completed intact')
 import re
 eval_docs=set()
 for r in rows(BASE/'data_manifest/full_manifest.jsonl'):
  if r['task']=='DocVQA_VAL':
   m=re.fullmatch(r'(.+)_p(\d+)',Path(r['source_image_path']).stem)
   if not m:raise ValueError('unparseable document')
   eval_docs.add(m.group(1))
 counts=document_stats(eval_docs,rows(aud/'rows.jsonl'))
 if counts['train_rows']!=result['counts']['rows']:raise ValueError('row accounting')
 out={'scope':'metadata diagnostic of distinct DocVQA document identifiers; no questions, answers or predictions read; no grouping changes','counts':counts,'inputs':{'rows_sha256':sha(aud/'rows.jsonl'),'audit_result_sha256':sha(aud/'result.json'),'eval_manifest_sha256':sha(BASE/'data_manifest/full_manifest.jsonl'),'source_sha256':sha(Path(__file__))},'limitations':['Pages from the same document are not automatically duplicates, but document-level dependence may affect generalization.','Evaluation document IDs are available for DocVQA only; image-level graph still covers all three tasks.','Requires independent verification before any use in a training decision.']}
 with (HERE/'document_overlap_001.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(counts))
if __name__=='__main__':main()
