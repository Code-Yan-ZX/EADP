"""Validate eval filename-to-document/page mapping without extracting answers."""
import collections,hashlib,json,os,re
from pathlib import Path
import pyarrow.csv as csv
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 summary=json.loads((BASE/'data_manifest/manifest_summary.json').read_text());src=summary['sources']['DocVQA_VAL'];p=Path(src['path'])
 if sha(p)!=src['sha256']:raise ValueError('evaluation dataset changed')
 columns=['ucsf_document_id','ucsf_document_page_no','image_path']
 conv=csv.ConvertOptions(include_columns=columns,column_types={k:'string' for k in columns})
 table=csv.read_csv(p,read_options=csv.ReadOptions(block_size=16*1024*1024,use_threads=False),parse_options=csv.ParseOptions(delimiter='\t'),convert_options=conv)
 if table.column_names!=columns:raise ValueError('unexpected projection')
 counts=collections.Counter()
 for doc,page,path in zip(*(table[k].to_pylist() for k in columns)):
  match=re.fullmatch(r'(.+)_p(\d+)',Path(path or '').stem)
  if not match or not page or not str(page).isdigit():counts['unparseable']+=1;continue
  counts['rows']+=1;counts['same_doc']+=int(match.group(1)==doc);counts['page_delta_'+str(int(match.group(2))-int(page))]+=1
 if counts!={'rows':5349,'same_doc':5349,'page_delta_-1':5349}:raise ValueError('eval document/page filename mapping is not uniformly zero-based')
 result={'status':'passed','columns':columns,'counts':dict(counts),'eval_source_sha256':src['sha256'],'manifest_summary_sha256':sha(BASE/'data_manifest/manifest_summary.json'),'source_sha256':sha(__file__),'scope':'projected non-answer eval metadata only; filename page index equals original page minus one','answer_or_question_values_extracted':False,'predictions_opened':False,'training_performed':False}
 with (HERE/'eval_path_semantics_001.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed','rows':counts['rows'],'page_delta':-1}))
if __name__=='__main__':main()
