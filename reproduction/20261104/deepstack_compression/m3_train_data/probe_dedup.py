"""Image-only feasibility against all evaluation images; no answers read.

This does not produce an eligible training manifest: full TRAIN availability,
transitive grouping across all TRAIN images, and source provenance remain open.
"""
import collections,hashlib,importlib.util,json,os,re,sys,time
from pathlib import Path
import pyarrow.parquet as pq
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;MAN=BASE/'data_manifest'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 receipt=read(HERE/'probe_001/receipt.json');file=HERE/'probe_001/shard.parquet'
 if receipt.get('status')!='passed' or sha(file)!=receipt['actual_sha256'] or receipt.get('GT_opened') is not False:raise ValueError('unverified train shard')
 summary=read(MAN/'manifest_summary.json')
 for name in ('images.jsonl','full_manifest.jsonl'):
  if sha(MAN/name)!=summary['outputs_sha256'][name]:raise ValueError('evaluation manifest changed')
 spec=importlib.util.spec_from_file_location('_eadp_train_dedup_build',MAN/'build.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 if Path(mod.__file__).resolve()!=(MAN/'build.py').resolve() or mod.pillow_version!=summary['pillow'] or mod.scipy.__version__!=summary['scipy']:raise ValueError('image digest implementation or version differs')
 eval_images=[json.loads(x) for x in (MAN/'images.jsonl').open() if x.strip()]
 raw_set={x['byte_sha256'] for x in eval_images};rgb_set={x['rgb_sha256'] for x in eval_images};phashes=[int(x['phash'],16) for x in eval_images]
 eval_docs=set();n_doc_path=0;unparsed=0
 for x in (json.loads(l) for l in (MAN/'full_manifest.jsonl').open() if l.strip()):
  if x['task']!='DocVQA_VAL':continue
  n_doc_path+=1;stem=Path(x.get('source_image_path') or '').stem;match=re.fullmatch(r'(.+)_p(\d+)',stem)
  if match:eval_docs.add((match.group(1),str(int(match.group(2)))))
  else:unparsed+=1
 if len(eval_images)!=5652 or n_doc_path==0:raise ValueError('evaluation image inventory')
 parquet=pq.ParquetFile(file);columns=['image','question_id','data_split','other_metadata'];qids=set();cache={};counts=collections.Counter();image_docs={};rows=0
 for batch in parquet.iter_batches(batch_size=32,columns=columns):
  for record in batch.to_pylist():
   rows+=1;qid=record['question_id'];meta=record['other_metadata'];im=record['image'];split=record['data_split']
   if split!='train' or not isinstance(qid,int) or qid in qids or not isinstance(meta,dict) or not isinstance(im,dict) or not isinstance(im.get('bytes'),bytes):raise ValueError('unexpected train metadata')
   qids.add(qid);raw=im['bytes'];rawhash=hashlib.sha256(raw).hexdigest();doc=str(meta.get('ucsf_document_id') or '');page=str(meta.get('ucsf_document_page_no') or '')
   if not doc or not page.isdigit():raise ValueError('missing train document/page')
   semantic=(doc,str(int(page)))
   if rawhash in image_docs and image_docs[rawhash]!=semantic:raise ValueError('same bytes with differing train semantic id')
   image_docs[rawhash]=semantic
   if rawhash not in cache:
    _,rgb,bits=mod.describe(raw);cache[rawhash]={'byte_match':rawhash in raw_set,'rgb_match':rgb in rgb_set,'phash_le4':any((bits-x).bit_count()<=4 for x in [])}
    # XOR Hamming, not arithmetic distance; all evaluation images included.
    cache[rawhash]['phash_le4']=any((bits^x).bit_count()<=4 for x in phashes)
    cache[rawhash]['semantic_match']=semantic in eval_docs
   status=cache[rawhash]
   counts['rows']+=1
   for k,value in status.items():counts['rows_'+k]+=int(value)
 for value in cache.values():
  for k,flag in value.items():counts['unique_'+k]+=int(flag)
 counts['unique_images']=len(cache)
 if rows!=parquet.metadata.num_rows or rows!=1201 or counts['rows']!=rows:raise ValueError('shard denominator')
 result={'status':'completed','scope':'first pinned third-party TRAIN shard, image-only feasibility; no question text, answers or predictions parsed','counts':dict(counts),'eval_unique_byte_images':len(eval_images),'eval_doc_question_rows':n_doc_path,'eval_doc_paths_unparsed':unparsed,'train_shard_sha256':sha(file),'train_receipt_sha256':sha(HERE/'probe_001/receipt.json'),'eval_image_manifest_sha256':sha(MAN/'images.jsonl'),'eval_full_manifest_sha256':sha(MAN/'full_manifest.jsonl'),'digest_implementation_sha256':sha(MAN/'build.py'),'pillow':mod.pillow_version,'scipy':mod.scipy.__version__,'GT_opened':False,'predictions_opened':False,'training_performed':False,'limits':['Single first shard is not a representative or admissible TRAIN split.','The per-image matches do not implement cross-TRAIN transitive grouping.','The third-party repack provenance remains unverified against the original access-controlled release.','A perceptual hash can miss semantically related pages; document grouping and near-duplicate audit remain necessary.']}
 out=HERE/'dedup_probe_001.json'
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'completed','rows':rows,'unique_images':len(cache),'matches':{k:v for k,v in counts.items() if 'match' in k or 'phash' in k},'eval_doc_paths_unparsed':unparsed}))
if __name__=='__main__':main()
