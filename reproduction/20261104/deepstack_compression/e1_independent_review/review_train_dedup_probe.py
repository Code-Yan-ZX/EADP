"""Recompute the first training shard image-only dedup probe independently."""
import argparse,collections,hashlib,io,json,os,re
from pathlib import Path
import numpy as np,pyarrow.csv as csv,pyarrow.parquet as pq,scipy
from scipy.fft import dctn
from PIL import Image,ImageOps,__version__ as pillow
ROOT=Path(__file__).resolve().parent.parent;D=ROOT/'m3_train_data';M=ROOT/'data_manifest'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert a.out.resolve()==ROOT/'e1_independent_review/train_dedup_probe002_review.json'
 proof=read(D/'dedup_probe_002.json');sem=read(D/'eval_path_semantics_001.json');summary=read(M/'manifest_summary.json');rec=read(D/'probe_001/receipt.json')
 for mon,script in (('monitor_dedup_probe003','probe_dedup_v2.py'),('monitor_eval_paths001','validate_eval_paths.py')):
  s=read(D/mon/'started.json');t=read(D/mon/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and any(Path(x).name==script for x in s['command'])
 assert proof['source_sha256']==sha(D/'probe_dedup_v2.py') and proof['digest_implementation_sha256']==sha(M/'build.py') and proof['eval_page_semantics_sha256']==sha(D/'eval_path_semantics_001.json') and sem['source_sha256']==sha(D/'validate_eval_paths.py')
 assert proof['train_shard_sha256']==sha(D/'probe_001/shard.parquet')==rec['actual_sha256'] and proof['train_receipt_sha256']==sha(D/'probe_001/receipt.json')
 for n,k in [('images.jsonl','eval_image_manifest_sha256'),('full_manifest.jsonl','eval_full_manifest_sha256')]:assert sha(M/n)==proof[k]==summary['outputs_sha256'][n]
 assert pillow==proof['pillow']==summary['pillow'] and scipy.__version__==proof['scipy']==summary['scipy']
 cols=['ucsf_document_id','ucsf_document_page_no','image_path'];src=summary['sources']['DocVQA_VAL'];assert sha(src['path'])==src['sha256']==sem['eval_source_sha256']
 table=csv.read_csv(src['path'],read_options=csv.ReadOptions(block_size=16*1024*1024,use_threads=False),parse_options=csv.ParseOptions(delimiter='\t'),convert_options=csv.ConvertOptions(include_columns=cols,column_types={k:'string' for k in cols}))
 assert table.column_names==cols and len(table)==5349
 for doc,page,path in zip(*(table[k].to_pylist() for k in cols)):
  m=re.fullmatch(r'(.+)_p(\d+)',Path(path or '').stem);assert m and str(page).isdigit() and m[1]==doc and int(m[2])+1==int(page)
 images=[json.loads(x) for x in (M/'images.jsonl').open()];full=[json.loads(x) for x in (M/'full_manifest.jsonl').open()];assert len(images)==5652
 assert set(x['image_index'] for x in full)==set(range(len(images)))
 byte={x['byte_sha256'] for x in images};pixels={x['rgb_sha256'] for x in images};hashes=[int(x['phash'],16) for x in images]
 docrows=[x for x in full if x['task']=='DocVQA_VAL'];docs=set()
 for row in docrows:
  m=re.fullmatch(r'(.+)_p(\d+)',Path(row.get('source_image_path') or '').stem);assert m;docs.add((m[1],str(int(m[2])+1)))
 assert len(docrows)==5349
 counts=collections.Counter();seen={};ids=set();file=pq.ParquetFile(D/'probe_001/shard.parquet')
 for batch in file.iter_batches(batch_size=24,columns=['image','question_id','data_split','other_metadata']):
  for r in batch.to_pylist():
   assert r['data_split']=='train' and isinstance(r['question_id'],int) and r['question_id'] not in ids;ids.add(r['question_id']);data=r['image']['bytes'];h=hashlib.sha256(data).hexdigest();doc=str(r['other_metadata']['ucsf_document_id']);page=str(r['other_metadata']['ucsf_document_page_no']);assert page.isdigit();key=(doc,str(int(page)))
   if h not in seen:
    with Image.open(io.BytesIO(data)) as im:
     rgb=ImageOps.exif_transpose(im).convert('RGB');w,hgt=rgb.size;ph=hashlib.sha256(f'{w}x{hgt}|RGB|'.encode()+rgb.tobytes()).hexdigest();grey=np.asarray(rgb.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
    f=dctn(grey,type=2,norm='ortho')[:8,:8].flatten();m=np.median(f[1:]);bits=int(''.join('1' if x>m else '0' for x in f),2)
    seen[h]=(key,{'byte_match':h in byte,'rgb_match':ph in pixels,'phash_le4':any((bits^x).bit_count()<=4 for x in hashes),'semantic_match':key in docs})
   assert seen[h][0]==key;counts['rows']+=1
   for k,v in seen[h][1].items():counts['rows_'+k]+=int(v)
 for _,v in seen.values():
  for k,flag in v.items():counts['unique_'+k]+=int(flag)
 counts['unique_images']=len(seen);assert dict(counts)==proof['counts'] and len(ids)==1201 and len(seen)==321
 out={'status':'passed','independent_review':True,'source_sha256':sha(__file__),'probe_sha256':sha(D/'dedup_probe_002.json'),'rows':len(ids),'unique_images':len(seen),'eval_images':len(images),'eval_doc_rows':len(docrows),'counts':dict(counts),'labels_extracted':False,'predictions_opened':False,'scope':'Full underlying TSV and parquet contain answer fields; read projected metadata/images only and did not extract or parse answer values. One third-party shard is insufficient for an eligible training set.'}
 with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed','rows':len(ids),'unique_images':len(seen)}))
if __name__=='__main__':main()
