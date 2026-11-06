"""Reconstruct completed TRAIN image audit independently; no answers or predictions."""
import argparse,collections,hashlib,io,json,os,re
from pathlib import Path
import numpy as np,pyarrow.parquet as pq
from PIL import Image,ImageOps
from scipy.fft import dctn
ROOT=Path(__file__).resolve().parent.parent;D=ROOT/'m3_train_data';M=ROOT/'data_manifest';RUN=D/'train_audit_001';MON=D/'monitor_train_audit001';PROGRESS=ROOT/'e1_independent_review/train_audit_reconstruction001'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def log(**v):
 with (PROGRESS/'progress.jsonl').open('a') as f:f.write(json.dumps(v)+'\n');f.flush();os.fsync(f.fileno())
def describe(raw):
 with Image.open(io.BytesIO(raw)) as im:
  rgb=ImageOps.exif_transpose(im).convert('RGB');w,h=rgb.size;pix=hashlib.sha256(f'{w}x{h}|RGB|'.encode()+rgb.tobytes()).hexdigest();small=np.asarray(rgb.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
 v=dctn(small,type=2,norm='ortho')[:8,:8].flatten();med=np.median(v[1:]);bits=int(''.join('1' if x>med else '0' for x in v),2);return (w,h,pix,bits)
class DSU:
 def __init__(self,n):self.p=list(range(n))
 def find(self,x):
  while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
  return x
 def union(self,x,y):
  a,b=self.find(x),self.find(y)
  if a!=b:self.p[b]=a

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert a.out.resolve()==ROOT/'e1_independent_review/train_audit001_review.json'
 # Nothing from the unfinished audit is read before these checks.
 start=read(MON/'started.json');end=read(MON/'terminal.json');cfg=read(D/'train_audit_execution_001.json');pre=read(ROOT/'e1_independent_review/train_audit_prelaunch_review.json');assert pre['status']=='passed' and pre['independent_review'] is True and all(sha(p)==d for p,d in pre['source_sha256'].items())
 assert end['reason']=='completed' and end['worker_returncode']==0 and end['remaining_process_group']==[] and start['worker_pid']==start['process_group']==end['worker_pid'] and start['command']==[cfg['python_executable'],cfg['worker'],cfg['run_id']] and start['timeout_seconds']==cfg['timeout_seconds'] and start['stale_seconds']==cfg['stale_seconds'] and start['snapshot_seconds']==cfg['snapshot_seconds']
 result=read(RUN/'result.json');assert result['status']=='provisional_completed' and result['answer_or_question_values_extracted'] is False and result['predictions_opened'] is False and result['training_performed'] is False and result['counts']['rows']==39463
 for name,digest in result['outputs'].items():assert sha(RUN/name)==digest
 assert result['inputs']['protocol_sha256']==sha(D/'train_audit_protocol_001.json') and result['inputs']['receipt_sha256']==sha(D/'train_fetch_001/receipt.json') and result['inputs']['worker_sha256']==sha(D/'train_audit.py') and result['inputs']['core_sha256']==sha(D/'train_audit_core.py') and result['inputs']['execution_config_sha256']==sha(D/'train_audit_execution_001.json')
 PROGRESS.mkdir(exist_ok=False);log(event='begun')
 images=lines(RUN/'images.jsonl');rows=lines(RUN/'rows.jsonl');evals=lines(M/'images.jsonl');full=lines(M/'full_manifest.jsonl');receipt=read(D/'train_fetch_001/receipt.json');inventory=read(D/'source_inventory_001.json');assert len(rows)==39463 and len(evals)==5652 and len(receipt['shards'])==len(inventory['train_shards'])==36
 assert all(x['image_index']==i for i,x in enumerate(images));byraw={x['byte_sha256']:i for i,x in enumerate(images)};assert len(byraw)==len(images)
 for i,e in enumerate(evals):
  o=images[i];assert (e['byte_sha256'],e['rgb_sha256'],e['phash'],e['width'],e['height'])==(o['byte_sha256'],o['rgb_sha256'],o['phash'],o['width'],o['height']) and o['is_eval_image'] is True
 seenraw=set();qids=set();at=0
 for shard_idx,(item,shard) in enumerate(zip(inventory['train_shards'],receipt['shards'])):
  p=Path(shard['path']);assert p.stat().st_size==item['bytes']==shard['bytes'] and sha(p)==item['lfs_sha256']==shard['sha256']
  k=0
  for batch in pq.ParquetFile(p).iter_batches(batch_size=24,columns=['image','question_id','data_split','other_metadata']):
   for r in batch.to_pylist():
    rec=rows[at];qid=r['question_id'];raw=r['image']['bytes'];h=hashlib.sha256(raw).hexdigest();meta=r['other_metadata'];doc=str(meta['ucsf_document_id']).strip();page=str(int(str(meta['ucsf_document_page_no']).strip()))
    assert r['data_split']=='train' and qid not in qids and (rec['question_id'],rec['shard_index'],rec['row_index'],rec['image_byte_sha256'],rec['document_id'],rec['page_one_based'])==(qid,shard_idx,k,h,doc,page);qids.add(qid)
    if h not in seenraw:
     w,ht,rgb,bits=describe(raw);im=images[byraw[h]];assert (w,ht,rgb,f'{bits:016x}')==(im['width'],im['height'],im['rgb_sha256'],im['phash']);seenraw.add(h)
    at+=1;k+=1
  assert k==pq.ParquetFile(p).metadata.num_rows;log(event='shard_verified',index=shard_idx,rows=at,unique_images=len(seenraw))
 assert at==len(rows)==len(qids)
 dsu=DSU(len(images));pixels={};buckets=collections.defaultdict(list);near=0
 for i,im in enumerate(images):
  if im['rgb_sha256'] in pixels:dsu.union(i,pixels[im['rgb_sha256']])
  else:pixels[im['rgb_sha256']]=i
  bits=int(im['phash'],16);keys=[(j,(bits>>s)&((1<<(12 if j==4 else 13))-1)) for j,s in enumerate((0,13,26,39,52))];candidates=set()
  for k in keys:candidates.update(buckets[k])
  for j in candidates:
   if (bits^int(images[j]['phash'],16)).bit_count()<=4:dsu.union(i,j);near+=1
  for k in keys:buckets[k].append(i)
 existing={}
 for i,e in enumerate(evals):
  gid=e['group_id']
  if gid in existing:dsu.union(i,existing[gid])
  else:existing[gid]=i
 semantic={}
 def addsem(key,idx):
  if key in semantic:dsu.union(idx,semantic[key])
  else:semantic[key]=idx
 for r in full:
  if r['task']=='DocVQA_VAL':
   m=re.fullmatch(r'(.+)_p(\d+)',Path(r['source_image_path']).stem);assert m;addsem((m[1],str(int(m[2])+1)),r['image_index'])
 for r in rows:addsem((r['document_id'],r['page_one_based']),byraw[r['image_byte_sha256']])
 comps=collections.defaultdict(list)
 for i in range(len(images)):comps[dsu.find(i)].append(i)
 group={};eligible=[];excluded=0
 for members in comps.values():
  gid=hashlib.sha256(('group|'+'|'.join(sorted(images[i]['byte_sha256'] for i in members))).encode()).hexdigest();is_eval=any(images[i]['is_eval_image'] for i in members)
  if is_eval:excluded+=1
  else:eligible.append(gid)
  for i in members:group[i]=(gid,is_eval)
 order=sorted(eligible,key=lambda g:(hashlib.sha256(('20261105|TRAIN_FIT_VAL|'+g).encode()).hexdigest(),g));cut=len(order)*4//5;splits={g:('fit' if i<cut else 'validation') for i,g in enumerate(order)};tally=collections.Counter();gcounts=collections.Counter()
 for i,im in enumerate(images):
  gid,ev=group[i];split='excluded_eval' if ev else splits[gid];assert im['group_id']==gid and im['split']==split
 for row in rows:
  gid,ev=group[byraw[row['image_byte_sha256']]];split='excluded_eval' if ev else splits[gid];assert row['group_id']==gid and row['split']==split;tally[split]+=1;gcounts[gid]+=1
 image_counts=collections.Counter(gid for gid,_ in group.values());expected={g:{'group_id':g,'split':splits[g],'images':image_counts[g],'questions':gcounts[g]} for g in eligible}
 reported=lines(RUN/'groups.jsonl');assert len(reported)==len(expected) and all(row==expected[row['group_id']] for row in reported)
 counts=result['counts'];assert counts['near_edges']==near and counts['eval_connected_components']==excluded and counts['eligible_groups']==len(eligible) and counts['rows_by_split']==dict(tally)
 out={'status':'passed','independent_review':True,'reviewer_sha256':sha(__file__),'formal_result_sha256':sha(RUN/'result.json'),'run_started_sha256':sha(MON/'started.json'),'run_terminal_sha256':sha(MON/'terminal.json'),'rows_verified':len(rows),'train_images_recomputed':len(seenraw),'all_images':len(images),'groups_verified':len(comps),'outputs_sha256':result['outputs'],'labels_extracted':False,'predictions_opened':False,'training_performed':False,'limits':['Source Parquet contains answer/question columns; only projected image/metadata values were parsed.','Different pages of a document are not automatically grouped.','This is provisional third-party source eligibility, not permission to train.']}
 with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed','rows':len(rows),'groups':len(comps)}))
if __name__=='__main__':main()
