"""Full provisional label-free TRAIN/evaluation image overlap audit.

Input Parquet files contain answer and question columns; only the four listed
columns are projected and no answer or question values are extracted or parsed.
Results are provisional pending independent review and source provenance.
"""
import collections,hashlib,importlib.util,json,os,re,shutil,sys,time
import numpy,pyarrow,PIL,scipy
from pathlib import Path
import pyarrow.parquet as pq
from train_audit_core import Graph,group_id,fit_val
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;MAN=BASE/'data_manifest'
COLUMNS=['image','question_id','data_split','other_metadata']
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def records(path):
 with Path(path).open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def semantic_from_eval(path):
 m=re.fullmatch(r'(.+)_p(\d+)',Path(path or '').stem)
 if not m:raise ValueError('unrecognized DocVQA page path')
 return (m.group(1),str(int(m.group(2))+1))
def log(out,event,**fields):
 with (out/'progress.jsonl').open('a') as f:
  f.write(json.dumps({'event':event,'unix':time.time(),**fields})+'\n');f.flush();os.fsync(f.fileno())
def check_execution():
 config=read(HERE/'train_audit_execution_001.json');run_id=config['run_id'];expected_progress=str(HERE/('train_audit_'+run_id)/'progress.jsonl');monitor=HERE/config['supervisor_run_dir'].removeprefix('m3_train_data/')
 if sys.argv!=[config['worker'],run_id] or sys.executable!=config['python_executable'] or sys.version!=config['python_version'] or numpy.__version__!=config['numpy'] or pyarrow.__version__!=config['pyarrow'] or PIL.__version__!=config['pillow'] or scipy.__version__!=config['scipy']:raise ValueError('frozen execution or dependency mismatch')
 if sha(HERE/'train_audit_protocol_001.json')!=config['protocol_sha256'] or shutil.disk_usage(HERE).free<config['minimum_disk_free_bytes']:raise ValueError('protocol or disk guard')
 mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
 if mem.get('MemAvailable',0)<config['minimum_memory_available_bytes']:raise ValueError('memory guard')
 # Parent writes started.json immediately after spawning. Bound the startup race.
 started=monitor/'started.json'
 for _ in range(100):
  if started.exists():
   try:m=read(started);break
   except (json.JSONDecodeError,OSError):pass
  time.sleep(.1)
 else:raise ValueError('missing supervisor start record')
 parent_cmd=Path(f'/proc/{os.getppid()}/cmdline').read_bytes().split(b'\0');parent_cmd=[x.decode() for x in parent_cmd if x]
 if len(parent_cmd)<2 or parent_cmd[1]!=config['supervisor'] or m['worker_pid']!=os.getpid() or m['process_group']!=os.getpgrp() or os.getpgrp()!=os.getpid() or m['cwd']!=str(BASE) or m['command']!=[config['python_executable'],config['worker'],run_id] or m['progress_paths']!=[expected_progress] or m['timeout_seconds']!=config['timeout_seconds'] or m['stale_seconds']!=config['stale_seconds'] or m['snapshot_seconds']!=config['snapshot_seconds'] or m['poll_seconds']!=config['poll_seconds'] or m['grace_seconds']!=config['grace_seconds']:raise ValueError('unexpected supervisor process or bounds')
 return config

def main():
 config=check_execution();protocol=read(HERE/'train_audit_protocol_001.json');initial=read(BASE/'initial_plan.json')
 if time.time()+config['timeout_seconds']>initial['budget_end_unix']:raise ValueError('insufficient remaining budget')
 out=HERE/('train_audit_'+sys.argv[1]);out.mkdir(exist_ok=False);log(out,'started')
 for path,expected in protocol['inputs_sha256'].items():
  if sha(path)!=expected:raise ValueError('protocol input changed: '+path)
 summary=read(MAN/'manifest_summary.json');sem=read(HERE/'eval_path_semantics_001.json')
 if sem.get('status')!='passed' or sem.get('counts')!={'rows':5349,'same_doc':5349,'page_delta_-1':5349} or sem.get('source_sha256')!=sha(HERE/'validate_eval_paths.py') or sem.get('answer_or_question_values_extracted') is not False:raise ValueError('page mapping proof')
 sem_terminal=read(HERE/'monitor_eval_paths001/terminal.json');sem_started=read(HERE/'monitor_eval_paths001/started.json')
 if sem_started.get('command')!=['/home/dell/miniconda3/bin/python','m3_train_data/validate_eval_paths.py'] or sem_started.get('cwd')!=str(BASE):raise ValueError('page mapping command')
 if sem_terminal.get('reason')!='completed' or sem_terminal.get('worker_returncode')!=0 or sem_terminal.get('remaining_process_group')!=[] or sem_terminal.get('worker_pid')!=sem_started.get('worker_pid'):raise ValueError('page mapping completion')
 if sha(summary['sources']['DocVQA_VAL']['path'])!=sem['eval_source_sha256']:raise ValueError('DocVQA source changed')
 for name in ('images.jsonl','full_manifest.jsonl'):
  if sha(MAN/name)!=summary['outputs_sha256'][name]:raise ValueError('evaluation manifest changed')
 terminal=read(HERE/'monitor_fetch001/terminal.json');started=read(HERE/'monitor_fetch001/started.json');receipt=read(HERE/'train_fetch_001/receipt.json');inventory=read(HERE/'source_inventory_001.json')
 if started.get('command')!=['/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python','m3_train_data/fetch_train.py','--run-id','001'] or started.get('cwd')!=str(BASE) or started.get('progress_paths')!=[str(HERE/'train_fetch_001/progress.jsonl')]:raise ValueError('fetch command')
 if terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[] or terminal.get('worker_pid')!=started.get('worker_pid') or receipt.get('status')!='completed' or receipt.get('source_inventory_sha256')!=sha(HERE/'source_inventory_001.json') or receipt.get('source_revision')!=inventory['repository_revision']:raise ValueError('fetch completion evidence')
 if len(inventory['train_shards'])!=36 or len(receipt['shards'])!=36:raise ValueError('incomplete train source')
 shard_paths=[]
 for i,(item,got) in enumerate(zip(inventory['train_shards'],receipt['shards'])):
  path=HERE/'train_fetch_001'/'shards'/Path(item['path']).name
  if item['path']!=f'data/train-{i:05d}-of-00036.parquet' or Path(got['path'])!=path or got['source_path']!=item['path'] or got['bytes']!=item['bytes'] or got['sha256']!=item['lfs_sha256'] or path.stat().st_size!=item['bytes'] or sha(path)!=item['lfs_sha256']:raise ValueError('train shard identity')
  shard_paths.append(path);log(out,'shard_integrity',index=i)
 spec=importlib.util.spec_from_file_location('_frozen_manifest_build',MAN/'build.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 if mod.pillow_version!=summary['pillow'] or mod.scipy.__version__!=summary['scipy']:raise ValueError('fingerprint dependency version')
 g=Graph();eval_images=list(records(MAN/'images.jsonl'))
 if len(eval_images)!=5652:raise ValueError('eval image count')
 for i,x in enumerate(eval_images):
  if x['index']!=i:raise ValueError('eval image ordering')
  j=g.add(x['byte_sha256'],x['rgb_sha256'],int(x['phash'],16),eval_group=x['group_id'],is_eval=True)
  if j!=i:raise ValueError('eval raw duplicate')
  g.items[j]['width']=x['width'];g.items[j]['height']=x['height']
 docrows=0
 for x in records(MAN/'full_manifest.jsonl'):
  if x['task']=='DocVQA_VAL':
   docrows+=1;i=x['image_index'];im=eval_images[i]
   g.add(im['byte_sha256'],im['rgb_sha256'],int(im['phash'],16),semantic=semantic_from_eval(x['source_image_path']),eval_group=im['group_id'],is_eval=True)
 if docrows!=5349:raise ValueError('eval doc row count')
 log(out,'eval_loaded',images=len(eval_images),doc_rows=docrows)
 qids=set();rows=[];expected_rows=0
 for shard_index,path in enumerate(shard_paths):
  if time.time()>initial['budget_end_unix']-120:raise TimeoutError('deadline protection')
  parquet=pq.ParquetFile(path);expected_rows+=parquet.metadata.num_rows;seen=0
  for batch in parquet.iter_batches(batch_size=32,columns=COLUMNS):
   for record in batch.to_pylist():
    qid=record['question_id'];meta=record['other_metadata'];im=record['image']
    if record['data_split']!='train' or isinstance(qid,bool) or not isinstance(qid,int) or qid in qids or not isinstance(meta,dict) or not isinstance(im,dict) or not isinstance(im.get('bytes'),bytes):raise ValueError('unexpected training row')
    qids.add(qid);raw=im['bytes'];rawhash=hashlib.sha256(raw).hexdigest();doc=str(meta.get('ucsf_document_id') or '').strip();page=str(meta.get('ucsf_document_page_no') or '').strip()
    if not doc or not page.isdecimal() or int(page)<1:raise ValueError('invalid train page metadata')
    if rawhash in g.raw:
     existing=g.items[g.raw[rawhash]];rgb=existing['rgb'];bits=existing['phash']
    else:
     wh,rgb,bits=mod.describe(raw)
    idx=g.add(rawhash,rgb,bits,semantic=(doc,str(int(page))))
    if 'width' not in g.items[idx]:g.items[idx]['width'],g.items[idx]['height']=wh
    rows.append((qid,shard_index,seen,idx,doc,str(int(page))));seen+=1
   if seen%320==0:log(out,'shard_progress',index=shard_index,rows=seen,total_rows=len(rows),unique_images=len(g.items))
  if seen!=parquet.metadata.num_rows:raise ValueError('shard row count')
  log(out,'shard_completed',index=shard_index,rows=seen,total_rows=len(rows),unique_images=len(g.items))
 if len(rows)!=expected_rows or len(qids)!=len(rows):raise ValueError('row accounting')
 components=g.components();nodegroup={};eligible={};excluded=0
 for members in components.values():
  items=[g.items[i] for i in members];gid=group_id(items);is_eval=any(x['is_eval'] for x in items)
  for i in members:nodegroup[i]=(gid,is_eval)
  if is_eval:excluded+=1
  else:
   if gid in eligible:raise ValueError('group id collision')
   eligible[gid]=len(items)
 splits=fit_val(eligible)
 counts=collections.Counter();group_rows=collections.Counter()
 with (out/'rows.jsonl').open('x') as f:
  for qid,shard,row,idx,doc,page in rows:
   gid,excluded_eval=nodegroup[idx];split='excluded_eval' if excluded_eval else splits[gid]
   counts[split]+=1;group_rows[gid]+=1
   f.write(json.dumps({'question_id':qid,'shard_index':shard,'row_index':row,'image_byte_sha256':g.items[idx]['raw'],'group_id':gid,'split':split,'document_id':doc,'page_one_based':page},sort_keys=True)+'\n')
  f.flush();os.fsync(f.fileno())
 with (out/'images.jsonl').open('x') as f:
  for i,item in enumerate(g.items):
   gid,is_eval_group=nodegroup[i];split='excluded_eval' if is_eval_group else splits[gid]
   f.write(json.dumps({'image_index':i,'byte_sha256':item['raw'],'rgb_sha256':item['rgb'],'phash':f"{item['phash']:016x}",'width':item['width'],'height':item['height'],'is_eval_image':item['is_eval'],'group_id':gid,'split':split},sort_keys=True)+'\n')
  f.flush();os.fsync(f.fileno())
 with (out/'groups.jsonl').open('x') as f:
  for gid in sorted(eligible):f.write(json.dumps({'group_id':gid,'split':splits[gid],'images':eligible[gid],'questions':group_rows[gid]},sort_keys=True)+'\n')
  f.flush();os.fsync(f.fileno())
 result={'status':'provisional_completed','scope':'all pinned third-party TRAIN shards; no question or answer values extracted; source files contain these columns','counts':{'rows':len(rows),'train_unique_images':len(set(r[3] for r in rows)),'all_unique_images':len(g.items),'eval_images':len(eval_images),'eval_connected_components':excluded,'eligible_groups':len(eligible),'fit_groups':sum(s=='fit' for s in splits.values()),'validation_groups':sum(s=='validation' for s in splits.values()),'rows_by_split':dict(counts),'near_edges':g.near_edges},'read_columns':COLUMNS,'inputs':{'protocol_sha256':sha(HERE/'train_audit_protocol_001.json'),'receipt_sha256':sha(HERE/'train_fetch_001/receipt.json'),'core_sha256':sha(HERE/'train_audit_core.py'),'worker_sha256':sha(__file__),'execution_config_sha256':sha(HERE/'train_audit_execution_001.json'),'supervisor_sha256':sha(BASE/'monitor/supervise.py')},'outputs':{'rows.jsonl':sha(out/'rows.jsonl'),'images.jsonl':sha(out/'images.jsonl'),'groups.jsonl':sha(out/'groups.jsonl')},'answer_or_question_values_extracted':False,'predictions_opened':False,'training_performed':False,'limitations':protocol['limitations']+['Third-party repackage equivalence to the official release remains unverified.','Different pages of the same document are not automatically grouped.','This provisional audit must pass independent review before any training use.']}
 with (out/'result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 log(out,'completed',rows=len(rows));print(json.dumps(result['counts']))
if __name__=='__main__':main()
