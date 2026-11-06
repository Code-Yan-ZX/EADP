"""Independent prelaunch review of label-free train image grouping; no labels/training."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent.parent;D=ROOT/'m3_train_data'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert a.out.resolve()==ROOT/'e1_independent_review/train_audit_prelaunch_review.json'
 paths=[D/x for x in ('train_audit.py','train_audit_core.py','test_train_audit_core.py','train_audit_execution_001.json','train_audit_protocol_001.json')]+[ROOT/'monitor/supervise.py'];before={str(p):sha(p) for p in paths}
 config=read(D/'train_audit_execution_001.json');protocol=read(D/'train_audit_protocol_001.json');assert config['protocol_sha256']==sha(D/'train_audit_protocol_001.json') and protocol['read_columns']==['image','question_id','data_split','other_metadata'] and config['timeout_seconds']==7200 and config['stale_seconds']==1800 and config['snapshot_seconds']==3600 and config['minimum_disk_free_bytes']>=20*1024**3 and config['minimum_memory_available_bytes']>=8*1024**3
 for p,d in protocol['inputs_sha256'].items():assert sha(p)==d
 for m,cmd in [('monitor_eval_paths001',['/home/dell/miniconda3/bin/python','m3_train_data/validate_eval_paths.py']),('monitor_fetch001',['/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python','m3_train_data/fetch_train.py','--run-id','001'])]:
  start=read(D/m/'started.json');end=read(D/m/'terminal.json');assert start['command']==cmd and start['cwd']==str(ROOT) and start['worker_pid']==start['process_group']==end['worker_pid'] and end['reason']=='completed' and end['worker_returncode']==0 and end['remaining_process_group']==[]
 receipt=read(D/'train_fetch_001/receipt.json');inventory=read(D/'source_inventory_001.json');assert receipt['status']=='completed' and receipt['source_inventory_sha256']==sha(D/'source_inventory_001.json') and receipt['source_revision']==inventory['repository_revision'] and len(receipt['shards'])==len(inventory['train_shards'])==36
 n=0
 for i,(entry,got) in enumerate(zip(inventory['train_shards'],receipt['shards'])):
  path=D/'train_fetch_001/shards'/Path(entry['path']).name;assert entry['path']==f'data/train-{i:05d}-of-00036.parquet' and got['source_path']==entry['path'] and got['path']==str(path) and got['sha256']==entry['lfs_sha256'] and got['bytes']==entry['bytes']==path.stat().st_size
  n+=pq.ParquetFile(path).metadata.num_rows
 assert n==39463
 c=subprocess.run([config['python_executable'],'-m','unittest','-v','test_train_audit_core'],cwd=D,text=True,capture_output=True,timeout=120);assert c.returncode==0 and 'Ran 8 tests' in c.stderr and c.stderr.rstrip().endswith('OK')
 c2=subprocess.run([config['python_executable'],str(D/'train_audit.py'),'001'],cwd=ROOT,text=True,capture_output=True,timeout=30);assert c2.returncode!=0 and ('frozen execution or dependency mismatch' in c2.stderr or 'missing supervisor start record' in c2.stderr)
 assert before=={str(p):sha(p) for p in paths}
 out={'status':'passed','independent_review':True,'reviewer_sha256':sha(__file__),'source_sha256':before,'footer_rows':n,'source_shards':36,'worker_rehashes_shards':True,'labels_extracted':False,'training_performed':False,'limits':['Provisional image-level audit only; training remains unapproved.','Different pages of the same document do not automatically join.','All source files contain answer/question fields; projected columns omit their values.','Third-party repack identity to official train release remains unverified.']}
 with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed','shards':36,'footer_rows':n}))
if __name__=='__main__':main()
