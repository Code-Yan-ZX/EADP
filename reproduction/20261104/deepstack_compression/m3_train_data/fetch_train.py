"""Fetch the frozen public third-party TRAIN-only shard inventory, no parsing."""
import argparse,hashlib,json,os,shutil,time,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
 if not a.run_id.isalnum():raise ValueError('invalid run id')
 source=HERE/'source_inventory_001.json';v=json.loads(source.read_text());entries=v['train_shards'];rev=v['repository_revision'];initial=json.loads((HERE.parent/'initial_plan.json').read_text())
 if len(entries)!=36 or len(rev)!=40 or time.time()+7200>initial['budget_end_unix'] or shutil.disk_usage(HERE).free<50*1024**3:raise ValueError('inventory, deadline or disk bound')
 out=HERE/('train_fetch_'+a.run_id);out.mkdir(exist_ok=False);shards=out/'shards';shards.mkdir();events=out/'progress.jsonl'
 def log(event,**fields):
  with events.open('a') as f:f.write(json.dumps({'event':event,'unix':time.time(),**fields})+'\n');f.flush();os.fsync(f.fileno())
 log('started',planned_shards=36,total_bytes=v['total_train_bytes']);results=[]
 for index,item in enumerate(entries):
  if time.time()>initial['budget_end_unix']-120:raise TimeoutError('deadline protection')
  path=item['path'];name=Path(path).name
  if not path.startswith('data/train-') or not name.endswith('.parquet') or len(item['lfs_sha256'])!=64:raise ValueError('unexpected training object')
  target=shards/name
  if index==0:
   existing=HERE/'probe_001/shard.parquet';receipt=json.loads((HERE/'probe_001/receipt.json').read_text())
   if receipt['status']!='passed' or receipt['actual_sha256']!=item['lfs_sha256'] or sha(existing)!=item['lfs_sha256']:raise ValueError('first shard probe differs')
   os.link(existing,target);log('reused_verified_probe',index=index,bytes=item['bytes'])
  else:
   last_error=None
   for attempt in range(1,4):
    partial=shards/(name+'.attempt'+str(attempt)+'.part');count=0;h=hashlib.sha256();log('download_started',index=index,attempt=attempt)
    try:
     req=urllib.request.Request(v['repository']+'/resolve/'+rev+'/'+path+'?download=true',headers={'User-Agent':'EADP-research-source-audit/1.0'})
     with urllib.request.urlopen(req,timeout=60) as response,partial.open('xb') as f:
      while True:
       block=response.read(1024*1024)
       if not block:break
       count+=len(block)
       if count>item['bytes']:raise ValueError('download exceeds expected size')
       h.update(block);f.write(block)
       if count%(64*1024*1024)==0:log('download_progress',index=index,attempt=attempt,bytes=count)
      f.flush();os.fsync(f.fileno())
     if count!=item['bytes'] or h.hexdigest()!=item['lfs_sha256']:raise ValueError('download size or SHA mismatch')
     os.rename(partial,target);log('download_completed',index=index,bytes=count,attempt=attempt);last_error=None;break
    except Exception as exc:
     last_error=type(exc).__name__;log('download_failed',index=index,attempt=attempt,bytes=count,error_type=last_error)
   if last_error:raise RuntimeError('train shard download attempts exhausted: '+name+' '+last_error)
  if target.stat().st_size!=item['bytes'] or sha(target)!=item['lfs_sha256']:raise ValueError('stored shard identity mismatch')
  results.append({'path':str(target.resolve()),'source_path':path,'bytes':item['bytes'],'sha256':item['lfs_sha256']})
 if len(results)!=36:raise ValueError('incomplete inventory')
 result={'status':'completed','source_inventory_sha256':sha(source),'source_revision':rev,'shards':results,'GT_opened':False,'predictions_opened':False,'training_performed':False,'scope':'public third-party TRAIN-only parquet bytes; no rows or answers parsed; no eligible split until source and full evaluation overlap audit','finished_unix':time.time()}
 with (out/'receipt.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 log('completed',shards=36);print(json.dumps({'status':'completed','shards':36,'bytes':sum(x['bytes'] for x in results)}))
if __name__=='__main__':main()
