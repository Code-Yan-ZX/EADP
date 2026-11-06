"""Download one pinned public train shard for schema/dedup feasibility only."""
import argparse,datetime,hashlib,json,os,time,urllib.error,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
 if not a.run_id.isalnum():raise ValueError('invalid run id')
 source=HERE/'source_inventory_001.json';v=json.loads(source.read_text());entry=v['train_shards'][0];revision=v['repository_revision'];out=HERE/('probe_'+a.run_id);out.mkdir(exist_ok=False)
 if not entry['path'].startswith('data/train-') or len(revision)!=40:raise ValueError('nontrain source')
 url=v['repository']+'/resolve/'+revision+'/'+entry['path']+'?download=true';partial=out/'shard.parquet.part';completed=out/'shard.parquet';progress=out/'progress.jsonl';start=time.time();h=hashlib.sha256();size=0
 def log(row):
  with progress.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
 log({'event':'start','expected_bytes':entry['bytes'],'unix':start})
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'EADP-research-source-audit/1.0'})
  with urllib.request.urlopen(req,timeout=60) as r,partial.open('xb') as f:
   while True:
    buf=r.read(1024*1024)
    if not buf:break
    size+=len(buf)
    if size>entry['bytes']:raise ValueError('download exceeds frozen size')
    h.update(buf);f.write(buf)
    if size%(16*1024*1024)==0:log({'event':'progress','bytes':size,'unix':time.time()})
   f.flush();os.fsync(f.fileno())
  digest=h.hexdigest()
  if size!=entry['bytes'] or digest!=entry['lfs_sha256']:raise ValueError('download size or SHA mismatch')
  os.rename(partial,completed);log({'event':'complete','bytes':size,'unix':time.time()})
  result={'status':'passed','source_inventory_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_revision':revision,'entry':entry,'actual_bytes':size,'actual_sha256':digest,'GT_opened':False,'predictions_opened':False,'scope':'one public third-party train shard downloaded; no rows, images or answers parsed; not an eligible training split','started_unix':start,'finished_unix':time.time()}
  with (out/'receipt.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  print(json.dumps({'status':'passed','bytes':size,'sha256':digest}))
 except Exception as exc:
  log({'event':'failed','error_type':type(exc).__name__,'bytes':size,'unix':time.time()})
  raise
if __name__=='__main__':main()
