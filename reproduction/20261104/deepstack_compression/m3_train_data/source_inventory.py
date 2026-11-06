"""Read public repository metadata only; no dataset rows, labels or images."""
import datetime,json,os,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO='pixparse/docvqa-single-page-questions'
def get(path):
 req=urllib.request.Request('https://huggingface.co/api/datasets/'+REPO+path,headers={'User-Agent':'EADP-research-source-audit/1.0'})
 with urllib.request.urlopen(req,timeout=30) as f:return json.load(f)
def main():
 info=get('');revision=info['sha']
 if len(revision)!=40:raise ValueError('unexpected repository revision')
 tree=get('/tree/'+revision+'?recursive=true&expand=true')
 entries=[{'path':x['path'],'bytes':x.get('size'),'git_blob_oid':x.get('oid'),'lfs_sha256':x.get('lfs',{}).get('oid')} for x in tree if x.get('type')=='file' and x['path'].startswith('data/train-') and x['path'].endswith('.parquet')]
 entries.sort(key=lambda x:x['path'])
 if len(entries)!=36 or any(not isinstance(x['bytes'],int) or len(x.get('lfs_sha256') or '')!=64 for x in entries):raise ValueError('unexpected train inventory')
 result={'status':'metadata_collected','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repository':'https://huggingface.co/datasets/'+REPO,'repository_revision':revision,'source_card':'https://huggingface.co/datasets/'+REPO+'/blob/'+revision+'/README.md','original_site':'https://site.docvqa.org/datasets/docvqa','scope':'public third-party repack metadata only; train parquet objects, no rows/images/references downloaded or parsed','train_shards':entries,'total_train_bytes':sum(x['bytes'] for x in entries),'limitations':['Original project directs downloads through a login-gated portal; this is a public third-party repack and original equivalence is not established by metadata alone.','Train eligibility requires cross-task exact and near-duplicate exclusion against all evaluation splits before teacher data preparation.','No training, scoring, model or label access occurred.'],'GT_opened':False,'predictions_opened':False,'files_downloaded':False}
 with (ROOT/'source_inventory_001.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':result['status'],'revision':revision,'train_shards':len(entries),'total_train_bytes':result['total_train_bytes']}))
if __name__=='__main__':main()
