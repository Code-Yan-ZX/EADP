"""Independent reconstruction of batch-2 DEV selection, without predictions."""
import collections,hashlib,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def main():
 sample=HERE/'sample_002.json';p=read(sample);first=read(ROOT/'m3_oracle/sample_001.json');e1=read(ROOT/'e1/e1full001_completion.json');inputs=Path(e1['files']['inputs']['path']);panel=ROOT/'data_manifest/panel_manifest.jsonl'
 t=read(HERE/'monitor_freeze002/terminal.json');s=read(HERE/'monitor_freeze002/started.json');cmd=s['command'];cwd=Path(s['cwd'])
 assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in cmd if Path(x).name=='freeze_sample.py']==[(HERE/'freeze_sample.py').resolve()]
 assert p['batch']==2 and p['selection_after_first_batch_scores'] is True and p['selection_uses_scores'] is False and p['GT_opened'] is False and p['predictions_opened'] is False and all(sha(k)==v for k,v in p['source_sha256'].items())
 assert sha(inputs)==e1['files']['inputs']['sha256'];records=lines(inputs);lookup={(x['task'],str(x['qid'])):x for x in records};dev=[x for x in lines(panel) if x['split']=='DEV'];assert len(records)==len(lookup)==len(dev)==384 and len({x['group_id'] for x in dev})==384
 old=first['examples'];oldgroups={x['group_id'] for x in old};assert len(old)==len(oldgroups)==128
 selected=[]
 for task,n in (('DocVQA_VAL',64),('TextVQA_VAL',32),('AI2D_TEST',32)):
  group=[x for x in dev if x['task']==task];assert len(group)==128
  group.sort(key=lambda x:(hashlib.sha256(('20261104|'+x['group_id']).encode()).digest(),x['group_id']))
  assert [x['group_id'] for x in old if x['task']==task]==[x['group_id'] for x in group[:n]]
  for row in [x for x in group if x['group_id'] not in oldgroups][:n]:
   item=lookup[(task,str(row['qid']))];assert item['group_id']==row['group_id'] and sha(item['image_path'])==item['image_sha256'] and hashlib.sha256(json.dumps(item['message'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()==item['message_sha256']
   selected.append({k:item[k] for k in ('task','qid','group_id','image_path','image_sha256','message_sha256')})
 assert p['examples']==selected and p['counts']==dict((('DocVQA_VAL',64),('TextVQA_VAL',32),('AI2D_TEST',32))) and len(selected)==128 and not ({x['group_id'] for x in selected}&oldgroups)
 out={'status':'passed','independent_review':True,'questions':128,'images_hashed':128,'counts':p['counts'],'overlap_first_batch':0,'GT_opened':False,'predictions_opened':False,'scoring_performed':False,
      'sample_sha256':sha(sample),'first_sample_sha256':sha(ROOT/'m3_oracle/sample_001.json'),'freeze_source_sha256':sha(HERE/'freeze_sample.py'),'freeze_started_sha256':sha(HERE/'monitor_freeze002/started.json'),'freeze_terminal_sha256':sha(HERE/'monitor_freeze002/terminal.json'),'reviewer_sha256':sha(__file__),
      'scope':'selection from existing DEV inputs only; second exploratory look after first batch, no new labels or predictions','finished_unix':time.time()}
 with (HERE/'sample_002_independent_review.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','questions':len(selected),'overlap_first_batch':0}))
if __name__=='__main__':main()
