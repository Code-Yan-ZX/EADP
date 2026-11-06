"""Independently reconstruct the complete PriorTR DEV input manifest, no labels."""
import collections,hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 b=Path(p).read_bytes()
 if b and not b.endswith(b'\n'):raise ValueError('partial input manifest')
 return [json.loads(x) for x in b.splitlines()]
def main():
 sample_path=HERE/'sample_001.json';v=read(sample_path);e1path=BASE/'e1/e1full001_completion.json';e=read(e1path)
 assert e['status']=='passed' and e['GT_opened'] is False and e['scoring_performed'] is False and v['E1_proof_sha256']==sha(e1path)
 for name,field in (('inputs','input_sha256'),('panel','panel_sha256')):
  desc=e['files'][name];assert sha(desc['path'])==desc['sha256']==v[field]
 fs=HERE/'monitor_freeze001/started.json';ft=HERE/'monitor_freeze001/terminal.json';s,t=read(fs),read(ft);cwd=Path(s['cwd'])
 assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid']==s['process_group']
 assert [(cwd/x).resolve() for x in s['command'] if Path(x).name=='freeze_panel.py']==[(HERE/'freeze_panel.py').resolve()]
 assert v['freeze_source_sha256']==sha(HERE/'freeze_panel.py') and v['design_after_existing_dev_results'] is True
 assert v['GT_opened'] is False and v['prediction_opened'] is False and v['scoring_performed'] is False
 inputs=lines(e['files']['inputs']['path']);panel=lines(e['files']['panel']['path']);dev=[x for x in panel if x['split']=='DEV'];index={(x['task'],str(x['qid'])):x for x in dev}
 assert len(inputs)==len(dev)==len(index)==384 and len({(x['task'],str(x['qid'])) for x in inputs})==384
 fields=('task','qid','group_id','image_path','image_sha256','message','message_sha256');expected=[]
 for row in inputs:
  key=(row['task'],str(row['qid']));assert key in index and index[key]['group_id']==row['group_id']
  assert not any(k in row for k in ('answer','answers','ground_truth','prediction'))
  assert sha(row['image_path'])==row['image_sha256'] and hashlib.sha256(json.dumps(row['message'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()==row['message_sha256']
  assert [x['value'] for x in row['message'] if x.get('type')=='image']==[row['image_path']]
  expected.append({k:row[k] for k in fields})
 assert v['sample']==expected and v['physical_generations']==1152 and v['arms']==['stock_native','author_default','priortr_k256']
 counts=dict(collections.Counter(x['task'] for x in expected));assert counts=={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128} and len({x['group_id'] for x in expected})==384
 previous={x['group_id'] for x in read(BASE/'e6/priortr/dev32/sample_001.json')['sample']};anchors={x['group_id'] for x in lines(BASE/'data_manifest/anchor_exclusion/anchor_groups.jsonl')}
 overlap={'prior_dev32_groups':len(previous & {x['group_id'] for x in expected}),'historical_e0_anchor_groups':len(anchors & {x['group_id'] for x in expected})}
 assert overlap['prior_dev32_groups']==32
 result={'status':'passed','independent_review':True,'sample_sha256':sha(sample_path),'reviewer_sha256':sha(__file__),'freeze_source_sha256':sha(HERE/'freeze_panel.py'),
         'freeze_started_sha256':sha(fs),'freeze_terminal_sha256':sha(ft),'input_sha256':e['files']['inputs']['sha256'],'panel_sha256':e['files']['panel']['sha256'],
         'questions_verified':384,'counts':counts,'prior_exposure_overlap':overlap,'GT_opened':False,'predictions_opened':False,'scoring_performed':False,
         'limits':['Entire existing DEV panel, frozen after earlier DEV findings; no prospective pre-score selection claim.',
                   'All 1152 outputs are to be newly generated; this audit contains no model outputs or labels.',
                   'PriorTR SDPA environment differs from E1/E4 FA2; no direct timing or numerical equivalence claim.']}
 out=HERE/'sample_001_independent_review.json'
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed','questions':384,'overlap':overlap}))
if __name__=='__main__':main()
