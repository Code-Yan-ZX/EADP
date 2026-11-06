"""CPU preflight to bind light-adapter anchors to accepted E1, E4 and B.

 Separate interpreter prevents E1 and E4-C modules named contract colliding.
 Reads only accepted DEV diagnostics and no reference answers or scores.
"""
import argparse,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
from identity_gate import validate_b_identity

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 b=validate_b_identity()
 sys.path[:0]=[str(BASE/'e4_panel_v2'),str(BASE/'e4'),str(BASE/'e1_analysis')]
 from panel_v2_imports import load_local
 from identity_worker import read_jsonl
 gate=load_local('panel_gate')
 e1=BASE/'e1/e1full001_completion.json';es=BASE/'e1/monitorverifye1full001/started.json';et=BASE/'e1/monitorverifye1full001/terminal.json'
 smoke=BASE/'e4/smoke_001';sp=BASE/'e4/smoke_001_verification.json';ss=BASE/'e4/monitor_verify_smoke001/started.json';st=BASE/'e4/monitor_verify_smoke001/terminal.json'
 proof,inputs=gate.validated_dev(e1,es,et);gate.validate_smoke_gate(e1,smoke,sp,ss,st);review=gate.validate_smoke_review(sp,st)
 anchors=json.loads((BASE/'e4_b_identity/b_001/frozen.json').read_text())['anchors']
 from light_smoke_plan import make_plan
 plan=make_plan(anchors);lookup={(x['task'],str(x['qid'])):x for x in inputs};rows=[]
 for anchor in anchors:
  key=(anchor['task'],str(anchor['qid']));row=lookup.get(key)
  if row is None or any(row.get(k)!=anchor[k] for k in ('task','group_id','message_sha256','image_sha256')):raise ValueError('B/E1 anchor mismatch')
  rows.append(dict(row,split='DEV'))
 wanted={(x['task'],str(x['qid'])) for x in rows}
 b_records=[x for x in read_jsonl(BASE/'e4_b_identity/b_001/records.jsonl') if (x['task'],str(x['qid'])) in wanted]
 old_on=[]
 for name in ('records.jsonl','forced_records.jsonl'):
  old_on.extend(x for x in read_jsonl(smoke/name) if (x['task'],str(x['qid'])) in wanted and x['budget']==256 and x['ds_on'] is True)
 if len(b_records)!=8 or len(old_on)!=4:raise ValueError('accepted reference inventory mismatch')
 for row in rows:
  candidates=[x for x in b_records if x['task']==row['task'] and str(x['qid'])==str(row['qid'])]
  starts={x['image_start'] for x in candidates};processors=[x['processor'] for x in candidates]
  if len(starts)!=1 or not processors or any(x!=processors[0] for x in processors):raise ValueError('B processor identity inconsistent')
  row['image_start']=next(iter(starts));row['processor_expected']=processors[0]
 payload={'status':'passed','scope':'accepted DEV diagnostic references only; no GT or score','b_gate':b,'e4_review':review,'plan':plan,'rows':rows,'b_records':b_records,'old_on':old_on}
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x') as f:json.dump(payload,f,ensure_ascii=False,separators=(',',':'));f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed','anchors':len(rows),'B_records':len(b_records),'old_ON_records':len(old_on),'new_generations_planned':len(plan['jobs'])}))
if __name__=='__main__':main()
