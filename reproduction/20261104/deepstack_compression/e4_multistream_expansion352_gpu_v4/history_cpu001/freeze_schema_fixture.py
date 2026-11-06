"""Freeze nine accepted-reference identity fields; no answer/prediction values saved."""
import hashlib,importlib.util,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;P=B/'e4_multistream_expansion352'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 out=D/'schema_fixture001.json'
 if out.exists():raise ValueError('exists')
 spec=importlib.util.spec_from_file_location('_schema_fixture_contract',D/'full_reference.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 selected=json.loads((P/'selection001.json').read_text())['pilot'];wanted={(r['task'],str(r['qid'])):r for r in selected};ep=B/'e1/e1full001_completion.json';eq=json.loads(ep.read_text());ej=Path(eq['files']['journal']['path']);fp=B/'e4_panel_v2/panel_003_verification.json';fq=json.loads(fp.read_text());fj=B/'e4_panel_v2/panel_003/records.jsonl'
 if eq.get('status')!='passed' or eq['files']['journal']['sha256']!=sha(ej) or fq.get('status')!='passed' or fq['files_sha256'].get(str(fj.resolve()))!=sha(fj):raise ValueError('accepted identity sources')
 e1=[];e4=[]
 with ej.open() as f:
  for line in f:
   r=json.loads(line);key=(r.get('task'),str(r.get('qid')))
   if key in wanted and r.get('arm')=='P3-D1' and r.get('forced') is False:
    if any(k not in r for k in m.FIELDS):raise ValueError('incomplete E1 fields')
    e1.append(dict({k:r[k] for k in m.FIELDS},arm='P3-D1',forced=False))
 fields=set(m.E4_MAP.values())|{'task','qid','group_id','message_sha256','image_sha256','budget','ds_on','actual_injection_verified','selection_mode'}
 with fj.open() as f:
  for line in f:
   r=json.loads(line);key=(r.get('task'),str(r.get('qid')))
   if key in wanted and r.get('budget')==256 and r.get('ds_on') is True:
    if any(k not in r for k in fields):raise ValueError('incomplete E4 fields')
    e4.append({k:r[k] for k in fields})
 if len(e1)!=9 or len(e4)!=9 or len({(r['task'],str(r['qid'])) for r in e1})!=9 or len({(r['task'],str(r['qid'])) for r in e4})!=9 or sha(ej)!=eq['files']['journal']['sha256'] or sha(fj)!=fq['files_sha256'][str(fj.resolve())]:raise ValueError('coverage or source changed')
 obj={'status':'frozen','selected':selected,'e1_fields':e1,'e4_fields':e4,'e1_proof_sha256':sha(ep),'e1_journal_sha256':sha(ej),'e4_proof_sha256':sha(fp),'e4_journal_sha256':sha(fj),'selection_sha256':sha(P/'selection001.json'),'freeze_script_sha256':sha(__file__),'predictions_opened':True,'prediction_scope':'accepted E1/E4 mixed records parsed only for identity fields; no answer or output text retained','GT_opened':False,'scoring_performed':False,'GPU_used':False,'errors':[]}
 with out.open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
