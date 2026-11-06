"""Extract frozen posthoc features only after independent evidence gate."""
import argparse,hashlib,importlib.util,json,math,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent
PINS={'gate': '2faa248ea5c1c298ee31fd98cb75c03a205f5e3a38165f6e7af79227c21627df', 'features': '0a27e3b35deb0c6f8b8a493e820d8504e9100382f5e8f27a197bf0c9c67d3a6f'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(name,path,digest):
 path=Path(path).resolve()
 if sha(path)!=digest or name in sys.modules:raise ValueError('unreviewed/preloaded module')
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def rows(path):
 with Path(path).open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def identities(pilot,score_rows,panel_rows):
 target={(x['task'],str(x['qid'])):x for x in pilot};scores={};panel={}
 for r in score_rows:
  key=(r['task'],str(r['qid']))
  if key not in target:continue
  if key in scores or r['group_id']!=target[key]['group_id']:raise ValueError('ambiguous score identity')
  vals=[]
  for k in (128,256,512):
   v=r['arms']['K'+str(k)+'-D1'];a=float(v['score']);b=float(v['independent_score'])
   if not math.isfinite(a) or not math.isfinite(b) or not 0<=a<=1 or not 0<=b<=1 or abs(a-b)>1e-12:raise ValueError('bad independently scored value')
   vals.append(a)
  scores[key]=vals
 for r in panel_rows:
  key=(r['task'],str(r['qid']));k=r.get('budget')
  if key not in target or r.get('ds_on') is not True or k not in (128,256,512):continue
  ident=(key,k)
  if ident in panel or any(r.get(x)!=target[key][x] for x in ('task','group_id','message_sha256','image_sha256')) or r.get('actual_injection_verified') is not True:raise ValueError('ambiguous panel identity')
  panel[ident]=True
 if len(scores)!=32 or len(panel)!=96 or any((key,k) not in panel for key in target for k in (128,256,512)):raise ValueError('incomplete score/panel identity cohort')
 return scores
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
 if a.out.resolve()!=D/'result001.json' or a.out.exists():raise ValueError('unexpected output')
 gate=load('_multistream_budget_gate',D/'gate.py',PINS['gate']);frozen=gate.prelaunch();upstream=gate.prior(frozen)
 feature=load('_multistream_budget_features',D/'features.py',PINS['features']);pilot=json.loads((B/'e5/selector_real32/sample_manifest_001.json').read_text())['pilot']
 scores=identities(pilot,rows(B/'e4_analysis/score_panel003/scores.jsonl'),rows(B/'e4_panel_v2/panel_003/records.jsonl'))
 import torch
 result=[]
 for i,row in enumerate(pilot):
  path=B/'e5/selector_real32/run_001'/('image_%03d'%i)/'capture.pt'
  if frozen['capture_inventory_sha256'].get(str(path))!=sha(path):raise ValueError('capture archive hash mismatch')
  capture=torch.load(path,map_location='cpu',weights_only=True);main,ds=capture['actual']['visual']
  if len(ds)!=3:raise ValueError('three DeepStack streams required')
  arrays=[x.detach().to(dtype=torch.float64,device='cpu').numpy() for x in (main,*ds)];values=feature.compute(arrays);s=scores[(row['task'],str(row['qid']))]
  result.append({'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'features':values,'score128':s[0],'score256':s[1],'score512':s[2],'delta512_128':s[2]-s[0]})
  del arrays,capture,main,ds
 summary=feature.analyze(result);gate.prelaunch()
 if gate.sha(B/'e4_analysis/score_panel003/scores.jsonl')!=upstream['scores_sha256'] or gate.sha(B/'e4_panel_v2/panel_003/records.jsonl')!=upstream['records_sha256']:raise ValueError('upstream changed during extraction')
 value={'status':'passed','questions':32,'task_counts':{'DocVQA_VAL':11,'TextVQA_VAL':11,'AI2D_TEST':10},'source_sha256':sha(__file__),'plan_freeze_sha256':sha(D/'plan_freeze001.json'),'code_plan_sha256':sha(D/'code_plan001.json'),'upstream':upstream,'rows':result,'correlations':summary,'GT_opened':False,'predictions_opened':True,'scoring_performed':False,'GPU_used':False,'scope':'posthoc 32-image descriptive correlation; mixed DEV prediction files parsed only for identity and pre-existing scores; no label, training, routing or confirmatory inference','errors':[]}
 with a.out.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
