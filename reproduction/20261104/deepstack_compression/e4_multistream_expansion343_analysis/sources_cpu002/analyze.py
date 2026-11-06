"""Complete-only 384-image posthoc descriptive analysis; no new GT."""
import argparse,hashlib,importlib.util,json,math,os,sys
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;B=D.parent
PINS={'gate':'fc0d7a78e5142ceb6bd3b921e7aeae59c408afb3f1c30e9da197b49123f76bae','stats':'c2bce04062c2c88b6d8e42766a0bc15fa404edc7638448be6bac41f15706da03'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(name,path,digest):
 if sha(path)!=digest or name in sys.modules:raise ValueError('changed/preloaded module')
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def key(x):return x['task'],str(x['qid'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();out=a.out.resolve()
 if out!=D/'analysis001.json' or out.exists():raise ValueError('output path')
 gate=load('_ms343_analysis_gate',D/'gate.py',PINS['gate']);proof=gate.accepted();st=load('_ms343_analysis_stats',D/'stats.py',PINS['stats'])
 orig=read(B/'e5/selector_real32/sample_manifest_001.json')['pilot'];sel=read(B/'e4_multistream_expansion352/selection001.json');pilot=sel['pilot'];pk={key(x) for x in pilot};remaining=[x for x in sel['remaining'] if key(x) not in pk]
 cohorts=(orig,pilot,remaining);assert list(map(len,cohorts))==[32,9,343]
 chosen={key(r):r for c in cohorts for r in c}
 if len(chosen)!=384:raise ValueError('cohort overlap')
 e1=read(B/'e1/e1full001_completion.json');inputs=Path(e1['files']['inputs']['path'])
 if sha(inputs)!=e1['files']['inputs']['sha256']:raise ValueError('E1 input hash')
 seen={}
 for r in lines(inputs):
  k=key(r)
  if k in chosen:
   if k in seen or not st.accepted_input_identity(r,chosen[k]):raise ValueError('E1 identity')
   seen[k]=True
 if set(seen)!=set(chosen):raise ValueError('E1 complete cohort')
 panels={};scores={}
 for r in lines(B/'e4_panel_v2/panel_003/records.jsonl'):
  k=key(r);budget=r.get('budget')
  if k not in chosen or r.get('ds_on') is not True or budget not in (128,256,512):continue
  if (k,budget) in panels or r.get('actual_injection_verified') is not True or any(str(r.get(x))!=str(chosen[k][x]) for x in ('task','qid','group_id','image_sha256','message_sha256')):raise ValueError('E4 panel identity')
  panels[k,budget]=True
 for r in lines(B/'e4_analysis/score_panel003/scores.jsonl'):
  k=key(r)
  if k not in chosen:continue
  if k in scores or r['group_id']!=chosen[k]['group_id']:raise ValueError('score identity')
  vals=[]
  for budget in (128,256,512):
   arm=r['arms']['K'+str(budget)+'-D1'];s=arm['score'];ind=arm['independent_score']
   if type(s) not in (int,float) or type(ind) not in (int,float) or not math.isfinite(s) or not math.isfinite(ind) or not 0<=s<=1 or not 0<=ind<=1 or abs(s-ind)>1e-12:raise ValueError('bad author/independent score')
   vals.append(s)
  scores[k]=vals
 if len(panels)!=1152 or set(scores)!=set(chosen):raise ValueError('incomplete panel or scores')
 features={};old=read(B/'e4_multistream_budget_diagnostic_v2/result001.json')
 if old.get('status')!='passed' or old.get('questions')!=32:raise ValueError('old result')
 for r in old['rows']:
  k=key(r)
  if k in features or k not in {key(x) for x in orig} or r['group_id']!=chosen[k]['group_id'] or any(abs(float(r['score'+str(b)])-scores[k][i])>1e-12 for i,b in enumerate((128,256,512))):raise ValueError('old result identity/score')
  features[k]=r['features']
 def append_capture(rows,run):
  for i,target in enumerate(rows):
   k=key(target);rec=read(run/'captures'/('image_%03d.json'%i));norm=np.load(run/'captures'/('image_%03d_norms.npy'%i),allow_pickle=False)
   if k in features or rec.get('status')!='passed' or rec.get('hooks_clean') is not True or rec.get('index')!=i or any(str(rec.get(x))!=str(target[x]) for x in ('task','qid','group_id','image_sha256','message_sha256')):raise ValueError('capture identity')
   fresh=st.norm_features(norm)
   if set(fresh)!=set(rec['features']):raise ValueError('feature keys')
   for name,v in fresh.items():
    x=rec['features'][name]
    if not ((v is None and x is None) or (v is not None and x is not None and math.isclose(v,x,rel_tol=1e-12,abs_tol=1e-12))):raise ValueError('feature mismatch')
   features[k]=fresh
 append_capture(pilot,B/'e4_multistream_expansion352_gpu_v4/run001');append_capture(remaining,B/'e4_multistream_expansion343_gpu/run001')
 if set(features)!=set(chosen):raise ValueError('features incomplete')
 rows=[dict(task=k[0],qid=k[1],group_id=chosen[k]['group_id'],image_sha256=chosen[k]['image_sha256'],message_sha256=chosen[k]['message_sha256'],features=features[k],scores=scores[k],delta512_128=scores[k][2]-scores[k][0]) for c in cohorts for r in c for k in (key(r),)]
 summary=st.summarize(rows)
 if gate.accepted()!=proof:raise ValueError('source evidence changed during analysis')
 value={'status':'passed','questions':384,'cohort_counts':[32,9,343],'summary':summary,'rows':rows,'upstream':proof,'GT_opened':False,'predictions_opened':True,'scoring_performed':False,'training':False,'scope':'posthoc same-DEV descriptive association; no causal, confirmatory, routing, or significance claims','limits':['32 and 9 subsets were already observed; all 384 remain same DEV.','Cross-task groups can repeat; 128 image groups are unique within each task.','343 and nine full high-dimensional tensors not saved: norm association remains runtime capture evidence.','Existing DEV mixed predictions parsed only for identity and pre-existing scores.'],'errors':[]}
 with out.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
