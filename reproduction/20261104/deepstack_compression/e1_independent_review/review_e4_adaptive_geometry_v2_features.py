"""Recompute all accepted DEV selector-geometry features, without labels or scores."""
import argparse,hashlib,importlib.util,json,math,os
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e4_adaptive_geometry_v2';OUT=I/'e4_adaptive_geometry_v2_features_review.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def independent(pos,start,ids):
 if len(ids)!=128 or len(set(ids))!=128 or any(type(x) is not int or not 0<=x<1024 for x in ids):raise ValueError('selection identity')
 coords=[(pos[1][0][start+i],pos[2][0][start+i]) for i in range(1024)]
 if any(type(a) is not int or type(b) is not int for a,b in coords) or len(set(coords))!=1024:raise ValueError('grid identity')
 lo=[min(x[j] for x in coords) for j in (0,1)]; hi=[max(x[j] for x in coords) for j in (0,1)];extent=[hi[j]-lo[j]+1 for j in (0,1)]
 if min(extent)<2 or extent[0]*extent[1]!=1024:raise ValueError('grid extent')
 chosen=[coords[i] for i in ids];norm=[((a-lo[0])/(extent[0]-1),(b-lo[1])/(extent[1]-1)) for a,b in chosen]
 nearest=sum(min(math.hypot(a-c,b-d) for j,(c,d) in enumerate(norm) if i!=j) for i,(a,b) in enumerate(norm))/128
 cells8={((a-lo[0])*8//extent[0],(b-lo[1])*8//extent[1]) for a,b in chosen};counts={}
 for a,b in chosen:
  cell=((a-lo[0])*4//extent[0],(b-lo[1])*4//extent[1]);counts[cell]=counts.get(cell,0)+1
 entropy=-sum((n/128)*math.log2(n/128) for n in counts.values())
 return nearest,len(cells8),entropy
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.resolve()!=OUT.resolve() or OUT.exists():raise ValueError('unplanned output')
 path=D/'admission.py'
 if sha(path)!='ec47e61d03abd9fb0a4f582df791e08b02bd90d96c902ac8640d8ff2c37bdee4':raise ValueError('admission source changed')
 spec=importlib.util.spec_from_file_location('_independent_geometry_features_gate',path);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
 approval=g.admit();mon=D/'monitor_extract001';feature=D/'features001';g.terminal(mon,D/'extract.py',feature);f=read(feature/'extraction.json')
 if f.get('status')!='completed' or f.get('n')!=384 or f.get('GT_opened') is not False or f.get('score_values_opened') is not False or f.get('approval')!=approval or f.get('extractor_sha256')!=sha(D/'extract.py') or f.get('feature_code_sha256')!=sha(D/'features.py') or f.get('features_sha256')!=sha(feature/'features.jsonl'):raise ValueError('extraction evidence')
 inputs={(r['task'],str(r['qid'])):r for r in rows(B/'e1/input_preparation_002/inputs.jsonl')}
 features={}
 for v in rows(feature/'features.jsonl'):
  key=(v['task'],str(v['qid']))
  if key in features:raise ValueError('duplicate feature')
  features[key]=v
 if len(inputs)!=384 or set(inputs)!=set(features):raise ValueError('feature inventory')
 seen=set();counts={}
 for record in rows(B/'e4_panel_v2/panel_003/records.jsonl'):
  if record.get('budget')!=128 or record.get('ds_on') is not True or record.get('forced') is not False:continue
  key=(record['task'],str(record['qid']));inp=inputs.get(key);row=features.get(key)
  if key in seen or inp is None or row is None or any(record.get(k)!=inp.get(k) for k in ('task','group_id','image_sha256','message_sha256')) or any(row.get(k)!=record.get(k) for k in ('task','group_id','image_sha256')):raise ValueError('source/feature binding')
  near,occ,entropy=independent(record['full_position_ids'],record['image_start'],record['spatial_ids'])
  if abs(row['nearest_mean']-near)>1e-12 or row['occupied8']!=occ or abs(row['entropy4']-entropy)>1e-12:raise ValueError('feature mismatch')
  seen.add(key);counts[key[0]]=counts.get(key[0],0)+1
 if seen!=set(inputs) or counts!={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128}:raise ValueError('task denominator')
 proof={'status':'passed','independent_review':True,'questions':384,'GT_opened':False,'score_values_opened':False,'prediction_records_parsed_for_geometry':True,'features_sha256':sha(feature/'features.jsonl'),'extraction_sha256':sha(feature/'extraction.json'),'extraction_started_sha256':sha(mon/'started.json'),'extraction_terminal_sha256':sha(mon/'terminal.json'),'reviewer_sha256':sha(__file__),'errors':[],'limitations':['Existing DEV prediction records are parsed only for input identity and selection/position; no labels or scores are accessed.','Summary-level E4 source evidence; no new tensor reconstruction.']}
 with OUT.open('x') as fp:json.dump(proof,fp,indent=2);fp.write('\n');fp.flush();os.fsync(fp.fileno())
if __name__=='__main__':main()
