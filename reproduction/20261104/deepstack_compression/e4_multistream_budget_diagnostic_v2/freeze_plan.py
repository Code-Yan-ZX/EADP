"""Freeze an identity/analysis protocol before feature or score inspection."""
import hashlib,json,os,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 dst=D/'plan_freeze001.json'
 if dst.exists():raise ValueError('already frozen')
 manifest=B/'e5/selector_real32/sample_manifest_001.json';data=json.loads(manifest.read_text());pilot=data['pilot']
 from collections import Counter
 if len(pilot)!=32 or Counter(x['task'] for x in pilot)!={'DocVQA_VAL':11,'TextVQA_VAL':11,'AI2D_TEST':10} or len({(x['task'],str(x['qid'])) for x in pilot})!=32 or len({(x['task'],x['group_id']) for x in pilot})!=32:raise ValueError('pilot identity inventory')
 selector=B/'e1_independent_review/selector_real32_run001_review.json';sr=json.loads(selector.read_text());score=B/'e4_analysis/independent_score_review/verification001.json';er=json.loads(score.read_text())
 if sr.get('status')!='passed' or sr.get('independent_review') is not True or sr.get('images')!=32 or sr.get('GT_opened') is not False or sr.get('errors')!=[] or er.get('status')!='passed' or er.get('questions')!=384 or er.get('errors')!=[]:raise ValueError('upstream metadata not accepted')
 arrays=sr.get('array_sha256',{});captures={k:v for k,v in arrays.items() if k.endswith('/capture.pt')}
 if len(captures)!=32 or any(not Path(k).is_file() for k in captures):raise ValueError('capture inventory')
 paths=[D/'PLAN.md',D/'freeze_plan.py',manifest,selector,score]
 result={'status':'frozen','plan_sha256':sha(D/'PLAN.md'),'source_sha256':{str(p.resolve()):sha(p) for p in paths},'capture_inventory_sha256':captures,'upstream_score_file_sha256':er['score_sha256'],'task_counts':{'DocVQA_VAL':11,'TextVQA_VAL':11,'AI2D_TEST':10},'images':32,'feature_or_score_values_opened':False,'GT_opened':False,'predictions_opened':False,'gpu_used':False,'created_unix':time.time(),'scope':'posthoc protocol freeze only; no extraction or execution authorization'}
 with dst.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
