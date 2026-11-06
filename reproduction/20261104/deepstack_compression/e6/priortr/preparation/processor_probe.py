"""CPU compare frozen DEV inputs with E1 exact processor tensor hashes."""
import hashlib,json,os
from pathlib import Path
import torch
from transformers import AutoProcessor
from admission import BASE,E1,admit,jsonl,sha
from preprocess import prepare

def meta(t):
 z=t.detach().contiguous().cpu()
 return {'dtype':str(z.dtype),'shape':list(z.shape),'sha256':hashlib.sha256(z.view(torch.uint8).numpy().tobytes()).hexdigest()}

def main():
 here=Path(__file__).resolve().parent;manifest=here/'sample_manifest.json';m=json.loads(manifest.read_text());proof,rows=admit()
 if m['rows']!=rows or m['e1_completion_sha256']!=sha(E1/'e1full001_completion.json'):raise ValueError('frozen sample identity changed')
 recipe=json.loads((BASE.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json').read_text())['recipe']
 processor=AutoProcessor.from_pretrained(recipe['model_path'],local_files_only=True,trust_remote_code=False)
 ref={}
 for r in jsonl(proof['files']['journal']['path']):
  key=(r['task'],str(r['qid']))
  if r['arm']=='P3-D0' and r['forced'] is False:ref[key]=r['processor']
 checks=[]
 for row in rows:
  actual=prepare(processor,row['message']);actual=actual.to(torch.bfloat16);seen={k:meta(actual[k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
  expected=ref[(row['task'],str(row['qid']))];matched={k:seen[k]==expected[k] for k in seen}
  checks.append({'task':row['task'],'qid':row['qid'],'matches':matched,'actual':seen})
 result={'status':'passed' if all(all(x['matches'].values()) for x in checks) else 'failed','n':len(checks),'checks':checks,'manifest_sha256':sha(manifest),'preprocess_sha256':sha(here/'preprocess.py'),'verifier_sha256':sha(__file__),'device':'cpu','cuda_initialized':torch.cuda.is_initialized()}
 out=here/'processor_check002.json'
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':result['status'],'n':len(checks),'cuda_initialized':result['cuda_initialized']}))
 if result['status']!='passed' or result['cuda_initialized']:raise SystemExit(1)
if __name__=='__main__':main()
