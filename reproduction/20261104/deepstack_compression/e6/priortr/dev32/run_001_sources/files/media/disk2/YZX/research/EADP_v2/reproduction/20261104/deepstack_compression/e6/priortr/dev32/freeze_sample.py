"""Freeze a label-free PriorTR 32-question DEV diagnostic design."""
import hashlib,json,os,time
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def main():
 proof=BASE/'e1/e1full001_completion.json';e1=json.loads(proof.read_text());inp=e1['files']['inputs'];panel=e1['files']['panel']
 if sha(inp['path'])!=inp['sha256'] or sha(panel['path'])!=panel['sha256'] or e1['GT_opened'] is not False or e1['scoring_performed'] is not False:raise ValueError('E1 input proof')
 def lines(p):
  with open(p,encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
 rows=lines(inp['path']);dev={(r['task'],str(r['qid'])):r for r in lines(panel['path']) if r.get('split')=='DEV'}
 if len(rows)!=384 or len(dev)!=384 or len({(r['task'],str(r['qid'])) for r in rows})!=384:raise ValueError('DEV inventory')
 for r in rows:
  if (r['task'],str(r['qid'])) not in dev or r['group_id']!=dev[(r['task'],str(r['qid']))]['group_id'] or any(k in r for k in ('answer','answers','prediction','ground_truth')):raise ValueError('DEV join or label')
 counts={'DocVQA_VAL':11,'TextVQA_VAL':11,'AI2D_TEST':10};selected=[]
 def rank(r):return hashlib.sha256(('20261105|priortr-dev32|'+r['task']+'|'+str(r['qid'])+'|'+r['group_id']).encode()).hexdigest()
 for task,n in counts.items():
  selected.extend(sorted((r for r in rows if r['task']==task),key=lambda r:(rank(r),str(r['qid'])))[:n])
 keys=('task','qid','group_id','image_path','image_sha256','message','message_sha256')
 sample=[{k:r[k] for k in keys} for r in selected]
 if len(sample)!=32 or Counter(r['task'] for r in sample)!=counts or len({r['group_id'] for r in sample})!=32:raise ValueError('sample inventory')
 plan={'stage':'prospective_diagnostic_only','created_unix':time.time(),'selection_after_e4_dev_scores':True,'selection':'SHA256 20261105|priortr-dev32|task|qid|group_id, ascending; 11/11/10 by task; input manifest only','E1_proof_sha256':sha(proof),'input_sha256':inp['sha256'],'panel_sha256':panel['sha256'],'freeze_source_sha256':sha(__file__),'sample':sample,'physical_generations':96,'arms':['stock_native','author_default','priortr_k256'],'recipe':'identical E1 image, prompt, generation parameters and weights; isolated fixed PriorTR environment, SDPA backend; default uses original 0.1111 ratio and layer3; k256 explicitly keep_tokens=256 and layer3','diagnostic':'Per-question processor hashes, full image tower and main/DS, selected ids, P3 gathering, layer lengths and actual natural generations; stock-native comparisons report first-logit and output differences to E1/E4 without assuming cross-backend identity; timing only rough wall clock not a latency benchmark. First fixed question per task retains full arrays, others retain checks and hashes. Record all outputs without scores; 32-question subset cannot select a winner.','safety':'Before execution require CPU contracts, independent source/protocol review, actual same-environment no-GPU preflight, independent accepted PriorTR single-question evidence, exclusive GPU, bounded container lifecycle and no labels; freeze source and full evidence.','GT_opened':False,'prediction_opened':False,'scoring_performed':False}
 out=HERE/'sample_001.json'
 with out.open('x') as f:json.dump(plan,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'frozen','questions':len(sample),'planned_generations':96}))
if __name__=='__main__':main()
