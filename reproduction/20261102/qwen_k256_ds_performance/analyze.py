import json,statistics as st,hashlib
from pathlib import Path
import numpy as np
R=Path('/media/disk2/YZX/research/EADP_v2');Q=R/'reproduction/20261028/qwen';P=R/'reproduction/20261102/qwen_k256_ds_performance'
panel=json.loads((Q/'fixed16_k256_ds_toggle/plan.json').read_text())['panel']
rows=[json.loads(x) for x in (P/'observations.jsonl').read_text().splitlines()]; assert len(rows)==64
complete=json.loads((P/'complete.json').read_text());assert complete['n_measured']==64
assert hashlib.sha256((P/'observations.jsonl').read_bytes()).hexdigest()==complete['observations_sha256']
out={'n':len(rows),'modes':{},'natural_token_ids_equal_prior':0,'all_selected_ids_equal_prior':64,'limitations':['one measured call per arm per sample, 16 fixed DocVQA questions','same single model, paired and alternate order; not all tasks or devices','natural output length differs; fixed32 changes generation termination','peak is PyTorch allocated, not nvidia-smi total or all external CUDA allocations','prefill GPU event includes on-side gather and injection plus first language-model forward','wall includes image reading, preprocessing, visual encoder, pruning, generation, token decode and lightweight instrumentation; excludes model load and scoring']}
for i,row in enumerate(panel):
 for arm in ('off','on'):
  sample=[x for x in rows if x['mode']=='natural' and x['arm']==arm and x['position']==row['position']];assert len(sample)==1
  with np.load(Q/'fixed16_k256_ds_toggle/execution'/f'{i:02d}_ds_{arm}.npz') as z:
   assert sample[0]['output_ids']==z['output_ids'][0].tolist();out['natural_token_ids_equal_prior']+=1
for mode in ('natural','fixed32'):
 arms={a:[r for r in rows if r['mode']==mode and r['arm']==a] for a in ('off','on')}
 assert all(len(x)==16 for x in arms.values())
 pairs=[]
 for p in panel:
  off=next(r for r in arms['off'] if r['position']==p['position']);on=next(r for r in arms['on'] if r['position']==p['position'])
  assert off['ordered_selection_sha256']==on['ordered_selection_sha256']
  if mode=='fixed32':assert on['output_tokens']==off['output_tokens']==32
  pairs.append((off,on))
 def stats(x):return {'mean':st.mean(x),'median':st.median(x),'min':min(x),'max':max(x)}
 def field(k,divisor=1):
  return {a:stats([r[k]/divisor for r in v]) for a,v in arms.items()}|{'on_minus_off_paired':stats([(b[k]-a[k])/divisor for a,b in pairs]),'relative_mean_percent':100*(st.mean([b[k] for a,b in pairs])/st.mean([a[k] for a,b in pairs])-1)}
 out['modes'][mode]={'n_pairs':16,'seconds':field('seconds'),'prefill_gpu_ms':field('prefill_gpu_ms'),'peak_allocated_MiB':field('peak_allocated_bytes',1024**2),'output_tokens':field('output_tokens'),'output_length_changed_pairs':sum(a['output_tokens']!=b['output_tokens'] for a,b in pairs),'no_eos':{arm:sum(not any(t in (151643,151645) for t in r['output_ids']) for r in v) for arm,v in arms.items()},'hit_2048':{arm:sum(r['output_tokens']==2048 for r in v) for arm,v in arms.items()},'last_token_eos':{arm:sum(r['output_ids'][-1] in (151643,151645) for r in v) for arm,v in arms.items()},'response_changed_pairs':sum(a['response']!=b['response'] for a,b in pairs)}
(P/'summary.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(json.dumps(out,indent=2,ensure_ascii=False))
