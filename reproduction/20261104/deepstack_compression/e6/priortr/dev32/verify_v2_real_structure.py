"""No-GT regression of verifier v2 conversion on one saved multi-token anchor."""
import hashlib,importlib.util,json,os,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;PREP=HERE.parent/'preparation';sys.path.insert(0,str(PREP))
spec=importlib.util.spec_from_file_location('_priortr_v2_smoke_verify',PREP/'smoke_verify.py');smoke=importlib.util.module_from_spec(spec);spec.loader.exec_module(smoke)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 path=HERE/'run_001/data/records.jsonl';chosen=None
 with path.open() as f:
  for line in f:
   r=json.loads(line)
   if r['arrays']['kind']=='full_arrays' and len(r['output_ids'])>1:chosen=r;break
 if chosen is None:raise ValueError('no multi-token anchor')
 r=chosen;t=r['trace'];converted={'arrays':r['arrays'],'case':{'stock_native':'stock','author_default':'author_default','priortr_k256':'k256'}[r['arm']],'calls':t['calls'],'layers':t['layers'],'injections':t['injections'],'prunes':t['prunes'],'ids':r['output_ids'],'forced':False,'eos':r['eos'],'eos_ids':r['eos_ids'],'truncated':r['truncated'],'rope_delta':r['rope_delta']}
 with np.load(r['arrays']['path'],allow_pickle=False) as arrays:smoke.validate_record(converted,arrays,r['processor'],36)
 result={'status':'passed','scope':'one already-generated multi-token diagnostic anchor; no GT/scoring','task':r['task'],'qid':str(r['qid']),'arm':r['arm'],'output_token_count':len(r['output_ids']),'records_sha256':sha(path),'anchor_sha256':sha(r['arrays']['path']),'original_verifier_sha256':sha(HERE/'panel_verify.py'),'v2_verifier_sha256':sha(HERE/'panel_verify_v2.py'),'regression_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False}
 out=HERE/'verify_v2_real_structure001.json'
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed','tokens':len(r['output_ids'])}))
if __name__=='__main__':main()
