"""Freeze scoring sources/contract before reading target predictions or labels."""
import hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def main():
 files=[D/x for x in ('PLAN.md','paired.py','score.py','admission.py','test_paired.py','test_score.py','test_gate_cli.py','test_main_synthetic.py','cpu_check.py','freeze_score.py')]
 frozen=json.loads((B/'e1/e1full001_score/frozen.json').read_text());source={str(p):sha(p) for p in files}
 for p,h in frozen['code_sha256'].items():
  if sha(p)!=h:raise ValueError('accepted scorer source changed')
  source[p]=h
 for p in (B/'e1/e1full001_score/frozen.json',B/'e6/pixelprune_t020_dev384/plan001.json'):
  source[str(p)]=sha(p)
 plan={'status':'frozen','target_predictions_opened':False,'GT_opened':False,'source_sha256':source,
       'questions':384,'questions_per_task':128,'seeds':[20261104,20261105,20261106],
       'bootstrap_draws':10000,'bootstrap_seed':20261104,'runtime_versions':frozen['runtime_versions'],
       'limits':['post-DEV exploratory','equal K is not equal compute or equal image resolution','no selection or sample replacement']}
 with (D/'score_plan001.json').open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
