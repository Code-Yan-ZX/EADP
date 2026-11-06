"""Freeze pre-measurement 32-image analysis, no predictions/GT/GPU."""
import collections,hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent
EXPECTED='94b8a159f4a787b541b9f41b27a62921e3a8a6a0b44ff648d8d35d46d5725f7f'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 out=D/'analysis_plan_freeze001.json'
 if out.exists():raise ValueError('already frozen')
 pilot=D/'pilot_plan001.json';doc=D/'ANALYSIS_PLAN.md';before={str(x.resolve()):sha(x) for x in (pilot,doc,Path(__file__))}
 if before[str(pilot.resolve())]!=EXPECTED:raise ValueError('pilot plan changed')
 p=json.loads(pilot.read_text());counts={'DocVQA_VAL':11,'TextVQA_VAL':11,'AI2D_TEST':10};jobs=p['jobs'];measured=[x for x in jobs if x['phase']=='measure'];warm=[x for x in jobs if x['phase']=='warmup']
 if len(jobs)!=1520 or len(measured)!=1280 or len(warm)!=240:raise ValueError('workload mismatch')
 cells=collections.Counter((x['task'],x['group_id'],x['budget'],x['protocol']) for x in measured)
 if any(v!=5 for v in cells.values()) or len(cells)!=32*8:raise ValueError('missing repeats')
 groups={(x['task'],x['group_id']) for x in measured}
 if collections.Counter(t for t,_ in groups)!=counts or len(groups)!=32:raise ValueError('sample mismatch')
 for task,group in groups:
  if {(k,prot) for t,g,k,prot in cells if (t,g)==(task,group)}!={(k,prot) for k in (128,256,512,1024) for prot in ('natural','fixed32')}:raise ValueError('budget/protocol inventory')
 text=doc.read_text()
 for token in ('10000','20261105','K256','31次decode','不把5次重复','不完整时不发布'):
  if token not in text:raise ValueError('missing analysis boundary '+token)
 after={str(x.resolve()):sha(x) for x in (pilot,doc,Path(__file__))}
 if before!=after:raise ValueError('source changed')
 proof={'status':'frozen','analysis_plan_sha256':before[str(doc.resolve())],'pilot_plan_sha256':EXPECTED,'freeze_source_sha256':before[str(Path(__file__).resolve())],'n_images':32,'by_task':counts,'measured_jobs':1280,'warmup_jobs':240,'gpu_authorized':False,'GT_opened':False,'predictions_opened':False,'timing_results_opened':False,'posthoc_exploratory':True,'limitations':['prior DEV and pilot evidence already observed','repeat measurements are not independent images','fixed32 includes 31 decode forwards after prefill','background interference may not be attributable per sample']}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('32-image analysis plan frozen')
if __name__=='__main__':main()
