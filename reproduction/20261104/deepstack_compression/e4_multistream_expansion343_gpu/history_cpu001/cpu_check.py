"""Freeze v2 CPU source/test evidence; not an execution authorization."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;P=B/'e4_multistream_expansion352';V=B/'e4_multistream_expansion352_gpu_v4';F=B/'e4_multistream_expansion343'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 out=D/'cpu_check001.json'
 if out.exists():raise ValueError('already frozen')
 prior=json.loads((V/'cpu_check002.json').read_text());p=json.loads((P/'selection001.json').read_text())
 if prior.get('status')!='passed' or any(sha(k)!=v for k,v in prior['source_sha256_after'].items()) or p.get('status')!='frozen' or len(p['pilot'])!=9:raise ValueError('prior sources')
 paths={Path(k) for k in prior['source_sha256_after']}|set(D.glob('*.py'))|{D/'execution_plan001.json',D/'formal_plan001.json',V/'schema_fixture001.json',V/'monitor_freeze_schema001/started.json',V/'monitor_freeze_schema001/terminal.json',V/'cpu_check002.json',F/'PLAN.md',F/'plan_freeze001.json',F/'monitor_freeze_plan001/started.json',F/'monitor_freeze_plan001/terminal.json'};before={str(x.resolve()):sha(x) for x in sorted(paths)}
 tests=[]
 for directory,pattern in ((P,'test_norm_contract.py'),(P,'test_pilot_contract.py'),(P,'test_selection.py'),(D,'test_*.py')):
  t=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(directory),'-p',pattern,'-v'],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
  tests.append(t.stdout+t.stderr)
  if t.returncode:raise ValueError('CPU tests failed '+t.stderr[-2000:])
 after={str(x.resolve()):sha(x) for x in sorted(paths)}
 if before!=after:raise ValueError('source changed')
 value={'status':'passed','source_sha256_before':before,'source_sha256_after':after,'tests':tests,'GPU_used':False,'model_loaded':False,'GT_opened':False,'predictions_opened':True,'prediction_scope':'accepted E1/E4 mixed journals were parsed earlier only for frozen identity schema; fixture excludes answers and outputs','feature_values_opened':False,'score_values_opened':False,'scope':'343-image CPU source freeze; real container and GPU not yet approved','errors':[]}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
