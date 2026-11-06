"""192-image analysis; proof gates finish before loading timing artifacts."""
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;ENV=B/'e4_c_multibudget_environment192_v2'
PIN={'gate': 'd62358ae199fafb409244daa896cd0d98cd4f7fef4ad5c17657025d085eaa088', 'metrics': '75ad63b8aa1cf129848d12192d9cdcad0bdc27cbcce030b023a339e58037310e', 'detail': '46ee3805e63400c257be1d37c58b052b47ff9bb526908abce32944236ac0398e', 'observer': '1e3c088969380282649c9c0f88efdfd61855623b0928b93a2c0a71af9c286e13'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(name,path,digest):
 path=Path(path).resolve()
 if sha(path)!=digest or name in sys.modules:raise ValueError('unreviewed/preloaded analysis dependency')
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def health(path,namespace):
 out=[]
 for row in rows(path):
  v=row.get('health_snapshot_window_monotonic_ns')
  if row.get('time_namespace')!=namespace or not isinstance(v,list) or len(v)!=2 or any(type(x) is not int for x in v) or v[0]>v[1]:raise ValueError('bad health window')
  out.append(v)
 if not out:raise ValueError('missing health window')
 return out
def calculate(jobs,events,windows,namespace,metrics,detail):
 summary=metrics.summarize(jobs,events);observation=detail.report(jobs,events,windows,namespace)
 return {'summary':summary,'jobs_detail':observation['jobs_detail'],'health_overlap':observation['health_overlap'],'hourly_health_coverage':observation['hourly_health_coverage'],'manual_check_evidence':observation['manual_check_evidence'],'unknown':observation['unknown'],'samples_excluded':observation['samples_excluded']}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
 if a.out.resolve()!=D/'analysis001.json' or a.out.exists():raise ValueError('unexpected analysis output')
 gate=load('_e4c_analysis192_gate',D/'analysis_gate.py',PIN['gate']);approval=gate.admit()
 # The above checks both formal/independent result chains and actual exits.
 metrics=load('_e4c_analysis192_metrics',B/'e4_c_multibudget_analysis192_v2/metrics.py',PIN['metrics']);detail=load('_e4c_analysis192_windows',B/'e4_c_multibudget_analysis192_independent/window_detail.py',PIN['detail']);validator=load('_e4c_analysis192_observer',ENV/'validate.py',PIN['observer'])
 import numpy as np
 if np.__version__!='2.2.6':raise ValueError('NumPy version mismatch')
 jobs=read(B/'e4_c_multibudget_expansion192/proposal002.json')['jobs'];events=rows(B/'e4_c_multibudget_worker192_v6/run_001/journal.jsonl');launch=read(B/'e4_c_multibudget_worker192_v6/run_001/launch.json');namespace=launch['time_namespace']
 obs=rows(ENV/'samples001.jsonl');w=validator.validate(obs,namespace);manual=[]
 for x in rows(ENV/'manual_checks001.jsonl'):
  l=x.get('start_monotonic_ns');r=x.get('end_monotonic_ns')
  if x.get('event')!='manual_health_check' or x.get('time_namespace')!=namespace or type(l) is not int or type(r) is not int or l>r or not x.get('start_utc') or not x.get('end_utc') or not isinstance(x.get('details'),dict):raise ValueError('invalid manual health observation')
  manual.append([l,r])
 windows={'model_health':health(B/'e4_c_multibudget_worker192_v6/monitor_run001/health.jsonl',namespace),'query':[[a,b] for a,b,c in w],'query_first_fsync':[[a,c] for a,b,c in w],'watcher_health':health(ENV/'monitor_watch001/health.jsonl',namespace),'manual_health':manual}
 result=calculate(jobs,events,windows,namespace,metrics,detail);result.update(status='completed',approval=approval,GT_opened=False,scoring_performed=False,predictions_opened=True,scope='posthoc 192-image expansion following observed 32-image pilot; same-image comparisons; complete panel only')
 with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
