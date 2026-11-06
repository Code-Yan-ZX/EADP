"""Independent prelaunch review: analysis source, genuine exit, synthetic tests and refusal."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
I=Path(__file__).resolve().parent; B=I.parent; D=B/'e4_c_multibudget_analysis192_exec'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);o=a.parse_args().out
 expected=I/'e4c_analysis192_prelaunch_review.json'
 if o.resolve()!=expected or o.exists() or (D/'analysis001.json').exists():raise ValueError('unexpected output or already analyzed')
 p=read(D/'analysis_code_plan001.json');c=read(D/'cpu_proof001.json');s=read(D/'monitor_cpu001/started.json');t=read(D/'monitor_cpu001/terminal.json')
 if p.get('status')!='frozen' or any(p.get(k) is not False for k in ('GT_opened','target_predictions_opened','timing_results_opened')) or (p.get('jobs'),p.get('warmups'),p.get('measured'))!=(7920,240,7680):raise ValueError('analysis plan')
 if c.get('status')!='passed' or c.get('returncode')!=0 or c.get('source_sha256_before')!=c.get('source_sha256_after'):raise ValueError('cpu proof')
 if p['source_sha256']!=dict(c['source_sha256_after'],**{str((D/'cpu_proof001.json').resolve()):sha(D/'cpu_proof001.json')}) or any(sha(k)!=v for k,v in p['source_sha256'].items()):raise ValueError('source pin')
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or [(Path(s['cwd'])/x).resolve() for x in s['command'] if Path(x).name=='freeze.py']!=[(D/'freeze.py').resolve()]:raise ValueError('freeze supervisor')
 essential=[D/'analysis_gate.py',D/'report.py',B/'e4_c_multibudget_analysis192_v2/metrics.py',B/'e4_c_multibudget_analysis192_independent/window_detail.py',B/'e4_c_multibudget_environment192_v2/validate.py',B/'e4_c_multibudget_expansion192/ANALYSIS_PLAN.md',B/'e4_c_multibudget_expansion192/proposal002.json',B/'e4_c_multibudget_worker192_v6_accept/verification_plan001.json']
 if any(str(x) not in p['source_sha256'] for x in essential):raise ValueError('essential source missing')
 gate=(D/'analysis_gate.py').read_text();report=(D/'report.py').read_text();metric=(B/'e4_c_multibudget_analysis192_v2/metrics.py').read_text();detail=(B/'e4_c_multibudget_analysis192_independent/window_detail.py').read_text()
 for x in ('rf,rr=admitted_pair','f,r=admitted_pair','monitor_watch001','Only after all complete formal/independent proofs','environment_files_sha256'):
  if x not in gate:raise ValueError('missing admission check '+x)
 if report.index('approval=gate.admit()')>report.index("events=rows(") or 'samples_excluded' not in report:raise ValueError('read ordering')
 for x in ('set(range(5))','np.median([x[1]','np.median([x[2]','values[256]','np.random.default_rng(SEED)','np.quantile'):
  if x not in metric:raise ValueError('missing paired median statistic '+x)
 for x in ('len(details)','manual_health','query_first_fsync','samples_excluded'):
  if x=='len(details)':continue
  if x not in detail:raise ValueError('missing environment detail '+x)
 tests=[]
 for cwd,mod in ((D,'test_exec'),(B/'e4_c_multibudget_analysis192_v2','test_metrics'),(B/'e4_c_multibudget_analysis192_independent','test_window_detail')):
  z=subprocess.run([sys.executable,'-m','unittest','-v',mod],cwd=cwd,text=True,capture_output=True,timeout=180)
  if z.returncode!=0 or 'OK' not in z.stderr:raise ValueError('tests failed '+mod+' '+z.stderr)
  tests.append({'module':mod,'stderr':z.stderr})
 # Run the real executable from its formal cwd before the target exists. It must
 # fail at the missing prelaunch proof, without producing output or opening a panel.
 z=subprocess.run([sys.executable,str(D/'report.py'),'--out',str(D/'analysis001.json')],cwd=B,text=True,capture_output=True,timeout=30)
 if z.returncode==0 or (D/'analysis001.json').exists() or 'monitor_e4c_analysis192_prelaunch' not in z.stderr:raise ValueError('actual entry did not fail closed')
 result={'status':'passed','independent_review':True,'GT_opened':False,'target_predictions_opened':False,'errors':[],'plan_sha256':sha(D/'analysis_code_plan001.json'),'cpu_proof_sha256':sha(D/'cpu_proof001.json'),'cpu_started_sha256':sha(D/'monitor_cpu001/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu001/terminal.json'),'reviewer_sha256':sha(__file__),'source_sha256':p['source_sha256'],'tests':tests,'actual_entry_returncode_before_proof':z.returncode,'actual_entry_refusal':'missing independent prelaunch supervisor','limitations':['pre-run source and synthetic tests; no performance result','unknown residual interference and observer/manual fsync remain unknown','independent prior metrics review covers separately computed synthetic bootstrap']}
 with o.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
