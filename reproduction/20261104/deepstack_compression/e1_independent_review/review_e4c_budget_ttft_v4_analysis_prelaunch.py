"""No timing-value read: independent source and synthetic analysis review."""
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path
B=Path(__file__).resolve().parents[1];D=B/'e4_c_budget_ttft_analysis_v4';I=B/'e1_independent_review';OUT=I/'e4c_budget_ttft_v4_analysis_prelaunch_review.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def run():
 mf=D/'source_manifest001.json';m=read(mf);start=D/'monitor_cpu002/started.json';end=D/'monitor_cpu002/terminal.json';s=read(start);t=read(end)
 if sha(mf)!='1b947abb3ce3b78049ae72da2ddf3413ad577f3e8ea7e5997bf73936b7834020' or m.get('cpu_started_sha256')!=sha(start) or m.get('cpu_terminal_sha256')!=sha(end) or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid'):raise ValueError('CPU validation evidence')
 if s['command'][1:]!=['-m','unittest','discover','-s','e4_c_budget_ttft_analysis_v4','-p','test_*.py','-v'] or 'Ran 4 tests' not in (D/'monitor_cpu002/stderr.log').read_text():raise ValueError('CPU validation command')
 for path,value in m['source_sha256'].items():
  if sha(path)!=value:raise ValueError('source changed')
 plan=read(D/'analysis_plan001.json');supp=read(D/'analysis_plan001_supplement.json')
 if sha(D/'analysis_plan001.json')!='70241e4486258a067d971ce0c016649b4cf6fafc2c7eb7d32d68c9cd643268dd' or supp['original_plan_sha256']!=sha(D/'analysis_plan001.json') or supp['background_overlap_sha256']!=sha(B/'e4_c_budget_ttft_accept_v3/background_overlap001.json') or plan.get('created_before_GPU') is not True:raise ValueError('plans')
 spec=importlib.util.spec_from_file_location('_independent_analysis',D/'analyze.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
 jobs=[];events=[];health=[];background=[]
 for phase,count in [('warmup',1),('measure',3)]:
  for rep in range(count):
   for task in ('D','T','A'):
    for k in (128,256,512,1024):
     j={'task':task,'qid':task,'group_id':task,'phase':phase,'replicate':rep,'budget':k,'order':len(jobs)};jobs.append(j)
     val={128:100,256:200,512:400,1024:800}[k]+rep*10
     b={'t0_before_image_read_ns':1000,'first_token_selected_ns':1000+val};events.extend([{'event':'started','job':j},{'event':'raw_generation','job':j},{'event':'completed','job':j,'payload':{'boundaries':b,'output_ids':[2]}}]);health.append({'order':j['order'],'ttft_health_overlap_indices':[]});background.append({'order':j['order'],'ttft_ps_query_overlap':[],'ttft_ps_first_fsync_overlap':[]})
 x=a.summarize(jobs,events,health,background)
 cell=next(c for c in x['cells'] if c['task']=='D' and c['budget']==128)
 if len(x['samples'])!=48 or len(x['cells'])!=12 or cell['paired_ratios_to_k256']!=[100/200,110/210,120/220] or cell['median_ttft_ms']!=110/1e6:raise ValueError('synthetic paired statistics')
 # Prior admission only checks proof metadata/hashes; it does not decode journal values.
 a.admit()
 return {'status':'passed','independent_review':True,'errors':[],'GT_opened':False,'scoring_performed':False,'source_manifest_sha256':sha(mf),'source_sha256':m['source_sha256'],'cpu_started_sha256':sha(start),'cpu_terminal_sha256':sha(end),'reviewer_sha256':sha(__file__),'limitations':['Exploratory three-anchor descriptive statistics; no population inference.','Background supplement was frozen after overlap counts were observed, before interpreting latency.','Source prelaunch review did not decode actual timing values.']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=OUT:raise ValueError('output')
 v=run()
 with a.out.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
