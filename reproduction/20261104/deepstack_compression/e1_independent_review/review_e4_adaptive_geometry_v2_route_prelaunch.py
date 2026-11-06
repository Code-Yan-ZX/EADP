import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');D=B/'e4_adaptive_geometry_v2';R=B/'e1_independent_review'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(mon,contains):
 s=read(mon/'started.json');t=read(mon/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and t['worker_pid']==s['worker_pid'] and contains in s['command']
 return {'started_sha256':sha(mon/'started.json'),'terminal_sha256':sha(mon/'terminal.json')}
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',required=True);a=a.parse_args();assert not (D/'analysis001').exists()
 route=read(D/'route_freeze001.json');freeze=read(D/'analysis_freeze005.json')
 assert route['status']=='frozen_before_feature_values' and route['feature_values_opened'] is False and route['score_values_opened'] is False and route['GT_opened'] is False
 assert freeze['status']=='gate_fixed_after_feature_reviewer_before_score_values' and freeze['feature_file_already_generated'] is True and freeze['feature_values_inspected_by_independent_reviewer'] is True and freeze['score_values_parsed_for_this_analysis'] is False and freeze['GT_opened'] is False
 for inv in (route['source_sha256'],freeze['source_sha256']):
  for path,h in inv.items():assert sha(path)==h,path
 mons={'route_cpu':terminal(D/'monitor_route_cpu001','test_route.py'),'analysis_cpu':terminal(D/'monitor_analysis_cpu005','test_analysis.py'),'extract':terminal(D/'monitor_extract001','e4_adaptive_geometry_v2/extract.py'),'feature_review':terminal(R/'monitor_e4_adaptive_geometry_v2_features','e1_independent_review/review_e4_adaptive_geometry_v2_features.py')}
 proof=read(R/'e4_adaptive_geometry_v2_features_review.json');assert proof['status']=='passed' and proof['independent_review'] is True and proof['questions']==384 and proof['GT_opened'] is False and proof['score_values_opened'] is False and proof['errors']==[]
 assert proof['reviewer_sha256']==sha(R/'review_e4_adaptive_geometry_v2_features.py') and proof['extraction_started_sha256']==sha(D/'monitor_extract001/started.json') and proof['extraction_terminal_sha256']==sha(D/'monitor_extract001/terminal.json') and proof['features_sha256']==sha(D/'features001/features.jsonl') and proof['extraction_sha256']==sha(D/'features001/extraction.json')
 for test,count in [('test_route','2'),('test_analysis','3')]:
  r=subprocess.run([sys.executable,'-m','unittest','-v',test],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=60);assert r.returncode==0 and 'Ran '+count+' tests' in r.stderr,(test,r.stderr)
 out={'status':'passed','independent_review':True,'GT_opened':False,'score_values_opened':False,'scope':'posthoc DEV-only source and route prelaunch; actual analysis remains separately gated','route_freeze_sha256':sha(D/'route_freeze001.json'),'analysis_freeze_sha256':sha(D/'analysis_freeze005.json'),'feature_review_sha256':sha(R/'e4_adaptive_geometry_v2_features_review.json'),'reviewer_sha256':sha(__file__),'supervision':mons,'independent_cpu_tests':5,'limitations':['The feature file existed and was independently inspected before analysis-gate fixes; allocation rules were frozen earlier.','Three geometry and three random rules must all be reported; batch ranking is exploratory and is not an online policy.','Mean token budget is not equal compute or latency; potential two-pass generation costs are excluded.','This prelaunch does not parse the existing score journal; the separate runtime gate must validate it before analysis.'],'errors':[]}
 with open(a.out,'x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
