"""Independent prelaunch source audit; no new panel predictions or GT."""
import argparse,importlib.util,json,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;S=B/'e6/pixelprune_t020_dev384_analysis';R=B/'e1_independent_review'
sys.path.insert(0,str(S));import admission
def read(path):return json.loads(Path(path).read_text())
def terminal(mon):
 s,t=read(mon/'started.json'),read(mon/'terminal.json')
 assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid']
 return s
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==admission.PRE and Path(admission.__file__).resolve()==S/'admission.py'
 admission.supervisor(admission.CM,S/'cpu_check.py',admission.CPU.parent)
 mon=R/'monitor_pixel_t020_score_cpu_review';s=terminal(mon);assert (Path(s['cwd'])/s['command'][1]).resolve()==S/'cpu_check.py'
 for path in (admission.CM/'stderr.log',mon/'stderr.log'):
  log=path.read_text();assert 'Ran 8 tests' in log and log.rstrip().endswith('OK') and 'test_actual_main_full_denominator_and_gate' in log
 actual=admission.source_hashes();c=read(admission.CPU);ind=read(R/'pixel_t020_score_cpu_review/verification.json')
 for x in (c,ind):assert x['status']=='passed' and x['GT_opened'] is False and x['real_predictions_opened'] is False and x['source_sha256_before']==x['source_sha256_after']==actual
 assert c['test_stderr_sha256']==admission.sha(admission.CM/'stderr.log') and ind['test_stderr_sha256']==admission.sha(mon/'stderr.log')
 plan=read(admission.PLAN);assert admission.sha(admission.PLAN)=='5c83009aea551eb23b481ad939938b01a5476026dccb3742744ae0d61a9f1987'
 assert plan['status']=='frozen_before_panel_results' and plan['arms']==['pixel_t020','pixel_t0','e4_k256_d1','e4_k1024_d1']
 assert plan['contrasts']=={'t020_minus_t0':['pixel_t020','pixel_t0'],'t020_minus_e4_k256':['pixel_t020','e4_k256_d1'],'t020_minus_e4_k1024':['pixel_t020','e4_k1024_d1']}
 assert plan['bootstrap']=={'unit':'image group','paired':True,'replicates':10000,'seed':20261104,'confidence':0.95,'multiplicity_adjustment':'none'} and plan['threshold']==0.2
 # The prior gate reads proof metadata and hashes prior files; it never parses the new panel.
 prior=admission.legacy().admission();assert prior['pixel_formal_sha256'] and prior['pixel_review_sha256']
 smoke=read(B/'e1_independent_review/pixel_t020_result_review.json');assert smoke['status']=='passed' and smoke['independent_review'] is True and smoke['images']==4 and smoke['GT_opened'] is False
 v=read(B/'e6/pixelprune_t020_accept/verification_plan001.json');assert v['expected_questions']==384 and v['panel_plan_sha256']==admission.sha(B/'e6/pixelprune_t020_dev384/plan001.json')
 for path,h in v['source_sha256'].items():assert admission.sha(path)==h
 out={'status':'passed','independent_review':True,'scope':'pixel_t020_dev384_score_prelaunch','GT_opened':False,'predictions_opened':False,'errors':[],'source_sha256':actual,'plan_sha256':admission.sha(admission.PLAN),'cpu_proof_sha256':admission.sha(admission.CPU),'cpu_started_sha256':admission.sha(admission.CM/'started.json'),'cpu_terminal_sha256':admission.sha(admission.CM/'terminal.json'),'reviewer_sha256':admission.sha(__file__),'limitations':['post-hoc DEV only; pairwise intervals unadjusted','dynamic K is not equal compute and no latency claim','prior output files hashed for source verification but new panel predictions not parsed in prelaunch','new panel complete formal and independent acceptance still required by score admission']}
 with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush()
 print('passed')
if __name__=='__main__':main()
