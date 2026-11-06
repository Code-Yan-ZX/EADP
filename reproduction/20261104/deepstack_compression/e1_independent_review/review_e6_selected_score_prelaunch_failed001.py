"""Independent no-data Div/CD DEV scorer source review."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;S=ROOT/'e6/selected_analysis';sys.path.insert(0,str(S));import admission

def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert a.out.resolve()==admission.PRE
 cs,ct=admission.supervisor(admission.CMON,S/'cpu_check.py');c=read(admission.CPU);actual=admission.source_hashes();assert c['status']=='passed' and c['GT_opened'] is False and c['real_predictions_opened'] is False and c['source_sha256_before']==c['source_sha256_after']==actual
 log=(admission.CPU.parent/'stderr.log').read_text();assert 'Ran 15 tests' in log and log.rstrip().endswith('OK') and c['stderr_sha256']==admission.sha(admission.CPU.parent/'stderr.log')
 frozen=admission.e1_identity();assert admission.sha(S/'plan.json')==admission.PLAN_SHA
 result={'status':'passed','independent_review':True,'scope':'selected_score_prelaunch','GT_opened':False,'predictions_opened':False,'errors':[],'source_sha256':actual,'plan_sha256':admission.PLAN_SHA,'cpu_proof_sha256':admission.sha(admission.CPU),'cpu_started_sha256':admission.sha(admission.CMON/'started.json'),'cpu_terminal_sha256':admission.sha(admission.CMON/'terminal.json'),'e1_score_frozen_sha256':admission.sha(admission.FROZEN),'e1_score_review_sha256':admission.sha(admission.E1REVIEW),'reviewer_sha256':admission.sha(__file__),'limits':['DEV exploratory unadjusted comparisons; not a compute-matched benchmark.','Formal and independent full-panel completion remain separate gates before any predictions or labels are read.']}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','sources':len(actual)}))
if __name__=='__main__':main()
