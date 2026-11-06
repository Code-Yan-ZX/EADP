"""Independent no-new-prediction/no-label Pixel DEV scoring source audit."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;S=ROOT/'e6/pixelprune_dev384_analysis';sys.path.insert(0,str(S));import admission

def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 assert a.out.resolve()==admission.PRE
 admission.supervisor(admission.CM,S/'cpu_check.py',admission.CPU.parent)
 m=ROOT/'e1_independent_review/monitor_pixel_score_cpu_review';started=read(m/'started.json');terminal=read(m/'terminal.json')
 assert terminal['reason']=='completed' and terminal['worker_returncode']==0 and terminal['remaining_process_group']==[] and started['worker_pid']==terminal['worker_pid']
 assert 'Ran 6 tests' in (m/'stderr.log').read_text() and (m/'stderr.log').read_text().rstrip().endswith('OK')
 c=read(admission.CPU);actual=admission.source_hashes();assert c['status']=='passed' and c['GT_opened'] is False and c['real_predictions_opened'] is False and c['source_sha256_before']==c['source_sha256_after']==actual
 assert c['test_stderr_sha256']==admission.sha(admission.CM/'stderr.log') and 'Ran 6 tests' in (admission.CM/'stderr.log').read_text()
 admission.e1_identity();plan=read(admission.PLAN);assert plan['denominator']==384 and plan['bootstrap']=={'unit':'image_group','replicates':10000,'seed':20261104,'interval':'95% percentile','paired':True}
 formal=admission.A/'verification001.json';review=admission.R/'pixel_dev384_result_review.json'
 admission.supervisor(admission.A/'monitor_verify001',admission.A/'verify.py',formal);admission.supervisor(admission.R/'monitor_pixel_dev384_result',admission.R/'review_pixel_dev384_result.py',review)
 f=read(formal);r=read(review);assert f['status']==r['status']=='passed' and f['questions']==r['questions']==384 and f['GT_opened'] is r['GT_opened'] is False and r['independent_review'] is True and r['formal_sha256']==admission.sha(formal)
 assert len(r['per_question_files_sha256'])==768 and all(f['files_sha256'].get(k)==v for k,v in r['per_question_files_sha256'].items())
 out={'status':'passed','independent_review':True,'scope':'pixel_dev384_score_prelaunch','GT_opened':False,'predictions_opened':False,'errors':[],'source_sha256':actual,'plan_sha256':admission.sha(admission.PLAN),'cpu_proof_sha256':admission.sha(admission.CPU),'cpu_started_sha256':admission.sha(admission.CM/'started.json'),'cpu_terminal_sha256':admission.sha(admission.CM/'terminal.json'),'e1_score_frozen_sha256':admission.sha(admission.FROZEN),'e1_score_review_sha256':admission.sha(admission.R/'e1full001_score_review.json'),'reviewer_sha256':admission.sha(__file__),'limitations':['DEV-only exploratory, task-wise and uncorrected','dynamic K versus fixed K changes compute; no equal-compute or latency claim','references verified separately before any scoring']}
 with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush()
 print('passed')
if __name__=='__main__':main()
