"""Independent source review of post-hoc document-disjoint metadata plan."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;D=ROOT/'m3_train_docsplit';sys.path.insert(0,str(D));import run

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert a.out.resolve()==D/'independent_review.json'
 s=run.read(D/'monitor_cpu003/started.json');t=run.read(D/'monitor_cpu003/terminal.json');cmd=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'] and [(cwd/x).resolve() for x in cmd if Path(x).name=='cpu_validation.py']==[D/'cpu_validation.py'] and cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==D/'cpu_check_003'
 cpu=run.read(D/'cpu_check_003/verification.json');src=cpu['source_sha256_after'];assert cpu['status']=='passed' and cpu['GT_opened'] is False and cpu['predictions_opened'] is False and cpu['gpu_used'] is False and cpu['training_performed'] is False and cpu['source_sha256_before']==src and len(src)==6 and all(run.sha(p)==h for p,h in src.items());log=(D/'cpu_check_003/stderr.log').read_text();assert cpu['stderr_sha256']==run.sha(D/'cpu_check_003/stderr.log') and 'Ran 4 tests' in log and 'Ran 2 tests' in log and log.rstrip().endswith('OK')
 p=run.read(D/'plan_001.json');assert run.sha(D/'plan_001.json')==run.PLAN_SHA and p['stage']=='post_hoc_document_disjoint_exploration' and p['row_count']==39463 and p['docvqa_eval_rows']==5349 and p['labels_opened'] is False and p['predictions_opened'] is False and p['gpu_used'] is False and p['training_performed'] is False and all(run.sha(k)==v for k,v in p['evidence_sha256'].items())
 pre=run.read(ROOT/'e1_independent_review/train_audit_prelaunch_review.json');ind=run.read(ROOT/'e1_independent_review/train_audit001_review.json');doc=run.read(ROOT/'e1_independent_review/document_overlap002_review.json');run.check_reviewers(pre,ind,doc)
 result={'status':'passed','independent_review':True,'reviewer_sha256':run.sha(__file__),'cpu_sha256':run.sha(D/'cpu_check_003/verification.json'),'cpu_started_sha256':run.sha(D/'monitor_cpu003/started.json'),'cpu_terminal_sha256':run.sha(D/'monitor_cpu003/terminal.json'),'source_sha256':src,'GT_opened':False,'predictions_opened':False,'limits':['Post-hoc source metadata only; no training is authorized.','Exact document IDs and pHash can miss semantic variants.']}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','source_count':len(src)}))
if __name__=='__main__':main()
