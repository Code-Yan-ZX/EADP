#!/usr/bin/env python3
"""Independent pre-run audit of complete-only R-res 384 scoring and paired stats."""
import argparse,hashlib,importlib.util,json,math,os,subprocess,sys
from pathlib import Path
import numpy as np
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/rres_k256_full_analysis_v2';RUN=B/'e6/rres_k256_dev384';E=B/'e6/rres_k256_dev384_exec'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(m,script):
 s=read(m/'started.json');t=read(m/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and (Path(s['cwd'])/s['command'][1]).resolve()==script.resolve()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out.resolve();assert out==I/'rres_k256_full_score_prelaunch_review.json' and not out.exists() and not (D/'score001').exists() and not (RUN/'run001').exists()
 term(D/'monitor_freeze001',D/'freeze_score.py');term(D/'monitor_cpu001',D/'cpu_check.py');p=read(D/'score_plan001.json');c=read(D/'cpu_proof001.json')
 assert p['status']=='frozen' and p['GT_opened'] is False and p['target_predictions_opened'] is False and c['status']=='passed' and c['GT_opened'] is False and c['target_predictions_opened'] is False and c['source_sha256_before']==c['source_sha256_after']==p['source_sha256'] and all(sha(k)==v for k,v in p['source_sha256'].items()) and c['score_plan_sha256']==sha(D/'score_plan001.json')
 assert len(c['test_logs'])==5 and all(sha(D/(k+'_cpu001.log'))==v for k,v in c['test_logs'].items())
 sys.path.insert(0,str(D));import admission,paired
 assert admission.RUN==RUN and admission.EXEC==E
 # Existing proof-chain validation hashes accepted mixed records but does not parse scored rows or GT.
 assert len(admission.references())==5
 # Genuine CLI must fail at its prelaunch proof, leaving target and GT inaccessible.
 z=subprocess.run([sys.executable,str(D/'score.py'),'--out',str(D/'score001')],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True,timeout=90)
 assert z.returncode!=0 and 'monitor_rres_k256_full_score_prelaunch' in z.stderr and not (D/'score001').exists(),z.stderr[-1200:]
 # Independent fixed-draw bootstrap, with 3 seeds averaged within each question.
 seeds=(20261104,20261105,20261106);rows=[]
 for i in range(5):rows.append({'qid':str(i),'group_id':'g'+str(i//2),'scores':{'rres':i/5,'eadp':(4-i)/5,'e1_full':.6,seeds[0]:.2,seeds[1]:.4,seeds[2]:.6}})
 x=paired.analyze(rows,replicates=500,seed=17);groups=sorted({r['group_id'] for r in rows});by={g:[r for r in rows if r['group_id']==g] for g in groups};rng=np.random.default_rng(17);draw=rng.integers(0,len(groups),(500,len(groups)));dif=[]
 for indexes in draw:
  selected=[r for i in indexes for r in by[groups[int(i)]]];dif.append(100*sum(r['scores']['rres']-r['scores']['eadp'] for r in selected)/len(selected))
 assert math.isclose(x['rres_minus']['eadp']['difference_pp'],100*sum(r['scores']['rres']-r['scores']['eadp'] for r in rows)/len(rows),abs_tol=1e-12) and np.allclose(x['rres_minus']['eadp']['ci95_pp'],np.quantile(dif,[.025,.975])) and math.isclose(x['arm_mean_pct']['random_three_seed_mean'],40.,abs_tol=1e-12)
 value={'status':'passed','independent_review':True,'scope':'rres_k256_full_score_prelaunch','GT_opened':False,'target_predictions_opened':False,'errors':[],'score_plan_sha256':sha(D/'score_plan001.json'),'cpu_proof_sha256':sha(D/'cpu_proof001.json'),'reviewer_sha256':sha(__file__),'checks':['real accepted E1 and 3-seed random proof chain; only hash mixed artifacts','actual entry refuses before prelaunch and target/GT','executor/monitor path separation checked','independent group bootstrap and within-question seed averaging','source and 5 real CPU test supervisors'],'limitations':['prior DEV and three pilot cases already observed; not independent confirmation','same visual-token count does not establish equal compute']}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
