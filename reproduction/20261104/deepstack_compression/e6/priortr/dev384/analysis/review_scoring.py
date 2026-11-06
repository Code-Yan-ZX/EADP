"""Independent DEV384 scoring source review; no prediction or answer parsing."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;DEV=HERE.parent;BASE=DEV.parents[2]
sys.path.insert(0,str(HERE))
import score_panel as panel,score_core as core
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert a.out.resolve()==HERE/'prelaunch_review.json'
 start=HERE/'monitor_core005/started.json';end=HERE/'monitor_core005/terminal.json';s,t=read(start),read(end);cmd=s['command'];cwd=Path(s['cwd'])
 assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid']
 assert '-m' in cmd and 'unittest' in cmd and 'discover' in cmd and cmd.count('-s')==1 and (cwd/cmd[cmd.index('-s')+1]).resolve()==HERE
 log=(HERE/'monitor_core005/stderr.log').read_text();assert 'Ran 8 tests' in log and log.rstrip().endswith('OK')
 panel.validate_imports();e1,score,proof=panel.accepted_e1()
 p=read(DEV/'scoring_plan_001.json');assert p['GT_opened'] is False and p['scoring_performed'] is False and p['bootstrap']=={'unit':'image group','replicates':10000,'seed':20261104,'interval':'two-sided percentile 95%; exploratory, no familywise correction'}
 assert p['arms']==[*core.ARMS,core.REFERENCE] and p['contrasts']==[a+' - '+b for a,b in core.CONTRASTS]
 # Independent arithmetic for nonconstant paired group bootstrap using exactly
 # the frozen random draws. All rows below are synthetic; no data is read.
 values=np.array([[0.,.2,.5,.1],[1.,.4,.6,.3],[.5,.9,.1,.7],[.2,.3,.8,.4]])
 groups=['a','a','b','c'];configs=(*core.ARMS,core.REFERENCE)
 rows=[{'group_id':g,'arms':{k:{'score':float(v)} for k,v in zip(configs,row)}} for g,row in zip(groups,values)]
 result=core.analyze(rows,replicates=137,seed=37);draw=np.random.default_rng(37).integers(0,3,size=(137,3));sets=[[0,1],[2],[3]]
 for left,right in core.CONTRASTS:
  i,j=configs.index(left),configs.index(right);delta=values[:,i]-values[:,j];samples=[float(np.mean([delta[k] for group in trial for k in sets[group]]))*100 for trial in draw]
  got=result['contrasts'][left+' - '+right];assert np.isclose(got['difference_pp'],100*delta.mean()) and np.allclose(got['percentile_95_ci_pp'],np.quantile(samples,[.025,.975]))
 required=[DEV/'scoring_plan_001.json',HERE/'score_core.py',HERE/'test_score_core.py',HERE/'test_score_gate.py',HERE/'score_panel.py',DEV/'gate.py',BASE/'e1_analysis/metrics.py',BASE/'e1_analysis/score.py',BASE/'e1_analysis/paired_stats.py',panel.metrics.VQA,panel.metrics.MISC,panel.metrics.MATCH]
 src={str(Path(x).resolve()):sha(x) for x in required}
 result={'status':'passed','independent_review':True,'GT_opened':False,'scoring_performed':False,'reviewer_sha256':sha(__file__),
         'core_started_sha256':sha(start),'core_terminal_sha256':sha(end),'source_sha256':src,'e1_score_review_sha256':sha(proof),
         'scope':'scoring source and synthetic statistics only; full prediction and labels remain unopened'}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','sources':len(src),'synthetic_contrasts':len(core.CONTRASTS)}))
if __name__=='__main__':main()
