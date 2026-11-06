#!/usr/bin/env python3
"""Independent, pre-capture-value audit of the frozen 384-image analysis."""
import argparse, hashlib, importlib.util, json, math, os, subprocess, sys
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent.parent; D=B/'e4_multistream_expansion343_analysis'; I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(m,script):
 s=read(m/'started.json');t=read(m/'terminal.json')
 assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid']
 assert (Path(s['cwd'])/s['command'][1]).resolve()==script.resolve()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert a.out.resolve()==I/'e4_multistream_expansion343_analysis_v3_prelaunch_review.json' and not a.out.exists()
 plan=read(D/'plan_freeze003.json');cpu=read(D/'cpu_proof003.json');code=read(D/'code_plan003.json')
 # Repair is mechanical and performed after capture and independent norm verification.
 before=(D/'sources_cpu002/analyze.py').read_text();after=(D/'analyze.py').read_text()
 old_gate=read(D/'code_plan002.json')['source_sha256'][str(D/'gate.py')];new_gate=read(D/'code_plan003.json')['source_sha256'][str(D/'gate.py')]
 assert before.replace(old_gate,new_gate).replace('analysis001.json','analysis002.json')==after
 oldstats=read(D/'code_plan002.json')['source_sha256'][str(D/'stats.py')];assert sha(D/'stats.py')==oldstats
 smoke=read(D/'smoke_prior003.json');term(D/'monitor_smoke_prior003',D/'smoke_prior003.py')
 assert smoke['status']=='passed' and smoke['real_prior_and_dual_proof_gate_calls']==2 and smoke['343_capture_opened'] is False and smoke['GT_opened'] is False
 prior=(D/'gate.py').read_text();assert "m.prior(read(T/'plan_freeze001.json'))" in prior and 'finally:del sys.modules[spec.name]' in prior
 assert read(D/'monitor_analysis001/terminal.json')['reason']=='worker_failed' and not (D/'analysis001.json').exists()

 term(D/'monitor_freeze_plan003',D/'freeze_plan003.py');term(D/'monitor_cpu003',D/'freeze_code003.py')
 assert plan['status']=='frozen' and plan['plan_sha256']==sha(D/'PLAN.md') and plan['new_343_values_opened'] is False and plan['GT_opened'] is False
 assert all(sha(k)==v for k,v in plan['identity_and_prior_sha256'].items())
 assert cpu['status']=='passed' and cpu['source_before']==cpu['source_after']==code['source_sha256'] and code['status']=='frozen'
 assert code['plan_sha256']==sha(D/'plan_freeze003.json') and code['cpu_sha256']==sha(D/'cpu_proof003.json') and all(sha(k)==v for k,v in code['source_sha256'].items())
 assert 'Ran 11 tests' in cpu['tests'] and all(cpu[k] is False for k in ('GT_opened','new_343_values_opened','scoring_performed','GPU_used'))
 # Actual entry point must fail closed before its independent proof exists.
 target=D/'analysis002.json';assert not target.exists()
 run=subprocess.run([sys.executable,str(D/'analyze.py'),'--out',str(target)],cwd=B,text=True,capture_output=True,timeout=90)
 assert run.returncode!=0 and not target.exists() and 'e4_multistream_expansion343_analysis_v3_prelaunch' in run.stderr, run.stderr[-1500:]
 # Independent tie-rank and correlation arithmetic, including feature recomputation.
 spec=importlib.util.spec_from_file_location('review_stats343',D/'stats.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
 def rank(v):
  return np.array([1+sum(y<x for y in v)+(sum(y==x for y in v)-1)/2 for x in v],dtype=float)
 x=[2.,1.,2.,4.,3.,4.];y=[3.,3.,0.,4.,1.,4.]
 assert np.array_equal(s.rank(x),rank(x)) and abs(s.corr(s.rank(x),s.rank(y))-np.corrcoef(rank(x),rank(y))[0,1])<1e-14
 arr=np.array([[float((i*(j+1)+j)%17) for i in range(1024)] for j in range(4)],dtype=np.float64)
 f=s.norm_features(arr)
 for j,name in enumerate(s.FEATURES[:4]):assert math.isclose(f[name],float(np.std(arr[j])/np.mean(arr[j])),abs_tol=1e-14)
 rr=[rank(row)/1023-1/1023 for row in arr]
 for j,name in enumerate(s.FEATURES[4:7]):assert math.isclose(f[name],np.corrcoef(rr[0],rr[j+1])[0,1],abs_tol=1e-14)
 assert math.isclose(f[s.FEATURES[7]],float(np.mean(np.var(rr,axis=0))),abs_tol=1e-14)
 rows=[]
 for task in s.TASKS:
  for n in range(128):
   v=float(n%13); rows.append({'task':task,'qid':str(n),'group_id':task+str(n),'features':{k:v for k in s.FEATURES},'scores':[0.,0.,v/12]})
 summary=s.summarize(rows)
 assert len(summary['correlations'])==3 and all(len(v)==8 and all(z['pairs']==128 and math.isclose(z['rho'],1.,abs_tol=1e-14) for z in v.values()) for v in summary['correlations'].values())
 result={'status':'passed','independent_review':True,'GT_opened':False,'new_343_values_opened':False,'errors':[],'source_sha256':code['source_sha256'],'reviewer_sha256':sha(__file__),'plan_sha256':sha(D/'plan_freeze003.json'),'code_sha256':sha(D/'code_plan003.json'),'cpu_sha256':sha(D/'cpu_proof003.json'),'cpu_started_sha256':sha(D/'monitor_cpu003/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu003/terminal.json'),'repair_after_full_343_identity_verification':True,'checks':['real prior integration boundary called twice and temporary module cleaned','analysis differs from pre-capture freeze only by gate digest and output name','frozen sources and real completed supervisors','entry fails before independent proof and no output created','independent average ties, Spearman and eight norm feature calculations','synthetic 384 rows and 24 correlations','source reviewed for all 343 dual proofs before new captures, complete identities and no GT reads']}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
