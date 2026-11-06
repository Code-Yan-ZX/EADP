"""Independent synthetic-only review of frozen warm192 analysis v3."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[1];D=B/'e6/pixel_wavefront_t020_warm192';E=B/'e6/pixel_wavefront_t020_warm192_extra_checks';I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if (len(cmd)<2 or (cwd/cmd[1]).resolve()!=script.resolve() or s.get('worker_pid')!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()))):raise ValueError('supervision')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
 if a.out.resolve()!=(I/'pixel_t020_warm192_analysis_prelaunch_review.json').resolve():raise ValueError('output')
 analysis=read(D/'analysis_plan001.json');gpu=read(D/'gpu_plan001.json');formal=read(D/'formal_plan001.json');source=gpu['source_sha256']
 terminal(E/'monitor_gpu_plan_freeze003',E/'freeze_gpu_plans_v3.py')
 if any(sha(k)!=v for k,v in source.items()) or analysis['source_sha256']!=source or formal['source_sha256']!=source or analysis['gpu_plan_sha256']!=sha(D/'gpu_plan001.json') or analysis['formal_plan_sha256']!=sha(D/'formal_plan001.json') or formal['gpu_plan_sha256']!=sha(D/'gpu_plan001.json') or analysis.get('draws')!=10000 or analysis.get('bootstrap_seed')!=20261105:raise ValueError('frozen plans')
 for monitor,script in [('monitor_observer_guard_real_cpu003','observer_guard_real_cpu_v3.py'),('monitor_observer_actual_integration_v3_001','observer_actual_integration_v3.py'),('monitor_formal_observer_smoke_v3_002','test_formal_observer_v3_002.py')]:
  terminal(E/monitor,E/script)
 s=read(E/'monitor_observer_guard_cpu003/started.json');t=read(E/'monitor_observer_guard_cpu003/terminal.json')
 if s['command'][1:4]!=['-m','unittest','discover'] or s['command'][-1]!='test_observer_guard_v3.py' or s.get('worker_pid')!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[]:raise ValueError('observer synthetic supervisor')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
 result=subprocess.run([sys.executable,'-m','unittest','-v','test_stats','test_analysis_gate'],cwd=D,env=env,capture_output=True,text=True,timeout=120)
 if result.returncode:raise ValueError('independent analysis tests: '+result.stderr[-1000:])
 subprocess.run([sys.executable,'-m','unittest','-v','test_observer_guard_v3'],cwd=E,env=env,capture_output=True,text=True,timeout=120,check=True)
 spec=importlib.util.spec_from_file_location('_warm192_independent_stats',D/'stats.py');stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)
 tasks=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST');design={'blocks':[]};records=[];blocks=[[] for _ in range(4)]
 for ti,task in enumerate(tasks):
  for k in range(64):
   row={'task':task,'qid':str(k),'group_id':task+str(k)};blocks[k//16].append(row)
   for arm in ('author','wavefront'):
    dur=1000+3*k+ti+(20+(k%7) if arm=='wavefront' else 0)
    records.append({**row,'arm':arm,'boundaries':{'t0_before_image_read_ns':100,'first_token_selected_ns':100+dur},'first_token_id':4,'selected_indices':[0,1],'eos_ids':[2],'first_token_is_eos':False})
 design['blocks']=blocks;actual=stats.calculate(design,records)
 rng=np.random.default_rng(20261105)
 for task in tasks:
  entries=[x for x in actual['per_question'] if x['task']==task]
  diff=np.array([x['wavefront_ns']-x['author_ns'] for x in entries],dtype=float);rat=np.array([x['wavefront_ns']/x['author_ns'] for x in entries],dtype=float)
  sample=rng.integers(0,64,size=(10000,64));r=actual['tasks'][task]
  if r['images']!=64 or r['paired_difference_median_ns']!=float(np.median(diff)) or r['paired_ratio_median']!=float(np.median(rat)) or r['paired_difference_bootstrap_95_ns']!=[float(x) for x in np.quantile(np.median(diff[sample],axis=1),[.025,.975])] or r['paired_ratio_bootstrap_95']!=[float(x) for x in np.quantile(np.median(rat[sample],axis=1),[.025,.975])]:raise ValueError('independent paired bootstrap')
 if actual['pairs']!=192 or actual['observations']!=384:raise ValueError('synthetic denominator')
 if (D/'analysis001.json').exists():raise ValueError('unexpected analysis result already exists')
 reject=subprocess.run([sys.executable,str(E/'analysis_v3.py'),'--out',str(D/'analysis001.json')],cwd=B,env=env,capture_output=True,text=True,timeout=120)
 if reject.returncode==0 or (D/'analysis001.json').exists():raise ValueError('analysis did not fail closed')
 if any(sha(k)!=v for k,v in source.items()):raise ValueError('source changed')
 out={'status':'passed','independent_review':True,'GT_opened':False,'timing_results_opened':False,'errors':[],
      'plan_sha256':sha(D/'analysis_plan001.json'),'reviewer_sha256':sha(__file__),'source_sha256':source,
      'gpu_plan_sha256':sha(D/'gpu_plan001.json'),'formal_plan_sha256':sha(D/'formal_plan001.json'),
      'synthetic_pairs':192,'synthetic_observations':384,'observer_cpu_and_real_integration':True,
      'limits':['post-hoc 192 DEV one-token TTFT, not quality or full-generation inference','all rows and interference unknowns retained; no real timing read']}
 with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
