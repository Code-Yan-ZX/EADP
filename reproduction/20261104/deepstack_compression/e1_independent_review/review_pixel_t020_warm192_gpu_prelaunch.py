"""Independent final source/admission audit, no timing or GPU access."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parents[1];D=B/'e6/pixel_wavefront_t020_warm192';E=B/'e6/pixel_wavefront_t020_warm192_extra_checks';I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if (len(cmd)<2 or (cwd/cmd[1]).resolve()!=script.resolve() or s.get('worker_pid')!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()))):raise ValueError('supervisor')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
 if a.out.resolve()!=(I/'pixel_t020_warm192_gpu_prelaunch_review.json').resolve():raise ValueError('output')
 gp=D/'gpu_plan001.json';fp=D/'formal_plan001.json';apath=D/'analysis_plan001.json';p=read(gp);f=read(fp);analysis=read(apath);source=p['source_sha256'];design=read(D/'design001.json')
 terminal(E/'monitor_gpu_plan_freeze003',E/'freeze_gpu_plans_v3.py')
 if (any(sha(k)!=v for k,v in source.items()) or p['jobs']!=design['processes'] or len(p['jobs'])!=8 or p.get('design_sha256')!=sha(D/'design001.json') or p.get('GT_opened') is not False or p.get('scoring_performed') is not False or p.get('outer_timeout_seconds')!=21600 or p.get('grace_seconds')!=300 or p.get('inner_timeout_seconds')!=2400 or p.get('cleanup_seconds')!=300 or p.get('budget_source_sha256')!=sha(B/'initial_plan.json') or f.get('gpu_plan_sha256')!=sha(gp) or analysis.get('gpu_plan_sha256')!=sha(gp) or analysis.get('formal_plan_sha256')!=sha(fp) or f['source_sha256']!=source or analysis['source_sha256']!=source):raise ValueError('frozen GPU plan or source')
 pairs=p['jobs'];arms=[('author','wavefront'),('wavefront','author'),('wavefront','author'),('author','wavefront')]
 if any(pairs[2*i]['arm']!=arms[i][0] or pairs[2*i+1]['arm']!=arms[i][1] or pairs[2*i]['measured']!=pairs[2*i+1]['measured'] or len(pairs[2*i]['measured'])!=48 for i in range(4)):raise ValueError('paired schedule')
 checks=[('pixel_t020_warm192_preflight_result_review.json','review_pixel_t020_warm192_preflight_result.py','monitor_pixel_t020_warm192_preflight_result'),('pixel_t020_warm192_analysis_prelaunch_review.json','review_pixel_t020_warm192_analysis_prelaunch.py','monitor_pixel_t020_warm192_analysis_prelaunch')]
 for proof,script,mon in checks:
  terminal(I/mon,I/script,I/proof);q=read(I/proof)
  if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('errors')!=[] or q.get('reviewer_sha256')!=sha(I/script):raise ValueError('prior independent review')
 if read(I/checks[1][0]).get('plan_sha256')!=sha(apath):raise ValueError('analysis proof')
 for mon,script in [('monitor_host_entry004','host_entry_smoke004.py'),('monitor_observer_guard_real_cpu003','observer_guard_real_cpu_v3.py'),('monitor_observer_actual_integration_v3_001','observer_actual_integration_v3.py'),('monitor_formal_observer_smoke_v3_002','test_formal_observer_v3_002.py'),('monitor_postrun_entry_smoke_v3_001','postrun_entry_smoke_v3.py'),('monitor_one_token_generate001','one_token_generate.py')]:terminal(E/mon,E/script)
 host=read(E/'host_entry004.json');integration=read(E/'observer_actual_integration_v3_001.json');one=read(E/'one_token_generate001.json')
 if host.get('status')!='passed' or integration.get('status')!='passed' or one.get('status')!='passed' or one.get('actual_transformers_generate') is not True or integration.get('GPU_and_docker_queries_stubbed') is not True:raise ValueError('actual dependency/observer/first-token smoke')
 guard=integration['actual_guard'];need=p['outer_timeout_seconds']+p['grace_seconds']
 if guard.get('required_remaining_seconds')!=need or min(guard['deadline_lower_monotonic_ns'],guard['supervisor_deadline_lower_monotonic_ns'])<guard['checked_monotonic_ns']+need*10**9 or not integration['actual_health_first'].get('health_snapshot_window_monotonic_ns'):raise ValueError('actual observer and supervisor duration')
 # Independent synthetic suspended-clock check: 10 s suspend must move the lower bound earlier, never later.
 path=E/'observer_guard_v3.py';spec=importlib.util.spec_from_file_location('_independent_warm_observer_guard',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 normal=m.lifetime(100000,100,1010*10**9,1010*10**9,23000,21900);suspended=m.lifetime(100000,100,1010*10**9,1020*10**9,23000,21900)
 if normal['observer_start_lower_monotonic_ns']-suspended['observer_start_lower_monotonic_ns']!=10*10**9:raise ValueError('suspend conservatism')
 try:m.lifetime(100000,100,2300*10**9,2300*10**9,23000,21900)
 except ValueError:pass
 else:raise ValueError('insufficient observer accepted')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');r=subprocess.run([sys.executable,'-m','unittest','-v','test_observer_guard_v3'],cwd=E,env=env,capture_output=True,text=True,timeout=120)
 if r.returncode:raise ValueError('observer CPU tests')
 if any(sha(k)!=v for k,v in source.items()):raise ValueError('source changed during review')
 out={'status':'passed','independent_review':True,'scope':'pixel_t020_warm192_gpu_prelaunch','GT_opened':False,'scoring_performed':False,'errors':[],
      'source_sha256':source,'plan_sha256':sha(gp),'formal_plan_sha256':sha(fp),'analysis_plan_sha256':sha(apath),
      'analysis_review_sha256':sha(I/'pixel_t020_warm192_analysis_prelaunch_review.json'),
      'preflight_result_sha256':sha(I/'pixel_t020_warm192_preflight_result_review.json'),'reviewer_sha256':sha(__file__),
      'limits':['approval proves readiness only; root must verify GPU idle, deadline and live observer at launch',
                'post-hoc one-token paired DEV timing, not full-generation accuracy or steady-state guarantee',
                'existing DEV predictions opened solely for identity/source hashes; no GT or new timing read']}
 with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
