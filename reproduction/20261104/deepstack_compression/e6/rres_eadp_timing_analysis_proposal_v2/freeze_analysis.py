"""Freeze report paths and proof contract; does not open timing results or GT."""
import argparse,hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];E=B/'e6/rres_eadp_timing_execution';I=B/'e1_independent_review';END=1794065414

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def term(d,script,out=None):
 a=read(d/'started.json');b=read(d/'terminal.json');cmd=a['command'];cwd=Path(a['cwd'])
 if (a.get('worker_pid')!=b.get('worker_pid') or b.get('reason')!='completed' or b.get('worker_returncode')!=0 or b.get('remaining_process_group')!=[] or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()))):raise ValueError('supervision')
def main():
 a=argparse.ArgumentParser()
 for option in ('launcher','snapshot','verifier','execution-plan'):a.add_argument('--'+option,type=Path,required=True)
 args=a.parse_args();out=D/'analysis_plan001.json'
 if out.exists():raise ValueError('do not overwrite frozen plan')
 for name in ('launcher','snapshot','verifier','execution_plan'):
  if getattr(args,name).resolve().parent!=E.resolve():raise ValueError('unplanned execution path')
 if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')) or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU environment')
 stat=read(D/'statistics_plan001.json');cpu=read(D/'analysis_cpu_proof002.json');term(D/'monitor_statistics_freeze001',D/'freeze_statistics.py');term(D/'monitor_analysis_cpu002',D/'cpu_check002.py')
 if (stat.get('status')!='frozen' or cpu.get('status')!='passed' or cpu.get('tests')!=22 or cpu.get('statistics_plan_sha256')!=sha(D/'statistics_plan001.json') or stat.get('numpy_version')!=cpu.get('numpy_version') or cpu.get('GT_opened') is not False or cpu.get('timing_results_opened') is not False or cpu.get('errors')!=[] or any(sha(k)!=v for k,v in cpu['source_sha256'].items()) or read(B/'initial_plan.json')['budget_end_unix']!=END):raise ValueError('CPU/stat source')
 journals=[E/f'run001/job_{i:03d}/observations/journal.jsonl' for i in range(12)]
 completions=[E/f'run001/job_{i:03d}/observations/worker_complete.json' for i in range(12)]
 observer=E/'observations001.jsonl';run_health=E/'monitor_run001/health.jsonl';observer_health=E/'monitor_observer001/health.jsonl';auto=E/'automatic_health_windows_snapshot001.jsonl';auto_e=E/'automatic_health_events_snapshot001.jsonl';auto_m=E/'automatic_health_snapshot001.json'
 proof=I/'rres_eadp_timing_analysis_prelaunch_review.json';reviewer=I/'review_rres_eadp_timing_analysis_prelaunch.py';mon=I/'monitor_rres_eadp_timing_analysis_prelaunch'
 formal=E/'verification001.json';ind=I/'rres_eadp_timing_result_review.json';indreview=I/'review_rres_eadp_timing_result.py'
 source=dict(cpu['source_sha256']);source.update({str(D/'analysis_cpu_proof002.json'):sha(D/'analysis_cpu_proof002.json'),str(D/'monitor_analysis_cpu002/started.json'):sha(D/'monitor_analysis_cpu002/started.json'),str(D/'monitor_analysis_cpu002/terminal.json'):sha(D/'monitor_analysis_cpu002/terminal.json'),str(Path(__file__)):sha(__file__)})
 plan={'status':'frozen','scope':'posthoc 32-image paired timing; no real results read prelaunch','images':32,'jobs':380,'warmups':60,'measured':320,'processes':12,'budget_end_unix':END,'GT_opened':False,'timing_results_opened':False,'numpy_version':stat['numpy_version'],'source_sha256':source,'schedule_path':str(B/'e6/rres_eadp_timing_pilot_v4/schedule001.json'),'journal_paths':list(map(str,journals)),'worker_completion_paths':list(map(str,completions)),'observer_path':str(observer),'run_health_path':str(run_health),'observer_health_path':str(observer_health),'automatic_health_windows_snapshot':str(auto),'automatic_health_events_snapshot':str(auto_e),'automatic_health_snapshot_metadata':str(auto_m),'required_result_paths':list(map(str,completions+[observer,run_health,observer_health,auto,auto_e,auto_m])),
 'prelaunch':{'proof':str(proof),'reviewer':str(reviewer),'monitor':str(mon),'analysis_plan':str(out),'statistics_plan':str(D/'statistics_plan001.json'),'cpu_proof':str(D/'analysis_cpu_proof002.json')},
 'future':{'formal':str(formal),'independent':str(ind),'run_monitor':str(E/'monitor_run001'),'launcher':str(args.launcher.resolve()),'observer_monitor':str(E/'monitor_observer001'),'observer':str(E/'observer.py'),'snapshot_monitor':str(E/'monitor_snapshot001'),'snapshot':str(args.snapshot.resolve()),'formal_monitor':str(E/'monitor_verify001'),'verifier':str(args.verifier.resolve()),'independent_monitor':str(I/'monitor_rres_eadp_timing_result'),'reviewer':str(indreview),'execution_plan':str(args.execution_plan.resolve()),'source_manifest':str(E/'source_snapshot001/manifest.json')},'errors':[]}
 with out.open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
