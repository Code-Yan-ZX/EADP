import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');D=B/'e4_c_multibudget_worker192_v6';I=B/'e1_independent_review';A=B/'e4_c_multibudget_worker192_v6_accept';R=B/'e4_c_multibudget_references192';ENV=B/'e4_c_multibudget_environment192_v2';AN=B/'e4_c_multibudget_analysis192_exec'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cmd=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()]
 if out is not None:assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
def load(name,path):
 assert name not in sys.modules;sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();p=read(D/'gpu_plan001.json');src=p['source_sha256'];assert len(src)==263 and all(sha(k)==v for k,v in src.items())
 assert p['stage']=='192-image four-budget timing' and (p['jobs'],p['warmups'],p['measured'])==(7920,240,7680) and p['GT_opened'] is False and p['GPU_authorized_by_plan'] is False and (p['inner_timeout_seconds'],p['outer_timeout_seconds'],p['stale_seconds'],p['acceptance_reserve_seconds'])==(28680,28800,1800,3600) and p['proposal_sha256']==sha(B/'e4_c_multibudget_expansion192/proposal002.json')
 terminal(A/'monitor_freeze_gpu_plan001',A/'freeze_gpu_plan.py');terminal(B/'e4_c_multibudget_worker192_v6_launch_dryrun/monitor_run004',B/'e4_c_multibudget_worker192_v6_launch_dryrun/run.py')
 dry=read(B/'e4_c_multibudget_worker192_v6_launch_dryrun/proof001.json');assert dry['status']=='passed' and dry['launcher_sha256']==sha(D/'launch.py') and dry['dryrun_source_sha256']==sha(B/'e4_c_multibudget_worker192_v6_launch_dryrun/run.py') and dry['bootstrap_real'] is True and dry['container_command_real'] is True and dry['gpu_used'] is False and dry['docker_executed'] is False and dry['model_loaded'] is False
 cpu=I/'e4c_worker192_v6_cpu_result_review.json';cs=I/'review_e4c_worker192_v6_cpu_result.py';terminal(I/'monitor_e4c_worker192_v6_cpu_result',cs,cpu);cr=read(cpu);assert cr['status']=='passed' and cr['independent_review'] is True and cr['reviewer_sha256']==sha(cs) and cr['inputs_verified']==222 and cr['GPU_authorized'] is False
 formal=I/'e4c_worker192_v6_accept_prelaunch_review.json';fs=I/'review_e4c_worker192_v6_accept_prelaunch.py';terminal(I/'monitor_e4c_worker192_v6_accept_prelaunch',fs,formal);fr=read(formal);assert fr['status']=='passed' and fr['independent_review'] is True and fr['reviewer_sha256']==sha(fs) and fr['plan_sha256']==sha(A/'verification_plan001.json') and fr['errors']==[] and all(sha(k)==v for k,v in fr['source_sha256'].items())
 analysis=I/'e4c_analysis192_prelaunch_review.json';ars=I/'review_e4c_analysis192_prelaunch.py';terminal(I/'monitor_e4c_analysis192_prelaunch',ars,analysis);ar=read(analysis);assert ar['status']=='passed' and ar['independent_review'] is True and ar['GT_opened'] is False and ar['target_predictions_opened'] is False and ar['errors']==[] and ar['reviewer_sha256']==sha(ars) and ar['plan_sha256']==sha(AN/'analysis_code_plan001.json') and all(sha(k)==v for k,v in ar['source_sha256'].items())
 env=I/'e4c_environment192_v2_prelaunch_review.json';es=I/'review_e4c_environment192_v2_prelaunch.py';terminal(I/'monitor_e4c_environment192_v2_prelaunch',es,env);er=read(env);ep=read(ENV/'plan001.json');assert er['status']=='passed' and er['reviewer_sha256']==sha(es) and er['plan_sha256']==sha(ENV/'plan001.json') and all(sha(k)==v for k,v in er['source_sha256'].items()) and ep['watcher_duration_seconds']>=p['outer_timeout_seconds']+ep['minimum_cleanup_seconds']
 # Fixed32 gate independently rechecks both supervised proofs, 888 identities and source snapshots; no new predictions are generated.
 fg=load('_independent_192_fixed_gate',D/'fixed_reference_gate.py');refs=fg.load(read(B/'e4_c_multibudget_expansion192/proposal002.json'));assert len(refs)==888
 cg=load('_independent_192_cpu_gate',D/'cpu_gate.py');assert cg.admit()['status']=='passed'
 assert not (D/'run_001').exists()
 out={'status':'passed','independent_review':True,'GT_opened':False,'scoring_performed':False,'errors':[],'gpu_plan_sha256':sha(D/'gpu_plan001.json'),'cpu_preflight_review_sha256':sha(cpu),'fixed_formal_sha256':sha(B/'e4_c_multibudget_references192_accept/verification001.json'),'fixed_review_sha256':sha(I/'e4c_multibudget_references192_result_review.json'),'analysis_review_sha256':sha(analysis),'environment_review_sha256':sha(env),'formal_prelaunch_review_sha256':sha(formal),'reviewer_sha256':sha(__file__),'source_sha256':src,'dryrun_sha256':sha(B/'e4_c_multibudget_worker192_v6_launch_dryrun/proof001.json'),'fixed_references_checked':888,'scope':'prelaunch review only; actual observer, GPU availability, deadline and supervisor must be checked at launch','limitations':['Synthetic launcher dry-run does not run Docker or GPU.','Real source snapshot occurs after run and endpoint checks cannot exclude transient replacement.','All timing samples retained; unrecorded host activity and write tails remain unknown.'],'existing_DEV_predictions_opened':True}
 with open(a.out,'x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
