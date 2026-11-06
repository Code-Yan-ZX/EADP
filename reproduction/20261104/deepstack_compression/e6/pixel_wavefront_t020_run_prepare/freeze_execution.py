"""Freeze future GPU identity plan only after reviewed real no-GPU preflight."""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];RUN=B/'e6/pixel_wavefront_model_t020_run';I=B/'e1_independent_review';P=B/'e6/pixel_wavefront_model_t020'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(folder,script,out):
 s=read(folder/'started.json');t=read(folder/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (cwd/cmd[1]).resolve()!=script.resolve() or cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve():raise ValueError('review process incomplete')
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def freeze():
 report=I/'pixel_wavefront_t020_run_preflight_result_002_review.json';reviewer=I/'review_pixel_wavefront_t020_run_preflight_result_002.py';mon=I/'monitor_pixel_wavefront_t020_run_preflight_result_002'
 if not report.is_file():raise ValueError('real no-GPU result review required')
 term(mon,reviewer,report);r=read(report);pre=RUN/'preflight_plan004.json';cpu=RUN/'cpu_check006/verification.json';cs=RUN/'monitor_cpu006/started.json';ct=RUN/'monitor_cpu006/terminal.json'
 checks={'plan_sha256':pre,'worker_preflight_sha256':RUN/'run_preflight001/worker_preflight.json','launch_sha256':RUN/'run_preflight001/launch.json','exit_sha256':RUN/'run_preflight001/exit.json','started_sha256':RUN/'monitor_preflight001/started.json','terminal_sha256':RUN/'monitor_preflight001/terminal.json','reviewer_sha256':reviewer}
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('gpu_authorized') is not False or r.get('GT_opened') is not False or r.get('errors')!=[] or any(r.get(k)!=sha(v) for k,v in checks.items()):raise ValueError('real no-GPU result proof mismatch')
 spec=load('_independent_t020_gpu_source_inventory',RUN/'source_inventory.py');sources=spec.hashes();preplan=read(pre);cp=read(cpu)
 if preplan['source_sha256']!=sources or cp.get('status')!='passed' or cp.get('source_sha256_before')!=sources or cp.get('source_sha256_after')!=sources or cp.get('existing_DEV_predictions_opened') is not True:raise ValueError('source/CPU changed')
 upstream=load('_independent_t020_gpu_execution_upstream',RUN/'execution_upstream.py');up=upstream.verify();base=read(P/'preflight_plan002.json');schedule=load('_independent_t020_gpu_schedule',RUN/'schedule_contract.py');jobs=schedule.schedule(base['images'])
 if preplan.get('jobs')!=jobs or len(jobs)!=6:raise ValueError('six arm schedule changed')
 initial=read(B/'initial_plan.json');budget_sha=sha(B/'initial_plan.json');outer=6480;reserve=3600
 if time.time()<initial['budget_start_unix'] or time.time()+outer+reserve>initial['budget_end_unix']:raise ValueError('insufficient deadline for run and acceptance reserve')
 return {'stage':'t020 three-anchor six-container identity run','threshold':.2,'jobs':jobs,'source_sha256':sources,'preflight_result_sha256':sha(report),'preflight_result_started_sha256':sha(mon/'started.json'),'preflight_result_terminal_sha256':sha(mon/'terminal.json'),'upstream_execution_review_sha256':up['review_sha256'],'cpu_sha256':sha(cpu),'cpu_started_sha256':sha(cs),'cpu_terminal_sha256':sha(ct),'budget_source_sha256':budget_sha,'inner_timeout_seconds':6300,'outer_timeout_seconds':outer,'per_arm_timeout_seconds':900,'grace_seconds':90,'stale_seconds':1200,'minimum_free_bytes':6442450944,'acceptance_reserve_seconds':reserve,'GT_opened':False,'scoring_performed':False,'existing_DEV_predictions_opened':True,'limitations':['Identity only, not a timing experiment; three post-hoc DEV anchors.','Each arm has a fresh process and retains raw tokens, full selected indices and prefill/three DS tensors; per-step KV are fingerprints only.','No full model equivalence or timing claim; reviewer of prior no-GPU preflight authored candidate.','Weight payload SHA checked in earlier no-GPU preflight; this run compares file stat before/after, which cannot rule out transient changes.']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
 if a.out.resolve()!=RUN/'execution_plan001.json':raise ValueError('unexpected plan output')
 result=freeze()
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
