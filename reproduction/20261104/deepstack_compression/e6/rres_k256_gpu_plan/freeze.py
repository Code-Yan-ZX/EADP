"""Freeze the three-anchor no-GT verifier and GPU plan after both noGPU reviews."""
import hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];E=B/'e6/rres_k256_execution';R=B/'e6/rres_k256_runtime_preflight';P=B/'e6/rres_k256_pilot_v2';I=B/'e1_independent_review'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def check(monitor,reviewer,proof):
 s=read(monitor/'started.json');t=read(monitor/'terminal.json');cwd=Path(s['cwd']);cmd=s['command'];q=read(proof)
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (cwd/cmd[1]).resolve()!=reviewer or cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=proof or q.get('status')!='passed' or q.get('independent_review') is not True or q.get('GPU_authorized') is not False or q.get('GT_opened') is not False or q.get('errors')!=[] or q.get('reviewer_sha256')!=sha(reviewer):raise ValueError('noGPU independent review')
 return q

def write(path,data):
 with path.open('x') as f:json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 old=I/'rres_k256_processor_v2_preflight_result_review.json';oldr=I/'review_rres_k256_processor_v2_preflight_result.py';oldm=I/'monitor_rres_k256_processor_v2_preflight_result';q=check(oldm,oldr,old)
 now=I/'rres_k256_runtime_preflight_result_review.json';nowr=I/'review_rres_k256_runtime_preflight_result.py';nowm=I/'monitor_rres_k256_runtime_preflight_result';r=check(nowm,nowr,now)
 if q.get('images')!=3 or q.get('worker_sha256')!=sha(P/'run_preflight001/worker_preflight.json') or r.get('worker_sha256')!=sha(R/'run001/worker.json'):raise ValueError('noGPU worker proof')
 cpu=read(D/'cpu_proof001.json');ct=read(D/'monitor_cpu001/terminal.json')
 if cpu['status']!='passed' or cpu['before']!=cpu['after'] or ct['reason']!='completed' or ct['worker_returncode']!=0 or ct['remaining_process_group']!=[]:raise ValueError('formal host smoke')
 rt=read(R/'plan001.json');source=dict(rt['source_sha256']);source.update(cpu['after']);extra=[D/'cpu_proof001.json',D/'monitor_cpu001/started.json',D/'monitor_cpu001/terminal.json',R/'plan001.json',R/'run001/worker.json',R/'run001/launch.json',R/'run001/container_identity.json',R/'run001/exit.json',R/'monitor_run001/started.json',R/'monitor_run001/terminal.json',now,nowr,nowm/'started.json',nowm/'terminal.json',D/'freeze.py']
 for path in extra:source[str(path.absolute())]=sha(path)
 if any(sha(k)!=v for k,v in source.items()):raise ValueError('frozen source changed')
 verification={'status':'frozen_before_gpu','run':'e6/rres_k256_execution/run001','run_supervisor':'e6/rres_k256_execution/monitor_run001','snapshot_supervisor':'e6/rres_k256_execution/monitor_snapshot001','formal_supervisor':'e6/rres_k256_execution/monitor_verify001','verifier':str(E/'verify.py'),'verifier_sha256':sha(E/'verify.py'),'images':3,'GT_opened':False,'scoring_performed':False,'output':'e6/rres_k256_execution/verification001.json','scope':'full 3 anchor natural 512 processor, 3 DS/position/cache identity, no equal-compute or speed claim'}
 vp=D/'verification_plan001.json';write(vp,verification);source[str(vp.absolute())]=sha(vp)
 plan={'status':'ready_for_independent_review','images':3,'visual_tokens':256,'resolution':512,'GT_opened':False,'scoring_performed':False,'timing_claims':False,'worker_timeout_seconds':3600,'supervisor_timeout_seconds':3720,'grace_seconds':120,'source_sha256':source,'preflight_result_sha256':sha(old),'preflight_started_sha256':sha(oldm/'started.json'),'preflight_terminal_sha256':sha(oldm/'terminal.json'),'runtime_preflight_result_sha256':sha(now),'runtime_preflight_started_sha256':sha(nowm/'started.json'),'runtime_preflight_terminal_sha256':sha(nowm/'terminal.json'),'verification_plan_sha256':sha(vp),'limitations':['three exploratory DEV anchors','512 image path differs in vision compute from high-resolution K256','source snapshot post-run; no full visual/DS/KV archives']}
 write(E/'gpu_plan001.json',plan)
if __name__=='__main__':main()
