"""Independent GPU prelaunch review; no GPU/container starts."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/rres_eadp_identity3_execution';PY=Path('/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def clean(m,script=None,out=None):
 s=read(m/'started.json');t=read(m/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if cwd.resolve()!=B or s['worker_pid']!=t['worker_pid'] or t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[]:raise ValueError('supervisor '+str(m))
 if script is not None and [ (cwd/x).resolve() for x in cmd if Path(x).name==script.name ]!=[script.resolve()]:raise ValueError('supervisor script')
 if out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()):raise ValueError('supervisor output')
 return s
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();dest=I/'rres_eadp_identity3_gpu_prelaunch_review.json'
 if a.out.resolve()!=dest or dest.exists() or (D/'run001').exists():raise ValueError('unplanned output or GPU already started')
 plan=read(D/'gpu_plan001.json');formal=read(D/'verification_plan001.json');sources=plan['source_sha256']
 if plan['scope']!='rres_eadp_identity3_gpu' or plan['anchors']!=[['DocVQA_VAL','64588'],['TextVQA_VAL','36511'],['AI2D_TEST','207406']] or plan['arms']!=['rres','eadp'] or plan['GT_opened'] is not False or plan['timing_claims'] is not False or any(sha(k)!=v for k,v in sources.items()):raise ValueError('plan/source')
 if formal['gpu_plan_sha256']!=sha(D/'gpu_plan001.json') or formal['verifier_sha256']!=sha(D/'verify.py') or formal['snapshot_sha256']!=sha(D/'snapshot.py') or formal['GT_opened'] is not False or formal['timing_claims'] is not False:raise ValueError('formal plan')
 clean(D/'monitor_freeze_gpu001',D/'freeze_gpu.py');clean(D/'monitor_cpu006');clean(D/'host_smoke002/monitor_run001',D/'host_entry_smoke.py');clean(B/'e6/rres_eadp_paired_timing_assessment/monitor_control_lifecycle001')
 pre=I/'rres_eadp_identity3_execution_preflight_result_review.json';reviewer=I/'review_rres_eadp_identity3_execution_preflight_result.py';clean(I/'monitor_rres_eadp_identity3_execution_preflight_result',reviewer,pre);q=read(pre);run=D/'cpu_preflight001';mon=D/'monitor_cpu_preflight001'
 files={'plan_sha256':D/'cpu_plan001.json','worker_sha256':run/'worker_preflight.json','launch_sha256':run/'launch.json','identity_sha256':run/'container_identity.json','exit_sha256':run/'exit.json','started_sha256':mon/'started.json','terminal_sha256':mon/'terminal.json','reviewer_sha256':reviewer}
 if q['status']!='passed' or q['independent_review'] is not True or q['preflight_only'] is not True or q['GPU_authorized'] is not False or q['GT_opened'] is not False or q['existing_predictions_opened'] is not True or q['errors']!=[] or any(q.get(k)!=sha(v) for k,v in files.items()) or plan['execution_preflight_review_sha256']!=sha(pre):raise ValueError('actual noGPU independent result')
 clean(mon,D/'cpu_launch.py',run)
 if plan['preflight_review_sha256']!=sha(I/'rres_eadp_identity3_v2_preflight_result_review.json'):raise ValueError('prior identity noGPU')
 p1=str((D/'run001/rres/observations/progress.jsonl').resolve());p2=str((D/'run001/eadp/observations/progress.jsonl').resolve());cmd=[str(PY),str(B/'monitor/supervise.py'),'--run-dir',str(D/'monitor_run001'),'--timeout','3600','--stale','1200','--progress',p1,'--progress',p2,'--grace-seconds','120','--',str(PY),str(D/'controller.py'),'--run-id','001']
 if plan['run_command']!=cmd or [plan['container_timeout_seconds'],plan['outer_timeout_seconds'],plan['stale_seconds'],plan['outer_grace_seconds']]!=[1600,3600,1200,120]:raise ValueError('command or bounds')
 base=[str(PY),str(B/'monitor/supervise.py'),'--run-dir'];snap=base+[str(D/'monitor_snapshot001'),'--timeout','900','--',str(PY),str(D/'snapshot.py')];verify=base+[str(D/'monitor_verify001'),'--timeout','900','--',str(PY),str(D/'verify.py'),'--out',str(D/'verification001.json')]
 if formal['snapshot_command']!=snap or formal['verify_command']!=verify:raise ValueError('formal commands')
 old=read(B/'e6/rres_k256_dev384/run001/launch.json')['command'];new=read(D/'host_smoke002/run001/rres/launch.json')['command'];old[old.index('--name')+1]=new[new.index('--name')+1];image=next(x for x in old if x.startswith('sha256:'));old=old[:old.index(image)+1]+new[new.index(image)+1:]
 if old!=new or new.count('--device')!=4 or new.count('--mount')!=12:raise ValueError('actual host command template')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PAIRED_TIMING_CANDIDATE='rres_eadp_timing_pilot_v4',IDENTITY_TIMING_CANDIDATE='rres_eadp_identity3_v2');tests=[]
 for d in (D,B/'e6/rres_eadp_paired_timing_assessment'):
  t=subprocess.run([str(PY),'-m','unittest','discover','-s',str(d),'-p','test_*.py'],cwd=B,env=env,text=True,capture_output=True,timeout=180)
  if t.returncode:raise ValueError('independent tests '+t.stderr)
  tests.append(t.stderr)
 if any(sha(k)!=v for k,v in sources.items()) or (D/'run001').exists():raise ValueError('source drift or run race')
 result={'status':'passed','independent_review':True,'scope':'identity3_gpu_prelaunch','errors':[],'GT_opened':False,'plan_sha256':sha(D/'gpu_plan001.json'),'verification_plan_sha256':sha(D/'verification_plan001.json'),'reviewer_sha256':sha(__file__),'source_sha256':sources,'independent_tests':tests,'execution_preflight_review_sha256':sha(pre),'limits':['Only three anchors, two arms; diagnostic synchronizations and hooks preclude timing claims.','No comparable prior EADP prefill logits; raw tensor arrays not archived.','Endpoint hashes/stat and post-run snapshot cannot rule out temporary changes.','Fake Docker cleanup is not actual GPU cleanup proof; real run still requires root scheduling and full formal independent result acceptance.']}
 with dest.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
