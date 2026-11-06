"""Independent approval of the bounded 384 runtime import preflight only."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/rres_k256_dev384_exec';F=B/'e6/rres_k256_dev384';A=B/'e6/rres_k256_full_analysis_v2';ROOT=B.parents[2]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 assert type(s['worker_pid']) is int and s['worker_pid']>0 and s['worker_pid']==t['worker_pid'] and t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()],str(d)
 if out is not None:assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
 return s

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert not a.out.exists()
 planfile=D/'runtime_preflight_plan001.json';p=read(planfile);cpu=read(D/'cpu_proof001.json');terminal(D/'monitor_cpu_freeze001',D/'cpu_check.py');terminal(D/'monitor_runtime_freeze001',D/'freeze_runtime.py')
 assert p['scope']=='rres384_runtime_import_noGPU' and p['GPU_authorized'] is False and p['GPU_used'] is False and p['GT_opened'] is False and p['scoring_performed'] is False and p['errors']==[] and cpu['status']=='passed' and cpu['GPU_used'] is False and cpu['GT_opened'] is False and cpu['source_sha256_before']==cpu['source_sha256_after']
 source=p['source_sha256'];assert all(source.get(k)==v for k,v in cpu['source_sha256_after'].items()) and all(sha(k)==v for k,v in source.items()) and not any(k.startswith('/data/models/huggingface/') for k in source)
 pre=I/'rres_k256_dev384_preflight_result_review.json';pr=I/'review_rres_k256_dev384_preflight_result.py';terminal(I/'monitor_rres_k256_dev384_preflight_result',pr,pre);q=read(pre)
 assert q['status']=='passed' and q['independent_review'] and q['images']==384 and q['GPU_authorized'] is False and q['GT_opened'] is False and q['errors']==[] and q['plan_sha256']==sha(F/'preflight_plan001.json') and q['worker_preflight_sha256']==sha(F/'run_preflight001/worker_preflight.json') and q['processor_probe_sha256']==sha(F/'run_preflight001/probe/processor_probe.json') and q['reviewer_sha256']==sha(pr) and p['preflight_result_sha256']==sha(pre)
 score=I/'rres_k256_full_score_prelaunch_review.json';sr=I/'review_rres_k256_full_score_prelaunch.py';terminal(I/'monitor_rres_k256_full_score_prelaunch',sr,score);sq=read(score)
 assert sq['status']=='passed' and sq['independent_review'] and sq['scope']=='rres_k256_full_score_prelaunch' and sq['GT_opened'] is False and sq['target_predictions_opened'] is False and sq['errors']==[] and sq['score_plan_sha256']==sha(A/'score_plan001.json') and sq['cpu_proof_sha256']==sha(A/'cpu_proof001.json') and sq['reviewer_sha256']==sha(sr) and p['score_prelaunch_sha256']==sha(score)
 hostcode=D/'host_entry_smoke004.py';host=read(D/'host_entry_smoke004.json');terminal(D/'host_smoke_case004/monitor_run001',hostcode)
 assert host['status']=='passed' and host['GPU_queried'] is False and host['model_loaded'] is False and host['GT_opened'] is False and host['launcher_sha256']==sha(D/'launch.py') and host['script_sha256']==sha(hostcode)
 text=hostcode.read_text();assert "name=='_rres384_gpu_dispatch'" in text and "name=='_rres_gpu_author_loader'" in text and "patch('subprocess.Popen'" in text and "runtime.gpu_pids=stop" in text
 failure=read(D/'host_smoke_case001/monitor_run001/terminal.json');old=read(D/'host_smoke_case001/run001/exit.json');assert failure['reason']=='worker_failed' and failure['worker_returncode']==1 and old['container_stopped_verified'] is False and old['container_exit_code']==1
 cid=old['last_inspect']['id'];obj=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=15).stdout)[0];assert obj['Id']==cid and not obj['State']['Running'] and obj['State']['Pid']==0 and obj['State']['ExitCode']==1 and not obj['State']['OOMKilled']
 tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-p','test_*.py'],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=180);assert tests.returncode==0 and 'Ran 6 tests' in tests.stderr,(tests.stdout,tests.stderr)
 admit=D/'runtime_admit.py';assert sha(admit) in (D/'runtime_probe.py').read_text() and sha(admit) in (D/'runtime_preflight_launch.py').read_text()
 out=D/'runtime_preflight001';assert not out.exists();r=subprocess.run([sys.executable,str(D/'runtime_preflight_launch.py'),'--out',str(out)],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120);assert r.returncode!=0 and not out.exists() and ('terminal.json' in r.stderr or 'supervisor' in r.stderr),r.stderr
 loaderpath=B/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py';spec=importlib.util.spec_from_file_location('_rres384_runtime_review_loader',loaderpath);loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader);runtime=loader.load_author(source);author=read(ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json');cmd=runtime.container_command(author,[str(D/'runtime_probe.py'),'--out',str(out/'worker_preflight.json')],False,'synthetic-review-only')
 assert cmd.count('--rm')==1 and '--gpus' not in cmd and '--device' not in cmd and 'CUDA_VISIBLE_DEVICES=' in cmd and cmd[cmd.index('--network')+1]=='none' and author['container_image'] in cmd
 assert all(sha(k)==v for k,v in source.items())
 result={'status':'passed','independent_review':True,'preflight_only':True,'GPU_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'source_sha256':source,'plan_sha256':sha(planfile),'cpu_sha256':sha(D/'cpu_proof001.json'),'cpu_started_sha256':sha(D/'monitor_cpu_freeze001/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu_freeze001/terminal.json'),'processor_preflight_result_sha256':sha(pre),'score_prelaunch_sha256':sha(score),'host_smoke_sha256':sha(D/'host_entry_smoke004.json'),'host_smoke_started_sha256':sha(D/'host_smoke_case004/monitor_run001/started.json'),'host_smoke_terminal_sha256':sha(D/'host_smoke_case004/monitor_run001/terminal.json'),'accidental_host_smoke_container_id':cid,'reviewer_sha256':sha(__file__),'model_loaded':False,'GPU_used':False,'scope':'only no-device worker imports, not authorization for model/GPU','limitations':['worker formal CPU tests and host mock do not prove model generation','001 host smoke accidentally started a GPU-device container; worker rejected output directory before model/CUDA, supervisor failed and fresh Docker inspect found stopped exit1','accepted DEV records hashed/parsed only for identity; no new predictions or GT'],'errors':[]}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('384 noGPU runtime preflight independent review passed')
if __name__=='__main__':main()
