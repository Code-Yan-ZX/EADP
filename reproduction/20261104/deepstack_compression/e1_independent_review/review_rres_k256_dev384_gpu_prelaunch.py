"""Independent source and evidence review; does not start Docker or use a GPU."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent; I=B/'e1_independent_review'; F=B/'e6/rres_k256_dev384'; E=B/'e6/rres_k256_dev384_exec'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def clean(p):
 s=read(p/'started.json'); t=read(p/'terminal.json')
 if s['worker_pid']!=t['worker_pid'] or t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[]:raise ValueError('supervision '+str(p))
 return s
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
 if a.out.resolve()!=I/'rres_k256_dev384_gpu_prelaunch_review.json' or a.out.exists():raise ValueError('output')
 sys.path.insert(0,str(F));import pilot_gate
 pilot=pilot_gate.require();gp=F/'gpu_plan001.json';fp=F/'verification_plan001.json';plan=read(gp);formal=read(fp);source=plan['source_sha256']
 if len(source)!=271 or any(sha(k)!=v for k,v in source.items()):raise ValueError('GPU source inventory')
 if plan['scope']!='rres384_gpu_generation' or plan['images']!=384 or plan['resolution']!=512 or plan['visual_tokens']!=256 or plan['GT_opened'] is not False or plan['scoring_performed'] is not False or plan['GPU_used_at_freeze'] is not False or (plan['inner_timeout_seconds'],plan['outer_timeout_seconds'],plan['grace_seconds'])!=(21000,21600,300) or plan['pilot_review_sha256']!=pilot['review_sha256']:raise ValueError('GPU contract')
 if formal['gpu_plan_sha256']!=sha(gp) or formal['verifier_sha256']!=sha(E/'verify.py') or formal['cpu_proof_sha256']!=sha(E/'cpu_proof001.json') or formal['GT_opened'] is not False or formal['scoring_performed'] is not False:raise ValueError('formal contract')
 freeze=clean(E/'gpu_plan_tools/monitor_freeze001');cpu=clean(E/'monitor_cpu_freeze001'); formal_cpu=clean(E/'monitor_formal_cpu002');sm=clean(E/'host_smoke_case004/monitor_run001')
 if Path(freeze['cwd'],freeze['command'][1]).resolve()!=E/'gpu_plan_tools/freeze_gpu.py':raise ValueError('freeze command')
 cp=read(E/'cpu_proof001.json')
 if cp['status']!='passed' or cp['source_sha256_before']!=cp['source_sha256_after'] or cp['GPU_used'] is not False or cp['GT_opened'] is not False or cp['errors']!=[] or any(sha(k)!=v or source.get(k)!=v for k,v in cp['source_sha256_before'].items()):raise ValueError('CPU source proof')
 host=read(E/'host_entry_smoke004.json')
 if host['status']!='passed' or host['GPU_queried'] is not False or host['model_loaded'] is not False or host['launcher_sha256']!=sha(E/'launch.py') or host['script_sha256']!=sha(E/'host_entry_smoke004.py'):raise ValueError('actual launcher smoke')
 if Path(sm['cwd'],sm['command'][1]).resolve()!=E/'host_entry_smoke004.py':raise ValueError('smoke command')
 for mon,script,proof in [('rres_k256_dev384_preflight_result','review_rres_k256_dev384_preflight_result.py','rres_k256_dev384_preflight_result_review.json'),('rres_k256_dev384_runtime_result','review_rres_k256_dev384_runtime_result.py','rres_k256_dev384_runtime_result_review.json'),('rres_k256_full_score_prelaunch','review_rres_k256_full_score_prelaunch.py','rres_k256_full_score_prelaunch_review.json')]:
  pilot_gate.terminal(I/('monitor_'+mon),I/script,I/proof)
  q=read(I/proof)
  if q['status']!='passed' or q['independent_review'] is not True or q['GT_opened'] is not False or q['errors']!=[] or q['reviewer_sha256']!=sha(I/script):raise ValueError('independent evidence '+mon)
 pre=I/'rres_k256_dev384_preflight_result_review.json';run=I/'rres_k256_dev384_runtime_result_review.json';score=I/'rres_k256_full_score_prelaunch_review.json'
 if read(pre)['images']!=384 or read(run)['GPU_authorized'] is not False or read(score)['target_predictions_opened'] is not False or any(plan[k]!=sha(v) for k,v in [('preflight_result_sha256',pre),('runtime_result_sha256',run),('score_review_sha256',score)]):raise ValueError('prior evidence binding')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
 test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(E),'-p','test_*.py'],cwd=B,env=env,capture_output=True,text=True,timeout=180)
 if test.returncode:raise RuntimeError('independent CPU tests: '+test.stderr)
 if any(sha(k)!=v for k,v in source.items()):raise ValueError('sources changed during review')
 out={'status':'passed','independent_review':True,'scope':'rres384_gpu_prelaunch','GT_opened':False,'scoring_performed':False,'errors':[],'source_sha256':source,'plan_sha256':sha(gp),'formal_plan_sha256':sha(fp),'preflight_result_sha256':sha(pre),'runtime_result_sha256':sha(run),'score_review_sha256':sha(score),'reviewer_sha256':sha(__file__),'cpu_test_stderr':test.stderr,'limitations':['pilot anchor differences are reported without filtering 384 results','source snapshot follows the run and endpoint weight stats cannot exclude temporary changes','full visual and KV tensors are not archived; observations use online hashes and metadata','512 resolution and K256 do not establish equal compute','GPU scheduling is controlled separately']}
 with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
