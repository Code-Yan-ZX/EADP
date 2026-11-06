#!/usr/bin/env python3
"""Independent no-device-only preflight review for full-DEV wavefront."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/pixel_wavefront_t020_full_dev'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(m,script):
 s=read(m/'started.json');t=read(m/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and (Path(s['cwd'])/s['command'][1]).resolve()==script.resolve()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out.resolve();assert out==D/'independent_preflight_review.json' and not out.exists() and not (D/'run_preflight001').exists() and not (D/'run_run001').exists()
 term(D/'monitor_freeze002',D/'freeze.py');term(D/'monitor_cpu002',D/'cpu_validation.py')
 p=read(D/'plan002.json');c=read(D/'cpu_proof002.json');assert p['questions']==384 and p['GT_opened'] is False and p['scoring_performed'] is False and c['status']=='passed' and c['plan_sha256']==sha(D/'plan002.json') and c['source_sha256']==p['source_sha256'] and all(sha(k)==v for k,v in p['source_sha256'].items())
 assert sha(D/'dependency_smoke_cpu002.log')==c['dependency_smoke_sha256'] and sha(D/'launch_rejection_cpu002.log')==c['launch_rejection_sha256'] and all(sha(D/(name+'_cpu002.log'))==digest for name,digest in c['test_logs'].items())
 required=[D/x for x in ('admission.py','baseline.py','compare.py','worker.py','launch.py','verify.py','snapshot.py','finish_guard.py')]+[B.parents[2]/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/diagnose.py']
 assert all(str(x.resolve()) in p['source_sha256'] for x in required)
 sys.path.insert(0,str(D));import admission,baseline,compare
 checked=admission.check_plan();assert checked['baseline']['questions']==384 and len(checked['baseline']['files_sha256'])==768 and checked['wavefront_prior']['candidate_sha256']==sha(B/'e6/pixel_wavefront_cpu/candidate.py')
 assert len(read(baseline.ORIGINAL/'plan001.json')['anchors'])==384
 text=(D/'worker.py').read_text();assert 'Pred2DSelector._select_2d_loco_anchored=wavefront_selector' in text and 'for r in chosen:' in text and 'weight_sha256_verified' in text and 'torch.cuda.is_initialized()' in text
 s=(D/'compare.py').read_text();assert "'selection_order_equal': ordered" in s and "'output_tokens_equal': tokens" in s and 'len(keys) != 384' in s
 # Real entrypoint must still refuse before creating a container.
 x=subprocess.run([sys.executable,str(D/'launch.py'),'--stage','preflight'],cwd=B,capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=120)
 assert x.returncode!=0 and 'monitor_pixel_wavefront_full_preflight' in x.stderr and not (D/'run_preflight001').exists()
 fields={'plan_sha256':D/'plan002.json','reviewer_sha256':Path(__file__),'cpu_proof_sha256':D/'cpu_proof002.json','cpu_started_sha256':D/'monitor_cpu002/started.json','cpu_terminal_sha256':D/'monitor_cpu002/terminal.json'}
 v={'status':'passed','independent_review':True,'scope':'preflight','errors':[],'GT_opened':False,'GPU_used':False,'checks':['384 prior dual-accepted identities and 768 source hashes','frozen source identity including actual diagnose import','CPU tests and true working-directory dependency smoke','actual unreviewed entry refuses before container','ordered selection and tokens compare all rows without discarding mismatches'],'limitations':['only authorizes no-device preflight','preflight must check actual same-image Docker and weight hashes','posthoc same DEV; high-dimensional prior logits/KV not archived']};v.update({k:sha(x) for k,x in fields.items()})
 with out.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
