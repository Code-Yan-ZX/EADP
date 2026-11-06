#!/usr/bin/env python3
"""Independent final host/acceptance admission review, no model or 343 values."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e4_multistream_expansion343_launch';V=B/'e4_multistream_expansion343_gpu'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(m,script):
 s=read(m/'started.json');t=read(m/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group'] and s['worker_pid']==t['worker_pid'] and (Path(s['cwd'])/s['command'][1]).resolve()==script.resolve()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out.resolve();assert out==I/'e4_multistream_expansion343_guard_gpu_review.json' and not out.exists() and not (V/'run001').exists()
 term(D/'monitor_cpu_check001',D/'cpu_check.py');term(D/'monitor_host_smoke003',D/'launch.py');term(I/'monitor_e4_multistream_expansion343_gpu_review',I/'review_e4_multistream_expansion343_gpu_review.py')
 c=read(D/'cpu_check001.json');f=read(D/'formal_plan001.json');worker=read(I/'e4_multistream_expansion343_gpu_review.json');p=read(I/'e4_multistream_expansion343_preflight001_result_review.json');a=read(I/'e4_multistream_expansion343_analysis_prelaunch_review.json')
 assert c['status']=='passed' and c['source_sha256_before']==c['source_sha256_after'] and all(sha(k)==v for k,v in c['source_sha256_after'].items()) and c['GT_opened'] is False and c['GPU_used'] is False and 'Ran 8 tests' in c['tests']
 assert f['images']==343 and f['GT_opened'] is False and f['scoring_performed'] is False and f['automatic_retry'] is False
 assert worker['status']=='passed' and worker['independent_review'] is True and worker['preflight_only'] is False and p['status']=='passed' and p['independent_review'] is True and p['images']==343 and a['status']=='passed' and a['independent_review'] is True and not any((worker['errors'],p['errors'],a['errors']))
 spec=importlib.util.spec_from_file_location('independent_host_gate',D/'gpu_gate.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g);assert g.analysis_prelaunch()==sha(I/'e4_multistream_expansion343_analysis_prelaunch_review.json')
 # Check the real executable fails before this independent proof exists.
 z=subprocess.run([sys.executable,str(D/'launch.py'),'--run-id','001'],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True,timeout=90)
 assert z.returncode!=0 and not (V/'run001').exists() and 'monitor_e4_multistream_expansion343_guard_gpu_review' in z.stderr,z.stderr[-1000:]
 # Complete-only contract must reject omissions and altered E1 fields.
 spec=importlib.util.spec_from_file_location('independent_complete_contract',D/'verify_contract.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
 selected=[{'task':'x','qid':str(i)} for i in range(343)];ok=[dict(r,status='passed') for r in selected];assert v.complete(selected,ok)
 try:v.complete(selected,ok[:-1]);raise AssertionError('omission accepted')
 except ValueError:pass
 ref={k:k for k in v.FIELDS};assert v.actual_identity(dict(ref),ref)
 bad=dict(ref);bad['visual_sha256']='tampered'
 try:v.actual_identity(bad,ref);raise AssertionError('identity tamper accepted')
 except ValueError:pass
 fields={'reviewer_sha256':Path(__file__),'cpu_sha256':D/'cpu_check001.json','cpu_started_sha256':D/'monitor_cpu_check001/started.json','cpu_terminal_sha256':D/'monitor_cpu_check001/terminal.json','worker_gpu_review_sha256':I/'e4_multistream_expansion343_gpu_review.json','preflight_result_review_sha256':I/'e4_multistream_expansion343_preflight001_result_review.json','formal_plan_sha256':D/'formal_plan001.json','analysis_review_sha256':I/'e4_multistream_expansion343_analysis_prelaunch_review.json'}
 value={'status':'passed','independent_review':True,'GT_opened':False,'GPU_used':False,'new_343_values_opened':False,'errors':[],'source_sha256':c['source_sha256_after'],'checks':['host source/CPU proofs and actual dependency smoke','entry refuses without independent guard proof','pre-reviewed 384 analysis binding','343 complete denominator and E1 identity tamper rejection','formal verifier and source snapshot reviewed before capture']}
 value.update({k:sha(x) for k,x in fields.items()})
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
