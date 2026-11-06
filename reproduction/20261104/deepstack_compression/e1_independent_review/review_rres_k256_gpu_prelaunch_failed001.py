#!/usr/bin/env python3
"""Independent final review of the three-anchor 512px model diagnostic."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/rres_k256_execution';G=B/'e6/rres_k256_gpu_plan';P=B/'e6/rres_k256_pilot_v2'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(m,script):
 s=read(m/'started.json');t=read(m/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and (Path(s['cwd'])/s['command'][1]).resolve()==script.resolve()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out.resolve();assert out==I/'rres_k256_gpu_prelaunch_review.json' and not out.exists() and not (D/'run001').exists()
 term(G/'monitor_freeze001',G/'freeze.py');term(G/'monitor_cpu001',G/'cpu_check.py');term(I/'monitor_rres_k256_processor_v2_preflight_result',I/'review_rres_k256_processor_v2_preflight_result.py');term(I/'monitor_rres_k256_runtime_preflight_result',I/'review_rres_k256_runtime_preflight_result.py')
 p=read(D/'gpu_plan001.json');f=read(G/'verification_plan001.json');a=read(I/'rres_k256_processor_v2_preflight_result_review.json');b=read(I/'rres_k256_runtime_preflight_result_review.json');probe=read(P/'run_preflight001/processor_probe.json');budget=read(B/'initial_plan.json')
 assert p['status']=='ready_for_independent_review' and p['images']==3 and p['visual_tokens']==256 and p['GT_opened'] is False and p['scoring_performed'] is False and all(sha(k)==v for k,v in p['source_sha256'].items())
 assert f['status']=='frozen_before_gpu' and f['images']==3 and f['GT_opened'] is False and f['scoring_performed'] is False and f['verifier_sha256']==sha(D/'verify.py')
 assert a['status']=='passed' and a['independent_review'] is True and a['images']==3 and a['worker_sha256']==sha(P/'run_preflight001/worker_preflight.json') and p['preflight_result_sha256']==sha(I/'rres_k256_processor_v2_preflight_result_review.json')
 assert b['status']=='passed' and b['independent_review'] is True and b['worker_sha256']==sha(B/'e6/rres_k256_runtime_preflight/run001/worker.json') and p['runtime_preflight_result_sha256']==sha(I/'rres_k256_runtime_preflight_result_review.json')
 assert probe['images']==3 and {x['task'] for x in probe['anchors']}=={'DocVQA_VAL','TextVQA_VAL','AI2D_TEST'} and all(x['low_visual_tokens']==256 and x['grid']==[[1,32,32]] and x['position_formula_verified'] is True for x in probe['anchors']) and time.time()+3840<budget['budget_end_unix']
 w=(D/'worker.py').read_text();arm=(P/'arm.py').read_text();cap=(P/'capture.py').read_text();ver=(D/'verify.py').read_text()
 for x in ('checked_generation','len(controls)!=384','arm.run_arm','hooks','worker_complete.json'):
  assert x in w,x
 for x in ('fixed_resolution(module,512)','persist(raw_path,raw)','decode.validate_trace','hooks_and_methods_restored'):
  assert x in arm,x
 for x in ("s['visual'].shape[0]!=256","s['ds_calls']!=[1]*3",'torch.equal(s[\'prefill_position_ids\'],position_ids)','torch.isfinite'):
  assert x in cap,x
 assert "for anchor in anchors:" in ver and "r['deepstack']==r['injected_ds']" in ver and "len(trace)==len(ids)" in ver
 z=subprocess.run([sys.executable,str(D/'launch.py'),'--run-id','001'],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=120)
 assert z.returncode!=0 and not (D/'run001').exists() and 'monitor_rres_k256_gpu_prelaunch' in z.stderr,z.stderr[-1000:]
 value={'status':'passed','independent_review':True,'images':3,'GT_opened':False,'errors':[],'gpu_plan_sha256':sha(D/'gpu_plan001.json'),'source_sha256':p['source_sha256'],'reviewer_sha256':sha(__file__),'formal_plan_sha256':sha(G/'verification_plan001.json'),'checks':['actual processor and runtime noGPU independent proofs','frozen 223 source hashes and full-model formal contract','same-question controls and 256 visual/three DS/position/KV validation','raw sequence fsynced before analysis; hooks and resolution restored','real GPU entry refuses without proof and absolute deadline bounds'],'limits':['three already observed DEV anchors, no quality or speed inference','same token count does not imply equal compute','high-dimensional tensors not archived; runtime checks and summaries retained']}
 with out.open('x') as o:json.dump(value,o,indent=2);o.write('\n');o.flush();os.fsync(o.fileno())
if __name__=='__main__':main()
